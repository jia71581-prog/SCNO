"""TRAIN-only POD auxiliary targets and fixed, local VAL references.

Training pressures stay at TRAIN_ROOT/shards/{row:06d}/pressure_pa.npy (Pa).
POD arrays live in data/: codes [500,512] float64; query_indices [25600]
int64 flatten C-order (time,z,x); query_coordinates [25600,3] float32 are
(t_seconds,x/2000,z/2000). VAL files use original roster row IDs, not 0..15.
No model imports, fitted normalizers, optimization, TEST or paper labels.
"""
from scno_release_paths import local_asset as _scno_local_asset
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '4')
sys.dont_write_bytecode = True
import numpy as np

RUN = Path(__file__).resolve().parent
DATA = RUN / 'data'
TRAIN_ROOT = Path(_scno_local_asset('project/data/marmousi_pglopt_train500_raw400_20261020'))
MIGRATION = Path(_scno_local_asset('migration'))
RESTORE = TRAIN_ROOT.parents[1] / 'runs/deeponet_marmousi_repair_20261024/DATA_RESTORE_STATUS.json'
PRESSURE_SCALE = 1.252277344221455e-8
SEED = 20261025
SOURCE_KEYS = ('source_x_m', 'source_z_m', 'source_f0_hz', 'source_t0_s', 'source_amplitude')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def save_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush(); os.fsync(handle.fileno())
    if path.exists():
        if sha(tmp) != sha(path):
            tmp.unlink()
            raise ValueError(f'Refusing to change existing artifact: {path}')
        tmp.unlink()
    else:
        os.link(tmp, path); tmp.unlink()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                shape=list(array.shape), dtype=str(array.dtype))


def identity():
    fields = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(fields[19]))


def fixed_query_indices(frames=400, nz=201, nx=201, per_time=64, seed=SEED):
    if not 0 < per_time <= nz * nx:
        raise ValueError('Invalid spatial query count')
    rng = np.random.default_rng(seed)
    return np.concatenate([np.sort(rng.choice(nz * nx, per_time, replace=False))
                           + t * nz * nx for t in range(frames)]).astype(np.int64)


def select_validation_sources(train_sources, val_sources):
    train = np.asarray(train_sources, dtype=np.float64)
    val = np.asarray(val_sources, dtype=np.float64)
    if train.ndim != 2 or train.shape[1] != 5 or val.shape != (200, 5):
        raise ValueError('Expected source matrices [N,5] and [200,5]')
    if not np.isfinite(train).all() or not np.isfinite(val).all():
        raise ValueError('Nonfinite source parameters')
    rows = np.linspace(0, 199, 16, dtype=int)
    chosen = val[rows].copy()
    for dtype in (np.float64, np.float32):
        train_keys = {tuple(s) for s in train.astype(dtype)}
        val_keys = [tuple(s) for s in chosen.astype(dtype)]
        if len(set(val_keys)) != 16:
            raise ValueError(f'duplicate selected VAL sources in {dtype}')
        if train_keys.intersection(val_keys):
            raise ValueError(f'TRAIN/VAL source overlap in {dtype}')
    return rows, chosen


def normalize_pressure(values):
    return np.asarray(values, dtype=np.float64) / PRESSURE_SCALE


def pod_source_codes(normalized_train, rank=512):
    y = np.asarray(normalized_train, dtype=np.float64)
    if y.ndim != 2 or not np.isfinite(y).all() or rank < y.shape[0]:
        raise ValueError('POD requires finite TRAIN matrix and rank >= records')
    eig, u = np.linalg.eigh(y @ y.T / y.shape[1])
    eig, u = eig[::-1].copy(), u[:, ::-1].copy()
    signs = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
    u *= np.where(signs == 0, 1, signs)
    codes = np.zeros((y.shape[0], rank), dtype=np.float64)
    codes[:, :y.shape[0]] = np.sqrt(y.shape[0]) * u
    return codes, eig


def validate_field(field, frames=400, nz=201, nx=201):
    if field.shape != (frames, nz, nx) or field.dtype != np.float32:
        raise ValueError('Invalid pressure shape/dtype')
    if not np.isfinite(field).all() or np.any(field[:, 0]) or not np.any(field):
        raise ValueError('Pressure must be finite, nonzero, with exact zero free surface')


def bind_inputs():
    complete = read(TRAIN_ROOT / 'DATASET_COMPLETE.json')
    dataset = read(TRAIN_ROOT / 'DATASET.json')
    restore = read(RESTORE)
    if complete['status'] != 'COMPLETE' or complete['completed_records'] != 500:
        raise ValueError('Original TRAIN500 completion missing')
    if not restore.get('all_pressure_files_original_sha256_exact') or restore['status'] != 'COMPLETE':
        raise ValueError('Byte-exact local TRAIN restoration is not complete')
    if complete['dataset_sha256'] != sha(TRAIN_ROOT / 'DATASET.json'):
        raise ValueError('Dataset manifest changed')
    if dataset['pressure_scale_pa'] != PRESSURE_SCALE:
        raise ValueError('Inherited normalizer differs')
    train_roster = read(TRAIN_ROOT / 'TRAIN_ROSTER.json')['records']
    train_sources = np.load(TRAIN_ROOT / 'sources_fp64.npy', allow_pickle=False)
    if len(train_roster) != 500 or train_sources.shape != (500, 5):
        raise ValueError('TRAIN roster/source shape differs')
    np.testing.assert_array_equal(train_sources, np.asarray([r['source'] for r in train_roster], np.float64))
    records = []
    for item in complete['records']:
        row = item['row']
        folder = TRAIN_ROOT / 'shards' / f'{row:06d}'
        receipt_path = folder / 'RECORD.json'
        receipt = read(receipt_path)
        p = folder / 'pressure_pa.npy'
        expected = receipt['files']['pressure_pa.npy']
        stat = p.stat()
        if sha(receipt_path) != item['receipt_sha256'] or expected['sha256'] != item['pressure_sha256']:
            raise ValueError(f'Original receipt mismatch for TRAIN {row}')
        if stat.st_size != expected['bytes'] or stat.st_mtime_ns > RESTORE.stat().st_mtime_ns:
            raise ValueError(f'TRAIN {row} changed since verified restoration')
        a = np.load(p, mmap_mode='r', allow_pickle=False)
        if a.shape != (400, 201, 201) or a.dtype != np.float32:
            raise ValueError(f'TRAIN {row} shape/dtype mismatch')
        del a
        records.append(dict(row=row, path=str(p), original_verified_sha256=expected['sha256'],
                            bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                            receipt_sha256=item['receipt_sha256']))
    if sorted(r['row'] for r in records) != list(range(500)):
        raise ValueError('TRAIN rows are not exactly 0..499')
    names = ['DATASET.json', 'DATASET_COMPLETE.json', 'TRAIN_ROSTER.json', 'VAL_ROSTER.json',
             'PHYSICS_CONFIG.json', 'normalization_ul.json', 'velocity401.npy', 'velocity201.npy',
             'sources_fp64.npy', 'sources_fp32.npy', 'times_fp32.npy', 'full_frames.npy',
             'native_times_fp64.npy', 'native_times_fp32.npy']
    binding = dict(status='BOUND', train_root=str(TRAIN_ROOT), count=500,
                   metadata={n: sha(TRAIN_ROOT / n) for n in names},
                   restored_pressure_verification=dict(path=str(RESTORE), sha256=sha(RESTORE),
                       note='Full pressure SHA verified by completed restoration; this phase reads selected TRAIN values only.'),
                   records=sorted(records, key=lambda r: r['row']), pressure_scale_pa=PRESSURE_SCALE)
    write(DATA / 'TRAIN_BINDING.json', binding)
    return binding


def prepare_pod():
    start = time.monotonic()
    if (DATA / 'POD_COMPLETE.json').exists():
        verify_artifacts(read(DATA / 'POD_COMPLETE.json')['artifacts']); return
    binding = bind_inputs()
    q = fixed_query_indices()
    ti = q // (201 * 201); zi = q // 201 % 201; xi = q % 201
    times = np.load(TRAIN_ROOT / 'times_fp32.npy', allow_pickle=False)
    coords = np.column_stack((times[ti], xi * .005, zi * .005)).astype(np.float32)
    y = np.empty((500, len(q)), dtype=np.float64)
    for row, record in enumerate(binding['records']):
        a = np.load(record['path'], mmap_mode='r', allow_pickle=False)
        y[row] = normalize_pressure(a.reshape(-1)[q])
        del a
        if row % 50 == 0:
            print(json.dumps(dict(stage='POD_SAMPLING', row=row, elapsed_seconds=time.monotonic()-start)), flush=True)
    codes, spectrum = pod_source_codes(y)
    orth_error = float(np.max(np.abs(codes @ codes.T / 500 - np.eye(500))))
    if orth_error > 1e-12:
        raise ValueError(f'Source code orthogonality failure: {orth_error}')
    arrays = {'pod_codes.npy': codes, 'query_indices.npy': q,
              'query_coordinates.npy': coords, 'sampled_spectrum.npy': spectrum}
    artifacts = {name: save_array(DATA / name, a) for name, a in arrays.items()}
    report = dict(status='COMPLETE', **identity(), train_records=500, queries=25600,
                  sampling=dict(seed=SEED, points_per_time=64, frames=400, order='flat C-order time,z,x',
                                coordinates='float32 (t_seconds,x_m/2000,z_m/2000)', replacement=False),
                  pressure_scale_pa=PRESSURE_SCALE, normalization_refitted=False,
                  gram='float64 Y@Y.T / 25600; all 500 eigenvectors, descending eigenvalues',
                  eigenvector_sign='largest absolute entry positive per column',
                  code_definition='[sqrt(500)*U, zeros(500,12)]', role='TRAIN auxiliary targets only, never inference inputs',
                  sampled_normalized_matrix_sha256=hashlib.sha256(y.tobytes(order='C')).hexdigest(),
                  spectrum_min=float(spectrum[-1]), spectrum_max=float(spectrum[0]),
                  source_orthogonality_max_absolute_error=orth_error, artifacts=artifacts,
                  train_binding_sha256=sha(DATA / 'TRAIN_BINDING.json'),
                  optimizer_updates=0, elapsed_seconds=time.monotonic()-start,
                  numpy_version=np.__version__, preparation_source_sha256=sha(__file__))
    write(DATA / 'POD_COMPLETE.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'elapsed_seconds', 'source_orthogonality_max_absolute_error')}), flush=True)


def verify_artifacts(artifacts):
    for a in artifacts.values():
        if sha(a['path']) != a['sha256']:
            raise ValueError(f'Prepared artifact changed: {a["path"]}')


def freeze_solver():
    target = DATA / 'solver_snapshot'
    src = TRAIN_ROOT / 'source_snapshot/src'
    expected = {str(p.relative_to(src)): sha(p) for p in src.rglob('*.py')}
    target.mkdir(parents=True, exist_ok=True)
    for name, digest in expected.items():
        dest_source = target / 'src' / name
        if dest_source.exists() and sha(dest_source) != digest:
            raise ValueError(f'Frozen solver source changed: {name}')
        if not dest_source.exists():
            dest_source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, dest_source)
    compiled = MIGRATION / 'compiled_lwc84_strict.py'
    dest = target / compiled.name
    if dest.exists() and sha(dest) != sha(compiled):
        raise ValueError('Compiled solver source changed')
    if not dest.exists():
        shutil.copy2(compiled, dest)
    for name, digest in expected.items():
        if sha(target / 'src' / name) != digest:
            raise ValueError(f'Frozen solver source mismatch: {name}')
    manifest = dict(original_snapshot=str(src), source_files=expected,
                    compiled_step=dict(path=str(dest), sha256=sha(dest)))
    write(target / 'MANIFEST.json', manifest)
    return target


def make_solver(physics, device, frames, progress):
    import torch
    from label_quality_fd.data_generation.raw_dataset import NodalLWC84Solver
    from label_quality_fd.data_generation.grid import AcousticGrid, BoundaryConfig
    from compiled_lwc84_strict import install
    class ProgressSolver(NodalLWC84Solver):
        def _restrict_output(self, physical):
            out = super()._restrict_output(physical)
            self.saved_count = getattr(self, 'saved_count', 0) + 1
            progress(self.saved_count - 1, frames)
            return out
    b = physics['boundaries']
    boundary = BoundaryConfig(**{key: b[key] for key in ('top', 'left', 'right', 'bottom',
        'npml', 'cpml_target_reflection', 'cpml_polynomial_order', 'cpml_outside_physical_domain')})
    solver = ProgressSolver(grid=AcousticGrid(nx=401, nz=401, dx_m=5., dz_m=5., lx_m=2000.,
        lz_m=2000., centering='node'), boundaries=boundary, dt_s=.000125,
        output_times_s=np.arange(frames, dtype=np.float64)*.00125, c_ref_mps=6750.,
        device=device, dtype=torch.float32, kappa_max=b['kappa_max'],
        minimum_frequency_hz=b['minimum_frequency_hz'], output_restriction_factor=2,
        cuda_graphs=device.startswith('cuda'), block_step_count=10 if device.startswith('cuda') else None)
    return install(solver) if device.startswith('cuda') else solver


def prepare_validation():
    import torch
    torch.set_num_threads(4)
    folder = DATA / 'validation'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'COMPLETE.json').exists():
        report = read(folder / 'COMPLETE.json')
        verify_artifacts(report['artifacts']); return
    start = time.monotonic()
    snapshot = freeze_solver()
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(DATA / 'torchinductor_cache'))
    os.environ.setdefault('TRITON_CACHE_DIR', str(DATA / 'triton_cache'))
    sys.path[:0] = [str(snapshot / 'src'), str(snapshot)]
    physics = read(TRAIN_ROOT / 'PHYSICS_CONFIG.json')
    train_sources = np.load(TRAIN_ROOT / 'sources_fp64.npy', allow_pickle=False)
    roster = read(TRAIN_ROOT / 'VAL_ROSTER.json')['records']
    rows, sources = select_validation_sources(train_sources, [r['source'] for r in roster])
    for row in rows:
        if roster[row]['split'] != 'validation' or roster[row]['source_c401_sha256'] != read(TRAIN_ROOT / 'TRAIN_ROSTER.json')['records'][0]['source_c401_sha256']:
            raise ValueError('VAL source roster medium/split mismatch')
    velocity = np.load(TRAIN_ROOT / 'velocity401.npy', allow_pickle=False)
    frames = np.load(TRAIN_ROOT / 'full_frames.npy', allow_pickle=False)
    native = np.load(TRAIN_ROOT / 'native_times_fp64.npy', allow_pickle=False)
    if not np.array_equal(native, np.arange(801, dtype=np.float64)*.00125):
        raise ValueError('Original native times differ from solver output')
    if frames.shape != (400,) or len(np.unique(frames)) != 400 or frames.min() < 0 or frames.max() >= 801:
        raise ValueError('Invalid original full400 frame mapping')
    input_artifacts = {'sources_fp64.npy': save_array(folder / 'sources_fp64.npy', sources),
                       'sources_fp32.npy': save_array(folder / 'sources_fp32.npy', sources.astype(np.float32)),
                       'rows.npy': save_array(folder / 'rows.npy', rows)}
    metadata = dict(status='INPUTS_READY', validation_rows=rows.tolist(), records=[dict(roster_row=int(r),
        source_registered_fp64=sources[i].tolist(), file=f'{r:06d}.npy', original_record=roster[r]) for i,r in enumerate(rows)],
        training_root=str(TRAIN_ROOT), pressure_scale_pa=PRESSURE_SCALE, shape=[400,201,201],
        dtype='float32', units='Pa', source_keys=SOURCE_KEYS, train_val_disjoint_fp32_and_fp64=True,
        original_metadata={n:dict(path=str(TRAIN_ROOT/n),sha256=sha(TRAIN_ROOT/n)) for n in
            ['velocity401.npy','velocity201.npy','times_fp32.npy','native_times_fp64.npy','full_frames.npy','PHYSICS_CONFIG.json','VAL_ROSTER.json']},
        input_artifacts=input_artifacts, solver_manifest_sha256=sha(snapshot/'MANIFEST.json'),
        local_only=True, optimizer_updates=0, validation_used_for_training=False)
    write(folder / 'metadata.json', metadata)
    receipts = {}
    def status(stage, **kw):
        value = dict(status=stage, **identity(), elapsed_seconds=time.monotonic()-start,
                     completed_records=len(receipts), total_records=16, **kw)
        write(folder / 'STATUS.json', value); print(json.dumps(value), flush=True)
    def solve(selected, count, device):
        t0 = time.monotonic()
        def progress(i, total):
            if i % 100 == 0 or i == total-1:
                status('GENERATING', device=device, rows=rows[selected].tolist(), frame=i, total_frames=total)
        solver = make_solver(physics, device, count, progress)
        with torch.no_grad():
            fn = solver.simulate_device if device.startswith('cuda') else solver.simulate
            result = fn(np.repeat(velocity[None],len(selected),axis=0),
                        **{k:sources[selected,j] for j,k in enumerate(SOURCE_KEYS)})
        field = result.wavefield_device if device.startswith('cuda') else result.wavefield
        return field, solver, result, time.monotonic()-t0
    # Every invocation independently checks the first selected VAL source.
    cpu, solver, result, cpu_seconds = solve([0], 151, 'cpu')
    cpu = cpu.copy()
    del solver, result
    gc.collect()
    gpu_seconds = 0.
    prefix_relative = None
    pending = list(range(16))
    while pending:
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        capacity = int((free - 3*2**30) // (220*2**20))
        if capacity < 1:
            raise RuntimeError(f'Insufficient currently free GPU memory: {free} bytes')
        batch = min(8, len(pending), capacity)
        selected = pending[:batch]
        status('GPU_BATCH_ADMITTED', rows=rows[selected].tolist(), batch_size=batch, free_gpu_bytes=free)
        field, solver, result, elapsed = solve(selected, 801, 'cuda:0')
        gpu_seconds += elapsed
        if 0 in selected:
            prefix = field[selected.index(0), :151].cpu().numpy().astype(np.float64)
            ref = cpu[0].astype(np.float64)
            prefix_relative = float(np.linalg.norm((prefix-ref).ravel()) / np.linalg.norm(ref.ravel()))
            if not np.isfinite(prefix_relative) or prefix_relative > 1e-5:
                raise ValueError(f'CPU/GPU prefix mismatch: {prefix_relative}')
            write(folder / 'CPU_GPU_PREFIX.json', dict(status='PASS',frames=151,roster_row=int(rows[0]),
                relative_l2_fp64=prefix_relative,threshold=1e-5,cpu_seconds=cpu_seconds))
        for j, pos in enumerate(selected):
            values = field[j, frames].cpu().numpy()
            validate_field(values)
            path = folder / f'{rows[pos]:06d}.npy'
            receipt = save_array(path, values)
            restored = np.load(path, allow_pickle=False)
            validate_field(restored)
            if not np.array_equal(restored, values):
                raise ValueError('Validation pressure readback mismatch')
            receipt.update(roster_row=int(rows[pos]),units='Pa',finite=True,free_surface_exact_zero=True,readback_equal=True)
            write(path.with_suffix('.json'), receipt)
            receipts[path.name] = receipt
            status('WRITING', last_row=int(rows[pos]))
            del restored, values
        del field, solver, result
        gc.collect(); torch.cuda.empty_cache()
        pending = pending[batch:]
    report = dict(status='COMPLETE', **identity(), artifacts=receipts, records=16,
        metadata_sha256=sha(folder / 'metadata.json'), input_artifacts=input_artifacts,
        CPU_GPU_prefix_frames=151, CPU_GPU_relative_l2=prefix_relative,
        cpu_seconds=cpu_seconds,gpu_seconds=gpu_seconds,elapsed_seconds=time.monotonic()-start,
        local_only=True, optimizer_updates=0, TEST_pressure_read=False,paper_pressure_read=False,
        pressure_processing='original 401 solver -> exact spatial stride2; full801 -> original 400 indices',
        preparation_source_sha256=sha(__file__),solver_manifest_sha256=sha(snapshot/'MANIFEST.json'))
    write(folder / 'COMPLETE.json', report)
    status('COMPLETE', CPU_GPU_relative_l2=prefix_relative)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['pod', 'validation', 'all'])
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    import fcntl
    # Phase-private duplicate guards; never exclude unrelated GPU jobs.
    from contextlib import ExitStack
    with ExitStack() as stack:
        phases = ['pod', 'validation'] if args.phase == 'all' else [args.phase]
        for phase in phases:
            handle = stack.enter_context((DATA / f'{phase}.lock').open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(DATA / f'{args.phase}_PROCESS.json', dict(**identity(), argv=sys.argv, source_sha256=sha(__file__)))
        try:
            if args.phase in ('pod', 'all'):
                prepare_pod()
            if args.phase in ('validation', 'all'):
                prepare_validation()
        except BaseException as error:
            write(DATA / f'{args.phase}_FAILED.json', dict(**identity(), error=repr(error)))
            raise


if __name__ == '__main__':
    main()
