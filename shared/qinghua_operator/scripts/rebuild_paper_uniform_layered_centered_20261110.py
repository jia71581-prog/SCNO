#!/usr/bin/env python3
"""Rebuild the paper TRAIN934 with every source at the domain centre.

Inputs (velocities, rosters, normalizer, 801-frame time protocol) are copied from
the frozen local GPU corpus.  The only physical change is the source: all 934
records use [1000, 1000, 15, 0.1, 1] m/Hz/s/Pa.  Wavefields are stored with
lossless LZF compression so the 801-frame corpus fits on the root filesystem.
"""
from __future__ import annotations
from scno_release_paths import local_asset as _scno_local_asset

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.dont_write_bytecode = True
OLD = ROOT/'data/paper_uniform500_layered434_raw801_local_gpu_20261022'
# local_assets/home/Data is a symlink to /data (37 GB free).  The regenerated corpus
# must live on the root filesystem, so use an absolute /home path there.
BASE = Path(_scno_local_asset('centered'))
RUN = BASE/'runs/paper_uniform_layered_centered_rebuild_20261110'
DATA = BASE/'dataset_uniform500_layered434_raw801_centered'
CENTER = (1000.0, 1000.0, 15.0, 0.1, 1.0)
SOURCE_KEYS = ('source_x_m', 'source_z_m', 'source_f0_hz', 'source_t0_s', 'source_amplitude')
COPY_INPUTS = ('velocity401.npy', 'velocity201.npy', 'time_s.npy', 'native400_indices.npy',
               'normalization_ul.json', 'TRAIN_ROSTER.json', 'VAL_ROSTER.json',
               'TEST_ROSTER.json', 'CONFIG.json', 'GENERATION_ROSTER.json')
STOP = False
IO_LOCK = threading.Lock()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.partial')
    with tmp.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def stop(*_):
    global STOP
    STOP = True


def prepare_inputs():
    """Publish copied frozen inputs with a centered-source override."""
    import numpy as np
    if DATA.exists():
        return verify_inputs()
    for name in COPY_INPUTS:
        if not (OLD/name).is_file():
            raise FileNotFoundError(OLD/name)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+DATA.name+'.inputs.', dir=DATA.parent))
    try:
        for name in COPY_INPUTS:
            shutil.copyfile(OLD/name, staging/name)
        roster = read(staging/'GENERATION_ROSTER.json')
        records = roster['records']
        if len(records) != 934:
            raise ValueError('Expected 934 TRAIN records')
        for record in records:
            previous = {
                'source': record.get('source'),
                'source_solver': record.get('source_solver'),
                'source_model': record.get('source_model'),
                'source_precision_provenance': record.get('source_precision_provenance'),
            }
            record['source_original_before_centering'] = previous
            record['source'] = list(CENTER)
            record['source_solver'] = list(CENTER)
            record['source_model'] = list(CENTER)
            record['source_precision_provenance'] = {
                'kind': 'centered_source_override_20261110',
                'user_instruction': 'move every source to the domain centre and regenerate',
                'center_m': [CENTER[0], CENTER[1]],
            }
        roster['source_override'] = {
            'center_m': [CENTER[0], CENTER[1]],
            'f0_hz': CENTER[2], 't0_s': CENTER[3], 'amplitude': CENTER[4],
            'applies_to_records': 934, 'utc': utc(),
        }
        write(staging/'GENERATION_ROSTER.json', roster)
        config = read(staging/'CONFIG.json')
        source = config['physics_config']['source']
        source['x_m'] = CENTER[0]
        source['z_m'] = CENTER[1]
        source['f0_hz'] = CENTER[2]
        source['t0_s'] = CENTER[3]
        source['amplitude'] = CENTER[4]
        source['centered_override_utc'] = utc()
        write(staging/'CONFIG.json', config)
        np.save(staging/'sources_solver.npy', np.tile(np.asarray(CENTER, np.float64), (934, 1)))
        np.save(staging/'sources_model.npy', np.tile(np.asarray(CENTER, np.float32), (934, 1)))
        # Validate copied velocity identities against the unchanged roster hashes.
        fine = np.load(staging/'velocity401.npy', mmap_mode='r', allow_pickle=False)
        sharp = np.load(staging/'velocity201.npy', mmap_mode='r', allow_pickle=False)
        if fine.shape != (934, 401, 401) or fine.dtype != np.float32:
            raise ValueError('velocity401 shape/dtype')
        if sharp.shape != (934, 201, 201) or sharp.dtype != np.float32:
            raise ValueError('velocity201 shape/dtype')
        for index, record in enumerate(records):
            if array_sha(fine[index]) != record['source_c401_sha256']:
                raise ValueError(f'fine401 velocity hash mismatch row {index}')
            if array_sha(sharp[index]) != record['velocity_sha256']:
                raise ValueError(f'sharp201 velocity hash mismatch row {index}')
        del fine, sharp
        old_manifest = read(OLD/'INPUT_MANIFEST.json')
        artifacts = {}
        for path in sorted(staging.iterdir()):
            if path.name == 'INPUT_MANIFEST.json':
                continue
            with path.open('rb') as handle:
                os.fsync(handle.fileno())
            entry = dict(path=path.name, sha256=sha(path), size_bytes=path.stat().st_size)
            if path.suffix == '.npy':
                value = np.load(path, mmap_mode='r', allow_pickle=False)
                entry.update(shape=list(value.shape), dtype=str(value.dtype))
                del value
            artifacts[path.name] = entry
        manifest = dict(old_manifest)
        manifest.update(
            schema='PAPER_UNIFORM500_LAYERED434_RAW801_CENTERED_INPUTS_V1',
            status='INPUTS_PREPARED_NO_PRESSURE',
            created_utc=utc(), data_dir=str(DATA),
            artifacts=artifacts,
            GPU_initialized_by_helper=False,
            source_override=dict(
                center_m=[CENTER[0], CENTER[1]], f0_hz=CENTER[2], t0_s=CENTER[3],
                amplitude=CENTER[4], applies_to_records=934,
                changed_artifacts=['CONFIG.json', 'GENERATION_ROSTER.json',
                                   'sources_solver.npy', 'sources_model.npy'],
                utc=utc(),
                provenance='velocities, rosters, normalizer and 801/400 time protocol copied unchanged from '+str(OLD)),
        )
        manifest['checks'] = dict(manifest['checks'])
        manifest['checks'].update(source_centered_all_records=True, source_xy_unique=1,
                                  source_protocol='single point source at domain centre [1000,1000] m')
        manifest['source_bindings'] = dict(manifest['source_bindings'])
        manifest['source_bindings']['scripts/rebuild_paper_uniform_layered_centered_20261110.py'] = sha(Path(__file__).resolve())
        write(staging/'INPUT_MANIFEST.json', manifest)
        if DATA.exists():
            raise FileExistsError(DATA)
        os.rename(staging, DATA)
    except BaseException:
        raise
    return verify_inputs()


def verify_inputs():
    import numpy as np
    manifest = read(DATA/'INPUT_MANIFEST.json')
    if manifest['data_dir'] != str(DATA):
        raise ValueError('Input manifest data_dir mismatch')
    for name, entry in manifest['artifacts'].items():
        path = DATA/name
        if path.stat().st_size != entry['size_bytes'] or sha(path) != entry['sha256']:
            raise ValueError('Input artifact changed: '+name)
    roster = read(DATA/'GENERATION_ROSTER.json')['records']
    solver = np.load(DATA/'sources_solver.npy', allow_pickle=False)
    model = np.load(DATA/'sources_model.npy', allow_pickle=False)
    expected = np.tile(np.asarray(CENTER, np.float64), (934, 1))
    if solver.shape != (934, 5) or not np.array_equal(solver, expected):
        raise ValueError('solver sources are not all centered')
    if model.shape != (934, 5) or not np.array_equal(model, expected.astype(np.float32)):
        raise ValueError('model sources are not all centered')
    if any(list(r['source_solver']) != list(CENTER) or list(r['source_model']) != list(CENTER) for r in roster):
        raise ValueError('roster source override incomplete')
    if len(roster) != 934 or sum(r['family'] == 'uniform' for r in roster) != 500:
        raise ValueError('roster family counts')
    times = np.load(DATA/'time_s.npy', allow_pickle=False)
    if not np.array_equal(times, np.arange(801, dtype=np.float64)*.00125):
        raise ValueError('801 time protocol changed')
    return dict(status='PASS', data_dir=str(DATA), records=934,
                source='[1000.0, 1000.0, 15.0, 0.1, 1.0]',
                stored_frames=801, selected_frames=400,
                input_manifest_sha256=sha(DATA/'INPUT_MANIFEST.json'))


def teacher(physics, *, device='cuda'):
    import numpy as np
    import torch
    from label_quality_fd.data_generation.raw_dataset import NodalLWC84Solver
    from label_quality_fd.data_generation.grid import AcousticGrid, BoundaryConfig
    b = physics['boundaries']
    boundary = BoundaryConfig(**{key: b[key] for key in ('top', 'left', 'right', 'bottom',
        'npml', 'cpml_target_reflection', 'cpml_polynomial_order', 'cpml_outside_physical_domain')})
    return NodalLWC84Solver(
        grid=AcousticGrid(nx=401, nz=401, dx_m=5., dz_m=5., lx_m=2000., lz_m=2000., centering='node'),
        boundaries=boundary, dt_s=.000125,
        output_times_s=np.arange(801, dtype=np.float64)*.00125,
        c_ref_mps=6750., device=device, dtype=torch.float32,
        kappa_max=b['kappa_max'], minimum_frequency_hz=b['minimum_frequency_hz'],
        output_restriction_factor=2, cuda_graphs=str(device).startswith('cuda'),
        block_step_count=10 if str(device).startswith('cuda') else None)


def relative_fp32(predicted, reference):
    import numpy as np
    a, b = np.asarray(predicted, np.float32), np.asarray(reference, np.float32)
    residual = np.subtract(a, b, dtype=np.float32)
    numerator = np.sum(np.square(residual, dtype=np.float32), dtype=np.float32)
    denominator = np.sum(np.square(b, dtype=np.float32), dtype=np.float32)
    if not np.isfinite(numerator) or not denominator > 0:
        raise ValueError('Invalid FP32 relative-error reduction')
    return float(np.sqrt(np.divide(numerator, denominator, dtype=np.float32), dtype=np.float32))


def choose_batch(remaining, free_bytes, per_record_bytes, fixed_bytes, maximum=256):
    if min(remaining, free_bytes, per_record_bytes, maximum) <= 0 or fixed_bytes < 0:
        raise ValueError('Invalid batch-admission inputs')
    capacity = int((.85*free_bytes-fixed_bytes)//per_record_bytes)
    if capacity < 1:
        return 0
    chosen = min(remaining, maximum, capacity)
    return 16*(chosen//16) if chosen >= 16 and chosen < remaining else chosen


def record_path(index):
    return DATA/'records'/f'{index:06d}.h5'


def receipt_path(index):
    return DATA/'receipts'/f'{index:06d}.json'


def load_completed():
    completed = {}
    for index in range(934):
        record, receipt = record_path(index), receipt_path(index)
        if not record.exists() and not receipt.exists():
            continue
        if not record.exists() or not receipt.exists():
            raise RuntimeError(f'partial record/receipt pair at row {index}')
        value = read(receipt)
        if value['status'] != 'PASS' or record.stat().st_size != value['bytes'] or sha(record) != value['sha256']:
            raise RuntimeError(f'corrupt committed record row {index}')
        completed[index] = value
    return completed


def save_record(row, field, velocity, solver_source, model_source, source_map, solver_metric):
    import h5py
    import numpy as np
    from label_quality_fd.data_generation.source import bilinear_point_source
    record_id = int(row['row'])
    if field.shape != (801, 201, 201) or field.dtype != np.float32:
        raise ValueError('Full FP32 native801 field required')
    if not np.isfinite(field).all() or np.any(field[0]) or np.any(field[:, 0, :]) or not np.any(field):
        raise ValueError('Finite/zero-initial/free-surface/nonzero checks failed')
    expected = bilinear_point_source(CENTER[0], CENTER[1], nx=401, nz=401, dx_m=5., dz_m=5.,
                                     centering='node').source_map.astype(np.float32)
    if not np.array_equal(source_map, expected) or np.count_nonzero(source_map) > 4:
        raise ValueError('Centered source map mismatch')
    if not np.array_equal(np.asarray(solver_source, np.float32), model_source):
        raise ValueError('Solver and model source encodings disagree')
    path, receipt = record_path(record_id), receipt_path(record_id)
    if path.exists() or receipt.exists():
        raise ValueError('Refusing to overwrite a committed record')
    path.parent.mkdir(exist_ok=True); receipt.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix('.h5.partial')
    with h5py.File(temporary, 'x') as handle:
        handle.attrs['sample_id'] = row['sample_id']; handle.attrs['row'] = record_id
        handle.attrs['family'] = row['family']; handle.attrs['split'] = 'train'
        handle.attrs['physical_sources_per_record'] = 1
        handle.attrs['pressure_units'] = 'Pa'; handle.attrs['pressure_axis_order'] = 'TZX'
        handle.attrs['spatial_restriction'] = 'fine401[::2,::2], no filter'
        handle.attrs['source_center_m'] = json.dumps([CENTER[0], CENTER[1]])
        handle.create_dataset('wavefield', data=field, chunks=(1, 201, 201),
                              compression='lzf', shuffle=True)
        handle['receiver_raw201'] = field[:, 2, :]
        handle['velocity201'] = velocity
        handle['source_solver'] = solver_source
        handle['source_parameters'] = model_source
        handle.flush()
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    with h5py.File(temporary, 'r') as handle:
        for start in range(0, 801, 32):
            if not np.array_equal(handle['wavefield'][start:start+32], field[start:start+32]):
                raise ValueError('Wavefield write/readback mismatch')
        for name, array in [('velocity201', velocity), ('source_solver', solver_source),
                            ('source_parameters', model_source),
                            ('receiver_raw201', field[:, 2, :])]:
            if not np.array_equal(handle[name][:], array):
                raise ValueError('Metadata or receiver readback mismatch')
    os.replace(temporary, path)
    value = dict(status='PASS', row=record_id, sample_id=row['sample_id'],
                 family=row['family'], file=str(path.relative_to(DATA)), sha256=sha(path),
                 bytes=path.stat().st_size, shape=list(field.shape), dtype='float32',
                 compression='lzf', physical_sources_per_record=1,
                 source_center_m=[CENTER[0], CENTER[1]],
                 source_stencil_nonzero_nodes=int(np.count_nonzero(source_map)),
                 finite=True, zero_initial=True, zero_free_surface=True,
                 nonzero_final_frame=bool(np.any(field[-1])),
                 maximum_abs_pressure_pa=float(np.max(np.abs(field))),
                 full_write_readback_equal=True, solver_metrics=solver_metric, utc=utc())
    write(receipt, value)
    return value


def build_views(roster, velocity, solver_sources, model_sources):
    import h5py
    import numpy as np
    count = len(roster)
    times = np.load(DATA/'time_s.npy', allow_pickle=False)
    indices = np.load(DATA/'native400_indices.npy', allow_pickle=False)
    layout = h5py.VirtualLayout((count, 801, 201, 201), np.float32)
    receiver = h5py.VirtualLayout((count, 801, 201), np.float32)
    for row in roster:
        i = int(row['row']); name = f'records/{i:06d}.h5'
        if not (DATA/name).is_file():
            raise ValueError('Missing VDS source: '+name)
        layout[i] = h5py.VirtualSource(name, 'wavefield', shape=(801, 201, 201))
        receiver[i] = h5py.VirtualSource(name, 'receiver_raw201', shape=(801, 201))
    temporary = DATA/'dataset_v1.h5.partial'
    with h5py.File(temporary, 'x', libver='latest') as handle:
        handle.create_virtual_dataset('wavefield', layout, fillvalue=np.nan)
        handle.create_virtual_dataset('receiver_raw201', receiver, fillvalue=np.nan)
        handle['velocity'] = velocity
        handle['source_parameters'] = model_sources
        handle['source_solver'] = solver_sources
        handle['time_s'] = times; handle['native400_indices'] = indices
        handle['x_m'] = np.arange(201, dtype=np.float32)*10
        handle['z_m'] = np.arange(201, dtype=np.float32)*10
        dtype = h5py.string_dtype('utf-8')
        for key, values in [('sample_id', [r['sample_id'] for r in roster]),
                            ('medium_type', [r['family'] for r in roster]),
                            ('split', ['train']*count)]:
            handle.create_dataset(key, data=np.asarray(values, dtype=object), dtype=dtype)
        handle.attrs['schema'] = 'PAPER_TRAIN934_RAW801_CENTERED_V1'
        handle.attrs['axis_order'] = 'NTZX'
        handle.attrs['physical_sources_per_record'] = 1
        handle.attrs['units'] = 'Pa'
        handle.attrs['pressure_compression'] = 'lzf lossless'
        handle.attrs['source_center_m'] = json.dumps([CENTER[0], CENTER[1]])
    os.replace(temporary, DATA/'dataset_v1.h5')
    selected = h5py.VirtualLayout((count, 400, 201, 201), np.float32)
    source = h5py.VirtualSource('dataset_v1.h5', 'wavefield', shape=(count, 801, 201, 201))
    for j, index in enumerate(indices):
        selected[:, j] = source[:, int(index)]
    temporary = DATA/'train400.h5.partial'
    with h5py.File(temporary, 'x', libver='latest') as handle:
        handle.create_virtual_dataset('wavefield', selected, fillvalue=np.nan)
        handle['time_s'] = times[indices]; handle['native_indices'] = indices
        handle['velocity'] = velocity; handle['source_parameters'] = model_sources
        handle['source_solver'] = solver_sources
        handle.attrs['selection'] = 'exact original native400 indices; no interpolation'
        handle.attrs['axis_order'] = 'NTZX'
    os.replace(temporary, DATA/'train400.h5')
    with h5py.File(DATA/'dataset_v1.h5', 'r') as all_frames, h5py.File(DATA/'train400.h5', 'r') as selected_frames:
        for i in [0, 133, 433, 434, 933]:
            with h5py.File(DATA/f'records/{i:06d}.h5', 'r') as record:
                for j in [0, 7, 8, 399]:
                    t = int(indices[j]); expected = record['wavefield'][t]
                    if not np.array_equal(all_frames['wavefield'][i, t], expected):
                        raise ValueError('Native801 VDS readback differs')
                    if not np.array_equal(selected_frames['wavefield'][i, j], expected):
                        raise ValueError('Native400 VDS time selection differs')
    return dict(status='PASS', source_records=count, original_frames=801,
                exact_selected_frames=400, VDS_relative_paths=True,
                physical_frame_interpolation=False, pressure_compression='lzf lossless')


def worker():
    import signal
    import numpy as np
    import torch
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    verify = verify_inputs()
    RUN.mkdir(parents=True, exist_ok=True)
    write(RUN/'STATUS.json', dict(status='STARTING', input_verification=verify, utc=utc()))
    config = read(DATA/'CONFIG.json')
    physics = config['physics_config']
    roster = read(DATA/'GENERATION_ROSTER.json')['records']
    velocity = np.load(DATA/'velocity401.npy', mmap_mode='r', allow_pickle=False)
    coarse = np.load(DATA/'velocity201.npy', mmap_mode='r', allow_pickle=False)
    sources = np.load(DATA/'sources_solver.npy', allow_pickle=False)
    model_sources = np.load(DATA/'sources_model.npy', allow_pickle=False)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    (DATA/'records').mkdir(exist_ok=True); (DATA/'receipts').mkdir(exist_ok=True)
    started = time.monotonic(); completed = load_completed(); propagations = []
    mutex = threading.Lock(); last_progress = 0.

    def budget():
        if STOP:
            raise InterruptedError('generation stopped by signal')

    def status(stage, **extra):
        with mutex:
            write(RUN/'STATUS.json', dict(status='RUNNING', stage=stage, completed_records=len(completed),
                total_records=934, elapsed_s=time.monotonic()-started, utc=utc(), **extra))

    def solve(ids):
        nonlocal last_progress
        budget(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic(); ids = np.asarray(ids, dtype=np.int64)
        solver = teacher(physics, device='cuda')
        s = sources[ids]
        with torch.no_grad():
            output = solver.simulate_device(np.ascontiguousarray(velocity[ids]),
                **{key: s[:, j] for j, key in enumerate(SOURCE_KEYS)})
        torch.cuda.synchronize()
        fields = output.wavefield_device.cpu().numpy()
        info = dict(batch_size=len(ids), rows=ids.tolist(), seconds=time.monotonic()-t0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            full_output_shape=list(fields.shape),
            includes_full801_allocation_and_D2H=True, utc=utc())
        maps = output.source_map_solver; metrics = output.metrics
        del output, solver
        gc.collect(); torch.cuda.empty_cache()
        propagations.append(info); write(RUN/'PROPAGATIONS.json', propagations)
        return fields, maps, metrics, info

    def commit(ids, bundle):
        fields, maps, metrics, info = bundle
        for j, i in enumerate(ids):
            budget()
            value = save_record(roster[i], fields[j], coarse[i], sources[i], model_sources[i], maps[j], metrics[j])
            with mutex:
                if i in completed:
                    raise ValueError('Duplicate committed row')
                completed[i] = value
                write(DATA/'PROGRESS.json', dict(completed_records=len(completed), total_records=934,
                    completed_rows=sorted(completed), utc=utc()))
                write(RUN/'STATUS.json', dict(status='RUNNING', stage='RECORD_COMMITTED',
                    completed_records=len(completed), total_records=934,
                    elapsed_s=time.monotonic()-started, utc=utc()))

    warmup_path = RUN/'GPU_PREFLIGHT.json'
    if warmup_path.exists() and completed:
        preflight = read(warmup_path)
        per_record = float(preflight['measured_per_record_bytes'])
        fixed = float(preflight['measured_fixed_bytes'])
        maximum = int(preflight.get('max_batch', 256))
    else:
        status('GPU_PREFLIGHT')
        probe_ids = [0, 133, 200, 433, 434, 500, 766, 933]
        singles = [solve([i]) for i in [0, 434]]
        probe = solve(probe_ids)
        errors = [relative_fp32(probe[0][probe_ids.index(i)], singles[j][0][0]) for j, i in enumerate([0, 434])]
        if max(errors) > 2e-6:
            raise ValueError('Full801 B1/B8 independence failed: '+str(errors))
        small = max(item[3]['peak_allocated_bytes'] for item in singles)
        per_record = (probe[3]['peak_allocated_bytes']-small)/7
        fixed = max(0., small-per_record)
        if per_record <= 0:
            raise ValueError('Measured memory slope is not positive')
        maximum = 256
        write(warmup_path, dict(status='PASS', B1_B8_full801_relative_L2_FP32=errors,
            measured_per_record_bytes=per_record, measured_fixed_bytes=fixed,
            max_batch=maximum, B1_peaks=[item[3] for item in singles], B8_peak=probe[3],
            source_centered=[CENTER[0], CENTER[1]], utc=utc()))
        for i in probe_ids:
            if i not in completed:
                value = save_record(roster[i], probe[0][probe_ids.index(i)], coarse[i], sources[i],
                                    model_sources[i], probe[1][probe_ids.index(i)], probe[2][probe_ids.index(i)])
                with mutex:
                    completed[i] = value
                    write(DATA/'PROGRESS.json', dict(completed_records=len(completed), total_records=934,
                        completed_rows=sorted(completed), utc=utc()))
        del singles, probe
        gc.collect(); torch.cuda.empty_cache()
    remaining = [i for i in range(934) if i not in completed]
    write(RUN/'BATCH_EXECUTION.json', dict(batch_policy='largest safe batch under 85% free VRAM, max 256',
        measured_per_record_bytes=per_record, measured_fixed_bytes=fixed, max_batch=maximum,
        resumed_records=len(completed), total_records=934, pressure_compression='lzf lossless',
        source_center_m=[CENTER[0], CENTER[1]], utc=utc()))
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='record-writer') as pool:
        pending = None
        while remaining:
            budget()
            if pending is not None and pending.done():
                pending.result(); pending = None
            free, total = torch.cuda.mem_get_info()
            chosen = choose_batch(len(remaining), free, per_record, fixed, maximum)
            if chosen < 1:
                raise RuntimeError('Insufficient free VRAM for one complete record')
            ids = remaining[:chosen]
            if shutil.disk_usage(DATA).free < 5*1024**3 + len(ids)*95*1024**2:
                raise RuntimeError('Insufficient disk reserve for compressed batch')
            write(RUN/'BATCH_SELECTION.json', dict(chosen_batch=chosen, remaining_records=len(remaining),
                free_bytes=free, total_bytes=total, measured_per_record_bytes=per_record,
                measured_fixed_bytes=fixed, fraction=.85, max_batch=maximum, utc=utc()))
            try:
                bundle = solve(ids)
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                write(RUN/'ALLOCATION_OOM.json', dict(batch=chosen, completed_records=len(completed), utc=utc()))
                if chosen <= 1:
                    raise
                maximum = max(1, chosen//2)
                continue
            if pending is not None:
                pending.result()
            pending = pool.submit(commit, ids, bundle)
            del bundle
            remaining = remaining[chosen:]
        if pending is not None:
            pending.result()
    status('BUILDING_EXACT_TIME_VIEWS')
    if set(completed) != set(range(934)):
        raise ValueError('Incomplete TRAIN934 row coverage')
    import h5py
    with h5py.File(DATA/'records/000000.h5', 'r') as handle:
        if handle['wavefield'].compression != 'lzf':
            raise ValueError('pressure compression is not lzf')
    views = build_views(roster, coarse, sources, model_sources)
    receipts = [completed[i] for i in range(934)]
    counts = {family: sum(r['family'] == family for r in receipts) for family in ['uniform', 'layered']}
    if counts != {'uniform': 500, 'layered': 434}:
        raise ValueError('Family counts differ from paper')
    result = dict(status='COMPLETE', records=934, families=counts, shape=[934, 801, 201, 201],
        selected_shape=[934, 400, 201, 201], axis_order='NTZX', dtype='float32', units='Pa',
        physical_sources_per_record=1, independent_GPU_batch_cases=True,
        source_center_m=[CENTER[0], CENTER[1]],
        source_protocol='every record uses one point source at the domain centre',
        pressure_compression='lzf lossless', dataset_path=str(DATA), views=views,
        record_bytes=sum(r['bytes'] for r in receipts), generation_wall_s=time.monotonic()-started,
        propagation_calls=len(propagations), largest_measured_batch=max(p['batch_size'] for p in propagations),
        largest_measured_allocated_bytes=max(p['peak_allocated_bytes'] for p in propagations),
        largest_measured_reserved_bytes=max(p['peak_reserved_bytes'] for p in propagations),
        normalization_refitted=False, old_dataset_modified=False, training_started=False,
        checks=dict(input_manifest_sha256=sha(DATA/'INPUT_MANIFEST.json'),
                    gpu_preflight_sha256=sha(RUN/'GPU_PREFLIGHT.json')),
        dataset_file_sha256=sha(DATA/'dataset_v1.h5'), train400_file_sha256=sha(DATA/'train400.h5'), utc=utc())
    write(DATA/'RECEIPTS_INDEX.json', dict(records=receipts))
    write(DATA/'DATASET_COMPLETE.json', result)
    write(RUN/'RESULT.json', result)
    write(RUN/'STATUS.json', dict(result, stage='FINISHED', completed_records=934, total_records=934))


def launch():
    verify = verify_inputs()
    if not RUN.exists():
        RUN.mkdir(parents=True)
    free = shutil.disk_usage(DATA.parent).free
    expected = 934*90*1024**2
    if free < expected + 10*1024**3:
        raise OSError(f'Insufficient disk: free={free} expected<{expected + 10*1024**3}')
    if (RUN/'LAUNCH.json').exists() and not (RUN/'STOP_REQUEST.json').exists():
        raise ValueError('Already launched; refusing duplicate')
    import subprocess
    with (RUN/'worker.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'worker'],
            cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
    value = dict(status='DETACHED_WORKER_STARTED', pid=process.pid, utc=utc(),
                 input_verification=verify, free_bytes=free, expected_record_bytes=expected,
                 total_records=934, batch_policy='largest safe batch under 85% free VRAM, max 256')
    write(RUN/'LAUNCH.json', value)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify', 'worker', 'launch'])
    mode = parser.parse_args().mode
    if mode == 'prepare':
        print(json.dumps(prepare_inputs(), ensure_ascii=False))
    elif mode == 'verify':
        print(json.dumps(verify_inputs(), ensure_ascii=False))
    elif mode == 'worker':
        try:
            worker()
        except BaseException:
            error = dict(status='INCOMPLETE', error=traceback.format_exc(), utc=utc())
            write(RUN/'ERROR.json', error); write(RUN/'STATUS.json', error)
            raise
    else:
        launch()
