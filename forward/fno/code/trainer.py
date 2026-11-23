"""Checkpointed local FNO continuation with explicit uniform401/min-max bindings."""
from __future__ import annotations
from scno_release_paths import local_asset as _scno_local_asset

import argparse
import copy
import scno_posix_lock as fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
from datetime import datetime, timezone

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import h5py
import numpy as np
import torch
from torch import nn

ROOT = Path(_scno_local_asset('project'))
PACKAGED_ROOT = Path(__file__).resolve().parents[3]
SHARED_SOURCE = PACKAGED_ROOT / 'shared/qinghua_operator/src'
DATA_CONTRACT = Path(__file__).resolve().parents[1] / 'data_contract'
DATA = Path(os.environ.get('FNO_DATA_ROOT', ROOT / 'data/acoustic_uniform401_20261027'))
STAGE = ROOT / 'fno_uniform401_minmax_20261028'
RUNTIME = Path(_scno_local_asset('runtime-fno/fno401'))
sys.path[:0] = [str(SHARED_SOURCE), str(SHARED_SOURCE / 'baselines/raw934'),
                str(DATA_CONTRACT / 'code')]
from unit_range_loader import encode_pressure
from baselines.raw934.train_fno_raw934 import FNO2d as ULFNO
from baselines.marmousi_pde_20261020.models import build_model as build_marmousi

PROFILE = json.loads((DATA_CONTRACT / 'normalization_unit_range.json').read_text())
OLD_SCALE = 1.252277344221455e-8
KNOWN = np.array([44, 48, 51, 55, 58, 62, 66, 69], dtype=np.int64)
PARENTS = {
    'ul': ROOT / 'runs/uniform_layered934_fno_continue200_20261021/latest.pt',
    'marmousi': ROOT / 'runs/marmousi_fno_total200_20261021/latest.pt',
}
STOP = False


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def set_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


class CompatibleFNO(nn.Module):
    """The public pressure channels are min-max; a fixed layer preserves the old feature codec."""
    def __init__(self, net, case):
        super().__init__()
        self.net = net
        self.case = case

    def forward(self, x):
        first = 1 if self.case == 'ul' else 8
        known = (x[:, first:first + 8] * PROFILE['pressure_half_range_pa']
                 + PROFILE['pressure_center_pa']) / OLD_SCALE
        compatible = torch.cat((x[:, :first], known, x[:, first + 8:]), dim=1)
        return self.net(compatible)


def make_model(case, inherit=True):
    if case == 'ul':
        net = ULFNO(cin=14, width=55, modes=25, layers=4, cout=401)
        head_keys = {'proj.2.weight', 'proj.2.bias'}
        head = net.proj[-1]
    else:
        net = build_marmousi('fno', target_parameters=None)
        net.parent.head[-1] = nn.Conv2d(net.parent.head[-1].in_channels, 401, 1)
        head_keys = {'parent.head.4.weight', 'parent.head.4.bias'}
        head = net.parent.head[-1]
    migration = dict(new_output_frames=401, inherited_tensor_count=0, all_compatible_tensors_equal=True)
    if inherit:
        parent = torch.load(PARENTS[case], map_location='cpu', weights_only=False, mmap=True)
        body = {k: v for k, v in parent['model'].items() if k not in head_keys}
        loaded = net.load_state_dict(body, strict=False)
        assert set(loaded.missing_keys) == head_keys and not loaded.unexpected_keys
        assert all(torch.equal(net.state_dict()[k], v) for k, v in body.items())
        migration.update(inherited_tensor_count=len(body), parent_path=str(PARENTS[case]),
                         parent_sha256=sha(PARENTS[case]), reset_temporal_tensors=sorted(head_keys),
                         historical_training_epochs=200, optimizer='Fresh AdamW for corrected time/normalization stage')
    nn.init.zeros_(head.weight)
    nn.init.constant_(head.bias, PROFILE['normalized_physical_zero'])
    model = CompatibleFNO(net.float(), case)
    migration['parameters'] = sum(p.numel() for p in model.parameters())
    return model, migration


class Corpus:
    def __init__(self, case, cache_records):
        self.case = case
        self.path = DATA / 'dataset_v1.h5'
        with h5py.File(self.path, 'r') as f:
            families = f['family'].asstr()[...]
            selected = np.isin(families, ['uniform', 'layered']) if case == 'ul' else families == 'marmousi'
            self.rows = np.flatnonzero(selected).tolist()
            assert len(self.rows) == (934 if case == 'ul' else 500)
            self.sources = f['source_parameters'][...]
            self.source_solver = f['source_solver'][...]
            self.source_files = f['source_files'].asstr()[...]
            self.ids = f['sample_id'].asstr()[...]
            self.families = families
            times = f['time_s'][...]
            assert np.array_equal(times, np.arange(401, dtype=np.float64) * .0025)
            assert f['wavefield'].shape[1:] == (401, 201, 201)
        self.cache = {}
        self.source_identities = {}
        provenance = json.loads((DATA / 'combined_provenance.json').read_text())['records']
        assert [r['sample_id'] for r in provenance] == self.ids.tolist()
        self.expected_payload_hashes = [r['selected_payload_sha256'] for r in provenance]
        self.cache_records = min(cache_records, len(self.rows))
        self.energy = {}
        grid = np.arange(201, dtype=np.float32) * 10.
        self.z, self.x = np.meshgrid(grid, grid, indexing='ij')

    def record(self, row):
        source_path = (DATA / self.source_files[row]).resolve()
        assert source_path.is_file(), str(source_path)
        stat = source_path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if row in self.cache:
            assert identity == self.source_identities[row], 'A cached physical source file changed'
            return self.cache[row]
        with h5py.File(self.path, 'r') as f:
            pa = f['wavefield'][row]
            velocity = f['velocity'][row]
        assert pa.shape == (401, 201, 201) and pa.dtype == np.float32 and np.isfinite(pa).all()
        assert not pa[0].any() and not pa[:, 0].any()
        assert hashlib.sha256(memoryview(np.ascontiguousarray(pa)).cast('B')).hexdigest() == self.expected_payload_hashes[row], 'Selected401 physical payload SHA mismatch'
        self.source_identities[row] = identity
        self.energy[row] = float(np.square(pa, dtype=np.float64).sum())
        y = encode_pressure(pa, PROFILE)
        del pa
        assert float(y.min()) >= -1.0000002 and float(y.max()) <= 1.0000002
        source = self.sources[row]
        c = (velocity - np.float32(3728.63427734375)) / np.float32(2141.111083984375)
        known = y[KNOWN]
        if self.case == 'ul':
            # Same nodal unit-mass bilinear source map as the historical baseline.
            sx, sz = self.source_solver[row, :2] / 10.
            ix, iz = int(np.floor(sx)), int(np.floor(sz))
            ax, az = sx - ix, sz - iz
            smap = np.zeros((201, 201), np.float32)
            for dz, wz in [(0, 1 - az), (1, az)]:
                for dx, wx in [(0, 1 - ax), (1, ax)]:
                    if wz * wx:
                        smap[iz + dz, ix + dx] += wz * wx
            assert abs(smap.sum(dtype=np.float64) - 1) < 1e-7
            maps = np.stack((c, *known, smap, 2 * self.x / 2000 - 1, 2 * self.z / 2000 - 1,
                             (self.x - source[0]) / 2000, (self.z - source[1]) / 2000))
        else:
            scaled = source / np.array([2000., 2000., 50., 1.2, 1.], np.float32)
            constants = np.broadcast_to(scaled[:, None, None], (5, 201, 201))
            maps = np.stack((c, *constants, 2 * self.x / 2000 - 1, 2 * self.z / 2000 - 1, *known))
        maps = np.ascontiguousarray(maps, dtype=np.float32)
        y.flags.writeable = False
        maps.flags.writeable = False
        result = (maps, y)
        if len(self.cache) < self.cache_records:
            self.cache[row] = result
        return result

    def batch(self, rows, device):
        records = [self.record(row) for row in rows]
        x = torch.from_numpy(np.stack([r[0] for r in records])).to(device)
        y = torch.from_numpy(np.stack([r[1] for r in records])).to(device)
        return x, y

    def weights(self, device):
        stats = json.loads((DATA / 'normalization_stats.json').read_text())['families']
        names = ['uniform', 'layered'] if self.case == 'ul' else ['marmousi']
        energy = sum(np.square(np.array(stats[n]['per_frame_rms_pa'])) * stats[n]['records'] for n in names)
        energy /= sum(stats[n]['records'] for n in names)
        w = 1 / np.maximum(energy, energy.max() * 1e-3)
        return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def configuration(args, migration):
    names = ['normalization_unit_range.json', 'normalization.json', 'code/unit_range_loader.py', 'combined_provenance.json', 'DATASET_COMPLETE.json', 'FINAL_VERIFICATION.json', 'time_s.npy']
    return dict(schema='FNO_UNIFORM401_MINMAX_CONTINUATION_V1', case=args.case, epochs=args.epochs,
                records=934 if args.case == 'ul' else 500, additional_record_exposures=args.epochs * (934 if args.case == 'ul' else 500),
                batch_initial=args.batch, lr=1e-4, weight_decay=1e-6, optimizer='new AdamW; weights-only spatial continuation',
                objective='full401 supervised MSE, fixed per-frame inverse TRAIN physical energy weights, mean1, floor1e-3',
                all_output_frames_learned=True, copied_output_pressure_frames=0, time_s=[float(x) for x in np.arange(401) * .0025],
                known_input_indices=KNOWN.tolist(), known_input_times_s=(KNOWN * .0025).tolist(),
                known_input_rule='Explicit supported uniform401 instants near the original observed-time landmarks; no pressure interpolation',
                normalization=PROFILE, metadata_sha256={n: sha(DATA / n) for n in names},
                code_sha256=sha(__file__), inherited_migration=migration, local_only=True,
                historical_epochs=200, new_stage_epochs_reported_separately=True)


def snapshot(model, opt, progress, cfg):
    return dict(schema=cfg['schema'], config=cfg, model=cpu(model.state_dict()), optimizer=cpu(opt.state_dict()),
                progress=copy.deepcopy(progress), rng=cpu(rng()))


def save(run, model, opt, progress, cfg, name='latest.pt'):
    state = snapshot(model, opt, progress, cfg)
    tmp = run / (name + '.tmp')
    torch.save(state, tmp)
    readback = torch.load(tmp, map_location='cpu', weights_only=False)
    assert equal(state, readback), 'Full model/Adam/RNG/cursor checkpoint readback failed'
    os.replace(tmp, run / name)
    return state


def restore(case, state, device):
    model, _ = make_model(case, inherit=False)
    model.to(device).load_state_dict(state['model'], strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6, foreach=False)
    opt.load_state_dict(state['optimizer'])
    assert equal(model.state_dict(), state['model']) and equal(opt.state_dict(), state['optimizer'])
    set_rng(state['rng'])
    return model, opt, copy.deepcopy(state['progress'])


def train(args):
    run = RUNTIME / args.case
    run.mkdir(parents=True, exist_ok=True)
    owner = (run / 'owner.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    worker = dict(pid=os.getpid(), proc_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
                  started_utc=utc(), command=sys.argv, cwd=os.getcwd())
    write(run / 'WORKER.json', worker)
    random.seed(20261028)
    np.random.seed(20261028)
    torch.manual_seed(20261028)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    device = 'cuda:0'
    data = Corpus(args.case, args.cache_records)
    model, migration = make_model(args.case)
    cfg = configuration(args, migration)
    write(run / 'MIGRATION.json', migration)
    if (run / 'CONFIG.json').exists():
        assert json.loads((run / 'CONFIG.json').read_text()) == cfg, 'Frozen configuration changed'
    else:
        write(run / 'CONFIG.json', cfg)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6, foreach=False)
    p = dict(epoch=0, offset=0, updates=0, exposures=0, actual_batch=args.batch, best_train_panel_relative_l2=None)
    if (run / 'latest.pt').exists():
        state = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
        assert state['config'] == cfg
        del model, opt
        model, opt, p = restore(args.case, state, device)
    else:
        save(run, model, opt, p, cfg)
    def stop(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    w = data.weights(device)
    def status(phase, **extra):
        write(run / 'STATUS.json', dict(status=phase, utc=utc(), **worker, **p,
              target_epochs=args.epochs, target_exposures=cfg['additional_record_exposures'],
              pressure_range=[-1, 1], time_frames=401, time_dt_s=.0025, cache_records=len(data.cache), **extra))
    def panel():
        rows = [data.rows[0], data.rows[-1]]
        if args.case == 'ul':
            rows = [next(r for r in data.rows if data.families[r] == n) for n in ['uniform', 'layered']]
        total_sse = 0.
        for row in rows:
            x, y = data.batch([row], device)
            with torch.no_grad():
                prediction = model(x)
                total_sse += float((prediction - y).square().sum(dtype=torch.float64)) * PROFILE['pressure_half_range_pa'] ** 2
            del x, y, prediction
        return float(np.sqrt(total_sse / sum(data.energy[r] for r in rows)))
    if not (run / 'INITIAL_PANEL.json').exists():
        status('INITIAL_TRAIN_PANEL')
        write(run / 'INITIAL_PANEL.json', dict(relative_l2=panel(), scope='two fixed TRAIN records, all401 frames, decoded physical Pa', utc=utc()))
    started = time.monotonic()
    try:
        while p['epoch'] < args.epochs and not STOP:
            order = np.random.default_rng(np.random.SeedSequence([20261028, p['epoch']])).permutation(len(data.rows))
            while p['offset'] < len(order) and not STOP:
                count = min(p['actual_batch'], len(order) - p['offset'])
                rows = [data.rows[int(i)] for i in order[p['offset']:p['offset'] + count]]
                status('TRAINING', current_rows=rows)
                t = time.monotonic()
                x, y = data.batch(rows, device)
                opt.zero_grad(set_to_none=True)
                try:
                    prediction = model(x)
                    assert prediction.shape == y.shape == (count, 401, 201, 201)
                    loss = ((prediction - y).square().mean(dim=(2, 3)) * w[None]).mean()
                    assert bool(torch.isfinite(loss))
                    loss.backward()
                    assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters())
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=False)
                    assert bool(torch.isfinite(norm))
                except torch.cuda.OutOfMemoryError:
                    # No optimizer update has happened, so retry the same cursor safely.
                    if count == 1:
                        raise
                    opt.zero_grad(set_to_none=True)
                    del x, y
                    if 'prediction' in locals():
                        del prediction
                    if 'loss' in locals():
                        del loss
                    gc.collect()
                    torch.cuda.empty_cache()
                    p['actual_batch'] = max(1, count // 2)
                    append(run / 'OOM_RETRIES.jsonl', dict(utc=utc(), offset=p['offset'], batch=p['actual_batch']))
                    continue
                opt.step()
                torch.cuda.synchronize()
                value = float(loss.detach())
                p['offset'] += count
                p['updates'] += 1
                p['exposures'] += count
                report = dict(utc=utc(), **p, batch=count, loss=value, gradient_norm=float(norm), seconds=time.monotonic() - t,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(), cache_records=len(data.cache))
                append(run / 'metrics.jsonl', report)
                print(json.dumps(report), flush=True)
                del prediction, loss, y
                if p['updates'] == 8:
                    state = save(run, model, opt, p, cfg)
                    with torch.no_grad():
                        expected = model(x[:1]).detach().cpu()
                    del model, opt
                    gc.collect()
                    torch.cuda.empty_cache()
                    model, opt, p = restore(args.case, state, device)
                    with torch.no_grad():
                        actual = model(x[:1]).detach().cpu()
                    assert torch.equal(expected, actual), 'Fresh CUDA restore prediction mismatch'
                    write(run / 'CUDA_RESTORE8.json', dict(status='PASS', update=8, model_Adam_RNG_cursor_exact=True,
                          prediction_bitwise=True, full401=True, source='actual corrected-data training', utc=utc()))
                    del state, expected, actual
                del x
                if p['updates'] == 9:
                    save(run, model, opt, p, cfg)
                    write(run / 'LAUNCH_VERIFIED.json', dict(status='PASS', actual_restored_object_update9=True,
                          exposures=p['exposures'], full401_gradients_finite=True, utc=utc()))
                elif p['updates'] % 32 == 0:
                    save(run, model, opt, p, cfg)
                status('TRAINING')
            if p['offset'] == len(order):
                p['epoch'] += 1
                p['offset'] = 0
                rel = panel()
                if p['best_train_panel_relative_l2'] is None or rel < p['best_train_panel_relative_l2']:
                    p['best_train_panel_relative_l2'] = rel
                    save(run, model, opt, p, cfg, 'best.pt')
                save(run, model, opt, p, cfg)
                append(run / 'epochs.jsonl', dict(utc=utc(), **p, train_panel_relative_l2=rel,
                       exact_once_per_epoch=True, train_records=len(data.rows)))
        save(run, model, opt, p, cfg)
        phase = 'STOPPED_BY_USER' if STOP else 'COMPLETE'
        status(phase, elapsed_seconds=time.monotonic() - started)
        write(run / 'RESULT.json', dict(status=phase, utc=utc(), **p, historical_epochs=200,
              new_time_normalization_stage=True, heldout_evaluation_executed=False))
    except BaseException as error:
        status('FAILED', error=repr(error))
        write(run / 'ERROR.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=utc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['ul', 'marmousi'], required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--cache-records', type=int, default=1000)
    train(parser.parse_args())
