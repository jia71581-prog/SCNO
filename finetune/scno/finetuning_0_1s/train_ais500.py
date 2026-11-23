#!/usr/bin/env python3
"""500 supervised updates on the uniform 0--1 s grid from the pretrained operator.

Every update scans all 801 reference frames and adds an AIS-sampled LWC PDE
residual.  The pretrained operator is ``marmousi_epoch65.pt`` with the frozen
full-400 readout patch and rank-8 refiner LoRA, matching the existing
full-supervision refit architecture.
"""
from scno_release_paths import local_asset as _scno_local_asset
from pathlib import Path
import os, sys, json, time, math, hashlib, traceback, argparse, copy, scno_posix_lock as fcntl

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

R = Path(_scno_local_asset('runtime'))
PACKAGE = Path(__file__).resolve().parents[3]
RUNTIME = PACKAGE / 'finetune/scno/runtime'
WORK = R / 'ours_pretrained_uniform801_ais500_20261108'
BASE = R / 'ours_full400_receiver_refit'
ROOT = PACKAGE / 'shared/qinghua_operator'
CHECKPOINT = Path(_scno_local_asset('project/paper_assets_main_fresh_20261013_marmousi_20261019/models/marmousi_epoch65.pt'))
SUPPORT = RUNTIME / 'final_algorithm_revision/evidence_code_v4/support'
FULL801_CODE = RUNTIME / 'ours_finetuning_full801_soap_continue_20261107/code'
sys.path.insert(0, str(RUNTIME / BASE.name / 'code'))
sys.path.insert(0, str(RUNTIME / 'primary_evidence_revision/early_nonlinear_probe'))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(SUPPORT))
sys.path.insert(0, str(FULL801_CODE))

from refit import load
from model_adapter import FrozenBaseCache, _Rank8Weight
from probe import MarmSession
from native400_runtime.cf_expanded_lora import ConvLoRA, LinearLoRA
from full801_policy import (adaptive_frame_weights, pooled_percent,
                            window_errors_801, window_errors_400compat)
from batch400_memory import enable_block_checkpointing
from operators import (AISSampler, make_valid_mask, residual_scale_squared,
                      lwc_residual)

REF801 = R / 'fno_fulltime_supervised/reference/training_reference_pa.npy'
TIME801 = R / 'fno_fulltime_supervised/reference/time_s.npy'
HEAD_FIT = BASE / 'pretrained_init/HEAD_FIT.pt'
DT = 0.00125
N = 801
DELTA = 0.0005
PDE_WEIGHT = 0.001
MIN_T = 0.0005
MAX_T = 1.0 - 0.0005
SEED = 20261108
CHUNK = 267
UPDATES = 500
SCHEMA = 'PRETRAINED_UNIFORM801_AIS500_V1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def atomic_torch(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def receiver_percent(prediction, reference, indices=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if indices is not None:
        prediction = prediction[indices]
        reference = reference[indices]
    denominator = float(np.linalg.norm(reference))
    if denominator <= 1e-30:
        return None
    return 100.0 * float(np.linalg.norm(prediction - reference) / denominator)


def receiver_rmse_pa(prediction, reference, indices=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if indices is not None:
        prediction = prediction[indices]
        reference = reference[indices]
    return float(np.sqrt(np.mean(np.square(prediction - reference))))


def trap_weights(times):
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or not np.all(np.diff(times) > 0):
        raise ValueError('Strictly increasing time vector required')
    weights = np.empty_like(times)
    weights[0] = (times[1] - times[0]) / 2
    weights[-1] = (times[-1] - times[-2]) / 2
    weights[1:-1] = (times[2:] - times[:-2]) / 2
    return weights


def time_weighted_percent(prediction, reference, times):
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    weights = trap_weights(times)
    denominator = float(np.sum(weights * np.square(reference)))
    if denominator <= 1e-30:
        return None
    return 100.0 * float(np.sqrt(np.sum(weights * np.square(prediction - reference)) / denominator))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=UPDATES)
    parser.add_argument('--chunk', type=int, default=CHUNK)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def stat(stage, **kw):
        write(OUT / 'STATUS.json', dict(stage=stage, pid=os.getpid(), **kw))
        print(stage, kw, flush=True)

    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 3090':
        raise RuntimeError('Expected RTX3090 GPU0')
    free, total = torch.cuda.mem_get_info()
    reserve = 768 * 1024 ** 2
    if free < 3 * 1024 ** 3:
        raise RuntimeError('Insufficient GPU free memory before init')
    torch.cuda.set_per_process_memory_fraction(min(.98, (free - reserve) / total), 0)
    wall = time.perf_counter()

    ckpt_path = CHECKPOINT
    ckpt_sha = sha(ckpt_path)
    head_sha = sha(HEAD_FIT)
    truth_sha = sha(REF801)
    time_sha = sha(TIME801)

    stat('load')
    model, x, ckpt, _ = load('pretrained_init')
    if Path(ckpt) != ckpt_path:
        raise RuntimeError(f'Unexpected pretrained checkpoint {ckpt}')
    x['velocity'] = x['velocity'].to('cuda')
    x['source'] = x['source'].to('cuda')
    scale = float(x['source'][4]) * x['pressure_scale_pa']
    times_np = np.load(TIME801).astype(np.float64)
    if times_np.shape != (N,):
        raise RuntimeError('Expected 801 reference times')
    if abs(times_np[0]) > 1e-12 or abs(times_np[-1] - 1.0) > 1e-9:
        raise RuntimeError('Reference times must span 0-1 s')
    if not np.allclose(np.diff(times_np), DT, rtol=0, atol=1e-12):
        raise RuntimeError('Reference time grid is not uniform 1.25 ms')
    times = torch.as_tensor(times_np, device='cuda', dtype=torch.float32)
    truth = np.load(REF801, mmap_mode='r', allow_pickle=False)
    if truth.shape != (N, 201, 201):
        raise RuntimeError(f'Unexpected reference shape {truth.shape}')

    # Frozen readout patch, then the same 42 rank-8 refiner sites used by the
    # existing full-supervision refits.  The pretrained checkpoint has no LoRA.
    head = model.refiner.output
    if isinstance(head, nn.Conv2d):
        gen = torch.Generator(device='cpu').manual_seed(SEED)
        head = ConvLoRA(head, 1, gen).to('cuda')
        model.refiner.output = head
    patch = torch.load(HEAD_FIT, map_location='cpu', weights_only=False)
    with torch.no_grad():
        head.A.copy_(patch['A'])
        head.B.copy_(patch['B'])
        head.base.bias.copy_(patch['bias'])
    model.eval().requires_grad_(False)
    for name, module in [(n, v) for n, v in model.named_modules()
                         if isinstance(v, (ConvLoRA, LinearLoRA))]:
        parent, leaf = name.rsplit('.', 1)
        replacement = copy.deepcopy(module.base)
        with torch.no_grad():
            replacement.weight.copy_(module.weight)
        if replacement.bias is not None:
            with torch.no_grad():
                replacement.bias.copy_(module.bias)
        model.get_submodule(parent)._modules[leaf] = replacement
    sites = [(n, v) for n, v in model.named_modules()
             if n.startswith('refiner.') and not n.startswith('refiner.dg.')
             and isinstance(v, (nn.Conv2d, nn.Linear)) and not v.weight.is_complex()
             and (not isinstance(v, nn.Conv2d) or v.groups == 1)]
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    params = []
    for _, module in sites:
        low = _Rank8Weight(module.weight, gen)
        parametrize.register_parametrization(module, 'weight', low)
        params.extend((low.A, low.B))
    if len(sites) != 42 or sum(p.numel() for p in params) != 121648:
        raise RuntimeError('Unexpected LoRA architecture')
    model.eval()
    if any(p.requires_grad for p in model.base.parameters()) or not model.refiner.dg.enabled:
        raise RuntimeError('Unexpected base/refiner trainability')
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if len(names) != len(params):
        raise RuntimeError('Trainable parameter list does not match LoRA sites')

    t400 = x['full_times'].detach().cpu().numpy().astype(np.float64)
    idx400 = np.array([int(np.argmin(np.abs(times_np - t))) for t in t400], dtype=np.int64)
    if len(np.unique(idx400)) != 400 or idx400[0] != 88 or idx400[-1] != 800:
        raise RuntimeError('Could not map the 400-frame times onto the 801-frame grid')
    if not np.all(np.abs(times_np[idx400] - t400) < 3e-8):
        raise RuntimeError('400-frame time mapping is not exact')
    idx488 = np.r_[np.arange(88, dtype=np.int64), idx400]
    ref_receiver = np.asarray(truth[:, 2, 50], dtype=np.float64)

    stat('cache_801')
    cache = FrozenBaseCache(model, x['velocity'], x['source'], cache_device='cpu', block=1)
    session = MarmSession(model, x, cache)
    cache.prefill(times)

    energy = np.empty(N, dtype=np.float64)
    for i in range(N):
        y = np.asarray(truth[i], np.float64) / scale
        energy[i] = np.square(y).mean()
    if not np.isfinite(energy).all() or (energy < 0).any() or energy.sum() <= 0:
        raise RuntimeError('Invalid reference energy')

    def target(indices):
        return torch.as_tensor(np.array(truth[indices], copy=True),
                               device='cuda', dtype=torch.float32) / scale

    def scan_eval(chunk):
        result = np.zeros(N, np.float64)
        receiver = np.zeros(N, np.float64)
        with torch.no_grad(), session.training_scope():
            for j in range(0, N, chunk):
                ix = np.arange(j, min(j + chunk, N))
                z = session.query(times[ix], grad=False)
                y = target(ix)
                per = (z - y).square().mean((1, 2))
                result[ix] = per.double().cpu().numpy()
                receiver[ix] = (z[:, 2, 50].double() * scale).cpu().numpy()
                del z, y, per
        return result, receiver

    def metrics_from(mse, receiver):
        full801 = pooled_percent(mse, energy)
        full400 = pooled_percent(mse[idx400], energy[idx400])
        post392 = pooled_percent(mse[idx400[8:]], energy[idx400[8:]])
        obs8 = pooled_percent(mse[idx400[:8]], energy[idx400[:8]])
        return {
            'full801_percent': full801,
            'full400_percent': full400,
            'post_early392_percent': post392,
            'observation8_0p11_0p1725_percent': obs8,
            'first8_uniform_absolute_rmse_Pa': float(scale * np.sqrt(np.mean(mse[:8]))),
            'first8_uniform_reference_rms_Pa': float(scale * np.sqrt(np.mean(energy[:8]))),
            'receiver801_percent': receiver_percent(receiver, ref_receiver),
            'receiver488_percent': receiver_percent(receiver, ref_receiver, idx488),
            'receiver488_time_weighted_percent': time_weighted_percent(
                receiver[idx488], ref_receiver[idx488], times_np[idx488]),
            'receiver400_percent': receiver_percent(receiver, ref_receiver, idx400),
            'receiver_observation8_percent': receiver_percent(
                receiver, ref_receiver, idx400[:8]),
            'receiver_observation8_absolute_rmse_Pa': receiver_rmse_pa(
                receiver, ref_receiver, idx400[:8]),
            'receiver_early88_absolute_rmse_Pa': receiver_rmse_pa(
                receiver, ref_receiver, np.arange(88)),
            'receiver_early88_reference_rms_Pa': float(
                np.sqrt(np.mean(np.square(ref_receiver[:88])))),
            'window_errors_801_percent': window_errors_801(mse, energy),
            'window_errors_400compat_percent': window_errors_400compat(
                mse, energy, idx400),
        }

    baseline_path = OUT / 'BASELINE_801.json'
    baseline_mse_path = OUT / 'baseline_mse.npy'
    baseline_receiver_path = OUT / 'baseline_receiver_Pa.npy'
    energy_path = OUT / 'reference_energy.npy'
    reuse_baseline = False
    if (baseline_path.exists() and baseline_mse_path.exists()
            and baseline_receiver_path.exists() and energy_path.exists()):
        saved = json.loads(baseline_path.read_text())
        reuse_baseline = (
            saved.get('checkpoint_sha256') == ckpt_sha
            and saved.get('head_fit_sha256') == head_sha
            and saved.get('truth_sha256') == truth_sha
            and saved.get('time_sha256') == time_sha
            and saved.get('frames') == N
            and saved.get('baseline_mse_sha256') == sha(baseline_mse_path)
            and saved.get('baseline_receiver_sha256') == sha(baseline_receiver_path)
            and saved.get('reference_energy_sha256') == sha(energy_path))
    if reuse_baseline:
        baseline_mse = np.load(baseline_mse_path)
        baseline_receiver = np.load(baseline_receiver_path)
        saved_energy = np.load(energy_path)
        if (baseline_mse.shape != (N,) or baseline_receiver.shape != (N,)
                or not np.allclose(saved_energy, energy, rtol=0, atol=1e-12)):
            reuse_baseline = False
    if not reuse_baseline:
        stat('baseline_801')
        baseline_mse, baseline_receiver = scan_eval(args.chunk)
        np.save(baseline_mse_path, baseline_mse)
        np.save(baseline_receiver_path, baseline_receiver)
        np.save(energy_path, energy)
    initial_metrics = metrics_from(baseline_mse, baseline_receiver)

    # The first eight 0--0.00875 s reference fields are numerically zero
    # (RMS ~2.4e-17 Pa), so they are supervised as data but cannot define the
    # AIS residual scale.  Use the eight registered observation frames at
    # 0.11--0.1725 s for that fixed scale.
    early_scale_values = torch.as_tensor(
        np.array(truth[idx400[:8]], copy=True), device='cuda', dtype=torch.float32) / scale
    nu_squared = residual_scale_squared(early_scale_values, f0=15.)
    nu_value = float(nu_squared.detach().cpu())
    source_xy = (float(x['source'][0]), float(x['source'][1]))
    mask = make_valid_mask(shape=(201, 201), h=10., source=source_xy, halo=8, radius=90.)
    sampler = AISSampler(mask, n_times=N, tile_size=16, beta_uniform=.1,
                         ema=.9, seed=SEED)
    first8_uniform_rms_pa = float(np.sqrt(np.mean(
        np.square(np.asarray(truth[:8], dtype=np.float64)))))

    baseline_record = {
        'schema': SCHEMA,
        'checkpoint': str(ckpt_path),
        'checkpoint_sha256': ckpt_sha,
        'head_fit': str(HEAD_FIT),
        'head_fit_sha256': head_sha,
        'truth': str(REF801),
        'truth_sha256': truth_sha,
        'time_grid': str(TIME801),
        'time_grid_sha256': time_sha,
        'frames': N,
        'scale_Pa': scale,
        'initial_metrics': initial_metrics,
        'initial_full801_percent': initial_metrics['full801_percent'],
        'initial_full400_percent': initial_metrics['full400_percent'],
        'first8_uniform_reference_rms_Pa': first8_uniform_rms_pa,
        'first8_uniform_zero_note': (
            '0--0.00875 s reference fields are numerically zero; they are '
            'supervised as data and their residual scale is not used for AIS.'),
        'ais_normalization_frame_indices': [int(idx400[0]), int(idx400[7])],
        'ais_normalization_time_range_s': [float(times_np[idx400[0]]),
                                          float(times_np[idx400[7]])],
        'ais_nu_squared': nu_value,
        'ais_mask_valid_nodes': int(mask.sum()),
        'ais_tiles': int(len(sampler.n_nodes)),
        'baseline_mse_sha256': sha(baseline_mse_path),
        'baseline_receiver_sha256': sha(baseline_receiver_path),
        'reference_energy_sha256': sha(energy_path),
    }
    write(baseline_path, baseline_record)
    stat('baseline_ready', initial_full801=initial_metrics['full801_percent'],
         initial_full400=initial_metrics['full400_percent'],
         receiver488=initial_metrics['receiver488_time_weighted_percent'])

    cfg = {
        'schema': SCHEMA,
        'scope': ('single instance; all 801 uniform 0--1 s reference frames '
                  'supervised on every optimizer update; no unseen-time claim'),
        'initialization': 'marmousi_epoch65.pt + frozen full-400 HEAD_FIT + rank-8 refiner LoRA (121648 parameters)',
        'checkpoint': str(ckpt_path),
        'checkpoint_sha256': ckpt_sha,
        'head_fit': str(HEAD_FIT),
        'head_fit_sha256': head_sha,
        'truth': str(REF801),
        'truth_sha256': truth_sha,
        'time_grid': str(TIME801),
        'time_grid_sha256': time_sha,
        'frames_per_update': N,
        'batch_size': N,
        'optimizer': 'AdamW lr1e-4 weight_decay0 betas(.9,.999), gradient clip1',
        'target_updates': int(args.updates),
        'early_window': '0.11--0.1725 s (eight registered observation frames)',
        'weighting': ('EMA beta .9 of frame MSE; w=temporal_prior*(1+3*clip(EMA/q90,0,1)); '
                      'prior2 for first100 and last100; all 801 frames active'),
        'data_loss': 'sum(w_i*mean((p_i-y_i)^2))/(801*mean_801(w_i*mean(y_i^2)))',
        'ais': {
            'enabled': True,
            'pde_weight': PDE_WEIGHT,
            'n_time_draws': 8,
            'n_tile_draws': 8,
            'tile_size': 16,
            'beta_uniform': .1,
            'ema': .9,
            'seed': SEED,
            'delta_s': DELTA,
            'nu_squared': nu_value,
            'normalization_frames': [int(idx400[0]), int(idx400[7])],
            'normalization_note': ('first eight 0--1 s frames are numerically zero; '
                                   'the fixed scale uses the first eight nonzero '
                                   '0.11--0.1725 s observation frames'),
            'mask_nodes': int(mask.sum()),
            'tiles': int(len(sampler.n_nodes)),
        },
        'selection': 'lowest pooled full801 field error; receiver not used',
        'selection_uses_receiver': False,
        'pde_receiver_loss': False,
        'new_data_generation': False,
        'start_from_pretrained': True,
    }
    write(OUT / 'CONFIG.json', cfg)

    resume_path = OUT / 'resume.pt'
    resume_exists = resume_path.exists()
    if resume_exists:
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        if (state.get('schema') != SCHEMA
                or state.get('checkpoint_sha256') != ckpt_sha
                or state.get('head_fit_sha256') != head_sha
                or state.get('truth_sha256') != truth_sha
                or state.get('time_sha256') != time_sha
                or state.get('frames') != N):
            raise RuntimeError('Resume state does not match this run')
        set_active_state(model, state['active'])
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0., betas=(.9, .999))
        opt.load_state_dict(state['optimizer'])
        ema = np.asarray(state['ema'], dtype=np.float64)
        weights = np.asarray(state['weights'], dtype=np.float64)
        best_error = float(state['best_error'])
        best_step = int(state['best_step'])
        best_state = state['best_state']
        complete = int(state['complete'])
        args.chunk = int(state['chunk'])
        sampler.load_state_dict(state['sampler'])
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        stat('resume', complete=complete, best_step=best_step,
             best_percent=best_error, chunk=args.chunk)
    else:
        complete = 0
        best_error = initial_metrics['full801_percent']
        best_step = 0
        best_state = active_state(model)
        ema = baseline_mse.copy()
        weights = adaptive_frame_weights(ema)
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0., betas=(.9, .999))
        write(OUT / 'START_801.json', {
            'schema': SCHEMA,
            'checkpoint_sha256': ckpt_sha,
            'head_fit_sha256': head_sha,
            'truth_sha256': truth_sha,
            'time_sha256': time_sha,
            'supervision_frames': N,
            'supervision_time_range_s': [0.0, 1.0],
            'supervision_dt_s': DT,
            'initial_full801_percent': initial_metrics['full801_percent'],
            'initial_full400_percent': initial_metrics['full400_percent'],
            'initial_receiver488_time_weighted_percent':
                initial_metrics['receiver488_time_weighted_percent'],
            'optimizer': cfg['optimizer'],
            'ais': cfg['ais'],
            'target_updates': int(args.updates),
        })

    start_complete = int(complete)
    if ema.shape != (N,) or weights.shape != (N,):
        raise RuntimeError('Resume EMA/weights have the wrong length')
    if args.chunk < 16:
        raise RuntimeError('Training chunk must be at least 16')

    def scan_train(chunk, weights_now, track_mem=False):
        result = np.zeros(N, np.float64)
        den = float(np.mean(weights_now * energy))
        mem = {}
        data_total = 0.0
        opt.zero_grad(set_to_none=True)
        with session.training_scope():
            for j in range(0, N, chunk):
                ix = np.arange(j, min(j + chunk, N))
                y = target(ix)
                ww = torch.as_tensor(weights_now[ix], device='cuda', dtype=torch.float32)
                z = session.query(times[ix], grad=True)
                per = (z - y).square().mean((1, 2))
                loss = (per * ww).sum() / (N * den)
                result[ix] = per.detach().double().cpu().numpy()
                data_total += float(loss.detach())
                loss.backward()
                if track_mem:
                    torch.cuda.synchronize()
                    mem['minimum_free'] = min(mem.get('minimum_free', 2 ** 63 - 1),
                                              int(torch.cuda.mem_get_info()[0]))
                del z, y, ww, per, loss
        return result, data_total, mem

    def scan_train_retry(chunk, weights_now, track_mem=False):
        while True:
            try:
                result, data_total, mem = scan_train(chunk, weights_now,
                                                     track_mem=track_mem)
                return result, data_total, chunk, mem
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                chunk = max(16, chunk - 32)
                stat('oom_retry_scan', chunk=chunk)
                if chunk < 16:
                    raise RuntimeError('OOM prevents a safe training chunk')

    def ais_loss_step():
        plan = sampler.draw(mode='ais', n_time_draws=8, n_tile_draws=8)
        selected = np.asarray(plan['time_indices'], dtype=np.int64)
        n = len(selected)
        t = times[torch.as_tensor(selected, device='cuda')].clamp(MIN_T, MAX_T)
        query_times = torch.cat((t - DELTA, t, t + DELTA))
        with session.training_scope():
            z = session.query(query_times, grad=True)
            residual = lwc_residual(z[:n], z[n:2 * n], z[2 * n:], x['velocity'])
            energies = sampler.tile_energies(residual, nu_squared)
            loss = sampler.loss(energies, plan)
        finite = bool(torch.isfinite(loss) and torch.isfinite(energies).all())
        if finite:
            (PDE_WEIGHT * loss).backward()
            sampler.commit(plan, energies.detach())
        return (float(loss.detach()) if torch.isfinite(loss) else None,
                finite, selected, float(energies.detach().mean()))

    blocks = enable_block_checkpointing(model)
    if len(blocks) != 10:
        raise RuntimeError('Expected 10 checkpointed blocks')

    def fingerprint():
        h = hashlib.sha256()
        for p in params:
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
            state = opt.state.get(p, {})
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                value = state.get(key)
                if torch.is_tensor(value):
                    h.update(value.detach().cpu().contiguous().numpy().tobytes())
                elif value is not None:
                    h.update(repr(value).encode())
        return h.hexdigest()

    hist = None
    if not resume_exists:
        stat('ais_initial_history')
        hist = np.zeros((N, len(sampler.n_nodes)), dtype=np.float64)
        with torch.no_grad(), session.training_scope():
            for j in range(0, N, 32):
                ix = np.arange(j, min(j + 32, N))
                n = len(ix)
                t = times[ix].clamp(MIN_T, MAX_T)
                query_times = torch.cat((t - DELTA, t, t + DELTA))
                z = session.query(query_times, grad=False)
                residual = lwc_residual(z[:n], z[n:2 * n], z[2 * n:], x['velocity'])
                e = sampler.tile_energies(residual, nu_squared)
                hist[ix] = e.detach().double().cpu().numpy()
                del z, residual, e
        bad = ~np.isfinite(hist).all(axis=1)
        if bad.any():
            hist[bad] = 0.0
        sampler.initialize_history(hist)
        np.save(OUT / 'ais_initial_history.npy', hist)
        stat('ais_initial_ready', bad_rows=int(bad.sum()),
             mean=float(hist.mean()), p90=float(np.quantile(hist, .9)),
             max=float(hist.max()))

        stat('preflight')
        before = fingerprint()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.perf_counter()
        probe_mse, probe_data, chosen_chunk, probe_mem = scan_train_retry(
            args.chunk, weights, track_mem=True)
        finite_grad = all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                          for p in params)
        probe_sampler = copy.deepcopy(sampler)
        plan = probe_sampler.draw(mode='ais', n_time_draws=8, n_tile_draws=8)
        selected = np.asarray(plan['time_indices'], dtype=np.int64)
        n = len(selected)
        t = times[torch.as_tensor(selected, device='cuda')].clamp(MIN_T, MAX_T)
        query_times = torch.cat((t - DELTA, t, t + DELTA))
        with session.training_scope():
            z = session.query(query_times, grad=True)
            residual = lwc_residual(z[:n], z[n:2 * n], z[2 * n:], x['velocity'])
            energies = sampler.tile_energies(residual, nu_squared)
            pde_loss = sampler.loss(energies, plan)
            finite_pde = bool(torch.isfinite(pde_loss) and torch.isfinite(energies).all())
            if finite_pde:
                (PDE_WEIGHT * pde_loss).backward()
        torch.cuda.synchronize()
        probe_seconds = time.perf_counter() - tick
        peak = int(torch.cuda.max_memory_allocated())
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        after = fingerprint()
        free_after = int(torch.cuda.mem_get_info()[0])
        if not finite_grad or not finite_pde:
            raise RuntimeError('Preflight produced nonfinite data or PDE gradients')
        if before != after:
            raise RuntimeError('Preflight changed parameters or optimizer state')
        if probe_mse.shape != (N,) or not np.isfinite(probe_mse).all():
            raise RuntimeError('Preflight returned invalid frame MSE')
        if free_after < reserve:
            raise RuntimeError('Reserved GPU memory margin violated after preflight')
        preflight = {
            'status': 'PASS',
            'frames': N,
            'chunk': chosen_chunk,
            'checkpoint_blocks': blocks,
            'probe_seconds': probe_seconds,
            'probe_peak_allocated_bytes': peak,
            'probe_free_after_bytes': free_after,
            'minimum_free_bytes_after_chunk': probe_mem.get('minimum_free'),
            'finite_data_gradients': finite_grad,
            'finite_pde_gradient': finite_pde,
            'model_and_optimizer_unchanged': before == after,
            'probe_data_loss': probe_data,
            'probe_pde_loss': float(pde_loss.detach()),
            'probe_selected_times': selected.tolist(),
            'optimizer_step_executed': False,
        }
        write(OUT / 'PREFLIGHT.json', preflight)
        stat('preflight_pass', chunk=chosen_chunk, peak_bytes=peak,
             seconds=probe_seconds, pde_loss=preflight['probe_pde_loss'])
        args.chunk = chosen_chunk
        if args.preflight:
            return

    step_times = []
    data_losses = []
    pde_losses = []
    grad_norms = []
    selected_counts = np.zeros(N, dtype=np.int64)
    pde_nonfinite = 0
    history = (json.loads((OUT / 'FULL_METRICS.json').read_text())
               if (OUT / 'FULL_METRICS.json').exists() else [])
    ais_log = OUT / 'AIS_HISTORY.jsonl'
    if resume_exists and ais_log.exists():
        for line in ais_log.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            data_losses.append(float(record['data_loss']))
            grad_norms.append(float(record['grad_norm']))
            if record.get('pde_finite') and record.get('pde_loss') is not None:
                pde_losses.append(float(record['pde_loss']))
                selected = np.asarray(record['selected_time_indices'], dtype=np.int64)
                selected_counts[selected] += 1
            else:
                pde_nonfinite += 1
    while complete < int(args.updates):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        frame_mse, data_loss, args.chunk, _ = scan_train_retry(
            int(args.chunk), weights)
        current = pooled_percent(frame_mse, energy)
        if current < best_error:
            best_error = current
            best_step = complete
            best_state = active_state(model)
            atomic_torch(OUT / 'best_active.pt', {
                'active': best_state,
                'complete': best_step,
                'full801_percent': best_error,
                'source': 'pre_update',
            })
        pde_loss, pde_finite, selected, e_mean = ais_loss_step()
        if not pde_finite:
            pde_nonfinite += 1
        else:
            pde_losses.append(pde_loss)
            selected_counts[selected] += 1
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            params, 1.0, error_if_nonfinite=True))
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError('Nonfinite or zero total gradient; no step')
        opt.step()
        complete += 1
        ema = .9 * ema + .1 * frame_mse
        weights = adaptive_frame_weights(ema)
        data_losses.append(data_loss)
        grad_norms.append(grad_norm)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        step_times.append(elapsed)

        record = {
            'complete': complete,
            'preupdate_full801_percent': current,
            'best_step': best_step,
            'best_percent': best_error,
            'data_loss': data_loss,
            'pde_loss': pde_loss,
            'pde_finite': pde_finite,
            'pde_energy_mean': e_mean,
            'grad_norm': grad_norm,
            'selected_time_indices': selected.tolist(),
            'selected_times_s': times_np[selected].tolist(),
            'step_s': elapsed,
        }
        with ais_log.open('a') as fp:
            fp.write(json.dumps(record) + '\n')

        snapshot = {
            'schema': SCHEMA,
            'checkpoint_sha256': ckpt_sha,
            'head_fit_sha256': head_sha,
            'truth_sha256': truth_sha,
            'time_sha256': time_sha,
            'frames': N,
            'complete': complete,
            'active': active_state(model),
            'optimizer': opt.state_dict(),
            'optimizer_type': 'AdamW',
            'ema': ema,
            'weights': weights,
            'best_error': best_error,
            'best_step': best_step,
            'best_state': best_state,
            'chunk': int(args.chunk),
            'sampler': sampler.state_dict(),
            'torch_rng': torch.get_rng_state(),
            'numpy_rng': np.random.get_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(),
        }
        atomic_torch(resume_path, snapshot)

        if complete % 50 == 0:
            eval_mse, eval_receiver = scan_eval(int(args.chunk))
            eval_record = {
                'complete': complete,
                **metrics_from(eval_mse, eval_receiver),
            }
            if eval_record['full801_percent'] < best_error:
                best_error = eval_record['full801_percent']
                best_step = complete
                best_state = active_state(model)
                atomic_torch(OUT / 'best_active.pt', {
                    'active': best_state,
                    'complete': best_step,
                    'full801_percent': best_error,
                    'source': 'periodic_eval',
                })
            with (OUT / 'PERIODIC_EVAL.jsonl').open('a') as fp:
                fp.write(json.dumps(eval_record) + '\n')
            snapdir = OUT / 'snapshots'
            snapdir.mkdir(exist_ok=True)
            atomic_torch(snapdir / f'active_{complete:04d}.pt', {
                'schema': SCHEMA,
                'complete': complete,
                'active': active_state(model),
                'eval_mse': eval_mse,
                'eval_receiver_Pa': eval_receiver,
                'eval_record': eval_record,
                'checkpoint_sha256': ckpt_sha,
                'truth_sha256': truth_sha,
                'time_sha256': time_sha,
            })
            stat('periodic_eval', **eval_record)

        if complete <= 2 or complete % 10 == 0:
            eta = (int(args.updates) - complete) * float(np.mean(step_times[-10:]))
            stat('train', complete=complete, chunk=int(args.chunk),
                 metric_state_step=complete - 1,
                 preupdate_full801_percent=current,
                 best_step=best_step, best_percent=best_error,
                 data_loss=data_loss, pde_loss=pde_loss,
                 grad_norm=grad_norm, last_step_s=elapsed,
                 eta_optimization_s=eta,
                 window_errors_percent=window_errors_801(frame_mse, energy))
        if complete % 50 == 0:
            history.append({
                'complete': complete,
                'metric_state_step': complete - 1,
                'preupdate_full801_percent': current,
                'best_step': best_step,
                'best_percent': best_error,
                'step_s': elapsed,
            })
            write(OUT / 'FULL_METRICS.json', history)

    last_mse, last_receiver = scan_eval(int(args.chunk))
    last_error = pooled_percent(last_mse, energy)
    last_state = active_state(model)
    if last_error < best_error:
        best_error = last_error
        best_step = complete
        best_state = last_state

    def full_payload(state, step):
        set_active_state(model, state)
        return {
            'schema': SCHEMA,
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'active': state,
            'added_updates': step,
            'config': cfg,
            'sites': [n for n, _ in sites],
            'checkpoint_sha256': ckpt_sha,
            'head_fit_sha256': head_sha,
        }

    atomic_torch(OUT / 'last_full.pt', full_payload(last_state, complete))
    atomic_torch(OUT / 'best.pt', full_payload(best_state, best_step))
    set_active_state(model, best_state)
    model.eval()
    stat('final_best_prediction', best_step=best_step, best_percent=best_error)

    prediction = np.lib.format.open_memmap(
        OUT / 'prediction801_pa.npy', mode='w+', dtype=np.float32, shape=(N, 201, 201))
    receiver = np.zeros(N, np.float64)
    best_mse = np.zeros(N, np.float64)
    with torch.no_grad(), session.training_scope():
        for j in range(0, N, int(args.chunk)):
            ix = np.arange(j, min(j + int(args.chunk), N))
            z = session.query(times[ix], grad=False)
            pa = (z * scale).cpu().numpy()
            prediction[ix] = pa
            receiver[ix] = pa[:, 2, 50]
            y = np.asarray(truth[ix], np.float64) / scale
            best_mse[ix] = np.square(z.cpu().numpy().astype(np.float64) - y).mean((1, 2))
            del z, pa, y
    prediction.flush()
    del prediction

    np.savetxt(
        OUT / 'per_frame_metrics.csv',
        np.column_stack([np.arange(N), times_np, baseline_mse, best_mse, last_mse,
                         energy, weights]),
        delimiter=',',
        header='frame_index,time_s,initial_mse,best_mse,last_mse,reference_mse,final_adaptive_weight',
        comments='')
    with (OUT / 'receiver_trace.csv').open('w') as stream:
        stream.write('time_s,ours_Pa,reference_Pa\n')
        np.savetxt(stream, np.column_stack((times_np, receiver, ref_receiver)),
                   delimiter=',', fmt='%.18e')

    final_metrics = metrics_from(best_mse, receiver)
    init = initial_metrics
    full801_gain = init['full801_percent'] - final_metrics['full801_percent']
    full400_gain = init['full400_percent'] - final_metrics['full400_percent']
    receiver_gain = (init['receiver488_time_weighted_percent']
                     - final_metrics['receiver488_time_weighted_percent'])
    hist_final = sampler.history
    if hist is not None:
        hist_initial = hist
    elif (OUT / 'ais_initial_history.npy').exists():
        hist_initial = np.load(OUT / 'ais_initial_history.npy')
    else:
        hist_initial = hist_final
    top = np.argsort(selected_counts)[-20:][::-1]
    ais_summary = {
        'enabled': True,
        'pde_weight': PDE_WEIGHT,
        'nu_squared': nu_value,
        'mask_nodes': int(mask.sum()),
        'tiles': int(len(sampler.n_nodes)),
        'n_time_draws': 8,
        'n_tile_draws': 8,
        'tile_size': 16,
        'beta_uniform': .1,
        'ema': .9,
        'seed': SEED,
        'pde_steps': len(pde_losses),
        'pde_nonfinite_steps': pde_nonfinite,
        'pde_loss_mean': float(np.mean(pde_losses)) if pde_losses else None,
        'pde_loss_p90': float(np.quantile(pde_losses, .9)) if pde_losses else None,
        'data_loss_mean': float(np.mean(data_losses)) if data_losses else None,
        'grad_norm_mean': float(np.mean(grad_norms)) if grad_norms else None,
        'selected_time_counts_top20': [
            {'frame_index': int(i), 'time_s': float(times_np[i]),
             'count': int(selected_counts[i])} for i in top],
        'selected_time_count_total': int(selected_counts.sum()),
        'final_history_mean': float(hist_final.mean()),
        'final_history_p90': float(np.quantile(hist_final, .9)),
        'final_history_max': float(hist_final.max()),
        'initial_history_mean': float(hist_initial.mean()),
        'initial_history_p90': float(np.quantile(hist_initial, .9)),
    }
    write(OUT / 'AIS_SUMMARY.json', ais_summary)

    result = {
        'status': ('COMPLETE' if full801_gain > 0 else
                   'COMPLETE_NO_FULL801_IMPROVEMENT'),
        'schema': SCHEMA,
        'checkpoint': str(ckpt_path),
        'checkpoint_sha256': ckpt_sha,
        'head_fit': str(HEAD_FIT),
        'head_fit_sha256': head_sha,
        'new_stage_updates': complete,
        'start_complete': start_complete,
        'additional_updates': int(complete - start_complete),
        'target_updates': int(args.updates),
        'supervision_frames_per_update': N,
        'supervision_time_range_s': [0.0, 1.0],
        'supervision_dt_s': DT,
        'optimizer': cfg['optimizer'],
        'initial_metrics': init,
        'best_metrics': final_metrics,
        'initial_full801_percent': init['full801_percent'],
        'best_full801_percent': final_metrics['full801_percent'],
        'last_full801_percent': last_error,
        'initial_full400_percent': init['full400_percent'],
        'best_full400_percent': final_metrics['full400_percent'],
        'initial_receiver488_time_weighted_percent':
            init['receiver488_time_weighted_percent'],
        'best_receiver488_time_weighted_percent':
            final_metrics['receiver488_time_weighted_percent'],
        'full801_improved': bool(full801_gain > 0),
        'full400_improved': bool(full400_gain > 0),
        'receiver488_improved': bool(receiver_gain > 0),
        'full801_gain_percent_points': float(full801_gain),
        'full400_gain_percent_points': float(full400_gain),
        'receiver488_gain_percent_points': float(receiver_gain),
        'full801_relative_reduction_percent': float(
            100.0 * full801_gain / init['full801_percent']),
        'best_step': int(best_step),
        'selection': 'lowest pooled full801 field error; receiver not used',
        'selection_uses_receiver': False,
        'ais': ais_summary,
        'physical_chunk': int(args.chunk),
        'wall_s': time.perf_counter() - wall,
        'peak_gpu_bytes': int(torch.cuda.max_memory_allocated()),
        'best_sha256': sha(OUT / 'best.pt'),
        'last_sha256': sha(OUT / 'last_full.pt'),
        'prediction_sha256': sha(OUT / 'prediction801_pa.npy'),
        'receiver_trace_sha256': sha(OUT / 'receiver_trace.csv'),
        'first8_uniform_zero_note': baseline_record['first8_uniform_zero_note'],
    }
    write(OUT / 'RESULT.json', result)
    stat('complete', status=result['status'],
         initial_full801=result['initial_full801_percent'],
         best_full801=result['best_full801_percent'],
         initial_full400=result['initial_full400_percent'],
         best_full400=result['best_full400_percent'],
         receiver_initial=result['initial_receiver488_time_weighted_percent'],
         receiver_best=result['best_receiver488_time_weighted_percent'],
         best_step=best_step)


def active_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad}


def set_active_state(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(state[n])


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise
