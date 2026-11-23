#!/usr/bin/env python3
"""Fine-tune the released Marmousi operator on the 801-frame 0--4 s LWC8-4 reference.

Initialization: marmousi_epoch65.pt plus the frozen full-400 readout patch.
The frozen base is queried over 0--4 s by relaxing only its inherited domain
bound (the feature time scale stays 1 s).  The complete 772,273-parameter
refiner (including its DG block and readout) is trained; the 13.8M-parameter
base remains frozen.  Every update supervises all 801 frames at 5 ms.

Loss: per-frame relative MSE with a finite energy floor, so late-time frames
whose reference amplitude is four orders of magnitude smaller are not ignored.
A dynamic hard-frame factor built from an EMA of each frame's relative error
can emphasise the currently hardest frames; its strength is annealed over a
configured update interval.  Selection reports both the pooled full-801 metric
and an equal-per-second balanced metric.
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

R = Path(_scno_local_asset('runtime'))
PACKAGE = Path(__file__).resolve().parents[3]
RUNTIME = PACKAGE / 'finetune/scno/runtime'
WORK = R / 'ours_pretrained_marmousi4s_lwc_20261108'
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
from probe import MarmSession
from native400_runtime.cf_expanded_lora import ConvLoRA, LinearLoRA
from full801_policy import pooled_percent

REF = WORK / 'reference/reference4s_801_pa.npy'
TIME = WORK / 'reference/time_s_801.npy'
RECEIVER = WORK / 'reference/receiver_201x801_pa.npy'
HEAD_FIT = BASE / 'pretrained_init/HEAD_FIT.pt'
OUT = None
N = 801
DT = 0.005
DURATION = 4.0
SEED = 20261108
SCHEMA = 'PRETRAINED_MARMOUSI_4S_LWC801_V1'
UPDATES = 1000
CHUNK = 64
ENERGY_FLOOR_FRACTION = 1e-4


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


class BaseCache4s:
    """Frozen base predictions at arbitrary times in [0,4]; feature scale stays 1 s."""
    def __init__(self, model, velocity, source, cache_device='cpu'):
        self.model = model
        self.device = model.device
        self.cache_device = torch.device(cache_device)
        self.c, self.f = model._inputs(velocity, source)
        with torch.no_grad(), model.scopes():
            self.prepared, self.grid, _ = model._prepare_cf(self.c, self.f)
        z, x = torch.meshgrid(model.z_m, model.x_m, indexing='ij')
        coordinates = torch.stack((2*x/2000-1, 2*z/2000-1,
                                   (x-self.f[0, 0])/2000, (z-self.f[0, 1])/2000))[None]
        self.static = torch.cat((self.c/3000-1, self.c.new_zeros(1, 8, 201, 201),
                                 model._source_map(self.f, 201, 10.), coordinates), 1).detach()
        self._values = {}
        self.generated_queries = 0

    @staticmethod
    def _key(t):
        values = torch.as_tensor(t).detach().cpu().to(torch.float32).reshape(-1).numpy()
        if not np.isfinite(values).all() or (values < 0).any() or (values > DURATION).any():
            raise ValueError('BaseCache4s times must lie in [0,4]')
        return tuple(int(v) for v in values.view(np.uint32))

    def _base_values(self, times):
        m = self.model
        base101 = m.base.dense_normalized(self.prepared, times[None], dense_grid=self.grid,
                                          time_block=1, apply_correction=True)
        pa = m.normalizer.decode_pressure(base101, self.f[:, 4])
        pa201 = m._bicubic201(pa[0, :, None])
        return m.normalizer.encode_pressure(pa201[:, 0][None], self.f[:, 4])[0, :, None]

    def prefill(self, times, block=8):
        values = torch.as_tensor(times, device=self.device, dtype=torch.float32).reshape(-1)
        keys = self._key(values)
        pairs = [(k, v.detach()) for k, v in zip(keys, values) if k not in self._values]
        for start in range(0, len(pairs), block):
            current = pairs[start:start+block]
            t = torch.stack([v for _, v in current])
            with torch.no_grad(), self.model.scopes():
                base = self._base_values(t)
            if not torch.isfinite(base).all():
                raise FloatingPointError('Nonfinite frozen base')
            for (key, _), value in zip(current, base):
                self._values[key] = value.detach().to(self.cache_device).clone()
            self.generated_queries += len(current)
        return self

    def get(self, times):
        keys = self._key(times)
        missing = [k for k in keys if k not in self._values]
        if missing:
            raise KeyError('BaseCache4s prefill did not include every query time')
        return torch.stack([self._values[k] for k in keys]).to(self.device)


def attach_readout_patch(model, head_fit):
    """Collapse the fitted rank-1 readout patch into the plain convolution."""
    head = model.refiner.output
    if isinstance(head, nn.Conv2d):
        gen = torch.Generator(device='cpu').manual_seed(SEED)
        head = ConvLoRA(head, 1, gen).to(model.device)
        model.refiner.output = head
    patch = torch.load(head_fit, map_location='cpu', weights_only=False)
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
    return patch


def relax_base_domain(model):
    def relaxed(self, prepared, time_s):
        values = torch.as_tensor(time_s, dtype=torch.float32, device=self.device)
        records = prepared.source_parameters.shape[0]
        if values.ndim == 1:
            values = values[None].expand(records, -1)
        if values.ndim != 2 or values.shape[0] != records or values.shape[1] == 0:
            raise ValueError('dense time_s must be [time] or [record,time]')
        if (not torch.isfinite(values).all() or torch.any(values < 0)
                or torch.any(values > DURATION)):
            raise ValueError('dense times must lie in [0,4]')
        return values
    import types
    model.base._expand_dense_times = types.MethodType(relaxed, model.base)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--updates', type=int, default=UPDATES)
    ap.add_argument('--chunk', type=int, default=CHUNK)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--early-weight', type=float, default=0.1)
    ap.add_argument('--hard-weight', type=float, default=0.0,
                    help='initial strength of the EMA hard-frame factor (0 disables)')
    ap.add_argument('--hard-final', type=float, default=None,
                    help='final strength after annealing (default: keep --hard-weight)')
    ap.add_argument('--hard-anneal-start', type=int, default=0)
    ap.add_argument('--hard-anneal-end', type=int, default=0)
    ap.add_argument('--hard-ema', type=float, default=0.9)
    ap.add_argument('--hard-quantile', type=float, default=0.9)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--pilot', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'snapshots').mkdir(exist_ok=True)
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
    free, total = torch.cuda.mem_get_info()
    if free < 3 * 1024 ** 3:
        raise RuntimeError('Insufficient free GPU memory')
    torch.cuda.set_per_process_memory_fraction(min(.98, (free - 512 * 1024 ** 2) / total), 0)
    wall = time.perf_counter()

    ckpt = ROOT / 'paper_assets_main_fresh_20261013_marmousi_20261019/models/marmousi_epoch65.pt'
    stat('load')
    model, x, ckpt, _ = load('pretrained_init')
    if Path(ckpt) != ckpt:
        raise RuntimeError(f'Unexpected pretrained checkpoint {ckpt}')
    x['velocity'] = x['velocity'].to('cuda')
    x['source'] = x['source'].to('cuda')
    scale = float(x['source'][4]) * x['pressure_scale_pa']
    times_np = np.load(TIME).astype(np.float64)
    if times_np.shape != (N,) or abs(times_np[0]) > 1e-12 or abs(times_np[-1] - DURATION) > 1e-12:
        raise RuntimeError('Expected the 801-frame 0--4 s axis')
    if not np.allclose(np.diff(times_np), DT, rtol=0, atol=1e-12):
        raise RuntimeError('Expected a uniform 5 ms axis')
    times = torch.as_tensor(times_np, device='cuda', dtype=torch.float32)
    truth = np.load(REF, mmap_mode='r', allow_pickle=False)
    if truth.shape != (N, 201, 201):
        raise RuntimeError(f'Unexpected reference {truth.shape}')

    patch = attach_readout_patch(model, HEAD_FIT)
    relax_base_domain(model)
    model.refiner.requires_grad_(True)
    params = list(model.refiner.parameters())
    if sum(p.numel() for p in params) != 772273:
        raise RuntimeError('Unexpected refiner parameter count')
    if any(p.requires_grad for p in model.base.parameters()):
        raise RuntimeError('Base must remain frozen')
    stat('cache_801')
    cache = BaseCache4s(model, x['velocity'], x['source'], cache_device='cpu')
    cache.prefill(times_np, block=8)
    session = MarmSession(model, x, cache)

    energy = np.empty(N, dtype=np.float64)
    for i in range(N):
        y = np.asarray(truth[i], np.float64) / scale
        energy[i] = np.square(y).mean()
    if not np.isfinite(energy).all() or energy.sum() <= 0:
        raise RuntimeError('Invalid reference energy')
    e_floor = max(float(energy.max()) * ENERGY_FLOOR_FRACTION, 1e-300)
    rel_weights = 1.0 / np.maximum(energy, e_floor)
    rel_weights /= float(rel_weights.mean())
    weights = torch.as_tensor(rel_weights, device='cuda', dtype=torch.float32)
    energy_sum = float(energy.sum())
    energy_early_sum = float(energy[:200].sum())

    ref_receiver = np.asarray(truth[:, 2, 50], np.float64)
    base_refiner_state = {k: v.detach().cpu().clone() for k, v in model.refiner.state_dict().items()}
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0., betas=(.9, .999))

    def scan(chunk, train=False, train_weights=None):
        mse = np.empty(N, np.float64)
        receiver = np.empty(N, np.float64)
        base_mse = np.empty(N, np.float64)
        base_receiver = np.empty(N, np.float64)
        loss_total = 0.0
        w_np = weights if train_weights is None else train_weights
        if train:
            opt.zero_grad(set_to_none=True)
        context = torch.enable_grad() if train else torch.no_grad()
        with context, session.training_scope():
            for j in range(0, N, chunk):
                ix = np.arange(j, min(j + chunk, N))
                ix_t = torch.as_tensor(ix, device='cuda', dtype=torch.long)
                y = torch.as_tensor(np.array(truth[ix], copy=True),
                                    device='cuda', dtype=torch.float32) / scale
                z = session.query(times[ix_t], grad=train)
                per = (z - y).square().mean((1, 2))
                mse[ix] = per.detach().double().cpu().numpy()
                receiver[ix] = (z[:, 2, 50].double() * scale).detach().cpu().numpy()
                if not train:
                    b = cache.get(times[ix_t])[:, 0]
                    b = torch.cat((torch.zeros_like(b[:, :1]), b[:, 1:]), 1)
                    base_mse[ix] = (b - y).square().mean((1, 2)).double().cpu().numpy()
                    base_receiver[ix] = (b[:, 2, 50].double() * scale).cpu().numpy()
                    del b
                else:
                    ww = (w_np[ix_t].to(torch.float32) if torch.is_tensor(w_np)
                          else torch.as_tensor(w_np[ix], device='cuda', dtype=torch.float32))
                    rel = per * ww
                    pooled = per.sum() / energy_sum
                    early = per[ix_t < 200]
                    early_pooled = early.sum() / energy_early_sum
                    # The per-frame relative term drives late-time accuracy;
                    # the pooled all-time and explicit 0--1 s terms keep the
                    # high-energy early window from being sacrificed.
                    loss = (rel.sum() / N + 0.01 * pooled
                            + args.early_weight * early_pooled)
                    loss_total += float(rel.sum().detach() / N)
                    loss.backward()
                del y, z, per
        if train:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            return mse, receiver, loss_total, float(grad_norm)
        return mse, receiver, base_mse, base_receiver

    def metrics(mse, receiver):
        per_second = {f'{a}-{a+1}s': pooled_percent(
            mse[a*200:min((a+1)*200, N)], energy[a*200:min((a+1)*200, N)])
            for a in range(4)}
        return {
            'full801_percent': pooled_percent(mse, energy),
            'per_second_percent': per_second,
            'balanced_per_second_percent': float(np.mean(list(per_second.values()))),
            'three_window_percent': {
                'early': pooled_percent(mse[:267], energy[:267]),
                'middle': pooled_percent(mse[267:534], energy[267:534]),
                'late': pooled_percent(mse[534:], energy[534:])},
            'receiver_percent': 100.0 * float(
                np.linalg.norm(receiver - ref_receiver) / np.linalg.norm(ref_receiver)),
        }

    stat('baseline_801')
    initial_mse, initial_receiver, base_mse, base_receiver = scan(args.chunk, train=False)
    initial = metrics(initial_mse, initial_receiver)
    base_only = metrics(base_mse, base_receiver)
    np.save(OUT / 'baseline_mse.npy', initial_mse, allow_pickle=False)
    np.save(OUT / 'base_only_mse.npy', base_mse, allow_pickle=False)
    np.save(OUT / 'reference_energy.npy', energy, allow_pickle=False)
    gate = {
        'schema': SCHEMA, 'checkpoint': str(ckpt), 'checkpoint_sha256': sha(ckpt),
        'head_fit': str(HEAD_FIT), 'head_fit_sha256': sha(HEAD_FIT),
        'reference': str(REF), 'reference_sha256': sha(REF),
        'times': str(TIME), 'times_sha256': sha(TIME),
        'frames': N, 'dt_s': DT, 'duration_s': DURATION,
        'trainable': 'full refiner (772273 parameters, including DG block and readout)',
        'frozen': 'base (13805766 parameters)',
        'loss': ('mean_t w_t h_t mse_t / N + 0.01 pooled_all + '
                 'early_weight * pooled_0_1s; '
                 'w_t proportional to 1/max(energy_t, 1e-4 max energy); '
                 'h_t = 1 + alpha * clip(EMA_rel_t / q90, 0, 1)'),
        'early_weight': args.early_weight,
        'hard_weight': args.hard_weight,
        'hard_final': args.hard_final,
        'hard_anneal_start': args.hard_anneal_start,
        'hard_anneal_end': args.hard_anneal_end,
        'hard_ema': args.hard_ema,
        'hard_quantile': args.hard_quantile,
        'gpu_free_before_gib': free / 1024 ** 3,
        'initial_metrics': initial,
        'base_only_metrics': base_only,
    }
    write(OUT / 'BASELINE_4S.json', gate)
    if args.preflight:
        stat('preflight_pass', initial=initial)
        return

    update0 = 0
    best_step = 0
    best_balanced = initial['balanced_per_second_percent']
    best_full = initial['full801_percent']
    best_state = copy.deepcopy(base_refiner_state)
    best_full_state = copy.deepcopy(base_refiner_state)
    hard_ema = None
    resume_path = OUT / 'resume.pt'
    if resume_path.exists():
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        if (state.get('schema') == SCHEMA and state.get('reference_sha256') == gate['reference_sha256']
                and state.get('head_fit_sha256') == gate['head_fit_sha256']):
            model.refiner.load_state_dict(state['refiner'], strict=True)
            opt.load_state_dict(state['optimizer'])
            update0 = int(state['complete'])
            best_step = int(state['best_step'])
            best_balanced = float(state['best_balanced'])
            best_full = float(state['best_full'])
            best_state = state['best_refiner']
            best_full_state = state['best_full_refiner']
            if 'hard_ema' in state:
                hard_ema = np.asarray(state['hard_ema'], dtype=np.float64).copy()
            stat('resume', complete=update0, best_step=best_step,
                 best_balanced=best_balanced, best_full=best_full)
    if args.hard_weight > 0:
        if hard_ema is None:
            stat('hard_ema_init')
            current_mse, _, _, _ = scan(args.chunk, train=False)
            hard_ema = current_mse / np.maximum(energy, e_floor)
            hard_ema[0] = 0.0
        if hard_ema.shape != (N,) or not np.isfinite(hard_ema).all():
            raise RuntimeError('Invalid hard-frame EMA')
        stat('hard_ema_ready', median=float(np.median(hard_ema)),
             p90=float(np.quantile(hard_ema, .9)),
             maximum=float(hard_ema.max()))
    total_updates = args.updates if args.pilot <= 0 else args.pilot
    start_update = update0

    def hard_alpha_at(update):
        if args.hard_weight <= 0:
            return 0.0
        final = args.hard_weight if args.hard_final is None else args.hard_final
        if args.hard_anneal_end > args.hard_anneal_start:
            frac = min(max((update - args.hard_anneal_start)
                           / float(args.hard_anneal_end - args.hard_anneal_start), 0.), 1.)
            return args.hard_weight + frac * (final - args.hard_weight)
        return args.hard_weight

    trajectory = []
    for update in range(start_update + 1, total_updates + 1):
        t0 = time.perf_counter()
        hard_now = 0.0
        train_weights = weights
        if args.hard_weight > 0:
            hard_now = hard_alpha_at(update)
            q = max(float(np.quantile(hard_ema[1:], args.hard_quantile)), 1e-30)
            hard_w = 1.0 + hard_now * np.clip(hard_ema / q, 0., 1.)
            hard_w[0] = 1.0
            hard_w /= float(hard_w.mean())
            hard_w_t = torch.as_tensor(hard_w, device='cuda', dtype=torch.float32)
            train_weights = weights * hard_w_t
            train_weights = train_weights / train_weights.mean()
        mse, receiver, rel_value, grad_norm = scan(
            args.chunk, train=True, train_weights=train_weights)
        step_s = time.perf_counter() - t0
        if args.hard_weight > 0:
            rel_now = mse / np.maximum(energy, e_floor)
            rel_now[0] = 0.0
            hard_ema = args.hard_ema * hard_ema + (1. - args.hard_ema) * rel_now
        current = metrics(mse, receiver)
        if current['balanced_per_second_percent'] < best_balanced:
            best_balanced = current['balanced_per_second_percent']
            best_step = update
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.refiner.state_dict().items()}
        if current['full801_percent'] < best_full:
            best_full = current['full801_percent']
            best_full_state = {k: v.detach().cpu().clone()
                               for k, v in model.refiner.state_dict().items()}
        trajectory.append(dict(update=update, step_s=step_s, grad_norm=grad_norm,
                               relative_loss=rel_value, hard_alpha=hard_now,
                               **current))
        if update % 10 == 0 or update == total_updates:
            with (OUT / 'TRAJECTORY.jsonl').open('a') as stream:
                stream.write(json.dumps(trajectory[-1]) + '\n')
            stat('train', complete=update, total=total_updates,
                 step_s=step_s, best_step=best_step,
                 best_balanced=best_balanced, best_full=best_full,
                 full801=current['full801_percent'],
                 per_second=current['per_second_percent'],
                 receiver=current['receiver_percent'],
                 grad_norm=grad_norm, hard_alpha=hard_now)
        if update % 50 == 0 or update == total_updates:
            atomic_torch(OUT / f'snapshots/active_{update:04d}.pt', {
                'schema': SCHEMA, 'complete': update,
                'refiner': {k: v.detach().cpu() for k, v in model.refiner.state_dict().items()},
                'metrics': current})
            atomic_torch(resume_path, {
                'schema': SCHEMA, 'complete': update,
                'reference_sha256': gate['reference_sha256'],
                'head_fit_sha256': gate['head_fit_sha256'],
                'refiner': {k: v.detach().cpu() for k, v in model.refiner.state_dict().items()},
                'optimizer': opt.state_dict(),
                'best_step': best_step, 'best_balanced': best_balanced,
                'best_full': best_full,
                'best_refiner': best_state, 'best_full_refiner': best_full_state,
                'hard_ema': hard_ema, 'hard_alpha': hard_now})
            atomic_torch(OUT / 'last_full.pt', {
                'schema': SCHEMA, 'complete': update,
                'refiner': {k: v.detach().cpu() for k, v in model.refiner.state_dict().items()},
                'metrics': current})
    torch.cuda.synchronize()

    model.refiner.load_state_dict(best_state, strict=True)
    stat('final_best_prediction', best_step=best_step)
    best_mse, best_receiver, _, _ = scan(args.chunk, train=False)
    final = metrics(best_mse, best_receiver)
    prediction = np.lib.format.open_memmap(OUT / 'prediction801_pa.npy', mode='w+',
                                           dtype=np.float32, shape=(N, 201, 201))
    with torch.no_grad(), session.training_scope():
        for j in range(0, N, args.chunk):
            ix = np.arange(j, min(j + args.chunk, N))
            z = session.query(times[torch.as_tensor(ix, device='cuda')], grad=False)
            prediction[ix] = (z * scale).cpu().numpy()
    prediction.flush()
    del prediction
    np.savetxt(OUT / 'receiver_trace.csv',
               np.column_stack((times_np, best_receiver, ref_receiver)),
               delimiter=',', header='time_s,ours_Pa,reference_Pa', comments='', fmt='%.18e')
    with (OUT / 'per_frame_metrics.csv').open('w') as stream:
        stream.write('frame_index,time_s,baseline_mse,final_mse,reference_energy,'
                     'baseline_receiver_Pa,final_receiver_Pa,reference_receiver_Pa\n')
        for i in range(N):
            stream.write(f'{i},{times_np[i]:.18g},{initial_mse[i]:.18g},{best_mse[i]:.18g},'
                         f'{energy[i]:.18g},{initial_receiver[i]:.18g},'
                         f'{best_receiver[i]:.18g},{ref_receiver[i]:.18g}\n')
    atomic_torch(OUT / 'best.pt', {
        'schema': SCHEMA, 'complete': total_updates, 'best_step': best_step,
        'refiner': best_state, 'metrics': final,
        'baseline_metrics': initial,
        'reference_sha256': gate['reference_sha256'],
        'head_fit_sha256': gate['head_fit_sha256'],
        'checkpoint_sha256': gate['checkpoint_sha256']})
    atomic_torch(OUT / 'best_full801.pt', {
        'schema': SCHEMA, 'refiner': best_full_state,
        'best_full': best_full})
    result = {
        'status': 'COMPLETE', 'schema': SCHEMA,
        'updates': total_updates, 'start_update': start_update,
        'best_step': best_step,
        'initial_metrics': initial,
        'final_metrics': final,
        'best_full801_percent': best_full,
        'selection': 'equal-per-second balanced score for best.pt; full-801 pooled also saved',
        'energy_floor_fraction': ENERGY_FLOOR_FRACTION,
        'trainable_parameters': sum(p.numel() for p in params),
        'frozen_parameters': sum(p.numel() for p in model.base.parameters()),
        'peak_gpu_bytes': int(torch.cuda.max_memory_allocated()),
        'elapsed_s': time.perf_counter() - wall,
        'prediction_sha256': sha(OUT / 'prediction801_pa.npy'),
        'reference_sha256': gate['reference_sha256'],
        'head_fit_sha256': gate['head_fit_sha256'],
        'checkpoint_sha256': gate['checkpoint_sha256'],
        'scope': 'single-instance 0--4 s Marmousi fine-tune; 801 saved and predicted frames at 5 ms',
    }
    write(OUT / 'RESULT.json', result)
    stat('complete', final=final, best_step=best_step,
         elapsed_s=result['elapsed_s'])


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT is not None and OUT.is_dir():
            write(OUT / 'FAILURE.json', {'traceback': traceback.format_exc()})
        raise
