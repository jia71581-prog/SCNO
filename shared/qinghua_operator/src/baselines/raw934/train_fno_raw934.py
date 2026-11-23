#!/usr/bin/env python3
"""Parameter-matched vanilla FNO-2D baseline on the joint raw934 TRAIN cache.

Purpose. The paper's layered/uniform baseline rows were trained on the
nyquist15 UL protocol (1000 records, 713-frame window at 15 Hz). The main
method now trains on the raw934 cache (934 records, IC8 + future392, raw nodal
NO_FILTER, source x free in 204.7..1799.2 m). Comparing those two is a protocol
confound, not a method difference. This entrypoint removes it by training the
same FNO-2D recipe on exactly the data, frames and metric the main method uses.

Provenance. Derived from `src/baselines/fno/train_fno_ul.py` (which descends
from the audited FS-25 `train_fno_pino1.py`), keeping the recipe: Li et al.
2021 FNO-2D with time-as-output-channels, four spectral convolution layers,
per-frame inverse-energy weighted MSE, AdamW wd=1e-6, lr 1e-3 cosine (0.1 floor
factor) with linear warmup, bf16 autocast, grad clip 1.0, checkpoint by
eval_every on an in-sample TRAIN-only dev panel.

Registered differences from the UL entrypoint, each forced by the raw934
contract:

- Data: `data/joint_uniform500_layered434_raw_20261011/train400_cache/`,
  read through `data_raw934.py`; 934 TRAIN records (434 layered + 500 uniform).
- Output: `cout = 392` stored future frames (146..800), i.e. exactly the frames
  the main method is scored on, instead of the UL 713-frame window.
- Input: 14 channels, adding the source-offset coordinate pair
  `((x-x_s)/2000, (z-z_s)/2000)` to the UL 12, because the raw934 source x is
  per-record. Velocity uses the frozen normalizer's center/scale rather than
  the UL `vel/6000`.
- Parameter match: width 55, modes 25 give 15,195,881 parameters, which is
  -0.27% against the UL FNO's 15,237,180 and +0.71% against the KNO's
  15,088,523, inside the +/-2% tolerance either way.
- Batching: a pre-generated per-step index schedule drawn without replacement,
  each rank taking its slice, so the global batch is the recipe's 8 records.
- No `set_per_process_memory_fraction`: the workspace forbids self-imposed VRAM
  caps; peak use is reported from the real allocator instead.
- Targets are memmapped, not staged in /dev/shm. The UL entrypoint cached an
  fp16 copy because its HDF5 source was slow; here `target.npy` is already an
  fp32 memmap, so a second cache would only duplicate 29 GB.

Split discipline. TRAIN only. The dev panel is drawn from TRAIN for monitoring;
no validation or test roster is opened, and no checkpoint selection uses
anything outside TRAIN.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

# 2026-11-10: local root repair for the centered-source restart.  The original
# remote worker path is unavailable on this host; the architecture is unchanged.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "baselines" / "raw934"))

from data_raw934 import (  # noqa: E402
    DEV_PANEL_ROWS,
    GRID,
    N_FUTURE,
    Raw934TrainCache,
)


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        scale = 1.0 / (cin * cout)
        self.m1, self.m2 = m1, m2
        self.w = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x.float())
        out = torch.zeros(B, self.w.shape[1], H, W // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        wc = torch.view_as_complex(self.w)
        out[:, :, :self.m1, :self.m2] = torch.einsum(
            'bixy,ioxy->boxy', xf[:, :, :self.m1, :self.m2], wc)
        out[:, :, -self.m1:, :self.m2] = torch.einsum(
            'bixy,ioxy->boxy', xf[:, :, -self.m1:, :self.m2], wc.conj())
        return torch.fft.irfft2(out, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, cin=14, width=55, modes=25, layers=4, cout=N_FUTURE):
        super().__init__()
        self.lift = nn.Conv2d(cin, width, 1)
        self.sp = nn.ModuleList([SpectralConv2d(width, width, modes, modes)
                                 for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, cout, 1))

    def forward(self, x):
        h = self.lift(x)
        for s, p in zip(self.sp, self.pw):
            h = torch.nn.functional.gelu(s(h) + p(h))
        return self.proj(h)


class StepBatchDataset(torch.utils.data.Dataset):
    """Item k = this rank's slice of the pre-generated step-k batch.

    The stack is assembled per item from the memmap rather than preloaded:
    934 x 14 x 201 x 201 fp32 is 2.1 GB, which would be fine in RAM but would
    also be duplicated per worker, and the target window is 29.6 GB regardless.
    """

    def __init__(self, schedule, rank, per_rank, cache, ids):
        self.schedule = schedule
        self.rank = rank
        self.per_rank = per_rank
        self.cache = cache
        self.ids = ids

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, step):
        mine = self.schedule[step, self.rank * self.per_rank:(self.rank + 1) * self.per_rank]
        xs, ys = [], []
        for row in mine:
            stack, target = self.cache.fno_item(self.ids[int(row)])
            xs.append(stack)
            ys.append(target)
        return torch.stack(xs), torch.stack(ys)


def frame_weights(cache: Raw934TrainCache, is_main: bool, path: Path) -> torch.Tensor:
    """Per-frame inverse energy, averaged over the 934 TRAIN records.

    The UL entrypoint had rank 0 compute this once into the cache directory.
    The cache directory here is a frozen, SHA-verified payload, so the vector is
    written into the run directory instead; the formula is unchanged.
    """
    if is_main and not path.exists():
        energy = cache.all_frame_energy()                     # [934,392] fp64
        mean = energy.mean(axis=0)
        np.save(str(path) + '.tmp.npy', mean)
        os.replace(str(path) + '.tmp.npy', path)
    if is_main:
        print(json.dumps({'event': 'frame_weights_ready', 'path': str(path)}), flush=True)
    dist.barrier()
    return torch.from_numpy(np.load(path)).float()


def panel_eval(model, device, cache, ids) -> dict:
    """In-sample TRAIN-only monitoring. Never used for the reported score."""
    was_training = model.training
    model.eval()
    per_family: dict[str, list] = {}
    per_record = {}
    with torch.no_grad():
        for row in DEV_PANEL_ROWS:
            sid = ids[row]
            stack, target = cache.fno_item(sid)
            pred = model(stack[None].to(device).float())[0].float().cpu()
            truth = target.float()
            rel = float((pred - truth).norm() / truth.norm().clamp_min(1e-30))
            per_family.setdefault(cache.families[row], []).append(rel)
            per_record[sid] = round(rel, 6)
    if was_training:
        model.train()
    every = [value for values in per_family.values() for value in values]
    return {
        'families': {family: round(float(np.mean(values)), 6)
                     for family, values in per_family.items()},
        'panel_mean': round(float(np.mean(every)), 6),
        'scoring_frames': f'{N_FUTURE} stored future frames (146..800), normalized scale',
        'scope': 'in-sample TRAIN panel, monitoring only',
        **per_record,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--global-batch', type=int, default=8)
    parser.add_argument('--width', type=int, default=55)
    parser.add_argument('--modes', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=300)
    parser.add_argument('--eval-every', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=0, help='bounded smoke only')
    parser.add_argument('--verify-arrays', action='store_true',
                        help='full-file SHA check of the frozen cache before training')
    args = parser.parse_args()

    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if args.global_batch % world:
        raise ValueError('global batch must divide across ranks')
    per_rank = args.global_batch // world
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl')
    is_main = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True

    run = Path(args.run_dir)
    if is_main:
        run.mkdir(parents=True, exist_ok=True)
    elif not run.exists():
        raise FileNotFoundError('run directory must exist before ranks start')

    cache = Raw934TrainCache()
    if args.verify_arrays:
        for name in ('known_ic.npy', 'target.npy', 'energy.npy'):
            cache.verify_array(name)
    ids = list(cache.ids)
    records = len(ids)

    model = FNO2d(width=args.width, modes=args.modes, cout=N_FUTURE).to(device)
    parameters = sum(value.numel() for value in model.state_dict().values())
    ddp = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=args.lr, weight_decay=1e-6)

    steps = args.max_steps or math.ceil(records * args.epochs / args.global_batch)
    w_frame = frame_weights(cache, is_main, run / 'w_frame_fp64_r934.npy')
    w_frame = (1.0 / w_frame.clamp_min(w_frame.max() * 1e-3))
    w_frame = (w_frame / w_frame.mean()).to(device)

    if is_main:
        (run / 'run_meta.json').write_text(json.dumps({
            'parameters': parameters,
            'cache': str(cache.cache),
            'cache_view_id': cache.view_id,
            'normalizer_sha256': cache.normalizer.source_sha256,
            'records': records,
            'families': {f: cache.families.count(f) for f in set(cache.families)},
            'epochs': args.epochs,
            'steps': steps,
            'world_size': world,
            'global_batch': args.global_batch,
            'per_rank_batch': per_rank,
            'cout': N_FUTURE,
            'input_channels': 14,
            'recipe': 'vanilla FNO-2D (Li 2021), time-as-channels, dense supervision, '
                      'per-frame inverse-energy weights, lr cosine+warmup, bf16; '
                      'raw934 IC8+future392 protocol',
            'scope': 'TRAIN only; no validation or test roster opened',
            'args': vars(args),
        }, indent=1))
        print(json.dumps({'event': 'start', 'parameters': parameters,
                          'steps': steps, 'world': world}), flush=True)

    metrics = open(run / f'metrics_rank{rank}.jsonl', 'a') if is_main else None
    generator = np.random.default_rng(args.seed * 13)
    schedule = np.stack([generator.choice(records, size=args.global_batch, replace=False)
                         for _ in range(steps)])
    dataset = StepBatchDataset(schedule, rank, per_rank, cache, ids)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.workers,
        pin_memory=True, prefetch_factor=2 if args.workers > 0 else None,
        persistent_workers=args.workers > 0)

    def lr_at(step):
        scale = min(1.0, (step + 1) / args.warmup)
        progress = min(1.0, (step + 1) / steps)
        return args.lr * scale * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    started = time.time()
    for step, (xb, yb) in enumerate(loader):
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = ddp(xb)
        loss = (((pred.float() - yb) ** 2).mean(dim=(2, 3)) * w_frame[None]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        optimizer.step()
        if is_main and (step % 10 == 0 or step == steps - 1):
            with torch.no_grad():
                rel = float((pred.float() - yb).norm() / yb.norm().clamp_min(1e-30))
            metrics.write(json.dumps({
                'step': step, 'loss': float(loss.detach()), 'gn': float(grad_norm),
                'batch_relL2': round(rel, 4), 'lr': lr_at(step),
                'peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2),
                'sec': round(time.time() - started, 1)}) + '\n')
            metrics.flush()
            if step % 200 == 0:
                print(json.dumps({'step': step, 'batch_relL2': round(rel, 4),
                                  'sec': round(time.time() - started, 1)}), flush=True)
        if args.eval_every and step > 0 and step % args.eval_every == 0:
            if is_main:
                print(json.dumps({'event': 'dev_panel', 'step': step,
                                  **panel_eval(model, device, cache, ids)}), flush=True)
                tmp = run / 'ckpt_latest.pt.tmp'
                torch.save({'model_state': model.state_dict(), 'step': step,
                            'args': vars(args), 'parameters': parameters}, tmp)
                tmp.rename(run / 'ckpt_latest.pt')
            dist.barrier()

    if is_main:
        tmp = run / 'ckpt_latest.pt.tmp'
        torch.save({'model_state': model.state_dict(), 'step': steps,
                    'args': vars(args), 'parameters': parameters}, tmp)
        tmp.rename(run / 'ckpt_latest.pt')
        final = panel_eval(model, device, cache, ids)
        (run / 'TRAIN_DEV_FINAL.json').write_text(json.dumps({
            'note': ('in-sample TRAIN panel, monitoring only; the reported score '
                     'comes from a separate frozen-protocol evaluation'),
            **final}, indent=1))
        print(json.dumps({'event': 'final_dev_panel', **final}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
