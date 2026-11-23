#!/usr/bin/env python3
"""局部推进算子(chunk operator)训练 —— 归一化域(2026-10-14)。

设计 ../01_design/DESIGN_LOCAL_CHUNK_OPERATOR.md

形式:known8(8 帧) + 速度 -> 接下来的 K 帧;自回归走完 392 帧预报窗口。
前向次数 = ceil(392 / K),这是相对数值解(401@5m 需 800 步 x 11.46 ms ~ 9.2 s)的速度来源。

## 为什么在归一化域训练(用户令,且是必要条件)

cache 已归一化(`pressure_scale_pa = 1.252277344221455e-08`,
`target_units = "UL-normalized signed amplitude exactly once"`),预测后乘回即为物理 Pa。
物理域下场约 1e-8,任何经 dt^2=1.56e-8 的修正相对贡献约 1e-10,低于 FP32 ~1e-7 精度,
**加法直接丢弃、上游梯度精确为零**(今日实测)。归一化域下相对贡献约 1e-2,FP32 充裕。
relL2 尺度不变,故指标两域可比;数组本身不可互换。

## 判据(自回归口径,唯一可部署)

留出记录、逐帧平均 relL2:PASS < 4.64%(同输入数值解);WEAK < 10.95%;否则 FAIL。
teacher-forced 仅作上界参照 —— 它每段消耗真实 known8,不可部署。

## rollout 稳定化开关(2026-10-14,默认全关 = 保持既有行为)

已实测的靶子:free-running(可部署)31.17% vs teacher-forced 6.82%,差 4.57x 且随训练
扩大;chunk0(真 known8、零自回归)17.005% 占 free 的 0.546 —— 即自回归复合误差最多
只占 14pp,这是 rollout 稳定化的上界。漂移场内部加权(内部 1.109,各边界带 <1.0),
所以不是边界问题。三个开关都针对「训练时网络很少消化自己的输出」这一条:

1. `--bptt-segments N`:最后 N 段之间**不** detach,梯度可穿过自回归边界。现状是段间
   一律 `ctx.detach()`,于是「我的输出会变成我的输入」这件事从未进入梯度。N=0 等于现状。
2. `--scheduled-sampling p`(+ `--ss-anneal-steps M` 线性退火到 0):每个段边界以概率 p
   用真值上下文替代自己的输出。p=1 是 teacher forcing,p=0 是现状(全自回归)。
   **只在训练生效**;`run_record(grad=False)` 内部硬置零,评测绝不喂真值。
3. `--seg-weight late`:按自回归深度给晚段更大权重,总权重与 uniform 同为 1。

三者默认关闭时训练一步的梯度与改动前**逐位相同**(见
`../04_runs/ROLLOUT_STABILISATION_SMOKE.json` 的 bitwise 检验);这是不破坏既有臂
可比性的前提。为此 uniform 路径保留原来的 `sum(terms)/nseg` 写法而非等价的加权和
——`(a+b)/n` 与 `a/n+b/n` 在 FP32 下不逐位相等。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np



def pfm(P, Y):
    s = np.square(P - Y).reshape(len(P), -1).sum(1)
    e = np.square(Y).reshape(len(Y), -1).sum(1)
    return float(np.sqrt(s / e).mean())


def segment_weights(nseg, scheme):
    """Per-segment loss weights inside one rollout; **sums to 1** for every scheme.

    Why the sum is pinned to 1: the existing loss is `sum_j term_j / nseg`, i.e.
    uniform weights 1/nseg that already sum to 1.  A weighting whose sum differed
    would rescale the gradient and therefore silently change the effective learning
    rate, so a `late`-vs-`uniform` comparison would no longer be a comparison of
    weighting alone.

    `late` = linear ramp w_j ∝ (j+1), j being the autoregressive depth inside the
    rollout (0 = fed a true/own context from the step's start, nseg-1 = deepest).
    Rationale: the measured gap is 31.17% free-running vs 6.82% teacher-forced,
    while chunk0 (true known8, zero autoregression) is already 17.005% -- so the
    part that weighting can touch is the ~14pp that accumulates WITH depth, and the
    deepest segments are where that signal lives.  Linear rather than exponential on
    purpose: an exponential ramp at nseg=4 puts ~70% of the weight on one segment,
    which cuts the effective number of scored segments per step to about one and
    raises gradient variance; the measured per-chunk error growth is not explosive
    enough to justify that trade.
    """
    if scheme == 'uniform':
        return [1.0 / nseg] * nseg
    if scheme == 'late':
        tot = nseg * (nseg + 1) / 2.0           # sum_{j=1..nseg} j
        return [(j + 1) / tot for j in range(nseg)]
    raise ValueError(f'unknown seg-weight scheme {scheme!r}')


def build_operator(*, K, ctx, width, modes, rank, trunk_width, nz, nx, dz_m, dx_m, device):
    """The backbone is ChunkMIONet -- see ../01_design/INHERITANCE.md.

    It keeps the parts of the production design that measured well (multi-branch
    multiplicative fusion, phase-aligned source-relative features, the straight-ray
    travel-time branch, the parallel-local + identity-skip spectral block, the
    1.0/0.1 coarse-local mixing) and drops the parts that measured badly (capacity
    piled on the medium branch, spectral truncation as a constraint, a fully shared
    DeepONet-style trunk basis, and the one-shot all-400-frame global form).
    """
    from chunk_mionet import ChunkMIONet
    return ChunkMIONet(K=K, ctx=ctx, rank=rank, width=width, modes=modes,
                       trunk_width=trunk_width, nz=nz, nx=nx,
                       dz_m=dz_m, dx_m=dx_m).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, help='Path for this training run')
    ap.add_argument('--data', required=True, help='Path to the prepared Marmousi data')
    ap.add_argument('--K', type=int, required=True, help='frames produced per forward')
    ap.add_argument('--ctx', type=int, default=8, help='input context frames (known8)')
    ap.add_argument('--records', type=int, default=200)
    ap.add_argument('--heldout', type=int, nargs='+', default=[900, 901, 902, 903])
    ap.add_argument('--width', type=int, default=48)
    ap.add_argument('--modes', type=int, default=20)
    ap.add_argument('--rank', type=int, default=128,
                    help='MIONet fusion rank (branch/trunk share it; '
                         'fusion.py asserts equal branch ranks)')
    ap.add_argument('--trunk-width', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--rollout-segments', type=int, default=2,
                    help='segments trained autoregressively per step (1 = teacher only). '
                         'With --rollout-curriculum this is the FINAL depth.')
    ap.add_argument('--rollout-curriculum', default='',
                    help='depth schedule "d0:f0,d1:f1,..." where each dN is a rollout '
                         'depth and fN the fraction of --steps at which it starts, e.g. '
                         '"2:0,6:0.15,14:0.4,28:0.7". Empty = fixed --rollout-segments. '
                         'This is the one lever that addresses the depth mismatch no '
                         'other method touches: training at 6 segments while deploying '
                         'at 28 is a 4.67x gap. Three independent papers (List et al. '
                         '1->4->16, Solver-in-the-Loop, Kohl et al.) agree a schedule '
                         'beats a fixed depth, and List et al. report long unrollings '
                         'early in training HURT accuracy -- hence ramping, not '
                         'starting deep.')
    # --- rollout stabilisation (all three default to the pre-existing behaviour) ---
    ap.add_argument('--bptt-segments', type=int, default=0,
                    help='number of trailing segments whose boundaries are NOT '
                         'detached, so gradient flows across the autoregressive '
                         'feedback. 0 = detach every boundary (current behaviour)')
    ap.add_argument('--scheduled-sampling', type=float, default=0.0,
                    help='per-boundary probability of substituting the TRUE context '
                         'for the net output during training only. 0 = off')
    ap.add_argument('--ss-anneal-steps', type=int, default=0,
                    help='linearly anneal --scheduled-sampling to 0 over this many '
                         'steps. 0 = hold the probability constant')
    ap.add_argument('--seg-weight', choices=('uniform', 'late'), default='uniform',
                    help='per-segment loss weighting inside a rollout; `late` puts '
                         'more weight on deeper autoregressive steps (sum of '
                         'weights is 1 either way, so the effective lr is unchanged)')
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--max-hours', type=float, default=10.0)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--no-tf32', dest='tf32', action='store_false',
                    help='keep full FP32 matmul (slower; only for numerics checks)')
    ap.set_defaults(tf32=True)
    a = ap.parse_args()

    import torch

    # TF32 for the matmul-shaped work (every Conv2d and every spectral einsum).
    # This model is almost entirely matmul, and FP32 is the slow path on Ada and
    # Hopper.  Accuracy headroom is wide: the gate is 4.64e-2 relL2 while TF32's
    # relative perturbation is ~1e-3 on a forward, and the cache's own float32
    # round-trip is already 2.6e-8.  Disable with --no-tf32 if a numerics question
    # ever needs the slow path back.
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True       # fixed shapes every step

    run = Path(a.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    dev = a.device
    DATA = Path(a.data).resolve()
    cfg = json.loads((DATA / 'CONFIG.json').read_text())
    scale_pa = float(cfg['pressure_scale_pa'])
    tgt = np.load(DATA / 'train400_cache/target.npy', mmap_mode='r')
    ic = np.load(DATA / 'train400_cache/known_ic.npy', mmap_mode='r')
    vel = np.load(DATA / 'train400_cache/known_velocity.npy', mmap_mode='r')
    nt = int(tgt.shape[2])
    gz, gx = int(tgt.shape[-2]), int(tgt.shape[-1])
    print(f'[data] {DATA.name}  target {tgt.shape}  known_ic {ic.shape}')
    # Caliber guard.  The network emits O(1) values, so the TARGET must be O(1) too.
    # The remote cache is already UL-normalised (absmax ~1); a physical-Pa cache
    # (absmax ~1e-8) would make every relative error ~1e8 regardless of training --
    # today this exact mismatch produced 1.8e10% and cost several debugging rounds.
    probe_absmax = float(np.abs(np.asarray(tgt[0, 0, nt // 2], dtype=np.float64)).max())
    if probe_absmax < 1e-3:
        norm = 1.0 / probe_absmax
        print(f'[caliber] target looks PHYSICAL (absmax {probe_absmax:.4g}); applying '
              f'on-the-fly normalisation x{norm:.6g}. Restore = divide by it.')
    else:
        norm = 1.0
        print(f'[caliber] target already normalised (absmax {probe_absmax:.4g}); '
              f'pressure_scale_pa={scale_pa:.9g} restores physical Pa (multiply once)')
    print(f'[grid] {gz}x{gx}  window {nt} frames  K={a.K}  forwards/record='
          f'{math.ceil(nt / a.K)}')

    train_ids = [i for i in range(a.records) if i not in a.heldout]
    held = list(a.heldout)
    print(f'[split] train {len(train_ids)} records, held-out {held}')

    if a.bptt_segments == 0 and a.scheduled_sampling == 0.0 and a.seg_weight == 'uniform':
        print('[rollout] stabilisation OFF (bitwise-identical to pre-20261014 arms)')
    else:
        _w = segment_weights(min(a.rollout_segments, math.ceil(nt / a.K)), a.seg_weight)
        print(f'[rollout] bptt_segments={a.bptt_segments}  '
              f'scheduled_sampling={a.scheduled_sampling} '
              f'(anneal {a.ss_anneal_steps} steps, TRAIN ONLY)  '
              f'seg_weight={a.seg_weight} {[round(w, 4) for w in _w]} '
              f'sum={sum(_w):.6f}')

    zz, xx = np.meshgrid(np.arange(gz) / (gz - 1), np.arange(gx) / (gx - 1), indexing='ij')
    xy = torch.tensor(np.stack([zz, xx]), dtype=torch.float32, device=dev)[None]

    op = build_operator(K=a.K, ctx=a.ctx, width=a.width, modes=a.modes,
                        rank=a.rank, trunk_width=a.trunk_width,
                        nz=gz, nx=gx, dz_m=10.0, dx_m=10.0, device=dev)
    nparam = sum(p.numel() for p in op.parameters())
    ntrunk = sum(p.numel() for n, p in op.named_parameters() if n.startswith('trunk'))
    nbranch = sum(p.numel() for n, p in op.named_parameters()
                  if n.split('.')[0] in ('medium', 'source', 'travel', 'ctx_branch'))
    print(f'[model] ChunkMIONet {nparam} params  trunk {ntrunk}  branches {nbranch} '
          f'(capacity on the trunk by design: it is the only group measured to '
          f'rotate the output subspace)')
    opt = torch.optim.AdamW(op.parameters(), lr=a.lr, weight_decay=1e-6)

    src_all = np.ascontiguousarray(
        np.load(DATA / 'train400_cache/known_source.npy', mmap_mode='r')[:, 0, :],
        dtype=np.float32)
    tax = np.ascontiguousarray(
        np.load(DATA / 'train400_cache/known_time_axis.npy', mmap_mode='r')[0],
        dtype=np.float64).reshape(-1)
    fut = np.asarray(cfg.get('future_frames', cfg['full_frames'][-nt:]), dtype=np.int64)

    def velocity_of(rid):
        # ChunkMIONet normalises the velocity internally, so pass m/s here.
        v = np.asarray(vel[rid], dtype=np.float32).reshape(1, 1, gz, gx)
        return torch.tensor(v, device=dev)

    def source_of(rid):
        return torch.tensor(src_all[rid], device=dev)[None]

    def context0(rid):
        arr = np.asarray(ic[rid, 0], dtype=np.float32) * norm
        return torch.tensor(arr, device=dev)[None]

    def frames_of(rid):
        arr = np.asarray(tgt[rid, 0], dtype=np.float32) * norm
        return torch.tensor(arr, device=dev)

    def run_record(rid, segments, grad, seg0=0, ss_p=0.0):
        """Autoregressive rollout; returns (loss tensor or None, prediction array).

        `seg0` is the chunk index the rollout STARTS at.  Training uses a random
        seg0 and takes the true frames ending at that point as context; evaluation
        always uses seg0=0 with segments<=0, i.e. the deployable full rollout from
        known8.

        `ss_p` is the scheduled-sampling probability for THIS call and is forced to
        zero whenever `grad` is False.  That is the hard interlock behind the
        caliber red line: the evaluation path (`seg0=0, segments=0, grad=False`)
        cannot be fed true frames mid-rollout no matter what the caller passes.

        Why seg0 exists: with a fixed seg0=0 and segments=R, the loss only ever
        covers the first R*K of the window's nt frames (R=4, K=14 -> 56 of 392),
        so 86% of the window was never trained while every eval scored all 28
        chunks.  That is visible in the K-curve diagnosis as a monotone rise in
        per-chunk error even under teacher forcing, where the chunks are
        independent and compounding cannot explain it.
        """
        v = velocity_of(rid)
        src5 = source_of(rid)
        Y = frames_of(rid)
        total_seg = math.ceil(nt / a.K)
        nseg = total_seg if segments <= 0 else min(segments, total_seg)
        seg0 = max(0, min(int(seg0), total_seg - nseg))
        if seg0 == 0:
            ctx = context0(rid)
        else:
            # true context ending just before this chunk, so the segment is trained
            # on a real state rather than on whatever the net drifted to
            end = seg0 * a.K
            ctx = Y[end - a.ctx:end][None]
        # Caliber interlock: scheduled sampling is a TRAINING device only.
        ss_p = float(ss_p) if grad else 0.0
        # `bptt_segments` counts back from the END of the rollout.  The boundary
        # after rollout-local segment j feeds segment j+1, so it is retained iff
        # both j and j+1 sit inside the trailing N segments, i.e. j >= nseg - N.
        # N=0 detaches every boundary (the pre-existing behaviour) and N=1 also
        # retains nothing, since a single segment has no interior boundary.
        nb = max(0, min(int(a.bptt_segments), nseg)) if grad else 0
        weights = segment_weights(nseg, a.seg_weight) if grad else None
        uniform_w = (a.seg_weight == 'uniform')
        preds, loss = [], None
        ss_used = 0
        ctxt = torch.enable_grad() if grad else torch.no_grad()
        with ctxt:
            # time-invariant branches and the ray integration: once per rollout,
            # not once per chunk (identical result, so this is pure removal)
            static = op.static_context(v, src5)
            for s in range(seg0, seg0 + nseg):
                j = s - seg0                     # autoregressive depth in this step
                # start time of this chunk, in seconds on the solver clock
                t_chunk = float(tax[int(fut[min(s * a.K, nt - 1)])])
                out = op(ctx, v, src5, t_chunk, static=static)[0]   # [K,gz,gx]
                lo = s * a.K
                hi = min(lo + a.K, nt)
                take = out[:hi - lo]
                preds.append(take.detach())
                if grad:
                    y = Y[lo:hi]
                    num = torch.square(take - y).sum()
                    den = torch.square(y).sum().clamp_min(1e-12)
                    # The uniform branch keeps the original `sum(terms)/nseg`
                    # arithmetic instead of the algebraically equal weighted sum,
                    # because (a+b)/n and a/n+b/n are not bitwise equal in FP32 and
                    # the default path must reproduce earlier arms exactly.
                    if uniform_w:
                        loss = (num / den) if loss is None else loss + num / den
                    else:
                        term = weights[j] * (num / den)
                        loss = term if loss is None else loss + term
                # autoregressive: feed back our OWN last ctx frames
                if take.shape[0] >= a.ctx:
                    ctx = take[-a.ctx:][None]
                else:
                    ctx = torch.cat((ctx[0, take.shape[0]:], take), 0)[None]
                # Scheduled sampling: hand the NEXT segment the true context
                # instead of ours.  Draw only when active, so the default path
                # consumes no RNG and stays bitwise identical.
                if ss_p > 0.0 and hi >= a.ctx and np.random.random() < ss_p:
                    ctx = Y[hi - a.ctx:hi][None]
                    ss_used += 1
                elif grad and j < nseg - nb:
                    # outside the BPTT tail: cut the graph at this boundary
                    ctx = ctx.detach()
        if grad and uniform_w:
            loss = loss / max(1, nseg)
        return loss, torch.cat(preds, 0)

    def ss_prob_at(it):
        """Scheduled-sampling probability at step `it` (1-based), linearly annealed.

        Held constant when --ss-anneal-steps is 0.  Annealing to zero matters for
        this target: the deployable caliber is pure free-running, so a run that ends
        with p>0 was never trained on the thing it is scored on.
        """
        if a.scheduled_sampling <= 0.0:
            return 0.0
        if a.ss_anneal_steps <= 0:
            return float(a.scheduled_sampling)
        frac = max(0.0, 1.0 - (it - 1) / float(a.ss_anneal_steps))
        return float(a.scheduled_sampling) * frac

    def evaluate():
        out = {}
        for h in held:
            # deployable caliber: full rollout from known8, no true frames fed back
            _, P = run_record(h, segments=0, grad=False, ss_p=0.0)
            # relL2 is scale-free, so score in whatever domain training used
            Y = np.asarray(tgt[h, 0], dtype=np.float64) * norm
            out[h] = pfm(P.double().cpu().numpy(), Y)
        return float(np.mean(list(out.values()))), out

    stop = run / 'STOP_REQUEST.json'
    hist = []
    best = float('inf')
    bad = 0
    t0 = time.time()
    (run / 'CONFIG.json').write_text(json.dumps(dict(
        schema='CHUNK_OPERATOR_TRAIN_V1', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        args=vars(a), params=nparam, pressure_scale_pa=scale_pa,
        window_frames=nt, forwards_per_record=math.ceil(nt / a.K),
        training_window_coverage=dict(
            scheme='random seg0 per sample; true frames as context at seg0>0',
            segments_per_step=a.rollout_segments,
            frames_per_step=a.rollout_segments * a.K,
            total_frames=nt,
            seg0_range=[0, max(0, math.ceil(nt / a.K) - a.rollout_segments)],
            note=('a fixed seg0=0 would train only the first segments*K frames '
                  'while every eval scores the whole window')),
        rollout_stabilisation=dict(
            bptt_segments=a.bptt_segments,
            scheduled_sampling=a.scheduled_sampling,
            ss_anneal_steps=a.ss_anneal_steps,
            seg_weight=a.seg_weight,
            segment_weights=segment_weights(
                min(a.rollout_segments, math.ceil(nt / a.K)), a.seg_weight),
            all_default=(a.bptt_segments == 0 and a.scheduled_sampling == 0.0
                         and a.seg_weight == 'uniform'),
            note=('defaults reproduce the pre-2026-10-14 arms bitwise; scheduled '
                  'sampling is train-only (forced to 0 when grad=False)')),
        data=DATA.name, train_records=len(train_ids), heldout=held,
        criteria=dict(pass_below=0.0464, weak_below=0.1095,
                      reference_numeric_solver=0.0464,
                      reference_linear_step_propagator_free=0.1095,
                      reference_global_operator=0.233306),
        caliber='normalised domain; relL2 scale-free; restore = multiply by pressure_scale_pa',
        applied_normalisation=None,  # filled below once `norm` is known
    ), ensure_ascii=False, indent=1))
    cfgp = json.loads((run / 'CONFIG.json').read_text())
    cfgp['applied_normalisation'] = norm
    cfgp['target_probe_absmax'] = probe_absmax
    (run / 'CONFIG.json').write_text(json.dumps(cfgp, ensure_ascii=False, indent=1))

    seg_total_all = math.ceil(nt / a.K)
    if a.rollout_curriculum:
        stages = []
        for part in a.rollout_curriculum.split(','):
            d_s, f_s = part.split(':')
            stages.append((min(int(d_s), seg_total_all), float(f_s)))
        stages.sort(key=lambda s: s[1])
        print(f'[curriculum] depth schedule {stages} over {a.steps} steps '
              f'(deploy depth is {seg_total_all})')
    else:
        stages = [(min(a.rollout_segments, seg_total_all), 0.0)]

    def depth_at(it):
        frac = it / max(1, a.steps)
        d = stages[0][0]
        for dd, ff in stages:
            if frac >= ff:
                d = dd
        return d

    cur_depth = depth_at(1)
    last_depth = cur_depth
    for it in range(1, a.steps + 1):
        if stop.exists():
            print('[stop] STOP_REQUEST.json'); break
        if (time.time() - t0) / 3600 > a.max_hours:
            print('[stop] wall-clock budget'); break
        cur_depth = depth_at(it)
        if cur_depth != last_depth:
            # run_record derives its own per-segment weights from nseg, so nothing
            # else needs rebuilding when the depth changes
            print(f'[curriculum] iter {it}: rollout depth {last_depth} -> {cur_depth}',
                  flush=True)
            last_depth = cur_depth
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        ss_now = ss_prob_at(it)
        for _ in range(a.batch):
            rid = int(np.random.choice(train_ids))
            # random window position, so the loss covers all nt frames over training
            seg_total = math.ceil(nt / a.K)
            seg0 = int(np.random.randint(0, max(1, seg_total - cur_depth + 1)))
            loss, _ = run_record(rid, segments=cur_depth, grad=True,
                                 seg0=seg0, ss_p=ss_now)
            (loss / a.batch).backward()
            tot += float(loss)
        gn = torch.nn.utils.clip_grad_norm_(op.parameters(), 1.0)
        if not math.isfinite(tot):
            print(f'[diverged] iter {it}'); break
        opt.step()
        if it % 50 == 0:
            extra = f'  ss {ss_now:.3f}' if a.scheduled_sampling > 0 else ''
            print(f'  iter {it:6d}  loss {tot/a.batch:.6e}  gn {float(gn):.3e}  '
                  f'{time.time()-t0:.0f}s{extra}', flush=True)
        if it % a.eval_every == 0:
            m, per = evaluate()
            hist.append(dict(iter=it, heldout_mean=m, per_record=per,
                             seconds=time.time() - t0))
            (run / 'HISTORY.jsonl').write_text(
                '\n'.join(json.dumps(h, ensure_ascii=False) for h in hist) + '\n')
            print(f'[eval] iter {it}  held-out autoregressive per-frame-mean {m*100:.4f}%',
                  flush=True)
            if m < best - 1e-6:
                best, bad = m, 0
                torch.save(dict(net=op.state_dict(), iter=it, heldout_mean=m), run / 'best.pt')
            else:
                bad += 1
                if bad >= a.patience:
                    print(f'[stop] no improvement in {bad} evals'); break

    verdict = 'PASS' if best < 0.0464 else 'WEAK' if best < 0.1095 else 'FAIL'
    (run / 'TERMINAL.json').write_text(json.dumps(dict(
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        best_heldout_autoregressive=best, verdict=verdict, evals=len(hist),
        seconds=time.time() - t0, K=a.K,
        criteria='PASS < 4.64% (numeric solver); WEAK < 10.95%; else FAIL',
        scope='held-out source positions on a single marmousi medium; not family-level',
    ), ensure_ascii=False, indent=1))
    print(f'\n[terminal] best held-out autoregressive {best*100:.4f}%  verdict {verdict}')


if __name__ == '__main__':
    main()
