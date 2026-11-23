#!/usr/bin/env python3
"""局部推进算子的主干:继承 MIONet 多分支乘性融合 + 相位对齐 + 走时(2026-10-14)。

继承决策与依据见 ../01_design/INHERITANCE.md。摘要:

继承(有实测支撑)
  1. **多分支乘性融合**(fusion.py:195-199):product = v * s * travel * trunk,再求和。
     但容量偏向 **trunk** 一侧 —— 实测 base_coordinate(5.2 万)梯度范数 1.851 全场最大、
     是唯一同时降误差与地板的组;base_medium(1006 万)范数 0.0853 且地板反升,
     按范数重配 lr 实测 −1.31%。故容量堆在系数侧无用。
  2. **相位对齐坐标特征**(features.py:24-110):dx/Lx、dz/Lz、radius、tau=t−t0−travel、
     phase=2πf0τ、Gabor 窗。真值侧实测:按震源对齐后共享基地板 40.79%→25.63%,
     主角余弦 0.04→0.37。
  3. **走时分支**(travel_time.py):tau 是相位结构的物理锚点,推进一段的相位推移
     可由走时直接给出,不必让网络从零学。
  4. **并联局部路径 + 恒等 skip**(spectral.py:313-325):value + gelu(norm(spectral + local))。
     链路因此不受谱截断限制(cutoff 只限谱路径;精修器是纯卷积无截断)。

不继承(已判负):一次性出 400 帧的全局形式;容量堆 base_medium;谱截断约束;
DeepONet 式完全共享 trunk 基(真值侧上界 43% vs 专属 24.86%,反向)。

本模块在**归一化域**工作(pressure_scale_pa = 1.252277344221455e-08,恢复=乘一次)。
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def straight_ray_tau(velocity_mps, src_xz_m, *, nz, nx, dz_m, dx_m, samples=12):
    """Straight-ray travel time from the source to every node, [B,1,nz,nx] seconds.

    Same construction as the inherited travel_time.py: integrate sampled slowness
    along each straight source-to-node segment.  Kept local so this module has no
    import dependency on the production snapshot tree.
    """
    B = velocity_mps.shape[0]
    dev = velocity_mps.device
    z = torch.arange(nz, device=dev, dtype=velocity_mps.dtype) * dz_m
    x = torch.arange(nx, device=dev, dtype=velocity_mps.dtype) * dx_m
    zz, xx = torch.meshgrid(z, x, indexing='ij')
    sx = src_xz_m[:, 0].view(B, 1, 1)
    sz = src_xz_m[:, 1].view(B, 1, 1)
    # Ray sampling with an explicit batch axis.  The earlier version broadcast the
    # sample axis against the batch axis ([S,1,1] vs [B,1,1]) and so only worked at
    # B=1; any larger batch raised a shape error.  Keep the axes separate: sample
    # points are [B,S,nz,nx] and each record's rays are sampled in its own velocity.
    t = torch.linspace(0.0, 1.0, samples, device=dev, dtype=velocity_mps.dtype)
    t = t.view(1, samples, 1, 1)
    sx4 = sx.view(B, 1, 1, 1)
    sz4 = sz.view(B, 1, 1, 1)
    xx4 = xx.view(1, 1, nz, nx)
    zz4 = zz.view(1, 1, nz, nx)
    px = sx4 + (xx4 - sx4) * t                                # [B,S,nz,nx]
    pz = sz4 + (zz4 - sz4) * t
    gx = 2.0 * px / max(1e-12, (nx - 1) * dx_m) - 1.0
    gz = 2.0 * pz / max(1e-12, (nz - 1) * dz_m) - 1.0
    # grid_sample wants [N,H,W,2]; fold (B,S) into N so one call covers every ray.
    grid = torch.stack((gx, gz), dim=-1).reshape(B * samples, nz, nx, 2)
    vrep = velocity_mps.unsqueeze(1).expand(B, samples, 1, nz, nx).reshape(
        B * samples, 1, nz, nx)
    v = F.grid_sample(vrep, grid, mode='bilinear', align_corners=True)
    mean_slowness = (1.0 / v.clamp_min(1.0)).view(B, samples, 1, nz, nx).mean(dim=1)
    dist = torch.sqrt((xx4 - sx4) ** 2 + (zz4 - sz4) ** 2 + 1e-12)   # [B,1,nz,nx]
    return mean_slowness * dist, dist, mean_slowness


def phase_aligned_features(velocity_mps, src5, t_s, *, nz, nx, dz_m, dx_m,
                           fourier_bands=6, gabor_scales=(0.01, 0.025, 0.05, 0.1),
                           domain_t_s=1.0, tau_tt=None):
    """Inherited phase-aligned feature block, [B,C,nz,nx] at one instant.

    Source-relative geometry (dx, dz, radius), travel-time-aligned time
    (tau = t - t0 - travel), the Ricker phase and its Gabor envelopes.  These are
    exactly the quantities the production coordinate encoder builds; the real-value
    evidence for keeping them is the source-centred basis probe (floor 40.79% ->
    25.63%, principal cosines 0.04 -> 0.37).
    """
    B = velocity_mps.shape[0]
    dev = velocity_mps.device
    dt_ = velocity_mps.dtype
    z = torch.arange(nz, device=dev, dtype=dt_) * dz_m
    x = torch.arange(nx, device=dev, dtype=dt_) * dx_m
    zz, xx = torch.meshgrid(z, x, indexing='ij')
    Lx = (nx - 1) * dx_m
    Lz = (nz - 1) * dz_m
    sx = src5[:, 0].view(B, 1, 1, 1)
    sz = src5[:, 1].view(B, 1, 1, 1)
    f0 = src5[:, 2].view(B, 1, 1, 1)
    t0 = src5[:, 3].view(B, 1, 1, 1)
    # `tau_tt` is time-invariant; the caller may pass a hoisted copy so an
    # R-segment rollout integrates the rays once instead of R times.
    if tau_tt is None:
        tau_tt, dist, _ = straight_ray_tau(velocity_mps, src5[:, :2],
                                           nz=nz, nx=nx, dz_m=dz_m, dx_m=dx_m)
    else:
        sx_ = src5[:, 0].view(B, 1, 1, 1)
        sz_ = src5[:, 1].view(B, 1, 1, 1)
        dist = torch.sqrt((xx.view(1, 1, nz, nx) - sx_) ** 2
                          + (zz.view(1, 1, nz, nx) - sz_) ** 2 + 1e-12)
    X = xx.view(1, 1, nz, nx).expand(B, 1, nz, nx)
    Z = zz.view(1, 1, nz, nx).expand(B, 1, nz, nx)
    dxr = (X - sx) / Lx
    dzr = (Z - sz) / Lz
    radius = dist / math.sqrt(Lx ** 2 + Lz ** 2)
    rel_t = (t_s - t0.squeeze(-1).squeeze(-1).squeeze(-1).view(B, 1, 1, 1))
    tau = rel_t - tau_tt
    phase = 2.0 * math.pi * f0 * tau
    # Scalar-per-record quantities (f0 here) are [B,1,1,1] and must be expanded to
    # the grid before concatenation, or torch.cat sees a size-1 spatial axis.
    ones = torch.ones(B, 1, nz, nx, device=dev, dtype=dt_)
    feats = [X / Lx, Z / Lz, dxr, dzr, radius, tau_tt / domain_t_s, tau / domain_t_s,
             (f0 / 50.0) * ones, torch.sin(phase), torch.cos(phase)]
    # periodic basis on the SOURCE-RELATIVE axes and on tau
    freqs = (2.0 ** torch.arange(fourier_bands, device=dev, dtype=dt_)) * math.pi
    for fr in freqs:
        for ax in (dxr, dzr, tau / domain_t_s):
            feats.append(torch.sin(ax * fr))
            feats.append(torch.cos(ax * fr))
    for sc in gabor_scales:
        w = torch.exp(-0.5 * (tau / sc) ** 2)
        feats.append(w * torch.sin(phase))
        feats.append(w * torch.cos(phase))
    return torch.cat(feats, dim=1)


class AdaptiveWaveAct(nn.Module):
    """Time-conditioned oscillatory activation: `a(t)*sin(w(t)*x) + b(t)*gelu(x)`.

    Why condition on t rather than learn one fixed frequency.  The wavefield's
    dominant spatial wavenumber is not constant over the forecast window: the
    measured spectral centroid of the truth rises from 0.0526 to 0.0682
    cycles/cell between chunk 1 and chunk 27 (1.30x), which in the units this
    activation sees corresponds to w rising from roughly 66 to 86.  A single
    learned frequency is therefore wrong at both ends of the window, and this
    matters precisely where the model is worst: teacher-forced error climbs 2.06x
    along the window (4.84% -> 9.97%) with the rollout removed, so the deficit is
    in the per-segment map at late times, not in error accumulation.

    A small MLP maps the scalar chunk time to per-channel amplitude modulations.
    This is FiLM-style conditioning, which is what the rest of the backbone
    already does with time, so it introduces no new mechanism -- only the
    nonlinearity now participates.

    Initialisation is the load-bearing detail.  The MLP last layer is zeroed, so
    at step 0 every sinusoidal amplitude is exactly 0, the GELU coefficient is
    exactly 1, and the module is BITWISE equal to GELU.  A fresh arm therefore
    starts from the same function as every arm already measured, the oscillatory
    path can only be added by the optimiser, and a failure of this idea degrades
    to the known baseline rather than to something new and untested.

    A MULTI-FREQUENCY BANK, not one learnable frequency.  The obvious design --
    one learnable `w` with the amplitude `a` starting at 0 so the module begins
    as exactly GELU -- is broken, and measurably so: since
    `d/dw [a*sin(w*x)] = a*x*cos(w*x)`, an `a` of exactly 0 makes `dL/dw` exactly
    0, so `w` never moves, while `a` can only become non-zero through the
    gradient it does receive.  Measured on the real module across three seeds:
    `|grad w| = 0.000e+00` while `|grad a| ~ 6e-01`.  `w` is then a fixed
    hyperparameter masquerading as an initial value.

    So the frequencies are FIXED and the coefficients are learned.  `n_freq`
    bands at `w_k = w_base * 2^k` cover a range instead of betting on one value,
    and only their amplitudes `a_k(t)` are learnable -- which have honest
    gradients from step 0.  This is the Rowdy-activation construction
    (Jagtap et al., Neurocomputing 2022): a fixed frequency series, learned
    coefficients, coefficients initialised at zero.

    `w_base` is O(1)-O(10), not SIREN omega_0 = 30, and not the 40-200 rad a naive
    wavenumber conversion suggests.  The reason is that `sin` here sees
    POST-GROUPNORM FEATURE VALUES (measured std 1.0000), not spatial coordinates:
    at w >= 10 the correlation between `sin(w*x)` and `x` has already collapsed
    from +0.92 to -0.003, i.e. the sinusoidal branch stops tracking its input and
    becomes high-frequency hash.  SIREN omega_0 = 30 is inseparable from its
    companion `1/omega_0` layer scaling, whose purpose is exactly to keep the
    argument of each `sin` at unit standard deviation.
    """

    def __init__(self, channels, w_base=2.0, n_freq=3, hidden=32):
        super().__init__()
        self.channels = int(channels)
        self.n_freq = int(n_freq)
        # fixed, geometrically spaced frequencies -- never trained
        self.register_buffer(
            'freqs', w_base * 2.0 ** torch.arange(self.n_freq, dtype=torch.float32))
        self.a = nn.Parameter(torch.zeros(1, self.n_freq, channels, 1, 1))
        self.b = nn.Parameter(torch.ones(1, channels, 1, 1))
        # t -> per-channel (a_0..a_{n-1}, b); zeroed last layer => identity at init
        self.film = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(),
            nn.Linear(hidden, (self.n_freq + 1) * channels))
        with torch.no_grad():
            self.film[-1].weight.zero_()
            self.film[-1].bias.zero_()

    def forward(self, x, t_s=None):
        a, b = self.a, self.b
        if t_s is not None:
            # a 3-vector of t so the MLP sees both the raw time and its phase
            tt = torch.as_tensor(
                [float(t_s), math.sin(2.0 * math.pi * float(t_s)),
                 math.cos(2.0 * math.pi * float(t_s))],
                device=x.device, dtype=x.dtype).view(1, 3)
            d = self.film(tt).view(1, self.n_freq + 1, self.channels, 1, 1)
            a = a + d[:, :self.n_freq]
            b = b + d[:, self.n_freq]
        out = b * F.gelu(x)
        for k in range(self.n_freq):
            out = out + a[:, k] * torch.sin(self.freqs[k] * x)
        return out


class WaveAct(nn.Module):
    """Oscillatory activation for wave fields: `a*sin(w*x) + b*gelu(x)`.

    Why not plain GELU everywhere.  The solution of the acoustic wave equation is
    oscillatory in both space and time, but GELU is monotone and saturating, so a
    network built from it represents an oscillation only by superposing many
    layers of ramps.  A sinusoidal unit represents one directly -- this is the
    SIREN observation (Sitzmann et al. 2020), where periodic activations fit
    signals and their derivatives far better than ReLU-family units, and the
    reason Fourier-feature inputs help PINNs on wave problems.

    Why it is a MIX rather than pure sin.  Pure sinusoidal networks are
    notoriously initialisation-sensitive (SIREN needs its own scheme, w0=30 on
    the first layer), and this backbone is already trained and working with
    GELU; swapping the nonlinearity outright would discard that.  Keeping a
    learnable GELU branch means the block can fall back to its current behaviour,
    so the sinusoidal path can only add.

    `a` starts at 0 and `b` at 1, which makes the module EXACTLY GELU at
    initialisation -- a fresh run therefore starts from the same function as the
    arms already measured, and the optimiser decides how much oscillation to
    bring in.  `w` is learnable per channel and starts at 1.
    """

    def __init__(self, channels=None, w0=1.0):
        super().__init__()
        shape = (1, channels, 1, 1) if channels else (1,)
        self.a = nn.Parameter(torch.zeros(shape))
        self.b = nn.Parameter(torch.ones(shape))
        self.w = nn.Parameter(torch.full(shape, float(w0)))

    def forward(self, x):
        return self.a * torch.sin(self.w * x) + self.b * F.gelu(x)


class SpectralBlock(nn.Module):
    """Inherited parallel-local + identity-skip spectral block.

    `value + gelu(norm(spectral + local))` -- the skip and the 3x3 local path are
    why the chain is NOT band-limited by the spectral truncation (today's
    correction: the cutoff only limits the spectral path).
    """

    def __init__(self, ch, modes):
        super().__init__()
        self.m = int(modes)
        scale = 1.0 / (ch * ch)
        self.w = nn.Parameter(scale * torch.randn(ch, ch, self.m, self.m, dtype=torch.cfloat))
        self.local = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm = nn.GroupNorm(1, ch)
        self.act = AdaptiveWaveAct(ch)

    def forward(self, v, t_s=None):
        b, c, h, wd = v.shape
        ft = torch.fft.rfft2(v, norm='ortho')
        out = torch.zeros_like(ft)
        m = min(self.m, ft.shape[-2] // 2, ft.shape[-1])
        out[..., :m, :m] = torch.einsum('bixy,ioxy->boxy', ft[..., :m, :m], self.w[..., :m, :m])
        spec = torch.fft.irfft2(out, s=(h, wd), norm='ortho')
        return v + self.act(self.norm(spec + self.local(v)), t_s)


class ChunkMIONet(nn.Module):
    """known8 + velocity + source -> next K frames, via MIONet multiplicative fusion.

    Branch / trunk split follows the inherited design, with the capacity
    deliberately on the TRUNK side (the only group measured to rotate the output
    subspace); the medium branch stays small because extra capacity there was
    measured to be useless (grad norm 0.0853, floor rises, lr rebalance -1.31%).
    """

    def __init__(self, *, K, ctx=8, rank=128, width=48, modes=20, blocks=4,
                 trunk_width=256, trunk_layers=3, nz=201, nx=201,
                 dz_m=10.0, dx_m=10.0, fourier_bands=6,
                 gabor_scales=(0.01, 0.025, 0.05, 0.1), legacy_trunk=False):
        super().__init__()
        self.K, self.ctx, self.rank = int(K), int(ctx), int(rank)
        self.nz, self.nx, self.dz_m, self.dx_m = nz, nx, dz_m, dx_m
        self.fourier_bands = int(fourier_bands)
        self.gabor_scales = tuple(float(s) for s in gabor_scales)
        nfeat = 10 + 6 * self.fourier_bands + 2 * len(self.gabor_scales)

        # --- trunk: capacity lives here (it sets the basis directions) ---
        # Measured: three 384-channel 3x3 layers at full 201x201 cost 11.87 ms,
        # which was 69% of the whole 23.58 ms forward -- the single dominant cost.
        # Running the trunk body at half resolution and adding a full-resolution
        # 1x1 bypass cuts that to 2.77 ms (4.28x, forward 2.13x overall) while
        # parameter count goes UP (2.91M vs 2.74M), so this trades resolution of
        # the analytic-feature mixing, not capacity.  Half, not quarter: the trunk
        # is the group measured to set the output basis directions, so its spatial
        # detail is the last thing to give away (stride 4 bought only 2.54x
        # overall for a 51x51 body).
        self.trunk_in = nn.Conv2d(nfeat, trunk_width, 3, stride=2, padding=1)
        # explicit lists rather than Sequential: the activations take t_s
        self.trunk_acts = nn.ModuleList(
            AdaptiveWaveAct(trunk_width) for _ in range(trunk_layers))
        self.trunk_convs = nn.ModuleList(
            nn.Conv2d(trunk_width, trunk_width, 3, padding=1)
            for _ in range(trunk_layers - 1))
        self.trunk_out = nn.Conv2d(trunk_width, rank, 1)
        self.trunk_fine = nn.Conv2d(nfeat, rank, 1)   # full-resolution bypass
        # `legacy_trunk=True` restores the pre-2026-10-14 full-resolution trunk so
        # that checkpoints written by runs started before the split can be read.
        # The two are NOT interchangeable: the half-resolution variant is 4.28x
        # faster on the trunk and holds slightly MORE parameters, so accuracy
        # numbers never cross between them.
        if legacy_trunk:
            tl = [nn.Conv2d(nfeat, trunk_width, 1), nn.GELU()]
            for _ in range(trunk_layers - 1):
                tl += [nn.Conv2d(trunk_width, trunk_width, 3, padding=1), nn.GELU()]
            self.legacy_trunk = nn.Sequential(*tl, nn.Conv2d(trunk_width, rank, 1))
            for name in ('trunk_in', 'trunk_body', 'trunk_out', 'trunk_fine'):
                delattr(self, name)
        else:
            self.legacy_trunk = None

        # --- branches: small, they only weight the trunk's directions ---
        self.medium = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, rank, 1))
        self.source = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, rank))
        self.travel = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, rank, 1))
        self.ctx_branch = nn.Sequential(
            nn.Conv2d(ctx, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, rank, 1))

        # --- spectral body on the fused field, then the chunk head ---
        self.lift = nn.Conv2d(rank, width, 1)
        self.body = nn.ModuleList(SpectralBlock(width, modes) for _ in range(blocks))
        self.head_in = nn.Conv2d(width, width, 1)
        self.head_act = AdaptiveWaveAct(width)
        self.head_out = nn.Conv2d(width, K, 1)
        # inherited two-scale mixing: coarse (fused product) at 1.0, local at 0.1
        self.coarse_scale = nn.Parameter(torch.tensor(1.0))
        self.local_scale = nn.Parameter(torch.tensor(0.1))
        self.local_path = nn.Sequential(
            nn.Conv2d(ctx + 1, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, K, 3, padding=1))
        # Small output init.  The chain is autoregressive: whatever the first
        # forward emits becomes the next input, so an output at the default
        # initialisation scale is amplified chunk after chunk (measured: held-out
        # error 1.8e10% at iteration 3).  Scaling the two output convolutions down
        # starts the rollout near zero and lets the optimiser grow it.  NOT exactly
        # zero: a zero output conv severs the gradient to everything upstream
        # (d loss/d body = head^T @ d loss/d out), which is how today's wide-trunk
        # and dt^2 failures both presented -- silently, with the head still moving.
        for mod in (self.head_out, self.local_path[-1]):
            with torch.no_grad():
                mod.weight.mul_(1e-3)
                if mod.bias is not None:
                    mod.bias.zero_()

    def static_context(self, velocity_mps, src5):
        """Everything that does NOT depend on t, computed once per rollout.

        `straight_ray_tau` integrates slowness along 12 sampled points of every
        source-to-node ray; it takes no time argument, so in an R-segment rollout
        the identical result was being recomputed R times here and another R times
        inside `phase_aligned_features`.  The branches fed only by velocity and
        source are time-invariant for the same reason.  Hoisting them is pure
        arithmetic removal -- the returned tensors keep their graph, so velocity
        and source branches still train.
        """
        tau_tt, dist, slow = straight_ray_tau(
            velocity_mps, src5[:, :2], nz=self.nz, nx=self.nx,
            dz_m=self.dz_m, dx_m=self.dx_m)
        B = velocity_mps.shape[0]
        travel_rank = self.travel(torch.cat((tau_tt, dist / dist.amax().clamp_min(1e-12),
                                             slow * 6000.0), dim=1))
        medium_rank = self.medium(velocity_mps / 6000.0)
        source_rank = self.source(src5 / torch.tensor(
            [2000.0, 2000.0, 50.0, 1.0, 1.0], device=src5.device)).view(B, self.rank, 1, 1)
        return dict(tau_tt=tau_tt, travel_rank=travel_rank,
                    medium_rank=medium_rank, source_rank=source_rank,
                    vel_scaled=velocity_mps / 6000.0)

    def forward(self, ctx_frames, velocity_mps, src5, t_s, static=None):
        """ctx_frames [B,ctx,nz,nx] (normalised), velocity [B,1,nz,nx] m/s,
        src5 [B,5] = (x_m, z_m, f0, t0, amp), t_s scalar start time of this chunk.

        `static` is an optional dict from `static_context()`; passing it across a
        rollout skips the time-invariant work without changing the result.
        """
        B = ctx_frames.shape[0]
        if static is None:
            static = self.static_context(velocity_mps, src5)
        feats = phase_aligned_features(
            velocity_mps, src5, t_s, nz=self.nz, nx=self.nx,
            dz_m=self.dz_m, dx_m=self.dx_m, fourier_bands=self.fourier_bands,
            gabor_scales=self.gabor_scales, tau_tt=static['tau_tt'])
        if self.legacy_trunk is not None:
            # full-resolution trunk, as used by runs started before 2026-10-14 20:00
            trunk_rank = self.legacy_trunk(feats)
        else:
            h = self.trunk_in(feats)
            for i, act in enumerate(self.trunk_acts):
                h = act(h, t_s)
                if i < len(self.trunk_convs):
                    h = self.trunk_convs[i](h)
            h = self.trunk_out(h)
            h = F.interpolate(h, size=feats.shape[-2:], mode='bilinear',
                              align_corners=True)
            trunk_rank = h + self.trunk_fine(feats)
        travel_rank = static['travel_rank']
        medium_rank = static['medium_rank']
        source_rank = static['source_rank']
        ctx_rank = self.ctx_branch(ctx_frames)
        # inherited multiplicative fusion; the sum over rank is the solve
        product = medium_rank * source_rank * travel_rank * trunk_rank * ctx_rank
        fused = self.lift(product / math.sqrt(self.rank))
        for blk in self.body:
            fused = blk(fused, t_s)
        coarse = self.head_out(self.head_act(self.head_in(fused), t_s))
        local = self.local_path(torch.cat((ctx_frames, static['vel_scaled']), dim=1))
        return self.coarse_scale * coarse + self.local_scale * local


__all__ = ['ChunkMIONet', 'AdaptiveWaveAct', 'WaveAct', 'SpectralBlock', 'phase_aligned_features', 'straight_ray_tau']
