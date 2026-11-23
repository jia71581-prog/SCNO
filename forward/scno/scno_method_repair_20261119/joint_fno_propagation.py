"""Condition a learned, compact pressure pulse on direct and free-surface-reflected travel times."""

import math
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PropagationContext(NamedTuple):
    source: torch.Tensor
    speed_condition: torch.Tensor
    travel_times: torch.Tensor
    delay_step: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


def _cubic_delay_weights(fraction):
    u2, u3 = fraction.square(), fraction.pow(3)
    return torch.stack((-.5 * fraction + u2 - .5 * u3, 1. - 2.5 * u2 + 1.5 * u3,
                        .5 * fraction + 2. * u2 - 1.5 * u3, -.5 * u2 + .5 * u3), dim=2)


class JointFNOPropagation(nn.Module):
    """Separable propagation feature with learned pulse coefficients in normalized-pressure units.

    Travel-time interpolation has finite endpoints. The pulse uses compact, nonperiodic
    cubic B-splines in source periods. Both paths share the learned source waveform.
    """

    def __init__(self, native_times, *, delay_bins=None, wavelet_knots=49, dx=10., ray_samples=12,
                 grid_shape=(201, 201), lag_range=(-2., 4.), reference_speed=2500.):
        super().__init__()
        axis = torch.as_tensor(native_times, dtype=torch.float32).detach().clone()
        if axis.ndim != 1 or len(axis) < 2 or not torch.isfinite(axis).all() or not (axis[1:] > axis[:-1]).all():
            raise ValueError("expected a finite increasing native pressure axis")
        delay_bins = (256 if len(axis) <= 401 else 512) if delay_bins is None else delay_bins
        nz, nx = grid_shape
        if min(delay_bins, wavelet_knots) < 4 or min(ray_samples, nz - 1, nx - 1) < 1:
            raise ValueError("the delay, pulse and ray grids must contain enough samples")
        if not (0 < dx and 0 < reference_speed and lag_range[0] < lag_range[1]):
            raise ValueError("invalid spatial spacing, velocity unit or pulse lag range")
        self.delay_bins, self.wavelet_knots = int(delay_bins), int(wavelet_knots)
        self.dx, self.reference_speed = float(dx), float(reference_speed)
        self.lag_min, self.lag_max = map(float, lag_range)
        self.pulse_step = (self.lag_max - self.lag_min) / (wavelet_knots - 1)
        self.register_buffer("native_times", axis)
        self.register_buffer("delay_ids", torch.arange(delay_bins, dtype=torch.float32), persistent=False)
        self.register_buffer("offsets", torch.arange(-1, 3), persistent=False)
        self.register_buffer("midpoints", (torch.arange(ray_samples) + .5) / ray_samples, persistent=False)
        self.register_buffer("x_m", (torch.arange(nx) * self.dx).reshape(1, 1, nx), persistent=False)
        self.register_buffer("z_m", (torch.arange(nz) * self.dx).reshape(1, nz, 1), persistent=False)
        # No starting waveform is prescribed. The second bank learns its dependence on source velocity.
        self.waveform = nn.Parameter(torch.zeros(2, wavelet_knots))
        self.spreading_adjustment = nn.Parameter(torch.zeros(()))
        self.receiver_speed_adjustment = nn.Parameter(torch.zeros(()))
        self.path_speed_adjustment = nn.Parameter(torch.zeros(()))

    def _straight_travel(self, velocity, source, distance):
        batch, _, height, width = velocity.shape
        f = self.midpoints[None, None, :, None, None]
        sx = source[:, 0, None, None, None, None]
        sz = torch.stack((source[:, 1], -source[:, 1]), dim=1)[:, :, None, None, None]
        x = (sx + f * (self.x_m - sx)).expand(-1, 2, -1, height, -1)
        z = (sz + f * (self.z_m - sz)).abs().expand(-1, -1, -1, -1, width)
        grid = torch.stack((2 * x / ((width - 1) * self.dx) - 1,
                            2 * z / ((height - 1) * self.dx) - 1), dim=-1)
        slowness = F.grid_sample(velocity.reciprocal(), grid.reshape(batch, -1, width, 2),
                                 mode="bilinear", align_corners=True, padding_mode="border")
        return distance * slowness.reshape(batch, 2, len(self.midpoints), height, width).mean(dim=2)

    def prepare(self, velocity, source, travel_times=None):
        """Map input-only geometry to compact nodal interpolation weights for the two paths."""
        velocity = torch.as_tensor(velocity, device=self.x_m.device, dtype=self.x_m.dtype)
        source = torch.as_tensor(source, device=self.x_m.device, dtype=self.x_m.dtype).reshape(-1, 5)
        if velocity.ndim == 2:
            velocity = velocity[None, None]
        elif velocity.ndim == 3:
            velocity = velocity[:, None]
        height, width = self.z_m.shape[-2], self.x_m.shape[-1]
        if velocity.ndim != 4 or velocity.shape[1:] != (1, height, width) or len(velocity) not in (1, len(source)):
            raise ValueError("expected one shared or one registered velocity grid per source")
        velocity = velocity.expand(len(source), -1, -1, -1)
        sx, sz = source[:, 0, None, None], source[:, 1, None, None]
        rx = self.x_m - sx
        rz = torch.stack((self.z_m - sz, self.z_m + sz), dim=1)
        offsets = torch.stack(torch.broadcast_tensors(rx[:, None], rz), dim=-1)
        distance = torch.linalg.vector_norm(offsets, dim=-1)
        if travel_times is None:
            travel = self._straight_travel(velocity, source, distance)
        else:
            travel = torch.as_tensor(travel_times, dtype=velocity.dtype, device=velocity.device)
            if travel.shape != distance.shape:
                raise ValueError("supply physical direct and reflected travel times [B,2,Z,X]")
        xy = torch.stack((2 * source[:, 0] / ((width - 1) * self.dx) - 1,
                          2 * source[:, 1] / ((height - 1) * self.dx) - 1), dim=-1)
        source_speed = F.grid_sample(velocity, xy[:, None, None], mode="bilinear",
                                     align_corners=True, padding_mode="border").flatten()
        cs = source_speed[:, None, None, None]
        path_speed = torch.where(travel > 0, distance / travel.clamp_min(1e-12), cs)
        spread = .5 + .5 * self.spreading_adjustment.tanh()
        envelope = torch.exp(-.5 * spread * torch.log1p((distance / self.dx).square())
                             + self.receiver_speed_adjustment.tanh() * torch.log(velocity / cs)
                             + self.path_speed_adjustment.tanh() * torch.log((path_speed / cs).clamp_min(1e-12)))
        limit = self.native_times[-1] - source[:, 3] - (self.lag_min - 2 * self.pulse_step) / source[:, 2]
        span = torch.minimum(travel.flatten(1).amax(1), limit).clamp_min(self.native_times[1] - self.native_times[0])
        delay_step = span / (self.delay_bins - 1)
        position = travel.flatten(2) / delay_step[:, None, None]
        floor = torch.floor(position)
        indices = (floor.to(torch.long)[:, :, None] + self.offsets[None, None, :, None]).clamp(0, self.delay_bins - 1)
        signed = torch.stack((envelope[:, 0], -envelope[:, 1]), dim=1)
        valid = ((self.z_m[:, None] > 0) & (travel <= span[:, None, None, None]) &
                 (source[:, 4, None, None, None] != 0))
        weights = _cubic_delay_weights(position - floor) * (signed * valid).flatten(2)[:, :, None]
        return PropagationContext(source, torch.log(source_speed / self.reference_speed),
                                  travel, delay_step, indices, weights)

    def spatial_basis(self, context):
        """Return [B,K,Z,X] for concatenation with the FNO pressure projection's spatial inputs."""
        batch, _, height, width = context.travel_times.shape
        basis = context.weights.new_zeros((batch, self.delay_bins, height * width))
        basis.scatter_add_(1, context.indices.flatten(1, 2), context.weights.flatten(1, 2))
        return basis.reshape(batch, self.delay_bins, height, width)

    def _pulse_weights(self, phase):
        position = (phase - self.lag_min) / self.pulse_step
        floor = torch.floor(position)
        u = position - floor
        indices = floor.to(torch.long)[..., None] + self.offsets
        weights = torch.stack(((1 - u).pow(3), 3*u.pow(3) - 6*u.square() + 4,
                               -3*u.pow(3) + 3*u.square() + 3*u + 1, u.pow(3)), dim=-1) / 6
        valid = (indices >= 0) & (indices < self.wavelet_knots)
        return indices.clamp(0, self.wavelet_knots - 1), weights * valid

    def time_rows(self, context, slots=None):
        """Return learned source-conditioned rows [B,T,K] on all or selected native output channels."""
        batch = len(context.source)
        times = self.native_times
        if slots is not None:
            slots = torch.as_tensor(slots, device=times.device)
            if slots.dtype not in (torch.int32, torch.int64) or slots.ndim not in (1, 2):
                raise ValueError("expected native integer time indices [T] or [B,T]")
            if slots.ndim == 2 and slots.shape[0] != batch:
                raise ValueError("per-source native indices must have one row per source")
            times = times.index_select(0, slots.reshape(-1)).reshape(slots.shape)
        if times.ndim == 1:
            times = times[None].expand(batch, -1)
        centers = context.delay_step[:, None] * self.delay_ids
        phase = ((times[:, :, None] - context.source[:, None, 3:4] - centers[:, None]) *
                 context.source[:, None, 2:3])
        knot, weight = self._pulse_weights(phase)
        amplitude = self.waveform[0] + context.speed_condition[:, None] * self.waveform[1]
        wave = amplitude.gather(1, knot.reshape(batch, -1)).reshape(knot.shape)
        return (wave * weight).sum(dim=-1) * (times > 0)[:, :, None]

    def forward(self, velocity, source, slots=None, *, travel_times=None):
        context = self.prepare(velocity, source, travel_times=travel_times)
        basis = self.spatial_basis(context)
        rows = self.time_rows(context, slots)
        return torch.bmm(rows, basis.flatten(2)).reshape(len(rows), rows.shape[1], *basis.shape[-2:])

    def calibration_design(self, context, times, flat_nodes):
        """Exact linear design [B,Q,2*wavelet_knots] for paired physical times and grid indices."""
        batch = len(context.source)
        times = torch.as_tensor(times, dtype=self.x_m.dtype, device=self.x_m.device)
        flat_nodes = torch.as_tensor(flat_nodes, dtype=torch.long, device=self.x_m.device)
        if times.ndim == 1:
            times = times[None].expand(batch, -1)
        if flat_nodes.ndim == 1:
            flat_nodes = flat_nodes[None].expand(batch, -1)
        if times.shape != flat_nodes.shape or times.ndim != 2 or len(times) != batch:
            raise ValueError("expected paired times and flat nodes for each source")
        select = flat_nodes[:, None, None].expand(-1, 2, 4, -1)
        indices = context.indices.gather(-1, select)
        amplitude = context.weights.gather(-1, select)
        arrivals = indices * context.delay_step[:, None, None, None]
        lag = times[:, None, None] - context.source[:, 3, None, None, None] - arrivals
        knot, weight = self._pulse_weights(lag * context.source[:, 2, None, None, None])
        knot = knot.permute(0, 3, 1, 2, 4).flatten(2)
        weight = (weight * amplitude[..., None]).permute(0, 3, 1, 2, 4).flatten(2)
        design = times.new_zeros((batch, times.shape[-1], self.wavelet_knots))
        design.scatter_add_(-1, knot, weight)
        design = design * (times > 0)[..., None]
        return torch.cat((design, design * context.speed_condition[:, None, None]), dim=-1)


def _calibration_queries(module, context, count, rng):
    times = module.native_times
    batch, _, nz, nx = context.travel_times.shape
    draw_shape = (batch, count)
    nodes = torch.tensor(rng.integers(nx, nz * nx, size=draw_shape), device=times.device)
    tau = context.travel_times.flatten(2).gather(2, nodes[:, None].expand(-1, 2, -1))
    delay = context.source[:, 3, None, None] + tau
    freq = context.source[:, 2, None, None]
    # Cover the pulse support and the neighbouring cubic travel-time interpolation knots.
    low = delay + (module.lag_min - 2 * module.pulse_step) / freq - 2 * context.delay_step[:, None, None]
    high = delay + (module.lag_max + 2 * module.pulse_step) / freq + 2 * context.delay_step[:, None, None]
    lo = torch.searchsorted(times, low.contiguous())
    hi = torch.searchsorted(times, high.contiguous(), right=True)
    empty = lo >= hi
    lo, hi = torch.where(empty, 0, lo), torch.where(empty, len(times), hi)
    whole = torch.zeros_like(nodes[:, None])
    lo = torch.cat((whole, lo), dim=1)
    hi = torch.cat((whole + len(times), hi), dim=1)
    u = torch.tensor(rng.random(draw_shape, dtype=np.float32), device=times.device)
    group = (u >= .25).long() + (u >= .625).long()
    lower = lo.gather(1, group[:, None]).squeeze(1)
    sizes = hi - lo
    length = sizes.gather(1, group[:, None]).squeeze(1)
    draws = torch.tensor(rng.random(draw_shape, dtype=np.float32), device=times.device)
    slots = lower + (length * draws).long().minimum(length - 1)
    probabilities = times.new_tensor([.25, .375, .375])[None, :, None]
    density = (probabilities * ((slots[:, None] >= lo) & (slots[:, None] < hi)) / sizes).sum(1)
    return nodes, slots, density


@torch.no_grad()
def fit_waveform(module, records, *, samples_per_record=32768, seed=261119, ridge_relative=1e-4, optimizer=None):
    """Fit training-only learned pulse coefficients with inverse-probability-weighted, streaming QR.

    Each item is (prepared_context, normalized_target) or additionally supplies a base
    prediction to subtract. Per-record weights use complete target pressure energy.
    """
    if samples_per_record < 1 or ridge_relative < 0:
        raise ValueError("expected a positive sample count and nonnegative ridge factor")
    rng = np.random.default_rng(seed)
    anchor = module.waveform.detach().double().flatten()
    moments = anchor.new_empty((0, anchor.numel() + 1))
    totals = []
    for record in records:
        if len(record) not in (2, 3):
            raise ValueError("supply context, target, and optionally a base prediction")
        context, target = record[:2]
        target = torch.as_tensor(target, device=anchor.device)
        if target.ndim == 3:
            target = target[None]
        batch, _, nz, nx = context.travel_times.shape
        if target.shape != (batch, len(module.native_times), nz, nx):
            raise ValueError("calibration requires the full registered target trajectory")
        nodes, slots, density = _calibration_queries(module, context, samples_per_record, rng)
        row = torch.arange(batch, device=anchor.device)[:, None]
        values = target.flatten(2)[row, slots, nodes].double()
        if len(record) == 3 and record[2] is not None:
            base = torch.as_tensor(record[2], device=anchor.device)
            base = base[None] if base.ndim == 3 else base
            if base.shape != target.shape:
                raise ValueError("residual calibration requires a base field aligned with the target")
            values -= base.flatten(2)[row, slots, nodes].double()
        energies = torch.stack([sum(slab.double().square().sum() for slab in seq.split(32)) for seq in target])
        keep = energies > 0
        weights = torch.where(keep[:, None], (nz - 1) * nx / (samples_per_record * density * energies[:, None]), 0.)
        informative = torch.zeros(batch, dtype=torch.long, device=anchor.device)
        for j in range(0, samples_per_record, 8192):
            sl = slice(j, j + 8192)
            design = module.calibration_design(context, module.native_times[slots[:, sl]], nodes[:, sl]).double()
            informative += (design != 0).any(-1).sum(1)
            augmented = torch.cat((design, values[:, sl, None]), dim=-1) * weights[:, sl, None].sqrt()
            moments = torch.linalg.qr(torch.cat((moments, augmented.flatten(0, 1))), mode="r").R
        totals.extend({"reference_energy": float(e), "sampled_rows": samples_per_record, "informative_rows": int(n)}
                      for e, n in zip(energies, informative))
    if not totals or not any(entry["reference_energy"] > 0 and entry["informative_rows"] > 0 for entry in totals):
        raise ValueError("calibration requires a non-silent training record with nonzero propagation design rows")
    x, y = moments[:, :-1], moments[:, -1]
    ridge = max(1e-12, ridge_relative * float(x.square().sum()) / anchor.numel())
    damping = math.sqrt(ridge)
    eye = torch.eye(anchor.numel(), device=anchor.device, dtype=anchor.dtype)
    theta = torch.linalg.lstsq(torch.cat((x, damping * eye)), torch.cat((y, damping * anchor))).solution
    candidate = theta.to(module.waveform.dtype).double()
    before = float((x @ anchor - y).square().sum())
    after = float((x @ candidate - y).square().sum())
    if not math.isfinite(after):
        raise RuntimeError("the fitted waveform has nonfinite weighted calibration error")
    if after <= before:
        module.waveform.copy_(candidate.reshape_as(module.waveform))
    else:
        after = before
    if optimizer is not None and not torch.equal(anchor, module.waveform.flatten().double()):
        optimizer.state.pop(module.waveform, None)
    return {"weighted_sse_before": before, "weighted_sse_after": after, "ridge": ridge,
            "seed": seed, "records": totals, "wavelet_coefficients": anchor.numel(),
            "sampled_rows": len(totals) * samples_per_record}
