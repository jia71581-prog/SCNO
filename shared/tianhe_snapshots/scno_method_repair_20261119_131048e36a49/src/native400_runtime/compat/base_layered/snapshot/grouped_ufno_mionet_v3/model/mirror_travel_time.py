"""Single free-surface mirror travel time (L2 carrier, design v2).

Fixed-path slowness integral along the specular single-bounce path
``source -> b -> query`` with the bounce point ``b`` on the free surface z=0.
In a homogeneous medium this is the exact image-source reflection travel time;
in a heterogeneous medium it is ONLY a fixed-polyline slowness integral and is
never claimed to be a Fermat/eikonal reflection travel time
(L2_CARRIER_DESIGN_v2.md section 2).

Contracts carried from the 20260914 review (G-fix 1-3, N-fix 1-3):

- Degenerate branches are pinned, not implicit: zs=0 -> b=s (first segment
  contributes 0); zq=0 -> b=q (second segment 0); zs=zq~0 -> the slowness
  integral along the surface segment s->q (implemented as alpha=1/2, which is
  that integral split into two half-segments); s==q at depth is NOT degenerate
  (alpha=1/2, path length 2*zs).
- Coordinates are clamped to z>=0 BEFORE the bounce arithmetic, and the
  surface branch triggers on zs+zq <= surface_eps_m: the repo's domain
  validation tolerance (travel_time._inside, ~2e-3 m on a 2 km extent) admits
  slightly negative z, and the exact float test zs+zq>0 would then divide by a
  tiny/negative denominator and push b_x out of the x extent unnoticed under
  grid_sample border padding.
- Sampler index order copies travel_time.py exactly: velocity [medium,1,z,x],
  point coordinates (x,z) in the last dim, grid stacked (grid_x, grid_z)
  (grid_sample last-dim order is (W,H) and W=x, H=z here), bilinear, border
  padding, align_corners=True.
- Per-segment trapezoid quadrature, default 256 samples/segment: the frozen
  constant from QUADRATURE_CONVERGENCE_20260914.json (64 points miss the
  0.05 rad @ 30 Hz gate by ~10x on train marmousi; 256 passes with max
  0.029 rad).  Integration runs in float64 regardless of input dtype
  (N-fix 3); the caller casts the returned seconds as needed.

The parent direct-path branch (travel_time.straight_ray_travel_time, uniform
mean-slowness rule, samples=12) is deliberately untouched: harmonising its
quadrature would break the A4 bitwise function-preservation acceptance of
design section 3 (G-fix 4).
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .travel_time import _inside, _validate_extent

_SURFACE_EPS_M = 1.0e-6


def _map_records(velocity: torch.Tensor, records: int, record_to_medium):
    if record_to_medium is None:
        if velocity.shape[0] == 1:
            return torch.zeros(records, dtype=torch.long, device=velocity.device)
        if velocity.shape[0] == records:
            return torch.arange(records, dtype=torch.long, device=velocity.device)
        raise ValueError("record_to_medium is required when medium and record counts differ")
    mapping = torch.as_tensor(record_to_medium, dtype=torch.long, device=velocity.device)
    if mapping.shape != (records,) or torch.any(mapping < 0) or torch.any(mapping >= velocity.shape[0]):
        raise ValueError("record_to_medium contains invalid medium indices")
    return mapping


def _segment_tau(
    velocity: torch.Tensor,
    start: torch.Tensor,
    stop: torch.Tensor,
    *,
    x_extent: tuple[float, float],
    z_extent: tuple[float, float],
    samples: int,
) -> torch.Tensor:
    """Trapezoid slowness integral along straight segments start->stop.

    start/stop: [record, query, 2] float64 (x, z).  velocity: [record, 1, z, x]
    float64 (already medium-mapped).  Returns [record, query] seconds.

    A zero-length segment integrates to exactly 0 (finite, never NaN): the
    sampled slowness is finite and the multiplying length is 0.  This is what
    makes the pinned degenerate branches fall out of the generic path.
    """
    x0, x1 = x_extent
    z0, z1 = z_extent
    alpha = torch.linspace(0.0, 1.0, samples, dtype=torch.float64, device=start.device)
    # [record, query, samples, 2]
    points = start[:, :, None, :] + alpha[None, None, :, None] * (stop - start)[:, :, None, :]
    grid_x = 2.0 * (points[..., 0] - x0) / (x1 - x0) - 1.0
    grid_z = 2.0 * (points[..., 1] - z0) / (z1 - z0) - 1.0
    # (grid_x, grid_z): grid_sample's last dim is (W-coord, H-coord), W=x, H=z.
    grid = torch.stack((grid_x, grid_z), dim=-1)
    records, queries = start.shape[0], start.shape[1]
    sampled = F.grid_sample(
        velocity,
        grid.reshape(records, queries * samples, 1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(records, queries, samples)
    slowness = sampled.reciprocal()
    length = torch.linalg.vector_norm(stop - start, dim=-1)
    return torch.trapezoid(slowness, dx=1.0 / (samples - 1), dim=-1) * length


def single_surface_mirror_travel_time(
    velocity_mps: torch.Tensor,
    source_xy_m: torch.Tensor,
    query_xy_m: torch.Tensor,
    *,
    x_extent_m: tuple[float, float],
    z_extent_m: tuple[float, float],
    record_to_medium: torch.Tensor | None = None,
    samples: int = 256,
    surface_eps_m: float = _SURFACE_EPS_M,
) -> torch.Tensor:
    """Travel time of the single free-surface bounce path, [record, query] s.

    Mirror-point arithmetic never leaves the physical domain: alpha_b is a
    ratio of clamped nonnegative depths, so it lies in [0, 1] and the bounce
    x is a convex combination of in-domain xs, xq.  The image point (xs, -zs)
    is never materialised and never validated.
    """
    velocity = torch.as_tensor(velocity_mps)
    if velocity.ndim != 4 or velocity.shape[1] != 1:
        raise ValueError("velocity_mps must be [medium,1,z,x]")
    if not velocity.dtype.is_floating_point:
        velocity = velocity.float()
    source = torch.as_tensor(source_xy_m, dtype=velocity.dtype, device=velocity.device)
    query = torch.as_tensor(query_xy_m, dtype=velocity.dtype, device=velocity.device)
    if source.ndim != 2 or source.shape[-1] != 2:
        raise ValueError("source_xy_m must be [record,2]")
    if query.ndim != 3 or query.shape[0] != source.shape[0] or query.shape[-1] != 2:
        raise ValueError("query_xy_m must be [record,query,2]")
    if samples < 2:
        raise ValueError("at least two quadrature samples are required")
    x_extent = _validate_extent(x_extent_m, "x")
    z_extent = _validate_extent(z_extent_m, "z")
    if not _inside(source[:, 0], *x_extent) or not _inside(source[:, 1], *z_extent):
        raise ValueError("source lies outside the physical domain")
    if not _inside(query[..., 0], *x_extent) or not _inside(query[..., 1], *z_extent):
        raise ValueError("query lies outside the physical domain")
    if not torch.isfinite(velocity).all() or torch.any(velocity <= 0):
        raise ValueError("velocity must be finite and positive")

    records = source.shape[0]
    mapping = _map_records(velocity, records, record_to_medium)
    velocity64 = velocity[mapping].double()
    src = source.double()
    qry = query.double()

    # Clamp BEFORE any bounce arithmetic (G-fix 2): the domain tolerance
    # admits z down to about -2e-3 m, and a negative depth must not reach the
    # alpha ratio or the surface-branch test.
    zs = src[:, 1].clamp_min(0.0)[:, None]            # [record, 1]
    zq = qry[..., 1].clamp_min(0.0)                   # [record, query]
    xs = src[:, 0][:, None]
    xq = qry[..., 0]

    denom = zs + zq
    at_surface = denom <= surface_eps_m
    alpha = torch.where(at_surface, torch.full_like(denom, 0.5), zs / denom.clamp_min(surface_eps_m))
    bounce_x = xs + alpha * (xq - xs)
    bounce = torch.stack((bounce_x, torch.zeros_like(bounce_x)), dim=-1)

    start = torch.stack((xs.expand_as(xq), zs.expand_as(zq)), dim=-1)
    stop = torch.stack((xq, zq), dim=-1)
    tau = (
        _segment_tau(velocity64, start, bounce, x_extent=x_extent, z_extent=z_extent, samples=samples)
        + _segment_tau(velocity64, bounce, stop, x_extent=x_extent, z_extent=z_extent, samples=samples)
    )
    if not torch.isfinite(tau).all():
        raise RuntimeError("mirror travel time produced non-finite values")
    return tau


def mirror_phase_features(
    tau_ref_s: torch.Tensor,
    time_s: torch.Tensor,
    t0_s: torch.Tensor,
    f0_hz: torch.Tensor,
    *,
    domain_t_s: float,
) -> torch.Tensor:
    """The 4 carrier features [tau/T, -sin(phi), -cos(phi), causal], stacked last.

    phi = 2*pi*f0*(t - t0 - tau_ref); the leading minus signs are the pi phase
    shift tagging the free-surface pressure reflection coefficient -1.  A
    heuristic polarity label absorbable by the zero-init carrier weights, NOT
    an amplitude model and NOT a boundary-condition guarantee
    (L2_CARRIER_DESIGN_v2.md section 2).  causal copies the direct branch's
    convention (features.py: sigmoid(tau / max(domain_t_s*0.01, 1e-6))).
    """
    tau = time_s - t0_s - tau_ref_s
    phase = 2.0 * math.pi * f0_hz * tau
    causal = torch.sigmoid(tau / max(domain_t_s * 0.01, 1.0e-6))
    features = torch.stack(
        (tau_ref_s / domain_t_s, -torch.sin(phase), -torch.cos(phase), causal),
        dim=-1,
    )
    if not torch.isfinite(features).all():
        raise RuntimeError("mirror phase features produced non-finite values")
    return features


__all__ = ["mirror_phase_features", "single_surface_mirror_travel_time"]
