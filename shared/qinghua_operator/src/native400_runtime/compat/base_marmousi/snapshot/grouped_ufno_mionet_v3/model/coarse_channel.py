"""Coarse-solver input channel for the point-query acoustic operator (2026-09-26).

Motivation (read-only probe products, PROBE_Q1.json / PROBE_COARSE.json):

  On the SAME 201x201 10 m velocity model, an order-8 Lax-Wendroff CFS-CPML solve
  (coarse_solver.py, 0.095 s for the full 401-frame 1 s record) beats the trained
  operator's late band on 9/9 dev9 records, and the residual energy ratio
  ||truth - coarse||^2 / ||truth - pred||^2 has median 0.0095.  Fitting the two-term
  model  Y ~ a*P + b*C  (P = operator prediction, C = coarse field, coefficients
  fitted against truth on the scored record = ORACLE upper bound) improves the late
  band by 71-97% on the layered / marmousi families.

This module is the minimal trainable form of that observation.  At every query point
(record, frame, z, x) it forms the pair

    u = [ operator's own normalized prediction , normalized coarse field ]

and adds a zero-initialised bias-free linear projection of u to the field that the
dense decoder / cycle / local path all see, i.e. at the SAME additive point as
arrival_head.MultiArrivalAdditiveHead (operator._dense_block_impl, the
`pressure = pressure + self._arrival_head_term(...)` line).

Zero start.  `projection.weight` is zero-initialised, so the channel output is exactly
0 at the fork and the model is bitwise identical to its parent.  The property is
asserted by coarse_identity_check.py, not assumed here.  With the weight open the
channel realizes exactly the probe's two-term family  delta = a*P + b*C  (a, b are the
two weights) with a small MLP refinement available via `hidden` > 0.

Unit contract.  Both inputs are NORMALIZED pressure (PhysicalNormalizer.encode_pressure
with the record's source amplitude), the same units as the fusion output and the
arrival-head term.  The coarse field is encoded once at cache-build time
(build_coarse_cache.py), so the injection itself is a pure gather + linear map.

Injection point and the free-surface factor.  The channel is added inside
_dense_block_impl BEFORE the final `raw * free_surface_factor(z)` multiply, so its
contribution receives the same tanh(z/20m)^2 taper the parent path does.  Measured on
the dev9 late band, only 0.01-0.44% of the residual energy sits in z < 20 m and the
truth energy there is 1e-4-4e-4 of the late total, so this choice is immaterial in
practice; the taper is kept for consistency with the parent.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class CoarseFieldChannel(nn.Module):
    """Zero-initialised [own prediction, coarse field] -> additive correction.

    Shapes (all in normalized pressure units):

      own       : [..., points]      the operator's prediction at the query points
      coarse    : [..., points]      the coarse solve at the same points
      returns   : [..., points]      the added correction (exactly 0 at init)

    With `hidden=0` the map is the probe's exact two-term family
    `a * own + b * coarse`.  With `hidden>0` a bias-free zero-output MLP is used
    instead so the channel can taper its reliance on the coarse field per point
    (the residual is a cumulative-phase error, so a constant b is a first order
    approximation only); the output layer stays zero-initialised either way.
    """

    INPUTS = 2

    def __init__(self, *, hidden: int = 0, gain: float = 1.0) -> None:
        super().__init__()
        if hidden < 0:
            raise ValueError("hidden must be >= 0")
        self.hidden = int(hidden)
        # Adam moves a zero-initialised output weight by lr regardless of gradient
        # scale; `gain` rescales the effective learning rate of the output layer the
        # same way arrival_head.gain does (there it is 1/hidden).
        self.gain = float(gain)
        if self.hidden == 0:
            self.projection = nn.Linear(self.INPUTS, 1, bias=False)
            nn.init.zeros_(self.projection.weight)
            self.mlp = None
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.INPUTS, self.hidden, bias=False),
                nn.GELU(),
                nn.Linear(self.hidden, 1, bias=False),
            )
            nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, own: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        if own.shape != coarse.shape:
            raise ValueError(
                f"own and coarse point sets disagree: {tuple(own.shape)} vs {tuple(coarse.shape)}")
        features = torch.stack((own, coarse), dim=-1)
        if self.mlp is None:
            delta = self.projection(features).squeeze(-1)
        else:
            delta = self.mlp(features.float()).squeeze(-1)
        return self.gain * delta

    @torch.no_grad()
    def set_two_term(self, a: float, b: float) -> None:
        """Write the probe's `a * own + b * coarse` coefficients directly (hidden=0 only).

        Used by the correction-only smoke: it installs the per-record least-squares
        (oracle) or a shared coefficient pair without any training.
        """
        if self.mlp is not None:
            raise ValueError("set_two_term requires hidden=0")
        self.projection.weight[0, 0] = float(a)
        self.projection.weight[0, 1] = float(b)

    @torch.no_grad()
    def two_term(self) -> tuple[float, float]:
        if self.mlp is not None:
            raise ValueError("two_term requires hidden=0")
        return float(self.projection.weight[0, 0]), float(self.projection.weight[0, 1])


def coarse_frame_indices(time_s: torch.Tensor, dt_s: float, n_frames: int) -> torch.Tensor:
    """Map query times [record, count] to cached coarse-frame indices.

    The dataset's saved axis is uniform (401 frames, 2.5 ms, verified unique diff) but is
    stored as fp32, whose quantum near t = 1.0 s is ~6e-5 s -- a fortieth of the 2.5e-3 s
    frame spacing.  The raw quotient therefore lands a small fraction off an integer, and a
    strict allclose test would reject valid frames.  The tolerance is a quarter frame: it
    absorbs fp32 representation error while still refusing an off-grid time (a half-frame
    window would silently accept a time 0.8 frames away and sample its neighbour).
    """
    if not (dt_s > 0.0):
        raise ValueError("coarse frame spacing must be positive")
    exact = time_s.double() / float(dt_s)
    index = exact.round()
    if not bool((exact - index).abs().max() < 0.25):
        raise ValueError(
            "query times are not aligned to the coarse frame grid; refusing to "
            "index the coarse cache (would sample a neighbouring frame)")
    index = index.long()
    if bool((index < 0).any()) or bool((index >= n_frames).any()):
        raise ValueError("coarse frame index out of range")
    return index


def gather_coarse(
    field: torch.Tensor,
    time_s: torch.Tensor,
    *,
    dt_s: float,
    points: int,
) -> torch.Tensor:
    """Sample a cached coarse field at the dense query (record, frame, point) set.

    `field` is [T, H, W] (one record) or [R, T, H, W]; `time_s` is [R, count].
    Returns [R, count, H*W], ordered exactly like `travel_time.dense_query_coordinates`
    / `_dense_block_impl`'s point axis (meshgrid(z, x, "ij") then flatten, i.e.
    z-major with x fastest), so the values line up with the operator's own output
    without any transpose.

    The dataset's frame grid is uniform, so the frame lookup is an exact round with a
    tolerance guard (`coarse_frame_indices`), never a search: a silent off-grid hit
    would inject the wrong frame.
    """
    if field.ndim == 3:
        field = field[None]
    if field.ndim != 4:
        raise ValueError(f"coarse field must be [T,H,W] or [R,T,H,W], got {tuple(field.shape)}")
    index = coarse_frame_indices(time_s, dt_s, field.shape[1])       # [R, count]
    flat = field.reshape(field.shape[0], field.shape[1], -1)         # [R, T, H*W]
    if flat.shape[-1] != points:
        raise ValueError(
            f"coarse field point count {flat.shape[-1]} does not match the dense grid {points}")
    if index.shape[0] != flat.shape[0]:
        raise ValueError("coarse field record count does not match the query set")
    # Per-record advanced index keeps this [R, count, points] without materialising a
    # [R, count, points, 3] index tensor; R is 1 on the training path.
    pieces = [flat[r].index_select(0, index[r]) for r in range(flat.shape[0])]
    return torch.stack(pieces, dim=0)


__all__ = ["CoarseFieldChannel", "coarse_frame_indices", "gather_coarse"]
