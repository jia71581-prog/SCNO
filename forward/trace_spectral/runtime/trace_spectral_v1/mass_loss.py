"""Original node-weighted objective in the assembled coefficient basis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .project_teacher import ProjectionRecord


WINDOWS = ((0, 200), (200, 480), (480, 801))
WINDOW_WEIGHTS = (.25, .35, .40)


def validate_windows(windows, frames: int) -> tuple[tuple[int, int], ...]:
    spans = tuple((int(lo), int(hi)) for lo, hi in windows)
    if (len(spans) != 3 or spans[0][0] != 0 or spans[-1][1] != frames
            or any(lo >= hi for lo, hi in spans)
            or any(left[1] != right[0] for left, right in zip(spans, spans[1:]))):
        raise ValueError('the three loss windows must partition the native time nodes')
    return spans


def basis_digest(sx, sz, t, windows) -> str:
    """Bind cached statistics to the actual free-coordinate decoder matrices."""
    digest = hashlib.sha256(json.dumps(windows).encode('ascii'))
    for basis in (sx, sz, t):
        array = torch.as_tensor(basis).detach().to(device='cpu', dtype=torch.float64).contiguous().numpy()
        digest.update(json.dumps(array.shape).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def window_weights(energy: torch.Tensor) -> torch.Tensor:
    if energy.ndim != 2 or energy.shape[-1] != 3:
        raise ValueError('teacher energy must have shape [batch, 3]')
    total = energy.sum(-1, keepdim=True)
    if not bool(torch.isfinite(energy).all() & torch.isfinite(total).all()
                & (energy >= 0).all() & (total > 0).all()):
        raise ValueError('non-finite, negative or zero-total teacher energy')
    return .005 * (.5 / total + .5 * energy.new_tensor(WINDOW_WEIGHTS)
                   / torch.maximum(energy, .02 * total))


def spatial_action(a: torch.Tensor, sz: torch.Tensor, sx: torch.Tensor) -> torch.Tensor:
    """Apply [out_z,in_z] and [out_x,in_x] factors to the final two axes."""
    return (sz @ a) @ sx.transpose(-1, -2)


def time_action(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply the temporal factor while retaining the final two spatial axes."""
    z, x = a.shape[-2:]
    return (t @ a.flatten(-2)).reshape(*a.shape[:-3], t.shape[0], z, x)


@dataclass(frozen=True)
class LossStats:
    loss: torch.Tensor
    objective: torch.Tensor
    rss: torch.Tensor
    score: torch.Tensor

    @property
    def raw_loss(self) -> torch.Tensor:
        return self.objective.mean()


class MassLoss(nn.Module):
    """Exact in real arithmetic, including per-window cached projection defects.

    Never builds the Kronecker matrix. The temporal submatrices restrict only
    the structural zeros of the native evaluation matrix, never small values.
    """

    def __init__(self, sx, sz, t, windows=WINDOWS):
        super().__init__()
        sx, sz, t = (torch.as_tensor(matrix, dtype=torch.float64).detach() for matrix in (sx, sz, t))
        self.windows = validate_windows(windows, t.shape[0])
        self.basis_signature = basis_digest(sx, sz, t, self.windows)
        self.coefficient_shape = (t.shape[1], sz.shape[1], sx.shape[1])
        self.register_buffer('gx', sx.T @ sx)
        self.register_buffer('gz', sz.T @ sz)
        for j, (lo, hi) in enumerate(self.windows):
            local_t = t[lo:hi]
            support = torch.where(local_t.ne(0).any(dim=0))[0]
            local_t = local_t.index_select(1, support)
            self.register_buffer(f'time_support_{j}', support)
            self.register_buffer(f'time_gram_{j}', local_t.T @ local_t)

    def _terms(self, coefficients: torch.Tensor, record: ProjectionRecord, dtype: torch.dtype):
        prediction = coefficients.to(dtype)
        delta = prediction - record.coefficients.to(device=prediction.device, dtype=dtype)
        gx = self.gx.to(device=prediction.device, dtype=dtype)
        gz = self.gz.to(device=prediction.device, dtype=dtype)
        space = spatial_action(delta, gz, gx)
        linear = record.linear.to(device=prediction.device, dtype=dtype)
        quadratic, cross = [], []
        for j in range(3):
            gram = getattr(self, f'time_gram_{j}').to(device=prediction.device, dtype=dtype)
            support = getattr(self, f'time_support_{j}').to(device=prediction.device)
            temporal = time_action(space.index_select(-3, support), gram)
            quadratic.append((delta.index_select(-3, support) * temporal).flatten(1).sum(-1))
            # A rounded center generally leaves cross terms even for modal
            # supports entirely within one window. Contract all directions.
            cross.append(-2 * (delta * linear[:, j]).flatten(1).sum(-1))
        return (torch.stack(quadratic, dim=-1), torch.stack(cross, dim=-1),
                record.floor.to(device=prediction.device, dtype=dtype))

    def forward(self, coefficients: torch.Tensor, record: ProjectionRecord,
                sample_weights: torch.Tensor | None = None) -> LossStats:
        if coefficients.ndim == 3:
            coefficients = coefficients.unsqueeze(0)
        if tuple(coefficients.shape[1:]) != self.coefficient_shape or coefficients.shape != record.coefficients.shape:
            raise ValueError('coefficient or batch shape differs from the cached projection')
        if record.basis_signature != self.basis_signature:
            raise ValueError('projection cache was produced by a different decoder or time partition')
        dtype = torch.float64 if coefficients.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=coefficients.device.type, enabled=False):
            terms = self._terms(coefficients, record, dtype)
            rss = sum(terms)
            if bool((rss < 0).any()) and dtype != torch.float64:
                # Recheck cancellation before classifying the error. Recompute
                # from the registered FP64 Grams, not from rounded products.
                terms = self._terms(coefficients, record, torch.float64)
                rss = sum(terms)
            if not bool(torch.isfinite(rss).all()):
                raise FloatingPointError('non-finite projected window residual')
            if bool((rss < 0).any()):
                # Explicitly reject significant inconsistencies. A few FP64 ulps
                # at a cancellation point must not alter the optimized objective.
                term_scale = sum(v.abs() for v in terms)
                allowance = 64 * torch.finfo(rss.dtype).eps * term_scale
                if bool((rss < -allowance).any()):
                    raise FloatingPointError('negative projected RSS beyond FP64 cancellation tolerance')
            energy = record.energy.to(device=rss.device, dtype=rss.dtype)
            weights = window_weights(energy)
            objective = (weights * rss).sum(-1)
            if sample_weights is not None:
                sample_weights = torch.as_tensor(sample_weights, device=objective.device,
                                                 dtype=objective.dtype).detach()
                if (sample_weights.shape != objective.shape or
                        not bool(torch.isfinite(sample_weights).all() & (sample_weights > 0).all())):
                    raise ValueError('invalid positive importance weights for the actual batch size')
                loss = (sample_weights * objective).mean()
            else:
                loss = objective.mean()
            # Roundoff-only negative RSS was checked above. Clean just the
            # detached sampling statistic; the objective retains its gradient.
            score = 2 * torch.sqrt((weights.square() * rss.detach().clamp_min(0)).sum(-1))
        return LossStats(loss, objective, rss, score)
