"""Offline weighted tensor-product projection of complete native pressure labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .mass_loss import WINDOWS, basis_digest, spatial_action, time_action, validate_windows, window_weights


def decode_tensor(sx, sz, t, coefficients: torch.Tensor) -> torch.Tensor:
    """Dense factors only. This reference path never builds the full operator D."""
    sx, sz, t = (torch.as_tensor(matrix, device=coefficients.device, dtype=coefficients.dtype)
                 for matrix in (sx, sz, t))
    return time_action(spatial_action(coefficients, sz, sx), t)


@dataclass(frozen=True)
class ProjectionRecord:
    coefficients: torch.Tensor  # [B, T_free, Z_free, X_free]
    energy: torch.Tensor        # [B, 3], original teacher energy
    floor: torch.Tensor         # [B, 3], cache-reconstructed residual energy
    linear: torch.Tensor        # [B, 3, T_free, Z_free, X_free], b_j
    basis_signature: str
    diagnostics: dict[str, Any]

    VERSION = 'weighted_qr_full_linear_v1'

    def state_dict(self) -> dict[str, Any]:
        return {'version': self.VERSION, 'coefficients': self.coefficients, 'energy': self.energy,
                'floor': self.floor, 'linear': self.linear, 'basis_signature': self.basis_signature,
                'diagnostics': self.diagnostics}

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> ProjectionRecord:
        if data.get('version') != cls.VERSION:
            raise ValueError('unsupported teacher projection cache format')
        return cls(**{key: data[key] for key in ('coefficients', 'energy', 'floor', 'linear',
                                                 'basis_signature', 'diagnostics')})

    def to(self, device=None, dtype=None) -> ProjectionRecord:
        return type(self)(*(v.to(device=device, dtype=dtype) for v in
                            (self.coefficients, self.energy, self.floor, self.linear)),
                          self.basis_signature, self.diagnostics)

    @classmethod
    def stack(cls, records: list[ProjectionRecord]) -> ProjectionRecord:
        if not records or len({r.basis_signature for r in records}) != 1:
            raise ValueError('records must use one decoder and time partition')
        common = records[0].diagnostics.copy()
        for key in ('reference_normal_relative', 'cached_normal_relative', 'relative_l2_full',
                    'relative_l2_windows', 'condition_time', 'rank_time', 'weighted_floor'):
            if key in common:
                common[key] = [item for record in records for item in record.diagnostics[key]]
        return cls(*(torch.cat([getattr(r, key) for r in records]) for key in
                     ('coefficients', 'energy', 'floor', 'linear')),
                   records[0].basis_signature, common)


def _checked_qr(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    q, r = torch.linalg.qr(matrix, mode='reduced')
    singular = torch.linalg.svdvals(r)
    tolerance = torch.finfo(torch.float64).eps * max(matrix.shape) * singular[0]
    rank = int((singular > tolerance).sum())
    if rank != matrix.shape[1]:
        raise ValueError('rank-deficient evaluation basis; no implicit truncation')
    condition = float(singular[0] / singular[-1])
    if condition > 1e8:
        raise ValueError(f'ill-conditioned reference basis (condition={condition:.3e})')
    return q, r, condition, rank


class TeacherProjector:
    """Reuses the fixed spatial QR across labels. Time weights are sample-specific."""

    def __init__(self, sx, sz, t, windows=WINDOWS):
        self.sx, self.sz, self.t = (torch.as_tensor(matrix, dtype=torch.float64).detach()
                                   for matrix in (sx, sz, t))
        if any(x.ndim != 2 or x.shape[0] < x.shape[1] or x.shape[1] == 0
               or not bool(torch.isfinite(x).all()) for x in (self.sx, self.sz, self.t)):
            raise ValueError('evaluation matrices must be finite and tall or square')
        if len({x.device for x in (self.sx, self.sz, self.t)}) != 1:
            raise ValueError('all three basis matrices must share a device')
        self.windows = validate_windows(windows, self.t.shape[0])
        self.basis_signature = basis_digest(self.sx, self.sz, self.t, self.windows)
        self.qx, self.rx, self.condition_x, self.rank_x = _checked_qr(self.sx)
        self.qz, self.rz, self.condition_z, self.rank_z = _checked_qr(self.sz)

    def _solve_space(self, values: torch.Tensor) -> torch.Tensor:
        rhs_z = self.qz.T @ values
        along_z = torch.linalg.solve_triangular(self.rz, rhs_z, upper=True)
        rhs_x = self.qx.T @ along_z.transpose(-1, -2)
        return torch.linalg.solve_triangular(self.rx, rhs_x, upper=True).transpose(-1, -2).contiguous()

    def _one(self, pressure: torch.Tensor, cache_dtype: torch.dtype, linear_dtype: torch.dtype) -> ProjectionRecord:
        energy = torch.stack([pressure[lo:hi].square().sum() for lo, hi in self.windows])
        weights = window_weights(energy[None])[0]
        common = weights.max()
        normalized = weights / common
        sqrt_weight = torch.cat([normalized[j].sqrt().expand(hi - lo)
                                 for j, (lo, hi) in enumerate(self.windows)])
        q, r, condition_t, rank_t = _checked_qr(sqrt_weight[:, None] * self.t)
        rhs = (q.T * sqrt_weight) @ pressure.flatten(1)
        temporal_solution = torch.linalg.solve_triangular(r, rhs, upper=True)
        reference = self._solve_space(temporal_solution.reshape(self.t.shape[1], *pressure.shape[-2:]))
        # This is D^T W D applied to the reference coefficients minus D^T W p,
        # with the same common weight rescaling as in the solve.
        native_time_modes = spatial_action(reference, self.sz, self.sx).flatten(1)
        normal = (r.T @ (r @ native_time_modes - rhs)).reshape(
            self.t.shape[1], *pressure.shape[-2:])
        normal = spatial_action(normal, self.sz.T, self.sx.T)
        target = spatial_action((r.T @ rhs).reshape(self.t.shape[1], *pressure.shape[-2:]),
                                self.sz.T, self.sx.T)
        target_norm = torch.linalg.vector_norm(target)
        reference_normal = float(torch.linalg.vector_norm(normal) / target_norm) if target_norm > 0 else 0.
        if reference_normal > 1e-10:
            raise FloatingPointError(f'weighted projection normal equation defect: {reference_normal:.3e}')
        del native_time_modes, normal, target, temporal_solution, rhs
        coefficients = reference.to(cache_dtype)
        del reference
        cache_native_modes = spatial_action(coefficients.double(), self.sz, self.sx)
        linear_shape = (3, self.t.shape[1], self.sz.shape[1], self.sx.shape[1])
        linear = torch.empty(linear_shape, device=pressure.device, dtype=linear_dtype)
        cached_normal = torch.zeros(linear_shape[1:], device=pressure.device, dtype=torch.float64)
        floor = []
        for j, (lo, hi) in enumerate(self.windows):
            local_t = self.t[lo:hi]
            support = torch.where(local_t.ne(0).any(dim=0))[0]
            local_t = local_t.index_select(1, support)
            residual = pressure[lo:hi] - time_action(cache_native_modes.index_select(0, support), local_t)
            floor.append(residual.square().sum())
            raw = time_action(residual, local_t.T)
            adjoint = spatial_action(raw, self.sz.T, self.sx.T)
            # Only remove structural zeros; do not threshold small values.
            linear[j].zero_().index_copy_(0, support, adjoint.to(linear_dtype))
            cached_normal.index_add_(0, support, normalized[j] * linear[j].index_select(0, support).double())
        floor = torch.stack(floor)
        cache_defect = float(torch.linalg.vector_norm(cached_normal) / target_norm) if target_norm > 0 else 0.
        ratios = [float((f/e).sqrt()) if e > 0 else None for f, e in zip(floor, energy)]
        diagnostics = {
            'reference_normal_relative': [reference_normal],
            'cached_normal_relative': [cache_defect],
            'relative_l2_full': [float((floor.sum() / energy.sum()).sqrt())],
            'relative_l2_windows': [ratios], 'weighted_floor': [float((weights * floor).sum())],
            'condition_x': self.condition_x, 'condition_z': self.condition_z,
            'condition_time': [condition_t], 'rank_time': [rank_t], 'rank_x': self.rank_x, 'rank_z': self.rank_z,
            'cache_dtype': str(cache_dtype), 'linear_dtype': str(linear_dtype),
        }
        return ProjectionRecord(coefficients[None], energy[None], floor[None], linear[None],
                                self.basis_signature, diagnostics)

    @torch.no_grad()
    def project(self, pressure, *, cache_dtype=torch.float32, linear_dtype=torch.float32) -> ProjectionRecord:
        if cache_dtype not in (torch.float64, torch.float32) or linear_dtype not in (torch.float64, torch.float32):
            raise ValueError('coefficient and residual FP16 caches require independent qualification')
        pressure = torch.as_tensor(pressure, dtype=torch.float64, device=self.t.device)
        if pressure.ndim == 3:
            pressure = pressure.unsqueeze(0)
        shape = (self.t.shape[0], self.sz.shape[0], self.sx.shape[0])
        if pressure.ndim != 4 or tuple(pressure.shape[1:]) != shape or not pressure.shape[0]:
            raise ValueError(f'expected [batch, {shape}] native pressure')
        if not bool(torch.isfinite(pressure).all()):
            raise ValueError('non-finite native pressure')
        with torch.autocast(device_type=pressure.device.type, enabled=False):
            if len(pressure) == 1:
                return self._one(pressure[0], cache_dtype, linear_dtype)
            return ProjectionRecord.stack([self._one(item, cache_dtype, linear_dtype) for item in pressure])


def project_teacher(sx, sz, t, pressure, *, windows=WINDOWS, cache_dtype=torch.float32,
                    linear_dtype=torch.float32) -> ProjectionRecord:
    return TeacherProjector(sx, sz, t, windows).project(pressure, cache_dtype=cache_dtype, linear_dtype=linear_dtype)
