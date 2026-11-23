"""LWC-84-consistent acoustic PDE residual loss (defect-matched, interior-masked).

Residual operator (matches the dataset generator's update formula, attribute
``lwc_update_formula``: ``p_np1 = 2p_n - p_nm1 + dt^2 (L p_n + q) + dt^4/12
(L (L p_n + q) + q_tt)``): in the source-free interior the label satisfies

    D2t[p] = L p + (dt^2/12) L(L p) + O(dt^4, dx^8),

where ``L = c^2 Laplacian_8`` (eighth-order centred, the generator's spatial
order) and ``D2t`` is the second central difference at the SAVED cadence.

Measured on stored labels (interior mask 20 cells, source-extinct frames):
plain ``D2t - L`` leaves 0.6-1.3% relative residual; adding the ``dt^2/12``
term drops it to 0.58% (marmousi) / 0.62% (layered) / 0.0000 (uniform).
That floor is the label's own defect, so the loss is DEFECT-MATCHED per
ACCURACY_DG_OPTIMIZATION_PLAN §9.2: we penalise the residual DIFFERENCE
``A(pred) - A(label)``, not ``A(pred)`` alone — §9.1 forbids forcing the raw
coarse residual to zero (it would push the network toward a different
discrete solution).

Time axis: three model evaluations at (t-dt, t, t+dt) via the caller's dense
head, NOT stored-frame adjacency — the pilot sampler's strata are neither
consecutive nor exact (continuous_fraction interpolates), so stored-frame
differencing is unavailable by construction.

Boundary handling: three-sided CPML (left/right/bottom, outside the physical
domain, with auxiliary memory state not present in these tensors) and a free
surface at z=0. The residual is meaningless where CPML memory acts and the
free-surface ghost extension is not reproduced here, so everything outside an
interior margin is masked out. Default margin 20 cells; measurements at 20 vs
40 cells agree to the third decimal, so 20 is not truncating real physics.

Normalisation: ``encode_pressure`` is a per-record LINEAR scale, so the
residual commutes with it. All terms here operate on normalised fields and
carry the same per-record scale; the defect-matched difference is finally
divided by ``RMS(D2t[label])`` per record, making the loss scale-free.
"""
from __future__ import annotations

import torch

# Eighth-order centred second-derivative stencil (the generator's spatial order).
_W8 = (-1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0,
       -205.0 / 72.0,
       8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0)
_R8 = 4


def laplacian8(field: torch.Tensor, *, dx_m: float) -> torch.Tensor:
    """Eighth-order centred Laplacian on [..., z, x]; valid only >= 4 cells in."""
    if field.shape[-1] < 2 * _R8 + 1 or field.shape[-2] < 2 * _R8 + 1:
        raise ValueError("field too small for the eighth-order stencil")
    weights = torch.tensor(_W8, dtype=field.dtype, device=field.device)
    out = torch.zeros_like(field)
    core = out[..., _R8:-_R8, _R8:-_R8]
    for k in range(2 * _R8 + 1):
        offset = k - _R8
        z_lo, z_hi = _R8 + offset, field.shape[-2] - _R8 + offset
        x_lo, x_hi = _R8 + offset, field.shape[-1] - _R8 + offset
        core += weights[k] * field[..., z_lo:z_hi, _R8:-_R8]
        core += weights[k] * field[..., _R8:-_R8, x_lo:x_hi]
    return out


def lwc_operator(field: torch.Tensor, c2: torch.Tensor, *, dx_m: float,
                 dt_s: float, high_order: bool = True) -> torch.Tensor:
    """Apply L p = c^2 lap8(p) (+ dt^2/12 L(L p) when high_order) per frame.

    ``field``: [record, time, z, x] normalised pressure frames.
    ``c2``:    [record, 1, z, x] squared velocity (m^2/s^2).
    """
    lap = laplacian8(field, dx_m=dx_m) / (dx_m * dx_m)
    l_p = c2 * lap
    if not high_order:
        return l_p
    lap2 = laplacian8(l_p, dx_m=dx_m) / (dx_m * dx_m)
    return l_p + (dt_s * dt_s / 12.0) * (c2 * lap2)


def interior_mask(shape_zx: tuple[int, int], *, margin: int,
                  device, dtype) -> torch.Tensor:
    """[z,x] mask that is 1 strictly inside ``margin`` cells from every edge."""
    z, x = shape_zx
    if 2 * margin >= min(z, x):
        raise ValueError("interior margin leaves no cells")
    mask = torch.zeros((z, x), device=device, dtype=dtype)
    mask[margin:-margin, margin:-margin] = 1.0
    return mask


def pde_defect_matched_loss(
    prediction_triplet: torch.Tensor,
    target_triplet: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    dx_m: float,
    dt_s: float,
    interior_margin: int = 20,
    high_order: bool = True,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Defect-matched LWC-84 residual on (t-dt, t, t+dt) triplets.

    ``prediction_triplet``/``target_triplet``: [record, frame, 3, z, x] where
    the length-3 axis holds fields at (t-dt, t, t+dt) for each supervised
    centre frame. ``velocity_mps``: [record, 1, z, x] (already gathered per
    record — the caller resolves record_to_medium).

    Returns the mean over records and frames of

        || M (A[pred] - A[label]) ||_2 / || M D2t[label] ||_2

    with A[u] = D2t[u] - L u (high-order L when enabled) and M the interior
    mask. Ill-posedness note: the denominator is the label's own inertial
    term over the interior, floored at eps; frames whose D2t vanishes
    identically contribute the numerator against eps and should be excluded
    upstream by the sampler's energy guard.
    """
    if prediction_triplet.shape != target_triplet.shape:
        raise ValueError("prediction/target triplet shapes must match")
    if prediction_triplet.ndim != 5 or prediction_triplet.shape[2] != 3:
        raise ValueError("triplets must be [record, frame, 3, z, x]")
    if velocity_mps.ndim != 4 or velocity_mps.shape[1] != 1:
        raise ValueError("velocity must be [record, 1, z, x]")
    records, frames, _, z, x = prediction_triplet.shape
    c2 = (velocity_mps.to(prediction_triplet.dtype) ** 2).unsqueeze(1)  # [r,1,1,z,x]

    def residual(triplet: torch.Tensor) -> torch.Tensor:
        prev, mid, nxt = triplet[:, :, 0], triplet[:, :, 1], triplet[:, :, 2]
        d2t = (nxt - 2.0 * mid + prev) / (dt_s * dt_s)
        l_mid = lwc_operator(mid, c2.squeeze(2), dx_m=dx_m, dt_s=dt_s,
                             high_order=high_order)
        return d2t - l_mid, d2t

    res_pred, _ = residual(prediction_triplet)
    res_targ, d2t_targ = residual(target_triplet)
    mask = interior_mask((z, x), margin=interior_margin,
                         device=prediction_triplet.device,
                         dtype=prediction_triplet.dtype)
    delta = (res_pred - res_targ) * mask
    denom = (d2t_targ * mask).flatten(start_dim=2).norm(dim=-1).clamp_min(eps)
    numer = delta.flatten(start_dim=2).norm(dim=-1)
    return (numer / denom).mean()


__all__ = ["laplacian8", "lwc_operator", "interior_mask", "pde_defect_matched_loss"]
