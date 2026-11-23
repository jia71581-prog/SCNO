"""Helpers used by the public training entry points."""
import math
import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

_STENCIL = (-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560)

def _laplacian8(field, stencil, spacing_squared):
    """Symmetric, zero-extended, centered stencil on the entire supplied patch."""
    flat = field.reshape(-1, 1, *field.shape[-2:])
    xx = F.conv2d(flat, stencil.reshape(1, 1, 1, 9), padding=(0, 4))
    zz = F.conv2d(flat, stencil.reshape(1, 1, 9, 1), padding=(4, 0))
    return ((xx + zz) / spacing_squared).reshape_as(field)

class _LocalResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, minus, middle, plus, velocity, h, delta):
        spacing_squared, step_squared = h**2, delta**2
        correction = step_squared / 12
        stencil = middle.new_tensor(_STENCIL)
        c2 = velocity.square()
        lap_middle = _laplacian8(middle, stencil, spacing_squared)
        first = c2 * lap_middle
        lap_first = _laplacian8(first, stencil, spacing_squared)
        second = c2 * lap_first
        ctx.spacing_squared, ctx.step_squared = spacing_squared, step_squared
        if ctx.needs_input_grad[3]:
            ctx.save_for_backward(c2, stencil, velocity, lap_middle, lap_first)
        else:
            ctx.save_for_backward(c2, stencil)
        return (plus - 2 * middle + minus) / step_squared - first - correction * second

    @staticmethod
    @once_differentiable
    def backward(ctx, q):
        c2, stencil, *medium = ctx.saved_tensors
        minus_grad, middle_grad, plus_grad, velocity_grad = (None,) * 4
        needs_minus, needs_middle, needs_plus, needs_velocity = ctx.needs_input_grad[:4]
        correction = ctx.step_squared / 12
        if needs_minus or needs_middle or needs_plus:
            time_grad = q / ctx.step_squared
            minus_grad = time_grad if needs_minus else None
            plus_grad = time_grad if needs_plus else None
        if needs_middle or needs_velocity:
            # L = diag(c²) Delta, so L^T q = Delta(diag(c²) q).
            adjoint_once = _laplacian8(c2 * q, stencil, ctx.spacing_squared)
        if needs_middle:
            adjoint_twice = _laplacian8(c2 * adjoint_once, stencil, ctx.spacing_squared)
            middle_grad = -2 * time_grad - adjoint_once - correction * adjoint_twice
        if needs_velocity:
            velocity, lap_middle, lap_first = medium
            # Differentiate the occurrence of c² in each of the two L applications.
            coefficient_grad = -q * lap_middle - correction * (q * lap_first + lap_middle * adjoint_once)
            velocity_grad = 2 * velocity * coefficient_grad.sum_to_size(velocity.shape)
        return minus_grad, middle_grad, plus_grad, velocity_grad, None, None

def lwc_residual_analytic(pminus, p0, pplus, velocity, h=10., delta=.0005):
    """Return the paper's local residual with its explicit first-order adjoint.

    Inputs follow the vendor API: [z,x] or [query,z,x] pressure and shared
    [z,x] positive velocity, cast to FP64. Indexing, the eight-node halo,
    source-exclusion mask, residual normalization and sample weights belong to
    the caller. PyTorch propagates the resulting cotangent and accumulates
    gradients for repeated time indices. For higher derivatives use the vendor
    residual through ordinary autograd.
    """
    h, delta = float(h), float(delta)
    if not math.isfinite(h) or h <= 0 or not math.isfinite(delta) or delta <= 0:
        raise ValueError("h and delta must be finite and positive")
    minus, middle, plus = (torch.as_tensor(field) for field in (pminus, p0, pplus))
    if any(field.ndim not in (2, 3) or field.is_complex() for field in (minus, middle, plus)):
        raise ValueError("Expected real [z,x] or [query,z,x] fields")
    if minus.shape != middle.shape or plus.shape != middle.shape:
        raise ValueError("The three pressure queries must have identical shapes")
    if minus.device != middle.device or plus.device != middle.device:
        raise ValueError("Pressure queries must be on one device")
    c = torch.as_tensor(velocity, device=middle.device, dtype=torch.float64)
    if c.shape != middle.shape[-2:] or not torch.isfinite(c).all() or not (c > 0).all():
        raise ValueError("Velocity must be finite, positive and match the spatial grid")
    return _LocalResidual.apply(minus.double(), middle.double(), plus.double(), c, h, delta)
