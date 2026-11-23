"""R17 exact discrete free-surface constraint, with no soft depth taper.

The physical output domain has an exact p(z=0)=0 top row.  Its left,
right and bottom edges are interfaces to *external* CFS-CPML halos;
they must not be zeroed or relabelled Dirichlet.  This module introduces
no parameters or persistent buffers.  Call install_exact_boundary after
model construction/loading and before every path's first forward.
"""
from __future__ import annotations

import inspect
import hashlib
from pathlib import Path
from typing import Any

import torch


def exact_nodal_factor(z_m: torch.Tensor) -> torch.Tensor:
    """Zero at the surface; bitwise one at every positive-depth node.

    This is a discrete output constraint, not a differentiable-in-space
    continuation or a CPML approximation.  Time AD commutes with it.
    """
    return (z_m > 0).to(dtype=z_m.dtype)


def install_exact_boundary(model: torch.nn.Module) -> torch.nn.Module:
    """Select the audited shared point/dense exact-row chokepoint.

    DDP wrappers are supported.  All existing parameters, state keys and
    parameter object identities remain unchanged, preserving old Adam state.
    A persisted boundary contract is required to restore this non-tensor mode.
    """
    target = getattr(model, "module", model)
    if not hasattr(target, "free_surface_mode") or not callable(
        getattr(target, "_free_surface_factor", None)
    ):
        raise TypeError("R17 requires the audited shared free-surface factor")
    before_keys = tuple(target.state_dict().keys())
    before_params = tuple(id(p) for p in target.parameters())
    target.free_surface_mode = "exact_row"
    z = torch.tensor([0.0, 1e-9, 10.0, 20.0, 2000.0], dtype=torch.float64)
    actual = target._free_surface_factor(z)
    if not torch.equal(actual, exact_nodal_factor(z)):
        raise RuntimeError("Model exact-row implementation does not match R17")
    if tuple(target.state_dict().keys()) != before_keys or tuple(
        id(p) for p in target.parameters()
    ) != before_params:
        raise RuntimeError("Boundary mode installation changed model state identity")
    return model


def project_top_exact(field: torch.Tensor, *, z_axis: int = -2) -> torch.Tensor:
    """Hard-project a complete composed wavefield after optional corrections.

    All positive-depth values and the physical side/bottom edges are untouched.
    The default matches [record,time,z,x].  The zero branch is independent of
    time and parameters, so first/second time derivatives obey the same top BC.
    """
    if field.ndim < 2:
        raise ValueError("Wavefield must have explicit depth and horizontal axes")
    axis = z_axis % field.ndim
    if field.shape[axis] < 1:
        raise ValueError("Cannot constrain an empty depth axis")
    top = field.narrow(axis, 0, 1)
    interior = field.narrow(axis, 1, field.shape[axis] - 1)
    return torch.cat((torch.zeros_like(top), interior), dim=axis)


def validate_exact_boundary(
    model: torch.nn.Module,
    *,
    normalized_field: torch.Tensor | None = None,
    physical_field: torch.Tensor | None = None,
    time_first: torch.Tensor | None = None,
    time_second: torch.Tensor | None = None,
    z_axis: int = -2,
) -> dict[str, Any]:
    """Fail closed on actual short-run fields/AD supplied by the trainer."""
    target = getattr(model, "module", model)
    if getattr(target, "free_surface_mode", None) != "exact_row":
        raise RuntimeError("R17 boundary mode was not installed after model restore")
    z = torch.tensor([0.0, 1e-9, 10.0, 20.0, 2000.0], dtype=torch.float64)
    if not torch.equal(target._free_surface_factor(z), exact_nodal_factor(z)):
        raise RuntimeError("Shared free-surface factor failed exact-row contract")
    checks: dict[str, Any] = {}
    for label, value in (
        ("normalized_field", normalized_field),
        ("physical_field", physical_field),
        ("time_first", time_first),
        ("time_second", time_second),
    ):
        if value is None:
            continue
        if value.ndim < 2 or not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{label} is not a finite wavefield")
        top = value.narrow(z_axis % value.ndim, 0, 1)
        if bool(torch.count_nonzero(top)):
            raise RuntimeError(f"{label} violates exact zero at z=0")
        checks[label] = {"top_max_abs": 0.0, "finite": True}
    return {
        "mode": "exact_row",
        "factor_exact": True,
        "positive_depth_factor_exact_one": True,
        "actual_field_checks": checks,
        "new_parameters": 0,
    }


def boundary_metadata(model: torch.nn.Module) -> dict[str, Any]:
    target = getattr(model, "module", model)
    path = Path(inspect.getfile(type(target))).resolve()
    return {
        "schema": "r17_exact_discrete_free_surface_v1",
        "mode": str(target.free_surface_mode),
        "top": "exact_p_zero_on_z0_nodal_row",
        "positive_depth_factor": 1.0,
        "soft_boundary_taper": False,
        "left_right_bottom": "external_cfs_cpml_not_output_dirichlet",
        "new_parameters": 0,
        "new_persistent_buffers": 0,
        "model_source": str(path),
        "model_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "time_derivative": "automatic_differentiation",
        "space_residual": "unchanged_lwc8",
        "exact_cpml_auxiliary_residual": False,
    }
