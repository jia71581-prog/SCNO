"""R21 native saved201 views; source encoding and PDE density stay distinct."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.util
import sys
import numpy as np
import torch

GRID_N = 201
DX_M = DZ_M = 10.0
DOMAIN_M = 2000.0
VIEW_SCHEMA = "R21_NATIVE_SAVED201_NO_SECOND_RESTRICTION_V1"
GENERATOR = Path(__file__).resolve().parents[1]/'sourcegen'
GENERATOR_SHAS = {
    "source.py": "736a46cdbc55cff7fbc0fe230e09f2177a29a692487641b123bfdf7a3237ae38",
    "restriction.py": "be53a93bced3a8aa7d3ca303e7f367444fb16ca564f2218a4fcec5ac04224792",
}

@lru_cache(maxsize=2)
def _generator_module(filename):
    path = GENERATOR / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != GENERATOR_SHAS[filename]:
        raise PermissionError("frozen generator helper changed: " + filename)
    name = "r21_native201_generator_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _clone_native201(field):
    if tuple(field.shape[-2:]) != (GRID_N, GRID_N):
        raise ValueError("R21 requires actual saved201 input, never resized101")
    if isinstance(field, torch.Tensor):
        if not field.is_floating_point() or not bool(torch.isfinite(field).all()):
            raise ValueError("finite floating native201 field required")
        return field.clone()
    array = np.asarray(field)
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("finite floating native201 field required")
    return array.copy()

def pressure_view(field):
    """Identity on stored201 interior/edges, with the exact physical top row."""
    result = _clone_native201(field)
    result[..., 0, :] = 0
    return result

def velocity_view(field):
    """Actual HDF saved201 velocity; no pressure boundary mask or normalization."""
    return _clone_native201(field)

def _source_coordinates(source_FP64):
    values = np.asarray(source_FP64, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("five finite original FP64 source fields required")
    x, z = float(values[0]), float(values[1])
    if not (0 <= x <= DOMAIN_M and 0 <= z <= DOMAIN_M):
        raise ValueError("source outside physical 2km domain")
    return x, z

def source_map201(source_FP64):
    """Generator-equivalent 10m bilinear encoding, sum of nodal values = 1."""
    x, z = _source_coordinates(source_FP64)
    source = _generator_module("source.py").bilinear_point_source(
        x, z, nx=GRID_N, nz=GRID_N, dx_m=DX_M, dz_m=DZ_M, centering="node")
    result = np.asarray(source.source_map, dtype=np.float32).copy()
    if result.shape != (GRID_N, GRID_N) or np.any(result < 0):
        raise AssertionError("nonnegative source201 encoding")
    if not np.isclose(result.sum(dtype=np.float64), 1., rtol=0, atol=2e-7):
        raise AssertionError("source201 encoding unit nodal mass")
    return result

def projected_source201(source_FP64):
    """PDE Q201 = one B5 restriction of physical fine401 delta_h, then top0.

    delta_h already contains 1/(5m*5m); do not divide by 25 again.
    This is not source_map201/100 and is not renormalized after restriction.
    Source amplitude is handled by pressure normalization/forcing caller.
    No wavefield, unknown truth, PDE solver or GPU is used here.
    """
    x, z = _source_coordinates(source_FP64)
    fine = _generator_module("source.py").bilinear_point_source(
        x, z, nx=401, nz=401, dx_m=5., dz_m=5., centering="node")
    density = np.asarray(fine.delta_h, dtype=np.float64)
    if not np.isclose(density.sum()*25., 1., rtol=0, atol=2e-14):
        raise AssertionError("fine401 delta_h unit physical integral")
    restricted = _generator_module("restriction.py").restrict_nodal_2x(density)
    result = pressure_view(np.asarray(restricted, dtype=np.float64))
    if result.dtype != np.float64 or result.shape != (GRID_N, GRID_N):
        raise AssertionError("PDE projected source201 shape/dtype")
    if np.any(result < 0) or not np.all(result[0] == 0):
        raise AssertionError("PDE projected source201 nonnegative exacttop")
    return result

def protocol():
    from protocol92 import FULL_FRAMES, IC_FRAMES, FORECAST_FRAMES
    return dict(schema=VIEW_SCHEMA, native_stored_frames=list(FULL_FRAMES),
                IC_pressure_frames=list(IC_FRAMES), forecast_frames=list(FORECAST_FRAMES),
                spatial_shape=[GRID_N, GRID_N], dx_m=DX_M, dz_m=DZ_M,
                domain_x_m=[0., DOMAIN_M], domain_z_m=[0., DOMAIN_M],
                pressure="raw saved201 Pa, identity interior, exacttop0, no extra B5",
                velocity="raw saved201 m/s, no extra B5 or interpolation",
                source_map="original generator 10m bilinear201 unit nodal mass",
                PDE_source="original fine401 delta_h -> single B5 -> exacttop0; no renormalization",
                anchor="first observed IC time 0.11s",
                time_spacing_native_steps=[7, 8],
                generator_helpers={name: dict(path=str(GENERATOR/name), SHA256=digest)
                                   for name, digest in GENERATOR_SHAS.items()})
