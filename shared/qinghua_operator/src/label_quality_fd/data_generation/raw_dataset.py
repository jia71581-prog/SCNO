"""401-node physical solves with unfiltered 201-node output and two layers."""
from __future__ import annotations

import math
import numpy as np

from .grid import AcousticGrid, BoundaryConfig
from .solver_lwc84_fused_device import DeviceResidentFusedLWC84CPMLSolver
from .velocity_models_lwc84 import load_marmousi_crop


class NodalLWC84Solver(DeviceResidentFusedLWC84CPMLSolver):
    """Only the observation operator changes; all physical steps are inherited."""

    def _restrict_output(self, physical):
        if self.output_restriction_factor != 2:
            raise ValueError("raw dataset requires 401-to-201 nodal stride two")
        return physical[..., ::2, ::2].clone()


def two_layer_velocity(grid: AcousticGrid, seed: int):
    rng = np.random.default_rng(int(seed))
    # Consume the historical layer-count draw. Existing two-layer seeds then
    # reconstruct exactly, while other seeds are deliberately forced to two.
    original_count = int(rng.integers(2, 6))
    minimum = 150.0
    thickness = minimum + (grid.lz_m - 2 * minimum) * rng.dirichlet(np.ones(2))
    for _ in range(10_000):
        values = np.sort(rng.uniform(1600.0, 6000.0, size=2))
        if values[1] - values[0] >= 200.0:
            break
    else:
        raise RuntimeError("cannot draw two layer velocities")
    style = "horizontal" if int(seed) % 2 == 0 else ("tilted" if int(seed) % 4 == 1 else "gently_wavy")
    shift = np.zeros_like(grid.x_m)
    amplitude = 0.0
    phase = None
    if style != "horizontal":
        clearance = min(float(thickness[0] - minimum), float(thickness[1] - minimum))
        amplitude = max(0.0, min(50.0, 0.45 * clearance))
        if style == "tilted":
            shift = amplitude * (2 * grid.x_m / grid.lx_m - 1)
        else:
            phase = float(rng.uniform(0, 2 * math.pi))
            shift = amplitude * np.sin(2 * math.pi * grid.x_m / grid.lx_m + phase)
    interface = float(thickness[0]) + shift
    velocity = values[(grid.z_m[:, None] >= interface[None, :]).astype(np.int64)].astype(np.float32)
    if np.unique(velocity).size != 2:
        raise ValueError("exactly two realised layer speeds required")
    meta = dict(seed=int(seed), n_layers=2, original_layer_count_draw=original_count,
                layer_order="ascending_with_depth", layer_velocity_mps=values.tolist(),
                base_interface_z_m=[float(thickness[0])], interface_style=style,
                interface_shift_amplitude_m=amplitude, interface_phase_rad=phase,
                minimum_realized_thickness_m=float(min(interface.min(), grid.lz_m-interface.max())))
    return velocity, meta


class MediumFactory:
    def __init__(self, config):
        self.config = config
        self.grid = AcousticGrid(nx=401,nz=401,dx_m=5.,dz_m=5.,lx_m=2000.,lz_m=2000.,centering="node")
        self.marmousi_cache = {}

    def make(self, row):
        if row["medium_type"] == "layered":
            return two_layer_velocity(self.grid, row["medium_parameters"]["seed"])[0]
        if row["medium_type"] != "marmousi":
            raise ValueError("TRAIN layered/marmousi only")
        p = row["medium_parameters"]
        key = (float(p["crop_x0_m"]),float(p["crop_z0_m"]),bool(p.get("mirror_x",False)))
        if key not in self.marmousi_cache:
            c = self.config["marmousi"]
            velocity, _ = load_marmousi_crop(c["velocity_file"],grid=self.grid,
                source_dx_m=c["source_dx_m"],source_dz_m=c["source_dz_m"],source_unit=c["velocity_unit"],
                crop_x0_m=key[0],crop_z0_m=key[1],interpolation=c["interpolation"])
            if key[2]: velocity=velocity[:,::-1].copy()
            self.marmousi_cache[key]=np.ascontiguousarray(velocity)
        return self.marmousi_cache[key]


def make_solver(config, *, device="cuda:0", nt_out=801):
    import torch
    b=config["boundaries"]
    boundary=BoundaryConfig(**{k:b[k] for k in ("top","left","right","bottom","npml",
        "cpml_target_reflection","cpml_polynomial_order","cpml_outside_physical_domain")})
    grid=AcousticGrid(nx=401,nz=401,dx_m=5.,dz_m=5.,lx_m=2000.,lz_m=2000.,centering="node")
    return NodalLWC84Solver(grid=grid,boundaries=boundary,dt_s=.000125,
        output_times_s=np.arange(nt_out,dtype=np.float64)*.00125,c_ref_mps=6750.,device=device,
        dtype=torch.float32,kappa_max=b["kappa_max"],minimum_frequency_hz=b["minimum_frequency_hz"],
        output_restriction_factor=2,cuda_graphs=True,block_step_count=10)
