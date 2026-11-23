"""Device-resident propagation and fixed, window-checked pressure directions."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch


@dataclass(frozen=True)
class FixedDirection:
    spatial: torch.Tensor
    temporal: torch.Tensor | None = None

    def rows(self, indices):
        if self.temporal is None:
            return self.spatial.index_select(0, indices)
        values = self.temporal.index_select(0, indices) @ self.spatial.flatten(1)
        return values.reshape(len(indices), *self.spatial.shape[-2:])

    def tensors(self):
        return (self.spatial,) if self.temporal is None else (self.spatial, self.temporal)

    def state_dict(self):
        return {"spatial": self.spatial.cpu(),
                "temporal": None if self.temporal is None else self.temporal.cpu()}


def stable_internal_step(cmax, spacing=10., registered_dt=.0025):
    if not isfinite(cmax) or cmax <= 0.:
        raise ValueError("Propagation requires finite positive nodal speeds")
    subdivisions = 2
    while True:
        dt = registered_dt / subdivisions
        qmax = (4096. / 315.) * (cmax * dt / spacing) ** 2
        if qmax < 12.:
            return dt, subdivisions, qmax
        subdivisions *= 2


def generate_native_proposal(velocity_mps, source, model, cfg, measure):
    from label_quality_fd.data_generation.grid import AcousticGrid, BoundaryConfig
    from label_quality_fd.data_generation.solver_lwc84_fused_device import DeviceResidentFusedLWC84CPMLSolver

    # The device-resident API accepts host nodal velocities and keeps its outputs on device.
    velocity_mps = np.asarray(velocity_mps, dtype=np.float32)
    cmax = float(velocity_mps.max())
    spacing, native_dt = cfg["spacing_m"], cfg["registered_dt_s"]
    dt, subdivisions, qmax = stable_internal_step(cmax, spacing, native_dt)
    block = cfg["block_step_count"]
    if subdivisions % block:
        raise ValueError("The graph block must divide the number of steps per output frame")
    with measure("propagation_prepare_s"):
        solver = DeviceResidentFusedLWC84CPMLSolver(
            grid=AcousticGrid(nx=201, nz=201, dx_m=spacing, dz_m=spacing, centering="node"),
            boundaries=BoundaryConfig(npml=cfg["npml"]), dt_s=dt,
            output_times_s=np.arange(401) * native_dt, c_ref_mps=6750., device=str(model.device),
            dtype=torch.float32, output_restriction_factor=cfg["output_restriction_factor"],
            cuda_graphs=cfg["cuda_graphs"], block_step_count=block)
    x, z, frequency, shift, amplitude = source[0].tolist()
    with measure("propagation_s"):
        result = solver.simulate_device(
            velocity_mps, source_x_m=x, source_z_m=z, source_f0_hz=frequency,
            source_t0_s=shift, source_amplitude=amplitude)
        pa = result.wavefield_device[0]
        if pa.shape != (401, 201, 201) or not torch.isfinite(pa).all() or pa[:, 0].count_nonzero():
            raise ValueError("The independent propagation violates the native pressure contract")
        normalized = model.normalizer.encode_pressure(pa[None], source[:, 4])[0].detach()
    return normalized, {"solver": type(solver).__name__, "dt_s": dt, "registered_dt_s": native_dt,
                        "cmax_mps": cmax, "qmax": qmax,
                        "registered_stencil_qmax": (4096. / 315.) * (cmax * native_dt / spacing)**2,
                        "internal_steps_per_frame": subdivisions, "block_step_count": block,
                        "spacing_m": spacing, "grid": [201, 201], "npml": cfg["npml"],
                        "boundary": "zero top plus three outer CFS-CPML sides",
                        "interface_policy": "original 10 m nodal velocities",
                        "cuda_capture_accounting": "capture inside simulate_device is included in propagation_s"}


@torch.no_grad()
def prepare_fixed_direction(baseline, proposal, *, tolerance=.002, max_rank=128):
    difference = (proposal - baseline).detach()
    flat = difference.flatten(1)
    wide = flat.double()
    values, basis = torch.linalg.eigh(wide @ wide.T)
    values = values.flip(0).clamp_min(0.)
    basis = basis.flip(1)
    width = min(max_rank, len(values))
    windows = {"full_401": slice(None), "early_160": slice(0, 160), "late_241": slice(160, None)}
    proposed_energy = {k: proposal[s].double().square().sum().clamp_min(1e-30) for k, s in windows.items()}
    captured = basis[:, :width].square() * values[:width]
    enough = torch.stack([
        wide[s].square().sum() - captured[s].sum(0).cumsum(0) <= tolerance**2 * proposed_energy[k]
        for k, s in windows.items()
    ]).all(0)
    feasible = enough.nonzero().flatten()
    if len(feasible):
        rank = int(feasible[0]) + 1
        time = basis[:, :rank].to(flat)
        time = time.masked_fill((flat.count_nonzero(dim=1) == 0)[:, None], 0.)
        spatial = (time.T @ flat).reshape(rank, *difference.shape[-2:])
        # The original top row is zero in each snapshot, including after synthesis.
        spatial[:, 0] = 0.
        direction = FixedDirection(spatial, time)
        restored = direction.rows(torch.arange(len(flat), device=flat.device))
        errors = {k: float(((restored[s] - difference[s]).double().square().sum()
                            / proposed_energy[k]).sqrt()) for k, s in windows.items()}
        if max(errors.values()) <= tolerance:
            return direction, {"rank": rank, "relative_to_proposal": errors, "fallback": False}
    return FixedDirection(difference), {"rank": None, "relative_to_proposal": dict.fromkeys(windows, 0.),
                                        "fallback": True}
