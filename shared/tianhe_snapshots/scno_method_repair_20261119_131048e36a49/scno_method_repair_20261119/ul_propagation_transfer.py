"""Generate and fit a label-free propagation correction for a frozen SCNO."""

from pathlib import Path
import sys

import numpy as np
import torch


def interface_velocity(velocity):
    """Average slowness squared over a coarse cell at a horizontal step.

    The 5 m nodal convention places the step midway between the fine nodes
    straddling it. The first lower 10 m node then has 1/4 upper-cell volume.
    """
    velocity = np.asarray(velocity, dtype=np.float32)
    if velocity.ndim != 2 or not np.all(velocity == velocity[:, :1]):
        raise ValueError("the current propagation transfer covers uniform or horizontally layered media")
    coarse = velocity.copy()
    transitions = np.flatnonzero(velocity[1:, 0] != velocity[:-1, 0]) + 1
    mixed = []
    for row in transitions:
        upper, lower = float(velocity[row - 1, 0]), float(velocity[row, 0])
        effective = (0.25 / upper**2 + 0.75 / lower**2)**-0.5
        coarse[row] = effective
        mixed.append({"row": int(row), "effective_velocity_mps": float(coarse[row, 0])})
    return coarse, mixed


def gen_pressure(velocity, source, device, solver_source_dir):
    sys.path.insert(0, str(Path(solver_source_dir).resolve()))
    from label_quality_fd.data_generation.grid import AcousticGrid, BoundaryConfig
    from label_quality_fd.data_generation.solver_lwc84_fused import FusedLWC84CPMLSolver

    n_z, n_x = velocity.shape
    solver = FusedLWC84CPMLSolver(
        grid=AcousticGrid(nx=n_x, nz=n_z, dx_m=10., dz_m=10., centering="node"),
        boundaries=BoundaryConfig(npml=20), dt_s=.000125, output_times_s=np.arange(401) * .0025,
        c_ref_mps=6750., device=str(device), dtype=torch.float32, output_restriction_factor=1,
        cuda_graphs=device.type == "cuda", block_step_count=10)
    x, z, f0, t0, amplitude = map(float, source)
    result = solver.simulate(velocity, source_x_m=x, source_z_m=z, source_f0_hz=f0,
                             source_t0_s=t0, source_amplitude=amplitude)
    pressure = torch.as_tensor(result.wavefield[0], device=device)
    if (pressure.shape != (401, n_z, n_x) or not torch.isfinite(pressure).all()
            or pressure[:, 0].count_nonzero() or pressure[0].count_nonzero()):
        raise ValueError("the propagation proposal violates the registered pressure contract")
    return pressure


def delayed_propagation_direction(base, proposal, times, last_observation, frequency):
    """Switch in the propagated correction over one source period."""
    phase = ((times - last_observation) * frequency).clamp(0., 1.)
    # Quintic transition has a continuous second derivative at both ends.
    switch = phase.pow(3) * (10. - 15. * phase + 6. * phase.square())
    return (proposal - base) * switch[:, None, None]


@torch.no_grad()
def compress_pressure_correction(base, correction, tolerance=.002):
    """Choose a time rank using the corrected pressure, with no reference labels."""
    flat = correction.flatten(1)
    flat64 = flat.double()
    gram = flat64 @ flat64.T
    spectrum, vectors = torch.linalg.eigh(gram)
    energy = spectrum.clamp_min(0.).flip(0).cumsum(0)
    target_energy = (base + correction).double().square().sum().clamp_min(1e-30)
    keep_energy = (energy[-1] - tolerance**2 * target_energy).clamp_min(0.)
    rank = min(int(torch.searchsorted(energy, keep_energy)) + 1, len(energy))
    temporal = vectors[:, -rank:].to(flat)
    temporal = temporal.masked_fill((flat.count_nonzero(dim=1) == 0)[:, None], 0.)
    spatial = (temporal.T @ flat).reshape(rank, *correction.shape[1:])
    reconstructed = (temporal @ spatial.flatten(1)).reshape_as(correction)
    error = float(((reconstructed - correction).double().square().sum() / target_energy).sqrt())
    return (temporal, spatial), reconstructed, error


@torch.no_grad()
def line_fit(objective, base, direction):
    def score(pressure, name):
        rows, inverse = objective.requests[name]
        return float(objective.loss_decoded(pressure[rows], inverse, name)[0])

    initial = score(base, "train")
    positive = score(base + direction, "train")
    negative = score(base - direction, "train")
    curvature = (positive + negative) * .5 - initial
    if curvature > 0.:
        weight = min(1., max(0., (negative - positive) / (4 * curvature)))
    else:
        weight = float(positive < initial)
    initial_probe = score(base, "probe")
    for coefficient in (weight, .5 * weight, 0.):
        candidate = base + coefficient * direction
        train = score(candidate, "train")
        probe = score(candidate, "probe")
        if train <= initial and probe <= initial_probe:
            return coefficient, {"train_before": initial, "train_after": train,
                                 "probe_before": initial_probe, "probe_after": probe}
    raise RuntimeError("zero propagation correction did not reproduce its original objective")


@torch.no_grad()
def fit_propagation_pressure(objective, solver_source_dir):
    from finetune_ul_transfer import synchronized_time

    model = objective.model
    if not getattr(model, "_ul_completed_pretraining", False):
        raise ValueError("propagation transfer requires the completed pretrained parent")
    effective, mixed = interface_velocity(objective.velocity.cpu().numpy())
    source = objective.context.source[0]
    started = synchronized_time(model)
    physical_pa = gen_pressure(effective, source.cpu().numpy(), model.device, solver_source_dir)
    simulation_seconds = synchronized_time(model) - started
    proposal = model.normalizer.encode_pressure(physical_pa[None], source[4:5])[0]
    rows = torch.arange(401, device=model.device)
    base = torch.cat([model.decode(objective.context, part, normalized=True)[0] for part in rows.split(32)])
    last_obs = model.full_times_s[objective.observation_indices[-1]]
    direction = delayed_propagation_direction(base, proposal, model.full_times_s, last_obs, source[2])
    objective.velocity = torch.as_tensor(effective, dtype=objective.velocity.dtype, device=model.device)
    objective.set_propagation_defect(proposal)
    coefficient, record = line_fit(objective, base, direction)
    compression = None
    if coefficient:
        correction = coefficient * direction
        factors, reconstructed, error = compress_pressure_correction(base, correction)
        probe_rows, inverse = objective.requests['probe']
        probe = float(objective.loss_decoded((base + reconstructed)[probe_rows], inverse, 'probe')[0])
        accepted = error <= .00201 and probe <= record['probe_before']
        objective.set_extra_pressure(reconstructed if accepted else correction)
        if accepted:
            objective.extra_pressure_factors = factors
        compression = {'rank': len(factors[1]), 'relative_pressure_reconstruction_error': error,
                       'used_factorization': accepted, 'probe_after_reconstruction': probe}
    return {"coefficient": coefficient, "solver_cuda_graphs": model.device.type == "cuda",
            "source_of_proposal": "independent LWC8-4 CPML solve from target velocity and source",
            "native_residual_rhs": "native stencil applied to the independent propagation proposal",
            "propagation_ramp_start_s": float(last_obs),
            "propagation_ramp_end_s": float(last_obs + 1. / source[2]),
            "solver_dt_s": .000125, "solver_spacing_m": 10., "simulation_seconds": simulation_seconds,
            "interface_rows": mixed, "compression": compression, **record}
