"""Physics-constrained transfer of a small pretrained coarse pressure basis."""

import math
from typing import NamedTuple

import torch
from torch.nn import functional as F


class CoarseModes(NamedTuple):
    time: torch.Tensor
    space: torch.Tensor


@torch.no_grad()
def pretrained_coarse_modes(model, context, rank=6):
    """Project pretrained BASE spatial SVD directions onto the coarsest space."""
    decoder = model.parent.decoder
    readout = decoder.proj[-1].weight.detach()
    if readout.ndim != 4 or tuple(readout.shape[-2:]) != (1, 1):
        raise ValueError('coarse transfer requires the registered pointwise BASE pressure readout')
    if not 1 <= rank <= min(readout.shape[:2]):
        raise ValueError('transfer rank exceeds the pretrained readout dimensions')
    (features, _) = decoder.prepare_spatial(context.features, detail_enabled=False)
    if features.shape[0] != 1:
        raise ValueError('derive a separate set of transfer modes for each instance')
    u, _, vh = torch.linalg.svd(readout[:, :, 0, 0].cpu().double(), full_matrices=False)
    spatial = torch.einsum('rh,hzx->rzx', vh[:rank].to(features), features[0])
    transform = model.detail_transform
    coarse, packets = transform.analyze(spatial[:, None])
    inactive = [torch.zeros_like(packet[..., 16:]) for packet in packets]
    spatial = F.pad(transform.synthesize(coarse, inactive)[:, 0], (0, 0, 1, 0))
    norms = torch.linalg.vector_norm(spatial.flatten(1).double(), dim=1)
    if (norms <= 1e-12).any() or not torch.isfinite(norms).all():
        raise ValueError('pretrained coarse spatial modes are degenerate')
    spatial = spatial / norms.to(spatial)[:, None, None]
    return CoarseModes(u[:, :rank].to(features), spatial)


class _QRAccumulator:
    def __init__(self, width, device):
        self.factor = torch.empty((0, width + 1), dtype=torch.float64, device=device)

    def add(self, design, target, weight):
        if weight == 0:
            return
        rows = torch.cat((design, target.reshape(-1, 1)), dim=1) * math.sqrt(weight)
        _, self.factor = torch.linalg.qr(torch.cat((self.factor, rows)), mode='r')

    def solve(self, ridge_relative):
        design, target = self.factor[:, :-1], self.factor[:, -1:]
        width = design.shape[1]
        ridge = max(1e-12, ridge_relative * design.square().sum().item() / width)
        identity = torch.eye(width, dtype=design.dtype, device=design.device)
        q, r = torch.linalg.qr(torch.cat((design, math.sqrt(ridge) * identity)))
        rhs = torch.cat((target, torch.zeros_like(identity[:, :1])))
        coeff = torch.linalg.solve_triangular(r, q.T @ rhs, upper=True).flatten()
        return coeff, ridge


@torch.no_grad()
def solve_coarse_physics_transfer(
        baseline, modes, velocity, source, observations, observation_indices,
        train_centers, train_weights, probe_centers, probe_weights, operators, objective,
        *, baseline_indices=None, delta=.0025, h=10., ridge_relative=1e-4):
    """Fit T A S using early observations, zero initial data and linear LWC residuals.

    All pressures are in the inherited normalized units. `baseline` contains
    predictions on either all native rows or the indicated subset. The only
    truth fields are `observations`. Residual probes contain disjoint time
    centers and no target pressure labels.
    """
    device = baseline.device
    pressure = baseline.double()
    time, space = (value.to(device=device, dtype=torch.float64) for value in modes)
    velocity = torch.as_tensor(velocity, device=device, dtype=torch.float64)
    source = torch.as_tensor(source, device=device, dtype=torch.float64).flatten()
    observed = torch.as_tensor(observations, device=device, dtype=torch.float64)
    observed_at = torch.as_tensor(observation_indices, device=device, dtype=torch.long)
    nt = time.shape[0]
    count, nz, nx = pressure.shape
    if (space.shape[1:] != (nz, nx) or source.shape != (5,)
            or velocity.shape != (nz, nx) or observed_at.shape != (2,) or observed.shape != (2, nz, nx)
            or not 0 <= observed_at[0] < observed_at[1] < nt or source[2] <= 0 or nt < 3):
        raise ValueError('inconsistent registered pressure, mode, source, or observation dimensions')
    if any(not torch.isfinite(tensor).all() for tensor in (pressure, time, space, velocity, source, observed)):
        raise ValueError('nonfinite coarse physics inputs')
    if torch.count_nonzero(pressure[:, 0]) or torch.count_nonzero(space[:, 0]):
        raise ValueError('baseline and coarse modes must satisfy the exact zero surface')
    weights = objective['weights']
    if any(not math.isfinite(value) or value < 0 for value in weights.values()) or ridge_relative <= 0:
        raise ValueError('invalid physics weights or ridge')
    pde_enabled = weights['lwc'] > 0
    train = [int(index) for index in train_centers]
    probe = [int(index) for index in probe_centers]
    if not train or not probe or set(train) & set(probe) or any(not 0 < i < nt - 1 for i in train + probe):
        raise ValueError('use nonempty disjoint native residual-center sets with adjacent rows')
    rows = (torch.arange(nt, device=device) if baseline_indices is None
            else torch.as_tensor(baseline_indices, device=device, dtype=torch.long))
    if rows.shape != (count,) or (rows < 0).any() or (rows >= nt).any() or rows.unique().numel() != count:
        raise ValueError('baseline rows must give one distinct native index per supplied prediction')
    index_of = torch.full((nt,), -1, dtype=torch.long, device=device)
    index_of[rows] = torch.arange(count, device=device)
    initial_rows = index_of[:3]
    observed_rows = index_of[observed_at]
    time_centers = torch.as_tensor(train + probe, device=device, dtype=torch.long)
    stencil = index_of[time_centers[:, None] + time_centers.new_tensor([-1, 0, 1])]
    if (initial_rows < 0).any() or (observed_rows < 0).any() or (pde_enabled and (stencil < 0).any()):
        raise ValueError('predictions must cover every requested observation, initial and residual row')
    train_weights = [float(value) for value in train_weights]
    probe_weights = [float(value) for value in probe_weights]
    for centers, time_weights in ((train, train_weights), (probe, probe_weights)):
        if len(centers) != len(time_weights) or any(not math.isfinite(value) or value <= 0 for value in time_weights):
            raise ValueError('each residual center requires a finite positive weight')
        if not math.isclose(sum(time_weights), 1., rel_tol=1e-6):
            raise ValueError('residual time weights must sum to one')

    omega = 2 * math.pi * float(source[2])
    obs_scale = weights['obs'] / max(float(observed.square().sum()), objective['observation_energy_floor'])
    ic_scale = weights['ic'] / (nz * nx)
    spatial = space.flatten(1)
    width = time.shape[1] * space.shape[0]

    def design(time_row, spatial_rows):
        return torch.einsum('i,jp->pij', time_row, spatial_rows).reshape(-1, width)

    derivative_weights = time.new_tensor([-3., 4., -1.]) / (2 * delta * omega)

    def residual(field, selected):
        return operators.lwc_residual(*field[selected.T].unbind(0), velocity, h=h, delta=delta)[:, mask]

    qr = _QRAccumulator(width, device)
    for observed_time, row, reference in zip(observed_at, observed_rows, observed):
        qr.add(design(time[observed_time], spatial), reference - pressure[row], obs_scale)
    qr.add(design(time[0], spatial), -pressure[initial_rows[0]], ic_scale)
    derivative = derivative_weights @ pressure[initial_rows].flatten(1)
    qr.add(design(derivative_weights @ time[:3], spatial), -derivative, ic_scale)
    if pde_enabled:
        mask = operators.make_valid_mask((nz, nx), h=h, source=(float(source[0]), float(source[1])), device=device)
        variance = float(operators.residual_scale_squared(observed, float(source[2]), objective['pde_energy_floor']))
        pde_scale = weights['lwc'] / (int(mask.count_nonzero()) * variance)
        masked_spatial = space[:, mask]
        first = velocity.square() * operators.laplacian8(space, h=h)
        second = velocity.square() * operators.laplacian8(first, h=h)
        masked_operator = (first + delta**2 / 12 * second)[:, mask]
        train_residual = residual(pressure, stencil[:len(train)])
        for index, weight, target in zip(train, train_weights, train_residual):
            dtt = (time[index + 1] - 2 * time[index] + time[index - 1]) / delta**2
            block = design(dtt, masked_spatial) - design(time[index], masked_operator)
            qr.add(block, -target, weight * pde_scale)
    coefficients, ridge = qr.solve(ridge_relative)
    scaled_space = coefficients.reshape(time.shape[1], space.shape[0]) @ spatial
    correction_on_rows = (time[rows] @ scaled_space).reshape_as(pressure)
    probe_time_weights = time.new_tensor(probe_weights)

    def probe_objective(field):
        observation_error = obs_scale * float((field[observed_rows] - observed).square().sum())
        initial_field = field[initial_rows].flatten(1)
        initial_error = ic_scale * float(initial_field[0].square().sum() + (derivative_weights @ initial_field).square().sum())
        residual_error = (pde_scale * float(probe_time_weights @ residual(field, stencil[len(train):]).square().sum(1))
                          if pde_enabled else 0.)
        return observation_error + initial_error + residual_error

    probe_before = probe_objective(pressure)
    fit_before = float(qr.factor[:, -1].square().sum())
    fit_after = fit_before
    probe_after = probe_before
    chosen = 0.
    for scale in (1., .5, .25, .125):
        trial = (correction_on_rows * scale).to(baseline)
        candidate_probe = probe_objective((baseline + trial).double())
        q_error = scale * (qr.factor[:, :-1] @ coefficients) - qr.factor[:, -1]
        candidate_fit = float(q_error.square().sum())
        if math.isfinite(candidate_probe) and candidate_probe <= probe_before and candidate_fit <= fit_before:
            chosen, probe_after, fit_after = scale, candidate_probe, candidate_fit
            break
    correction = ((time @ scaled_space) * chosen).to(baseline).reshape(nt, nz, nx)
    return correction, {
        'coarse_parameters': width, 'accepted_scale': chosen, 'ridge': ridge,
        'fit_objective_before': fit_before, 'fit_objective_after': fit_after,
        'physics_probe_objective_before': probe_before, 'physics_probe_objective_after': probe_after,
        'calibration_training_centers': train, 'calibration_probe_centers': probe,
        'baseline_pressure_rows_decoded': count,
    }
