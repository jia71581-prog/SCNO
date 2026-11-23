"""Six-column zero-observation calibration of a frozen neural pressure field."""

from copy import deepcopy
from math import isfinite, pi

import torch

from explicit_lwc_residual import lwc_residual_analytic
from source_forcing_moment import SourceForcingMoment, triangular_ricker


SCHEMA = "marmousi_ul_sourced_pressure_fit_v1"
GRID_M, DT_S, HALO_NODES, SOURCE_RADIUS_M = 10., .005, 8, 90.


def _time_axis(times):
    time = torch.as_tensor(times, dtype=torch.float64, device="cpu").detach()
    if time.ndim != 1 or len(time) < 24 or not torch.isfinite(time).all():
        raise ValueError("Expected a finite native 5 ms observation axis")
    exact = torch.arange(len(time), dtype=torch.float64)*DT_S
    if not torch.allclose(time, exact, rtol=0., atol=2.e-7):
        raise ValueError("Fitting and replay must share the zero-based 5 ms axis")
    return exact


def _source_tuple(source):
    values = torch.as_tensor(source, dtype=torch.float64, device="cpu").detach()
    if values.shape != (5,) or not torch.isfinite(values).all() or values.tolist()[1:] != [40., 8., .1875, 1.]:
        raise ValueError("Expected one frozen source (x, 40 m, 8 Hz, 0.1875 s, unit amplitude)")
    return values


def _time_polynomials(time, delay):
    # Scale the delay-centered polynomial without changing its column span.
    scale = max(delay, float(time[-1])-delay)
    relative = (time-delay)/scale
    modes = torch.stack((torch.ones_like(relative), relative, relative.square()))
    return modes


def _validated_centers(time, source, manual):
    delay, f0 = float(source[3]), float(source[2])
    if manual is None:
        pulse = torch.where((time >= delay-1.25/f0) & (time <= delay+1.25/f0))[0]
        pulse = pulse[(pulse >= 2) & (pulse < len(time)-2)]
        # Balance the split on the known forcing alone. A fixed phase on this
        # 5 ms axis can otherwise put nearly all pulse energy in one split.
        forcing_energy = triangular_ricker(time, f0, delay, DT_S).square()
        options = [pulse[offset::5] for offset in range(5)]
        options = [candidate for candidate in options if len(candidate) >= 6]
        if not options:
            raise ValueError("The observation axis needs enough samples of the known source pulse")

        def imbalance(candidate):
            even, odd = (float(forcing_energy[candidate[side::2]].sum()) for side in (0, 1))
            return abs(even-odd)/max(even+odd, 1.e-30)

        pulse = min(options, key=imbalance)
        # All fit/probe five-frame stencils are disjoint. A separate spread
        # samples propagation to the end of the record.
        first_late = int(pulse[-1])+5 if len(pulse) else len(time)
        last = len(time)-3
        if first_late <= last:
            late = torch.arange(first_late, last+1, 5)
            late = late[torch.linspace(0, len(late)-1, min(24, len(late))).round().long()]
        else:
            late = pulse[:0]
        fit = sorted([*pulse[::2].tolist(), *late[::2].tolist()])
        probe = sorted([*pulse[1::2].tolist(), *late[1::2].tolist()])
    else:
        fit, probe = (_frame_ids(manual[key], len(time)) for key in ("fit", "probe"))
        fit, probe = sorted(set(fit)), sorted(set(probe))
    if min(len(fit), len(probe)) < 3 or set(fit).intersection(probe) or min(fit+probe) < 2 or max(fit+probe) >= len(time)-2:
        raise ValueError("Provide separate fit/probe equation centers with a two-frame margin")
    return {"fit": fit, "probe": probe}


def _frame_ids(values, count):
    frames = torch.as_tensor(values, device="cpu")
    if frames.ndim != 1 or len(frames) == 0 or frames.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Pressure queries require a vector of integer frame IDs")
    if (frames < 0).any() or (frames >= count).any():
        raise ValueError("Requested pressure frame is outside the registered axis")
    return frames.tolist()


def _read_pressure(predict_pressure, slots, shape):
    pressure = torch.as_tensor(predict_pressure(slots))
    if pressure.shape != (len(slots), *shape) or not pressure.is_floating_point() or not torch.isfinite(pressure).all():
        raise ValueError("The frozen callback must return finite normalized [time,z,x] pressure")
    return torch.cat((torch.zeros_like(pressure[:, :1]), pressure[:, 1:]), dim=1)


def _sample_basis(predict_pressure, slots, shape, time, modes, frequency_hz):
    end = len(time)-1
    before = [max(0, index-1) if index != end else end-2 for index in slots]
    after = [min(end, index+1) if index != end else end-1 for index in slots]
    queries = sorted(set(slots).union(before, after))
    position = {frame: index for index, frame in enumerate(queries)}
    raw = _read_pressure(predict_pressure, queries, shape)
    p = raw[[position[frame] for frame in slots]].double()
    prior = raw[[position[frame] for frame in before]].double()
    later = raw[[position[frame] for frame in after]].double()
    on_end = torch.as_tensor([index == end for index in slots], device=raw.device)[:, None, None]
    difference = torch.where(on_end, 3.*p-4.*later+prior, later-prior)
    phase = difference/(2.*DT_S*2.*pi*frequency_hz)
    basis = modes[:, slots].to(p)
    keep = torch.as_tensor([index >= 3 for index in slots], device=p.device)
    weights = (basis*keep[None])[:, :, None, None]
    return p, torch.cat((weights*p[None], weights*phase[None]), dim=0), raw.dtype


def _interior_mask(velocity, source):
    nz, nx = velocity.shape
    z = torch.arange(nz, dtype=torch.float64, device=velocity.device)[:, None]
    x = torch.arange(nx, dtype=torch.float64, device=velocity.device)[None]
    xs, zs = source[:2].tolist()
    interior = ((z >= HALO_NODES) & (z < nz-HALO_NODES) & (x >= HALO_NODES) & (x < nx-HALO_NODES)
                & ((x*GRID_M-xs).square()+(z*GRID_M-zs).square() > SOURCE_RADIUS_M**2))
    if not interior.any():
        raise ValueError("There is no valid homogeneous equation region outside source and halo")
    return interior


class _NormalEquations:
    """Accumulate seven-column Grams, never a complete space-time design matrix."""

    def __init__(self):
        self.gram = torch.zeros((7, 7), dtype=torch.float64)
        self.rows = 0
        self.known_source_energy = 0.

    def add(self, columns, residual, known_source=None):
        matrix = torch.cat((columns.reshape(6, -1), residual.reshape(1, -1)), dim=0)
        self.gram += (matrix @ matrix.T).cpu()
        self.rows += matrix.shape[1]
        if known_source is not None:
            self.known_source_energy += float(known_source.square().sum())

    def normalized(self, use_source_energy):
        scale = self.known_source_energy if use_source_energy else self.rows
        if not isfinite(scale) or scale <= 1.e-12 or not torch.isfinite(self.gram).all():
            raise ValueError("Each split needs valid PDE rows and nonzero known source-pulse samples")
        return self.gram/scale


def _local_equations(p, phi, before, center, after, physical_times, velocity, interior, moment, frequency):
    nt, nz, nx = p.shape
    triplets = (before, center, after)
    p_states = tuple(p[which] for which in triplets)
    phi_states = tuple(phi[:, which] for which in triplets)
    wave_rate = 2.*pi*frequency
    p_residual = lwc_residual_analytic(*p_states, velocity, h=GRID_M, delta=DT_S)
    columns = lwc_residual_analytic(*(state.reshape(-1, nz, nx) for state in phi_states), velocity, h=GRID_M, delta=DT_S)
    columns = columns.reshape(6, len(center), nz, nx)
    base_source, known = moment.signature(*p_states, velocity, physical_times)
    # SourceForcingMoment.signature is linear in pressure. Apply it to the
    # six trial fields to obtain their exact source-equation columns.
    source_columns, _ = moment.signature(*(state.reshape(-1, nz, nx) for state in phi_states), velocity, physical_times.repeat(6))
    weak = (source_columns.reshape(6, len(center), -1), base_source-known, known)
    homogeneous = (columns[:, :, interior]/wave_rate**2, p_residual[:, interior]/wave_rate**2)
    return homogeneous, weak


def _accumulate(predict, velocity, time, modes, source, scale, centers, block):
    result = (_NormalEquations(), _NormalEquations())
    shape = tuple(velocity.shape)
    frequency = float(source[2])
    moment = SourceForcingMoment(source, pressure_scale=scale, spacing_m=GRID_M, fine_spacing_m=5.,
                                 observation_dt_s=DT_S, shape_zx=shape)
    for first in range(0, len(centers), block):
        group = centers[first:first+block]
        fold = sorted(set(group).union(frame-1 for frame in group).union(frame+1 for frame in group))
        location = {frame: index for index, frame in enumerate(fold)}
        index = ([location[frame+shift] for frame in group] for shift in (-1, 0, 1))
        p, phi, _ = _sample_basis(predict, fold, shape, time, modes, frequency)
        if first == 0:
            velocity = velocity.to(p)
            moment = moment.to(p.device)
            mask = _interior_mask(velocity, source)
        if velocity.device != p.device:
            raise ValueError("The frozen pressure callback changed device across equation blocks")
        main, weak = _local_equations(p, phi, *index, time[group].to(p), velocity, mask, moment, frequency)
        result[0].add(*main)
        result[1].add(*weak)
    return {"interior": result[0].normalized(False), "source": result[1].normalized(True),
            "rows": {"interior": result[0].rows, "source": result[1].rows}, "source_energy": result[1].known_source_energy}


def _column_ridge(gram, ridge):
    matrix, rhs = gram[:6, :6], -gram[:6, 6]
    energy = matrix.diagonal().clone()
    cutoff = max(1.e-30, float(energy.max())*1.e-13)
    active = energy > cutoff
    result = torch.zeros(6, dtype=torch.float64)
    if active.any():
        divisor = energy[active].sqrt()
        standardized = (matrix[active][:, active]/divisor[:, None])/divisor[None]
        standardized.diagonal().add_(ridge)
        result[active] = torch.linalg.solve(standardized, rhs[active]/divisor)/divisor
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite six-column ridge solution")
    return result, int(active.sum())


def _amplitude_and_delay(coefficients, modes, frequency):
    amplitude = float((coefficients[:3] @ modes).abs().max())
    lag = float((coefficients[3:] @ modes).abs().max()/(2.*pi*frequency))
    return amplitude, lag


def _quadratics(case, coefficients, source_weight):
    vector = torch.cat((coefficients, coefficients.new_ones(1)))
    interior, source = (max(0., float(vector.dot(case[label] @ vector))) for label in ("interior", "source"))
    return {"interior_loss": interior, "source_loss": source, "total": interior+source_weight*source}


def _line_optimum(case, direction, source_weight):
    """Minimize the unregularized sourced equation along the ridge direction."""
    gram = case["interior"]+source_weight*case["source"]
    matrix, mixed = gram[:6, :6], gram[:6, 6]
    curvature = float(direction.dot(matrix @ direction))
    linear = float(direction.dot(mixed))
    if not isfinite(curvature) or not isfinite(linear):
        raise ValueError("Nonfinite polynomial on the sourced ridge direction")
    eps = 64.*torch.finfo(direction.dtype).eps
    magnitude = direction.abs()
    curve_tol = eps*float(magnitude.dot(matrix.abs() @ magnitude))
    linear_tol = eps*float(magnitude.dot(mixed.abs()))
    if curvature < -curve_tol:
        raise ValueError("An equation Gram cannot curve downward in the ridge direction")
    if curvature <= curve_tol:
        if abs(linear) <= linear_tol:
            return None, "flat"
        return (0., "increasing_linear") if linear > 0. else (None, "decreasing_linear")
    if abs(linear) <= linear_tol:
        return 0., "quadratic"
    optimum = -linear/curvature
    if not isfinite(optimum):
        return (0., "increasing_linear") if linear > 0. else (None, "decreasing_linear")
    return optimum, "quadratic"


class SourcedPressureFit:
    """Serializable six-mode readout. Apply it to the same frozen callback."""

    def __init__(self, receipt):
        self.receipt = deepcopy(receipt)
        if receipt["schema"] != SCHEMA:
            raise ValueError("Unexpected sourced readout schema")
        self.time = _time_axis(receipt["times_s"])
        self.source = _source_tuple(receipt["source_x_z_f0_delay_amp"])
        self.modes = _time_polynomials(self.time, float(self.source[3]))
        self.shape = tuple(receipt["pressure_grid_zx"])
        self.coefficients = torch.as_tensor(receipt["coefficients"], dtype=torch.float64)
        if tuple(self.coefficients.shape) != (6,) or not torch.isfinite(self.coefficients).all():
            raise ValueError("The persisted sourced readout must contain six finite coefficients")
        if not receipt["accepted"] and self.coefficients.count_nonzero():
            raise ValueError("A rejected sourced readout must replay the original pressure")
        amplitude, lag = _amplitude_and_delay(self.coefficients, self.modes, float(self.source[2]))
        if amplitude > receipt["amplitude_limit"]+1.e-12 or lag > receipt["delay_limit_s"]+1.e-12:
            raise ValueError("Persisted coefficients exceed the sourced readout limits")

    @classmethod
    def from_receipt(cls, receipt):
        """Rebuild replay from a JSON-compatible receipt and the original callback."""
        return cls(receipt)

    def apply(self, predict_pressure, frame_ids, *, query_block=None):
        requested = _frame_ids(frame_ids, len(self.time))
        block = self.receipt["query_block"] if query_block is None else int(query_block)
        if not 1 <= block <= 64:
            raise ValueError("Replay query blocks must contain 1 to 64 frames")
        slots = sorted(set(requested))
        offsets = {frame: index for index, frame in enumerate(slots)}
        output = []
        for first in range(0, len(slots), block):
            frames = slots[first:first+block]
            if self.receipt["accepted"]:
                pressure, basis, dtype = _sample_basis(predict_pressure, frames, self.shape, self.time, self.modes, float(self.source[2]))
                changed = pressure + torch.einsum("k,ktzx->tzx", self.coefficients.to(pressure), basis)
                output.append(changed.to(dtype))
            else:
                output.append(_read_pressure(predict_pressure, frames, self.shape))
        unique = torch.cat(output)
        return unique[[offsets[frame] for frame in requested]]


@torch.no_grad()
def fit_sourced_pressure(predict_pressure, physical_velocity, source, times, pressure_scale, *, equation_centers=None,
                        query_block=8, ridge=.005, amplitude_limit=.06, delay_limit_s=.0006, source_weight=1.):
    """Fit the local sourced pressure equation with known velocity and source only.

    ``predict_pressure(sorted_unique_frame_ids)`` returns normalized pressure on
    the registered native grid. Its initial-condition startup is the caller's
    frozen pressure operation. This readout leaves its first three frames alone
    and enforces the zero-pressure free surface at every requested frame.
    """
    time, source = _time_axis(times), _source_tuple(source)
    velocity = torch.as_tensor(physical_velocity, dtype=torch.float64).detach()
    if velocity.ndim != 2 or not torch.isfinite(velocity).all() or not (velocity > 0.).all():
        raise ValueError("Physical velocity must be a finite positive [z,x] array in m/s")
    scales = (pressure_scale, ridge, amplitude_limit, delay_limit_s, source_weight)
    if not all(isfinite(float(scale)) and float(scale) > 0. for scale in scales) or not isinstance(query_block, int) or not 1 <= query_block <= 32:
        raise ValueError("Expected positive pressure and fit scales with at most 32 equation centers per block")
    scales = tuple(map(float, scales))
    pressure_scale, ridge, amplitude_limit, delay_limit_s, source_weight = scales
    frequency, delay = map(float, source[2:4])
    modes = _time_polynomials(time, delay)
    split = _validated_centers(time, source, equation_centers)
    cases = {part: _accumulate(predict_pressure, velocity, time, modes, source, pressure_scale, centers, query_block)
             for part, centers in split.items()}
    train = cases["fit"]
    coefficient, rank = _column_ridge(train["interior"]+source_weight*train["source"], ridge)
    measured_amplitude, measured_delay = _amplitude_and_delay(coefficient, modes, frequency)
    factor = min(1., amplitude_limit/max(measured_amplitude, 1.e-30), delay_limit_s/max(measured_delay, 1.e-30))
    nonzero = rank > 0 and bool(coefficient.count_nonzero())
    line = {part: _line_optimum(case, coefficient, source_weight) if nonzero else (None, "null_direction")
            for part, case in cases.items()}
    limits = [value for value, _ in line.values() if value is not None]
    alpha = max(0., min([factor, *limits])) if nonzero else 0.
    proposed = coefficient*alpha
    zero = torch.zeros_like(proposed)
    before = {part: _quadratics(case, zero, source_weight) for part, case in cases.items()}
    candidate = {part: _quadratics(case, proposed, source_weight) for part, case in cases.items()}
    accepted = (alpha > 0. and bool(proposed.count_nonzero())
                and all(candidate[part]["total"] <= before[part]["total"]+1.e-12*max(1., before[part]["total"]) for part in split)
                and candidate["fit"]["total"] < before["fit"]["total"])
    final = proposed if accepted else zero
    amplitude, lag = _amplitude_and_delay(final, modes, frequency)
    fit_support, probe_support = ({center+shift for center in centers for shift in (-2, -1, 0, 1, 2)} for centers in split.values())
    receipt = {"schema": SCHEMA, "times_s": time.tolist(), "pressure_grid_zx": list(velocity.shape),
               "source_x_z_f0_delay_amp": source.tolist(), "pressure_scale": pressure_scale,
               "coefficients": final.tolist(), "candidate_coefficients": proposed.tolist(), "accepted": bool(accepted),
               "query_block": query_block, "relative_column_ridge": ridge, "active_columns": rank, "source_weight": source_weight,
               "coefficient_limit_scale": factor, "continuous_step_scale": alpha,
               "fit_line_optimum": line["fit"][0], "probe_line_optimum": line["probe"][0],
               "line_optimum_status": {part: status for part, (_, status) in line.items()},
               "amplitude_limit": amplitude_limit, "delay_limit_s": delay_limit_s,
               "peak_amplitude_correction": amplitude, "peak_phase_equivalent_delay_s": lag,
               "equation_centers": split, "shared_frozen_query_frames": len(fit_support.intersection(probe_support)),
               "equations": {part: {"row_count": case["rows"], "known_source_energy": case["source_energy"],
                                    "baseline": before[part], "proposed": candidate[part],
                                    "applied": candidate[part] if accepted else before[part]} for part, case in cases.items()},
               "pressure_units": "normalized by frozen TRAIN pressure scale", "physics_scope": "weak forced moments and local interior residual",
               "observed_or_reference_waveforms_used": False, "time_steps_advanced": 0}
    return SourcedPressureFit(receipt)
