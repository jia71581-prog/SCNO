"""Fit the joint SCNO's final time rows to complete TRAIN pressure sequences."""

import math
from time import perf_counter

import torch


def _squared_norm(value):
    return torch.linalg.vector_norm(value, dtype=torch.float64).square()


class _ReducedQR:
    """Keep R, QᵀY, and the squared norm discarded by successive QR reductions."""

    def __init__(self, device):
        self.r = self.projected = None
        self.orthogonal_sse = torch.zeros((), dtype=torch.float64, device=device)

    def add(self, design, residual):
        if self.r is not None:
            design = torch.cat((self.r, design))
            residual = torch.cat((self.projected, residual))
        q, self.r = torch.linalg.qr(design, mode="reduced")
        self.projected = q.T @ residual
        self.orthogonal_sse += _squared_norm(torch.addmm(residual, q, self.projected, alpha=-1.))


def _output_fields(decoder):
    final = decoder.proj[-1]
    return (final.weight[:, :, 0, 0], *(layer.weight for layer in decoder.detail_time), final.bias[:, None])


@torch.no_grad()
def calibrate_time_rows(model, records, *, energy_floor=1e-12, spatial_chunk=2048, qr_dtype=None):
    """Fit normalized pressure on the 200×201 active grid, uniformly over all native times.

    Supply a one-pass iterable of TRAIN (single-source encoded context, normalized
    target) pairs. The target has shape [T,201,201] or [1,T,201,201]. Pass the
    training energy receipt's ``epsilon_E`` when using its corpus-specific floor.
    Each record contributes its squared pressure error divided by its floored
    active-grid energy. Only the base weight/bias and detail time weights change.
    ``qr_dtype`` selects float32 or float64 on the model device for spatial QR.
    The final reduction, anchored ridge solve and error reporting use float64.
    """
    start = perf_counter()
    if not math.isfinite(energy_floor) or energy_floor <= 0:
        raise ValueError("energy_floor must be finite and positive")
    if type(spatial_chunk) is not int or spatial_chunk < 1:
        raise ValueError("spatial_chunk must be a positive integer")
    fields = _output_fields(model.decoder)
    device = fields[0].device
    if qr_dtype is None:
        qr_dtype = torch.float32 if device.type == "cuda" else torch.float64
    if qr_dtype not in (torch.float32, torch.float64):
        raise ValueError("spatial QR requires torch.float32 or torch.float64")
    n_times = len(model.full_times_s)
    if any(field.ndim != 2 or field.shape[0] != n_times or field.device != device for field in fields):
        raise ValueError("all readout fields must share the native time axis and device")
    originals = [field.detach().clone() for field in fields]
    anchor = torch.cat(originals, dim=1).T.contiguous().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(anchor).all():
        raise ValueError("the current affine readout must contain finite parameters")
    device_anchor = anchor.to(device, qr_dtype)
    n_nodes = 200 * 201
    combined = _ReducedQR(torch.device("cpu"))
    count = floored = zero_source = 0
    before = 0.

    for context, target in records:
        features, source = context
        if (features.ndim != 4 or features.shape[0] != 1 or features.shape[-2:] != (201, 201)
                or source.shape != (1, 5) or source.device != device or features.device != device):
            raise ValueError("each record requires one model.encode context on the model's 201×201 grid")
        if not isinstance(target, torch.Tensor) or not target.is_floating_point():
            raise ValueError("each normalized target must be a real, floating-point tensor")
        if target.ndim == 4 and target.shape[0] == 1:
            target = target[0]
        if tuple(target.shape) != (n_times, 201, 201):
            raise ValueError("each calibration record must contain every native pressure frame in order")
        active = target[:, 1:].flatten(1)
        count += 1
        # Normalized inference multiplies its entire readout by zero for a zero-amplitude source.
        if source[0, 4].item() == 0:
            energy = sum(_squared_norm(chunk).item() for chunk in active.split(spatial_chunk, dim=1))
            if not math.isfinite(energy):
                raise ValueError("calibration targets must have finite energy")
            loss = energy / max(energy, energy_floor)
            combined.orthogonal_sse += loss
            before += loss
            floored += int(energy < energy_floor)
            zero_source += 1
            continue

        with torch.autocast(device_type=device.type, enabled=False):
            spatial = model.decoder.prepare_spatial(features)[0][0].flatten(1)
            if spatial.shape != (anchor.shape[0] - 1, n_nodes):
                raise ValueError("the base/detail basis does not match the final time-row parameters")
            gabor_space = model.gabor_spatial(features[:, :, 1:])[0].flatten(1)
            times = model._selected_times(None, 1)
            bank = model.gabor(times, source[:, 3:4], source[:, 2:3], model.duration_s)
            gabor_time = model.gabor_embedding(bank)[0].to(qr_dtype)
            local = _ReducedQR(device)
            anchor_sse = torch.zeros((), device=device, dtype=torch.float64)
            target_energy = torch.zeros_like(anchor_sse)
            for first in range(0, n_nodes, spatial_chunk):
                block = slice(first, min(first + spatial_chunk, n_nodes))
                basis = spatial[:, block].T.to(qr_dtype)
                design = torch.cat((basis, torch.ones_like(basis[:, :1])), dim=1)
                truth = active[:, block].T.to(device=device)
                target_energy += _squared_norm(truth)
                # Residual at the deployed anchor, including the frozen source-dependent Gabor contribution.
                residual = torch.addmm(truth.to(qr_dtype), gabor_space[:, block].T.to(qr_dtype),
                                       gabor_time.T, alpha=-1.)
                residual = torch.addmm(residual, design, device_anchor, alpha=-1.)
                anchor_sse += _squared_norm(residual)
                local.add(design, residual)

        energy = target_energy.item()
        baseline = anchor_sse.item()
        if not math.isfinite(energy) or not math.isfinite(baseline):
            raise ValueError("calibration requires finite active-grid targets, spatial bases, and predictions")
        denominator = max(energy, energy_floor)
        scale = 1. / math.sqrt(denominator)
        floored += int(energy < energy_floor)
        before += baseline / denominator
        combined.orthogonal_sse += local.orthogonal_sse.cpu() / denominator
        combined.add(local.r.to(device="cpu", dtype=torch.float64) * scale,
                     local.projected.to(device="cpu", dtype=torch.float64) * scale)

    if count == 0:
        raise ValueError("at least one TRAIN record is required")
    report = {"records": count, "frames": count * n_times, "frames_per_record": n_times,
              "active_nodes_per_frame": n_nodes, "observations": count * n_times * n_nodes,
              "energy_floor": float(energy_floor), "floored_records": floored, "zero_source_records": zero_source,
              "spatial_qr_dtype": str(qr_dtype), "before": before / count, "after": before / count,
              "accepted": False, "ridge_mu": None}
    if combined.r is not None:
        r, projected = combined.r, combined.projected
        n_columns = anchor.shape[0]
        mu = max(1e-12, 1e-4 * _squared_norm(r).item() / n_columns)
        identity = torch.eye(n_columns, dtype=r.dtype) * math.sqrt(mu)
        q, upper = torch.linalg.qr(torch.cat((r, identity)), mode="reduced")
        # For differences from the old readout, the ridge anchor is exactly zero.
        rhs = q[:r.shape[0]].T @ projected
        delta = torch.linalg.solve_triangular(upper, rhs, upper=True)
        proposed = (anchor + delta).T.split([field.shape[1] for field in fields], dim=1)
        candidates = [part.to(field) for part, field in zip(proposed, fields, strict=True)]
        deployed = torch.cat([part.to(device="cpu", dtype=torch.float64) for part in candidates], dim=1).T
        after = (combined.orthogonal_sse + _squared_norm(r @ (deployed - anchor) - projected)).item() / count
        if not math.isfinite(after) or not torch.isfinite(deployed).all():
            raise FloatingPointError("joint time-row calibration produced a non-finite candidate")
        report.update(ridge_mu=mu, candidate_after=after)
        if after <= report["before"]:
            try:
                for field, candidate in zip(fields, candidates, strict=True):
                    field.copy_(candidate)
            except BaseException:
                for field, original in zip(fields, originals, strict=True):
                    field.copy_(original)
                raise
            report.update(accepted=True, after=after)
    report["elapsed_s"] = perf_counter() - start
    return report
