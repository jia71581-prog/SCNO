"""Full-field, receiver and fixed orthogonal error diagnostics for SCNO V2."""

from contextlib import contextmanager
import math

import numpy as np
import torch


SELECTION_METRIC = "mean_record_relative_l2_squared"


@contextmanager
def evaluation_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def squared_norm(field):
    return field.double().square().sum().item()


@torch.no_grad()
def orthogonal_errors(model, target, parts):
    base = parts["base_pressure"].detach()
    prediction = parts["prediction"].detach()
    if base.shape != target.shape or prediction.shape != target.shape or target.ndim != 3:
        raise ValueError("validation requires aligned [time,z,x] field and reference arrays")
    if base[:, 0].count_nonzero().item() or prediction[:, 0].count_nonzero().item():
        raise ValueError("both pressure outputs must obey the exact free surface")
    base_error = target.double() - base.double()
    corrected_error = target.double() - prediction.double()
    active_error = base_error[:, None, 1:]
    transform = model.detail_transform
    coarse, packets = transform.analyze(active_error)
    coefficients = parts["detail_coefficients"]
    if len(coefficients) != len(packets):
        raise ValueError("there must be one coefficient array for every detail scale")
    coarse_sse = squared_norm(coarse)
    surface_sse = squared_norm(base_error[:, 0])
    before = []
    after = []
    for level, modes, predicted in zip(transform.levels, packets, coefficients):
        valid = level.valid_detail
        residual_detail = modes[..., 16:].masked_fill(~valid, 0)
        if predicted is None:
            predicted = torch.zeros_like(residual_detail)
        if predicted.ndim == residual_detail.ndim - 1:
            predicted = predicted.unsqueeze(1)
        if predicted.shape != residual_detail.shape or predicted[..., ~valid].count_nonzero().item():
            raise ValueError("incorrect detail shape or nonzero padding coefficients")
        before.append(squared_norm(residual_detail))
        after.append(squared_norm(residual_detail - predicted.detach()))
    es = squared_norm(target)
    base_sse = squared_norm(base_error)
    final_sse = squared_norm(corrected_error)
    gain = sum(before) - sum(after)
    scale = max(es, base_sse, final_sse, 1e-30)
    closure = max(abs(base_sse - coarse_sse - surface_sse - sum(before)),
                  abs(final_sse - coarse_sse - surface_sse - sum(after)),
                  abs((base_sse - final_sse) - gain)) / scale
    if not math.isfinite(closure) or closure > 2e-5:
        raise ArithmeticError(f"orthogonal error decomposition failed: normalized discrepancy {closure:.6g}")
    return {
        "reference_sse": es, "base_field_sse": base_sse, "field_sse": final_sse,
        "detail_before_sse": before, "detail_after_sse": after,
        "oracle_detail_floor_sse": coarse_sse, "reference_surface_sse": surface_sse,
        "correction_gain_sse": gain, "orthogonal_closure_relative": closure,
        "base_output_sse": squared_norm(base), "detail_output_sse": squared_norm(prediction - base),
        "output_elements": target.numel(),
    }


def combine_errors(observations):
    combined = {}
    for key in observations[0]:
        values = [row[key] for row in observations]
        if key in ("detail_before_sse", "detail_after_sse"):
            combined[key] = np.sum(values, axis=0).tolist()
        elif key == "orthogonal_closure_relative":
            combined[key] = max(values)
        else:
            combined[key] = sum(values)
    return combined


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def _relative_l2(numerator, denominator):
    relative_squared = _ratio(numerator, denominator)
    return math.sqrt(relative_squared) if relative_squared is not None else None


def evaluate(model, dataset, record_ids, slots=None, batch_times=8, receiver_ij=((2, 100),)):
    """Evaluate fixed records/query slots, with pooled energies in physical units."""
    record_ids = tuple(record_ids)
    if not record_ids or type(batch_times) is not int or batch_times < 1:
        raise ValueError("validation needs fixed records and a positive query batch")
    axis = np.asarray(dataset.times, dtype=np.float64)
    slots = np.arange(len(axis)) if slots is None else np.asarray(slots, dtype=np.int64)
    if len(slots) == 0 or slots.min() < 0 or slots.max() >= len(axis) or np.any(np.diff(slots) <= 0):
        raise ValueError("validation slots must be unique, sorted and on the native axis")
    receivers = [tuple(position) for position in receiver_ij]
    if any(len(position) != 2 or any(type(index) is not int or index < 0 for index in position)
           for position in receivers):
        raise ValueError("receiver positions must be nonnegative integer [z, x] indices")
    bands = np.empty(len(axis), dtype=np.int64)
    for index, ids in enumerate(np.array_split(np.arange(len(axis)), 3)):
        bands[ids] = index
    device = next(model.parameters()).device
    scale = float(dataset.metadata["pressure_scale_pa"])
    pooled = np.zeros(3)
    sums_time = np.zeros((3, 3))
    sums_recs = np.zeros((len(receivers), 3))
    all_details = []
    per_record = []
    with evaluation_mode(model):
        for record_id in record_ids:
            record_chunks = []
            physical_scale = None
            for start in range(0, len(slots), batch_times):
                interval = slots[start:start + batch_times]
                batch = dataset.read(record_id, interval)
                source_amplitude = float(batch["source"].reshape(-1)[4].item())
                pressure_unit = scale * source_amplitude
                if physical_scale is None:
                    physical_scale = pressure_unit ** 2
                elif physical_scale != pressure_unit ** 2:
                    raise ValueError("a record changed source amplitude between validation batches")
                target = batch["target"].to(device)
                parts = model(batch["velocity"], batch["source"], batch["times"], return_parts=True)
                errors = orthogonal_errors(model, target, parts)
                record_chunks.append(errors)
                pooled += physical_scale * np.array([errors["field_sse"], errors["base_field_sse"],
                                                     errors["reference_sse"]])
                reference = target.double()
                full_error = parts["prediction"].double() - reference
                base_error = parts["base_pressure"].double() - reference
                for band in range(3):
                    members = bands[interval] == band
                    if any(members):
                        mask = torch.from_numpy(members).to(device)
                        sums_time[band] += physical_scale * np.array([
                            squared_norm(value[mask]) for value in (full_error, base_error, reference)])
                for i, (row, column) in enumerate(receivers):
                    if row >= target.shape[1] or column >= target.shape[2]:
                        raise ValueError(f"validation receiver outside grid: {(row, column)}")
                    sums_recs[i] += physical_scale * np.array([
                        squared_norm(value[:, row, column]) for value in (full_error, base_error, reference)])
            errors = combine_errors(record_chunks)
            all_details.append(errors)
            per_record.append({
                "record_id": record_id,
                "relative_l2_squared": _ratio(errors["field_sse"], errors["reference_sse"]),
                "base_relative_l2_squared": _ratio(errors["base_field_sse"], errors["reference_sse"]),
                "reference_energy_normalized_pressure": errors["reference_sse"],
            })
    defined = [record for record in per_record if record["relative_l2_squared"] is not None]
    if not defined or pooled[2] <= 0:
        raise ValueError("validation queries contain no positive reference energy")

    def paired_ratio(sums):
        return {"relative_l2": _relative_l2(sums[0], sums[2]),
                "base_relative_l2": _relative_l2(sums[1], sums[2])}

    return {
        "record_ids": list(record_ids), "time_slots": slots.tolist(), "time_count": len(slots),
        "full_time_axis": len(slots) == len(axis),
        "zero_energy_records_excluded_from_record_mean": len(per_record) - len(defined),
        SELECTION_METRIC: float(np.mean([r["relative_l2_squared"] for r in defined])),
        "mean_record_base_relative_l2_squared": float(np.mean([r["base_relative_l2_squared"] for r in defined])),
        "pooled_relative_l2": _relative_l2(pooled[0], pooled[2]),
        "pooled_base_relative_l2": _relative_l2(pooled[1], pooled[2]),
        "time_bands": {name: paired_ratio(sums_time[i]) for i, name in enumerate(("early", "middle", "late"))},
        "receivers": [{"row": row, "column": column, **paired_ratio(sums_recs[i])}
                      for i, (row, column) in enumerate(receivers)],
        "per_record": per_record, "orthogonal_errors": combine_errors(all_details),
    }
