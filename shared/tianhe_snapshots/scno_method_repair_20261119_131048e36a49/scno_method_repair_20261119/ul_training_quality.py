"""Complete-axis uniform/layered diagnostics; research goals never stop training.

All field metrics use every 0--1 s native frame and every 201x201 node.
Receiver geometry is declared before scoring, with no prediction-driven
selection. Family pooling restores physical pressure units before summation.
"""

from collections import defaultdict
import math

import numpy as np
import torch


def _quantiles(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    names, levels = ("min", "q25", "median", "q75", "p90", "max"), (0., .25, .5, .75, .9, 1.)
    return {name: float(value) for name, value in zip(names, np.quantile(values, levels))} if values else {name: None for name in names}


def _field_moments(prediction, target):
    moments = torch.zeros(4, device=prediction.device, dtype=torch.float64)
    for begin in range(0, 401, 32):
        p = prediction[begin:begin + 32].double()
        y = target[begin:begin + 32].to(device=p.device, dtype=torch.float64)
        moments += torch.stack(((p - y).square().sum(), y.square().sum(), p.square().sum(), (p * y).sum()))
    values = moments.cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite pressure or squared field energy")
    return values


def fixed_receiver_metrics(predicted, reference, *, f0_hz, reference_field_energy,
                           dt_s=.0025, relative_trace_energy_floor=1e-12):
    """Full-trace statistics; phase uses a prespecified source-frequency band.

    Phase RMS is reference-spectrum-energy weighted in [0.5*f0, 1.5*f0].
    Reference-supported bins are selected independently of predictions. A
    missing predicted bin makes phase undefined, rather than omitting its error.
    """
    p = torch.as_tensor(predicted).detach().double().cpu().numpy()
    y = torch.as_tensor(reference).detach().double().cpu().numpy()
    result = {"relative_l2_percent": None, "signed_pearson": None, "peak_amplitude_ratio": None,
              "phase_rms_deg": None, "phase_status": "undefined",
              "phase_band_hz": [.5 * f0_hz, 1.5 * f0_hz], "frames": len(y)}
    if p.shape != (401,) or y.shape != (401,) or not np.isfinite(p).all() or not np.isfinite(y).all():
        return {**result, "status": "undefined_nonfinite_or_incomplete_trace"}
    reference_energy, prediction_energy = float(y @ y), float(p @ p)
    floor = max(1e-300, relative_trace_energy_floor * max(reference_field_energy, 0.))
    result.update(reference_trace_energy=reference_energy, prediction_trace_energy=prediction_energy,
                  reference_energy_floor=floor)
    if reference_energy <= floor:
        return {**result, "status": "undefined_low_energy_reference", "phase_status": "undefined_low_energy_reference"}
    result["relative_l2_percent"] = 100 * math.sqrt(float((p - y) @ (p - y)) / reference_energy)
    result["peak_amplitude_ratio"] = float(np.max(np.abs(p)) / np.max(np.abs(y)))
    pc, yc = p - p.mean(), y - y.mean()
    pe, ye = float(pc @ pc), float(yc @ yc)
    if ye <= floor:
        return {**result, "status": "undefined_static_reference", "phase_status": "undefined_static_reference"}
    if pe > floor:
        result["signed_pearson"] = float(np.clip(float(pc @ yc) / math.sqrt(pe * ye), -1., 1.))
    result["status"] = "measured" if pe > floor else "prediction_low_energy_or_static"
    frequency = np.fft.rfftfreq(401, dt_s)
    spectrum_y, spectrum_p = np.fft.rfft(y), np.fft.rfft(p)
    reference_power = np.abs(spectrum_y) ** 2
    band = (frequency >= .5 * f0_hz) & (frequency <= 1.5 * f0_hz)
    if not band.any() or float(reference_power[band].sum()) <= floor:
        result["phase_status"] = "undefined_low_energy_reference_band"
        return result
    band_peak = float(reference_power[band].max())
    supported = band & (reference_power >= band_peak * 1e-10) & (reference_power > 0)
    missing = np.abs(spectrum_p[supported]) ** 2 <= band_peak * 1e-12
    weights = reference_power[supported]
    result["phase_reference_supported_bins"] = int(supported.sum())
    result["phase_predicted_reference_energy_coverage"] = float(weights[~missing].sum() / weights.sum())
    if missing.any():
        result["phase_status"] = "undefined_prediction_missing_reference_supported_frequency"
        return result
    difference = np.angle(spectrum_p[supported] * np.conj(spectrum_y[supported]))
    result["phase_rms_deg"] = float(np.sqrt(np.dot(weights, difference ** 2) / weights.sum()) * 180 / math.pi)
    result["phase_status"] = "measured"
    return result


def ul_quality_alerts(current, history=(), *, stagnation_window=5, regression_relative=.05,
                      stagnation_relative=.005):
    """Return observations only; there is no early-stop or epoch-control action."""
    alerts = []
    compatible = [entry for entry in history if entry.get("record_ids") == current.get("record_ids")
                  and entry.get("receiver_ij") == current.get("receiver_ij")]
    if len(compatible) != len(history):
        alerts.append({"kind": "history_binding_changed", "message": "Trend comparisons use the same declared records and receiver geometry."})
    for family, summary in current.get("by_family", {}).items():
        value = summary["pooled_relative_l2_percent"]
        if value is None:
            continue
        previous = [entry.get("by_family", {}).get(family, {}).get("pooled_relative_l2_percent") for entry in compatible]
        previous = [float(x) for x in previous if x is not None and math.isfinite(float(x))]
        if previous and value > previous[-1] * (1 + regression_relative) + 1e-10:
            alerts.append({"kind": "validation_error_increased", "family": family,
                           "previous_percent": previous[-1], "current_percent": value})
        if len(previous) >= stagnation_window:
            window = previous[-stagnation_window:] + [value]
            improvement = (window[0] - min(window[1:])) / max(abs(window[0]), 1e-12)
            if improvement < stagnation_relative:
                alerts.append({"kind": "validation_error_stagnated", "family": family,
                               "evaluations": len(window), "best_relative_improvement": improvement})
    return alerts


@torch.inference_mode()
def evaluate_ul_quality(model, records, receiver_ij=((20, 60), (20, 100), (20, 140)),
                        targets=None, *, history=(), epoch=None):
    """Score complete UL trajectories and report research goals without stopping.

    Records are CPU/GPU row dictionaries with id/family/velocity/source/target.
    Targets default to uniform=2%, layered=3%; these are goals, not measured
    guarantees or stopping thresholds. Per-record scores are scale invariant;
    family pooling uses (normalizer.pressure_scale_pa * source_amplitude)^2.
    """
    targets = {"uniform": 2., "layered": 3.} if targets is None else dict(targets)
    if any(not math.isfinite(float(value)) or value <= 0 for value in targets.values()):
        raise ValueError("research goals must be positive finite percentages")
    receivers = tuple(tuple(node) for node in receiver_ij)
    if (not receivers or len(set(receivers)) != len(receivers)
            or any(len(node) != 2 or any(type(i) is not int or not 0 <= i < 201 for i in node) for node in receivers)):
        raise ValueError("declare distinct fixed receiver (row,column) indices on the 201-grid")
    network = getattr(model, "module", model)
    scale_pa = float(network.normalizer.metadata.pressure_scale_pa)
    if not math.isfinite(scale_pa) or scale_pa <= 0:
        raise ValueError("physical pooling needs the model's actual pressure normalization scale")
    device = network.device
    records = list(records)
    record_ids = sorted(str(row["id"]) for row in records)
    if not records or len(set(record_ids)) != len(records):
        raise ValueError("evaluate a nonempty roster of distinct records")
    report = {"epoch": epoch, "frames": 401, "dt_s": .0025, "time_interval_s": [0., 1.],
              "spatial_shape": [201, 201], "record_ids": record_ids,
              "receiver_ij": [list(node) for node in receivers], "per_record": {}, "by_family": {}, "alerts": [],
              "target_interpretation": "research_goals_only; diagnostics_do_not_stop_the_200_epoch_run",
              "phase_definition": "reference-spectrum-energy weighted wrapped phase in [0.5*f0,1.5*f0]",
              "pooling": "sum physical-pressure SSE / sum physical-pressure reference energy"}
    full_times = getattr(network, "full_times_s", None)
    if full_times is not None:
        axis = full_times.detach().double().cpu()
        expected = torch.arange(401, dtype=torch.float64) * .0025
        if axis.shape != expected.shape or not torch.allclose(axis, expected, atol=1e-6, rtol=0):
            report["alerts"].append({"kind": "native_axis_mismatch", "message": "UL quality requires all 401 native frames from 0 to 1 s."})
            return report
    was_training = model.training
    training_modes = {name: module.training for name, module in model.named_modules()}
    family_rows = defaultdict(list)
    try:
        model.eval()
        for row in records:
            key = str(row["id"])
            family = {"uniform": "uniform", "homogeneous": "uniform", "layered": "layered"}.get(row["family"])
            if family is None:
                raise ValueError(f"record {key} is not a declared uniform/layered family")
            result = {"family": family, "frames": 401, "status": "unscored", "relative_l2_percent": None,
                      "physical_sse_pa2": None, "physical_reference_energy_pa2": None, "receivers": []}
            report["per_record"][key] = result
            family_rows[family].append(result)
            try:
                reference = row["target"]
                source = row["source"].detach().double().cpu().reshape(-1)
                if reference.shape != (401, 201, 201) or source.shape != (5,) or not bool(torch.isfinite(source).all()) or source[2] <= 0:
                    raise ValueError("record needs its full 401x201x201 target and physical five-parameter source")
                prediction = model(row["velocity"].to(device), row["source"].to(device), normalized=True)
                if prediction.shape != (1, 401, 201, 201):
                    raise ValueError("model did not predict the complete registered pressure sequence")
                prediction = prediction[0]
                sse, energy, prediction_energy, dot = _field_moments(prediction, reference)
                physical_scale = scale_pa * float(source[4])
                physical_sse, physical_energy = sse * physical_scale ** 2, energy * physical_scale ** 2
                if not math.isfinite(physical_sse) or not math.isfinite(physical_energy):
                    raise ValueError("physical energy conversion is nonfinite")
                result.update(status="measured" if energy > 0 else "undefined_zero_reference_energy",
                    relative_l2_percent=100 * math.sqrt(sse / energy) if energy > 0 else None,
                    normalized_sse=sse, normalized_reference_energy=energy,
                    physical_sse_pa2=physical_sse, physical_reference_energy_pa2=physical_energy,
                    pressure_factor_pa=physical_scale,
                    spacetime_cosine=float(np.clip(dot / math.sqrt(energy * prediction_energy), -1., 1.)) if energy * prediction_energy > 0 else None,
                    top_max_abs=float(prediction[:, 0].abs().max()))
                if result["top_max_abs"] != 0:
                    report["alerts"].append({"kind": "nonzero_free_surface", "record_id": key, "top_max_abs": result["top_max_abs"]})
                for iz, ix in receivers:
                    trace = fixed_receiver_metrics(prediction[:, iz, ix], reference[:, iz, ix],
                                                   f0_hz=float(source[2]), reference_field_energy=energy)
                    result["receivers"].append({"receiver_ij": [iz, ix], "position_xz_m": [10 * ix, 10 * iz], **trace})
            except (ValueError, RuntimeError, OverflowError) as error:
                result.update(status="invalid", reason=str(error))
                report["alerts"].append({"kind": "invalid_prediction_or_reference", "record_id": key, "reason": str(error)})
        for family, rows in family_rows.items():
            valid = [row for row in rows if row["physical_sse_pa2"] is not None and row["status"] != "invalid"]
            numerator = sum(row["physical_sse_pa2"] for row in valid)
            denominator = sum(row["physical_reference_energy_pa2"] for row in valid)
            available_pooled = 100 * math.sqrt(numerator / denominator) if denominator > 0 else None
            complete_roster = len(valid) == len(rows)
            pooled = available_pooled if complete_roster else None
            target = targets.get(family)
            summary = {"records": len(rows), "scored_records": len(valid),
                "complete_roster": complete_roster,
                "pooled_relative_l2_percent": pooled, "physical_sse_pa2": numerator,
                "physical_reference_energy_pa2": denominator,
                "partial_available_pooled_relative_l2_percent": available_pooled if not complete_roster else None,
                "per_record_l2_percent_quantiles": _quantiles(row["relative_l2_percent"] for row in valid),
                "research_target_relative_l2_percent": target,
                "research_target_met": bool(pooled <= target) if target is not None and pooled is not None and len(valid) == len(rows) else None,
                "receiver_summary": []}
            for index, node in enumerate(receivers):
                traces = [row["receivers"][index] for row in valid if len(row["receivers"]) == len(receivers)]
                summary["receiver_summary"].append({"receiver_ij": list(node), "records": len(traces),
                    "pearson_quantiles": _quantiles(trace["signed_pearson"] for trace in traces),
                    "peak_amplitude_ratio_quantiles": _quantiles(trace["peak_amplitude_ratio"] for trace in traces),
                    "phase_rms_deg_quantiles": _quantiles(trace["phase_rms_deg"] for trace in traces),
                    "undefined_correlation_records": sum(trace["signed_pearson"] is None for trace in traces),
                    "undefined_phase_records": sum(trace["phase_rms_deg"] is None for trace in traces)})
            report["by_family"][family] = summary
        report["alerts"].extend(ul_quality_alerts(report, history))
        return report
    finally:
        model.train(was_training)
        # In stage B, frozen BASE BatchNorm/dropout can remain in eval mode
        # while the root model trains. Preserve those per-module modes too.
        for name, module in model.named_modules():
            module.training = training_modes[name]
