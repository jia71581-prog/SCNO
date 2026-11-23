"""Train-only, slot-masked affine pressure-detail calibration."""

from collections import defaultdict
import math
from numbers import Integral
from time import perf_counter

import numpy as np
import torch


def _slot_groups(mask):
    flat = mask.detach().cpu().reshape(-1, mask.shape[-1])
    patterns = defaultdict(list)
    for i, rows in enumerate(flat.T):
        if rows.any():
            patterns[rows.numpy().tobytes()].append(i)
    return [(torch.as_tensor(slots), flat[:, slots[0]]) for slots in patterns.values()]


class ReducedAffineFit:
    """Streaming QR, retaining only a triangular system and projected targets."""

    def __init__(self):
        self.triangular = self.projected = None
        self.rows = 0
        self.trace = 0.
        self.residual_constant = 0.

    def add(self, design, targets):
        self.rows += len(design)
        self.trace += design.square().sum().item()
        if self.triangular is not None:
            design = torch.cat((self.triangular, design))
            targets = torch.cat((self.projected, targets))
        q, self.triangular = torch.linalg.qr(design, mode="reduced")
        self.projected = q.T @ targets
        constant = targets.square().sum().item() - self.projected.square().sum().item()
        self.residual_constant += max(0., constant)

    def solve(self, prior):
        if self.triangular is None or self.projected is None:
            raise ValueError("calibration slot group has no observations")
        nvars = prior.shape[0]
        mu = max(1e-12, 1e-4 * self.trace / nvars)
        sigma = math.sqrt(mu)
        augmented = torch.cat((self.triangular, sigma * torch.eye(nvars, dtype=prior.dtype)))
        rhs = torch.cat((self.projected, sigma * prior))
        solution = torch.linalg.lstsq(augmented, rhs, driver="gels").solution
        baseline = (self.triangular @ prior - self.projected).square().sum().item()
        new_error = (self.triangular @ solution - self.projected).square().sum().item()
        if not torch.isfinite(solution).all() or new_error > baseline + 1e-10 * max(baseline, 1e-20):
            raise FloatingPointError("fixed-feature calibration failed its weighted training objective check")
        return solution, mu, baseline + self.residual_constant, new_error + self.residual_constant


@torch.no_grad()
def calibrate_detail_affines(model, dataset, energy, config, evaluate_validation=None, *, new_output_layers=None):
    """Fit output affines on TRAIN with fixed features and a shared pressure objective."""
    started = perf_counter()
    initialize = new_output_layers is not None
    layers = model.detail_output_layers
    transform = model.detail_transform
    if initialize:
        indices = tuple(new_output_layers)
        if (not indices or any(isinstance(i, bool) or not isinstance(i, Integral) for i in indices)
                or len(set(indices)) != len(indices) or any(i < 0 or i >= len(layers) for i in indices)):
            raise ValueError("B initialization requires distinct indices of new output affines")
        selected = sorted(map(int, indices))
        n_records = min(32, len(dataset.train_ids))
    else:
        if not callable(evaluate_validation):
            raise ValueError("stage-end calibration requires its fixed validation callback")
        selected = list(range(len(layers)))
        n_records = min(config.get("affine_calibration_records", 32), 32, len(dataset.train_ids))
    bands = np.array_split(np.arange(len(dataset.times)), 4)
    queries_per_band = 2
    n_times = len(bands) * queries_per_band
    if not 0 < n_records or any(len(band) < queries_per_band for band in bands):
        raise ValueError("calibration requires TRAIN records and enough frames in each complete-axis stratum")
    seed = config.get("affine_calibration_seed", config["seed"] + 331)
    rng = np.random.default_rng(seed)
    ids = [dataset.train_ids[i] for i in rng.choice(len(dataset.train_ids), n_records, replace=False)]
    per_time_sizes = torch.tensor([len(band) / queries_per_band for band in bands], dtype=torch.float64)
    per_time_sizes = per_time_sizes.repeat_interleave(queries_per_band)
    groups = {i: [(slots, valid, ReducedAffineFit())
                  for slots, valid in _slot_groups(transform.levels[i].valid_detail)] for i in selected}
    saved = {i: (layers[i].weight.detach().clone(), layers[i].bias.detach().clone()) for i in selected}
    phase = getattr(model, "phase", None)
    modes = [(module, module.training) for module in model.modules()]
    selections = []
    timings = {"validation_s": 0.}

    def restore():
        for i, (weight, bias) in saved.items():
            layers[i].weight.copy_(weight)
            layers[i].bias.copy_(bias)

    def observe(record, slots):
        example = dataset.read(record, slots)
        result = model(example["velocity"], example["source"], example["times"],
                       detail_enabled=True, return_parts=True)
        total, floor = float(energy["energies"][record]), float(energy["epsilon_E"])
        if not (math.isfinite(total) and total >= 0 and math.isfinite(floor) and floor > 0):
            raise ValueError("calibration requires a finite positive floored TRAIN energy")
        weights = per_time_sizes / (max(total, floor) * n_records)
        return example, result, weights

    model.eval()
    try:
        before_validation = after_validation = None
        if initialize:
            # Constructor placeholders are unrelated to the pressure baseline or ridge prior.
            for i in selected:
                layers[i].weight.zero_()
                layers[i].bias.zero_()
        else:
            tick = perf_counter()
            before_validation = evaluate_validation()
            timings["validation_s"] = perf_counter() - tick

        tick = perf_counter()
        before = _ErrorTotals(len(layers))
        for record in ids:
            slots = np.concatenate([np.sort(rng.choice(band, queries_per_band, replace=False))
                                    for band in bands])
            example, result, weights = observe(record, slots)
            wanted = before.add(transform, result, example["target"], weights)
            selections.append({"record_id": record, "slots": slots.tolist(),
                               "times_s": np.asarray(dataset.times)[slots].tolist()})
            # SCNO bypasses every output affine for zero source amplitude.
            if example["source"].reshape(-1)[4].item() == 0:
                continue
            sqrt_weights = weights.sqrt().reshape(-1, 1, 1)
            for level in selected:
                features = result["detail_last_hidden_features"][level]
                design = features.detach().cpu().double().reshape(n_times, -1, features.shape[-1])
                design = torch.cat((design, torch.ones_like(design[..., :1])), dim=-1)
                detail = wanted[level].detach().cpu().double().reshape(n_times, -1, wanted[level].shape[-1])
                for slot_ids, valid, fit in groups[level]:
                    fit.add((design[:, valid] * sqrt_weights).flatten(0, 1),
                            (detail[:, valid].index_select(-1, slot_ids) * sqrt_weights).flatten(0, 1))
        timings["fit_s"] = perf_counter() - tick

        tick = perf_counter()
        retained_reduced_bytes = equivalent_dense_bytes = 0
        group_report = []
        for level in selected:
            layer = layers[level]
            weight, bias = saved[level]
            current = torch.cat((weight, bias[:, None]), dim=1).cpu().double().T
            anchor = torch.zeros_like(current) if initialize else current
            for slot_ids, _, fit in groups[level]:
                if fit.rows == 0:
                    continue
                retained_reduced_bytes += (fit.triangular.numel() + fit.projected.numel()) * 8
                equivalent_dense_bytes += fit.rows * (current.shape[0] + slot_ids.numel()) * 8
                solution, mu, old_sse, new_sse = fit.solve(anchor[:, slot_ids])
                device_ids = slot_ids.to(layer.weight.device)
                coefficients = solution.to(layer.weight)
                if not coefficients.isfinite().all() or not coefficients[-1].to(layer.bias).isfinite().all():
                    raise FloatingPointError("calibrated parameters are outside the deployed dtype's finite range")
                layer.weight.index_copy_(0, device_ids, coefficients[:-1].T)
                layer.bias.index_copy_(0, device_ids, solution[-1].to(layer.bias))
                group_report.append({"level": level, "slots": slot_ids.tolist(), "rows": fit.rows,
                                     "mu": mu, "before": old_sse, "after": new_sse})
        timings["solve_s"] = perf_counter() - tick

        tick = perf_counter()
        after = _ErrorTotals(len(layers))
        for entry in selections:
            example, result, weights = observe(entry["record_id"], entry["slots"])
            after.add(transform, result, example["target"], weights)
        timings["paired_pressure_s"] = perf_counter() - tick
        field_gain = before.field - after.field
        detail_gain = sum(before.detail) - sum(after.detail)
        scale = max(before.field, after.field, 1e-10)
        closure = max(abs(before.field - before.reconstructed), abs(after.field - after.reconstructed),
                      abs(field_gain - detail_gain))
        fixed_change = abs(before.fixed - after.fixed)
        fixed_change += sum(abs(before.detail[i] - after.detail[i]) for i in range(len(layers)) if i not in groups)
        if not math.isfinite(closure) or max(closure, fixed_change) > 1e-10 + 1e-5 * scale:
            raise FloatingPointError("deployed pressure change disagrees with the complete coefficient change")
        if initialize:
            if after.field > before.field + 1e-12 + 1e-8 * before.field:
                raise FloatingPointError("zero-anchor initialization increased its weighted TRAIN pressure objective")
            accepted = True
        else:
            tick = perf_counter()
            after_validation = evaluate_validation()
            timings["validation_s"] += perf_counter() - tick
            accepted = (after.field <= before.field + 1e-8 * max(before.field, 1e-12) and
                        after_validation["mean_record_relative_l2_squared"] <=
                        before_validation["mean_record_relative_l2_squared"])
        trial = {}
        for name, parameter in model.named_roles()["DETAIL"]:
            if any(parameter is layers[i].weight or parameter is layers[i].bias for i in selected):
                trial[name] = parameter.detach().cpu().clone()
        if not accepted:
            restore()
        fit_frames = n_records * n_times
        timings["total_s"] = perf_counter() - started
        return {"phase": phase, "seed": seed, "anchor": "zero" if initialize else "current_affines",
                "rewritten_levels": selected, "train_selections": selections,
                "time_strata_sizes": [len(band) for band in bands], "draws_per_stratum": queries_per_band,
                "fit_frames": fit_frames, "pressure_frame_evaluations": 2 * fit_frames,
                "fit_pressure_frame_evaluations": fit_frames, "paired_pressure_frame_evaluations": fit_frames,
                "record_reads": 2 * n_records, "fit_record_reads": n_records, "paired_record_reads": n_records,
                "timing_s": timings,
                "calibration_field_loss_before": before.field, "calibration_field_loss_after": after.field,
                "calibration_coefficient_reconstructed_field_loss_before": before.reconstructed,
                "calibration_coefficient_reconstructed_field_loss_after": after.reconstructed,
                "weighted_detail_loss_by_level_before": before.detail, "weighted_detail_loss_by_level_after": after.detail,
                "weighted_field_gain": field_gain, "weighted_detail_coefficient_gain": detail_gain,
                "field_detail_gain_absolute_error": abs(field_gain - detail_gain),
                "field_coefficient_closure_absolute": closure,
                "fixed_coarse_and_unselected_detail_change_absolute": fixed_change,
                "fixed_coarse_field_loss": before.fixed, "slot_groups": group_report,
                "output_dtypes": {str(i): str(layers[i].weight.dtype) for i in selected},
                "retained_fit_state_bytes": retained_reduced_bytes,
                "dense_rows_for_same_fit_bytes": equivalent_dense_bytes,
                "validation_before": before_validation, "validation_after": after_validation,
                "accepted": accepted, "candidate_affines": trial}
    except BaseException:
        restore()
        raise
    finally:
        for module, mode in modes:
            module.training = mode


class _ErrorTotals:
    """Measure pressure and valid coefficient losses at the deployed precision."""
    def __init__(self, n_levels):
        self.detail = [0.] * n_levels
        self.field = self.fixed = 0.

    @property
    def reconstructed(self):
        return self.fixed + sum(self.detail)

    def add(self, transform, parts, target, weights):
        base, prediction = parts["base_pressure"], parts["prediction"]
        if base.shape != target.shape or prediction.shape != target.shape or target.ndim != 3:
            raise ValueError("calibration needs matched complete pressure fields")
        if base[:, 0].count_nonzero().item() or prediction[:, 0].count_nonzero().item():
            raise ValueError("calibration needs the deployed zero free-surface pressure")
        target = target.to(base.device, torch.float64)
        weights = weights.to(base.device)

        def weighted_sse(residual):
            return torch.dot(residual.double().square().flatten(1).sum(1), weights).item()

        residual = target - base.double()
        self.field += weighted_sse(target - prediction.double())
        coarse, packets = transform.analyze(residual[:, None, 1:])
        self.fixed += weighted_sse(coarse) + weighted_sse(residual[:, 0])
        wanted = []
        for i, (level, packet, candidate) in enumerate(zip(
                transform.levels, packets, parts["detail_coefficients"], strict=True)):
            detail = packet[..., 16:]
            mask = level.valid_detail
            if candidate.shape != detail.shape or mask.shape != detail.shape[2:]:
                raise ValueError("calibration coefficients do not match their valid-slot layout")
            self.detail[i] += weighted_sse((candidate.double() - detail.double()).masked_fill(~mask, 0.))
            wanted.append(detail)
        return wanted
