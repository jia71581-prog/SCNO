"""TRAIN-only phase-B residual fitting that preserves every trained old factor.

New, unused zero time columns can encode delta folded affines exactly. The old
trained columns/rows are never canonicalized or cleared. A first zero-delta
identity allocation is pressure-function preserving; QR-fitted deltas are
accepted only after an actual weighted pressure-error check on all requested
TRAIN calibration frames. True resume must restore the saved allocation.
"""

from copy import deepcopy
import hashlib
import math
from time import perf_counter

import torch

from joint_fno_calibration import _squared_norm
from wide_time_basis import coefficient_packets, prepare_coefficient_features, _tensor_hash


SCHEMA = "SCNO_WIDE_STAGE_B_PRESERVE_OLD_FACTORS_V1"


class _QR:
    def __init__(self):
        self.r = self.rhs = None
        self.trace = 0.0
        self.rows = 0

    def add(self, design, residual):
        self.trace += design.square().sum().item()
        self.rows += len(design)
        if self.r is not None:
            design = torch.cat((self.r, design))
            residual = torch.cat((self.rhs, residual))
        q, self.r = torch.linalg.qr(design, mode="reduced")
        self.rhs = q.T @ residual

    def solve(self):
        width = self.r.shape[1]
        mu = max(1e-12, 1e-4 * self.trace / width)
        ridge = torch.eye(width, dtype=self.r.dtype, device=self.r.device) * math.sqrt(mu)
        q, upper = torch.linalg.qr(torch.cat((self.r, ridge)), mode="reduced")
        # Delta has zero ridge anchor; current effective theta stays in the old
        # factors and in any other unmodified columns.
        rhs = q[:self.r.shape[0]].T @ self.rhs
        return torch.linalg.solve_triangular(upper, rhs, upper=True), mu


def _layouts(mask):
    flat = mask.detach().cpu().flatten(0, 1)
    groups = {}
    for slot in range(flat.shape[1]):
        layout = flat[:, slot]
        if layout.any():
            groups.setdefault(layout.numpy().tobytes(), [layout, []])[1].append(slot)
    return [(layout, torch.tensor(slots, dtype=torch.long)) for layout, slots in groups.values()]


def _pressure_error(prediction, target, weights):
    return ((prediction.double() - target.double()).square().flatten(2).sum(-1) * weights).sum().item()


def _coefficient_error(predicted, coarse, targets, weights):
    value = (coarse.double().square().flatten(2).sum(-1) * weights).sum()
    for p, y in zip(predicted, targets, strict=True):
        value += ((p.double() - y.double()).square().flatten(2).sum(-1) * weights).sum()
    return value.item()


def _source_and_sample_hash(source, target):
    digest = hashlib.sha256()
    for value in (source, target):
        value = value.detach().cpu().contiguous()
        digest.update(str((list(value.shape), str(value.dtype))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _unpack_record(model, row, draws, train_ids, test_ids, seen, energy_floor):
    key = str(row.get("id", row.get("record_id")))
    if key not in train_ids or key in test_ids or key not in draws or key in seen:
        raise ValueError("factory must supply each requested TRAIN calibration record exactly once")
    if str(row.get("split", "train")).lower() not in ("train", "training"):
        raise ValueError("calibration accepts TRAIN labels only")
    seen.add(key)
    draw = draws[key]
    context = row.get("context")
    if context is None:
        context = model.encode(row["velocity"], row["source"])
    features, source = context[:2]
    if len(features) != 1 or source.shape != (1, 5) or source[0, 4].item() == 0:
        raise ValueError("one nonzero-amplitude source is required per TRAIN calibration record")
    target = row["target"]
    if not torch.is_tensor(target) or not target.is_floating_point():
        raise ValueError("targets must be normalized pressure tensors")
    if target.ndim == 4 and len(target) == 1:
        target = target[0]
    if target.ndim != 3 or tuple(target.shape[-2:]) != tuple(features.shape[-2:]):
        raise ValueError("reference and encoded complete pressure grids differ")
    native_frames = len(model.full_times_s)
    raw_energy = row.get("record_energy", row.get("full_record_energy"))
    if raw_energy is None:
        if len(target) != native_frames:
            raise ValueError("eight sampled frames cannot define the original full-record energy")
        energy = _squared_norm(target).item()
    else:
        energy = float(raw_energy)
    if not math.isfinite(energy) or energy < 0:
        raise ValueError("supply finite full-native-record energy in normalized pressure units")
    slots = torch.tensor(draw["frame_indices"], dtype=torch.long, device=features.device)
    if len(target) == native_frames:
        selected = target.index_select(0, slots.to(target.device))
    elif len(target) == 8 and list(row.get("frame_indices", ())) == draw["frame_indices"]:
        selected = target
    else:
        raise ValueError("supply all native frames or the exact eight explicitly indexed calibration frames")
    selected = selected.detach().to(device=features.device, dtype=features.dtype)[None]
    if torch.count_nonzero(selected[:, :, 0]).item():
        raise ValueError("normalized references must satisfy the exactly enforced free surface")
    weights = torch.tensor(
        draw["weight_numerators"], dtype=torch.float64, device=features.device)[None] / max(energy, energy_floor)
    return key, context, selected, slots, weights, energy


@torch.no_grad()
def calibrate_wide_stage_b(
        model, records_factory, *, calibration_spec, train_ids, test_ids=(), optimizer=None,
        energy_floor=1e-12, expansion_receipt=None, expected_records=32,
        startup_mode="new_training", qr_device="cpu", pressure_tolerance=5e-5):
    """Perform the first phase-B 32 TRAIN x eight-frame residual initialization.

    records_factory() yields repeatable Trajectories-style mappings with id,
    velocity, source, and normalized target. context can replace velocity/source.
    A record_energy field may provide its precomputed ORIGINAL full-native
    normalized energy; without it the full target is required and read only.
    Eight-frame targets must additionally supply matching frame_indices.

    calibration_spec is StageController's fixed-seed roster, with two frames in
    each of four full-native strata and stratum_size/2 weight numerators.
    All old trained factor values, hidden layers, BASE and feature projection
    stay unchanged. Only certified newly added, zero-time columns may change.
    """
    if startup_mode != "new_training":
        raise ValueError("true resume restores its saved factors/receipt and must not recalibrate")
    if not callable(records_factory) or not math.isfinite(energy_floor) or energy_floor <= 0:
        raise ValueError("supply a repeatable TRAIN factory and positive finite energy floor")
    if type(expected_records) is not int or expected_records < 1:
        raise ValueError("expected_records must be positive; production uses exactly 32")
    if (getattr(model, "nonlinear_details", True) or getattr(model, "coarse_head", None) is not None
            or getattr(model, "detail_head", None) is not None or getattr(model, "basis_size", None) is None):
        raise ValueError("this stage-B fit supports the static wide head without a new coarse/nonlinear branch")
    parent, decoder = model.parent, model.parent.decoder
    rank, n_times = decoder.detail_latent, len(model.full_times_s)
    if getattr(parent, "residual_gain", None) is not None:
        if parent.residual_gain.detach().item() != 1.0 or parent.residual_gain.requires_grad:
            raise ValueError("inherit the already absorbed, frozen unit neural gain")
    detail_parameters = {
        id(p) for module in (decoder.detail_projection, decoder.detail_heads, decoder.detail_time)
        for p in module.parameters()}
    if any(p.requires_grad and id(p) not in detail_parameters for p in model.parameters()):
        raise ValueError("phase-B BASE must be frozen before residual calibration")
    previous = getattr(model, "wide_stage_b_calibration_receipt", None)
    if previous is None:
        previous = getattr(model, "repair_receipt", {}).get("wide_stage_b_calibration")
    if previous and previous.get("accepted"):
        raise ValueError("stage-B calibration already completed; restore its allocation instead of repeating")
    source_receipt = expansion_receipt or getattr(model, "basis_expansion_receipt", None)
    if (not source_receipt or source_receipt.get("operation") != "new_training_basis_expansion"
            or source_receipt.get("new_basis") != rank):
        raise ValueError("require provenance for newly added, unused output-factor columns")
    old_rank = source_receipt["old_basis"]
    if type(old_rank) is not int or not 0 < old_rank < rank or rank - old_rank < n_times:
        raise ValueError("not enough newly added basis columns to preserve all old factors and encode native-time deltas")
    train_ids, test_ids = set(map(str, train_ids)), set(map(str, test_ids))
    if not train_ids or train_ids & test_ids:
        raise ValueError("declare disjoint nonempty TRAIN and optional TEST manifests")
    spec = deepcopy(calibration_spec)
    if (spec.get("records") != expected_records or spec.get("frames_per_record") != 8
            or spec.get("draws_per_stratum") != 2 or spec.get("native_frames") != n_times
            or spec.get("pressure_frames") != expected_records * 8):
        raise ValueError("require the declared 32-record, four-stratum, eight-frame B calibration support")
    sizes = spec["stratum_sizes"]
    quotient, remainder = divmod(n_times, 4)
    if sizes != [quotient + int(i < remainder) for i in range(4)]:
        raise ValueError("calibration strata do not partition the complete native time axis")
    draws = {}
    for draw in spec["draws"]:
        key = str(draw["record_id"])
        slots, labels = draw["frame_indices"], draw["stratum_indices"]
        if (key in draws or key not in train_ids or key in test_ids or len(slots) != 8
                or len(set(slots)) != 8 or labels != [0, 0, 1, 1, 2, 2, 3, 3]):
            raise ValueError("invalid TRAIN calibration roster or four-by-two frame sampling")
        boundaries = [0]
        for size in sizes:
            boundaries.append(boundaries[-1] + size)
        if any(not boundaries[band] <= slot < boundaries[band + 1] for slot, band in zip(slots, labels, strict=True)):
            raise ValueError("a selected frame is outside its declared time stratum")
        if draw["weight_numerators"] != [sizes[band] / 2.0 for band in labels]:
            raise ValueError("weights must be stratum_size/(2*original_record_energy)")
        draws[key] = draw
    if len(draws) != expected_records or set(map(str, spec["train_record_ids"])) != set(draws):
        raise ValueError("the exact declared TRAIN calibration roster is required")
    fit_device = torch.device(qr_device)
    layouts = [_layouts(level.valid_detail) for level in decoder.detail_transform.levels]
    outputs, originals, chosen = [], [], []
    for scale, (head, time, level) in enumerate(zip(
            decoder.detail_heads, decoder.detail_time, decoder.detail_transform.levels, strict=True)):
        spatial = head[-1]
        slots = level.valid_detail.shape[-1]
        if spatial.in_features != 32 or time.weight.shape != (n_times, rank):
            raise ValueError("require the existing 32-hidden-feature/33-affine wide factor computation")
        provenance = source_receipt["scales"][scale]
        if (_tensor_hash(spatial.weight[:old_rank * slots]) != provenance["copied_spatial_weight_sha256"]
                or _tensor_hash(time.weight[:, :old_rank]) != provenance["copied_time_weight_sha256"]):
            raise ValueError("old trained factors changed since the recorded frozen A-stage expansion")
        unused = ((torch.count_nonzero(time.weight[:, old_rank:], dim=0) == 0)
                  .nonzero().flatten().add(old_rank).cpu().tolist())
        if len(unused) < n_times:
            raise ValueError("new columns have learned nonzero time factors; do not overwrite trained factors")
        chosen.append(torch.tensor(unused[:n_times], dtype=torch.long, device=time.weight.device))
        outputs.append((spatial, time, slots))
        originals.append((spatial.weight.clone(), spatial.bias.clone(), time.weight.clone()))
    modes = [(module, module.training) for module in model.modules()]
    start = perf_counter()
    accepted = mutated = False
    groups = [{} for _ in outputs]
    before_pressure = before_coefficients = 0.0
    first_audit_context = audit_before = None
    record_receipts = {}
    report = None
    try:
        model.eval()
        model.compiled_readout = None
        seen = set()
        for row in records_factory():
            key, context, target, time_slots, weights, energy = _unpack_record(
                model, row, draws, train_ids, test_ids, seen, energy_floor)
            features = context[0]
            if first_audit_context is None:
                first_audit_context = context
                audit_before = model.decode(context, normalized=True).detach()
            base = parent.decode(context, time_slots, normalized=True, detail_enabled=False)
            residual = target - base
            coarse, full_packets = decoder.detail_transform.analyze(residual[:, :, 1:])
            targets = tuple(
                packet[..., 16:].masked_fill(~level.valid_detail, 0)
                for packet, level in zip(full_packets, decoder.detail_transform.levels, strict=True))
            hidden = prepare_coefficient_features(decoder, features)
            current = coefficient_packets(decoder, hidden, time_slots)
            prediction = model.decode(context, time_slots, normalized=True)
            before_pressure += _pressure_error(prediction, target, weights)
            before_coefficients += _coefficient_error(current, coarse, targets, weights)
            record_receipts[key] = {
                "original_full_record_energy": energy,
                "source_sample_sha256": _source_and_sample_hash(context[1], target)}
            for scale, (h, target_coefficients, current_coefficients) in enumerate(zip(hidden, targets, current, strict=True)):
                h = h[0].detach().flatten(0, 1).to(device=fit_device, dtype=torch.float64)
                design = torch.cat((h, torch.ones_like(h[:, :1])), dim=1)
                residual_coefficients = (target_coefficients - current_coefficients)[0].detach().flatten(1, 2).to(
                    device=fit_device, dtype=torch.float64)
                for position, native_time in enumerate(draws[key]["frame_indices"]):
                    scale_weight = math.sqrt(weights[0, position].item())
                    for layout_id, (layout, slot_indices) in enumerate(layouts[scale]):
                        layout = layout.to(fit_device)
                        slot_indices = slot_indices.to(fit_device)
                        group = groups[scale].setdefault((native_time, layout_id), _QR())
                        group.add(
                            design[layout] * scale_weight,
                            residual_coefficients[position][layout][:, slot_indices] * scale_weight)
        if seen != set(draws):
            raise ValueError("TRAIN calibration factory omitted requested records")
        if abs(before_pressure - before_coefficients) > pressure_tolerance * max(before_pressure, before_coefficients, 1e-12):
            raise ValueError("masked detail/coarse and pressure objectives disagree before fitting")
        deltas, solve_receipts = [], []
        for scale, (spatial, _, slots) in enumerate(outputs):
            delta = torch.zeros((n_times, slots, 33), dtype=torch.float64, device=fit_device)
            solves = []
            for (native_time, layout_id), group in sorted(groups[scale].items()):
                theta, mu = group.solve()
                slot_indices = layouts[scale][layout_id][1].to(fit_device)
                delta[native_time, slot_indices] = theta.T
                solves.append({
                    "time_index": native_time, "layout": layout_id, "slots": slot_indices.cpu().tolist(),
                    "rows": group.rows, "ridge_mu": mu})
            if not torch.isfinite(delta).all():
                raise FloatingPointError("non-finite folded residual affine")
            deltas.append(delta)
            solve_receipts.append(solves)
        # Allocate zero deltas first. Only the explicitly certified unused
        # columns are touched; ORIGINAL trained time/spatial factors stay exact.
        mutated = True
        for (spatial, time, slots), reserve in zip(outputs, chosen, strict=True):
            spatial.weight.view(rank, slots, 32).index_fill_(0, reserve, 0)
            spatial.bias.view(rank, slots).index_fill_(0, reserve, 0)
            time.weight.index_copy_(1, reserve, torch.eye(n_times, device=time.weight.device, dtype=time.weight.dtype))
        audit_after = model.decode(first_audit_context, normalized=True)
        audit_difference = _squared_norm(audit_after - audit_before).sqrt().item()
        audit_norm = _squared_norm(audit_before).sqrt().item()
        audit_relative = audit_difference / max(audit_norm, 1e-30)
        if not torch.allclose(audit_after, audit_before, rtol=pressure_tolerance, atol=1e-6):
            raise ValueError("zero-delta allocation failed the complete native pressure identity")
        if n_times > 1:
            early_difference = (
                (audit_after[:, 1] - audit_after[:, 0]) - (audit_before[:, 1] - audit_before[:, 0]))
            derivative_relative = _squared_norm(early_difference).sqrt().item() / max(
                _squared_norm(audit_before[:, 1] - audit_before[:, 0]).sqrt().item(), 1e-30)
        else:
            derivative_relative = 0.0
        del audit_before, audit_after
        for (spatial, _, slots), reserve, delta in zip(outputs, chosen, deltas, strict=True):
            spatial.weight.view(rank, slots, 32).index_copy_(0, reserve, delta[:, :, :32].to(spatial.weight))
            spatial.bias.view(rank, slots).index_copy_(0, reserve, delta[:, :, 32].to(spatial.bias))
        after_pressure = after_coefficients = 0.0
        seen_after = set()
        for row in records_factory():
            key, context, target, time_slots, weights, energy = _unpack_record(
                model, row, draws, train_ids, test_ids, seen_after, energy_floor)
            if (energy != record_receipts[key]["original_full_record_energy"]
                    or _source_and_sample_hash(context[1], target) != record_receipts[key]["source_sample_sha256"]):
                raise ValueError("repeatable calibration factory changed its target/source/energy")
            base = parent.decode(context, time_slots, normalized=True, detail_enabled=False)
            coarse, packets = decoder.detail_transform.analyze((target - base)[:, :, 1:])
            targets = tuple(
                packet[..., 16:].masked_fill(~level.valid_detail, 0)
                for packet, level in zip(packets, decoder.detail_transform.levels, strict=True))
            current = coefficient_packets(decoder, context[0], time_slots)
            prediction = model.decode(context, time_slots, normalized=True)
            after_pressure += _pressure_error(prediction, target, weights)
            after_coefficients += _coefficient_error(current, coarse, targets, weights)
        if seen_after != set(draws):
            raise ValueError("verification omitted requested TRAIN calibration records")
        # Compare on the original common objective scale: at a near-zero
        # candidate, pressure subtraction can leave harmless roundoff larger
        # than a relative tolerance measured against that tiny remainder.
        objective_scale = max(before_pressure, before_coefficients, after_pressure, after_coefficients, 1e-12)
        if abs(after_pressure - after_coefficients) > pressure_tolerance * objective_scale:
            raise ValueError("masked coefficient and actual pressure objectives disagree after fitting")
        preserved_hashes = []
        for scale, ((spatial, time, slots), original) in enumerate(zip(outputs, originals, strict=True)):
            if (not torch.equal(spatial.weight[:old_rank * slots], original[0][:old_rank * slots])
                    or not torch.equal(spatial.bias[:old_rank * slots], original[1][:old_rank * slots])
                    or not torch.equal(time.weight[:, :old_rank], original[2][:, :old_rank])):
                raise RuntimeError("a trained old factor was changed")
            preserved_hashes.append({
                "scale": scale,
                "old_spatial_weight_before": _tensor_hash(original[0][:old_rank * slots]),
                "old_spatial_weight_after": _tensor_hash(spatial.weight[:old_rank * slots]),
                "old_spatial_bias_before": _tensor_hash(original[1][:old_rank * slots]),
                "old_spatial_bias_after": _tensor_hash(spatial.bias[:old_rank * slots]),
                "old_time_weight_before": _tensor_hash(original[2][:, :old_rank]),
                "old_time_weight_after": _tensor_hash(time.weight[:, :old_rank]),
            })
        accepted = math.isfinite(after_pressure) and after_pressure <= before_pressure
        cleared, rewritten = [], []
        parameter_names = {id(p): name for name, p in model.named_parameters()}
        if accepted:
            for spatial, time, _ in outputs:
                for parameter in (spatial.weight, spatial.bias, time.weight):
                    name = parameter_names[id(parameter)]
                    rewritten.append(name)
                    parameter.grad = None
                    if optimizer is not None and parameter in optimizer.state:
                        optimizer.state.pop(parameter)
                        cleared.append(name)
        report = {
            "schema": SCHEMA, "state": "complete" if accepted else "rejected",
            "accepted": accepted, "records": expected_records,
            "calibration_frames": expected_records * 8, "frames_per_record": 8,
            "native_frames": n_times, "calibration_spec": spec,
            "train_only": True, "energy_floor": float(energy_floor),
            "objective": "sum_record_stratum_weighted_full_pressure_sse_over_original_record_energy",
            "before": before_pressure / expected_records,
            "after": (after_pressure if accepted else before_pressure) / expected_records,
            "candidate_after": after_pressure / expected_records,
            "before_coefficient_objective": before_coefficients / expected_records,
            "candidate_after_coefficient_objective": after_coefficients / expected_records,
            "pressure_coefficient_change_relative_discrepancy": abs(
                (before_pressure - after_pressure) - (before_coefficients - after_coefficients)) / objective_scale,
            "coefficient_count_averaging": False, "invalid_detail_slots_excluded": True,
            "old_trained_basis": old_rank, "wide_basis": rank,
            "source_expansion_receipt": deepcopy(source_receipt),
            "preserved_trained_factor_hashes": preserved_hashes,
            "new_reserved_columns_per_scale": [indices.cpu().tolist() for indices in chosen],
            "new_rows_replaced": "previously unused spatial output rows only; old rows preserved",
            "delta_anchor": "zero; existing effective theta retained in unchanged old factors",
            "time_encoding": "identity on reserved new columns",
            "old_trained_factor_values_preserved": True,
            "complete_native_pressure_identity_relative_error": audit_relative,
            "initial_difference_identity_relative_error": derivative_relative,
            "identity_pressure_frames_evaluated": 2 * n_times,
            "calibration_pressure_frames_evaluated": 2 * expected_records * 8,
            "ridge_rule": "max(1e-12,1e-4*trace(X.T@X)/33)",
            "qr_device": str(fit_device), "solve_dtype": "torch.float64",
            "solves": solve_receipts, "record_receipts": record_receipts,
            "rewritten_parameters": rewritten, "optimizer_states_cleared": cleared,
            "optimizer_reset_scope": "modified output Parameters only; old factor values unchanged",
            "elapsed_s": perf_counter() - start,
        }
        model.wide_stage_b_calibration_receipt = deepcopy(report)
        model.repair_receipt["wide_stage_b_calibration"] = deepcopy(report)
        return report
    finally:
        if mutated and not accepted:
            for (spatial, time, _), original in zip(outputs, originals, strict=True):
                spatial.weight.copy_(original[0])
                spatial.bias.copy_(original[1])
                time.weight.copy_(original[2])
        for module, mode in modes:
            module.training = mode


def run_cpu_checks():
    """Small actual static-head/wrapper test; no production data or GPU."""
    from io import BytesIO
    from pathlib import Path
    from types import SimpleNamespace
    from typing import NamedTuple
    import importlib.util
    import sys
    from torch import nn
    from torch.nn import functional as F
    from joint_fno_decoder import JointFNOTimeDecoder, _time_projection
    from joint_fno_model import JointFNOSCNO
    from method_repair_model import CorrectedSCNOReadout
    from repair_stage_schedule import stage_b_calibration_spec
    from wide_base_readout import expand_base_readout
    from wide_time_basis import expand_decoder_time_basis, decode_coefficient_first

    spec_module = importlib.util.spec_from_file_location(
        "_wide_stage_b_cpu_spectral", Path(__file__).parent / "vendor/backbone/spectral_detail.py")
    spectral = importlib.util.module_from_spec(spec_module)
    sys.modules[spec_module.name] = spectral
    spec_module.loader.exec_module(spectral)

    class Context(NamedTuple):
        features: torch.Tensor
        source: torch.Tensor

    class Gabor(nn.Module):
        def forward(self, times, delay, frequency, duration):
            phase = (times - delay) * frequency * (2 * math.pi)
            return torch.stack((phase.sin(), phase.cos(), times / duration), dim=-1)

    class Parent(nn.Module):
        _selected_times = JointFNOSCNO._selected_times

        def __init__(self):
            super().__init__()
            self.encoder = nn.Conv2d(3, 5, 1)
            decoder = JointFNOTimeDecoder(
                5, torch.linspace(0, 1, 12),
                detail_transform=spectral.PolynomialPyramid((16, 17), merges=1))
            decoder, self.expansion = expand_decoder_time_basis(decoder, 800)
            self.decoder, _ = expand_base_readout(decoder, 800)
            self.detail_latent, self.duration_s = 800, 1.0
            self.fixed_branch = nn.Conv2d(5, 12, 1)
            self.gabor = Gabor()
            self.gabor_embedding = nn.Linear(3, 8)
            self.residual_gain = nn.Parameter(torch.tensor(1.), requires_grad=False)
            self.normalizer = SimpleNamespace(metadata=SimpleNamespace(pressure_scale_pa=1.0))

        @property
        def device(self):
            return self.full_times_s.device

        @property
        def full_times_s(self):
            return self.decoder.native_times

        @property
        def detail_transform(self):
            return self.decoder.detail_transform

        def encode(self, velocity, source):
            return Context(self.encoder(velocity), source)

        def decode(self, context, frame_indices=None, *, normalized=True, detail_enabled=True):
            if detail_enabled:
                prediction = decode_coefficient_first(self.decoder, context.features, frame_indices)
            else:
                hidden = context.features[:, :, 1:]
                for layer in list(self.decoder.proj.children())[:-1]:
                    hidden = layer(hidden)
                last = self.decoder.proj[-1]
                prediction = F.pad(_time_projection(
                    hidden, last.weight[:, :, 0, 0], last.bias, frame_indices), (0, 0, 1, 0))
            fixed = self.fixed_branch(context.features[:, :, 1:])
            if frame_indices is not None:
                fixed = fixed[:, frame_indices]
            return prediction + F.pad(fixed, (0, 0, 1, 0))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(61971)
        parent = Parent()
        model = CorrectedSCNOReadout(
            parent, nonlinear_details=False, coarsest_base=False,
            time_chunk=12, basis_size=800, restore_basis=True)
        model.basis_expansion_receipt = deepcopy(parent.expansion)
        model.requires_grad_(False)
        for module in (parent.decoder.detail_projection, parent.decoder.detail_heads, parent.decoder.detail_time):
            module.requires_grad_(True)
        teacher, failed = deepcopy(model), deepcopy(model)
        for head, time, level in zip(
                teacher.parent.decoder.detail_heads, teacher.parent.decoder.detail_time,
                teacher.detail_transform.levels, strict=True):
            slots = level.valid_detail.shape[-1]
            with torch.no_grad():
                time.weight[:, 8:20] = torch.eye(12)
                head[-1].weight.view(800, slots, 32)[8:20] = torch.randn(12, slots, 32) * .03
                head[-1].bias.view(800, slots)[8:20] = torch.randn(12, slots) * .02
        source = torch.tensor([[1000., 1000., 2., .05, 1.]])
        rows = []
        with torch.no_grad():
            for index in range(2):
                velocity = torch.randn(1, 3, 17, 17) * (index + 1)
                target = teacher.decode(teacher.encode(velocity, source))[0].detach()
                rows.append({"id": f"train{index}", "velocity": velocity, "source": source, "target": target})
        support = stage_b_calibration_spec(
            {"seed": 61971, "affine_calibration_records": 2, "affine_calibration_times_per_record": 8},
            [row["id"] for row in rows], model.full_times_s)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        for parameter in optimizer.param_groups[0]["params"]:
            optimizer.state[parameter] = {
                "step": torch.tensor(1.), "exp_avg": torch.ones_like(parameter),
                "exp_avg_sq": torch.ones_like(parameter)}
        report = calibrate_wide_stage_b(
            model, lambda: iter(rows), calibration_spec=support,
            train_ids=[row["id"] for row in rows], test_ids=["test0"],
            optimizer=optimizer, expected_records=2)
        assert report["accepted"] and report["after"] < report["before"]
        changed = set(report["rewritten_parameters"])
        assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items() if name not in changed)
        for scale, (head, time, level) in enumerate(zip(
                model.parent.decoder.detail_heads, model.parent.decoder.detail_time,
                model.detail_transform.levels, strict=True)):
            slots = level.valid_detail.shape[-1]
            prefix = f"parent.decoder.detail_heads.{scale}.4"
            assert torch.equal(head[-1].weight[:8 * slots], before[prefix + ".weight"][:8 * slots])
            assert torch.equal(head[-1].bias[:8 * slots], before[prefix + ".bias"][:8 * slots])
            assert torch.equal(time.weight[:, :8], before[f"parent.decoder.detail_time.{scale}.weight"][:, :8])
            assert torch.equal(head[-1].weight[20 * slots:], before[prefix + ".weight"][20 * slots:])
            assert torch.equal(time.weight[:, 20:], before[f"parent.decoder.detail_time.{scale}.weight"][:, 20:])
            assert head[-1].weight not in optimizer.state and time.weight not in optimizer.state
        assert model.parent.decoder.detail_projection.weight in optimizer.state
        for mode in ("resume", "new_training"):
            try:
                calibrate_wide_stage_b(
                    model, lambda: iter(rows), calibration_spec=support,
                    train_ids=[row["id"] for row in rows], startup_mode=mode, expected_records=2)
            except ValueError:
                pass
            else:
                raise AssertionError("resume or repeated initialization was permitted")
        failure_snapshot = {name: value.clone() for name, value in failed.state_dict().items()}
        calls = [0]

        def broken_factory():
            calls[0] += 1
            for row in rows:
                if calls[0] == 2:
                    raise RuntimeError("synthetic verification interruption")
                yield row

        try:
            calibrate_wide_stage_b(
                failed, broken_factory, calibration_spec=support,
                train_ids=[row["id"] for row in rows], expected_records=2)
        except RuntimeError as exc:
            assert "verification interruption" in str(exc)
        else:
            raise AssertionError("synthetic second-pass failure was not raised")
        assert all(torch.equal(value, failure_snapshot[name]) for name, value in failed.state_dict().items())
        stream = BytesIO()
        torch.save({"model": model.state_dict(), "repair_receipt": model.repair_receipt}, stream)
        stream.seek(0)
        payload = torch.load(stream, weights_only=True)
        failed.load_state_dict(payload["model"], strict=True)
        failed.repair_receipt = payload["repair_receipt"]
        with torch.no_grad():
            assert torch.equal(model.decode(model.encode(rows[0]["velocity"], source)),
                               failed.decode(failed.encode(rows[0]["velocity"], source)))
        return {
            "passed": True, "actual_wide_basis": 800, "actual_base_hidden_width": 800,
            "old_trained_time_and_spatial_factors_exact": True,
            "unreserved_new_factors_and_other_model_state_exact": True,
            "full_native_zero_delta_identity_error": report["complete_native_pressure_identity_relative_error"],
            "initial_difference_identity_error": report["initial_difference_identity_relative_error"],
            "weighted_full_pressure_before": report["before"],
            "weighted_full_pressure_after": report["after"],
            "coefficient_pressure_agreement_after": abs(
                report["candidate_after_coefficient_objective"] - report["candidate_after"]),
            "valid_masks_and_original_record_energy_weights": True,
            "selective_optimizer_reset": True, "resume_and_repeat_blocked": True,
            "interrupted_verification_rolls_back": True, "save_reload_prediction_exact": True,
        }


if __name__ == "__main__":
    import json
    import sys
    if sys.argv[1:] == ["--cpu-check"]:
        torch.set_num_threads(2)
        print(json.dumps(run_cpu_checks(), indent=2))
    else:
        print("Import calibrate_wide_stage_b; --cpu-check runs a CPU-only static-head check.")
