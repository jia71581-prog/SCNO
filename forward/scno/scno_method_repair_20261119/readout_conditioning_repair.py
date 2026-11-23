"""Function-preserving gain removal and TRAIN-only anchored residual readout fitting.

This is an explicit new-training transition, never an automatic resume hook.
Existing checkpoint files are not edited. Save a separate conditioned payload
with conditioning_checkpoint_payload and load it with
load_conditioned_checkpoint before restoring its optimizer/RNG/scheduler.
"""

from copy import deepcopy
import math
from pathlib import Path
from time import perf_counter

import torch

from joint_fno_calibration import _ReducedQR, _output_fields, _squared_norm


SCHEMA = "scno_gain_absorbed_readout_v1"
_SEMANTICS = {
    "schema": SCHEMA,
    "neural_gain": 1.0,
    "neural_gain_trainable": False,
    "readout_ridge_anchor": "current_effective_rows",
}


def _new_training_only(startup_mode):
    if startup_mode == "resume":
        raise ValueError("a true resume must not absorb gains or recalibrate learned parameters")
    if startup_mode not in ("fresh", "warm_start", "new_training"):
        raise ValueError("declare startup_mode as fresh, warm_start, new_training, or resume")


def _row_parameters(model):
    final = model.decoder.proj[-1]
    final_name = f"decoder.proj.{len(model.decoder.proj) - 1}"
    return {
        final_name + ".weight": final.weight,
        final_name + ".bias": final.bias,
        **{f"decoder.detail_time.{i}.weight": layer.weight
           for i, layer in enumerate(model.decoder.detail_time)},
    }


def _clear_moments(optimizer, parameters):
    cleared = []
    for parameter in parameters.values():
        parameter.grad = None
    if optimizer is not None:
        for name, parameter in parameters.items():
            if parameter in optimizer.state:
                optimizer.state.pop(parameter)
                cleared.append(name)
    return cleared


def _assert_conditioned(model):
    gain = model.residual_gain
    receipt = getattr(model, "readout_conditioning_receipt", {})
    if (gain.numel() != 1 or not torch.isfinite(gain).all()
            or gain.detach().item() != 1.0 or gain.requires_grad
            or receipt.get("schema") != SCHEMA):
        raise ValueError("first make an explicit function-preserving gain-absorption transition")


@torch.no_grad()
def absorb_neural_gain(model, *, startup_mode, optimizer=None):
    """Absorb gamma into every linear neural time readout, leaving carrier fixed.

    Bias, detail rows, and the affine Gabor embedding are all included.
    Unchanged upstream features keep the same pressure/input derivatives up to
    rounding. Gradients in the rescaled output-parameter coordinates change
    intentionally. Changed parameters lose old optimizer state.
    """
    _new_training_only(startup_mode)
    previous = getattr(model, "readout_conditioning_receipt", {})
    if previous.get("schema") == SCHEMA:
        _assert_conditioned(model)
        return {**deepcopy(previous), "already_conditioned": True}
    gain = model.residual_gain
    if gain.numel() != 1 or not torch.isfinite(gain).all():
        raise ValueError("expected one finite neural residual gain")
    gamma = gain.detach().item()
    if gamma == 0:
        raise ValueError("zero gain destroys information; choose an explicit fresh initialization instead")
    parameters = _row_parameters(model)
    parameters["gabor_embedding.weight"] = model.gabor_embedding.weight
    if model.gabor_embedding.bias is not None:
        parameters["gabor_embedding.bias"] = model.gabor_embedding.bias
    if any(parameter is None for parameter in parameters.values()):
        raise ValueError("this readout requires its base affine bias")
    proposed = {name: parameter.detach() * gamma for name, parameter in parameters.items()}
    if any(not torch.isfinite(value).all() for value in proposed.values()):
        raise FloatingPointError("gain absorption would overflow a readout parameter")
    for name, parameter in parameters.items():
        parameter.copy_(proposed[name])
        parameter.grad = None
    gain.fill_(1.0)
    gain.requires_grad_(False)
    gain.grad = None
    changed = {**parameters, "residual_gain": gain}
    cleared = _clear_moments(optimizer, changed)
    receipt = {
        "schema": SCHEMA, "transition": "function_preserving_gain_absorption",
        "startup_mode": startup_mode, "previous_gain": gamma,
        "neural_gain": 1.0, "neural_gain_trainable": False,
        "changed_parameters": list(changed), "optimizer_states_cleared": cleared,
        "calibration": None,
    }
    model.readout_conditioning_receipt = receipt
    if hasattr(model, "initialization_receipt"):
        model.initialization_receipt["readout_conditioning"] = deepcopy(receipt)
    return deepcopy(receipt)


def conditioning_checkpoint_payload(model):
    """Return a distinctly identified payload for saving to a NEW checkpoint path."""
    _assert_conditioned(model)
    payload = model.checkpoint_payload()
    payload["model_semantics"] = {
        **payload["model_semantics"], "readout_conditioning": deepcopy(_SEMANTICS)}
    payload["readout_conditioning_receipt"] = deepcopy(model.readout_conditioning_receipt)
    return payload


def load_conditioned_checkpoint(native_model, payload, *, freeze_encoder=None):
    """Restore conditioned weights without absorption, fitting, or optimizer edits.

    The caller restores optimizer, scheduler, RNG, sampler and stage state from
    the same payload. This function restores only the model and conditioning
    receipt, with the original strict grid/axis/normalization compatibility gate.
    """
    from joint_fno_transport_model import TravelTimeFNOSCNO

    if isinstance(payload, (str, Path)):
        payload = torch.load(payload, map_location="cpu", weights_only=True)
    semantics = dict(payload.get("model_semantics", {}))
    if semantics.pop("readout_conditioning", None) != _SEMANTICS:
        raise ValueError("expected an explicitly gain-absorbed conditioned checkpoint")
    receipt = payload.get("readout_conditioning_receipt", {})
    if receipt.get("schema") != SCHEMA:
        raise ValueError("missing conditioned checkpoint receipt")
    base_payload = {**payload, "model_semantics": semantics}
    model = TravelTimeFNOSCNO.from_checkpoint(
        native_model, base_payload, freeze_encoder=freeze_encoder)
    if model.residual_gain.detach().item() != 1.0:
        raise ValueError("conditioned checkpoint has a non-unit neural gain")
    model.residual_gain.requires_grad_(False)
    model.readout_conditioning_receipt = deepcopy(receipt)
    model.initialization_receipt["readout_conditioning"] = deepcopy(receipt)
    _assert_conditioned(model)
    return model


def _validate_record(model, item, expected_ids, seen):
    if not isinstance(item, (tuple, list)) or len(item) != 3:
        raise ValueError("records_factory must yield (TRAIN record_id, encoded_context, normalized_target)")
    record_id, context, target = item
    if record_id not in expected_ids:
        raise ValueError(f"calibration record {record_id!r} is not in the declared TRAIN partition")
    if record_id in seen:
        raise ValueError(f"duplicate calibration record {record_id!r}")
    seen.add(record_id)
    features, source = context[:2]
    if (len(context) != 3 or features.ndim != 4 or len(features) != 1
            or source.shape != (1, 5) or features.device != model.device
            or source.device != model.device):
        raise ValueError("each record must encode one source and its fixed propagation carrier")
    if not isinstance(target, torch.Tensor) or not target.is_floating_point():
        raise ValueError("normalized targets must be real floating-point pressure tensors")
    if target.ndim == 4 and len(target) == 1:
        target = target[0]
    if tuple(target.shape) != (len(model.full_times_s), *features.shape[-2:]):
        raise ValueError("supply every native frame in its original order, on the full pressure grid")
    if torch.count_nonzero(target[:, 0]).item() != 0:
        raise ValueError("the top reference row must satisfy the exactly enforced free surface")
    return record_id, context, target


def _record_error(model, context, target, spatial_chunk):
    prediction = model.decode(context, normalized=True)
    if prediction.shape != (1, *target.shape):
        raise ValueError("unexpected full-sequence prediction shape")
    active_prediction = prediction[0, :, 1:].flatten(1)
    active_target = target[:, 1:].flatten(1)
    sse = energy = 0.0
    for first in range(0, active_target.shape[1], spatial_chunk):
        block = slice(first, first + spatial_chunk)
        truth = active_target[:, block].to(device=prediction.device, dtype=torch.float64)
        error = active_prediction[:, block].to(torch.float64) - truth
        sse += _squared_norm(error).item()
        energy += _squared_norm(truth).item()
    if not math.isfinite(sse) or not math.isfinite(energy):
        raise ValueError("targets and predictions must be finite")
    return sse, energy


@torch.no_grad()
def calibrate_residual_time_rows(
        model, records_factory, *, train_ids, startup_mode, optimizer=None,
        energy_floor=1e-12, spatial_chunk=2048, qr_dtype=None, ridge_relative=1e-4):
    """Jointly fit native time rows to y - current carrier - frozen Gabor.

    records_factory() is repeatable and yields one (id, context, normalized full
    sequence) per declared TRAIN id. For the clean partition pass all TRAIN900
    ids, never any held-out id. Every native frame is used. References retain
    their ORIGINAL full-sequence energy denominator; subtracting the carrier
    does not alter that denominator. Native synthesis already excludes invalid
    detail slots and pads the free surface.

    The fit is anchored at the current effective rows. QR reductions and the
    augmented ridge solve avoid normal-equation inversion. A second pass through
    the actual model decoder must confirm nonincreasing normalized pressure SSE.
    """
    _new_training_only(startup_mode)
    _assert_conditioned(model)
    if not callable(records_factory):
        raise ValueError("a repeatable records_factory is required for deployed-error verification")
    expected_ids = frozenset(train_ids)
    if not expected_ids:
        raise ValueError("declare a nonempty TRAIN-only calibration partition")
    if (not math.isfinite(energy_floor) or energy_floor <= 0
            or not math.isfinite(ridge_relative) or ridge_relative <= 0
            or type(spatial_chunk) is not int or spatial_chunk <= 0):
        raise ValueError("energy_floor, ridge_relative and spatial_chunk must be positive")
    start = perf_counter()
    fields = _output_fields(model.decoder)
    device = fields[0].device
    if qr_dtype is None:
        qr_dtype = torch.float32 if device.type == "cuda" else torch.float64
    if qr_dtype not in (torch.float32, torch.float64):
        raise ValueError("QR requires float32 or float64")
    n_times = len(model.full_times_s)
    if any(field.ndim != 2 or len(field) != n_times or field.device != device for field in fields):
        raise ValueError("all readout rows must share the registered time axis and device")
    originals = [field.detach().clone() for field in fields]
    anchor = torch.cat(originals, dim=1).T.contiguous().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(anchor).all():
        raise ValueError("the current effective affine rows must be finite")
    device_anchor = anchor.to(device, qr_dtype)
    combined = _ReducedQR(torch.device("cpu"))
    module_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    seen, first_pass = set(), {}
    before = 0.0
    floored = zero_source = 0
    report = None
    candidate_installed = accepted = False
    try:
        for item in records_factory():
            record_id, context, target = _validate_record(model, item, expected_ids, seen)
            features, source = context[:2]
            before_sse, energy = _record_error(model, context, target, spatial_chunk)
            denominator = max(energy, energy_floor)
            before += before_sse / denominator
            floored += int(energy < energy_floor)
            first_pass[record_id] = {
                "energy": energy, "source": source.detach().cpu().tolist(),
                "before": before_sse / denominator}
            if source[0, 4].item() == 0:
                zero_source += 1
                combined.orthogonal_sse += before_sse / denominator
                continue
            with torch.autocast(device_type=device.type, enabled=False):
                spatial = model.decoder.prepare_spatial(features.detach())[0][0].flatten(1)
                n_nodes = spatial.shape[1]
                if spatial.shape[0] != anchor.shape[0] - 1:
                    raise ValueError("native valid spatial basis and affine row dimensions disagree")
                gabor_space = model.gabor_spatial(features.detach()[:, :, 1:])[0].flatten(1)
                times = model._selected_times(None, 1)
                bank = model.gabor(times, source[:, 3:4], source[:, 2:3], model.duration_s)
                gabor_time = model.gabor_embedding(bank)[0].to(qr_dtype)
                carrier_rows, carrier_basis = model.propagation_factors(context, None)
                carrier_rows = carrier_rows[0].detach().to(qr_dtype)
                carrier_basis = carrier_basis[0, :, 1:].detach().flatten(1)
                active_target = target[:, 1:].flatten(1)
                local = _ReducedQR(device)
                for first in range(0, n_nodes, spatial_chunk):
                    block = slice(first, min(first + spatial_chunk, n_nodes))
                    basis = spatial[:, block].T.to(qr_dtype)
                    design = torch.cat((basis, torch.ones_like(basis[:, :1])), dim=1)
                    residual = active_target[:, block].T.to(device=device, dtype=qr_dtype)
                    residual = torch.addmm(residual, carrier_basis[:, block].T.to(qr_dtype),
                                           carrier_rows.T, alpha=-1.0)
                    residual = torch.addmm(residual, gabor_space[:, block].T.to(qr_dtype),
                                           gabor_time.T, alpha=-1.0)
                    residual = torch.addmm(residual, design, device_anchor, alpha=-1.0)
                    local.add(design, residual)
            scale = 1.0 / math.sqrt(denominator)
            combined.orthogonal_sse += local.orthogonal_sse.cpu() / denominator
            combined.add(local.r.to(device="cpu", dtype=torch.float64) * scale,
                         local.projected.to(device="cpu", dtype=torch.float64) * scale)
        if seen != expected_ids:
            raise ValueError(f"records_factory omitted {len(expected_ids - seen)} declared TRAIN records")
        count = len(seen)
        report = {
            "schema": SCHEMA, "objective": "mean_record_sse_over_original_full_sequence_energy",
            "records": count, "record_ids": sorted(map(str, seen)),
            "frames_per_record": n_times, "frames": count * n_times,
            "original_reference_energy_denominator": True,
            "fixed_terms_subtracted": ["current_propagation_carrier", "current_gabor_field"],
            "valid_detail_slots": "native prepare_spatial mask; zero-extended top row excluded",
            "energy_floor": float(energy_floor), "floored_records": floored,
            "zero_source_records": zero_source, "spatial_qr_dtype": str(qr_dtype),
            "ridge_relative": float(ridge_relative), "ridge_mu": None,
            "before": before / count, "after": before / count, "accepted": False,
        }
        if combined.r is not None:
            r, projected = combined.r, combined.projected
            n_columns = anchor.shape[0]
            mu = max(1e-12, ridge_relative * _squared_norm(r).item() / n_columns)
            ridge = torch.eye(n_columns, dtype=r.dtype) * math.sqrt(mu)
            q, upper = torch.linalg.qr(torch.cat((r, ridge)), mode="reduced")
            rhs = q[:r.shape[0]].T @ projected
            delta = torch.linalg.solve_triangular(upper, rhs, upper=True)
            proposed = (anchor + delta).T.split([field.shape[1] for field in fields], dim=1)
            candidates = [part.to(field) for part, field in zip(proposed, fields, strict=True)]
            deployed = torch.cat(
                [part.to(device="cpu", dtype=torch.float64) for part in candidates], dim=1).T
            predicted_after = (
                combined.orthogonal_sse + _squared_norm(r @ (deployed - anchor) - projected)).item() / count
            if not math.isfinite(predicted_after) or not torch.isfinite(deployed).all():
                raise FloatingPointError("the anchored residual fit produced non-finite rows")
            report.update(ridge_mu=mu, reduced_qr_after=predicted_after)
            candidate_installed = True
            for field, candidate in zip(fields, candidates, strict=True):
                field.copy_(candidate)
            seen_after, after = set(), 0.0
            for item in records_factory():
                record_id, context, target = _validate_record(model, item, expected_ids, seen_after)
                sse, energy = _record_error(model, context, target, spatial_chunk)
                original = first_pass[record_id]
                if (not math.isclose(energy, original["energy"], rel_tol=1e-12, abs_tol=0.0)
                        or context[1].detach().cpu().tolist() != original["source"]):
                    raise ValueError("records_factory changed the reference or source between passes")
                after += sse / max(original["energy"], energy_floor)
            if seen_after != expected_ids:
                raise ValueError("verification omitted declared TRAIN records")
            actual_after = after / count
            report["candidate_deployed_after"] = actual_after
            accepted = actual_after <= report["before"]
            if accepted:
                report.update(accepted=True, after=actual_after,
                              optimizer_states_cleared=_clear_moments(optimizer, _row_parameters(model)))
        report["elapsed_s"] = perf_counter() - start
        model.readout_conditioning_receipt["calibration"] = deepcopy(report)
        if hasattr(model, "initialization_receipt"):
            model.initialization_receipt["readout_conditioning"] = deepcopy(model.readout_conditioning_receipt)
        return report
    finally:
        if candidate_installed and not accepted:
            for field, original in zip(fields, originals, strict=True):
                field.copy_(original)
        for module, mode in module_modes:
            module.training = mode


def run_cpu_checks():
    """Exercise real joint pressure assembly, gradient coordinates and ridge fit on CPU.

    The synthetic fixture supplies cheap spatial fields and a nonzero carrier;
    pressure assembly and source-conditioned Gabor dispatch use JointFNOSCNO's
    actual methods. These checks do not establish training accuracy or speed.
    """
    from io import BytesIO
    from types import SimpleNamespace
    from torch import nn
    from torch.nn import functional as F
    from joint_fno_model import JointFNOSCNO

    class Bank(nn.Module):
        def forward(self, times, delay, frequency, duration):
            phase = (times - delay) * frequency * (2 * math.pi)
            return torch.stack((phase.sin(), phase.cos(), times / duration), dim=-1)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Conv2d(4, 4, 1), nn.SiLU(), nn.Conv2d(4, 7, 1))
            self.detail_time = nn.ModuleList(nn.Linear(2, 7, bias=False) for _ in range(3))
            self.register_buffer("native_times", torch.linspace(0, 1, 7))

        def prepare_spatial(self, features, *, detail_enabled=True):
            active = features[:, :, 1:]
            base = self.proj[1](self.proj[0](active))
            packets = (active[:, :2], active[:, 1:3].square(), active[:, 2:].sin())
            return (torch.cat((base, *packets), dim=1), packets) if detail_enabled else (base, ())

    class Fixture(nn.Module):
        _selected_times = JointFNOSCNO._selected_times
        _project_once = JointFNOSCNO._project_once
        decode = JointFNOSCNO.decode

        def __init__(self):
            super().__init__()
            self.decoder = Decoder()
            self.feature_projection = nn.Conv2d(3, 4, 1)
            self.gabor = Bank()
            self.gabor_embedding = nn.Linear(3, 2)
            self.gabor_spatial = nn.Conv2d(4, 2, 1)
            self.carrier_spatial = nn.Conv2d(4, 2, 1)
            self.carrier_rows = nn.Parameter(torch.randn(1, 7, 2) * 2)
            self.residual_gain = nn.Parameter(torch.tensor(0.01953))
            self.duration_s = 1.0
            self.initialization_receipt = {}
            self.normalizer = SimpleNamespace(metadata=SimpleNamespace(pressure_scale_pa=3.0))

        @property
        def full_times_s(self):
            return self.decoder.native_times

        @property
        def device(self):
            return self.full_times_s.device

        def encode(self, value, source):
            features = self.feature_projection(value)
            carrier = F.pad(self.carrier_spatial(features[:, :, 1:]), (0, 0, 1, 0))
            return features, source, (self.carrier_rows, carrier)

        def propagation_factors(self, context, slots):
            rows, basis = context[2]
            return (rows if slots is None else rows[:, slots]), basis

        def checkpoint_payload(self):
            return {"model_semantics": {"variant": "cpu_check_fixture"}, "model": self.state_dict()}

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(310719)
        model = Fixture().double()
        raw = torch.randn(1, 3, 201, 201, dtype=torch.float64, requires_grad=True)
        source = torch.tensor([[1000., 1000., 2.5, 0.08, 1.]], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for parameter in model.parameters():
            optimizer.state[parameter] = {
                "step": torch.tensor(1.), "exp_avg": torch.ones_like(parameter),
                "exp_avg_sq": torch.ones_like(parameter)}
        changed = _row_parameters(model)
        changed["gabor_embedding.weight"] = model.gabor_embedding.weight
        changed["gabor_embedding.bias"] = model.gabor_embedding.bias
        row_parameter = model.decoder.proj[-1].weight
        upstream = model.feature_projection.weight
        before_prediction = model.decode(model.encode(raw, source))
        before_gradients = torch.autograd.grad(
            before_prediction.square().mean(), (raw, source, upstream, row_parameter))
        physical_before = model.decode(
            model.encode(raw.detach(), source.detach()), torch.tensor([1, 4]), normalized=False).detach()
        gamma = model.residual_gain.detach().item()
        receipt = absorb_neural_gain(model, startup_mode="warm_start", optimizer=optimizer)
        after_prediction = model.decode(model.encode(raw, source))
        after_gradients = torch.autograd.grad(
            after_prediction.square().mean(), (raw, source, upstream, row_parameter))
        pressure_difference = (before_prediction - after_prediction).abs().max().item()
        assert torch.allclose(before_prediction, after_prediction, rtol=1e-12, atol=1e-12)
        for old, new in zip(before_gradients[:3], after_gradients[:3], strict=True):
            assert torch.allclose(old, new, rtol=1e-11, atol=1e-12)
        assert torch.allclose(before_gradients[3], after_gradients[3] * gamma, rtol=1e-11, atol=1e-12)
        assert torch.allclose(physical_before, model.decode(
            model.encode(raw.detach(), source.detach()), torch.tensor([1, 4]), normalized=False),
            rtol=1e-12, atol=1e-12)
        assert all(parameter not in optimizer.state for parameter in changed.values())
        assert upstream in optimizer.state
        assert model.residual_gain not in optimizer.state and not model.residual_gain.requires_grad
        state_before_resume = {name: value.clone() for name, value in model.state_dict().items()}
        try:
            absorb_neural_gain(model, startup_mode="resume", optimizer=optimizer)
        except ValueError:
            pass
        else:
            raise AssertionError("true resume allowed gain absorption")
        assert all(torch.equal(state_before_resume[name], value) for name, value in model.state_dict().items())
        teacher = deepcopy(model)
        with torch.no_grad():
            for parameter in _row_parameters(teacher).values():
                parameter.add_(0.2 * torch.randn_like(parameter))
        examples = [
            (f"train-{i}", (i + 1) * torch.randn(1, 3, 201, 201, dtype=torch.float64),
             source.detach().clone()) for i in range(2)]
        with torch.no_grad():
            targets = {name: teacher.decode(teacher.encode(value, src))[0].detach()
                       for name, value, src in examples}

        def records_factory():
            for name, value, src in examples:
                yield name, model.encode(value, src), targets[name]

        # Independent augmented least-squares reference with the ORIGINAL
        # target energy; nonzero carrier and Gabor must both be subtracted.
        anchor = torch.cat([field.detach().clone() for field in _output_fields(model.decoder)], dim=1).T
        designs, residuals = [], []
        with torch.no_grad():
            for name, value, src in examples:
                context = model.encode(value, src)
                spatial = model.decoder.prepare_spatial(context[0])[0][0].flatten(1).T
                design = torch.cat((spatial, torch.ones_like(spatial[:, :1])), dim=1)
                truth = targets[name][:, 1:].flatten(1).T
                current = model.decode(context)[0, :, 1:].flatten(1).T
                denominator = truth.square().sum().sqrt()
                designs.append(design / denominator)
                residuals.append((truth - current) / denominator)
        design, residual = torch.cat(designs), torch.cat(residuals)
        mu = max(1e-12, 1e-4 * design.square().sum().item() / design.shape[1])
        augmented = torch.cat((design, math.sqrt(mu) * torch.eye(design.shape[1], dtype=torch.float64)))
        rhs = torch.cat((residual, torch.zeros(design.shape[1], residual.shape[1], dtype=torch.float64)))
        independent = anchor + torch.linalg.lstsq(augmented, rhs, driver="gels").solution
        fixed = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                 if parameter not in set(_row_parameters(model).values())}
        report = calibrate_residual_time_rows(
            model, records_factory, train_ids=[row[0] for row in examples],
            startup_mode="warm_start", optimizer=optimizer, spatial_chunk=4096)
        deployed = torch.cat([field.detach() for field in _output_fields(model.decoder)], dim=1).T
        assert report["accepted"] and report["after"] < report["before"]
        assert torch.allclose(deployed, independent, rtol=1e-8, atol=1e-9)
        assert all(torch.equal(parameter, fixed[name]) for name, parameter in model.named_parameters() if name in fixed)
        snapshot = {name: value.clone() for name, value in model.state_dict().items()}
        try:
            calibrate_residual_time_rows(
                model, records_factory, train_ids=["not-a-training-record"], startup_mode="warm_start")
        except ValueError:
            pass
        else:
            raise AssertionError("a non-TRAIN record entered calibration")
        assert all(torch.equal(snapshot[name], value) for name, value in model.state_dict().items())
        stream = BytesIO()
        torch.save(conditioning_checkpoint_payload(model), stream)
        stream.seek(0)
        payload = torch.load(stream, weights_only=True)
        restored = Fixture().double()
        restored.load_state_dict(payload["model"], strict=True)
        restored.residual_gain.requires_grad_(False)
        restored.readout_conditioning_receipt = payload["readout_conditioning_receipt"]
        _assert_conditioned(restored)
        first = examples[0]
        assert torch.equal(model.decode(model.encode(first[1], first[2])),
                           restored.decode(restored.encode(first[1], first[2])))
        return {
            "passed": True, "pressure_max_abs_change": pressure_difference,
            "input_source_upstream_gradient_preserved": True,
            "output_gradient_coordinate_relation_preserved": True,
            "physical_selected_rows_preserved": True,
            "optimizer_state_reset_is_selective": True,
            "resume_and_non_train_record_blocked": True,
            "serialized_predictions_exact": True,
            "calibration_matches_independent_weighted_lstsq": True,
            "calibration_before": report["before"], "calibration_after": report["after"],
            "gain_receipt": receipt,
        }


if __name__ == "__main__":
    import json
    import sys
    if sys.argv[1:] == ["--cpu-check"]:
        torch.set_num_threads(2)
        print(json.dumps(run_cpu_checks(), indent=2))
    else:
        print("Import for an explicit new-training transition; use --cpu-check for CPU checks.")
