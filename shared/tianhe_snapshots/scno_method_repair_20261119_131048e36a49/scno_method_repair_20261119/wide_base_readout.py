"""Function-preserving expansion of the joint SCNO BASE spatial readout.

UL: input -> 128 -> GELU -> T becomes input -> 800 -> GELU -> T.
Marmousi: input -> 89 -> GELU -> 67 -> GELU -> T widens only 67.
The old hidden slice and old final time columns are copied exactly. New hidden
rows are Xavier/nonzero; their final columns start at zero. This changes
capacity without changing the initial pressure function. No accuracy or timing
improvement is asserted.

The containing model owner must save base_readout_semantics and the expansion
receipt and allocate the matching BASE width before its strict model-state
load. The component restore below does that without a new-training transition.
Optimizer state for replaced Parameter objects must be rebuilt; true resume
restores the existing expanded architecture and saved optimizer/RNG/stage state.
"""

from copy import deepcopy
import hashlib
import io
import json

import torch
from torch import nn

from joint_fno_decoder import JointFNOTimeDecoder


SCHEMA = "SCNO_JOINT_BASE_READOUT_EXPANSION_V1"
_PROFILES = {"uniform_layered": (128,), "marmousi": (89, 67)}


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def base_readout_semantics(decoder):
    """Inspect actual modules, activation identity, native axis and BASE roles."""
    if type(decoder) is not JointFNOTimeDecoder or decoder.profile not in _PROFILES:
        raise TypeError("expected the reviewed JointFNOTimeDecoder computation")
    widths = _PROFILES[decoder.profile]
    projection = decoder.proj
    if type(projection) is not nn.Sequential or len(projection) != len(widths) * 2 + 1:
        raise ValueError("the BASE projection is not the recorded Conv/GELU sequence")
    convs = []
    for index, module in enumerate(projection):
        if index % 2:
            if type(module) is not nn.GELU:
                raise ValueError("preserve the recorded GELU activation")
            continue
        if (type(module) is not nn.Conv2d or module.kernel_size != (1, 1)
                or module.stride != (1, 1) or module.padding != (0, 0)
                or module.dilation != (1, 1) or module.groups != 1
                or module.padding_mode != "zeros" or module.bias is None):
            raise ValueError("expand only the reviewed biased one-by-one BASE convolutions")
        convs.append(module)
    if convs[0].in_channels != decoder.in_channels:
        raise ValueError("BASE input channels differ from the conditioning features")
    if any(left.out_channels != right.in_channels for left, right in zip(convs, convs[1:])):
        raise ValueError("BASE hidden and output channels do not compose")
    if len(widths) > 1 and convs[0].out_channels != widths[0]:
        raise ValueError("the earlier Marmousi hidden layer must retain its recorded width")
    axis = decoder.native_times
    if (axis.ndim != 1 or not len(axis) or not torch.isfinite(axis).all()
            or not (axis[1:] > axis[:-1]).all() or convs[-1].out_channels != len(axis)):
        raise ValueError("the BASE output must retain every registered physical time")
    hidden_index, output_index = len(projection) - 3, len(projection) - 1
    return {
        "schema": SCHEMA, "decoder_class": "joint_fno_decoder.JointFNOTimeDecoder",
        "profile": decoder.profile, "input_channels": decoder.in_channels,
        "base_hidden_width": convs[-2].out_channels,
        "preceding_hidden_widths": [layer.out_channels for layer in convs[:-2]],
        "hidden_layer_index": hidden_index, "time_readout_index": output_index,
        "activations": [
            {"index": index, "type": "GELU", "approximate": projection[index].approximate}
            for index in range(1, len(projection), 2)],
        "native_frames": len(axis),
        "native_axis_f32_sha256": hashlib.sha256(
            axis.detach().cpu().float().numpy().astype("<f4").tobytes()).hexdigest(),
        "active_grid": list(decoder.detail_transform.shape) if decoder.detail_transform is not None else None,
        "detail_latent": int(decoder.detail_latent),
        "role": "BASE", "parameterization": "affine_hidden_GELU_affine_native_time_rows",
    }


def base_parameter_roles(decoder, *, prefix=""):
    """Return semantic names, not a decision to unfreeze permanently fixed parameters."""
    base, detail = [], []
    for name, _ in decoder.named_parameters():
        (base if name.startswith("proj.") else detail).append(prefix + name)
    return {"BASE": base, "DETAIL": detail}


def _check_external(identity, semantics):
    if semantics is None:
        return
    checks = {
        "profile": identity["profile"], "output_frames": identity["native_frames"],
        "detail_latent": identity["detail_latent"],
        "native_axis_sha256": identity["native_axis_f32_sha256"],
    }
    if identity["active_grid"] is not None:
        checks["active_grid"] = identity["active_grid"]
    for name, expected in checks.items():
        if name not in semantics or semantics[name] != expected:
            raise ValueError(f"BASE checkpoint semantic mismatch: {name}")
    if "base_hidden_width" in semantics and semantics["base_hidden_width"] != identity["base_hidden_width"]:
        raise ValueError("source base_hidden_width metadata contradicts its actual module")


def _allocate_pair(decoder, width):
    """Allocate uninitialized, matching layers without consuming CPU/CUDA RNG."""
    hidden, final = decoder.proj[-3], decoder.proj[-1]
    expanded_hidden = nn.Conv2d(
        hidden.in_channels, width, 1, bias=True, device="meta",
        dtype=hidden.weight.dtype).to_empty(device=hidden.weight.device)
    expanded_final = nn.Conv2d(
        width, final.out_channels, 1, bias=True, device="meta",
        dtype=final.weight.dtype).to_empty(device=final.weight.device)
    for candidate, original in ((expanded_hidden, hidden), (expanded_final, final)):
        candidate.weight.requires_grad_(original.weight.requires_grad)
        candidate.bias.requires_grad_(original.bias.requires_grad)
        candidate.train(original.training)
    return expanded_hidden, expanded_final


def expand_base_readout(
        decoder, hidden_width=800, *, seed=261119,
        source_checkpoint_semantics=None, startup="new_training", inplace=False):
    """Return candidate_decoder, receipt, preserving all old pressure paths.

    Newly added output columns can receive gradients on the first update.
    Newly added spatial rows initially receive zero gradient because their
    output columns are zero; they receive gradients after those columns move.
    This is intentional function-preserving growth, not a symmetric zero init.
    """
    if startup != "new_training":
        raise ValueError("BASE expansion is a new-training transition; true resume restores saved expanded state")
    if type(hidden_width) is not int or hidden_width < 1:
        raise ValueError("hidden_width must be a positive integer")
    before = base_readout_semantics(decoder)
    _check_external(before, source_checkpoint_semantics)
    old_width = before["base_hidden_width"]
    if hidden_width < old_width:
        raise ValueError("this operation grows BASE capacity and never removes trained hidden features")
    candidate = decoder if inplace else deepcopy(decoder)
    receipt = {
        "schema": SCHEMA, "operation": "new_training_base_hidden_expansion",
        "seed": int(seed), "source_identity": before,
        "source_checkpoint_semantics": deepcopy(source_checkpoint_semantics),
        "old_hidden_width": old_width, "new_hidden_width": hidden_width,
        "old_hidden_rows": [0, old_width], "old_time_columns": [0, old_width],
        "new_initialization": {
            "hidden_weight": "xavier_uniform_gain_1_cpu_seeded", "new_hidden_bias": "zero",
            "new_time_columns": "zero", "time_bias": "copied"},
        "optimizer_action": "rebuild replaced Parameter states; true resume restores the saved optimizer",
        "role": "BASE",
        "function_preservation": "copied old hidden rows and old time columns; added time columns zero",
        "gradient_schedule": "new time columns first; new hidden rows after time columns update",
        "unchanged": ["earlier hidden layers", "GELU activations", "physical time axis",
                      "trained old hidden rows and time columns", "all detail branches and fixed transforms"],
    }
    if hidden_width == old_width:
        receipt["operation"] = "identity_no_base_growth"
        receipt["optimizer_action"] = "no parameter replacement"
    else:
        hidden, final = candidate.proj[-3], candidate.proj[-1]
        new_hidden, new_final = _allocate_pair(candidate, hidden_width)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        fresh = torch.empty(new_hidden.weight.shape, dtype=torch.float32, device="cpu")
        nn.init.xavier_uniform_(fresh, gain=1.0, generator=generator)
        with torch.no_grad():
            new_hidden.weight.copy_(fresh.to(new_hidden.weight))
            new_hidden.bias.zero_()
            new_hidden.weight[:old_width].copy_(hidden.weight)
            new_hidden.bias[:old_width].copy_(hidden.bias)
            new_final.weight.zero_()
            new_final.weight[:, :old_width].copy_(final.weight)
            new_final.bias.copy_(final.bias)
        receipt.update(
            old_hidden_weight_sha256=_tensor_hash(hidden.weight),
            copied_hidden_weight_sha256=_tensor_hash(new_hidden.weight[:old_width]),
            old_hidden_bias_sha256=_tensor_hash(hidden.bias),
            copied_hidden_bias_sha256=_tensor_hash(new_hidden.bias[:old_width]),
            old_time_weight_sha256=_tensor_hash(final.weight),
            copied_time_weight_sha256=_tensor_hash(new_final.weight[:, :old_width]),
            old_time_bias_sha256=_tensor_hash(final.bias),
            copied_time_bias_sha256=_tensor_hash(new_final.bias),
            new_hidden_nonzero=bool(torch.count_nonzero(new_hidden.weight[old_width:])),
            new_time_columns_zero=bool(torch.count_nonzero(new_final.weight[:, old_width:]) == 0),
        )
        candidate.proj[-3], candidate.proj[-1] = new_hidden, new_final
    candidate.base_hidden_width = hidden_width
    receipt["target_identity"] = base_readout_semantics(candidate)
    receipt["parameter_roles"] = base_parameter_roles(candidate, prefix="decoder.")
    candidate.base_expansion_receipt = deepcopy(receipt)
    return candidate, receipt


def base_decoder_checkpoint_payload(decoder, *, model_semantics=None):
    """Independent component payload. The owner additionally saves training state."""
    return {
        "schema": SCHEMA, "component": "joint_base_readout",
        "identity": base_readout_semantics(decoder),
        "model_semantics": deepcopy(model_semantics),
        "expansion_receipt": deepcopy(getattr(decoder, "base_expansion_receipt", None)),
        "parameter_roles": base_parameter_roles(decoder),
        "requires_grad": {name: parameter.requires_grad for name, parameter in decoder.named_parameters()},
        "module_training_flags": {name: module.training for name, module in decoder.named_modules()},
        "state_dict": {name: value.detach().cpu().clone() for name, value in decoder.state_dict().items()},
        "training_state_complete": False,
    }


def restore_base_decoder_checkpoint(template, payload, *, expected_model_semantics=None):
    """Strict component restore; no expansion initialization, fitting or calibration.

    Existing DETAIL widths must match the saved computation. Restore or allocate
    its time/detail architecture separately before calling this BASE helper.
    """
    if payload.get("schema") != SCHEMA or payload.get("component") != "joint_base_readout":
        raise ValueError("expected a separately versioned BASE readout component")
    if expected_model_semantics is not None and payload.get("model_semantics") != expected_model_semantics:
        raise ValueError("enclosing medium, source, grid or normalization semantics differ")
    source, saved_identity = base_readout_semantics(template), payload["identity"]
    for key in source:
        if key != "base_hidden_width" and source[key] != saved_identity.get(key):
            raise ValueError(f"saved BASE computation differs from template: {key}")
    width = saved_identity["base_hidden_width"]
    if type(width) is not int or width < 1:
        raise ValueError("saved BASE width is invalid")
    candidate = deepcopy(template)
    if source["base_hidden_width"] != width:
        candidate.proj[-3], candidate.proj[-1] = _allocate_pair(candidate, width)
    current, saved = candidate.state_dict(), payload["state_dict"]
    if current.keys() != saved.keys():
        raise ValueError("saved decoder has missing or unexpected component state")
    for name, value in current.items():
        target = saved[name]
        if not torch.is_tensor(target) or target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError(f"saved BASE component tensor is incompatible: {name}")
    candidate.load_state_dict(saved, strict=True)
    if base_readout_semantics(candidate) != saved_identity:
        raise ValueError("saved tensors contradict the recorded BASE semantic identity")
    parameters = dict(candidate.named_parameters())
    modules = dict(candidate.named_modules())
    if parameters.keys() != payload["requires_grad"].keys() or modules.keys() != payload["module_training_flags"].keys():
        raise ValueError("saved trainability or module-mode layout differs from the component")
    for name, parameter in parameters.items():
        parameter.requires_grad_(bool(payload["requires_grad"][name]))
    for name, module in modules.items():
        module.training = bool(payload["module_training_flags"][name])
    candidate.base_hidden_width = width
    candidate.base_expansion_receipt = deepcopy(payload.get("expansion_receipt"))
    if base_parameter_roles(candidate) != payload["parameter_roles"]:
        raise ValueError("saved BASE/DETAIL role attribution changed")
    return candidate


def run_cpu_checks():
    """Small-grid function/gradient/resume checks using the actual decoder."""
    results = []
    for profile in ("uniform_layered", "marmousi"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19817)
            original = JointFNOTimeDecoder(
                136, torch.linspace(0, 1, 11), profile=profile).double()
            features = torch.randn(2, 136, 9, 11, dtype=torch.float64, requires_grad=True)
            old_prediction = original(features)
            old_derivative = torch.autograd.grad(old_prediction.square().sum(), features)[0]
            old_state = {name: value.clone() for name, value in original.state_dict().items()}
            rng = torch.get_rng_state().clone()
            expanded, receipt = expand_base_readout(original, 800)
            assert torch.equal(rng, torch.get_rng_state())
            assert original.proj[-3].out_channels == _PROFILES[profile][-1]
            assert all(torch.equal(value, old_state[name]) for name, value in original.state_dict().items())
            new_prediction = expanded(features)
            new_derivative = torch.autograd.grad(new_prediction.square().sum(), features)[0]
            assert torch.allclose(old_prediction, new_prediction, atol=1e-12, rtol=1e-12)
            assert torch.allclose(old_derivative, new_derivative, atol=1e-11, rtol=1e-11)
            indices = torch.tensor([[1, 2, 9], [0, 7, 10]])
            assert torch.allclose(original(features, slots=indices), expanded(features, slots=indices),
                                  atol=1e-12, rtol=1e-12)
            assert expanded.proj[-2].approximate == original.proj[-2].approximate
            if profile == "marmousi":
                assert all(torch.equal(a, b) for a, b in zip(
                    original.proj[0].parameters(), expanded.proj[0].parameters(), strict=True))
            old_width = _PROFILES[profile][-1]
            hidden, final = expanded.proj[-3], expanded.proj[-1]
            target = torch.randn_like(new_prediction)
            expanded.zero_grad(set_to_none=True)
            loss = (expanded(features) - target).square().mean()
            loss.backward()
            first_time_grad = final.weight.grad[:, old_width:].norm().item()
            first_hidden_grad = hidden.weight.grad[old_width:].norm().item()
            assert first_time_grad > 0 and first_hidden_grad == 0
            with torch.no_grad():
                final.weight[:, old_width:].add_(final.weight.grad[:, old_width:], alpha=-.1)
            expanded.zero_grad(set_to_none=True)
            (expanded(features) - target).square().mean().backward()
            next_hidden_grad = hidden.weight.grad[old_width:].norm().item()
            assert next_hidden_grad > 0 and final.weight.grad[:, old_width:].norm().item() > 0
            # Check exact storage, freeze flags, mode and RNG preservation on restore.
            hidden.bias.requires_grad_(False)
            expanded.eval()
            stream = io.BytesIO()
            torch.save(base_decoder_checkpoint_payload(
                expanded, model_semantics={"scope": "cpu_test", "profile": profile}), stream)
            stream.seek(0)
            payload = torch.load(stream, weights_only=True)
            rng = torch.get_rng_state().clone()
            restored = restore_base_decoder_checkpoint(
                original, payload, expected_model_semantics={"scope": "cpu_test", "profile": profile})
            assert torch.equal(rng, torch.get_rng_state())
            assert torch.equal(expanded(features), restored(features))
            assert not restored.proj[-3].bias.requires_grad and not restored.training
            snapshot = {name: value.clone() for name, value in expanded.state_dict().items()}
            try:
                expand_base_readout(expanded, 800, startup="resume")
            except ValueError:
                pass
            else:
                raise AssertionError("true resume accepted a new-training expansion")
            assert all(torch.equal(snapshot[name], value) for name, value in expanded.state_dict().items())
            results.append({
                "profile": profile, "old_width": old_width, "new_width": 800,
                "pressure_max_abs_difference": (old_prediction - new_prediction).abs().max().item(),
                "input_derivative_max_abs_difference": (old_derivative - new_derivative).abs().max().item(),
                "first_new_time_gradient_norm": first_time_grad,
                "first_new_hidden_gradient_norm": first_hidden_grad,
                "next_new_hidden_gradient_norm": next_hidden_grad,
                "earlier_hidden_and_original_state_preserved": True,
                "save_reload_predictions_exact": True, "restore_rng_unchanged": True,
                "resume_expansion_rejected": True,
            })
    return {"passed": True, "checks": results}


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--cpu-check"]:
        torch.set_num_threads(2)
        print(json.dumps(run_cpu_checks(), indent=2))
    else:
        print("Import expand_base_readout; --cpu-check runs CPU-only checks.")
