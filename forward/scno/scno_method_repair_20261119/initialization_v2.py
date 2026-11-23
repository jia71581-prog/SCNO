"""SCNO warm starts with explicit model semantics and a tensor-level receipt."""

import hashlib
import json

import torch
from torch import nn

from lora_v2 import ADAPTER_KEYS, adapter_prefix, place_saved_adapters


DETAIL_PREFIX = "refiner.spectral_detail."
INITIALIZATION_SCHEMA = "scno_v2_residual_start_v1"
HEADS_PREFIX = DETAIL_PREFIX + "heads."
TRAINING_SCHEMA = "scno_training_v2"
LEGACY_DECODER_GRID_SIZE = 101


def _decoder_grid_size(value):
    if type(value) is not int or value not in (LEGACY_DECODER_GRID_SIZE, 201):
        raise ValueError("SCNO decoder_grid_size must be 101 or 201")
    return value


def _normalized_semantics(receipt):
    if not isinstance(receipt, dict):
        raise ValueError("model_semantics must be a mapping")
    grid = _decoder_grid_size(receipt.get("decoder_grid_size", LEGACY_DECODER_GRID_SIZE))
    return {**receipt, "decoder_grid_size": grid}


def semantics(model, dataset):
    return {
        "variant": "scno_v2_shared_gabor_orthogonal_pressure",
        "family": model.family,
        "model_spec_sha256": model.initialization_receipt["model_spec_sha256"],
        "physical_axis_sha256": dataset.binding["actual_times_sha256"],
        "normalization_sha256": dataset.binding["normalization_sha256"],
        "pressure_readout": "normalized_pa_output_exact_zero_top_local_polynomial_200x201",
        "decoder_grid_size": _decoder_grid_size(getattr(model, "decoder_grid_size", LEGACY_DECODER_GRID_SIZE)),
    }


def checkpoint_decoder_grid_size(payload):
    """Read the decoder grid independently of tensor shapes and the current model."""
    receipt = payload.get("model_semantics")
    declared = (_normalized_semantics(receipt)["decoder_grid_size"]
                if receipt is not None else LEGACY_DECODER_GRID_SIZE)
    if payload.get("schema") == TRAINING_SCHEMA:
        config = payload.get("binding", {}).get("config", {})
        model = config.get("model", {})
        if "decoder_grid_size" in model or config.get("training_version") == "scno_v2":
            bound = _decoder_grid_size(model.get("decoder_grid_size", LEGACY_DECODER_GRID_SIZE))
            if bound != declared:
                raise ValueError("checkpoint model semantics and training configuration declare different decoder grids")
    return declared


def checkpoint_semantics(payload, target):
    grid = checkpoint_decoder_grid_size(payload)
    if payload.get("model_semantics") is not None:
        return {**payload["model_semantics"], "decoder_grid_size": grid}
    binding = payload.get("binding", {})
    if payload.get("schema") != TRAINING_SCHEMA or binding.get("config", {}).get("training_version") != "scno_v2":
        return None
    config, dataset = binding["config"], binding.get("dataset", {})
    family = config.get("model", {}).get("family")
    if family is None:
        family = "layered" if dataset.get("kind") == "centered_uniform_layered" else "marmousi"
    # The retired V2 writer used the same sealed runtime and state layout.
    return {**target, "family": family, "physical_axis_sha256": dataset.get("actual_times_sha256"),
            "normalization_sha256": dataset.get("normalization_sha256"), "decoder_grid_size": grid}


def validate_resume_semantics(payload, target):
    """Check model semantics in addition to the caller's complete resume binding."""
    source = checkpoint_semantics(payload, target)
    if (source is None and payload.get("schema") == TRAINING_SCHEMA
            and checkpoint_decoder_grid_size(payload) == target["decoder_grid_size"] == LEGACY_DECODER_GRID_SIZE):
        return
    if source != target:
        raise ValueError("checkpoint model semantics differ from the current runtime")


def check_tensor(old, new):
    return isinstance(old, torch.Tensor) and old.shape == new.shape and old.dtype == new.dtype


def same_fixed_tensor(name, old, new, fixed):
    return name not in fixed or torch.equal(old.detach().cpu(), new.detach().cpu())


def initialize_run_weights(model, dataset, path=None, manifest=None):
    target = semantics(model, dataset)
    if manifest is not None and path is None:
        raise ValueError("a semantic manifest is bound to a specific warm-start checkpoint")
    result = {"schema": INITIALIZATION_SCHEMA, "mode": "fresh" if path is None else "warm_start",
              "model_semantics": target, "loaded": [], "missing": [], "incompatible": {},
              "xavier_initialized": [], "pending_output_layers": []}
    current = model.state_dict()
    state, payload = {}, {}
    if path is not None:
        path = path.resolve(strict=True)
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("model", payload)
        if not isinstance(state, dict):
            raise ValueError("a warm start requires a tensor state dictionary")
        result.update(path=str(path), sha256=digest,
                      source_completed_updates=payload.get("completed_updates", payload.get("next_update")))
        evidence = None
        if manifest is not None:
            evidence = json.loads(manifest.read_text(encoding="utf-8"))
            if evidence.get("checkpoint_sha256") != digest:
                raise ValueError("the warm-start semantic manifest has a different checkpoint digest")
        source, rejection = None, None
        try:
            source = checkpoint_semantics(payload, target)
            known_grid = source["decoder_grid_size"] if source is not None else checkpoint_decoder_grid_size(payload)
            if evidence is not None:
                claimed = _normalized_semantics(evidence.get("model_semantics"))
                result["manifest_model_semantics"] = claimed
                if claimed["decoder_grid_size"] != known_grid:
                    raise ValueError("the warm-start manifest cannot change the checkpoint decoder grid")
                source = claimed
        except ValueError as error:
            rejection = str(error)
        result["source_model_semantics"] = source
        if rejection is None and source != target:
            if known_grid != target["decoder_grid_size"]:
                old, new = known_grid, target["decoder_grid_size"]
                rejection = f"checkpoint decoder grid {old} x {old} differs from target {new} x {new}"
            else:
                rejection = "unverified or different model, physical axis or normalization"
        if rejection is not None:
            result["incompatible"] = {key: rejection
                                      for key, value in state.items() if isinstance(value, torch.Tensor)}
            result["missing"] = sorted(current.keys() - state.keys())
            result["rejected"] = True
            return result
        for index in range(len(model.detail_output_layers)):
            prefix = adapter_prefix(index)
            if any(prefix + key in state for key in ADAPTER_KEYS):
                base = current[prefix + "weight"]
                expected = {"base.weight": base, "base.bias": current[prefix + "bias"],
                            "U": base.new_empty((32, 4)), "V": base.new_empty((32, 4)),
                            "scaling": base.new_empty(())}
                bad = [key for key in ADAPTER_KEYS if not check_tensor(state.get(prefix + key), expected[key])]
                if bad or not torch.isfinite(state[prefix + "scaling"]) or state[prefix + "scaling"] <= 0:
                    result["incompatible"][prefix] = "incomplete or incompatible compensated rank-four module"
                    result["rejected"] = True
                    return result
        place_saved_adapters(model, state)
        current = model.state_dict()

    fixed = {key for key in current if key.startswith(DETAIL_PREFIX + "transform.")
             or key.startswith("base.coordinate_encoder.gabor.")
             or key == "base.coordinate_encoder.fourier_frequencies"}
    loaded = {}
    for name, value in current.items():
        if name not in state:
            result["missing"].append(name)
        elif not check_tensor(state[name], value):
            result["incompatible"][name] = "tensor dtype or shape differs"
        elif not same_fixed_tensor(name, state[name], value, fixed):
            result["incompatible"][name] = "fixed transform or time basis differs"
        else:
            loaded[name] = state[name]
    result["incompatible"].update({key: "absent from the target model"
                                   for key in sorted(state.keys() - current.keys())
                                   if isinstance(state[key], torch.Tensor)})
    result["loaded"] = sorted(loaded)
    if any(name in result["incompatible"] for name in fixed):
        result["rejected"] = True
        return result

    previous = payload.get("initializer") or {}
    from_phase_a = payload.get("phase") == "A" and payload.get("schema") == TRAINING_SCHEMA
    pending = set(previous.get("pending_output_layers", ())) if from_phase_a else set()
    for index in range(len(model.detail_output_layers)):
        prefix = f"{HEADS_PREFIX}{index}.4."
        found = [prefix + key in loaded for key in ("weight", "bias")]
        if any(found) and not all(found):
            result["incompatible"][prefix] = "only part of the pressure readout is compatible"
            result["rejected"] = True
            return result
        if not any(found):
            pending.add(index)
        elif from_phase_a and previous.get("schema") != INITIALIZATION_SCHEMA:
            # In the retired trainer a phase-A checkpoint stored untrained, exactly zero detail readouts.
            if all(loaded[prefix + key].count_nonzero().item() == 0 for key in ("weight", "bias")):
                pending.add(index)
    result["pending_output_layers"] = sorted(pending)
    model.load_state_dict({**current, **loaded}, strict=True)
    result["xavier_initialized"] = initialize_missing_features(model, loaded)
    return result


@torch.no_grad()
def initialize_missing_features(model, loaded):
    params = dict(model.named_parameters())
    branch = model.refiner.spectral_detail
    layers = [branch.fused_projection, branch.refined_projection, branch.source_projection, branch.time_projection]
    layers.extend(head[index] for head in branch.heads for index in (0, 2) if isinstance(head[index], nn.Linear))
    names = {id(parameter): name for name, parameter in params.items()}
    initialized = []
    for layer in layers:
        for key, parameter in layer.named_parameters(recurse=False):
            name = names[id(parameter)]
            if name not in loaded:
                if key == "weight":
                    nn.init.xavier_uniform_(parameter, gain=1.)
                else:
                    nn.init.zeros_(parameter)
                initialized.append(name)
    return sorted(initialized)
