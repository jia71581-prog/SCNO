"""Fresh-graph full-model fitting with an unbiased four-stratum time estimator.

Each loss calls the real model forward again. Updating the encoder or BASE
therefore changes the current context/base on the next step. Direct pressure
error equals correction - (reference-current_base), with the BASE gradient
retained; no detached or stale residual target enters this objective.
"""

from copy import deepcopy
import hashlib
import json
import math
from typing import NamedTuple

import torch

from repair_stage_schedule import capture_rng, restore_rng


SCHEMA = "scno_shared_full_model_stratified_resume_v1"


class FrameSample(NamedTuple):
    indices: torch.Tensor
    weights: torch.Tensor


class FourStratumFrameSampler:
    """Independent uniform samples without replacement within each time stratum."""

    def __init__(self, full_frames=801, samples_per_stratum=64, *, seed=261119):
        if (type(full_frames) is not int or type(samples_per_stratum) is not int
                or min(full_frames, samples_per_stratum) < 1 or full_frames // 4 < samples_per_stratum):
            raise ValueError("each of the four contiguous strata must contain enough distinct frame rows")
        self.full_frames = full_frames
        self.samples_per_stratum = samples_per_stratum
        quotient, remainder = divmod(full_frames, 4)
        self.sizes = tuple(quotient + int(i < remainder) for i in range(4))
        self.starts = (0, self.sizes[0], sum(self.sizes[:2]), sum(self.sizes[:3]))
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.drawn_records, self.sample_calls = 0, 0

    @property
    def semantics(self):
        return {"kind": "four_contiguous_time_strata_uniform_without_replacement",
                "full_frames": self.full_frames, "samples_per_stratum": self.samples_per_stratum,
                "stratum_sizes": list(self.sizes), "stratum_starts": list(self.starts),
                "weight": "stratum_size/samples_per_stratum"}

    def sample(self, batch_size=1, *, device=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("sample one positive record batch")
        rows = [torch.cat([torch.randperm(size, generator=self.generator)[:self.samples_per_stratum] + start
                           for start, size in zip(self.starts, self.sizes)]) for _ in range(batch_size)]
        weights = torch.tensor(self.sizes, dtype=torch.float32).repeat_interleave(self.samples_per_stratum)
        weights = weights / self.samples_per_stratum
        self.drawn_records += batch_size
        self.sample_calls += 1
        return FrameSample(torch.stack(rows).to(device=device), weights.expand(batch_size, -1).clone().to(device=device))

    def state_dict(self):
        return {"semantics": self.semantics, "generator_state": self.generator.get_state(),
                "drawn_records": self.drawn_records, "sample_calls": self.sample_calls}

    def load_state_dict(self, state):
        if state.get("semantics") != self.semantics:
            raise ValueError("resume must preserve the native frame axis and time-sampling distribution")
        if min(state.get("drawn_records", -1), state.get("sample_calls", -1)) < 0:
            raise ValueError("invalid sampler cursor")
        self.generator.set_state(state["generator_state"].cpu())
        self.drawn_records, self.sample_calls = state["drawn_records"], state["sample_calls"]


@torch.no_grad()
def full_reference_energy(reference, *, frame_chunk=32):
    """Cache this unchanged full-axis denominator once per reference record."""
    if reference.ndim != 4 or frame_chunk < 1:
        raise ValueError("use full normalized reference pressure [B,T,Z,X]")
    energy = torch.zeros(reference.shape[0], device=reference.device, dtype=torch.float64)
    for start in range(0, reference.shape[1], frame_chunk):
        energy += reference[:, start:start + frame_chunk].double().square().flatten(1).sum(1)
    return energy


def _selected_reference(reference, indices):
    return reference.gather(1, indices[..., None, None].expand(-1, -1, *reference.shape[-2:]))


def _relative_loss(prediction, target, weights, reference_energy, energy_floor):
    if prediction.shape != target.shape or prediction.ndim != 4 or weights.shape != prediction.shape[:2]:
        raise ValueError("matched pressure frames [B,K,Z,X] need one weight per frame")
    if not math.isfinite(energy_floor) or energy_floor <= 0 or reference_energy.shape != prediction.shape[:1]:
        raise ValueError("retain one full-reference energy and a positive energy floor per record")
    work_dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    error = prediction.to(work_dtype) - target.detach().to(device=prediction.device, dtype=work_dtype)
    frame_sse = error.square().flatten(2).sum(2)
    estimated_record_sse = (frame_sse.double() * weights.detach().to(device=prediction.device, dtype=torch.float64)).sum(1)
    denominator = reference_energy.detach().to(device=prediction.device, dtype=torch.float64).clamp_min(energy_floor)
    return (estimated_record_sse / denominator).mean()


def sampled_relative_pressure_loss(prediction, full_reference, sample, reference_energy, *, energy_floor=1e-12):
    """Mean record-normalized estimator of the original complete-axis pressure SSE."""
    if (full_reference.ndim != 4 or sample.indices.ndim != 2
            or prediction.shape[:2] != sample.indices.shape
            or full_reference.shape[0] != prediction.shape[0]):
        raise ValueError("keep full reference records and their per-record sampled native time indices")
    target = _selected_reference(full_reference, sample.indices.to(full_reference.device))
    return _relative_loss(prediction, target, sample.weights, reference_energy, energy_floor)


def fresh_sampled_model_loss(model, velocity, source, full_reference, sample, reference_energy, *, energy_floor=1e-12):
    """Recompute encoder, current BASE, detail and pressure graph on every call.

    This goes through forward (also compatible with a DDP wrapper). It never
    accepts a precomputed context or residual cache. The input truth is used
    exclusively in the supervised loss, not in model prediction.
    """
    prediction = model(velocity, source, frame_indices=sample.indices, normalized=True)
    return sampled_relative_pressure_loss(prediction, full_reference, sample, reference_energy,
                                           energy_floor=energy_floor)


@torch.no_grad()
def evaluate_complete_axis(model, velocity, source, full_reference, reference_energy, *, energy_floor=1e-12):
    """Evaluate all native frames, without sampling or a replacement prefix."""
    was_training = model.training
    try:
        model.eval()
        prediction = model(velocity, source, normalized=True)
        if prediction.shape != full_reference.shape or not bool(torch.isfinite(prediction).all()):
            raise ValueError("evaluation must return the complete finite native pressure sequence")
        weights = prediction.new_ones(prediction.shape[:2])
        loss = _relative_loss(prediction, full_reference, weights, reference_energy, energy_floor)
        return {"relative_squared_error": float(loss), "relative_l2_percent": 100 * math.sqrt(float(loss)),
                "frames": prediction.shape[1], "top_max_abs": float(prediction[:, :, 0].abs().max())}
    finally:
        model.train(was_training)


def _optimizer_roles(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    try:
        return [[names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]
    except KeyError as error:
        raise ValueError("optimizer contains a parameter outside the saved model") from error


def _model_payload(model):
    if callable(getattr(model, "checkpoint_payload", None)):
        return model.checkpoint_payload()
    return {"model": model.state_dict(), "model_type": type(model).__module__ + "." + type(model).__qualname__}


def _semantics(payload):
    return {key: deepcopy(payload[key]) for key in
            ("schema", "model_type", "model_semantics", "parent_semantics", "repair_settings") if key in payload}


def _binding(binding):
    canonical = json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.loads(canonical), hashlib.sha256(canonical.encode()).hexdigest()


def warm_start_checkpoint_payload(model, optimizer, scheduler, sampler, *, updates_completed,
                                  run_binding, training_state, record_sampler_state=None,
                                  scaler=None, phase="shared_full_model_warm_start"):
    """Serialize immediately after an accepted optimizer/scheduler update.

    The caller stores epoch/record/microbatch cursors in training_state and
    supplies any independent shuffled-record sampler state explicitly. Tensor
    states have standard state_dict reference semantics: save this payload
    before continuing to update weights.
    """
    if type(updates_completed) is not int or updates_completed < 0 or not isinstance(training_state, dict):
        raise ValueError("save a nonnegative accepted-update cursor and explicit training-stage state")
    payload = _model_payload(model)
    binding, digest = _binding(run_binding)
    return {**payload, "training_resume_schema": SCHEMA, "phase": phase,
            "updates_completed": updates_completed, "run_binding": binding, "run_binding_sha256": digest,
            "training_state": deepcopy(training_state), "record_sampler": deepcopy(record_sampler_state),
            "optimizer": optimizer.state_dict(), "optimizer_parameter_names": _optimizer_roles(model, optimizer),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None, "frame_sampler": sampler.state_dict(),
            "rng": capture_rng(),
            "resume_trainable_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
            "module_training_modes": {name: module.training for name, module in model.named_modules()}}


def restore_warm_start_training(model, optimizer, scheduler, sampler, payload, *, expected_run_binding,
                                scaler=None, expected_phase="shared_full_model_warm_start"):
    """True resume only: restore state, with no SVD, gain absorption or calibration."""
    binding, digest = _binding(expected_run_binding)
    if (payload.get("training_resume_schema") != SCHEMA or payload.get("phase") != expected_phase
            or payload.get("run_binding") != binding or payload.get("run_binding_sha256") != digest):
        raise ValueError("resume requires the saved phase, data binding and training settings")
    if _semantics(_model_payload(model)) != _semantics(payload):
        raise ValueError("resume requires identical architecture and physical/model semantics")
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if names != payload["resume_trainable_names"] or _optimizer_roles(model, optimizer) != payload["optimizer_parameter_names"]:
        raise ValueError("restore the saved trainable roles and optimizer parameter ordering before resume")
    if ((scheduler is None) != (payload["scheduler"] is None)
            or (scaler is None) != (payload["scaler"] is None)
            or sampler.semantics != payload["frame_sampler"]["semantics"]):
        raise ValueError("resume must preserve scheduler/scaler presence and the time sampling distribution")
    modules = dict(model.named_modules())
    if modules.keys() != payload["module_training_modes"].keys():
        raise ValueError("resume model module structure differs")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    sampler.load_state_dict(payload["frame_sampler"])
    model.train(payload["module_training_modes"][""])
    for name, training in payload["module_training_modes"].items():
        modules[name].training = training
    # Restore last: constructors, optimizer creation or loading must not move
    # the next CPU/CUDA/Python/NumPy random draw away from the saved state.
    rng = deepcopy(payload["rng"])
    # torch.load(map_location="cuda:...") may relocate RNG byte tensors too;
    # generator.set_state requires their CPU representation.
    rng["torch_cpu"] = rng["torch_cpu"].cpu()
    rng["torch_cuda"] = [value.cpu() for value in rng["torch_cuda"]]
    restore_rng(rng)
    return {"updates_completed": payload["updates_completed"], "phase": payload["phase"],
            "training_state": deepcopy(payload["training_state"]),
            "record_sampler_state": deepcopy(payload.get("record_sampler"))}
