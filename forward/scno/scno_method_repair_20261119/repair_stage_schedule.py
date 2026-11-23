"""Role-based A/B/C training state machine; no model initialization or GPU work.

Runner contract: begin_update(); zero_grad(); backward(); step(). For AMP, unscale
before clip_gradients(), perform scaler.step/update, then finish_update with the
actual success flag. Checkpoint BOTH model weights and this state, plus sampler,
data-normalization binding and scaler state. TEST records never select a model.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re
from time import perf_counter

import torch
from torch import nn


SCHEMA = "scno_repaired_staged_training_v1"
ACTIVE_ROLES = {"A": ("BASE",), "B": ("DETAIL",), "C": ("BASE", "DETAIL")}
DEFAULTS = {
    "A": {"peak_lr": 3e-4, "warmup_updates": 500, "final_lr": 3e-5},
    "B": {"peak_lr": 1e-3, "warmup_updates": 200, "final_lr": 1e-4},
    "C": {"base_lr": 3e-5, "detail_lr": 1e-4, "decay_factor": .1, "warmup_updates": 0},
}


def _fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_stage_plan(config, *, epochs=200, updates_per_epoch):
    """Map configured phase-update proportions to the ACTUAL epoch/batch budget.

    Existing phase_updates are exact when totals match. A different actual
    batch count preserves their cumulative proportions; phase_fractions, when
    supplied too, must agree up to original integer-rounding precision.
    Warmup scales with each phase's budget, retaining the original warmup/phase
    fraction. An explicit warmup_actual_updates overrides that mapping.
    """
    epochs = _positive_int(epochs, "epochs")
    updates_per_epoch = _positive_int(updates_per_epoch, "updates_per_epoch")
    total = epochs * updates_per_epoch
    source_total = _positive_int(config.get("total_updates", total), "configured total_updates")
    protocol = config.get("protocol", "staged")
    if protocol not in ("staged", "joint_from_start"):
        raise ValueError("protocol must be staged or joint_from_start")
    fractions = config.get("phase_fractions")
    counts = config.get("phase_updates")
    if fractions is not None:
        if (len(fractions) != 3 or any(not math.isfinite(x) or x <= 0 for x in fractions)
                or not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("phase_fractions must contain three positive fractions summing to one")
    if counts is not None:
        if (len(counts) != 3 or any(type(n) is not int or n < 1 for n in counts)
                or sum(counts) != source_total):
            raise ValueError("phase_updates must contain three positive counts summing to configured total_updates")
        if fractions is not None:
            for i in (1, 2):
                if abs(sum(counts[:i]) / source_total - sum(fractions[:i])) > 1.0 / source_total + 1e-12:
                    raise ValueError("phase_updates and phase_fractions describe conflicting phase boundaries")
        proportions = [n / source_total for n in counts]
    else:
        proportions = list(fractions or (.6, .2, .2))
    if protocol == "joint_from_start":
        ranges = [("C", 0, total)]
        source_lengths = {"C": source_total}
    else:
        first = round(total * proportions[0])
        second = round(total * sum(proportions[:2]))
        if not 0 < first < second < total:
            raise ValueError("actual update budget is too short to allocate all three phases")
        ranges = [("A", 0, first), ("B", first, second), ("C", second, total)]
        source_first = round(source_total * proportions[0])
        source_second = round(source_total * sum(proportions[:2]))
        source_lengths = {
            "A": source_first, "B": source_second - source_first, "C": source_total - source_second}
        if any(n < 1 for n in source_lengths.values()):
            raise ValueError("configured update budget cannot express three nonempty phases")
    stages = []
    for phase, begin, end in ranges:
        options = {
            "weight_decay": 1e-4, "gradient_clip_norm": 1.0,
            "betas": [.9, .999], "eps": 1e-8,
            **DEFAULTS[phase], **config.get("phases", {}).get(phase, {}),
        }
        requested = options["warmup_updates"]
        if type(requested) is not int or requested < 0:
            raise ValueError("warmup_updates must be a nonnegative integer")
        options["requested_warmup_updates"] = requested
        override = options.get("warmup_actual_updates", config.get("warmup_actual_updates"))
        if isinstance(override, dict):
            override = override.get(phase)
        if override is not None:
            if type(override) is not int or override < 0:
                raise ValueError("warmup_actual_updates must be a nonnegative integer")
            mapped = override
            mapping = "explicit_actual_updates"
        else:
            mapped = round(requested * (end - begin) / source_lengths[phase])
            mapped = max(int(requested > 0), mapped)
            mapping = "phase_budget_proportional"
        options["warmup_mapping"] = mapping
        options["mapped_warmup_before_clamp"] = mapped
        options["warmup_updates"] = min(mapped, max(0, end - begin - 1))
        for key in ("weight_decay", "gradient_clip_norm", "eps"):
            if not math.isfinite(options[key]) or options[key] < 0 or (key == "eps" and options[key] == 0):
                raise ValueError(f"{phase} has invalid {key}")
        if len(options["betas"]) != 2 or any(not 0 <= x < 1 for x in options["betas"]):
            raise ValueError("AdamW betas must be in [0,1)")
        lrs = {}
        for role in ACTIVE_ROLES[phase]:
            peak = options["peak_lr"] if phase in ("A", "B") else options[role.lower() + "_lr"]
            if phase in ("A", "B"):
                final = options["final_lr"]
            else:
                final = options.get(role.lower() + "_final_lr", options.get("final_lr", peak * options["decay_factor"]))
            if not math.isfinite(peak) or not math.isfinite(final) or not 0 < final <= peak:
                raise ValueError(f"{phase}/{role} needs 0 < final_lr <= peak_lr")
            lrs[role] = {"peak_lr": float(peak), "final_lr": float(final)}
        stages.append({
            "phase": phase, "begin": begin, "end": end, "updates": end - begin,
            "configured_phase_updates": source_lengths[phase],
            "start_epoch": begin / updates_per_epoch, "end_epoch": end / updates_per_epoch,
            "roles": list(ACTIVE_ROLES[phase]), "options": options, "learning_rates": lrs,
        })
    return {
        "schema": SCHEMA, "protocol": protocol, "epochs": epochs,
        "updates_per_epoch": updates_per_epoch, "total_updates": total,
        "configured_total_updates": source_total, "phase_budget_rescaled": total != source_total,
        "configured_phase_updates": list(counts) if counts is not None else None,
        "warmup_unit": "accepted_optimizer_updates_after_budget_mapping",
        "phase_fractions": proportions, "stages": stages,
    }


def cosine_ratio(position, length, warmup, final_ratio):
    """The first warmup step is peak/warmup; the final accepted update uses final LR."""
    position = min(max(position, 0), length - 1)
    if position < warmup:
        return (position + 1) / warmup
    peak_position = max(0, warmup - 1)
    remaining = length - 1 - peak_position
    fraction = (position - peak_position) / remaining if remaining else 0.0
    return final_ratio + (1 - final_ratio) * .5 * (1 + math.cos(math.pi * fraction))


def stage_b_calibration_spec(config, train_ids, native_times):
    """Fixed-seed TRAIN-only 32 x (four strata x two frames) request.

    Sample weight is numerator/max(original_record_energy, energy_floor).
    Dividing separately by each scale's coefficient count is not requested.
    """
    count = _positive_int(config.get("affine_calibration_records", 32), "affine_calibration_records")
    if config.get("affine_calibration_times_per_record", 8) != 8:
        raise ValueError("stage-B startup calibration requires exactly two frames in each of four strata")
    ids = sorted(set(train_ids), key=str)
    if len(ids) < count:
        raise ValueError("not enough TRAIN records for the declared calibration sample")
    times = torch.as_tensor(native_times).detach().cpu()
    if (times.ndim != 1 or len(times) < 8 or not torch.isfinite(times).all()
            or not (times[1:] > times[:-1]).all()):
        raise ValueError("calibration needs a finite increasing complete native time axis")
    size, extra = divmod(len(times), 4)
    sizes = [size + int(i < extra) for i in range(4)]
    strata, first = [], 0
    for n in sizes:
        strata.append(list(range(first, first + n)))
        first += n
    seed = int(config.get("seed", 0))
    generator = random.Random(seed)
    selected = generator.sample(ids, count)
    draws = []
    for record_id in selected:
        indices, labels, numerators = [], [], []
        for band, slots in enumerate(strata):
            chosen = sorted(generator.sample(slots, 2))
            indices.extend(chosen)
            labels.extend([band, band])
            numerators.extend([sizes[band] / 2.0] * 2)
        draws.append({
            "record_id": record_id, "frame_indices": indices,
            "times_s": times[indices].tolist(), "stratum_indices": labels,
            "weight_numerators": numerators,
        })
    return {
        "seed": seed, "train_record_ids": selected, "records": count,
        "frames_per_record": 8, "pressure_frames": count * 8,
        "native_frames": len(times), "stratum_sizes": sizes, "draws_per_stratum": 2,
        "denominator": "max(original_full_native_record_energy, energy_floor)",
        "target": "valid_detail_coefficients_of_reference_minus_current_frozen_base",
        "ridge_anchor": "current_effective_affine_readout", "draws": draws,
    }


def capture_rng():
    import numpy as np

    name, keys, position, gaussian, cached = np.random.get_state()
    return {
        "python": random.getstate(), "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "numpy": [name, keys.tolist(), position, gaussian, cached],
    }


def restore_rng(state):
    import numpy as np

    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))


class StageController:
    """An explicit accepted-update controller with strict role/split bindings.

    named_roles is a mapping BASE/DETAIL/FROZEN to parameter names or
    (name, Parameter) pairs. If omitted, model.named_roles() must supply it.
    BASE includes backbone, base pressure, temporal Gabor and carrier parameters.
    DETAIL includes the complete trainable detail branch. Fixed gain/bases
    belong to FROZEN; fixed bases implemented as buffers need no parameter role.
    """

    def __init__(
            self, model, config, *, epochs=200, updates_per_epoch, named_roles=None,
            train_ids, test_ids=(), validation_ids=(), native_times=None, startup_mode="fresh"):
        if startup_mode not in ("fresh", "warm_start", "new_training", "resume"):
            raise ValueError("unsupported startup_mode")
        self.model, self.config = model, deepcopy(config)
        self.plan = build_stage_plan(config, epochs=epochs, updates_per_epoch=updates_per_epoch)
        self.train_ids, self.test_ids, self.validation_ids = map(
            frozenset, (train_ids, test_ids, validation_ids))
        if (not self.train_ids or self.train_ids & self.test_ids or self.train_ids & self.validation_ids
                or self.validation_ids & self.test_ids):
            raise ValueError("TRAIN, validation and TEST partitions must be nonempty TRAIN and mutually disjoint")
        actual = dict(model.named_parameters())
        role_input = named_roles if named_roles is not None else model.named_roles()
        self.roles, assigned = {}, set()
        for role in ("BASE", "DETAIL", "FROZEN"):
            values = []
            for item in role_input.get(role, ()):
                name, parameter = (item, actual[item]) if isinstance(item, str) else item
                if name not in actual or parameter is not actual[name] or id(parameter) in assigned:
                    raise ValueError(f"invalid, duplicated or aliased role parameter {name}")
                values.append((name, parameter))
                assigned.add(id(parameter))
            self.roles[role] = values
        for name, parameter in actual.items():
            if id(parameter) not in assigned:
                if parameter.requires_grad:
                    raise ValueError(f"trainable parameter has no explicit role: {name}")
                self.roles["FROZEN"].append((name, parameter))
        if any(not self.roles[role] for role in ("BASE", "DETAIL")):
            raise ValueError("provide nonempty BASE and DETAIL role partitions")
        if any(role not in self.roles for role in role_input):
            raise ValueError("roles are BASE, DETAIL and FROZEN")
        self.native_times = (native_times if native_times is not None else getattr(model, "full_times_s", None))
        if self.config.get("affine_calibration", False) and self.native_times is None:
            raise ValueError("stage-B calibration requires the registered native time axis")
        axis = (torch.as_tensor(self.native_times).detach().cpu().tolist()
                if self.native_times is not None else None)
        self.binding = _fingerprint({
            "plan": self.plan, "roles": {
                role: [[name, list(p.shape), str(p.dtype)] for name, p in values]
                for role, values in self.roles.items()},
            "train_ids": sorted(map(str, self.train_ids)), "test_ids": sorted(map(str, self.test_ids)),
            "validation_ids": sorted(map(str, self.validation_ids)), "native_times": axis,
            "calibration": {
                key: self.config.get(key) for key in (
                    "affine_calibration", "affine_calibration_records",
                    "affine_calibration_times_per_record", "seed")},
        })
        self.completed_updates = 0
        self.current_phase = None
        self.optimizer = self.scheduler = None
        self.calibration = {"state": "not_started"}
        self.best = {}
        self._resume_pending = startup_mode == "resume"

    def _stage(self, phase=None):
        if phase is not None:
            return next(stage for stage in self.plan["stages"] if stage["phase"] == phase)
        return next((stage for stage in self.plan["stages"]
                     if stage["begin"] <= self.completed_updates < stage["end"]), None)

    def _set_trainability(self, phase):
        active = set(ACTIVE_ROLES[phase])
        for role, pairs in self.roles.items():
            for _, parameter in pairs:
                parameter.requires_grad_(role in active)
                if role not in active:
                    parameter.grad = None
        self.model.train()
        # Frozen BN statistics and dropout must stay frozen along with weights.
        for module in self.model.modules():
            parameters = list(module.parameters())
            if parameters and not any(p.requires_grad for p in parameters):
                module.eval()

    def _build_optimizer(self, stage):
        options, groups, schedules = stage["options"], [], []
        norm_types = (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm,
                      nn.GroupNorm, nn.LayerNorm, nn.LocalResponseNorm)
        if hasattr(nn, "RMSNorm"):
            norm_types += (nn.RMSNorm,)
        norm_ids = {id(p) for module in self.model.modules() if isinstance(module, norm_types)
                    for p in module.parameters(recurse=False)}
        exempt_names = set(getattr(self.model, "no_decay_names", ()))
        for role in stage["roles"]:
            schedule = stage["learning_rates"][role]
            peak, final = schedule["peak_lr"], schedule["final_lr"]
            buckets = {"decay": [], "no_decay": []}
            for name, parameter in self.roles[role]:
                exempt = name == "bias" or name.endswith(".bias") or id(parameter) in norm_ids or name in exempt_names
                buckets["no_decay" if exempt else "decay"].append(parameter)
            for suffix, parameters in buckets.items():
                if parameters:
                    groups.append({
                        "params": parameters, "role": role, "group_name": f"{role}_{suffix}", "lr": peak,
                        "weight_decay": 0.0 if suffix == "no_decay" else options["weight_decay"]})
                    schedules.append(
                        lambda position, ratio=final / peak:
                        cosine_ratio(position, stage["updates"], options["warmup_updates"], ratio))
        optimizer = torch.optim.AdamW(groups, betas=tuple(options["betas"]), eps=options["eps"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedules)
        return optimizer, scheduler

    def begin_update(self, calibration_callback=None):
        """Freeze roles, enter phase and perform first B calibration before gradients.

        The callback(model, spec) must atomically fit only DETAIL affine outputs
        on the supplied TRAIN frames, return its receipt, and use detached
        current frozen base features. It may return rewritten_parameters for an
        additional role check. A completed checkpointed calibration is not run
        again on resume, including a checkpoint before the first B optimizer step.
        """
        if self._resume_pending:
            raise RuntimeError("resume requires load_state_dict before any update")
        stage = self._stage()
        if stage is None:
            raise StopIteration("the configured training budget is complete")
        phase = stage["phase"]
        self._set_trainability(phase)
        changed = phase != self.current_phase
        if changed:
            self.model.zero_grad(set_to_none=True)
            self.optimizer, self.scheduler = self._build_optimizer(stage)
            self.current_phase = phase
        if phase == "B" and self.calibration["state"] != "complete":
            if self.config.get("affine_calibration", False):
                if calibration_callback is None:
                    raise RuntimeError("first B entry requires the declared residual-calibration callback")
                spec = stage_b_calibration_spec(self.config, self.train_ids, self.native_times)
                self.calibration = {"state": "running", "spec": spec}
                start = perf_counter()
                try:
                    with torch.no_grad():
                        receipt = calibration_callback(self.model, deepcopy(spec))
                    if not isinstance(receipt, dict):
                        raise ValueError("calibration callback must return a serializable receipt")
                    detail_names = {name for name, _ in self.roles["DETAIL"]}
                    if any(name not in detail_names for name in receipt.get("rewritten_parameters", ())):
                        raise ValueError("stage-B calibration rewrote a non-DETAIL parameter")
                    self.calibration = {
                        "state": "complete", "spec": spec, "receipt": deepcopy(receipt),
                        "pressure_frames": spec["pressure_frames"], "elapsed_s": perf_counter() - start,
                        "completed_updates_at_entry": self.completed_updates,
                    }
                except BaseException:
                    self.calibration = {"state": "not_started", "previous_attempt_failed": True}
                    raise
            else:
                self.calibration = {"state": "complete", "skipped_by_config": True,
                                    "pressure_frames": 0, "elapsed_s": 0.0}
        return {
            "phase": phase, "phase_changed": changed, "completed_updates": self.completed_updates,
            "optimizer": self.optimizer, "scheduler": self.scheduler,
            "learning_rates": {g["group_name"]: g["lr"] for g in self.optimizer.param_groups},
            "gradient_clip_norm": stage["options"]["gradient_clip_norm"],
        }

    def clip_gradients(self):
        if self.optimizer is None:
            raise RuntimeError("begin_update must precede backward/step")
        stage = self._stage()
        if stage is None or self.current_phase != stage["phase"]:
            raise RuntimeError("begin_update is required before stepping a new phase or completed budget")
        if self.scheduler.last_epoch != self.completed_updates - stage["begin"]:
            raise RuntimeError("scheduler and accepted-update cursor disagree")
        active = [p for group in self.optimizer.param_groups for p in group["params"]]
        threshold = self._stage(self.current_phase)["options"]["gradient_clip_norm"]
        if threshold:
            return torch.nn.utils.clip_grad_norm_(active, threshold, error_if_nonfinite=True)
        return None

    def finish_update(self, *, optimizer_was_stepped):
        """Advance exactly once after an accepted optimizer update, including AMP."""
        if not optimizer_was_stepped:
            return
        next_stage = self._stage()
        if self.optimizer is None or next_stage is None or self.current_phase != next_stage["phase"]:
            raise RuntimeError("begin_update is missing at a phase boundary")
        stage = self._stage(self.current_phase)
        if self.scheduler.last_epoch != self.completed_updates - stage["begin"]:
            raise RuntimeError("scheduler and accepted-update cursor disagree")
        self.scheduler.step()
        self.completed_updates += 1

    def step(self):
        """Non-AMP convenience: clip, update, schedule, commit accepted-update cursor."""
        norm = self.clip_gradients()
        self.optimizer.step()
        self.finish_update(optimizer_was_stepped=True)
        return norm

    def state_dict(self):
        if self.calibration["state"] == "running":
            raise RuntimeError("checkpoint only after atomic stage-B calibration completes")
        return {
            "schema": SCHEMA, "binding": self.binding, "plan": deepcopy(self.plan),
            "completed_updates": self.completed_updates, "current_phase": self.current_phase,
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "calibration": deepcopy(self.calibration), "best": deepcopy(self.best), "rng": capture_rng(),
        }

    def load_state_dict(self, state, *, restore_random_state=True):
        """Restore a strict matching state without any initialization/calibration."""
        if (state.get("schema") != SCHEMA or state.get("binding") != self.binding
                or state.get("plan") != self.plan):
            raise ValueError("training phase plan, parameter roles or partition binding changed")
        completed = state["completed_updates"]
        phase = state["current_phase"]
        if type(completed) is not int or not 0 <= completed <= self.plan["total_updates"]:
            raise ValueError("invalid saved accepted-update cursor")
        if state["calibration"].get("state") == "running":
            raise ValueError("an in-flight partial calibration cannot be resumed as complete")
        if (state["calibration"].get("state") == "complete"
                and not state["calibration"].get("skipped_by_config")
                and self.config.get("affine_calibration", False)):
            expected_spec = stage_b_calibration_spec(self.config, self.train_ids, self.native_times)
            if state["calibration"].get("spec") != expected_spec:
                raise ValueError("saved stage-B calibration roster or native frame indices changed")
        if phase is None:
            if completed or state["optimizer"] is not None or state["scheduler"] is not None:
                raise ValueError("unstarted stage state is inconsistent")
            optimizer = scheduler = None
        else:
            stage = self._stage(phase)
            if not stage["begin"] <= completed <= stage["end"]:
                raise ValueError("saved phase and update cursor disagree")
            if phase in ("B", "C") and self.plan["protocol"] == "staged":
                if state["calibration"].get("state") != "complete":
                    raise ValueError("saved B/C state has no completed B calibration receipt")
            self._set_trainability(phase)
            optimizer, scheduler = self._build_optimizer(stage)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if scheduler.last_epoch != completed - stage["begin"]:
                raise ValueError("saved scheduler is inconsistent with accepted updates")
        self.completed_updates, self.current_phase = completed, phase
        self.optimizer, self.scheduler = optimizer, scheduler
        self.calibration, self.best = deepcopy(state["calibration"]), deepcopy(state["best"])
        self._resume_pending = False
        if restore_random_state:
            restore_rng(state["rng"])
        return self

    def save_best(
            self, value, *, split_role, split_name, metric_name, record_ids, output_dir,
            payload_factory=None, extra_state=None, lower_is_better=True):
        """Save separate, explicitly identified best TRAIN_monitor or validation states."""
        role = split_role.upper()
        records = frozenset(record_ids)
        if role not in ("TRAIN_MONITOR", "VALIDATION"):
            raise ValueError("only TRAIN_MONITOR or VALIDATION may select best weights; TEST is final-only")
        allowed = self.train_ids if role == "TRAIN_MONITOR" else self.validation_ids
        if not records or not records <= allowed or records & self.test_ids:
            raise ValueError("selection record identities do not match the declared partition")
        if not math.isfinite(value):
            raise ValueError("selection metric must be finite")
        identity = {
            "split_role": role, "split_name": str(split_name), "metric_name": str(metric_name),
            "record_ids": sorted(map(str, records)), "lower_is_better": bool(lower_is_better)}
        key = f"{role}:{split_name}:{metric_name}"
        previous = self.best.get(key)
        if previous is not None:
            if previous["identity"] != identity:
                raise ValueError("a selection stream changed record identities or metric direction")
            better = value < previous["value"] if lower_is_better else value > previous["value"]
            if not better:
                return None
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{role.lower()}_{split_name}_{metric_name}")[:100]
        path = Path(output_dir) / f"best_{stem}_{_fingerprint(identity)[:10]}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "identity": identity, "value": float(value), "completed_updates": self.completed_updates,
            "phase": self.current_phase, "path": str(path)}
        self.best[key] = metadata
        try:
            payload = payload_factory() if payload_factory is not None else {"model": self.model.state_dict()}
            if "stage_schedule" in payload or "selection" in payload:
                raise ValueError("payload_factory must reserve stage_schedule and selection")
            payload = {**payload, "stage_schedule": self.state_dict(), "selection": deepcopy(metadata)}
            for name, content in (extra_state or {}).items():
                if name in payload:
                    raise ValueError(f"extra_state would overwrite protected checkpoint field {name}")
                payload[name] = content
            temporary = path.with_suffix(path.suffix + ".pending")
            torch.save(payload, temporary)
            temporary.replace(path)
        except BaseException:
            if previous is None:
                self.best.pop(key, None)
            else:
                self.best[key] = previous
            raise
        return metadata


def run_cpu_checks():
    """Dummy learning checks; no PDE data, production model or GPU is touched."""
    import tempfile

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Sequential(nn.Linear(3, 4), nn.Dropout(.2))
            self.detail = nn.Linear(4, 1)
            self.fixed = nn.Parameter(torch.tensor(1.), requires_grad=False)
            self.register_buffer("full_times_s", torch.linspace(0, 1, 401))

        def forward(self, value):
            return self.detail(self.base(value)) * self.fixed

        def named_roles(self):
            return {
                "BASE": [(name, p) for name, p in self.named_parameters() if name.startswith("base.")],
                "DETAIL": [(name, p) for name, p in self.named_parameters() if name.startswith("detail.")],
            }

    config = {
        "seed": 711, "protocol": "staged", "total_updates": 18, "phase_updates": [6, 6, 6],
        "phase_fractions": [1 / 3] * 3, "affine_calibration": True,
        "affine_calibration_records": 2, "affine_calibration_times_per_record": 8,
        "phases": {
            "A": {"peak_lr": .03, "warmup_updates": 2, "final_lr": .003},
            "B": {"peak_lr": .02, "warmup_updates": 2, "final_lr": .002},
            "C": {"base_lr": .01, "detail_lr": .02, "decay_factor": .1, "warmup_updates": 0}},
    }
    options = {
        "epochs": 3, "updates_per_epoch": 6, "train_ids": [f"train{i}" for i in range(4)],
        "test_ids": ["test0"], "validation_ids": ["validation0"]}
    generator = torch.Generator().manual_seed(177)
    inputs = [torch.randn(5, 3, generator=generator) for _ in range(18)]
    targets = [torch.randn(5, 1, generator=generator) for _ in range(18)]
    initial_rng = capture_rng()
    torch.manual_seed(588)
    initial_model = Dummy().state_dict()
    training_rng = capture_rng()
    callbacks = []

    def calibrate(model, spec):
        assert not any(p.requires_grad for p in model.base.parameters())
        assert all(p.requires_grad for p in model.detail.parameters())
        assert not model.base[1].training
        assert spec["pressure_frames"] == 16 and spec["stratum_sizes"] == [101, 100, 100, 100]
        for draw in spec["draws"]:
            assert len(set(draw["frame_indices"])) == 8
            assert draw["weight_numerators"] == [50.5, 50.5, 50., 50., 50., 50., 50., 50.]
        callbacks.append(True)
        model.detail.bias.add_(.01)
        return {"rewritten_parameters": ["detail.bias"], "pressure_frames": spec["pressure_frames"]}

    def run(model, control, end, sequence):
        while control.completed_updates < end:
            cursor = control.completed_updates
            entry = control.begin_update(calibrate)
            control.optimizer.zero_grad(set_to_none=True)
            loss = (model(inputs[cursor]) - targets[cursor]).square().mean()
            loss.backward()
            active = set(ACTIVE_ROLES[entry["phase"]])
            for role, values in control.roles.items():
                assert all((p.grad is not None) == (role in active) for _, p in values)
            sequence.append((entry["phase"], list(entry["learning_rates"].values())))
            control.step()

    first = Dummy()
    first.load_state_dict(initial_model)
    restore_rng(training_rng)
    controller = StageController(first, config, **options)
    full_sequence = []
    run(first, controller, 18, full_sequence)
    assert len(callbacks) == 1
    callbacks.clear()
    second = Dummy()
    second.load_state_dict(initial_model)
    restore_rng(training_rng)
    interrupted = StageController(second, config, **options)
    interrupted_sequence = []
    run(second, interrupted, 7, interrupted_sequence)
    saved = deepcopy(interrupted.state_dict())
    saved_weights = deepcopy(second.state_dict())
    resumed_model = Dummy()
    resumed_model.load_state_dict(saved_weights)
    resumed = StageController(resumed_model, config, startup_mode="resume", **options)
    try:
        resumed.begin_update(calibrate)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unrestored resume entered training")
    resumed.load_state_dict(saved)
    run(resumed_model, resumed, 18, interrupted_sequence)
    assert len(callbacks) == 1 and full_sequence == interrupted_sequence
    assert all(torch.equal(p, resumed_model.state_dict()[name]) for name, p in first.state_dict().items())
    for phase, first_index, final_index, expected_peak, expected_final in (
            ("A", 0, 5, .03, .003), ("B", 6, 11, .02, .002)):
        assert math.isclose(full_sequence[first_index][1][0], expected_peak / 2)
        assert math.isclose(full_sequence[first_index + 1][1][0], expected_peak)
        assert math.isclose(full_sequence[final_index][1][0], expected_final)
    assert math.isclose(full_sequence[-1][1][0], .001)
    assert math.isclose(full_sequence[-1][1][-1], .002)
    remap = build_stage_plan(
        {"total_updates": 50000, "phase_updates": [30000, 10000, 10000],
         "phase_fractions": [.6, .2, .2]}, epochs=200, updates_per_epoch=225)
    assert remap["total_updates"] == 45000
    assert [stage["updates"] for stage in remap["stages"]] == [27000, 9000, 9000]
    assert [stage["options"]["warmup_updates"] for stage in remap["stages"]] == [450, 180, 0]
    big_batch_config = {
        "total_updates": 50000, "phase_updates": [30000, 10000, 10000],
        "phase_fractions": [.6, .2, .2], "phases": {"B": {"warmup_actual_updates": 7}}}
    big_batch = build_stage_plan(big_batch_config, epochs=200, updates_per_epoch=2)
    assert [stage["options"]["warmup_updates"] for stage in big_batch["stages"]] == [4, 7, 0]
    assert big_batch["stages"][1]["options"]["warmup_mapping"] == "explicit_actual_updates"
    with tempfile.TemporaryDirectory(prefix="scno_stage_cpu_") as folder:
        assert Path(folder).resolve().parent == Path(tempfile.gettempdir()).resolve()
        monitor = resumed.save_best(
            .2, split_role="TRAIN_MONITOR", split_name="in_train", metric_name="relative_mse",
            record_ids=["train0"], output_dir=folder)
        validation = resumed.save_best(
            .3, split_role="VALIDATION", split_name="heldout_val", metric_name="relative_mse",
            record_ids=["validation0"], output_dir=folder)
        assert Path(monitor["path"]).is_file() and Path(validation["path"]).is_file()
        selected = torch.load(monitor["path"], weights_only=True)
        assert selected["selection"]["identity"]["split_role"] == "TRAIN_MONITOR"
        assert resumed.save_best(
            .4, split_role="TRAIN_MONITOR", split_name="in_train", metric_name="relative_mse",
            record_ids=["train0"], output_dir=folder) is None
        for role, ids in (("TEST", ["test0"]), ("VALIDATION", ["test0"]), ("TRAIN_MONITOR", ["test0"])):
            try:
                resumed.save_best(
                    .1, split_role=role, split_name="illegal", metric_name="relative_mse",
                    record_ids=ids, output_dir=folder)
            except ValueError:
                pass
            else:
                raise AssertionError("TEST selected a checkpoint")
    restore_rng(initial_rng)
    return {
        "passed": True, "role_freezing_and_gradients": True,
        "frozen_backbone_dropout_eval": True, "stage_b_calibrated_once_after_resume": True,
        "optimizer_scheduler_rng_resume_exact": True, "warmup_peak_final_lrs": True,
        "actual_200_epoch_phase_remapping": [27000, 9000, 9000],
        "warmup_scales_with_phase_budget_and_explicit_override": True,
        "fixed_seed_train_strata_weights": True,
        "best_train_monitor_validation_saved_separately": True, "test_selection_rejected": True,
    }


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--cpu-check"]:
        torch.set_num_threads(2)
        print(json.dumps(run_cpu_checks(), indent=2))
    else:
        print("Import StageController; --cpu-check runs the CPU state-machine checks.")
