"""Conservative shared-backbone LR intervention without losing Adam progress.

The stage state machine is inherited unchanged. The subclass only changes
optimizer construction plus its fixed identity metadata. Migration is an
explicit operation between controllers over the SAME model Parameter objects;
it never reloads/initializes weights, takes a step, recalibrates or resets epoch
progress. The driver saves the returned LR_INTERVENTION receipt with its new
checkpoint. Future resumes use the new binding and normal load_state_dict.
"""

from copy import deepcopy
import inspect
import math

import torch

from repair_stage_schedule import StageController, _fingerprint, cosine_ratio, capture_rng, restore_rng


SCHEMA = "SCNO_UL_SHARED_BACKBONE_RATE_INTERVENTION_V1"
DEFAULT_PREFIXES = (
    "parent.medium_encoder", "parent.source_encoder", "parent.multiscale_fuse",
    "parent.source_projection", "parent.map_projection", "parent.propagation",
    "parent.gabor_spatial", "parent.gabor_embedding",
)


def _clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(content) for key, content in value.items()}
    if isinstance(value, list):
        return [_clone(content) for content in value]
    if isinstance(value, tuple):
        return tuple(_clone(content) for content in value)
    return deepcopy(value)


def _prefix_match(name, prefixes):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


class BackboneRateController(StageController):
    """Multiply only listed BASE shared features/carrier/Gabor LRs by factor.

    BASE decoder.proj and all DETAIL parameters keep the configured role LR.
    Their warmup/cosine/final ratios and all AdamW hyperparameters are unchanged.
    No phase-transition, calibration, clipping or update methods are replaced.
    """

    def __init__(
            self, model, config, *, backbone_factor=None, backbone_prefixes=None,
            intervention_receipt=None, **kwargs):
        factor = config.get("backbone_lr_factor", .1) if backbone_factor is None else backbone_factor
        prefixes = (config.get("backbone_lr_prefixes", DEFAULT_PREFIXES)
                    if backbone_prefixes is None else backbone_prefixes)
        if not math.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("backbone_factor must be finite and in (0,1]")
        if (isinstance(prefixes, str) or not prefixes
                or any(not isinstance(prefix, str) or not prefix.rstrip(".") for prefix in prefixes)):
            raise ValueError("declare a nonempty list of exact parameter-module prefixes")
        self.backbone_factor = float(factor)
        self.backbone_prefixes = tuple(dict.fromkeys(prefix.rstrip(".") for prefix in prefixes))
        super().__init__(model, config, **kwargs)
        self.base_stage_binding = self.binding
        scaled = [
            name for name, _ in self.roles["BASE"]
            if self._shared(name, "BASE")]
        if not scaled:
            raise ValueError("the explicit shared-backbone prefixes matched no BASE parameters")
        self.backbone_lr_policy = {
            "schema": SCHEMA, "factor": self.backbone_factor, "prefixes": list(self.backbone_prefixes),
            "applies_to_role": "BASE", "decoder_projection_multiplier": 1.0,
            "detail_multiplier": 1.0, "scaled_parameter_names": sorted(scaled),
            "schedule_ratio": "unchanged phase-local warmup/cosine/final",
            "base_stage_binding": self.base_stage_binding,
        }
        # Fixed metadata is reproduced by the constructor on every resume, so
        # normal strict StageController loading needs no compatibility bypass.
        self.plan["backbone_lr_policy"] = deepcopy(self.backbone_lr_policy)
        self.binding = _fingerprint({
            "base_stage_binding": self.base_stage_binding, "backbone_lr_policy": self.backbone_lr_policy})
        self.lr_intervention_receipt = deepcopy(intervention_receipt)

    def _shared(self, name, role):
        return (role == "BASE" and not (
            name == "parent.decoder.proj" or name.startswith("parent.decoder.proj."))
            and _prefix_match(name, self.backbone_prefixes))

    def _build_optimizer(self, stage):
        # Reuse exactly the existing role/no-decay classification and defaults;
        # the temporary empty optimizer changes neither parameters nor moments.
        original, _ = super()._build_optimizer(stage)
        names = {id(p): name for name, p in self.model.named_parameters()}
        groups, schedules = [], []
        for old_group in original.param_groups:
            role = old_group["role"]
            schedule = stage["learning_rates"][role]
            for kind, multiplier in (("shared", self.backbone_factor), ("local", 1.0)):
                parameters = [
                    p for p in old_group["params"]
                    if self._shared(names[id(p)], role) == (kind == "shared")]
                if not parameters:
                    continue
                peak = schedule["peak_lr"] * multiplier
                group = {key: _clone(value) for key, value in old_group.items() if key != "params"}
                group.update(
                    params=parameters, lr=peak, initial_lr=peak, lr_multiplier=multiplier,
                    rate_kind=kind, source_group_name=old_group["group_name"],
                    group_name=old_group["group_name"] + "_" + kind)
                groups.append(group)
                schedules.append(
                    lambda position, ratio=schedule["final_lr"] / schedule["peak_lr"]:
                    cosine_ratio(
                        position, stage["updates"], stage["options"]["warmup_updates"], ratio))
        allowed = inspect.signature(torch.optim.AdamW).parameters
        defaults = {key: _clone(value) for key, value in original.defaults.items()
                    if key in allowed and key != "params"}
        optimizer = torch.optim.AdamW(groups, **defaults)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedules)
        return optimizer, scheduler


def _without_policy(plan):
    result = deepcopy(plan)
    result.pop("backbone_lr_policy", None)
    return result


def migrate_stage_controller(legacy_controller, new_controller):
    """Explicitly transfer accepted-update progress and every Adam state value.

    Both controllers must reference the same model and exact role Parameter
    identities. The receiving controller must be unused. The source controller
    is retired after migration; do not continue stepping both optimizers.
    Return LR_INTERVENTION metadata for the driver's JSON/checkpoint receipt.
    """
    old, new = legacy_controller, new_controller
    if not isinstance(new, BackboneRateController) or old is new or old.model is not new.model:
        raise ValueError("migration requires separate controllers over the SAME model object")
    old_base_binding = getattr(old, "base_stage_binding", old.binding)
    if (old_base_binding != new.base_stage_binding
            or _without_policy(old.plan) != _without_policy(new.plan)):
        raise ValueError("phase budgets, data partitions or original stage settings changed")
    if (new.optimizer is not None or new.scheduler is not None
            or new.current_phase is not None or new.completed_updates != 0):
        raise ValueError("the receiving controller must not have begun training")
    if old.calibration.get("state") == "running":
        raise ValueError("do not migrate an in-flight residual calibration")
    if old._resume_pending:
        raise ValueError("fully restore the legacy checkpoint before migrating it")
    for role in old.roles:
        if (role not in new.roles or len(old.roles[role]) != len(new.roles[role])
                or any(a != b or pa is not pb for (a, pa), (b, pb) in
                       zip(old.roles[role], new.roles[role], strict=True))):
            raise ValueError("role names or model Parameter identities changed")
    names = {id(p): name for name, p in old.model.named_parameters()}
    versions = [(p, p._version, p.requires_grad, p.grad) for p in old.model.parameters()]
    modes = [(module, module.training) for module in old.model.modules()]
    rng = capture_rng()
    optimizer = scheduler = None
    group_receipts = []
    state_count = 0
    try:
        if old.current_phase is None:
            if old.completed_updates != 0 or old.optimizer is not None or old.scheduler is not None:
                raise ValueError("unstarted source controller has inconsistent optimizer progress")
        else:
            stage = new._stage(old.current_phase)
            local_position = old.completed_updates - stage["begin"]
            if (not 0 <= local_position <= stage["updates"] or old.optimizer is None
                    or old.scheduler is None or old.scheduler.last_epoch != local_position):
                raise ValueError("legacy scheduler, phase and accepted-update cursor disagree")
            optimizer, scheduler = new._build_optimizer(stage)
            old_group_by_parameter = {}
            for group in old.optimizer.param_groups:
                for p in group["params"]:
                    if id(p) in old_group_by_parameter:
                        raise ValueError("legacy optimizer duplicated a Parameter")
                    old_group_by_parameter[id(p)] = group
            new_parameters = [p for group in optimizer.param_groups for p in group["params"]]
            if ({id(p) for p in new_parameters} != set(old_group_by_parameter)
                    or len(new_parameters) != len(old_group_by_parameter)):
                raise ValueError("migration changed the active optimizer parameter set")
            if any(id(p) not in old_group_by_parameter for p in old.optimizer.state):
                raise ValueError("legacy optimizer contains an orphan parameter state")
            metadata_keys = {
                "params", "lr", "initial_lr", "role", "group_name", "lr_multiplier",
                "rate_kind", "source_group_name"}
            for group in optimizer.param_groups:
                source = old_group_by_parameter[id(group["params"][0])]
                if any(old_group_by_parameter[id(p)] is not source for p in group["params"]):
                    raise ValueError("a new LR group merges incompatible old optimizer groups")
                # Retain all non-LR source settings, including actual foreach,
                # fused/capturable/AMSGrad options and weight decay.
                for key, value in source.items():
                    if key not in metadata_keys:
                        group[key] = _clone(value)
                peak = group["initial_lr"]
                ratio = stage["learning_rates"][group["role"]]
                current_lr = peak * cosine_ratio(
                    local_position, stage["updates"], stage["options"]["warmup_updates"],
                    ratio["final_lr"] / ratio["peak_lr"])
                old_multiplier = source.get("lr_multiplier", 1.0)
                expected = source["lr"] * group["lr_multiplier"] / old_multiplier
                if not math.isclose(current_lr, expected, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError("source LR no longer follows its recorded scheduler; do not silently overwrite it")
                group["lr"] = current_lr
                group_receipts.append({
                    "group_name": group["group_name"], "role": group["role"],
                    "old_group_name": source.get("group_name"), "rate_kind": group["rate_kind"],
                    "old_lr": float(source["lr"]), "new_lr": float(current_lr),
                    "multiplier": group["lr_multiplier"],
                    "parameter_names": [names[id(p)] for p in group["params"]],
                    "non_lr_optimizer_settings_preserved": True,
                })
            optimizer.defaults = _clone(old.optimizer.defaults)
            for parameter, state in old.optimizer.state.items():
                optimizer.state[parameter] = _clone(state)
                state_count += 1
            # Topology-dependent arrays are rebuilt; position and every other
            # schedule cursor field are transferred without taking a step.
            old_scheduler = old.scheduler.state_dict()
            migrated_scheduler = deepcopy(scheduler.state_dict())
            for key, value in old_scheduler.items():
                if key not in ("base_lrs", "_last_lr", "lr_lambdas"):
                    migrated_scheduler[key] = _clone(value)
            migrated_scheduler["base_lrs"] = [group["initial_lr"] for group in optimizer.param_groups]
            migrated_scheduler["_last_lr"] = [group["lr"] for group in optimizer.param_groups]
            scheduler.load_state_dict(migrated_scheduler)
            if scheduler.last_epoch != old.scheduler.last_epoch:
                raise RuntimeError("scheduler position changed during migration")
        if any(p._version != version or p.requires_grad != requires_grad or p.grad is not grad
               for p, version, requires_grad, grad in versions):
            raise RuntimeError("migration modified a model parameter, gradient or trainability flag")
        if any(module.training != mode for module, mode in modes):
            raise RuntimeError("migration modified model training/evaluation modes")
        new.completed_updates, new.current_phase = old.completed_updates, old.current_phase
        new.optimizer, new.scheduler = optimizer, scheduler
        new.calibration, new.best = deepcopy(old.calibration), deepcopy(old.best)
        new._resume_pending = False
        receipt = {
            "schema": SCHEMA, "event": "LR_INTERVENTION",
            "old_controller": type(old).__name__, "new_controller": type(new).__name__,
            "old_binding": old.binding, "new_binding": new.binding,
            "base_stage_binding": new.base_stage_binding,
            "backbone_lr_policy": deepcopy(new.backbone_lr_policy),
            "completed_updates_preserved": old.completed_updates,
            "total_updates_preserved": old.plan["total_updates"],
            "current_phase_preserved": old.current_phase,
            "scheduler_last_epoch_preserved": old.scheduler.last_epoch if old.scheduler is not None else None,
            "scheduler_step_count_preserved": old.scheduler.state_dict().get("_step_count") if old.scheduler is not None else None,
            "adam_state_parameter_count": state_count,
            "adam_state_action": "deep-cloned per Parameter; step and every moment retained",
            "calibration_and_best_preserved": True,
            "model_parameters_and_gradients_unchanged": True, "rng_unchanged": True,
            "epochs_restarted": False, "calibration_repeated": False,
            "groups": group_receipts,
        }
        new.lr_intervention_receipt = deepcopy(receipt)
        return receipt
    finally:
        restore_rng(rng)


def run_cpu_checks():
    """Real Adam moments and an independent per-parameter reference update."""
    from torch import nn

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.parent = nn.Module()
            for prefix in DEFAULT_PREFIXES:
                setattr(self.parent, prefix.split(".")[-1], nn.Linear(2, 2))
            self.parent.decoder = nn.Module()
            self.parent.decoder.proj = nn.Linear(2, 2)
            self.detail = nn.Linear(2, 1)
            self.fixed = nn.Parameter(torch.tensor(1.), requires_grad=False)

        def named_roles(self):
            return {
                "BASE": [(name, p) for name, p in self.named_parameters() if name.startswith("parent.")],
                "DETAIL": [(name, p) for name, p in self.named_parameters() if name.startswith("detail.")],
            }

        def forward(self, value):
            mixed = value
            for prefix in DEFAULT_PREFIXES:
                mixed = mixed + getattr(self.parent, prefix.split(".")[-1])(value).tanh() / 8
            return self.detail(self.parent.decoder.proj(mixed)) * self.fixed

    def equal_state(first, second):
        if torch.is_tensor(first):
            return torch.equal(first, second) and first.dtype == second.dtype and first.device == second.device
        if isinstance(first, dict):
            return first.keys() == second.keys() and all(equal_state(first[k], second[k]) for k in first)
        if isinstance(first, (list, tuple)):
            return len(first) == len(second) and all(equal_state(a, b) for a, b in zip(first, second, strict=True))
        return first == second

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7277)
        model = Dummy().double()
        config = {
            "protocol": "staged", "total_updates": 20, "phase_updates": [10, 4, 6],
            "phase_fractions": [.5, .2, .3], "affine_calibration": False,
            "phases": {
                "A": {"peak_lr": .03, "final_lr": .003, "warmup_updates": 3, "gradient_clip_norm": 0},
                "B": {"peak_lr": .02, "final_lr": .002, "warmup_updates": 1, "gradient_clip_norm": 0},
                "C": {"base_lr": .01, "detail_lr": .02, "decay_factor": .1, "gradient_clip_norm": 0}},
        }
        options = {"epochs": 2, "updates_per_epoch": 10, "train_ids": ["train0"], "test_ids": ["test0"]}
        old = StageController(model, config, **options)
        inputs = [torch.randn(5, 2, dtype=torch.float64) for _ in range(6)]
        targets = [torch.randn(5, 1, dtype=torch.float64) for _ in range(6)]
        for i in range(4):
            old.begin_update()
            old.optimizer.zero_grad(set_to_none=True)
            (model(inputs[i]) - targets[i]).square().mean().backward()
            old.step()
        assert all(state["step"].item() == 4 for state in old.optimizer.state.values())
        assert any(state["exp_avg"].count_nonzero() for state in old.optimizer.state.values())
        old.best = {"TRAIN_MONITOR:tiny:loss": {"value": .3, "completed_updates": 3}}
        old.calibration["marker"] = "retained"
        values = {name: value.clone() for name, value in model.state_dict().items()}
        states = {p: _clone(state) for p, state in old.optimizer.state.items()}
        rng = capture_rng()
        new = BackboneRateController(model, config, startup_mode="resume", **options)
        receipt = migrate_stage_controller(old, new)
        assert equal_state(rng, capture_rng())
        assert all(torch.equal(value, values[name]) for name, value in model.state_dict().items())
        assert all(equal_state(state, new.optimizer.state[p]) for p, state in states.items())
        assert new.completed_updates == old.completed_updates and new.current_phase == old.current_phase
        assert new.calibration == old.calibration and new.best == old.best
        assert new.scheduler.last_epoch == old.scheduler.last_epoch == 4
        assert new.scheduler.state_dict()["_step_count"] == old.scheduler.state_dict()["_step_count"]
        assert all(math.isclose(
            group["new_lr"], group["old_lr"] * (.1 if group["rate_kind"] == "shared" else 1.0))
                   for group in receipt["groups"])
        reference_model = deepcopy(model)
        reference_parameters = dict(reference_model.named_parameters())
        source_names = {id(p): name for name, p in model.named_parameters()}
        reference_groups = []
        for group in new.optimizer.param_groups:
            mapped = {key: _clone(value) for key, value in group.items() if key != "params"}
            mapped["params"] = [reference_parameters[source_names[id(p)]] for p in group["params"]]
            reference_groups.append(mapped)
        reference = torch.optim.AdamW(reference_groups)
        # Independent reference begins with the OLD real moments and the
        # intended per-parameter LR; it does not use migrated state tensors.
        for p, state in states.items():
            reference.state[reference_parameters[source_names[id(p)]]] = _clone(state)
        new.begin_update()
        new.optimizer.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        (model(inputs[4]) - targets[4]).square().mean().backward()
        (reference_model(inputs[4]) - targets[4]).square().mean().backward()
        new.step()
        reference.step()
        assert all(torch.equal(value, reference_model.state_dict()[name]) for name, value in model.state_dict().items())
        assert all(equal_state(state, reference.state[reference_parameters[source_names[id(p)]]])
                   for p, state in new.optimizer.state.items())
        saved = deepcopy(new.state_dict())
        resumed_model = deepcopy(model)
        resumed = BackboneRateController(resumed_model, config, startup_mode="resume", **options)
        resumed.load_state_dict(saved)
        assert resumed.scheduler.last_epoch == 5
        assert [g["lr"] for g in resumed.optimizer.param_groups] == [g["lr"] for g in new.optimizer.param_groups]
        assert all(equal_state(
            new.optimizer.state[p], resumed.optimizer.state[dict(resumed_model.named_parameters())[source_names[id(p)]]])
                   for p in new.optimizer.state)
        incompatible = BackboneRateController(
            resumed_model, config, backbone_factor=.2, startup_mode="resume", **options)
        try:
            incompatible.load_state_dict(saved)
        except ValueError:
            pass
        else:
            raise AssertionError("resume silently changed the recorded LR policy")
        try:
            new.load_state_dict(old.state_dict())
        except ValueError:
            pass
        else:
            raise AssertionError("legacy binding bypassed explicit migration")
        # C phase must retain DETAIL and projection rates while scaling BASE.
        stage_c = next(stage for stage in new.plan["stages"] if stage["phase"] == "C")
        opt_c, sched_c = new._build_optimizer(stage_c)
        assert all(g["lr_multiplier"] == 1 for g in opt_c.param_groups if g["role"] == "DETAIL")
        assert all(not (
            g["rate_kind"] == "shared" and any(p is model.parent.decoder.proj.weight for p in g["params"]))
                   for g in opt_c.param_groups)
        return {
            "passed": True, "real_adam_steps_before_migration": 4,
            "model_parameters_gradients_and_rng_unchanged": True,
            "adam_step_moments_and_other_state_exact": True,
            "completed_updates_phase_calibration_best_preserved": True,
            "scheduler_position_and_step_count_preserved": True,
            "backbone_lr_times_point_one_only": True,
            "next_update_matches_independent_reference_exactly": True,
            "new_binding_normal_resume_exact": True,
            "changed_policy_and_implicit_legacy_resume_rejected": True,
            "detail_and_projection_learning_rates_unchanged": True,
        }


if __name__ == "__main__":
    import json
    import sys
    if sys.argv[1:] == ["--cpu-check"]:
        torch.set_num_threads(2)
        print(json.dumps(run_cpu_checks(), indent=2))
    else:
        print("Import BackboneRateController and migrate_stage_controller; --cpu-check is CPU-only.")
