"""Joint direct-wave-operator and physics-guided update-strategy pretraining.

The epoch65 initializer supplies the architecture only. The entire actual
2000-step fitted model and grouped AdamW history replace it before return.
No observation enters the network. Complete400 labels supervise every task,
with the inherited receiver, IC and residual-AIS physics objectives unchanged.
Task weighting scales only newly arriving gradients. In the same training
transaction, a learned block controller changes the real operator update;
the complete post-update objective trains that controller by a one-step
conditional VJP. Historical gradients and optimizer moments are detached,
so this is explicitly truncated online learning, not a multi-step meta-gradient.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import paper_marmousi201_fullfit_20261020 as fullfit
from . import paper_marmousi_fullfield_20261020 as objective
from .paper201_time_axis_20261020 import native_time_axis
from .paper_marmousi_joint_ais_20261020 import PaperMarmousiJointAIS


ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "paper_assets_main_fresh_20261013_marmousi_20261019"
PARENT = ROOT / "runs/paper_marmousi201_single_source_fullfit_20261020/case_0/latest.pt"
PARENT_SHA = "60fb5ac2a2072e469d9be7250eb7058b848a5dba05954388a9bb825ea39f9db5"
CONSTRUCTOR = ASSET / "models/marmousi_epoch65.pt"
SCHEMA = "MARMOUSI_DIRECT400_ALLPARAM_CONTINUE_FIT2000_V1"
EXPECTED_TENSORS = 398
EXPECTED_PARAMETERS = 14552439

sha = objective.sha
cpu_tree = objective.cpu_tree
equal_tree = objective.equal_tree
immutable_digest = objective.immutable_digest
trainable_digest = objective.trainable_digest
fixed_supervision = objective.fixed_supervision
pressure_metrics = objective.pressure_metrics
assess = objective.assess
dg_geometry = objective.dg_geometry


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class BlockOptimizerConfig:
    block_embedding_dim: int = 8
    hidden_dim: int = 32
    min_scale: float = .1
    max_scale: float = 10.
    initial_scale: float = 1.
    epsilon: float = 1e-8
    total_steps: int = 2500

    def __post_init__(self):
        for name in ("block_embedding_dim", "hidden_dim", "total_steps"):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0, name + " must be positive integer")
        require(all(math.isfinite(getattr(self, name)) for name in ("min_scale", "max_scale", "initial_scale", "epsilon")),
                "finite block optimizer configuration required")
        require(0 < self.min_scale < self.initial_scale < self.max_scale and self.epsilon > 0,
                "positive bounded scale with interior initialization required")


class BlockLearnedOptimizer(torch.nn.Module):
    """Small shared controller; one real update multiplier per parameter block.

    Eight detached statistics describe current gradient/parameter/momentum/
    second-moment/update RMS and two directional cosines. Conditioning also
    uses sixteen physical task features, four detached loss observations and
    two step features. All parameter-sized paths are outside this small graph.
    initial_scale is the output-bias center: small nonzero final-layer weights
    initialize multipliers approximately there and let hidden layers receive
    first-step gradients. It is not an exact identity-initialization claim.
    """
    def __init__(self, parameter_names, config=None):
        super().__init__()
        self.parameter_names = tuple(parameter_names)
        require(bool(self.parameter_names) and len(set(self.parameter_names)) == len(self.parameter_names),
                "nonempty unique block names required")
        self.config = BlockOptimizerConfig() if config is None else config
        self.embedding = torch.nn.Embedding(len(self.parameter_names), self.config.block_embedding_dim)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(8 + 16 + 4 + 2 + self.config.block_embedding_dim, self.config.hidden_dim),
            torch.nn.Tanh(), torch.nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            torch.nn.Tanh(), torch.nn.Linear(self.config.hidden_dim, 1))
        torch.nn.init.normal_(self.embedding.weight, std=.02)
        torch.nn.init.normal_(self.network[-1].weight, std=1e-3)
        probability = (self.config.initial_scale-self.config.min_scale)/(self.config.max_scale-self.config.min_scale)
        torch.nn.init.constant_(self.network[-1].bias, math.log(probability/(1-probability)))

    def get_extra_state(self):
        return {"schema": SCHEMA + "_BLOCK_CONTROLLER", "parameter_names": self.parameter_names,
                "config": asdict(self.config), "conditioning_dimensions": [8, 16, 4, 2]}

    def set_extra_state(self, state):
        require(state == self.get_extra_state(), "block controller checkpoint identity/config mismatch")

    def forward(self, block_statistics, task_features, loss_features, step):
        require(type(step) is int and step >= 0, "nonnegative integer controller step required")
        for name, value, shape in (("block statistics", block_statistics, (len(self.parameter_names), 8)),
                                   ("task features", task_features, (16,)), ("loss features", loss_features, (4,))):
            require(isinstance(value, torch.Tensor) and value.shape == shape
                    and value.device == self.embedding.weight.device and value.dtype == self.embedding.weight.dtype,
                    name + " shape/dtype/device mismatch")
        require(torch.stack([torch.isfinite(value).all() for value in (block_statistics, task_features, loss_features)]).all(),
                "finite detached controller observations required")
        fraction = min((step+1)/self.config.total_steps, 1.)
        logarithm = min(math.log1p(step+1)/math.log1p(self.config.total_steps), 1.)
        common = torch.cat((task_features.detach().tanh(), loss_features.detach().tanh(),
                            task_features.new_tensor([fraction, logarithm])))
        block_ids = torch.arange(len(self.parameter_names), device=self.embedding.weight.device)
        features = torch.cat((block_statistics.detach(), common.expand(len(block_ids), -1),
                              self.embedding(block_ids)), dim=1)
        logits = self.network(features).flatten()
        return self.config.min_scale + (self.config.max_scale-self.config.min_scale)*logits.sigmoid()


def make_learned(model, cfg):
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    controller = BlockLearnedOptimizer(names, BlockOptimizerConfig(**cfg.get("block_learned_optimizer", {})))
    return controller.to(device=parameter.device, dtype=parameter.dtype)


def _active(model, learned, optimizer=None):
    active = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    require(tuple(name for name, _ in active) == learned.parameter_names, "active block identities changed")
    if optimizer is not None:
        optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        require(len(optimized) == len(active) and len({id(p) for p in optimized}) == len(active)
                and {id(p) for p in optimized} == {id(p) for _, p in active}, "base optimizer active coverage changed")
    return active


@torch.no_grad()
def _block_statistics(old, gradient, direction, moment, variance, epsilon):
    rms = lambda value: value.square().mean().add(epsilon**2).sqrt()
    gr, pr, mr, vr, dr = rms(gradient), rms(old), rms(moment), rms(variance), rms(direction)
    log_rms = lambda value: (value.log()/10.).tanh()
    cosine_m = (gradient*moment).mean()/(gr*mr)
    cosine_d = (gradient*direction).mean()/(gr*dr)
    relative_gradient = torch.log1p(gr/pr).tanh()
    return torch.stack((log_rms(gr), log_rms(pr), log_rms(mr), log_rms(vr), log_rms(dr),
                        cosine_m.clamp(-1, 1), cosine_d.clamp(-1, 1), relative_gradient))


def propose_update(model, baseAdamW, learned, task_features, loss_features, step):
    """Advance inherited AdamW history, then install a learned full-model step.

    d = native_AdamW(theta_old) - theta_old includes native per-group decay,
    learning rates and moment bias correction. The installed state is exactly
    theta_old + scale*d in FP32 arithmetic. Gradients/history/directions are
    detached observations. Only 398 scales and the controller retain a graph.
    Caller must treat propose + post-loss + finish_meta + AIS commits as one
    transaction and checkpoint only after the whole transaction succeeds.
    """
    require(isinstance(baseAdamW, torch.optim.AdamW), "native grouped AdamW required for inherited moment direction")
    active = _active(model, learned, baseAdamW)
    require(type(step) is int and step >= 0, "nonnegative integer controller step required")
    for name, value, shape in (("task features", task_features, (16,)), ("loss features", loss_features, (4,))):
        require(isinstance(value, torch.Tensor) and value.shape == shape
                and value.device == learned.embedding.weight.device and value.dtype == learned.embedding.weight.dtype
                and torch.isfinite(value).all(), "finite " + name + " with correct shape/dtype/device required")
    require(torch.stack([torch.isfinite(p).all() for p in learned.parameters()]).all(), "finite controller weights required")
    require(all(parameter.grad is not None for _, parameter in active), "every active block requires a gradient")
    require(torch.stack([torch.isfinite(parameter.grad).all() & torch.isfinite(parameter).all()
                         for _, parameter in active]).all(), "finite full-model gradient/state required")
    old = [parameter.detach().clone() for _, parameter in active]
    previous_steps = [float(baseAdamW.state[p]["step"]) if "step" in baseAdamW.state[p] else 0.
                      for _, p in active]
    baseAdamW.step()
    directions, statistics = [], []
    for old_value, (_, parameter) in zip(old, active):
        direction = (parameter.detach()-old_value).detach()
        state = baseAdamW.state[parameter]
        directions.append(direction)
        statistics.append(_block_statistics(old_value, parameter.grad.detach(), direction,
                                            state["exp_avg"].detach(), state["exp_avg_sq"].detach(),
                                            learned.config.epsilon))
    statistics = torch.stack(statistics)
    scales = learned(statistics, task_features, loss_features, step)
    require(torch.isfinite(scales).all(), "finite learned scales required")
    with torch.no_grad():
        for old_value, direction, scale, (_, parameter) in zip(old, directions, scales, active):
            parameter.copy_(old_value + scale.detach()*direction)
    baseAdamW.zero_grad(set_to_none=True)
    require(torch.stack([torch.isfinite(p).all() for _, p in active]).all(), "finite actual learned model update required")
    return {"schema": SCHEMA + "_ONE_STEP_CONTEXT", "names": learned.parameter_names,
            "scales": scales, "directions": directions, "block_statistics": statistics,
            "base_step_before": previous_steps,
            "base_step_after": [float(baseAdamW.state[p]["step"]) for _, p in active],
            "model_parameter_ids": tuple(id(p) for _, p in active), "finished": False,
            "step": step, "estimator": "one-step exact conditional VJP; truncated model/optimizer history"}


def finish_meta(model, learned, phiAdam, context, clip=1.):
    """Use post-update full400 gradients for the one-step controller VJP.

    No further main-model update is made. The controller update affects the
    next proposal. This is exact for this conditional one-step map, not an
    exact derivative through prior training steps or through observed gradients.
    """
    require(context.get("schema") == SCHEMA + "_ONE_STEP_CONTEXT" and not context["finished"],
            "fresh unconsumed one-step context required")
    require(math.isfinite(clip) and clip > 0, "positive finite phi gradient clip required")
    active = _active(model, learned)
    phi_parameters = [p for p in learned.parameters() if p.requires_grad]
    optimized_phi = [p for group in phiAdam.param_groups for p in group["params"]]
    require(len(optimized_phi) == len(phi_parameters)
            and len({id(p) for p in optimized_phi}) == len(phi_parameters)
            and {id(p) for p in optimized_phi} == {id(p) for p in phi_parameters},
            "controller optimizer must cover every active phi parameter once")
    require(context["names"] == learned.parameter_names
            and context["model_parameter_ids"] == tuple(id(p) for _, p in active), "one-step model/context mismatch")
    require(all(p.grad is not None for _, p in active), "post-update gradient required for every block")
    require(torch.stack([torch.isfinite(p.grad).all() for _, p in active]).all(), "finite post-update gradients required")
    coefficients = torch.stack([(p.grad.detach()*direction).sum()
                                for (_, p), direction in zip(active, context["directions"])]).detach()
    require(torch.isfinite(coefficients).all(), "finite block directional VJP required")
    phiAdam.zero_grad(set_to_none=True)
    surrogate = (context["scales"]*coefficients).sum()
    surrogate.backward()
    require(all(p.grad is not None for p in phi_parameters)
            and torch.stack([torch.isfinite(p.grad).all() for p in phi_parameters]).all(),
            "finite controller gradients required")
    grad_norm = torch.nn.utils.clip_grad_norm_(phi_parameters, clip, error_if_nonfinite=True)
    phiAdam.step()
    require(torch.stack([torch.isfinite(p).all() for p in phi_parameters]).all(), "finite updated controller required")
    for _, parameter in active:
        parameter.grad = None
    context["finished"] = True
    return {"controller_gradient_norm_before_clip": float(grad_norm),
            "scale_min": float(context["scales"].detach().min()),
            "scale_max": float(context["scales"].detach().max()),
            "block_vjp_l2": float(torch.linalg.vector_norm(coefficients)),
            "updated_model_blocks": len(active), "controller_step": context["step"]+1,
            "estimator": context["estimator"], "main_model_moved_again_in_finish": False}


def create_operator(cfg, device="cpu"):
    """Return latest fitted model, restored AdamW and explicit migration receipt.

    cfg requires native_config. continuation_checkpoint/SHA and constructor
    paths may be explicitly supplied, but their exact identities are fixed to
    the actual FIT2000 parent and original epoch65 architecture initializer.
    The inherited optimizer learning rates, decay, moments and step2000 are
    restored; no learning-rate reset or new optimizer clock is introduced.
    """
    parent = Path(cfg.get("continuation_checkpoint", PARENT)).resolve()
    expected_sha = cfg.get("continuation_checkpoint_sha256", PARENT_SHA)
    require(expected_sha == PARENT_SHA and sha(parent) == PARENT_SHA, "actual fitted2000 parent SHA required")
    checkpoint_receipt = json.loads((parent.parent / "CHECKPOINT.json").read_text())
    require(checkpoint_receipt["sha256"] == PARENT_SHA and checkpoint_receipt["steps"] == 2000
            and checkpoint_receipt["all_state_readback"], "fitted parent receipt mismatch")
    saved = torch.load(str(parent), map_location="cpu", mmap=True, weights_only=False)
    require(saved["schema"] == fullfit.SCHEMA and saved["steps"] == 2000
            and saved["checkpoint_parent_sha256"] == fullfit.PAPER_CHECKPOINT_SHA256,
            "complete original201 FIT2000 state required")
    require(saved["parallel_frame_batch"] == 400, "parent complete400 execution receipt required")
    native = copy.deepcopy(cfg["native_config"])
    native.update(parent_checkpoint=str((ASSET / "models/initializers/marmousi.pt").resolve()),
                  dense_time_block=400, refiner_time_block=400, checkpoint_queries=True)
    model = fullfit.build(native, cfg.get("constructor_checkpoint", CONSTRUCTOR), device="cpu")
    model.load_state_dict(saved["model"], strict=True)
    require(equal_tree(cpu_tree(model.state_dict()), saved["model"]), "latest full model migration differs")
    selection = fullfit.selection(model)
    require(selection["tensors"] == EXPECTED_TENSORS and selection["parameters"] == EXPECTED_PARAMETERS,
            "all398 tensors /14552439 inherited active parameters required")
    require(selection == saved["selection"], "inherited trainability/parameter order changed")
    require(vars(model.normalizer.metadata) == saved["normalizer"], "fixed parent RMS/normalizer changed")
    require(torch.equal(model.native_times_s.cpu(), native_time_axis()), "canonical801 teacher times required")
    canonical = model.native_times_s.detach().clone()
    legacy = torch.arange(801, dtype=torch.float32) * .00125
    indices = list(model.full_native_frames)
    old_times = saved["time_s"]
    require(old_times.dtype == torch.float32 and old_times.shape == (400,), "parent400 physical time record required")
    if torch.equal(old_times, canonical[indices]):
        original_axis = canonical
    else:
        require(torch.equal(old_times, legacy[indices]), "unexpected parent time protocol")
        original_axis = legacy
    # The known nonpersistent historical time-buffer error is the sole allowed
    # immutable-state migration. Validate the original digest with its exact
    # old axis, then restore the repaired teacher authority before any query.
    try:
        model.native_times_s.copy_(original_axis)
        require(immutable_digest(model) == saved["immutable_digest"], "parent frozen tensors/buffers changed")
    finally:
        model.native_times_s.copy_(canonical)
    model.to(device)
    optimizer = fullfit.make_optimizer(model)
    optimizer.load_state_dict(saved["optimizer"])
    require(equal_tree(cpu_tree(optimizer.state_dict()), saved["optimizer"]), "AdamW full state migration differs")
    require(len(optimizer.state) == EXPECTED_TENSORS, "every inherited active tensor needs AdamW history")
    for parameter, state in optimizer.state.items():
        require(float(state["step"].item()) == 2000., "inherited AdamW step2000 required")
        for name in ("exp_avg", "exp_avg_sq"):
            require(state[name].shape == parameter.shape and state[name].dtype == parameter.dtype
                    and torch.isfinite(state[name]).all(), "invalid inherited AdamW moment: " + name)
    masked = dict(model.named_parameters())[fullfit.MASKED_INPUT_NAME]
    require(bool(masked._backward_hooks), "original zero-IC gradient mask hook missing")
    groups = fullfit.optimizer_selection(model, optimizer)
    require([row["weight_decay"] for row in groups if row["name"] == "refiner_masked_input"] == [0.],
            "masked-input zero weight decay changed")
    model.train()
    # Preserve old task's sampler/RNG/assessment lineage as CPU history only;
    # new source tasks require their own samplers, never the old task's EMA.
    history_keys = ("sampler", "torch_rng", "cuda_rng", "numpy_rng", "python_rng", "source",
                    "assessments", "consecutive", "reference_sha256", "config_sha256")
    model.__dict__["_direct400_parent_history"] = cpu_tree({key: saved[key] for key in history_keys})
    receipt = dict(schema=SCHEMA, selection=selection, names=selection["names"],
                   tensors=EXPECTED_TENSORS, parameters=EXPECTED_PARAMETERS, parent_checkpoint=str(parent),
                   parent_sha256=PARENT_SHA, inherited_optimizer_steps=2000, inherited_fitted_task_steps=2000,
                   model_exact_latest_parent=True, optimizer_exact_latest_parent=True,
                   normalizer_unchanged=True, constructor_epoch65_values_replaced=True,
                   optimizer_groups=groups, parent_immutable_digest=saved["immutable_digest"],
                   current_immutable_digest=immutable_digest(model),
                   parent_registered_time_values_repaired=int((old_times != canonical[indices]).sum()),
                   canonical_teacher_native801=True, complete400_frame_batch=400,
                   direct_prediction_inputs=["velocity201", "source5"], measured_pressure_inputs=0,
                   parent_sampler_preserved_as_history=True, new_task_sampler_policy="cold per-source AIS")
    model.__dict__["_direct400_migration_receipt"] = receipt
    require(sha(parent) == PARENT_SHA, "parent checkpoint changed during migration")
    return model, optimizer, receipt


def parent_history(model):
    return cpu_tree(model.__dict__["_direct400_parent_history"])


def episode_inputs(model, item, device=None):
    """Normalize all400 Pa labels with the inherited fixed RMS, never refit it."""
    require(item["pressure_scale_pa"] == model.normalizer.metadata.pressure_scale_pa,
            "teacher pressure scale differs from inherited fixed RMS")
    device = model.device if device is None else device
    c = torch.from_numpy(np.array(item["velocity"], copy=True)).to(device)[None, None]
    f = torch.from_numpy(np.array(item["source"], copy=True)).to(device).reshape(1, 5)
    truth = torch.from_numpy(np.array(item["pressure_pa"], copy=True)).to(device)
    require(c.dtype == f.dtype == truth.dtype == torch.float32, "original FP32 inputs required")
    require(c.shape == (1, 1, 201, 201) and truth.shape == (400, 201, 201), "complete native201/all400 data required")
    require(np.array_equal(np.asarray(item["native_frame_indices"]), np.asarray(model.full_native_frames)),
            "registered400 frame IDs changed")
    require(np.array_equal(np.asarray(item["times"]), model.full_times_s.detach().cpu().numpy()),
            "exact teacher400 physical times required")
    scale = model.normalizer.metadata.pressure_scale_pa * float(f[0, 4])
    require(math.isfinite(scale) and scale > 0, "positive inherited pressure scale required")
    truth = truth / scale
    require(torch.isfinite(c).all() and (c > 0).all() and torch.isfinite(f).all()
            and torch.isfinite(truth).all(), "finite complete400 instance required")
    return c, f, truth


def backward_task(model, c, f, truth, sampler, weights=None, plan=None):
    """Accumulate one weighted all400 task; caller owns zero/step/AIS commit.

    weights = {weight: .25, receiver_weight: .01, physics_weight: .001,
               ic_weight: .01} performs one quarter of a four-task objective.
    Original statistics and AIS feedback remain unscaled. The returned feedback
    contains the sampler's pending plan; commit it only after successful outer
    optimizer.step, using sampler.commit(plan, selected_times, energies).
    """
    settings = dict(weight=1., receiver_weight=.01, physics_weight=.001, ic_weight=.01)
    if weights is not None:
        require(set(weights) <= set(settings), "unknown objective weight")
        settings.update(weights)
    require(all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                for value in settings.values()) and settings["weight"] > 0, "finite nonnegative objective weights required")
    supervision = fixed_supervision(truth)
    target_receiver = truth[:, 2, [50, 100, 150]]
    require(torch.isfinite(truth).all() and target_receiver.square().mean() > 0,
            "finite full400 labels and positive receiver energy required")
    op, valid = dg_geometry(model, f)
    require(torch.equal(sampler.valid_mask, valid.detach().cpu().flatten()), "task/source AIS mask mismatch")
    plan = sampler.plan() if plan is None else plan
    handles = []
    try:
        # AccumulateGrad hooks see only each newly computed gradient. Scaling
        # .grad after backward would also rescale previous tasks and be wrong.
        if settings["weight"] != 1.:
            for parameter in model.parameters():
                if parameter.requires_grad:
                    handles.append(parameter.register_hook(lambda gradient: gradient * settings["weight"]))
        metrics, selected, energies = objective.backward(
            model, c, f, truth, supervision, op, valid, plan,
            receiver_weight=settings["receiver_weight"], physics_weight=settings["physics_weight"],
            ic_weight=settings["ic_weight"])
    finally:
        for handle in handles:
            handle.remove()
    metrics.update(task_weight=settings["weight"], weighted_loss=settings["weight"] * metrics["loss"],
                   direct_cf_prediction=True, known8_copied=False)
    return metrics, {"plan": plan, "selected_times": selected, "energies": energies}
