"""Physics-informed coarse and rank-4 transfer for native-401 SCNO.

This entry never chooses observations or starts before completed pretraining.
The physical objective uses registered adjacent rows, not a continuous-time
decoder: delta=2.5 ms, 399 interior centers and a second-order one-sided IC
derivative. Legacy continuous-query fine-tuning remains a separate entry.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
import numpy as np

from finetune_v2 import stratified_centers
from lora_v2 import CompensatedRank4, prepared_second_affine, second_affine_tensors


SCHEMA = "SCNO_UL_NATIVE401_LORA_TRANSFER_V1"
MODEL_SCHEMA = "scno_time_conditioned_repair_v1"
WEIGHTS = {"obs": 1., "ic": .01, "lwc": .001, "bc": .01}
DELTA = .0025


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def native_axis(axis):
    value = torch.as_tensor(axis).detach().cpu().float()
    expected = torch.arange(401, dtype=torch.float32) * DELTA
    if value.shape != (401,) or not torch.isfinite(value).all() or not torch.allclose(value, expected, atol=1e-7, rtol=0):
        raise ValueError("transfer requires the complete native 401 rows, 0--1 s at 2.5 ms")
    return value


def validate_checkpoint(payload, config, binding, *, check_only=False, completion=None):
    if (payload.get("schema") != MODEL_SCHEMA or payload.get("config") != config
            or payload.get("dataset_binding") != binding):
        raise ValueError("corrected checkpoint requires exact config and sealed warm-corpus binding")
    settings = payload.get("repair_settings", {})
    if (settings.get("nonlinear_details") is not False or settings.get("coarsest_base") is not False
            or settings.get("basis_size") != 800 or settings.get("base_hidden_width") != 800):
        raise ValueError("expected the actual base800/basis800 repaired SCNO roles")
    semantics = payload.get("parent_semantics", {})
    if semantics.get("output_frames") != 401 or semantics.get("active_grid") != [200, 201]:
        raise ValueError("checkpoint physical axis or active pressure grid differs")
    if not check_only:
        if payload.get("epochs_completed") != 200 or payload.get("records_completed_in_epoch") != 0:
            raise ValueError("target-instance fitting is gated on 200 completed pretraining epochs")
        if not isinstance(completion, dict) or completion.get("state") != "COMPLETE" or completion.get("epochs_completed") != 200:
            raise ValueError("provide this run's COMPLETE/e200 STATUS.json before actual fitting")


def load_pretrained(checkpoint, expected_sha256, training_config, warm_protocol, device="cpu",
                    *, check_only=False, completion_receipt=None):
    """Strict, shape-and-semantic restoration; check-only never optimizes."""
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or sha256(checkpoint) != expected_sha256.lower()):
        raise ValueError("pretrained file SHA-256 mismatch")
    from data_v2 import open_corpus
    from model_v2 import build_model
    from method_repair_model import CorrectedSCNOReadout
    from prepare_ul_warm_protocol import load_ul_warm_protocol

    config = json.loads(Path(training_config).read_text(encoding="utf-8"))
    original = open_corpus(config)
    corpus = load_ul_warm_protocol(original, warm_protocol)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    completion = (json.loads(Path(completion_receipt).read_text(encoding="utf-8"))
                  if completion_receipt is not None else None)
    validate_checkpoint(payload, config, corpus.binding, check_only=check_only, completion=completion)
    if not check_only:
        # STATUS belongs to this exact run directory; an unrelated COMPLETE
        # status is insufficient authorization to fit an incomplete checkpoint.
        if Path(completion_receipt).resolve().parent != Path(checkpoint).resolve().parent:
            raise ValueError("completion receipt and complete resume checkpoint must share their run directory")
        updates = payload.get("stage_schedule", {}).get("completed_updates")
        if updates is not None and completion.get("accepted_updates") != updates:
            raise ValueError("COMPLETE status and checkpoint describe different accepted updates")
    model = CorrectedSCNOReadout.from_checkpoint(build_model(config, original, torch.device("cpu")), payload)
    native_axis(model.full_times_s)
    model.to(device).requires_grad_(False).eval()
    model._ul_completed_pretraining = not check_only
    return model, {"checkpoint_sha256": expected_sha256.lower(),
                   "epochs_completed": payload.get("epochs_completed"),
                   "run_binding_sha256": payload.get("run_binding_sha256"),
                   "completed_gate": not check_only,
                   "loader": "strict CorrectedSCNOReadout.from_checkpoint"}


def detail_heads(model):
    if (getattr(model, "variant", None) != MODEL_SCHEMA or model.nonlinear_details
            or model.coarsest_base or model.basis_size != 800):
        raise ValueError("LoRA must attach to the reviewed factorized repaired pressure path")
    decoder = model.parent.decoder
    if decoder.detail_latent != 800 or len(decoder.detail_heads) != 3:
        raise ValueError("expected three live rank800 detail coefficient predictors")
    for head in decoder.detail_heads:
        if (not isinstance(head, nn.Sequential) or len(head) != 5
                or not isinstance(head[0], nn.Linear) or not isinstance(head[1], nn.SiLU)
                or not isinstance(head[3], nn.SiLU) or not isinstance(head[4], nn.Linear)
                or tuple(head[0].weight.shape[:1]) != (32,) or head[4].in_features != 32
                or not isinstance(head[2], (nn.Linear, CompensatedRank4))):
            raise ValueError("unrecognized actual second-hidden/detail-output module semantics")
        layer = head[2] if isinstance(head[2], nn.Linear) else head[2].base
        if layer.weight.shape != (32, 32) or layer.bias is None:
            raise ValueError("actual second hidden affine must be 32 by 32 with bias")
    return decoder.detail_heads


def enable_lora(model, scaling=1.):
    """Compensate each pretrained matrix; never initialize an existing factor."""
    heads = detail_heads(model)
    edits = [CompensatedRank4.insert(head[2], scaling) if isinstance(head[2], nn.Linear) else head[2]
             for head in heads]
    model.requires_grad_(False).eval()
    model.compiled_readout = None
    for head, layer in zip(heads, edits):
        head[2] = layer
        layer.U.requires_grad_(True)
        layer.V.requires_grad_(True)
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if len(parameters) != 6 or sum(p.numel() for p in parameters.values()) != 768:
        raise AssertionError("only the six live rank-four factors (768 scalars) may be trained")
    return parameters


def adapter_state(model):
    return {f"parent.decoder.detail_heads.{i}.2.{name}": tensor.detach().cpu().clone()
            for i, head in enumerate(detail_heads(model))
            for name, tensor in second_affine_tensors(head[2]).items()}


def fixed_extra_pressure(value, device):
    if value is None:
        return None
    result = torch.as_tensor(value, dtype=torch.float32, device=device).detach().clone()
    if (result.shape != (401, 201, 201) or not torch.isfinite(result).all()
            or result[:, 0].count_nonzero()):
        raise ValueError("fixed extra_pressure must be finite normalized [401,201,201] with an exact zero top")
    return result


def predict_adapted(model, velocity, source, frame_indices=None, *, normalized=True):
    """Predict with the separately recorded fixed correction, if present."""
    return decode_adapted(model, model.encode(*model._inputs(velocity, source)), frame_indices, normalized=normalized)


def decode_adapted(model, context, frame_indices=None, *, normalized=True):
    result = model.decode(context, frame_indices, normalized=True)
    extra = getattr(model, "_ul_fixed_extra_pressure", None)
    if extra is not None:
        rows = (torch.arange(401, device=model.device) if frame_indices is None
                else torch.as_tensor(frame_indices, device=model.device, dtype=torch.long))
        if rows.ndim != 1:
            raise ValueError("the explicit single-instance correction expects shared native indices")
        result = result + extra[rows][None]
    if not normalized:
        result = model.normalizer.decode_pressure(result, context.source[:, 4])
    return result
def restore_adapter(model, payload, parent_sha256):
    if payload.get("schema") != SCHEMA or payload.get("parent_checkpoint_sha256") != parent_sha256:
        raise ValueError("adapter requires its exact authenticated completed pretrained checkpoint")
    state = payload["adapter_state"]
    expected = set()
    edits = []
    for i, head in enumerate(detail_heads(model)):
        prefix = f"parent.decoder.detail_heads.{i}.2."
        keys = {prefix + k for k in ("base.weight", "base.bias", "U", "V", "scaling")}
        expected.update(keys)
        if not keys.issubset(state):
            raise ValueError("incomplete saved live low-rank affine")
        layer, tensors = prepared_second_affine(head[2], {k[len(prefix):]: state[k] for k in keys})
        edits.append((head, layer, tensors))
    if set(state) != expected:
        raise ValueError("unexpected adapter tensors outside the live three-layer registry")
    for head, layer, tensors in edits:
        layer.load_state_dict(tensors, strict=True)
        head[2] = layer
    enable_lora(model)  # Existing factors are continued, without SVD.
    correction = payload.get("fixed_extra_pressure")
    factors = payload.get("fixed_extra_pressure_factors")
    if factors is not None:
        temporal, spatial = (tensor.to(model.device, dtype=torch.float32) for tensor in factors)
        if (correction is not None or temporal.ndim != 2 or temporal.shape[0] != 401
                or spatial.shape != (temporal.shape[1], 201, 201)):
            raise ValueError("expected one fixed correction representation with a shared time rank")
        correction = (temporal @ spatial.flatten(1)).reshape(401, 201, 201)
    model._ul_fixed_extra_pressure = fixed_extra_pressure(correction, model.device)
    return model


def residual_operators():
    path = Path(__file__).resolve().parent / "vendor/residual_support/operators.py"
    spec = importlib.util.spec_from_file_location("scno_ul_native401_registered_operators", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initial_derivative(first_three, delta=DELTA):
    return (-3 * first_three[0].double() + 4 * first_three[1].double() - first_three[2].double()) / (2 * delta)


def synchronized_time(model):
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    return perf_counter()


class NativePhysicalObjective:
    """The four registered losses, with a declared native discrete-time IC."""

    def __init__(self, model, velocity, source, observed_pa, observation_indices,
                 *, seed=37, weights=None, energy_floor=1e-24, extra_pressure=None):
        native_axis(model.full_times_s)
        self.model = model
        value = (extra_pressure if extra_pressure is not None
                 else getattr(model, "_ul_fixed_extra_pressure", None))
        self.set_extra_pressure(value)
        self.weights = dict(WEIGHTS if weights is None else weights)
        if (set(self.weights) != set(WEIGHTS) or any(not math.isfinite(w) or w < 0 for w in self.weights.values())
                or not math.isfinite(energy_floor) or energy_floor <= 0):
            raise ValueError("finite nonnegative four-term weights and a positive energy floor are required")
        v, s = model._inputs(velocity, source)
        if len(v) != 1 or len(s) != 1 or tuple(v.shape[-2:]) != (201, 201):
            raise ValueError("supply one target instance on the physical 201 by 201 grid")
        if not torch.isfinite(v).all() or not (v > 0).all() or not torch.isfinite(s).all() or s[0, 2] <= 0 or s[0, 4] <= 0:
            raise ValueError("velocity, source frequency and amplitude must be finite and positive")
        indices = torch.as_tensor(observation_indices, device=model.device)
        if (indices.dtype not in (torch.int32, torch.int64) or indices.ndim != 1 or not len(indices)
                or (indices < 0).any() or (indices > 400).any() or not (indices[1:] > indices[:-1]).all()):
            raise ValueError("explicit observation_indices must be strictly increasing integer native rows in [0,400]")
        observed = torch.as_tensor(observed_pa, dtype=torch.float32, device=model.device)
        if observed.shape != (len(indices), 201, 201) or not torch.isfinite(observed).all():
            raise ValueError("observed_pa must contain only the explicitly supplied registered pressure observations")
        self.observations = model.normalizer.encode_pressure(observed[None], s[:, 4])[0].detach()
        if self.observations.double().square().sum() <= 0:
            raise ValueError("observation normalization requires nonzero supplied pressure energy")
        self.observation_indices = indices.long()
        self.frequency = float(s[0, 2])
        self.velocity = v[0, 0].detach()
        self.operators = residual_operators()
        self.mask = self.operators.make_valid_mask((201, 201), source=s[0, :2].tolist(), halo=8, radius=90., device=model.device)
        self.norm = self.operators.residual_scale_squared(self.observations, self.frequency, energy_floor)
        self.energy_floor = energy_floor
        self.observation_energy = self.observations.double().square().sum().clamp_min(energy_floor)
        train, probe = stratified_centers(np.arange(1, 400, dtype=np.int64), 16, 8, seed)
        self.support = {name: (torch.as_tensor(c, device=model.device, dtype=torch.long),
                               torch.as_tensor(w, device=model.device, dtype=torch.float64))
                        for name, (c, w) in (("train", train), ("probe", probe))}
        self.requests = {}
        for name, (centers, _) in self.support.items():
            requested = torch.cat((self.observation_indices, centers.new_tensor([0, 1, 2])))
            if self.weights["lwc"]:
                requested = torch.cat((requested, centers - 1, centers, centers + 1))
            self.requests[name] = requested.unique(sorted=True, return_inverse=True)
        self.request_union = torch.cat([rows for rows, _ in self.requests.values()]).unique(sorted=True)
        self.cached_readout = None
        self.pde_defect = None
        # The entire encoder/BASE is frozen. Reusing this input-only context is
        # valid; LoRA remains downstream and is re-evaluated on every decode.
        with torch.no_grad():
            self.context = model.encode(v, s)

    def set_extra_pressure(self, value):
        self.extra_pressure = fixed_extra_pressure(value, self.model.device)
        self.model._ul_fixed_extra_pressure = self.extra_pressure
        self.extra_pressure_factors = None

    def decode(self, rows):
        decode = self.model.decode if self.cached_readout is None else self.cached_readout.decode
        return decode(self.context, rows, normalized=True)[0]

    @torch.no_grad()
    def set_propagation_defect(self, solver_pressure):
        # The independently propagated solution supplies the consistency
        # defect of the short native-time stencil at its stored rows.
        self.pde_defect = {
            name: self.operators.lwc_residual(solver_pressure[centers - 1], solver_pressure[centers],
                                              solver_pressure[centers + 1], self.velocity,
                                              h=10., delta=DELTA)[:, self.mask]
            for name, (centers, _) in self.support.items()
        } if self.weights["lwc"] else None

    def loss(self, which="train"):
        unique, inverse = self.requests[which]
        decoded = self.decode(unique)
        if self.extra_pressure is not None:
            decoded = decoded + self.extra_pressure[unique]
        return self.loss_decoded(decoded, inverse, which)

    def loss_decoded(self, decoded, inverse, which):
        centers, weights = self.support[which]
        values = decoded[inverse]
        nobs = len(self.observation_indices)
        predicted = values[:nobs]
        initial = values[nobs:nobs + 3]
        # These are exactly loss_observation/loss_initial/loss_bc from
        # vendor/fast_objective.py, without importing its legacy model runtime.
        obs = (predicted.double() - self.observations.double()).square().sum() / self.observation_energy
        ut0 = initial_derivative(initial)
        ic = (initial[0].double().square() + (ut0 / (2 * math.pi * self.frequency)).square()).mean()
        bc = predicted[:, 0].double().square().mean()
        if bc.detach().item() != 0.:
            raise AssertionError("the final zero-extended free surface must be exactly zero")
        if self.weights["lwc"]:
            fields = values[nobs + 3:]
            residual = self.operators.lwc_residual(*fields.split(len(centers)), self.velocity,
                                                   h=10., delta=DELTA)[:, self.mask]
            if self.pde_defect is not None:
                residual = residual - self.pde_defect[which]
            lwc = (residual.square().mean(1) * weights).sum() / self.norm
        else:
            lwc = obs.new_zeros(())
        terms = {"obs": obs, "ic": ic, "lwc": lwc, "bc": bc}
        total = sum(self.weights[k] * value for k, value in terms.items())
        if not torch.isfinite(total):
            raise FloatingPointError("nonfinite native physical objective")
        return total, {k: float(value.detach()) for k, value in terms.items()}

    def protocol(self):
        return {"frames": 401, "time_interval_s": [0., 1.], "delta_s": DELTA,
                "admissible_residual_center_count": 399, "admissible_residual_indices": [1, 399],
                "training_residual_indices": self.support["train"][0].cpu().tolist(),
                "probe_residual_indices": self.support["probe"][0].cpu().tolist(),
                "training_time_weights": self.support["train"][1].cpu().tolist(),
                "probe_time_weights": self.support["probe"][1].cpu().tolist(),
                "observation_indices": self.observation_indices.cpu().tolist(),
                "initial_derivative": "(-3*u[0]+4*u[1]-u[2])/(2*0.0025), second-order one-sided",
                "PDE": "registered LWC: Dtt-L-delta^2*L(L)/12; h=10m, adjacent native rows",
                "PDE_mask": "eight-node halo excluded; source distance >90m",
                "PDE_constraint_enabled": self.weights["lwc"] > 0,
                "PDE_rhs": ("native residual of independent 10 m propagated solution" if self.pde_defect is not None
                            else "zero on the source-excluded interior"),
                "weights": self.weights, "reference_used": "explicit observed_pa only",
                "fixed_extra_pressure": self.extra_pressure is not None,
                "extra_pressure_units": "normalized pressure; fixed, detached, zero top"}


@torch.no_grad()
def fit_coarse_pressure(objective, rank):
    from ul_coarse_physics_transfer import pretrained_coarse_modes, solve_coarse_physics_transfer

    if not getattr(objective.model, "_ul_completed_pretraining", False):
        raise ValueError("coarse fitting requires the COMPLETE/e200 parent")
    modes = pretrained_coarse_modes(objective.model, objective.context, rank)
    rows = objective.request_union
    correction, receipt = solve_coarse_physics_transfer(
        objective.decode(rows), modes, objective.velocity, objective.context.source,
        objective.observations, objective.observation_indices,
        *objective.support["train"], *objective.support["probe"], objective.operators,
        {"weights": objective.weights, "pde_energy_floor": objective.energy_floor,
         "observation_energy_floor": objective.energy_floor}, baseline_indices=rows, delta=DELTA)
    objective.set_extra_pressure(correction)
    return receipt


@torch.no_grad()
def insert_and_check(model, velocity, source, scaling=1.):
    """Check every native pressure row before and after the real insertion."""
    native_axis(model.full_times_s)
    v, s = model._inputs(velocity, source)
    context = model.encode(v, s)
    before = torch.cat([model.decode(context, rows, normalized=True).cpu()
                        for rows in torch.arange(401, device=model.device).split(32)], dim=1)
    matrices = [head[2].weight.detach().clone() for head in detail_heads(model)]
    initialization_started = synchronized_time(model)
    parameters = enable_lora(model, scaling)
    initialization_seconds = synchronized_time(model) - initialization_started
    after = torch.cat([model.decode(context, rows, normalized=True).cpu()
                       for rows in torch.arange(401, device=model.device).split(32)], dim=1)
    matrix_error = max(float((head[2].effective_weight() - matrix).abs().max()) for head, matrix in zip(detail_heads(model), matrices))
    torch.testing.assert_close(after, before, atol=5e-6, rtol=2e-5)
    torch.testing.assert_close(initial_derivative(after[0, :3]), initial_derivative(before[0, :3]), atol=.004, rtol=2e-5)
    return parameters, {"all_401_rows_checked": True,
                        "matrix_max_absolute_difference": matrix_error,
                        "pressure_max_absolute_difference": float((after - before).abs().max()),
                        "pressure_relative_l2_difference": float((after.double() - before.double()).norm() / before.double().norm().clamp_min(1e-30)),
                        "initial_discrete_derivative_max_absolute_difference": float((initial_derivative(after[0, :3]) - initial_derivative(before[0, :3])).abs().max()),
                        "trainable_factor_scalars": sum(p.numel() for p in parameters.values()),
                        "initialization_seconds": initialization_seconds}


def fit(model, objective, parent_sha256, *, updates=50, learning_rate=3e-4, gradient_clip=1., selection="probe"):
    if not getattr(model, "_ul_completed_pretraining", False):
        raise ValueError("actual fitting requires a strictly loaded COMPLETE/e200 parent; check-only cannot fit")
    if type(updates) is not int or updates < 1 or not 0 < learning_rate < math.inf or not 0 < gradient_clip < math.inf:
        raise ValueError("positive update budget, learning rate and gradient clip are required")
    if selection not in ("probe", "final"):
        raise ValueError("adapter selection must be probe or final")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if len(parameters) != 6 or sum(p.numel() for p in parameters) != 768:
        raise ValueError("insert the six live U,V factors before fitting")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.)
    with torch.no_grad():
        initial_train, train_parts = objective.loss("train")
        initial_probe, probe_parts = objective.loss("probe")
    best = {"probe": float(initial_probe), "update": 0, "state": adapter_state(model)}
    history = [{"update": 0, "train_total": float(initial_train), "probe_total": float(initial_probe), "train": train_parts, "probe": probe_parts}]
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective.loss("train")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("a live LoRA factor lacks a finite gradient")
        gradnorm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        if selection == "final" and update < updates:
            continue
        with torch.no_grad():
            train_loss, tp = objective.loss("train")
            probe_loss, pp = objective.loss("probe")
        history.append({"update": update, "train_total": float(train_loss), "probe_total": float(probe_loss),
                        "gradient_norm": float(gradnorm), "train": tp, "probe": pp})
        if selection == "final" or (float(probe_loss) < best["probe"] and float(train_loss) <= float(initial_train)):
            best = {"probe": float(probe_loss), "update": update, "state": adapter_state(model)}
    factors = objective.extra_pressure_factors
    selected = {"schema": SCHEMA, "parent_checkpoint_sha256": parent_sha256,
                "adapter_state": best["state"], "selected_update": best["update"],
                "fixed_extra_pressure": (None if objective.extra_pressure is None or factors is not None
                                         else objective.extra_pressure.detach().cpu().clone()),
                "fixed_extra_pressure_factors": (None if factors is None else tuple(tensor.detach().cpu() for tensor in factors)),
                "attempted_updates": updates, "selection": selection, "objective": objective.protocol(), "history": history}
    restore_adapter(model, selected, parent_sha256)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--warm-protocol", required=True, type=Path)
    parser.add_argument("--completion-receipt", type=Path)
    parser.add_argument("--instance", required=True, type=Path,
                        help="NPZ: velocity_mps, source_parameters, observed_pa, observation_indices")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true", help="load/insertion checks only; never train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variant", choices=("lora", "coarse", "coarse-lora", "propagation-lora"), default="lora")
    parser.add_argument("--solver-source", type=Path, help="LWC solver source directory for propagation-lora")
    parser.add_argument("--coarse-rank", type=int, default=6)
    parser.add_argument("--pde-weight", type=float, default=WEIGHTS["lwc"],
                        help="use zero for the matched no-PDE ablation")
    parser.add_argument("--uncached", action="store_true", help="use regular pressure readout for a timing comparison")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--selection", choices=("probe", "final"), default="probe")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sampling-seed", type=int, default=37)
    parser.add_argument("--extra-pressure", type=Path,
                        help="optional fixed normalized [401,201,201] NPY from a separately validated physics solve")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new transfer output directory")
    if not args.check_only and args.completion_receipt is None:
        parser.error("actual fitting requires the completed parent run's --completion-receipt")
    if args.variant != "lora" and args.extra_pressure is not None:
        parser.error("an internal and an external fixed pressure correction cannot be combined")
    if args.variant == "propagation-lora" and args.solver_source is None:
        parser.error("provide the propagation implementation with --solver-source")
    if args.coarse_rank < 1 or not math.isfinite(args.pde_weight) or args.pde_weight < 0:
        parser.error("coarse rank must be positive and the PDE weight finite and nonnegative")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = perf_counter()
    model, load = load_pretrained(args.checkpoint, args.sha256, args.training_config, args.warm_protocol,
                                 args.device, check_only=args.check_only, completion_receipt=args.completion_receipt)
    with np.load(args.instance, allow_pickle=False) as arrays:
        velocity = np.array(arrays["velocity_mps"], copy=True)
        source = np.array(arrays["source_parameters"], copy=True)
        observed = np.array(arrays["observed_pa"], copy=True)
        indices = np.array(arrays["observation_indices"], copy=True)
    # Explicitly validate target/observation contract before a long parity pass.
    extra = (np.load(args.extra_pressure, allow_pickle=False) if args.extra_pressure is not None else None)
    target_setup_started = synchronized_time(model)
    objective = NativePhysicalObjective(model, velocity, source, observed, indices, seed=args.sampling_seed,
                                       weights={**WEIGHTS, "lwc": args.pde_weight}, extra_pressure=extra)
    target_setup_seconds = synchronized_time(model) - target_setup_started
    _, insertion = insert_and_check(model, velocity, source)
    prepared = synchronized_time(model)
    setup_seconds, coarse_seconds, propagation_seconds, lora_seconds = None, None, None, None
    coarse = None
    propagation = None
    cache_receipt = None
    check = None
    result = {"check_only": True}
    if not args.uncached:
        from ul_cached_transfer_readout import CachedTransferReadout

        objective.cached_readout = CachedTransferReadout(model, objective.context, objective.request_union)
        cache_receipt = objective.cached_readout.receipt()
    initialized = synchronized_time(model)
    setup_seconds = initialized - prepared
    if args.check_only:
        check = {}
        with torch.no_grad():
            if objective.cached_readout is not None:
                direct = model.decode(objective.context, objective.request_union, normalized=True)[0]
                cached = objective.decode(objective.request_union)
                torch.testing.assert_close(cached, direct, atol=5e-6, rtol=2e-5)
                check["cached_pressure_max_absolute_difference"] = float((cached - direct).abs().max())
            if args.variant in ("coarse", "coarse-lora"):
                from ul_coarse_physics_transfer import pretrained_coarse_modes

                time, space = pretrained_coarse_modes(model, objective.context, args.coarse_rank)
                check["coarse_mode_shapes"] = {"time": list(time.shape), "space": list(space.shape)}
                check["coarse_modes_zero_top"] = not bool(space[:, 0].count_nonzero())
    else:
        if args.variant in ("coarse", "coarse-lora"):
            coarse = fit_coarse_pressure(objective, args.coarse_rank)
            coarse_seconds = synchronized_time(model) - initialized
        elif args.variant == "propagation-lora":
            from ul_propagation_transfer import fit_propagation_pressure

            propagation = fit_propagation_pressure(objective, args.solver_source)
            propagation_seconds = synchronized_time(model) - initialized
        if args.variant == "coarse":
            result = {"schema": SCHEMA, "parent_checkpoint_sha256": args.sha256.lower(),
                      "adapter_state": adapter_state(model), "fixed_extra_pressure": objective.extra_pressure.cpu(),
                      "selected_update": 0, "attempted_updates": 0, "objective": objective.protocol()}
        else:
            fitting = synchronized_time(model)
            result = fit(model, objective, args.sha256.lower(), updates=args.updates, learning_rate=args.lr,
                         selection=args.selection)
            lora_seconds = synchronized_time(model) - fitting
        result["variant"] = args.variant
        result["coarse"] = coarse
        result["propagation"] = propagation
    args.output.mkdir(parents=True, exist_ok=False)
    adapter_hash = None
    if not args.check_only:
        adapter_path = args.output / "adapter.pt"
        torch.save(result, adapter_path)
        adapter_hash = sha256(adapter_path)
    receipt = {"schema": SCHEMA, **load, "instance": str(args.instance), "instance_sha256": sha256(args.instance),
               "extra_pressure_sha256": sha256(args.extra_pressure) if args.extra_pressure is not None else None,
               "check_only": args.check_only, "insertion": insertion, "objective": objective.protocol(),
               "variant": args.variant, "coarse": coarse, "propagation": propagation, "adapter_sha256": adapter_hash,
               "pressure_readout_cache": cache_receipt, "compatibility_checks": check,
               "selected_update": result.get("selected_update"), "attempted_updates": result.get("attempted_updates"),
               "selection": result.get("selection"), "wall_seconds": perf_counter() - started,
               "adaptation_seconds": (None if args.check_only else target_setup_seconds + insertion["initialization_seconds"]
                                      + setup_seconds + (coarse_seconds or 0.) + (propagation_seconds or 0.) + (lora_seconds or 0.)),
               "elapsed_seconds_by_stage": {"load_and_insertion_parity": prepared - started - target_setup_seconds
                                                                          - insertion["initialization_seconds"],
                                            "target_setup": target_setup_seconds, "adapter_initialization": insertion["initialization_seconds"],
                                            "transfer_cache_setup": setup_seconds, "coarse_solve": coarse_seconds,
                                            "propagation_calibration": propagation_seconds,
                                            "lora_fit_and_probe_selection": lora_seconds}}
    (args.output / "RECEIPT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
