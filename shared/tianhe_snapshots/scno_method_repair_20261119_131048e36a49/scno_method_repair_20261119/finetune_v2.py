"""Output-only physical fine-tuning with a fixed residual-query bank."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
import json
import math
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.autograd import forward_ad
from torch.nn import functional as F

from lora_v2 import (CompensatedRank4, adapter_prefix, place_saved_adapters,
                     prepared_second_affine, second_affine_tensors)


POLICY = "scno_training_v2_cached_affine"
GRID = (201, 201)
TIMERS = ("setup_time", "cache_time", "optimization_time", "probe_time",
          "full_monitor_time", "transfer_time")


class FrozenQuery(NamedTuple):
    w: torch.Tensor
    scale: torch.Tensor
    hidden: torch.Tensor
    a1: torch.Tensor
    a2: torch.Tensor
    a3: torch.Tensor


class Timing:
    def __init__(self, device):
        self.device = torch.device(device)
        self.seconds = dict.fromkeys(TIMERS, 0.)
        self.stack = []
        self.started = perf_counter()

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        self.sync()
        start = perf_counter()
        self.stack.append(0.)
        try:
            yield
        finally:
            self.sync()
            elapsed = perf_counter() - start
            self.seconds[name] += elapsed - self.stack.pop()
            if self.stack:
                self.stack[-1] += elapsed

    def elapsed(self):
        self.sync()
        return perf_counter() - self.started


def exact_surface(field):
    return F.pad(field[..., 1:, :], (0, 0, 1, 0))


def synthesize_details(transform, fields, *, like):
    coefficients = [value.masked_fill(~level.valid_detail[None], 0)[:, None]
                    for value, level in zip(fields, transform.levels)]
    final = transform.levels[-1]
    origin = like.new_zeros((like.shape[0], 1, final.gz * 4, final.gx * 4))
    return transform.synthesize(origin, coefficients)


def affine_pressure(registry, transform, pack, tangent=None):
    """Exact analytic time tangent of the two affine readout branches."""
    conv = registry.pressure_readout
    if tangent is None:
        base = pack.w + pack.scale * conv(pack.hidden)
        details = (head.out(features) for head, features in zip(registry.detail_heads, pack[3:]))
    else:
        if conv.padding_mode != "zeros":
            raise ValueError("the inherited pressure convolution must use zero padding")
        conv_dt = F.conv2d(tangent.hidden, conv.weight, None, conv.stride,
                           conv.padding, conv.dilation, conv.groups)
        base = tangent.w + tangent.scale * conv(pack.hidden) + pack.scale * conv_dt
        details = (F.linear(features, head.out.weight, None)
                   for head, features in zip(registry.detail_heads, tangent[3:]))
    detail = F.pad(synthesize_details(transform, details, like=pack.w), (0, 0, 1, 0))
    return (exact_surface(base) + detail)[:, 0]


def affine_increment(registry, transform, features, reference):
    """Change in pressure for readout increments about the cached pretrained state."""
    conv = registry.pressure_readout
    weight, bias = reference[0]
    coarse = F.conv2d(features.hidden, conv.weight - weight, conv.bias - bias,
                      conv.stride, conv.padding, conv.dilation, conv.groups) * features.scale
    detail = (F.linear(value, head.out.weight - w, head.out.bias - b)
              for value, head, (w, b) in zip(features[3:], registry.detail_heads, reference[1:]))
    detail = F.pad(synthesize_details(transform, detail, like=features.w), (0, 0, 1, 0))
    return (exact_surface(coarse) + detail)[:, 0]


def bicubic_with_jvp(original, value):
    primal, tangent = forward_ad.unpack_dual(value)
    result = original(primal)
    return result if tangent is None else forward_ad.make_dual(result, original(tangent))


class InstanceReadout:
    def __init__(self, model, checkpoint_identity):
        from fast_scno import FastSCNO, _jvp

        self.model = model.requires_grad_(False).eval()
        self.model.checkpoint_queries = False
        self.encode_pre_affine = FastSCNO._encode
        self.prepare = FastSCNO._prepared_frozen
        self.registry_factory = FastSCNO.finetune_registry
        self.jvp = _jvp
        self.fast_registry = self.registry_factory(self)
        for head in self.fast_registry.detail_heads:
            if not isinstance(head.fc2, (nn.Linear, CompensatedRank4)):
                # Normalize a resident V1 adapter before binding a cache or optimizer.
                layer, state = prepared_second_affine(head.fc2, second_affine_tensors(head.fc2))
                layer.load_state_dict(state, strict=True)
                head.module[2] = layer
        self.fast_checkpoint_identity = checkpoint_identity
        self.lora_training = False
        self._second_revision = 0
        # The legacy decoder uses this lift for both pressure and its time tangent.
        self._bicubic201 = partial(bicubic_with_jvp, model._bicubic201)
        self.horizon = float(getattr(model, "duration_s", model.full_times_s[-1]))
        self.fast_registry.pressure_readout.requires_grad_(True)
        for head in self.fast_registry.detail_heads:
            head.out.requires_grad_(True)

    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def transform(self):
        return self.refiner.spectral_detail.transform

    def _times(self, times):
        value = torch.as_tensor(times, dtype=torch.float32, device=self.device).flatten()
        if not value.numel() or not bool((torch.isfinite(value) & (value >= 0.) & (value <= self.horizon)).all()):
            raise ValueError(f"physical query times must lie in [0, {self.horizon:g}] seconds")
        return value

    def enable_lora(self, scaling=1.):
        if self.lora_training:
            raise ValueError("rank-four factors are already trainable")
        layers = []
        inserted = 0
        for head in self.fast_registry.detail_heads:
            layer = head.fc2
            if isinstance(layer, nn.Linear):
                layer = CompensatedRank4.insert(layer, scaling)
                inserted += 1
            layers.append((head, layer))
        for head, layer in layers:
            head.module[2] = layer
            for factor in layer.lora_parameters():
                factor.requires_grad_(True)
        self.fast_registry = self.registry_factory(self)
        self.lora_training = True
        self._second_revision += 1
        return {"initialized_from_svd": inserted, "continued_existing": len(layers) - inserted,
                "scaling_by_head": [float(getattr(layer, "scaling", 1.)) for _, layer in layers]}

    def freeze_lora(self):
        for head in self.fast_registry.detail_heads:
            if not isinstance(head.fc2, nn.Linear):
                for factor in head.fc2.lora_parameters():
                    factor.requires_grad_(False)
        self.lora_training = False

    @property
    def query_cache_key(self):
        return "first_silu" if self.lora_training else ("second_silu", self._second_revision)

    def _encode(self, instance, times):
        first = self.encode_pre_affine(self, instance, times)
        if self.lora_training:
            return FrozenQuery(*first)
        last = [head.module[3](head.fc2(activation))
                for head, activation in zip(self.fast_registry.detail_heads, first[3:])]
        return FrozenQuery(*first[:3], *last)

    @torch.no_grad()
    def encode(self, instance, times):
        with self.scopes():
            chunks = [self._encode(instance, piece) for piece in times.split(self.refiner_time_block)]
        return FrozenQuery(*(torch.cat(parts, 0) for parts in zip(*chunks)))

    @torch.no_grad()
    def initial(self, instance):
        times = self._times([0.])
        with self.scopes():
            return self.jvp(lambda t: self._encode(instance, t), (times,), (torch.ones_like(times),))

    def decode(self, pack):
        if self.lora_training:
            features = [head.module[3](head.fc2(value))
                        for head, value in zip(self.fast_registry.detail_heads, pack[3:])]
            pack = FrozenQuery(*pack[:3], *features)
        return affine_pressure(self.fast_registry, self.transform, pack)

    def decode_initial(self, pack, tangent):
        if self.lora_training:
            return self.jvp(lambda *p: self.decode(FrozenQuery(*p)), pack, tangent)
        return (self.decode(pack), affine_pressure(self.fast_registry, self.transform, pack, tangent))

    def affine_readouts(self):
        registry = self.fast_registry
        expected = list(registry.pressure_readout.parameters())
        expected.extend(parameter for head in registry.detail_heads for parameter in head.out.parameters())
        ids = {id(parameter) for parameter in expected}
        return {name: p for name, p in self.model.named_parameters() if id(p) in ids}

    def trainable(self):
        allowed = {id(parameter) for parameter in self.affine_readouts().values()}
        if self.lora_training:
            allowed.update(id(factor) for head in self.fast_registry.detail_heads
                           for factor in head.fc2.lora_parameters())
        active = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        if {id(p) for p in active.values()} != allowed:
            raise AssertionError("trainable tensors differ from the registered readouts and adapters")
        return active

    def snapshot(self):
        values = dict(self.affine_readouts())
        for index, head in enumerate(self.fast_registry.detail_heads):
            values.update({adapter_prefix(index) + key: value
                           for key, value in second_affine_tensors(head.fc2).items()})
        return {name: value.detach().cpu().clone() for name, value in values.items()}

    @torch.no_grad()
    def restore(self, values):
        params = self.affine_readouts()
        remaining = values.keys() - params.keys()
        for name, parameter in params.items():
            value = values.get(name)
            if (not torch.is_tensor(value) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                raise ValueError(f"readout tensor shape or type changed: {name}")
        edits = []
        for index, head in enumerate(self.fast_registry.detail_heads):
            prefix = adapter_prefix(index)
            names = {name for name in remaining if name.startswith(prefix)}
            tensors = {name[len(prefix):]: values[name] for name in names}
            module, state = prepared_second_affine(head.fc2, tensors)
            if self.lora_training and module is not head.fc2:
                raise ValueError("cannot replace a layer while its factors belong to an optimizer")
            if state is not None:
                old = second_affine_tensors(head.fc2)
                if module is not head.fc2 or not all(torch.equal(old[key].cpu(), v.cpu()) for key, v in state.items()):
                    edits.append((head, module, state))
            remaining -= names
        if remaining:
            raise ValueError(f"unknown readout tensors: {sorted(remaining)}")
        for name, parameter in params.items():
            parameter.copy_(values[name])
        for head, module, state in edits:
            module.load_state_dict(state, strict=True)
            head.module[2] = module
        if edits:
            self.fast_registry = self.registry_factory(self)
            self._second_revision += 1


def pressure_axis(times, horizon, delta):
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or not len(times) or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > horizon
            or not 0 < 2 * delta <= horizon):
        raise ValueError("expected the chronological physical query axis for the model horizon")
    centers = np.minimum(times[times >= delta], horizon - delta)
    # The final center is clipped as in fast v1; keep each physical center once.
    return np.unique(centers.astype(np.float32))


def stratified_centers(centers, count_train, count_probe, seed):
    centers = np.asarray(centers)
    if (type(count_train) is not int or type(count_probe) is not int or count_train < 1 or count_probe < 1
            or len(centers) < count_train + count_probe):
        raise ValueError("insufficient admissible centers for disjoint training and probes")
    strata = math.gcd(count_train, count_probe)
    q_train, q_probe = count_train // strata, count_probe // strata
    rng = np.random.default_rng(seed)
    training, probe, w_training, w_probe = [], [], [], []
    for stratum in np.array_split(centers, strata):
        draw = rng.permutation(stratum)[:q_train + q_probe]
        training.extend(np.sort(draw[:q_train]))
        probe.extend(np.sort(draw[q_train:]))
        w_training.extend([len(stratum) / (len(centers) * q_train)] * q_train)
        w_probe.extend([len(stratum) / (len(centers) * q_probe)] * q_probe)
    return ((np.array(training), np.array(w_training)), (np.array(probe), np.array(w_probe)))


def stencil_times(centers, delta, horizon):
    minus, plus = centers - delta, centers + delta
    if bool((minus < -1e-7).any() or (plus > horizon + 1e-7).any()):
        raise ValueError("a residual stencil lies outside the model horizon")
    return torch.cat((minus.clamp_min(0.), centers, plus.clamp_max(horizon)))


class InstanceQueryBank:
    """Cache unique FP32 physical queries before the deterministic solve begins."""

    def __init__(self, readout, instance, queries, settings, timing):
        self.readout = readout
        self.mode = readout.lora_training
        self.cache_key = readout.query_cache_key
        self.timing = timing
        self.device = readout.device
        location = settings["storage"]
        if location not in ("cpu", "device") or int(settings["prepare_batch"]) < 1:
            raise ValueError("cache requires cpu/device storage and a positive preparation batch")
        self.storage = self.device if location == "device" else torch.device("cpu")
        chunks, pretrained = [], []
        self.bytes = 0
        self.index = {}
        registry = readout.fast_registry
        modules = [registry.pressure_readout, *(head.out for head in registry.detail_heads)]
        self.readouts_initial = () if self.mode else tuple(
            (module.weight.detach().clone(), module.bias.detach().clone()) for module in modules)
        with timing.measure("cache_time"):
            with timing.measure("transfer_time"):
                times = readout._times(queries).unique(sorted=True)
                keys = times.cpu().view(torch.int32).tolist()
            self.index = {key: index for index, key in enumerate(keys)}
            for subset in times.split(int(settings["prepare_batch"])):
                features = readout.encode(instance, subset)
                self.bytes += sum(t.numel() * t.element_size() for t in features)
                if not self.mode:
                    with torch.no_grad():
                        predicted = readout.decode(features)
                    self.bytes += predicted.numel() * predicted.element_size()
                    with timing.measure("transfer_time"):
                        pretrained.append(predicted.to(self.storage))
                if self.bytes > int(settings["max_bytes"]):
                    raise MemoryError("the fixed query bank exceeds cache.max_bytes")
                chunks.append(self.move(features, self.storage))
            self.features = FrozenQuery(*(torch.cat(parts, 0) for parts in zip(*chunks)))
            self.pretrained = None if self.mode else torch.cat(pretrained, 0)
            initial, tangent = readout.initial(instance)
            self.bytes += sum(t.numel() * t.element_size() for t in (*initial, *tangent))
            if self.bytes > int(settings["max_bytes"]):
                raise MemoryError("initial time tangents and fixed queries exceed cache.max_bytes")
            self.initial = (initial, tangent)

    def move(self, pack, device):
        if all(t.device == device for t in pack):
            return pack
        with self.timing.measure("transfer_time"):
            return FrozenQuery(*(value.to(device) for value in pack))

    def arrange(self, times):
        with self.timing.measure("transfer_time"):
            bits = times.detach().cpu().contiguous().view(torch.int32).tolist()
        return torch.tensor([self.index[key] for key in bits], dtype=torch.long, device=self.storage)

    def decode(self, indices):
        if self.readout.query_cache_key != self.cache_key:
            raise RuntimeError("the second hidden affine or optimization phase changed since caching")
        features = FrozenQuery(*(value.index_select(0, indices) for value in self.features))
        features = self.move(features, self.device)
        if self.pretrained is None:
            return self.readout.decode(features)
        with self.timing.measure("transfer_time"):
            pretrained = self.pretrained.index_select(0, indices).to(self.device)
        delta = affine_increment(self.readout.fast_registry, self.readout.transform, features, self.readouts_initial)
        return pretrained + delta


class FixedPhysicalObjective:
    def __init__(self, readout, instance, observations, obs_times, train, probe,
                 cfg, cache, operators, timing):
        from fast_objective import loss_initial, loss_observation

        self.readout, self.instance = readout, instance
        self.observations, self.obs_times = observations, obs_times
        self.operators, self.cfg, self.timing = operators, cfg, timing
        self.loss_observation, self.loss_initial = loss_observation, loss_initial
        self.frequency = float(instance[1][0, 2])
        self.velocity = instance[0][0, 0]
        self.mask = operators.make_valid_mask(tuple(self.velocity.shape), source=instance[1][0, :2].tolist(),
                                              halo=8, radius=90., device=readout.device)
        self.norm = operators.residual_scale_squared(observations, self.frequency, cfg["pde_energy_floor"])
        self.delta = float(cfg["delta_s"])
        self.batch = int(cfg["residual_batch"])
        if self.batch < 1:
            raise ValueError("objective.residual_batch must be positive")
        self.queries = {}
        encoded_times = [obs_times]
        for name, (centers, weights) in (("train", train), ("probe", probe)):
            centers = readout._times(centers)
            weights = torch.as_tensor(weights, dtype=torch.float64, device=readout.device)
            self.queries[name] = []
            for times, w in zip(centers.split(self.batch), weights.split(self.batch)):
                stencil = stencil_times(times, self.delta, readout.horizon)
                encoded_times.append(stencil)
                self.queries[name].append((stencil, w, len(times)))
        self.bank = InstanceQueryBank(readout, instance, torch.cat(encoded_times), cache, timing)
        self.obs_indices = self.bank.arrange(obs_times)
        self.queries = {name: [(self.bank.arrange(times), w, count) for times, w, count in groups]
                        for name, groups in self.queries.items()}

    def loss(self, which, *, backward=False):
        cfg = self.cfg
        weights = cfg["weights"]
        with torch.set_grad_enabled(backward), torch.backends.cudnn.flags(deterministic=True, benchmark=False):
            predicted = self.bank.decode(self.obs_indices)
            u0, ut0 = self.readout.decode_initial(*self.bank.initial)
            obs = self.loss_observation(predicted, self.observations, cfg["observation_energy_floor"])
            ic = self.loss_initial(u0, ut0, self.frequency)
            boundary = predicted[:, 0].detach().double().square().mean()
            if boundary.item() != 0.:
                raise AssertionError("the exact free surface has a nonzero pressure observation")
            first = weights["obs"] * obs + weights["ic"] * ic + weights["bc"] * boundary
            if backward:
                first.backward()
            detached_residual = first.new_zeros(())
            for indices, time_weights, count in self.queries[which]:
                fields = self.bank.decode(indices)
                r = self.operators.lwc_residual(*fields.split(count), self.velocity, h=10., delta=self.delta)
                component = (r[:, self.mask].square().mean(1) * time_weights).sum() / self.norm
                if backward:
                    (weights["lwc"] * component).backward()
                detached_residual += component.detach()
            result = first.detach() + weights["lwc"] * detached_residual
        parts = {"obs": float(obs.detach()), "ic": float(ic.detach()),
                 "lwc": float(detached_residual), "bc": float(boundary), "total": float(result)}
        if not all(math.isfinite(value) for value in parts.values()):
            raise FloatingPointError("nonfinite physical objective")
        return result, parts


def same_state(candidate, snapshot):
    return all(torch.equal(value, snapshot[name]) for name, value in candidate.items())


def gradient_probe(optimizer, objective, parameters):
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective.loss("train", backward=True)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("nonfinite or disconnected readout gradient")
        return loss

    return closure


def affine_checks(readout, objective, closure, parameters):
    """Measure cache parity, linearity and closure repeatability on fixed features."""
    snapshot = readout.snapshot()
    instance = objective.instance
    original = readout.model._bicubic201
    try:
        readout.model._bicubic201 = readout._bicubic201
        with torch.no_grad(), readout.scopes():
            native = readout.model(*instance[:2], objective.obs_times)
            fixed = objective.bank.decode(objective.obs_indices)
            times = readout._times([0.])
            direct0, direct_dt = readout.jvp(lambda t: readout.model(*instance[:2], t), (times,), (torch.ones_like(times),))
            cached0, cached_dt = readout.decode_initial(*objective.bank.initial)
        torch.testing.assert_close(fixed, native, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(cached0, direct0, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(cached_dt, direct_dt, atol=1e-5, rtol=8e-5)
        rng = torch.Generator(device="cpu").manual_seed(711)
        affine = readout.affine_readouts()
        a = {name: torch.randn(value.shape, generator=rng, dtype=value.dtype) * .001
             for name, value in affine.items()}
        b = {name: torch.randn(value.shape, generator=rng, dtype=value.dtype) * .001
             for name, value in affine.items()}
        shifted = []
        with torch.no_grad():
            for first, second in ((1, 0), (0, 1), (1, 1)):
                changed = {name: snapshot[name] + first * a[name] + second * b[name] for name in affine}
                readout.restore({**snapshot, **changed})
                shifted.append(objective.bank.decode(objective.obs_indices))
            error = shifted[2] - (shifted[0] + shifted[1] - fixed)
            torch.testing.assert_close(error, torch.zeros_like(error), atol=3e-6, rtol=0.)
    finally:
        readout.model._bicubic201 = original
        readout.restore(snapshot)
    first = closure().clone()
    g = [p.grad.detach().clone() for p in parameters]
    second = closure()
    torch.testing.assert_close(first, second, atol=1e-12, rtol=0.)
    grad_error = max(float((p.grad - previous).abs().max()) for p, previous in zip(parameters, g))
    for p, previous in zip(parameters, g):
        torch.testing.assert_close(p.grad, previous, atol=1e-8, rtol=1e-7)
    optimizer_gradient = {"objective_absolute_error": abs(float(first - second)), "gradient_max_absolute_error": grad_error}
    return {"cached_pressure_max_absolute_error": float((fixed - native).abs().max()),
            "cached_initial_pressure_max_absolute_error": float((cached0 - direct0).abs().max()),
            "cached_initial_tangent_max_absolute_error": float((cached_dt - direct_dt).abs().max()),
            "affine_superposition_max_absolute_error": float(error.abs().max()),
            "closure_repeatability": optimizer_gradient}


def phased_fit(readout, objective, optimizer, settings, timing, budget, *, use_lora=False, numerical_checks=False):
    parameters = list(readout.trainable().values())
    closure = gradient_probe(optimizer, objective, parameters)
    with timing.measure("probe_time"):
        _, start_train = objective.loss("train")
        _, start_probe = objective.loss("probe")
        checks = affine_checks(readout, objective, closure, parameters) if numerical_checks else {}
    initial = readout.snapshot()
    best = {"state": initial, "update": 0, "train": start_train, "probe": start_probe}
    history, counts = [], {"attempted_updates": 0, "num_updates": 0, "closure_evaluations": 0}
    status = "update_limit"
    last_train = start_train

    def counted_closure():
        counts["closure_evaluations"] += 1
        return closure()

    for _ in range(int(settings["max_updates"] if use_lora else settings["outer_step_cap"])):
        if budget is not None and timing.elapsed() >= budget:
            status = "wall_budget_exhausted"
            break
        if not use_lora and counts["closure_evaluations"] >= int(settings["closure_evaluation_budget"]):
            status = "closure_budget_exhausted"
            break
        previous = readout.snapshot()
        counts["attempted_updates"] += 1
        try:
            with timing.measure("optimization_time"):
                if use_lora:
                    counted_closure()
                    torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip_norm"], error_if_nonfinite=True)
                    optimizer.step()
                else:
                    optimizer.step(counted_closure)
                _, train = objective.loss("train")
                state = readout.snapshot()
                if not all(torch.isfinite(p).all() for p in state.values()):
                    raise FloatingPointError("nonfinite accepted parameters")
            if not use_lora and train["total"] > last_train["total"] + 2e-7 * max(1., abs(last_train["total"])):
                status = "line_search_did_not_decrease_objective"
                readout.restore(previous)
                break
            if same_state(state, previous):
                status = "stationary"
                break
            with timing.measure("probe_time"):
                _, probe = objective.loss("probe")
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            readout.restore(previous)
            status = f"optimization_failure: {type(exc).__name__}: {exc}"
            break
        counts["num_updates"] += 1
        update = counts["num_updates"]
        history.append({"update": update, "train": train, "probe": probe})
        last_train = train
        if probe["total"] < best["probe"]["total"] and train["total"] <= start_train["total"]:
            best = {"state": state, "update": update, "train": train, "probe": probe}
    readout.restore(best["state"])
    return best, {**counts, "stop_reason": status, "initial_training": start_train,
                  "initial_probe": start_probe, "selected_update": best["update"],
                  "selected_training": best["train"], "selected_probe": best["probe"],
                  "history": history, "numerical_checks": checks}


@torch.no_grad()
def certify_full_residual(readout, instance, objective, centers, timing, budget):
    operators, cfg = objective.operators, objective.cfg
    failures, total, processed = 0, 0, 0
    with timing.measure("full_monitor_time"):
        for times in centers.split(objective.batch):
            if budget is not None and timing.elapsed() >= budget:
                break
            values = readout.decode(readout.encode(instance, stencil_times(times, objective.delta, readout.horizon)))
            residual = operators.lwc_residual(*values.split(len(times)), objective.velocity, delta=objective.delta)
            stats = operators.full_grid_diagnostics(residual, objective.mask, f0=objective.frequency,
                                                   epsilon_r=cfg["epsilon_r"], epsilon_p=cfg["epsilon_p"])
            failures += stats["failure_count"]
            total += stats["units"]
            processed += len(times)
    completed = processed == len(centers)
    probability = failures / total if completed else None
    return {"full_grid_monitor_completed": completed,
            "full_grid_threshold_satisfied": probability < cfg["epsilon_p"] if completed else None,
            "full_grid_failure_probability": probability, "full_grid_processed_centers": processed,
            "full_grid_total_centers": len(centers), "full_grid_processed_units": total}


def fit_instance(model, checkpoint_identity, c, source, observations, observation_times, config,
                 *, observations_normalized=False):
    """Fit one wavefield with a deterministically evaluated physical objective."""
    from fast_objective import load_residual_operators

    timing = Timing(model.device)
    settings = config["affine"]
    cfg = config["objective"]
    cache = config["cache"]
    budget = config.get("wall_time_budget_s")
    if budget is not None and (not math.isfinite(budget) or budget < 0):
        raise ValueError("the wall budget must be null or nonnegative seconds")
    if (set(cfg["weights"]) != {"obs", "ic", "lwc", "bc"}
            or any(not math.isfinite(w) or w < 0 for w in cfg["weights"].values())):
        raise ValueError("the four registered physical weights must be finite and nonnegative")
    if (type(settings["outer_step_cap"]) is not int or settings["outer_step_cap"] < 0
            or type(settings["closure_evaluation_budget"]) is not int or settings["closure_evaluation_budget"] < 1):
        raise ValueError("invalid affine optimizer budget")
    certification = {"full_grid_monitor_completed": False, "full_grid_threshold_satisfied": None,
                     "full_grid_failure_probability": None}
    with timing.measure("setup_time"):
        readout = InstanceReadout(model, checkpoint_identity)
        operators = load_residual_operators(config.get("residual_operator_path"))
        with timing.measure("transfer_time"):
            cv, fv = readout._inputs(c, source)
            obs = torch.as_tensor(observations, dtype=cv.dtype, device=cv.device)
            obs_times = readout._times(observation_times)
        prescribed = cfg.get("observation_times_s")
        if (len(obs_times) != 2 or not bool(obs_times[0] < obs_times[1])
                or (prescribed is not None and not torch.equal(obs_times, readout._times(prescribed)))):
            raise ValueError("provide two ordered observation times matching the configured support")
        if obs.shape != (2, *GRID) or not bool(torch.isfinite(obs).all()) or float(fv[0, 4]) <= 0.:
            raise ValueError("the instance requires two finite pressure fields and positive source amplitude")
        if not observations_normalized:
            obs = readout.normalizer.encode_pressure(obs[None], fv[:, 4])[0]
        obs = obs.detach()
        if float(obs.square().sum()) <= 0.:
            raise ValueError("observation energy is zero on the supplied two snapshots")
        axis = readout.full_times_s.detach().cpu().numpy()
        centers = pressure_axis(axis, readout.horizon, float(cfg["delta_s"]))
        train, probe = stratified_centers(centers, config["training_residual_centers"],
                                         config["validation_residual_centers"], config["sampling_seed"])
        initial = readout.snapshot()
    affine_parameters = readout.trainable()
    counts = {"affine_readout_scalars": sum(value.numel() for value in affine_parameters.values()),
              "affine_readout_tensor_count": len(affine_parameters)}
    candidates = {"pretrained": initial}
    selection = "pretrained"
    affine_report = {"num_updates": 0, "stop_reason": "wall_budget_exhausted"}
    lora_report = {"num_updates": 0, "stop_reason": "disabled"}
    objective = instance = None
    if budget is None or timing.elapsed() < budget:
        with timing.measure("setup_time"):
            instance = readout.prepare(readout, cv, fv)
        objective = FixedPhysicalObjective(readout, instance, obs, obs_times, train, probe, cfg, cache, operators, timing)
        if budget is None or timing.elapsed() < budget:
            optimizer = torch.optim.LBFGS(list(readout.trainable().values()), lr=settings["lr"],
                                          history_size=int(settings["history_size"]), max_iter=1,
                                          line_search_fn="strong_wolfe")
            affine, affine_report = phased_fit(readout, objective, optimizer, settings, timing, budget,
                                               numerical_checks=bool(config.get("check_numerics", False)))
            candidates["affine"] = affine["state"]
            if affine["update"]:
                selection = "affine"
            lora = config["lora"]
            if lora["enabled"]:
                threshold = float(lora["minimum_remaining_probe_lwc"])
                eligible = (affine["update"] > 0 and affine["probe"]["lwc"] > threshold
                            and (budget is None or timing.elapsed() < budget))
                lora_report = {"num_updates": 0, "stop_reason": "affine_probe_or_budget_gate"}
                if eligible:
                    # Rebuild the bank using the first SiLU features before training the factors.
                    objective = None
                    with timing.measure("setup_time"):
                        initialization = readout.enable_lora(lora.get("scaling", 1.))
                        instance = readout.prepare(readout, cv, fv)
                    objective = FixedPhysicalObjective(readout, instance, obs, obs_times, train, probe, cfg, cache, operators, timing)
                    parameters = readout.trainable()
                    adapter = [p for name, p in parameters.items() if name.endswith((".U", ".V"))]
                    others = [p for name, p in parameters.items() if not name.endswith((".U", ".V"))]
                    optimizer = torch.optim.AdamW([{"params": others, "lr": lora["readout_lr"]},
                                                   {"params": adapter, "lr": lora["lora_lr"]}],
                                                  weight_decay=lora["weight_decay"])
                    adjusted, lora_report = phased_fit(readout, objective, optimizer, lora, timing, budget, use_lora=True)
                    lora_report["initialization"] = initialization
                    counts["lora_phase_scalars"] = sum(p.numel() for p in parameters.values())
                    counts["lora_factor_scalars"] = sum(p.numel() for p in adapter)
                    candidates["lora"] = adjusted["state"]
                    if adjusted["update"]:
                        selection = "lora"
    with timing.measure("transfer_time"):
        readout.freeze_lora()
        readout.restore(candidates[selection])
    if config.get("full_grid_monitor", False) and objective is not None:
        certification = certify_full_residual(readout, instance, objective, readout._times(centers), timing, budget)
    wall = timing.elapsed()
    report = {"policy_id": POLICY, "selected_candidate": selection, "pretrained_checkpoint_sha256": checkpoint_identity,
              "num_updates": affine_report["num_updates"] + lora_report["num_updates"],
              "affine": affine_report, "lora": lora_report, **counts,
              "observation_times_s": obs_times.detach().cpu().tolist(),
              "observation_input_units": "pressure/(A*sigma_u)" if observations_normalized else "Pa",
              "pressure_unit_pa": float(readout.normalizer.decode_pressure(torch.ones((1, 1, 1, 1), device=cv.device), fv[:, 4]).item()),
              "residual_axis_count": len(centers), "residual_axis_interval_s": [float(centers[0]), float(centers[-1])],
              "residual_training_centers_s": train[0].tolist(), "residual_probe_centers_s": probe[0].tolist(),
              "residual_training_center_weights": train[1].tolist(), "residual_probe_center_weights": probe[1].tolist(),
              "sampling_seed": config["sampling_seed"], "objective": cfg,
              "cached_query_count": 0 if objective is None else len(objective.bank.index),
              "cached_query_bytes": 0 if objective is None else objective.bank.bytes,
              "wall_time_budget_s": budget, "wall_budget_overrun_s": 0. if budget is None else max(0., wall - budget),
              "closure_budget_overrun": max(0, affine_report.get("closure_evaluations", 0) - settings["closure_evaluation_budget"]),
              "timing": {**timing.seconds, "total_wall_time": wall, "scope": "resident model to selected candidate and requested probes"},
              **certification}
    state = {"policy_id": POLICY, "pretrained_checkpoint_sha256": checkpoint_identity,
             "selected_candidate": selection, "lora_seed": config["lora"].get("seed"), "candidates": candidates}
    return report, state


def restorable_readout(model, checkpoint_identity, payload, candidate=None):
    if payload["policy_id"] != POLICY or payload["pretrained_checkpoint_sha256"] != checkpoint_identity:
        raise ValueError("the readout requires the original authenticated SCNO checkpoint")
    saved = payload["candidates"][candidate or payload["selected_candidate"]]
    readout = InstanceReadout(model, checkpoint_identity)
    readout.restore(saved)
    return readout


def load_pretrained(config, device):
    from fast_scno import sha256

    checkpoint = config["checkpoint"]
    if not checkpoint.get("path") or not checkpoint.get("sha256"):
        raise ValueError("configure the spectral checkpoint path and its SHA-256")
    path = Path(checkpoint["path"])
    identity = sha256(path)
    if identity != checkpoint["sha256"]:
        raise ValueError("the pretrained checkpoint digest does not match the configuration")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    dataset = None
    if checkpoint["format"] == "spectral_v1":
        from spectral_marmousi import SpectralMarmousiOperator

        model = SpectralMarmousiOperator(config.get("model", {}), device=device)
        state = saved.get("model", saved)
        place_saved_adapters(model, state)
        model.load_state_dict(state, strict=True)
    elif checkpoint["format"] == "v2":
        from data_v2 import open_corpus
        from initialization_v2 import semantics, validate_resume_semantics
        from model_v2 import build_model

        model_config_path = checkpoint.get("training_config")
        if not model_config_path:
            raise ValueError("configure the V2 training JSON used by the checkpoint")
        training_config = json.loads(Path(model_config_path).read_text(encoding="utf-8"))
        dataset = open_corpus(training_config)
        binding = saved.get("binding")
        if not binding or binding.get("config") != training_config or binding.get("dataset") != dataset.binding:
            raise ValueError("the V2 checkpoint must carry the configured physical axis, dataset and normalization binding")
        model = build_model(training_config, dataset, device)
        validate_resume_semantics(saved, semantics(model, dataset))
        model.load_state_dict(saved["model"], strict=True)
    else:
        raise ValueError(f"unknown checkpoint format: {checkpoint['format']}")
    if hasattr(model, "spectral_detail_enabled"):
        model.spectral_detail_enabled = True
    return model.eval(), identity, dataset


def instance_from_record(dataset, record_id, slots):
    if dataset is None:
        raise ValueError("corpus records require an authenticated V2 training configuration")
    try:
        record = next(value for value in dataset.train_ids if str(value) == str(record_id))
    except StopIteration as error:
        raise ValueError(f"{record_id} is not in the configured TRAIN split") from error
    if (len(slots) != 2 or slots[0] < 0 or slots[0] >= slots[1] or slots[1] >= len(dataset.times)):
        raise ValueError("choose two distinct ordered slots within the physical data axis")
    batch = dataset.read(record, np.asarray(slots, dtype=np.int64))
    return batch["velocity"], batch["source"], batch["target"], batch["times"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--instance", type=Path, help="NPZ containing c201_mps, source, observation_times_s, observed_pa")
    data.add_argument("--from-record", help="configured V2 training record used for a short diagnostic")
    parser.add_argument("--observation-slots", type=int, nargs=2, metavar=("FIRST", "SECOND"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--budget-s", type=float, help="soft resident-run wall time in seconds")
    parser.add_argument("--full-monitor", action="store_true")
    parser.add_argument("--check-numerics", action="store_true")
    parser.add_argument("--enable-lora", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("choose a new output directory")
    if bool(args.from_record is not None) != bool(args.observation_slots is not None):
        parser.error("--from-record requires --observation-slots")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.budget_s is not None:
        config["wall_time_budget_s"] = args.budget_s
    if args.full_monitor:
        config["full_grid_monitor"] = True
    if args.check_numerics:
        config["check_numerics"] = True
    if args.enable_lora:
        config["lora"]["enabled"] = True
    started = perf_counter()
    model, identity, dataset = load_pretrained(config, args.device)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    load_s = perf_counter() - started
    started = perf_counter()
    if args.instance is not None:
        with np.load(args.instance, allow_pickle=False) as archive:
            values = tuple(archive[name] for name in ("c201_mps", "source", "observed_pa", "observation_times_s"))
    else:
        values = instance_from_record(dataset, args.from_record, args.observation_slots)
    acquisition_s = perf_counter() - started
    report, payload = fit_instance(model, identity, *values, config, observations_normalized=args.from_record is not None)
    report["checkpoint_load_time"] = load_s
    report["input_acquisition_time"] = acquisition_s
    report["wall_time_with_input_and_load"] = load_s + acquisition_s + report["timing"]["total_wall_time"]
    report["instance_source"] = str(args.instance) if args.instance is not None else {
        "training_record_id": args.from_record, "observation_slots": args.observation_slots}
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(payload, args.out / "candidates.pt")
    (args.out / "RECEIPT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("selected_candidate", "num_updates", "affine_readout_scalars",
                                             "full_grid_monitor_completed", "timing")}), flush=True)


if __name__ == "__main__":
    main()
