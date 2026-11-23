"""Matched 50-update propagation-defect experiment on a sealed E200+U96 pair."""

from argparse import ArgumentParser
from contextlib import contextmanager
import json
from math import isfinite, pi
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from data_v2 import _write_json
from evaluate_ul_current_pair import (array_sha, open_reference, pool_case_scores, prepare_center_observations,
                                      prepare_external_observations, streamed_window_scores, tensor_sha)
from finetune_ul_transfer import initial_derivative, residual_operators, sha256
from finetune_v2 import stratified_centers
from train_ul_repaired_200 import save_atomic
from ul_current_cached_transfer import CachedCurrentTransferReadout, encode_instance, original_pressure
from ul_current_lora import (ADAPTER_SCHEMA, clone_current, current_adapter_state, current_layers,
                             insert_current_adapters, restore_current_adapter)
from ul_current_model_binding import load_current_model, verify_current_binding
from ul_current_propagation import FixedDirection, generate_native_proposal, prepare_fixed_direction


SCHEMA = "SCNO_CURRENT_E200_U96_LORA_PAIR_V1"
ROWS = 401
ARMS = ("defect_off_50", "defect_on_50")
ATOL, RTOL = 5e-6, 2e-5


class Timings(dict):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)
        self.cuda_event_ms = {}

    @contextmanager
    def __call__(self, name):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
        started = perf_counter()
        yield
        if self.device.type == "cuda":
            end.record()
            torch.cuda.synchronize(self.device)
            self.cuda_event_ms[name] = begin.elapsed_time(end)
        self[name] = perf_counter() - started


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@torch.no_grad()
def check_pressure_parity(actual, expected):
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    if actual[:, 0].count_nonzero():
        raise ValueError("The output free-surface row must be exactly zero")
    diff = actual.double() - expected.double()
    return {"max_abs_normalized": float(diff.abs().max()),
            "relative_l2": float((diff.square().sum() / expected.double().square().sum().clamp_min(1e-30)).sqrt())}


LATE_DEFECT_FIRST_FRAME = 160


class NativeCurrentObjective:
    def __init__(self, shared, inputs, proposal, cfg, *, late_defect_factor=1.):
        if not isfinite(late_defect_factor) or late_defect_factor < 0.:
            raise ValueError("The late propagation defect weight must be finite and nonnegative")
        model, instance = shared.model, shared.instance
        self.shared = shared
        self.cfg = cfg
        self.velocity = instance.velocity[0, 0]
        self.proposal = proposal.detach()
        self.frequency = float(instance.source[0, 2])
        self.op = residual_operators()
        self.mask = self.op.make_valid_mask(
            source=instance.source[0, :2].tolist(), halo=cfg["residual"]["halo_nodes"],
            radius=cfg["residual"]["excluded_source_radius_m"], device=model.device)
        speed = self.velocity[None, None]
        mixed = F.max_pool2d(speed, 17, stride=1, padding=8) != -F.max_pool2d(-speed, 17, stride=1, padding=8)
        self.interface_mask = mixed[0, 0] & self.mask
        target = torch.as_tensor(inputs.observed_pa, device=model.device)
        self.observations = model.normalizer.encode_pressure(target[None], instance.source[:, 4])[0].double()
        self.nu_squared = self.op.residual_scale_squared(self.observations, self.frequency)
        self.observation_energy = self.observations.square().sum()
        boundary = torch.zeros_like(self.velocity, dtype=torch.bool)
        boundary[:, 0] = boundary[:, -1] = boundary[-1] = True
        self.boundary_nodes = boundary.flatten().nonzero().flatten()
        spec = cfg["residual"]
        train, probe = stratified_centers(np.arange(1, 400), spec["training_centers"],
                                         spec["diagnostic_centers"], cfg["seed"])
        self.support = {}
        self.defect_weights = {}
        self.observation_rows = torch.as_tensor(inputs.observation_indices, device=model.device)
        initial = torch.arange(3, device=model.device)
        for name, (c, w) in zip(("train", "probe"), (train, probe)):
            centers = torch.as_tensor(c, device=model.device)
            weights = torch.as_tensor(w, device=model.device)
            requested = torch.cat((self.observation_rows, initial, centers - 1, centers, centers + 1))
            rows, inverse = requested.unique(sorted=True, return_inverse=True)
            traces = self.proposal[centers].flatten(1).index_select(1, self.boundary_nodes).double()
            denominator = (traces.square().mean(1) * weights).sum()
            denominator = denominator.clamp_min(1e-4 * self.proposal.double().square().mean()).clamp_min(1e-24)
            self.support[name] = (centers, weights, rows, inverse, traces, denominator)
            self.defect_weights[name] = (weights if late_defect_factor == 1. else
                                         weights * torch.where(centers >= LATE_DEFECT_FIRST_FRAME,
                                                               late_defect_factor, 1.))
        self.query_union = torch.cat([group[2] for group in self.support.values()]).unique(sorted=True)

    def stencil(self, minus, center, plus):
        spec = self.cfg["residual"]
        return self.op.lwc_residual(minus, center, plus, self.velocity,
                                    h=spec["spacing_m"], delta=spec["registered_dt_s"])

    def loss_terms(self, readout, which, *, include_defect, include_lora_trust=True):
        centers, weights, rows, inverse, traces, scale = self.support[which]
        pressure = readout.decode(rows)
        ordered = pressure[inverse]
        predicted, initial, minus, center, plus = ordered.split((2, 3, len(centers), len(centers), len(centers)))
        obs = (predicted.double() - self.observations).square().sum() / self.observation_energy
        rate = initial_derivative(initial) / (2. * pi * self.frequency)
        ic = (initial[0].double().square() + rate.square()).mean()
        side = center.flatten(1).index_select(1, self.boundary_nodes).double()
        side = ((side - traces).square().mean(1) * weights).sum() / scale
        terms = {"observation": obs, "initial": ic, "side_bottom_proposal_trace": side}
        if include_lora_trust:
            terms["lora_effective_weight_trust"] = torch.stack([
                layer.delta_weight().square().sum() / layer.base.weight.square().sum().clamp_min(1e-12)
                for layer in current_layers(readout.model)
            ]).mean()
        if include_defect:
            residual = self.stencil(minus.double() - self.proposal[centers - 1],
                                    center.double() - self.proposal[centers],
                                    plus.double() - self.proposal[centers + 1])[:, self.mask]
            terms["defect"] = (residual.square().mean(1) * self.defect_weights[which]).sum() / self.nu_squared
        return terms, pressure[:, 0]

    @torch.no_grad()
    def residual_diagnostics(self, normalized):
        difference = normalized.double() - self.proposal
        indices = torch.arange(1, 400, device=difference.device)
        masks = {"interior": self.mask, "mixed_speed_halo": self.interface_mask,
                 "single_speed_halo": self.mask & ~self.interface_mask}
        energy = {name: difference.new_zeros(()) for name in (*masks, "masked_short_wavelength_power_fraction")}
        spectral_total = difference.new_zeros(())
        kz = torch.fft.fftfreq(201, device=difference.device)
        kx = torch.fft.rfftfreq(201, device=difference.device)
        short = kz[:, None].square() + kx[None].square() >= .25**2
        multiplicity = torch.ones_like(kx)
        multiplicity[1:] = 2.
        for c in indices.split(16):
            r = self.stencil(difference[c - 1], difference[c], difference[c + 1])
            for name, mask in masks.items():
                energy[name] += r[:, mask].square().sum()
            power = torch.fft.rfft2(r * self.mask).abs().square() * multiplicity
            spectral_total += power.sum()
            energy["masked_short_wavelength_power_fraction"] += power[:, short].sum()
        out = {k: (float(energy[k] / (399 * count * self.nu_squared)) if count else None)
               for k, count in ((k, int(mask.sum())) for k, mask in masks.items())}
        out["masked_short_wavelength_power_fraction"] = float(
            energy["masked_short_wavelength_power_fraction"] / spectral_total.clamp_min(1e-30))
        out["centers"] = 399
        out["short_wavelength_threshold_cycles_per_node"] = .25
        out["normalization_nu_squared"] = float(self.nu_squared)
        return out

    def protocol(self):
        payload = {
            name: {"centers": centers.tolist(), "weights": weights.tolist(), "requested_rows": rows.tolist()}
            for name, (centers, weights, rows, *_rest) in self.support.items()
        }
        payload.update(observation_indices=self.observation_rows.tolist(),
                       shared_query_union=self.query_union.tolist(),
                       boundary_node_count=len(self.boundary_nodes), residual_nodes=int(self.mask.sum()),
                       mixed_speed_halo_nodes=int(self.interface_mask.sum()))
        return payload


def trainable_groups(readout):
    model = readout.model
    groups = {"detail": [], "u96": [], "gain": [model.propagation_gain]}
    for n, layer in enumerate(current_layers(model)):
        if n < 3 or readout.instance.eligible:
            groups["detail" if n < 3 else "u96"].extend(p for p in layer.lora_parameters() if p.requires_grad)
    return groups


def diagnostics(objective, readout, groups):
    terms, top = objective.loss_terms(readout, "train", include_defect=True)
    if top.count_nonzero():
        raise ValueError("Zero extension failed during adaptation")
    parameters = [parameter for group in groups.values() for parameter in group]
    report = {}
    for key in ("observation", "side_bottom_proposal_trace", "defect"):
        gradients = torch.autograd.grad(terms[key], parameters, retain_graph=True, allow_unused=True)
        grouped = {}
        cursor = 0
        for name, values in groups.items():
            count = len(values)
            g = [value for value in gradients[cursor:cursor + count] if value is not None]
            cursor += count
            grouped[name] = float(torch.stack([v.square().sum() for v in g]).sum().sqrt()) if g else 0.
            if name == "gain":
                grouped["gain_signed"] = float(g[0]) if g else 0.
        report[key] = grouped
    with torch.no_grad():
        probe, _ = objective.loss_terms(readout, "probe", include_defect=True)
    return {"train_loss_terms": {k: float(v.detach()) for k, v in terms.items()}, "bc_top": 0.,
            "probe_loss_terms": {k: float(v) for k, v in probe.items()},
            "gradient_by_loss": report, "gain": float(readout.model.propagation_gain.detach())}


class CurrentAdaptation:
    """One adapter and AdamW state, usable in either a single or grouped update."""

    def __init__(self, readout, objective, cfg, *, use_defect):
        self.readout, self.objective = readout, objective
        self.groups = trainable_groups(readout)
        self.lora = self.groups["detail"] + self.groups["u96"]
        self.gain = self.groups["gain"]
        opt_cfg, loss_cfg = cfg["optimizer"], cfg["loss"]
        self.optimizer = torch.optim.AdamW([{"params": self.lora, "lr": opt_cfg["lora_lr"]},
                                            {"params": self.gain, "lr": opt_cfg["propagation_gain_lr"]}],
                                           weight_decay=opt_cfg["weight_decay"])
        self.clip = opt_cfg["clip_each_parameter_group"]
        self.bounds = cfg["propagation"]["gain_bounds"]
        self.weights = {key: loss_cfg[key] for key in ("observation", "initial", "side_bottom_proposal_trace",
                                                      "lora_effective_weight_trust")}
        self.weights["defect"] = loss_cfg["defect_on" if use_defect else "defect_off"]

    def accumulate(self):
        self.optimizer.zero_grad(set_to_none=True)
        terms, top = self.objective.loss_terms(self.readout, "train", include_defect=self.weights["defect"] > 0.)
        loss = sum(self.weights[k] * t for k, t in terms.items())
        if not torch.isfinite(loss) or top.count_nonzero():
            raise FloatingPointError("Nonfinite loss or free surface violation")
        loss.backward()
        return loss.detach()

    def step(self, step):
        if any(p.grad is None for p in self.lora + self.gain):
            raise RuntimeError("An active LoRA factor or the gain was detached")
        if step == 1 and any(not torch.count_nonzero(p.grad) for p in self.lora + self.gain):
            raise RuntimeError("A nondegenerate instance did not activate every live factor")
        for parameters in (self.lora, self.gain):
            torch.nn.utils.clip_grad_norm_(parameters, self.clip, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            self.readout.model.propagation_gain.clamp_(*self.bounds)

    def receipt(self, before, after, updates):
        return {"step0": before, f"step{updates}": after, "weights": self.weights,
                "attempted_updates": updates, "selected_update": updates, "selection": "final",
                "effective_weight_change_frobenius": [float(layer.delta_weight().detach().norm())
                                                       for layer in current_layers(self.readout.model)]}


def fit_current_updates(readout, objective, cfg, timing, case_key, *,
                        use_defect, label, checkpoint_steps=(), on_checkpoint=None):
    fit = CurrentAdaptation(readout, objective, cfg, use_defect=use_defect)
    updates = cfg["updates"]
    with timing("step0_diagnostics_s"):
        before = diagnostics(objective, readout, fit.groups)
    with timing(f"fit_{updates}_s"):
        for step in range(1, updates + 1):
            loss = fit.accumulate()
            fit.step(step)
            if step in (10, 25, 50) or step in checkpoint_steps:
                print(f"{case_key} {label} step {step}/{updates} loss={float(loss):.6g} "
                      f"gain={float(fit.gain[0].detach()):.6g}", flush=True)
            if on_checkpoint is not None and step in checkpoint_steps:
                on_checkpoint(step, fit.optimizer)
    with timing(f"step{updates}_diagnostics_s"):
        after = diagnostics(objective, readout, fit.groups)
    return fit.receipt(before, after, updates)


def fit_current_50(readout, objective, arm, cfg, timing, case_key):
    return fit_current_updates(readout, objective, cfg, timing, case_key,
                               use_defect=arm == "defect_on_50", label=arm)


def check_render_gradients(readout, observation_rows):
    rows = torch.cat((observation_rows.new_tensor([0, 1, 2]), observation_rows,
                      observation_rows.new_tensor([179, 300, 399])))
    direct, cached = readout.direct(rows), readout.decode(rows)
    parity = check_pressure_parity(cached, direct)
    parameters = [p for group in trainable_groups(readout).values() for p in group]
    target = torch.linspace(.1, 1., direct.numel(), device=direct.device).reshape_as(direct)
    gradients = [torch.autograd.grad((value * target).double().square().mean(), parameters)
                 for value in (direct, cached)]
    parity["gradient_max_abs_difference"] = 0.
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=1e-3)
        parity["gradient_max_abs_difference"] = max(parity["gradient_max_abs_difference"], float((a-b).abs().max()))
    return parity


def delta_report(scores, min_improvement):
    p = {key: val["full_401"]["relative_l2_percent"] for key, val in scores.items()}
    base = p["current_baseline"] - p["defect_on_50"]
    paired = p["defect_off_50"] - p["defect_on_50"]
    return {"full401_relative_l2_percent": p, "baseline_gain_pp": base, "paired_gain_pp": paired,
            "propagation_only_gain_pp": p["propagation_only"] - p["defect_on_50"],
            "gain_only_to_lora_pp": p["gain_only_at_on_50"] - p["defect_on_50"],
            "pass_primary": base >= min_improvement and paired > 0.}


def run_case(original, master, identity, case, archived, cfg, directory):
    directory.mkdir()
    np.savez(directory / "observations_only.npz", **vars(case.inputs))
    instance_identity = case.inputs.identity()
    t = Timings(original.device)
    with t("instance_encode_s"):
        instance = encode_instance(original, case.inputs.velocity_mps, case.inputs.source_parameters)
    rows = torch.arange(ROWS, device=original.device)
    with t("baseline_decode_401_s"):
        baseline, head = original_pressure(original, instance, rows)
    with t("baseline_evaluation_s"), torch.no_grad():
        pa = original.normalizer.decode_pressure(baseline[None], instance.source[:, 4])[0]
        scores = streamed_window_scores(open_reference(case), {"current_baseline": pa})
    reproduced = scores["current_baseline"]["full_401"]["relative_l2_percent"]
    if abs(reproduced - archived) > .02:
        raise ValueError(f"{case.key}: {reproduced:.8f}% does not reproduce the stored {archived:.8f}%")
    print(f"{case.key} baseline {reproduced:.7f}% (archive {archived:.7f}%)", flush=True)

    proposal, solver = generate_native_proposal(case.inputs.velocity_mps, instance.source, original,
                                                cfg["propagation"], t)
    with t("direction_compression_s"):
        p = cfg["propagation"]
        direction, compression = prepare_fixed_direction(baseline, proposal, tolerance=p["compression_relative_tolerance"],
                                                         max_rank=p["compression_max_rank"])
    with t("train_cache_s"):
        shared = CachedCurrentTransferReadout(master, instance, direction, head)
        objective = NativeCurrentObjective(shared, case.inputs, proposal, cfg)
    protocol = objective.protocol()
    direction_hashes = [tensor_sha(x) for x in direction.tensors()]
    initial_states, arm_records = {}, {}
    with torch.no_grad():
        preds = {"current_baseline": pa,
                 "propagation_only": original.normalizer.decode_pressure(proposal[None], instance.source[:, 4])[0],
                 "compressed_proposal": original.normalizer.decode_pressure(
                     (baseline + direction.rows(rows))[None], instance.source[:, 4])[0]}
    for arm in ARMS:
        timing = Timings(original.device)
        with timing("arm_clone_s"):
            readout = shared.fork()
        init = current_adapter_state(readout.model)
        with timing("initial_parity_s"):
            with torch.no_grad():
                starting = readout.decode()
                parity = check_pressure_parity(starting, baseline)
                initial_states[arm] = {"prediction_sha256": tensor_sha(starting),
                                       "factors": {k: {n: tensor_sha(v) for n, v in layer.items()}
                                                   for k, layer in init["layers"].items()},
                                       "gain": float(init["gain"]), "parity": parity}
            if arm == ARMS[0]:
                gradient_parity = check_render_gradients(readout, objective.observation_rows)
            elif initial_states[ARMS[0]] != initial_states[arm]:
                raise RuntimeError("The two arms do not share exactly the same initialization")
        fit = fit_current_50(readout, objective, arm, cfg, timing, case.key)
        adapter = directory / f"{arm}.pt"
        with timing("export_s"):
            save_atomic(adapter, {"schema": ADAPTER_SCHEMA, "identity": identity,
                                  "instance": instance_identity, "updates": 50, "selection": "final",
                                  "adapter_state": current_adapter_state(readout.model),
                                  "direction": direction.state_dict(), "direction_hashes": direction_hashes,
                                  "sampling": protocol, "arm": arm, "config": cfg})
        with torch.no_grad():
            with timing("predict_401_cached_first_s"):
                normalized = readout.decode()
            with timing("predict_401_fresh_encoding_first_s"):
                fresh = readout.direct(fresh_encoding=True)
            for name, predict in (("predict_401_cached", readout.decode),
                                  ("predict_401_fresh_encoding", lambda: readout.direct(fresh_encoding=True))):
                measurements = Timings(original.device)
                for trial in range(3):
                    with measurements(str(trial)):
                        predict()
                timing[name + "_s"] = median(measurements.values())
                if measurements.cuda_event_ms:
                    timing.cuda_event_ms[name + "_s"] = median(measurements.cuda_event_ms.values())
            with timing("final_diagnostics_s"):
                fresh_parity = check_pressure_parity(fresh, normalized)
                restored = shared.fork()
                saved = torch.load(adapter, map_location="cpu", weights_only=True)
                restored.model = restore_current_adapter(restored.model, saved, identity, instance_identity)
                stored = FixedDirection(**{k: None if v is None else v.to(original.device)
                                           for k, v in saved["direction"].items()})
                if [tensor_sha(x) for x in stored.tensors()] != saved["direction_hashes"]:
                    raise ValueError("Saved direction differs from its hash")
                restored.direction = stored
                restored.fixed_stamp = restored._stamp()
                restore_parity = check_pressure_parity(restored.decode(), normalized)
                defect_metrics = objective.residual_diagnostics(normalized)
                preds[arm] = original.normalizer.decode_pressure(normalized[None], instance.source[:, 4])[0]
                if arm == "defect_on_50":
                    gain_only = baseline + readout.model.propagation_gain * direction.rows(rows)
                    preds["gain_only_at_on_50"] = original.normalizer.decode_pressure(
                        gain_only[None], instance.source[:, 4])[0]
        active = {id(p) for group in trainable_groups(readout).values() for p in group}
        roster = {name: {"active": id(p) in active, "count": p.numel()}
                  for name, p in readout.model.named_parameters() if p.requires_grad}
        arm_records[arm] = {
            "fit": fit, "timing_s": dict(timing), "cuda_event_ms": timing.cuda_event_ms,
            "adapter_sha256": sha256(adapter), "fresh_vs_cached": fresh_parity, "restored_vs_cached": restore_parity,
            "residual": defect_metrics, "trainable": roster,
        }
        del readout, restored, fresh, normalized
    with t("paired_evaluation_s"):
        if direction_hashes != [tensor_sha(x) for x in direction.tensors()]:
            raise ValueError("The fixed propagation direction was altered by training")
        scores = streamed_window_scores(open_reference(case), preds)
    shared_keys = ("instance_encode_s", "baseline_decode_401_s", "propagation_prepare_s",
                   "propagation_s", "direction_compression_s", "train_cache_s")
    for arm, record in arm_records.items():
        tm = record["timing_s"]
        tm["instance_total_s"] = sum(t[k] for k in shared_keys) + sum(tm[k] for k in ("arm_clone_s", "fit_50_s", "export_s"))
    result = {"record_id": case.record_id, "primary": case.primary, "model_identity": identity,
              "archived_baseline_percent": archived, "instance_identity": instance_identity,
              "reference_sha256": case.reference["sha256"], "solver": solver, "compression": compression,
              "direction_hashes": direction_hashes, "sampling": protocol,
              "sampling_sha256": array_sha(np.frombuffer(json.dumps(protocol, sort_keys=True).encode(), dtype=np.uint8)),
              "initializations": initial_states, "initial_gradient_parity": gradient_parity,
              "timing_s": dict(t), "cuda_event_ms": t.cuda_event_ms, "arms": arm_records, "scores": scores,
              "comparison": delta_report(scores, cfg["evaluation"]["baseline_min_improvement_pp"])}
    _write_json(directory / "CASE_RESULT.json", result)
    print(f"{case.key} scores {json.dumps(result['comparison']['full401_relative_l2_percent'])}", flush=True)
    return result


def run_pair(args):
    started = perf_counter()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (cfg["schema"], cfg["updates"], cfg["selection"], cfg["seed"], cfg["observation_indices"],
            cfg["loss"]["defect_off"], cfg["loss"]["defect_on"], cfg["current_readout_key"]) != (
            SCHEMA, 50, "final", 37, [41, 42], 0., .001, ["boundary_only", "model"]):
        raise ValueError("This entry is bound to the declared v1 pair protocol")
    binding = args.config.parent / cfg["binding"]
    _, head = verify_current_binding(binding)
    history = read_json(head.parent / "COMPARISON.json")["arms"]["boundary_only"]
    device = torch.device(cfg["device"])
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cohort = prepare_external_observations(args.external_root, cfg["evaluation"]["layered_roster_indices"])
    cohort += prepare_center_observations(args.center_instance_root, cfg["evaluation"]["centers"])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "RUN_CONFIG.yaml").write_bytes(args.config.read_bytes())
    _write_json(args.output / "STATUS.json", {"state": "STARTED", "total_cases": len(cohort)})
    shared_timing = Timings(device)
    with shared_timing("model_load_s"):
        original, identity = load_current_model(binding, args.training_config, args.warm_protocol,
                                                args.completion_receipt, device)
    with shared_timing("lora_init_s"):
        master = insert_current_adapters(clone_current(original))
    print(f"Loaded E200+U96 and prepared one shared factor initialization on {device}", flush=True)
    records = {}
    for case in cohort:
        historical = (history["heldout"]["per_record"][case.record_id] if case.primary
                      else history["centered_cases"][case.key])
        records[case.key] = run_case(original, master, identity, case, historical["relative_l2_percent"],
                                     cfg, args.output / case.key)
        _write_json(args.output / "STATUS.json", {"state": "RUNNING", "completed": list(records),
                                                  "total_cases": len(cohort)})
    primary = {k: v for k, v in records.items() if v["primary"]}
    pooled = pool_case_scores(v["scores"] for v in primary.values())
    report = {"schema": SCHEMA, "identity": identity, "timing_s": dict(shared_timing),
              "wall_s": perf_counter() - started,
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
              "cuda_event_ms": shared_timing.cuda_event_ms, "primary_records": list(primary),
              "primary_raw_scores": pooled,
              "primary": delta_report(pooled, cfg["evaluation"]["baseline_min_improvement_pp"]),
              "cases": records}
    _write_json(args.output / "COMPARISON.json", report)
    _write_json(args.output / "STATUS.json", {"state": "COMPLETE", "completed": list(records),
                                              "primary": report["primary"], "wall_s": report["wall_s"]})
    print(json.dumps(report["primary"], indent=2), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    for arg in ("config", "training-config", "warm-protocol", "completion-receipt",
                "external-root", "center-instance-root", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    run_pair(parser.parse_args())


if __name__ == "__main__":
    main()
