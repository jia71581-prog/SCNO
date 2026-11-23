"""Matched UL baseline/arrival trial from one immutable v3 checkpoint."""

from argparse import ArgumentParser
from copy import deepcopy
from inspect import signature
from pathlib import Path
import json
import math
import os
import signal
import time

import torch

import train_tianhe_ul_b5 as legacy
from arrival_phase import ArrivalReceivers, LocalArrivalPhase
from full_model_warm_start import FourStratumFrameSampler
from train_ul_repaired_200 import role_partition
from ul_backbone_lr_repair import BackboneRateController


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identical_state(current, saved):
    if isinstance(saved, torch.Tensor):
        return isinstance(current, torch.Tensor) and torch.equal(current.detach().cpu(), saved.detach().cpu())
    if isinstance(saved, dict):
        return current.keys() == saved.keys() and all(identical_state(current[k], saved[k]) for k in saved)
    if isinstance(saved, (tuple, list)):
        return len(current) == len(saved) and all(identical_state(a, b) for a, b in zip(current, saved))
    return current == saved


class ArrivalOptimizer:
    """Track only added weights, with the inherited local BASE AdamW settings."""

    group_names = ("BASE_decay_local", "BASE_no_decay_local")

    def __init__(self, residual, control, multiplier=1.):
        self.residual = residual
        self.multiplier = multiplier
        self.base_groups = {group["group_name"]: group for group in control.optimizer.param_groups}
        require(all(name in self.base_groups for name in self.group_names), "Missing original local BASE groups")
        adam_options = signature(torch.optim.AdamW).parameters
        defaults = {name: value for name, value in control.optimizer.defaults.items() if name in adam_options}
        named = tuple(residual.named_parameters())
        groups = []
        for no_decay, name in enumerate(self.group_names):
            options = {key: self.base_groups[name][key] for key in defaults}
            options["params"] = [parameter for key, parameter in named if key.endswith(".bias") == bool(no_decay)]
            groups.append(options)
        self.optimizer = torch.optim.AdamW(groups, **defaults)

    def begin_update(self):
        self.residual.train()
        self.optimizer.zero_grad(set_to_none=True)
        for side, name in zip(self.optimizer.param_groups, self.group_names):
            side["lr"] = self.multiplier * self.base_groups[name]["lr"]

    def step(self, clipping):
        norm = torch.nn.utils.clip_grad_norm_(self.residual.parameters(), clipping, error_if_nonfinite=True)
        self.optimizer.step()
        return norm.item()

    def state_dict(self):
        return {"model": self.residual.state_dict(), "optimizer": self.optimizer.state_dict(),
                "semantics": self.residual.semantics()}

    def restore(self, state):
        require(state["semantics"] == self.residual.semantics(), "Arrival branch semantics changed")
        self.residual.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])


def prepare_base(args, data, saved, device):
    binding = saved["binding"]
    require(binding["schema"] == "tianhe_uploaded_ul_scno_b5_801_sparse_4shot_v3"
            and binding["trainer_sha256"] == legacy.sha_file(legacy.__file__)
            and binding["dataset_binding"] == data.binding
            and binding["source_manifest_sha256"] == legacy.sha_file(args.source_manifest)
            and binding["ul_config_sha256"] == legacy.sha_file(args.ul_config),
            "Anchor must belong to this audited v3 trainer, dataset and source")
    model = legacy.build_marmousi_ul(data, binding["seed"], device, binding["query_block"],
                                     interface_correction=binding["velocity_interface_correction"])
    require(saved["model_semantics"] == model.parent.checkpoint_semantics()
            and saved["repair_settings"] == model.checkpoint_payload()["repair_settings"],
            "Anchor model semantics differ from the reviewed architecture")
    model.load_state_dict(saved["model"], strict=True)
    observations = deepcopy(data.observe).to(device=device, dtype=torch.float32)
    count = data.record_count("train")
    width = binding["effective_batch"]
    require(count == 3200 and width == 64 and count % width == 0, "Expected the exact four-shot training release")
    config = {**legacy.read_json(args.ul_config), "protocol": "joint_from_start",
              "affine_calibration": False, "seed": binding["seed"]}
    factor = binding["stage_plan"]["backbone_lr_policy"]["factor"]
    control = BackboneRateController(model, config, epochs=binding["epochs"], updates_per_epoch=count // width,
                                    named_roles=role_partition(model), train_ids=data.train_ids,
                                    validation_ids=data.validation_ids, test_ids=data.splits["test"],
                                    native_times=model.full_times_s, startup_mode="resume", backbone_factor=factor)
    control.load_state_dict(saved["controller"], restore_random_state=False)
    require(control.current_phase == "C" and control.completed_updates > 0,
            "Arrival trials start from an already-trained joint C anchor")
    sampler = FourStratumFrameSampler(len(data.times), binding["sampler"]["samples_per_stratum"], seed=binding["seed"])
    sampler.load_state_dict(saved["sampler"])
    require(sampler.drawn_records == saved["epoch"] * count + saved["offset"]
            == control.completed_updates * width, "Anchor data and optimizer cursors disagree")
    times, floor, objective = legacy.observation_objective(data, sampler, binding["objective"]["kind"], device)
    require(objective == binding["objective"], "Anchor objective changed")
    return model, observations, control, sampler, times, floor


@torch.no_grad()
def verify_start(data, model, reader, views, sampler, time_weights, floor):
    """Compare exact B5 and the original objective on identical TRAIN rows."""
    prior = sampler.state_dict()
    try:
        sampled = sampler.sample(1, device="cpu")
    finally:
        sampler.load_state_dict(prior)
    frames = sampled.indices[0]
    shot = data.sample("train", 0, frames.numpy())
    context = model.encode(shot["velocity"], shot["source"])
    baseline = reader(model.decode(context, frames.to(model.device)))
    other = views.decode(context, frames.to(model.device))
    require(torch.equal(baseline, other), "Initialized arrival branch changed anchor predictions")
    target = shot["target"].to(baseline)
    weights = sampled.weights[0].to(model.device) * time_weights[frames.to(model.device)]
    denominator = max(floor, shot["full_energy"])
    base_loss = ((baseline - target).square().sum(-1).double() * weights).sum() / denominator
    branch_loss = ((other - target).square().sum(-1).double() * weights).sum() / denominator
    require(torch.equal(base_loss, branch_loss), "Initialized arrival branch changed initial loss")
    return {"identical_b5": True, "identical_loss": float(base_loss),
            "model_id": shot["model_id"], "shot_id": shot["shot_id"], "frames": len(frames)}


def verify_heads(data, view):
    # A small actual label slice suffices to test that both zero-output heads
    # have a first-update gradient, without taking any optimizer step.
    slots = torch.tensor([12, 30, 80, 145])
    shot = data.sample("train", 0, slots.numpy())
    pressure = view(shot["velocity"], shot["source"], slots)
    target = shot["target"].to(pressure)
    score = (pressure[0] - target).square().mean()
    heads = (view.residual.amplitude.weight, view.residual.phase.weight)
    gradient = torch.autograd.grad(score, heads)
    norm = [float(value.norm()) for value in gradient]
    require(all(math.isfinite(value) and value > 0. for value in norm), "Expected positive initial gradients on both heads")
    return {"active_probe_times": slots.tolist(), "head_gradient_norms": dict(zip(("amplitude", "phase"), norm)),
            "arrival_parameters": sum(value.numel() for value in view.residual.parameters())}


def run(args):
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "Use one rank for each matched trial arm")
    device = torch.device(args.device)
    torch.set_num_threads(2 if device.type == "cpu" else 4)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(.94, device)
    data = legacy.CommittedMarmousi(args.dataset)
    checkpoint_path = args.output / "checkpoint.pt"
    original = torch.load(args.anchor, mmap=True, map_location="cpu", weights_only=True)
    saved = torch.load(checkpoint_path, mmap=True, map_location="cpu", weights_only=True) if args.resume else original
    start = original["controller"]["completed_updates"]
    stop_at = start + args.updates
    binding = saved["binding"]
    require(saved["binding"] == original["binding"], "Trial no longer matches the immutable anchor")
    manifest = {"version": 1, "variant": args.variant, "anchor_sha256": legacy.sha_file(args.anchor),
                "start": start, "target": stop_at, "legacy_binding": binding,
                "arrival_lr_multiplier": args.head_lr_multiplier if args.variant == "arrival" else None,
                "phase_source_sha256": legacy.sha_file(Path(__file__).with_name("arrival_phase.py")),
                "trial_source_sha256": legacy.sha_file(__file__)}
    artifact = args.output / "RUN.json"
    if args.resume:
        require(legacy.read_json(artifact) == json.loads(json.dumps(manifest)) and saved["trial"] == manifest,
                "Trial manifest mismatch")
    else:
        require(not artifact.exists() and not checkpoint_path.exists(), "Use a fresh trial output")
    model, readout, control, sampler, times, floor = prepare_base(args, data, saved, device)
    require(start <= control.completed_updates <= stop_at <= control.plan["total_updates"], "Invalid trial budget")
    if args.variant == "arrival":
        residual = LocalArrivalPhase(model.parent.decoder.in_channels).to(device)
        view = ArrivalReceivers(model, residual, readout)
        side = ArrivalOptimizer(residual, control, args.head_lr_multiplier)
        if args.resume:
            side.restore(saved["arrival"])
        network, project = view, torch.nn.Identity()
    else:
        side = None
        network, project = model, readout
    restored = control.state_dict()
    require(identical_state(restored["optimizer"], saved["controller"]["optimizer"])
            and identical_state(restored["scheduler"], saved["controller"]["scheduler"]),
            "Original AdamW state or scheduler changed when attaching the arrival sidecar")
    del restored
    if side is not None and not args.resume:
        model.eval()
        view.residual.eval()
        proof = verify_start(data, model, readout, view, sampler, times, floor)
        if args.verify_only:
            proof.update(verify_heads(data, view))
        print(json.dumps({"stage": "arrival_anchored", **proof, "original_optimizer_exact": True}), flush=True)
    legacy.restore_rank_rng(saved["rank_rngs"][0], device)
    epoch, offset = saved["epoch"], saved["offset"]
    micro = saved["microbatch"]
    last_validated_epoch = saved["last_validated_epoch"]
    del saved, original
    if args.verify_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with legacy.writer_lock(args.output / "writer.lock", True):
        if args.resume:
            legacy.trim_log(args.output / "metrics.jsonl", control.completed_updates)
        else:
            legacy.atomic_json(artifact, manifest)
        interrupted = False

        def request_stop(*_):
            nonlocal interrupted
            interrupted = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

        def report(state, **extra):
            legacy.atomic_json(args.output / "STATUS.json", {
                "state": state, "variant": args.variant, "accepted_updates": control.completed_updates,
                "epoch": epoch, "records_completed_in_epoch": offset, "microbatch": micro, **extra})

        def save():
            legacy.checkpoint(checkpoint_path, {**model.checkpoint_payload(), "binding": binding,
                "model_semantics": model.parent.checkpoint_semantics(), "controller": control.state_dict(),
                "sampler": sampler.state_dict(), "rank_rngs": [legacy.rank_rng(device)],
                "epoch": epoch, "offset": offset, "microbatch": micro, "last_validated_epoch": last_validated_epoch,
                "arrival": side.state_dict() if side is not None else None, "trial": manifest})

        width, count = binding["effective_batch"], data.record_count("train")
        current_order = None
        while control.completed_updates < stop_at and not interrupted:
            if current_order is None:
                current_order = legacy.epoch_pairs(data.shot_counts["train"], width, binding["seed"], epoch)
            selected = current_order[offset:offset + width]
            started = time.perf_counter()
            stage = control.begin_update()
            control.optimizer.zero_grad(set_to_none=True)
            if side is not None:
                side.begin_update()
            previous_sampler = sampler.state_dict()
            frame_sample = sampler.sample(len(selected), device="cpu")
            records = [data.sample("train", index, slots.numpy()) for index, slots in zip(selected, frame_sample.indices)]
            objective = 0.
            for first in range(0, len(records), micro):
                if interrupted:
                    break
                past = min(first + micro, len(records))
                local = legacy.micro_loss(network, project, records, frame_sample, slice(first, past), times, floor)
                require(torch.isfinite(local).item(), "Nonfinite trial objective")
                weighted = local * ((past - first) / len(records))
                weighted.backward()
                objective += weighted.item()
                del weighted, local
            if interrupted:
                sampler.load_state_dict(previous_sampler)
                break
            norm = control.step().item()
            side_norm = side.step(stage["gradient_clip_norm"]) if side is not None else None
            control.optimizer.zero_grad(set_to_none=True)
            if side is not None:
                side.optimizer.zero_grad(set_to_none=True)
            offset += width
            if offset == count:
                epoch, offset, current_order = epoch + 1, 0, None
            update = control.completed_updates
            record = {"completed_updates": update, "accepted_updates": update, "epoch": epoch, "offset": offset,
                      "train_b5_objective": objective, "optimizer_update_s": time.perf_counter() - started,
                      "BASE_gradient_norm": norm, "arrival_gradient_norm": side_norm,
                      "global_shots_in_update": width, "microbatch": micro, "learning_rates": stage["learning_rates"]}
            if side is not None:
                record["arrival_learning_rates"] = [group["lr"] for group in side.optimizer.param_groups]
            if device.type == "cuda":
                record["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                record["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
            if (not offset and epoch % binding["validation_every_epochs"] == 0) or update == stop_at:
                report("VALIDATING")
                record["validation"] = legacy.validate(data, network, project, binding["validation_probes"],
                                                       binding["validation_block"])
                last_validated_epoch = epoch
            with (args.output / "metrics.jsonl").open("a") as log:
                log.write(json.dumps(record, allow_nan=False) + "\n")
            committed = update in (start + 1, stop_at) or not offset
            if committed:
                save()
            report("TRAINING", checkpoint_committed=committed, optimizer_update_s=record["optimizer_update_s"])
        save()
        report("COMPLETE" if control.completed_updates == stop_at else "PAUSED_AT_UPDATE_BOUNDARY",
               checkpoint_committed=True)


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("dataset", "output", "anchor", "source-manifest", "ul-config"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--variant", choices=("baseline", "arrival"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--head-lr-multiplier", type=float, default=1.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    require(args.updates > 0 and not (args.resume and args.verify_only)
            and math.isfinite(args.head_lr_multiplier) and args.head_lr_multiplier > 0., "Invalid trial invocation")
    run(args)


if __name__ == "__main__":
    main()
