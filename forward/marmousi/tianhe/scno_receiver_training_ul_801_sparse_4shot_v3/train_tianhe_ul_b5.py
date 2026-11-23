"""Adapt uploaded UL SCNO training to Tianhe's committed single-shot B5 labels."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from heapq import heapify, heappop, heappush
from pathlib import Path
import argparse
import gc
import json
import math
import os
import random
import signal
import time

import numpy as np
import torch

from full_model_warm_start import FourStratumFrameSampler
from marmousi_ul_model_801 import build_marmousi_ul
from repair_stage_schedule import capture_rng, restore_rng
from train_ul_repaired_200 import role_partition
from train_v2 import trim_log, writer_lock
from ul_backbone_lr_repair import BackboneRateController
from tools.marmousi_target_dataset_refined.common import atomic_json, read_json, sha_file
from tianhe_observations import CommittedMarmousi
from ul_receiver_ranks import RankRuntime


def checkpoint(path, payload):
    pending = path.with_suffix(".pending")
    with pending.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def rank_rng(device):
    state = capture_rng()
    cuda = state.pop("torch_cuda")
    state["device_rng"] = cuda[device.index] if cuda and device.type == "cuda" else None
    return state


def restore_rank_rng(state, device):
    restore_rng({**state, "torch_cuda": []})
    if device.type == "cuda" and state["device_rng"] is not None:
        torch.cuda.set_rng_state(state["device_rng"], device=device)


def micro_loss(model, readout, shots, frame_sample, section, time_weights, energy_floor):
    """UL's resident sampled loss, with the reference and prediction in B5."""
    velocity = torch.stack([shot["velocity"] for shot in shots[section]]).to(model.device)
    source = torch.stack([shot["source"] for shot in shots[section]]).to(model.device)
    target = torch.stack([shot["target"] for shot in shots[section]]).to(model.device)
    frame_ids = frame_sample.indices[section].to(model.device)
    weights = frame_sample.weights[section].to(model.device, dtype=torch.float64)
    weights = weights * time_weights[frame_ids]
    denominator = torch.as_tensor([s["full_energy"] for s in shots[section]],
                                  device=model.device, dtype=torch.float64).clamp_min(energy_floor)
    pressure = model(velocity, source, frame_indices=frame_ids, normalized=True)
    predicted = readout(pressure)
    if predicted.shape != target.shape:
        raise ValueError("UL receiver projection does not match the committed label shape")
    frame_sse = (predicted - target).square().sum(-1)
    return ((frame_sse.double() * weights).sum(1) / denominator).mean()


@torch.no_grad()
def verify_projection(data, model, readout):
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(20261119)
    trial = torch.randn((3, 201, 201), dtype=torch.float64, generator=generator)
    trial[:, 0] = 0.
    torch.testing.assert_close(data.observe(trial), data.observe.explicit(trial),
                               rtol=2e-12, atol=3e-14)
    # Use different media and their independent sources in the same batch.
    slots = np.asarray([[0, 37, 303, 800], [0, 41, 453, 800]])
    counts = data.shot_counts["train"]
    picks = [9 % counts[0], data.offsets["train"][-2] + 28 % counts[-1]]
    shots = [data.sample("train", index, indices) for index, indices in zip(picks, slots)]
    velocity = torch.stack([shot["velocity"] for shot in shots]).to(model.device)
    source = torch.stack([shot["source"] for shot in shots]).to(model.device)
    pressure = model(velocity, source, frame_indices=torch.from_numpy(slots).to(model.device), normalized=True)
    if pressure.shape != (2, 4, 201, 201) or not torch.isfinite(pressure).all():
        raise ValueError("UL pressure forward returned an invalid field")
    if torch.count_nonzero(pressure[..., 0, :]):
        raise ValueError("UL pressure must vanish exactly at the top free surface")
    values = readout(pressure)
    reference = readout.explicit(pressure)
    torch.testing.assert_close(values, reference, rtol=2e-5, atol=1e-6)
    if values.shape != (2, 4, 201) or not torch.isfinite(values).all():
        raise ValueError("Invalid UL B5 receiver prediction")
    report = {
        "status": "PASS", "axis_frames": len(model.full_times_s), "exact_top_zero": True,
        "max_explicit_b5_difference": (values - reference).abs().max().item(),
        "sample_ids": [[shot["model_id"], shot["shot_id"]] for shot in shots],
        "slot_indices": slots.tolist(), "device": str(model.device),
        "record_dt_s": .005, "dense_prefix_decimated_view_only": data.dense_prefix_probe,
        "dataset_shots_by_split": data.shots_by_split, "sparse_source_roster": data.sparse,
        "normalization_training_shots": data.normalization_training_shots,
    }
    print(json.dumps({"stage": "preflight", **report}), flush=True)
    return report


def validation_plan(data, records):
    by_family = {}
    for index, model in enumerate(data.validation_ids):
        by_family.setdefault(data.rows[model]["family"], []).append(index)
    ordered = [by_family[family] for family in sorted(by_family)]
    # Prefer a spread of independent validation families before reusing any.
    rng = np.random.default_rng(271119)
    within = [rng.permutation(indices) for indices in ordered]
    model_indices = [
        int(indices[position])
        for position in range(max(map(len, within))) for indices in within if position < len(indices)
    ][:records]
    return [int(data.offsets["validation"][index]) + (17 * j + 7) % data.shot_counts["validation"][index]
            for j, index in enumerate(model_indices)]


def observation_objective(data, sampler, mode, device):
    sizes = sampler.sizes
    if mode == "late_balanced":
        # Favor the late wave train, where multi-path arrivals and the
        # receiver coda can be weak compared with the early direct pressure.
        # A single denominator is shared by every receiver and time bin.
        raw = (1., 1.25, 1.5, 2.)
        floor = max(1e-12, .01 * data.training_reference_energy_mean)
    else:
        raw, floor = (1., 1., 1., 1.), 1e-12
    scale = sum(size * weight for size, weight in zip(sizes, raw)) / sum(sizes)
    effective = [weight / scale for weight in raw]
    per_frame = torch.from_numpy(np.repeat(effective, sizes)).to(device=device, dtype=torch.float64)
    binding = {
        "kind": mode, "base_loss": "UL stratified record-relative squared B5 error",
        "time_windows_frames": list(sizes), "prescaled_window_weights": list(raw),
        "window_weights": effective, "window_weight_time_mean": 1.,
        "record_denominator": "max(full-shot sum of squared normalized B5 labels, energy_floor)",
        "normalized_shot_energy_floor": floor,
        "floor_provenance": "1% of mean normalized energy of the saved shots on frozen 128 TRAIN models"
                            if mode == "late_balanced" else "original UL numerical floor 1e-12",
        "no_trace_or_window_energy_normalization": True,
    }
    return per_frame, floor, binding


def epoch_pairs(shot_counts, effective_batch, seed, epoch):
    """Shuffle saved shots, spreading each velocity across different updates."""
    if not 1 <= effective_batch <= len(shot_counts) or min(shot_counts) < 1:
        raise ValueError("A training update needs no more than the number of training media")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    offsets = np.cumsum((0, *shot_counts), dtype=np.int64)
    total = int(offsets[-1])
    batches = [[] for _ in range(math.ceil(total / effective_batch))]
    open_batches = [(-min(effective_batch, total - i * effective_batch), i) for i in range(len(batches))]
    heapify(open_batches)
    # Place the longest rosters first. Giving each medium to distinct batches
    # with the most capacity spreads abundant media throughout the epoch.
    order = sorted(rng.permutation(len(shot_counts)), key=lambda model: -shot_counts[model])
    for model in order:
        reserved = []
        for slot in rng.permutation(shot_counts[model]):
            if not open_batches:
                # A highly skewed roster may need the same velocity twice in
                # one update. Preserve the shot after using all distinct batches.
                open_batches, reserved = reserved, []
                heapify(open_batches)
            free, batch = heappop(open_batches)
            batches[batch].append(offsets[model] + slot)
            if free + 1 < 0:
                reserved.append((free + 1, batch))
        for remaining in reserved:
            heappush(open_batches, remaining)
    # Leave the possibly smaller final batch last for the accepted-update cursor.
    batch_order = [*rng.permutation(len(batches) - 1), len(batches) - 1]
    return np.concatenate([rng.permutation(batches[batch]) for batch in batch_order])


@torch.no_grad()
def validate(data, model, readout, probes, chunk):
    """Keep the UL complete-time validation protocol on fixed VAL-only shots."""
    model.eval()
    slots = np.arange(len(data.times))
    windows = [part for part in np.array_split(slots, 4)]
    window_error = np.zeros(4, dtype=np.float64)
    window_energy = np.zeros(4, dtype=np.float64)
    family_totals = {}
    details = []
    for index in probes:
        shot = data.sample("validation", index, slots)
        context = model.encode(shot["velocity"], shot["source"])
        target = shot["target"].to(model.device, dtype=torch.float64)
        record_error, record_energy = 0., 0.
        for window_index, window in enumerate(windows):
            first, after = int(window[0]), int(window[-1]) + 1
            local_error = 0.
            for start in range(first, after, chunk):
                stop = min(start + chunk, after)
                indices = torch.arange(start, stop, device=model.device)
                pressure = model.decode(context, frame_indices=indices, normalized=True)
                prediction = readout(pressure)[0]
                local_error += (prediction.double() - target[start:stop]).square().sum().item()
            local_energy = target[first:after].square().sum().item()
            window_error[window_index] += local_error
            window_energy[window_index] += local_energy
            record_error += local_error
            record_energy += local_energy
        family = data.rows[shot["model_id"]]["family"]
        group = family_totals.setdefault(family, {"shots": 0, "error": 0., "energy": 0.})
        group["shots"] += 1
        group["error"] += record_error
        group["energy"] += record_energy
        details.append({"model_id": shot["model_id"], "shot_id": shot["shot_id"],
                        "family": family, "relative_l2": math.sqrt(record_error / record_energy)})
    squared_error, energy = float(window_error.sum()), float(window_energy.sum())
    if not energy > 0. or not math.isfinite(squared_error + energy):
        raise FloatingPointError("Invalid validation B5 energies")
    return {"pooled_relative_l2": math.sqrt(squared_error / energy),
            "frames_per_shot": len(slots), "receivers": 201, "probe_records": details,
            "pooled_by_family": {family: {"shots": values["shots"],
                                         "relative_l2": math.sqrt(values["error"] / values["energy"])}
                                 for family, values in family_totals.items()},
            "pooled_by_time_window": [
                {"first_s": float(data.times[window[0]]), "last_s": float(data.times[window[-1]]),
                 "frames": len(window), "reference_energy": float(reference),
                 "relative_l2": math.sqrt(error / reference) if reference > 0 else None}
                for window, error, reference in zip(windows, window_error, window_energy)
            ],
            "scope": "held-out validation single shots, complete B5 time axis, unweighted physical metrics"}


def training_loop(args, data, model, readout, preflight, ranks):
    output = args.output.resolve()
    train_records = data.record_count("train")
    microbatch = min(args.microbatch, args.effective_batch)
    updates_per_epoch = math.ceil(train_records / args.effective_batch)
    source_config = read_json(args.ul_config)
    # The linked UL job was already in joint C. The new Marmousi run starts
    # with fresh weights and a fresh C optimizer using those same roles and LR.
    config = {**source_config, "protocol": "joint_from_start", "affine_calibration": False, "seed": args.seed}
    control = BackboneRateController(
        model, config, epochs=args.epochs, updates_per_epoch=updates_per_epoch,
        named_roles=role_partition(model), train_ids=data.train_ids, validation_ids=data.validation_ids,
        test_ids=data.splits["test"], native_times=model.full_times_s,
        startup_mode="resume" if args.resume else "fresh", backbone_factor=args.backbone_factor,
    )
    sampler = FourStratumFrameSampler(len(data.times), args.times_per_stratum, seed=args.seed)
    time_weights, energy_floor, criterion = observation_objective(data, sampler, args.objective, model.device)
    probes = validation_plan(data, args.validation_records)
    binding = {
        "schema": "tianhe_uploaded_ul_scno_b5_801_sparse_4shot_v3", "data": str(data.root),
        "dataset_binding": data.binding, "source_manifest_sha256": sha_file(args.source_manifest),
        "trainer_sha256": sha_file(__file__), "ul_config_sha256": sha_file(args.ul_config),
        "model_adapter_sha256": sha_file(Path(__file__).with_name("marmousi_ul_model_801.py")),
        "b5_adapter_sha256": sha_file(Path(__file__).with_name("tianhe_observations.py")),
        "distributed_adapter_sha256": sha_file(Path(__file__).with_name("ul_receiver_ranks.py")),
        "model_semantics": model.parent.checkpoint_semantics(), "repair": model.repair_receipt,
        "basis_size": model.basis_size, "base_hidden_width": model.base_hidden_width,
        "model_checkpoint_initialization": "fresh; uploaded UL architecture without a source checkpoint",
        "model_normalization_metadata": data.metadata,
        "normalization": "frozen train128 numerical receiver RMS from their current saved source roster",
        "objective": criterion, "velocity_interface_correction": args.interface_correction,
        "selection": "held-out validation-only fixed shots; pooled relative L2 over all 801 times",
        "observation_times_per_shot": len(data.times), "observation_dt_s": .005,
        "optimizer": "uploaded UL BackboneRateController and role_partition (joint BASE/DETAIL phase C)",
        "validation_probes": probes, "validation_block": args.validation_block,
        "validation_every_epochs": args.validation_every,
        "test_labels_opened_by_trainer": False,
        "train_single_shots": train_records, "validation_single_shots": data.record_count("validation"),
        "dataset_single_shots_by_split": data.shots_by_split,
        "epoch_pair_order": "seeded shuffled saved shot slots; longest velocity rosters spread across updates; "
                            "every retained TRAIN model/source pair visited exactly once per epoch",
        "data_parallel": "stride one global roster by GLOBAL rank, sum shot-weighted gradients before AdamW; "
                         "the shared state can resume under a different rank count",
        "freeze_ic_weights_and_absorbed_gain": True,
        "stage_plan": control.plan, "sampler": sampler.semantics, "seed": args.seed,
        "effective_batch": args.effective_batch, "initial_microbatch": microbatch,
        "query_block": args.query_block, "epochs": args.epochs, "checkpoint_interval": args.checkpoint_interval,
        "torch_version": str(torch.__version__), "device_type": model.device.type,
    }
    binding_path, checkpoint_path = output / "RUN.json", output / "checkpoint.pt"
    epoch, offset, best = 0, 0, None
    last_validated_epoch = 0
    if args.resume:
        def load_checkpoint():
            if not binding_path.is_file() or read_json(binding_path) != json.loads(json.dumps(binding)):
                raise ValueError("UL Marmousi run bindings or settings changed")
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if (saved["binding"] != binding or saved["model_semantics"] != model.parent.checkpoint_semantics()
                    or saved["repair_settings"] != model.checkpoint_payload()["repair_settings"]):
                raise ValueError("UL time/architecture or dataset checkpoint mismatch")
            model.load_state_dict(saved["model"], strict=True)
            control.load_state_dict(saved["controller"], restore_random_state=False)
            sampler.load_state_dict(saved["sampler"])
            if sampler.drawn_records != saved["epoch"] * train_records + saved["offset"]:
                raise ValueError("UL sampler and dataset progress cursor disagree")
            if ranks.rank < len(saved["rank_rngs"]):
                restore_rank_rng(saved["rank_rngs"][ranks.rank], model.device)
            return (saved["epoch"], saved["offset"], saved["best_validation"],
                    saved["last_validated_epoch"], saved["microbatch"])

        epoch, offset, best, last_validated_epoch, microbatch = ranks.on_each(load_checkpoint)
        ranks.on_primary(lambda: trim_log(output / "metrics.jsonl", control.completed_updates))
    else:
        def prepare_output():
            incomplete_bootstrap = (binding_path.is_file() and not checkpoint_path.exists()
                                    and not (output / "metrics.jsonl").exists()
                                    and not (output / "best_validation.pt").exists()
                                    and read_json(binding_path) == json.loads(json.dumps(binding)))
            if checkpoint_path.exists() or (binding_path.exists() and not incomplete_bootstrap):
                raise ValueError("Existing training state requires --resume")
            atomic_json(binding_path, binding)
            atomic_json(output / "PREFLIGHT.json", preflight)

        ranks.on_primary(prepare_output)

    stop = False

    def stop_at_boundary(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stop_at_boundary)
    signal.signal(signal.SIGINT, stop_at_boundary)
    started = time.monotonic()
    last_checkpoint = control.completed_updates if args.resume else -1

    def publish(state, **extra):
        if not ranks.primary:
            return
        atomic_json(output / "STATUS.json", {
            "state": state, "pid": os.getpid(), "time_unix": time.time(),
            "updated_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.monotonic() - started,
            "dataset": str(data.root), "epochs_completed": epoch, "records_completed_in_epoch": offset,
            "accepted_updates": control.completed_updates, "microbatch": microbatch,
            "effective_batch": args.effective_batch, "best_validation": best,
            "distributed_world_size": ranks.size, "global_batch": args.effective_batch,
            "per_rank_full_batch": [len(range(rank, args.effective_batch, ranks.size))
                                    for rank in range(ranks.size)],
            "last_validated_epoch": last_validated_epoch,
            "phase": control.current_phase, "train_single_shots": train_records,
            "dataset_single_shots_by_split": data.shots_by_split,
            "objective": args.objective, "velocity_interface_correction": args.interface_correction,
            "epoch_pair_order": binding["epoch_pair_order"],
            "observation_times_per_shot": len(data.times),
            "projected_times_per_shot_per_update": 4 * args.times_per_stratum,
            "checkpoint_committed_updates": last_checkpoint,
            "checkpoint_committed": last_checkpoint == control.completed_updates, **extra,
        })

    def save(path=checkpoint_path):
        nonlocal last_checkpoint
        states = ranks.gather(rank_rng(model.device))

        def write():
            payload = model.checkpoint_payload()
            checkpoint(path, {**payload, "model_semantics": model.parent.checkpoint_semantics(),
                              "binding": binding, "controller": control.state_dict(),
                              "sampler": sampler.state_dict(), "epoch": epoch, "offset": offset,
                              "last_validated_epoch": last_validated_epoch,
                              "best_validation": best, "microbatch": microbatch,
                              "rank_rngs": states, "saved_world_size": ranks.size})

        ranks.on_primary(write)
        if path == checkpoint_path:
            last_checkpoint = control.completed_updates

    def validate_if_due():
        nonlocal last_validated_epoch, best
        if (offset != 0 or last_validated_epoch >= epoch or
                (epoch % args.validation_every != 0 and epoch != args.epochs)):
            return None
        if ranks.action(stop):
            return None

        def evaluate():
            publish("VALIDATING")
            return validate(data, model, readout, probes, args.validation_block)

        result = ranks.on_primary(evaluate, share_result=True)
        last_validated_epoch = epoch
        score = result["pooled_relative_l2"]
        if best is None or score < best["pooled_relative_l2"]:
            best = {"pooled_relative_l2": score, "epoch": epoch,
                    "accepted_updates": control.completed_updates}
            save(output / "best_validation.pt")
        save()
        return result

    if not args.resume:
        save()
    ranks.on_primary(lambda: publish("TRAINING"))
    # A checkpoint can precede a time-consuming validation. Complete such a
    # pending validation after resume before changing the corresponding weights.
    if args.resume:
        resumed_validation = validate_if_due()
        if resumed_validation is not None:
            def log_resumed_validation():
                print(json.dumps({"stage": "resumed_pending_validation", "epoch": epoch,
                                  "validation": resumed_validation}), flush=True)
                publish("TRAINING")

            ranks.on_primary(log_resumed_validation)
    mutable_buffers = [value for name, value in model.named_buffers()
                       if name.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    paused = False
    while epoch < args.epochs and not paused:
        order = epoch_pairs(data.shot_counts["train"], args.effective_batch, args.seed, epoch)
        while offset < train_records:
            if ranks.action(stop):
                paused = True
                break
            selected = order[offset:offset + args.effective_batch]
            update_started = time.monotonic()
            sampler_before = sampler.state_dict()

            def start_update():
                step = control.begin_update()
                global_sample = sampler.sample(len(selected), device="cpu")
                local = slice(ranks.rank, None, ranks.size)
                sample = type(global_sample)(global_sample.indices[local], global_sample.weights[local])
                records = [data.sample("train", index, frames.numpy())
                           for index, frames in zip(selected[local], sample.indices)]
                return step, sample, records

            controller_step, sample, records = ranks.on_each(start_update)
            stored_rng = rank_rng(model.device)
            stored_buffers = [buffer.clone() for buffer in mutable_buffers]
            while True:
                control.optimizer.zero_grad(set_to_none=True)
                loss_sum, retry_memory, problem, loss = 0., False, None, None
                try:
                    for start in range(0, len(records), microbatch):
                        if stop:
                            break
                        stop_at = min(start + microbatch, len(records))
                        loss = micro_loss(model, readout, records, sample, slice(start, stop_at),
                                          time_weights, energy_floor)
                        if not torch.isfinite(loss).item():
                            raise FloatingPointError("Nonfinite UL B5 training loss")
                        ratio = (stop_at - start) / len(selected)
                        (loss * ratio).backward()
                        loss_sum += float(loss.detach()) * ratio
                        del loss
                except torch.cuda.OutOfMemoryError as error:
                    loss = None
                    retry_memory = microbatch > 1
                    if not retry_memory:
                        problem = error
                except Exception as error:
                    problem = error
                # No backward collective precedes this consensus. All ranks
                # either reduce the same update, retry it, or stop together.
                action = ranks.action(3 if problem is not None else 2 if stop else int(retry_memory))
                if action == 3:
                    if problem is not None:
                        raise problem
                    raise RuntimeError("Another rank failed during the UL backward pass")
                if action == 0:
                    break
                control.optimizer.zero_grad(set_to_none=True)
                for buffer, saved_buffer in zip(mutable_buffers, stored_buffers):
                    buffer.copy_(saved_buffer)
                restore_rank_rng(stored_rng, model.device)
                if action == 2:
                    sampler.load_state_dict(sampler_before)
                    paused = True
                    break
                microbatch = max(1, microbatch // 2)
                gc.collect()
                torch.cuda.empty_cache()
                if ranks.primary:
                    print(json.dumps({"stage": "oom_retry_same_batch", "microbatch": microbatch}), flush=True)
            if paused:
                break
            parameters = [p for group in control.optimizer.param_groups for p in group["params"]]
            ranks.reduce_gradients(parameters)
            grad_norm = ranks.on_each(lambda: control.step().item())
            control.optimizer.zero_grad(set_to_none=True)
            loss_sum, = ranks.stats([loss_sum])
            offset += len(selected)
            epoch_finished = offset == train_records
            update = control.completed_updates
            if epoch_finished:
                epoch, offset = epoch + 1, 0
            elapsed, = ranks.stats([time.monotonic() - update_started], maximum=True)
            media = np.searchsorted(data.offsets["train"], selected, side="right") - 1
            metrics = {"completed_updates": update, "accepted_updates": update, "epochs_completed": epoch,
                       "records_completed_in_epoch": offset, "train_b5_objective": loss_sum,
                       "gradient_norm": grad_norm, "microbatch": microbatch,
                       "learning_rates": controller_step["learning_rates"], "time_unix": time.time(),
                       "optimizer_update_s": elapsed, "distributed_world_size": ranks.size,
                       "global_shots_in_update": len(selected),
                       "distinct_velocity_models_in_update": len(np.unique(media))}
            if model.device.type == "cuda":
                maxima = ranks.stats([torch.cuda.max_memory_allocated(model.device) / 2**30,
                                      torch.cuda.max_memory_reserved(model.device) / 2**30], maximum=True)
                metrics["cuda_peak_allocated_gib"], metrics["cuda_peak_reserved_gib"] = maxima
            if update == 1 or update % args.checkpoint_interval == 0 or epoch_finished:
                save()
            validation = validate_if_due() if epoch_finished else None
            if validation is not None:
                metrics["validation"] = validation
            def log_update():
                with (output / "metrics.jsonl").open("a") as history:
                    history.write(json.dumps(metrics, allow_nan=False) + "\n")
                print(json.dumps({"stage": "accepted_optimizer_update", **metrics}, allow_nan=False), flush=True)
                publish("TRAINING", train_b5_objective=loss_sum, gradient_norm=grad_norm,
                        optimizer_update_s=metrics["optimizer_update_s"],
                        cuda_peak_allocated_gib=metrics.get("cuda_peak_allocated_gib"))

            ranks.on_primary(log_update)
            if epoch_finished:
                break
    if control.completed_updates != last_checkpoint:
        save()
    finish = "COMPLETE" if epoch == args.epochs and last_validated_epoch == epoch else "PAUSED_AT_UPDATE_BOUNDARY"
    ranks.on_primary(lambda: publish(finish))


def run(args, ranks):
    device = ranks.device
    rank_seed = args.seed + 104729 * ranks.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed % 2**32)
    torch.manual_seed(rank_seed)
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    def initialize():
        if device.type == "cuda":
            if "A800" not in torch.cuda.get_device_name(device):
                raise ValueError("This training run expects a reserved A800 per rank")
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(.94, device)
        data = CommittedMarmousi(args.dataset, prefix_probe=args.prefix_probe,
                                 dense_prefix_probe=args.dense_prefix_probe)
        if len(data.rows) != args.expected_models:
            raise ValueError("The requested release model count differs")
        if not args.verify_only and args.effective_batch > len(data.train_ids):
            raise ValueError("Effective batch exceeds the number of distinct TRAIN velocity models")
        model = build_marmousi_ul(data, args.seed, device, args.query_block,
                                  interface_correction=args.interface_correction)
        readout = deepcopy(data.observe).to(device=device, dtype=torch.float32)
        return data, model, readout

    data, model, readout = ranks.on_each(initialize)
    ranks.share_initial_parameters(model)
    preflight = ranks.on_primary(lambda: verify_projection(data, model, readout))
    if args.verify_only:
        return
    with ExitStack() as stack:
        def lock_output():
            args.output.mkdir(parents=True, exist_ok=True)
            stack.enter_context(writer_lock(args.output / "writer.lock", True))

        ranks.on_primary(lock_output)
        try:
            training_loop(args, data, model, readout, preflight, ranks)
        except BaseException as error:
            if ranks.primary:
                path = args.output / "STATUS.json"
                previous = read_json(path) if path.exists() else {}
                atomic_json(path, {**previous, "state": "FAILED", "pid": os.getpid(),
                                   "time_unix": time.time(), "error": repr(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--ul-config", type=Path, required=True)
    parser.add_argument("--expected-models", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=261119)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--effective-batch", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--times-per-stratum", type=int, default=64)
    parser.add_argument("--query-block", type=int, default=64)
    parser.add_argument("--interface-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--objective", choices=("relative", "late_balanced"), default="late_balanced")
    parser.add_argument("--backbone-factor", type=float, default=.1)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--validation-every", type=int, default=2)
    parser.add_argument("--validation-records", type=int, default=6)
    parser.add_argument("--validation-block", type=int, default=256)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--prefix-probe", action="store_true")
    parser.add_argument("--dense-prefix-probe", action="store_true",
                        help="Verify only: read every fourth row of the archived 3201-frame prefix")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and (args.output is None or args.prefix_probe):
        parser.error("A training run needs --output and the fully validated release")
    if args.verify_only and args.resume:
        parser.error("--resume is for a training run")
    if args.dense_prefix_probe and not (args.verify_only and args.prefix_probe):
        parser.error("--dense-prefix-probe requires --prefix-probe and --verify-only")
    if (min(args.epochs, args.effective_batch, args.microbatch, args.times_per_stratum, args.query_block,
            args.checkpoint_interval, args.validation_every, args.validation_records, args.validation_block) < 1
            or args.times_per_stratum > 200):
        parser.error("Invalid epoch, batch or time counts")
    ranks = RankRuntime(args.device)
    try:
        run(args, ranks)
    finally:
        ranks.close()


if __name__ == "__main__":
    main()
