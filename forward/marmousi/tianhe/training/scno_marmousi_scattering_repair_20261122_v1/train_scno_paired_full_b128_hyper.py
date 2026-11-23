"""Train the background and signed-source operators using the committed Marmousi pairs."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
from time import monotonic
import json
import math
import os
import signal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pretrain_scno_mhc import atomic_checkpoint, pressure_frames, read, sha
from scno_direct_background_scattering import DirectBackgroundScattering
from scno_forward_source_propagator import SourcePropagator, smooth_velocity
from scno_learned_background import BackgroundNeuralOperator, ray_geometry
from tools.marmousi_target_dataset_refined.common import atomic_json, sha_array


SCHEMA = "marmousi_scno_paired_train_pressure_v1"
SEED = 593881
ORIGINAL_BATCH = 64
TIMES = 801
LOSS_COMPONENTS = ("pressure", "deep_pressure", "receiver", "late_receiver", "background", "scattering")
JOINT_LOSS_WEIGHTS = (.27, .15, .27, .14, .10, .07)


class PairedPressure(Dataset):
    def __init__(self, root, release, read_threads):
        status = read(root / "STATUS.json")
        index = root / "TRAIN_INDEX.json"
        plan = root / "PLAN.json"
        if (status["schema"] != SCHEMA or status["status"] != "COMPLETE"
                or status["exact_marmousi_labels"] != 0 or status["completed_pairs"] != 3200
                or status["completed_models"] != 800 or sha(index) != status["train_index_sha256"]
                or sha(plan) != status["plan_sha256"]):
            raise ValueError("A complete and verified ordinary TRAIN pressure-pair release is required")
        self.binding = {"pair_plan_sha256": status["plan_sha256"],
                        "pair_index_sha256": status["train_index_sha256"],
                        "velocity_release": release.binding}
        manifest = read(plan)
        if (manifest["background_sigma_m"] != 150. or manifest["frames"] != list(range(TIMES))
                or manifest["exact_target_waveforms_used"]
                or any(manifest["release"][name] != release.binding[name] for name in manifest["release"])):
            raise ValueError("Pressure pairs have a different velocity, background or time-axis contract")
        self.rows = []
        self.velocities = {}
        self.read_threads = read_threads
        self.read_pool = None
        self.rms = np.float32(release.rms)
        self.sources = release.sources.astype(np.float32)
        self.frames = pressure_frames(0)
        seen = Counter()
        for receipt in read(index):
            row = receipt["record"]
            model, shot = row["model"], row["shot"]
            if (model not in release.train_set or shot not in release.rosters[model]
                    or receipt["schema"] != SCHEMA or receipt["status"] != "COMMITTED"
                    or receipt["plan_sha256"] != status["plan_sha256"]
                    or row["velocity_sha"] != release.rows[model]["velocity_sha256"]
                    or row["parent"] != release.rows[model]["parent_group_id"]
                    or not np.array_equal(receipt["source"], release.sources[shot])):
                raise ValueError(f"Pressure pair leaves the fixed TRAIN roster: {model}, {shot}")
            folder = root / "train" / model / f"shot_{shot:02d}"
            if folder.relative_to(root).as_posix() != receipt["relative_directory"]:
                raise ValueError(f"Invalid pressure-pair folder: {folder}")
            if model not in self.velocities:
                velocity = np.load(release.root / release.rows[model]["velocity_path"], allow_pickle=False)
                if (velocity.shape != (201, 201) or velocity.dtype != np.float32
                        or sha_array(velocity) != row["velocity_sha"]):
                    raise ValueError(f"Velocity no longer matches pair labels: {model}")
                self.velocities[model] = velocity
            seen[model, shot] += 1
            self.rows.append((model, shot, folder))
        expected = {(model, shot) for model in release.train_ids for shot in release.rosters[model]}
        if set(seen) != expected or len(self.rows) != 3200 or max(seen.values()) != 1:
            raise ValueError("A TRAIN model/shot pair is missing, duplicated, or outside the committed roster")

    def __len__(self):
        return len(self.rows)

    def __getitems__(self, indices):
        if self.read_threads == 1:
            return [self[index] for index in indices]
        if self.read_pool is None:
            self.read_pool = ThreadPoolExecutor(max_workers=self.read_threads)
        return list(self.read_pool.map(self.__getitem__, indices))

    @staticmethod
    def sampled_pressure(path, shape, frames):
        # Small frame reads avoid filesystem readahead across the intervening samples.
        with path.open("rb", buffering=0) as file:
            version = np.lib.format.read_magic(file)
            read_header = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}.get(version)
            if read_header is None:
                raise ValueError(f"Unsupported NumPy array header: {path}, {version}")
            actual_shape, fortran_order, dtype = read_header(file)
            if actual_shape != shape or fortran_order or dtype != np.float32:
                raise ValueError(f"Invalid pressure field: {path}")
            offset = file.tell()
            os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
            values = np.empty((len(frames), *shape[1:]), dtype=dtype)
            frame_bytes = values[0].nbytes
            for row, frame in zip(values, frames, strict=True):
                read = os.preadv(file.fileno(), [row], offset + int(frame) * frame_bytes)
                if read != frame_bytes:
                    raise OSError(f"Incomplete pressure frame {frame}: {path}")
        return values

    def __getitem__(self, index):
        model, shot, folder = self.rows[index]
        def target(name, shape, frames=None):
            path = folder / f"{name}.npy"
            if frames is not None:
                return self.sampled_pressure(path, shape, frames) / self.rms
            array = np.load(path, allow_pickle=False)
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError(f"Invalid field contents for {model}, shot {shot}: {name}")
            return array / self.rms
        return (self.velocities[model], self.sources[shot],
                target("pressure", (TIMES, 201, 201), self.frames),
                target("background_pressure", (TIMES, 201, 201), self.frames),
                target("b5", (TIMES, 201)), target("background_b5", (TIMES, 201)))


def forward(model, velocity, source, slots, *, background_only=False):
    if background_only:
        coarse = model.background(*ray_geometry(smooth_velocity(velocity, 150.), source))
        background = model.geometry.native_time(coarse)
        p0, b0 = model.geometry(background[slots]), model.background_b5(background)
        return p0, b0, p0, b0
    background, effective = model.prepare(velocity, source)
    p0, b0 = model.geometry(background[slots]), model.background_b5(background)
    fields = model.propagate((background, effective))
    return model.pressure(fields, slots), model.b5(fields), p0, b0


def relative_squared(prediction, reference, *, energy_floor=None):
    axes = tuple(range(1, reference.ndim))
    denominator = reference.square().sum(axes)
    if energy_floor is not None:
        denominator = torch.maximum(denominator, energy_floor)
    numerator = (prediction - reference).square().sum(axes)
    return (numerator / denominator.clamp_min(1.e-8)).mean()


def training_loss(model, batch, slots, background_only, weights=JOINT_LOSS_WEIGHTS):
    velocity, source, pressure, background, receiver, background_receiver = batch
    p, b, p0, b0 = torch.vmap(
        lambda v, s: forward(model, v, s, slots, background_only=background_only))(velocity, source)
    bg = .65 * relative_squared(p0, background) + .35 * relative_squared(b0, background_receiver)
    if background_only:
        return bg
    deep = relative_squared(p[:, :, 50:], pressure[:, :, 50:],
                            energy_floor=.01 * pressure.square().sum((1, 2, 3)))
    late = relative_squared(b[:, 100:400], receiver[:, 100:400],
                            energy_floor=.02 * receiver.square().sum((1, 2)))
    scatter = relative_squared(p - p0, pressure - background,
                               energy_floor=.05 * pressure.square().sum((1, 2, 3)))
    components = (relative_squared(p, pressure), deep, relative_squared(b, receiver), late, bg, scatter)
    return sum(weight * component for weight, component in zip(weights, components, strict=True))


@torch.inference_mode()
def evaluate(model, archive, sources, *, detailed=False, split="validation"):
    totals = torch.zeros((4 if detailed else 2, 2), dtype=torch.float64, device=sources.device)
    slots = torch.arange(TIMES, device=sources.device)
    for index, (meta, _, _) in enumerate(archive.items[split]):
        pressure, receiver, velocity = archive.arrays(split, index)
        v = torch.as_tensor(np.array(velocity), device=sources.device)
        fields = model.propagate(model.prepare(v, sources[meta["shot_id"]]))
        ref_b = torch.as_tensor(np.array(receiver), device=sources.device) / archive.release.rms
        pred_b = model.b5(fields)
        totals[1] += torch.stack(((pred_b - ref_b).double().square().sum(), ref_b.double().square().sum()))
        if detailed:
            estimates, reference = pred_b[100:400], ref_b[100:400]
            totals[3] += torch.stack(((estimates - reference).double().square().sum(),
                                     reference.double().square().sum()))
        for block in slots.split(32):
            ref = torch.as_tensor(pressure[block.cpu().numpy()].copy(),
                                  device=sources.device) / archive.release.rms
            est = model.pressure(fields, block)
            totals[0] += torch.stack(((est - ref).double().square().sum(), ref.double().square().sum()))
            if detailed:
                estimates, reference = est[:, 50:], ref[:, 50:]
                totals[2] += torch.stack(((estimates - reference).double().square().sum(),
                                         reference.double().square().sum()))
    pressure_l2, receiver_l2, *subsets = (totals[:, 0] / totals[:, 1]).sqrt().tolist()
    result = {"full_pressure_l2": pressure_l2, "b5_l2": receiver_l2,
              f"ordinary_{split}_shots": len(archive.items[split]), "frames": TIMES}
    if detailed:
        result.update(deep_pressure_l2=subsets[0], late_b5_l2=subsets[1])
    return result


def publish(output, **fields):
    atomic_json(output / "STATUS.json", fields)
    print(json.dumps(fields, allow_nan=False), flush=True)


def select_microbatch(args, model, data, device):
    supported = (128, 120, 112, 104, 96, 88, 80, 72, 64, 48, 32, 24, 16, 8, 4, 2, 1)
    candidates = (args.microbatch,) if args.microbatch else tuple(dict.fromkeys(
        (args.effective_batch, *(size for size in supported if size <= args.effective_batch))))
    slots = torch.as_tensor(data.frames, device=device)
    indices = np.random.default_rng(SEED).choice(len(data), candidates[0], replace=False)
    cached = [data[i] for i in indices]

    def measure(size):
        batch = [torch.from_numpy(np.stack(values)).to(device)
                 for values in zip(*cached[:size], strict=True)]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = monotonic()
        for _ in range(2):
            check = training_loss(model, batch, slots, background_only=False, weights=args.loss_weights)
            check.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if not torch.isfinite(check):
            raise ValueError("Joint forward/loss preflight is nonfinite")
        return {"microbatch": size, "loss": float(check.detach()), "gradient_norm": float(gradient),
                "seconds_per_microbatch": (monotonic() - start) / 2.,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20}

    failures = []
    for size in candidates:
        try:
            result = measure(size)
        except torch.cuda.OutOfMemoryError:
            failures.append(size)
        else:
            result.update(oom_microbatches=failures, effective_batch=args.effective_batch)
            atomic_json(args.output / "MICROBATCH_PROBE.json", result)
            torch.cuda.empty_cache()
            return result
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"status": "MICROBATCH_OOM", "microbatch": size}), flush=True)
    raise RuntimeError("The registered joint training step did not fit at microbatch 1")


def backward_chunks(model, sample, slots, background_only, logical_size, microbatch,
                    accumulated, device, loss_weights=JOINT_LOSS_WEIGHTS, *, objective=training_loss):
    offset = 0
    total_loss = torch.zeros((), device=device)
    parameters = tuple(model.parameters())
    while offset < len(sample[0]):
        count = min(microbatch, len(sample[0]) - offset)
        # A failed backward may have written some gradients. Restore prior chunks before retrying.
        gradients = [p.grad.detach().clone() if (accumulated or offset) and p.grad is not None else None
                     for p in parameters]
        loss = None
        values = None
        try:
            values = [item[offset:offset + count].to(device, non_blocking=True) for item in sample]
            loss = objective(model, values, slots, background_only, loss_weights) * (count / logical_size)
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            if count == 1:
                raise
        else:
            total_loss += loss.detach()
            offset += count
            continue

        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = gradient
        del loss, values
        gc.collect()
        torch.cuda.empty_cache()
        microbatch = max(1, count - max(1, count // 8))
        print(json.dumps({"status": "RUNTIME_MICROBATCH_FALLBACK", "failed_microbatch": count,
                          "microbatch": microbatch, "effective_batch": logical_size}), flush=True)
    return total_loss, microbatch


def configure_optimizer(args, optimizer, previous):
    previous_policy = previous.get("optimizer_policy_sha256")
    if args.optimizer_policy is None:
        if previous_policy is not None:
            raise ValueError("The checkpoint needs its validation-based optimizer policy")
        return 1., None, None, [], 0

    identity = sha(args.optimizer_policy)
    config = read(args.optimizer_policy)
    plateau = config["plateau"]
    scale, clip = config["lr_scale"], config["grad_clip"]
    window = plateau["validation_window"]
    if (not 0 < scale <= 2. or not 0 < clip <= 10. or not 0 < plateau["factor"] < 1.
            or not 0 < plateau["min_lr_ratio"] < 1. or plateau["threshold"] < 0.
            or not isinstance(window, int) or window < 1 or plateau["patience"] < 1):
        raise ValueError("Invalid learning-rate or gradient-clip policy")
    if previous_policy is not None and previous_policy != identity:
        raise ValueError("The checkpoint policy differs from the declared policy")

    initial = [args.lr * .5 * scale, args.lr * scale]
    if previous_policy is None:
        for group, lr in zip(optimizer.param_groups, initial, strict=True):
            group["lr"] = lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=plateau["factor"], patience=plateau["patience"],
        threshold=plateau["threshold"], threshold_mode="abs",
        min_lr=[rate * plateau["min_lr_ratio"] for rate in initial])
    if previous_policy is not None:
        scheduler.load_state_dict(previous["optimizer_scheduler"])
    return clip, scheduler, identity, list(previous.get("recent_validation_scores", [])), window


def train(args, data, validation, model, identity):
    device = torch.device("cuda:0")
    optimizer = torch.optim.AdamW([
        {"params": model.background.parameters(), "lr": args.lr * .5},
        {"params": model.scatter.parameters(), "lr": args.lr}], weight_decay=.01)
    path = args.output / "LAST.pt"
    step = 0
    visited = 0
    best = math.inf
    previous = {}
    if path.exists():
        previous = torch.load(path, map_location=device, weights_only=True)
        if previous["plan_sha256"] != identity:
            raise ValueError("Checkpoint belongs to a different full paired training contract")
        if previous.get("loss_policy_sha256") not in (None, args.loss_policy_sha256):
            raise ValueError("Resume requires the checkpoint's registered loss policy")
        model.load_state_dict(previous["model"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        step, best = previous["step"], previous["best_score"]
        visited = previous.get("model_shot_pairs_seen", step * ORIGINAL_BATCH)
        if not 0 <= visited <= args.epochs * len(data):
            raise ValueError("Checkpoint cursor lies outside the TRAIN roster")
        print(json.dumps({"status": "RESUMED", "optimizer_step": step,
                          "model_shot_pairs_seen": visited}), flush=True)
    grad_clip, scheduler, policy, recent_scores, score_window = configure_optimizer(args, optimizer, previous)
    if scheduler is not None:
        print(json.dumps({"status": "OPTIMIZER_POLICY_ACTIVE", "optimizer_step": step,
                          "optimizer_policy_sha256": policy, "grad_clip": grad_clip,
                          "learning_rates": [group["lr"] for group in optimizer.param_groups]}), flush=True)
    source = torch.as_tensor(data.sources, device=device)
    run_start = monotonic()
    stop_requested = False
    active_microbatch = args.microbatch

    def finish_current_update(*_):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, finish_current_update)

    def save(file):
        atomic_checkpoint(file, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                     step=step, model_shot_pairs_seen=visited,
                                     effective_batch=args.effective_batch, microbatch=active_microbatch,
                                     optimizer_policy_sha256=policy,
                                     optimizer_scheduler=scheduler.state_dict() if scheduler else None,
                                     recent_validation_scores=recent_scores,
                                     loss_policy_sha256=args.loss_policy_sha256,
                                     best_score=best, plan_sha256=identity))

    for epoch in range(visited // len(data), args.epochs):
        start = monotonic()
        offset = visited % len(data)
        data.frames = pressure_frames(epoch)
        slots = torch.as_tensor(data.frames, device=device)
        order = np.random.default_rng([SEED, epoch]).permutation(len(data))[offset:].tolist()
        groups = [order[i:i + args.effective_batch] for i in range(0, len(order), args.effective_batch)]
        batches = [group[i:i + args.microbatch]
                   for group in groups for i in range(0, len(group), args.microbatch)]
        group_sizes = (len(group) for group in groups for _ in range(math.ceil(len(group) / args.microbatch)))
        loader = DataLoader(data, batch_sampler=batches, num_workers=args.workers,
                            pin_memory=True, prefetch_factor=2 if args.workers else None)
        model.train()
        background_only = epoch < args.background_epochs
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        last_log_step = step
        recent = []
        for sample, group_size in zip(loader, group_sizes, strict=True):
            count = len(sample[0])
            loss, active_microbatch = backward_chunks(model, sample, slots, background_only, group_size,
                                                     active_microbatch, accumulated, device, args.loss_weights)
            accumulated += count
            recent.append(loss)
            if accumulated != group_size:
                continue
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            visited += accumulated
            accumulated = 0
            step += 1
            at_epoch_end = visited % len(data) == 0
            if stop_requested and not at_epoch_end:
                save(path)
                publish(args.output, status="PAUSED", epoch=epoch + 1, optimizer_step=step,
                        effective_batch=args.effective_batch, model_shot_pairs_seen=visited, reason="SIGUSR1")
                return
            if step == 1 or step % 5 == 0 or at_epoch_end:
                updates = step - last_log_step
                average = float(torch.stack(recent).sum() / updates)
                last_log_step = step
                recent.clear()
                if step == 1 or (step % 10 == 0 and not at_epoch_end):
                    save(path)
                publish(args.output, status="TRAINING", stage="background" if background_only else "joint",
                        epoch=epoch + 1, epochs=args.epochs, optimizer_step=step, model_shot_pairs_seen=visited,
                        pairs_per_epoch=len(data), loss=average, gradient_norm=float(gradient),
                        grad_clip=grad_clip, learning_rates=[group["lr"] for group in optimizer.param_groups],
                        loss_policy_sha256=args.loss_policy_sha256,
                        microbatch=active_microbatch, effective_batch=args.effective_batch,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() // 2**20, elapsed_s=monotonic() - run_start)
        if accumulated or visited != (epoch + 1) * len(data):
            raise ValueError("The epoch did not visit the full shuffled TRAIN roster")
        model.eval()
        receipt = evaluate(model, validation, source, detailed=args.loss_policy is not None)
        if not all(math.isfinite(receipt[key]) for key in ("full_pressure_l2", "b5_l2")):
            raise ValueError("Nonfinite ordinary-validation score")
        score = .5 * (receipt["full_pressure_l2"] + receipt["b5_l2"])
        if scheduler is not None and not background_only:
            recent_scores.append(score)
            del recent_scores[:-score_window]
            if len(recent_scores) == score_window:
                rates_before = [group["lr"] for group in optimizer.param_groups]
                scheduler.step(sum(recent_scores) / score_window)
                rates_after = [group["lr"] for group in optimizer.param_groups]
                if rates_after != rates_before:
                    print(json.dumps({"status": "LEARNING_RATE_REDUCED", "epoch": epoch + 1,
                                      "before": rates_before, "after": rates_after}), flush=True)
        if not background_only and score < best:
            best = score
            save(args.output / "BEST.pt")
        epoch_report = {"epoch": epoch + 1, "optimizer_step": step, "epoch_s": monotonic() - start,
                        "grad_clip": grad_clip, "learning_rates": [group["lr"] for group in optimizer.param_groups],
                        **receipt}
        atomic_json(args.output / f"VAL_{epoch + 1:03d}.json", epoch_report)
        save(path)
        print(json.dumps(epoch_report, allow_nan=False), flush=True)
        if stop_requested and epoch + 1 < args.epochs:
            publish(args.output, status="PAUSED", epoch=epoch + 1, optimizer_step=step,
                    effective_batch=args.effective_batch, model_shot_pairs_seen=visited, reason="SIGUSR1")
            return
    publish(args.output, status="COMPLETE", epochs=args.epochs, optimizer_step=step,
            pairs_per_epoch=len(data), model_shot_pairs_seen=visited,
            ordinary_validation=read(args.output / f"VAL_{args.epochs:03d}.json"))


def run(args):
    from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
    from make_direction_labels import split_lock
    from tianhe_observations import CommittedMarmousi
    from train_joint_pressure import PressureArchive
    from train_no_pde_native import forbid_rollouts

    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(.92, device)
    args.loss_weights = JOINT_LOSS_WEIGHTS
    args.loss_policy_sha256 = None
    if args.loss_policy is not None:
        values = read(args.loss_policy)["weights"]
        if set(values) != set(LOSS_COMPONENTS):
            raise ValueError("The loss policy must specify all pressure, receiver, and scattering weights")
        args.loss_weights = tuple(values[name] for name in LOSS_COMPONENTS)
        if (any(not math.isfinite(weight) or weight <= 0. for weight in args.loss_weights)
                or not math.isclose(sum(args.loss_weights), 1.)):
            raise ValueError("Loss weights must be positive and sum to one")
        args.loss_policy_sha256 = sha(args.loss_policy)
    args.output.mkdir(parents=True, exist_ok=True)
    with split_lock(args.output / ".writer.lock"), forbid_rollouts():
        guard = forbid_exact_label_files(args.dataset)
        release = CommittedMarmousi(args.dataset)
        data = PairedPressure(args.pairs, release, args.read_threads)
        validation = PressureArchive(args.validation, release)
        if len(validation.items["validation"]) != 12:
            raise ValueError("The registered ordinary validation panel must contain 12 shots")
        basis_file = args.background / "TRAIN_BASIS.pt"
        background_file = args.background / "BACKGROUND_LAST.pt"
        scatter_file = args.scatter / "SCATTER_LAST.pt"
        basis = torch.load(basis_file, weights_only=True, map_location="cpu")
        background_state = torch.load(background_file, weights_only=True, map_location="cpu")
        scatter_state = torch.load(scatter_file, weights_only=True, map_location="cpu")
        if (basis["plan"] != sha(args.background / "PLAN.json") or background_state["plan"] != basis["plan"]
                or scatter_state["plan"] != sha(args.scatter / "PLAN.json")):
            raise ValueError("Warm-start weight and short-run receipts disagree")
        bg = BackgroundNeuralOperator(basis["basis"])
        bg.load_state_dict(background_state["model"], strict=True)
        scatter = SourcePropagator()
        scatter.load_state_dict(scatter_state["model"], strict=True)
        model = DirectBackgroundScattering(bg, scatter, release.observe).to(device)
        design = {**data.binding, "validation": validation.binding, "epochs": args.epochs,
                  "background_epochs": args.background_epochs, "effective_batch": args.effective_batch,
                  "lr": args.lr, "seed": SEED, "initial_weights": [sha(x) for x in (basis_file, background_file, scatter_file)],
                  "training_script_sha256": sha(Path(__file__)), "exact_target_label_reads": 0,
                  "exact_target_guard": guard, "online_numerical_propagation": False}
        plan = args.output / "PLAN.json"
        # The read counter in the guard can grow as subsequent contract readers check the velocity template.
        if plan.exists():
            previous = read(plan)
            revision = None
            if args.resume_reader_from is not None:
                original = sha(args.resume_reader_from)
                if previous["training_script_sha256"] != original:
                    raise ValueError("The declared original reader differs from the training plan")
                if previous["effective_batch"] != ORIGINAL_BATCH:
                    raise ValueError("Expected the original effective batch of 64 in the frozen plan")
                revision = {
                    "original_script_sha256": original,
                    "active_script_sha256": design["training_script_sha256"],
                    "training_plan_sha256": sha(plan),
                    "original_effective_batch": ORIGINAL_BATCH,
                    "active_effective_batch": args.effective_batch,
                    "pressure_io": "posix_fadvise_random_preadv",
                    "loader_workers": args.workers, "read_threads_per_worker": args.read_threads}
                if args.optimizer_policy is not None:
                    revision["optimizer_policy_sha256"] = sha(args.optimizer_policy)
                if args.loss_policy_sha256 is not None:
                    revision["loss_policy_sha256"] = args.loss_policy_sha256
                    revision["loss_weights"] = dict(zip(LOSS_COMPONENTS, args.loss_weights, strict=True))
                design["training_script_sha256"] = original
                design["effective_batch"] = ORIGINAL_BATCH
            if {k: v for k, v in previous.items() if k != "exact_target_guard"} != {
                    k: v for k, v in design.items() if k != "exact_target_guard"}:
                raise ValueError("Resume requires the same immutable data and optimization settings")
            if revision is not None:
                atomic_json(args.output / "READER_REVISION.json", revision)
        else:
            if args.resume_reader_from is not None:
                raise ValueError("Resuming with a new reader requires the original training plan")
            atomic_json(plan, design)
        publish(args.output, status="PROBING_MICROBATCH", pairs=len(data), validation_shots=12,
                gpu=torch.cuda.get_device_name(device), exact_marmousi_labels=0)
        probe = select_microbatch(args, model, data, device)
        args.microbatch = probe["microbatch"]
        publish(args.output, status="READY", pairs=len(data), validation_shots=12,
                probe=probe, microbatch=args.microbatch, effective_batch=args.effective_batch,
                gpu=torch.cuda.get_device_name(device), exact_marmousi_labels=0)
        train(args, data, validation, model, sha(plan))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("dataset", "pairs", "validation", "background", "scatter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--background-epochs", type=int, default=1)
    parser.add_argument("--effective-batch", type=int, default=ORIGINAL_BATCH)
    parser.add_argument("--microbatch", type=int, default=0, help="0 probes up to the effective batch")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--read-threads", type=int, default=1)
    parser.add_argument("--lr", type=float, default=.00055)
    parser.add_argument("--resume-reader-from", type=Path)
    parser.add_argument("--optimizer-policy", type=Path)
    parser.add_argument("--loss-policy", type=Path)
    config = parser.parse_args()
    if (not 1 <= config.effective_batch <= 128 or not 0 <= config.microbatch <= config.effective_batch
            or config.workers < 0
            or config.read_threads < 1
            or not 0 <= config.background_epochs < config.epochs or config.lr <= 0):
        parser.error("Specify a supported microbatch and a positive joint training budget")
    run(config)
