"""Staged SCNO V2 training with four global records and sixteen pressure frames per update."""

import argparse
from contextlib import contextmanager, nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch import distributed as dist
from torch import nn

from prefetch_v2 import PressureBatchPrefetch
from trainer_metrics import SELECTION_METRIC, combine_errors, evaluate, orthogonal_errors


GLOBAL_RECORDS = 4
FRAMES_PER_RECORD = 4
SUPERVISED_FRAMES = GLOBAL_RECORDS * FRAMES_PER_RECORD
CHECKPOINT_SCHEMA = "scno_training_v2"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_json(path, value):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, allow_nan=False) + "\n")


def save_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("wb") as output:
        torch.save(value, output)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def phase_ranges(config):
    total = config["total_updates"]
    if type(total) is not int or total < 1:
        raise ValueError("total_updates must be a positive integer")
    if config.get("global_records_per_update", 4) != 4 or config.get("frames_per_record", 4) != 4:
        raise ValueError("V2 requires four global records and four queries per record")
    protocol = config.get("protocol", "staged")
    if protocol == "joint_from_start":
        return [("C", 0, total)]
    if protocol != "staged":
        raise ValueError("protocol must be staged or joint_from_start")
    if "phase_updates" in config:
        counts = config["phase_updates"]
    else:
        fractions = config.get("phase_fractions", [.6, .2, .2])
        if len(fractions) != 3 or min(fractions) <= 0 or not math.isclose(sum(fractions), 1.):
            raise ValueError("three positive phase fractions summing to one are required")
        first, second = round(total * fractions[0]), round(total * (fractions[0] + fractions[1]))
        counts = (first, second - first, total - second)
    if len(counts) != 3 or any(type(n) is not int or n < 1 for n in counts) or sum(counts) != total:
        raise ValueError("the three accepted-update budgets must be positive integers adding to total_updates")
    a, b, _ = counts
    return [("A", 0, a), ("B", a, a + b), ("C", a + b, total)]


def cosine_ratio(step, length, warmup, final_ratio):
    position = min(step, length - 1)
    if position < warmup:
        return (position + 1) / warmup
    peak_position = max(0, warmup - 1)
    decay_length = length - 1 - peak_position
    progress = (position - peak_position) / decay_length if decay_length else 0.
    return final_ratio + (1. - final_ratio) * .5 * (1. + math.cos(math.pi * progress))


def build_optimizer(model, phase, length, config):
    """Build the exact active role partition and an accepted-update LR schedule."""
    defaults = {
        "A": {"peak_lr": 3e-4, "warmup_updates": 500, "final_lr": 3e-5},
        "B": {"peak_lr": 1e-3, "warmup_updates": 200, "final_lr": 1e-4},
        "C": {"base_lr": 3e-5, "detail_lr": 1e-4, "decay_factor": .1, "warmup_updates": 0},
    }
    options = {**defaults[phase], **config.get("phases", {}).get(phase, {})}
    weight_decay = float(options.get("weight_decay", 1e-4))
    requested_warmup = options["warmup_updates"]
    if type(requested_warmup) is not int or requested_warmup < 0 or weight_decay < 0 or not math.isfinite(weight_decay):
        raise ValueError("warm-up and weight decay must be finite and nonnegative")
    # Keep at least one decay step in shortened pilots.
    warmup = min(requested_warmup, max(0, length - 1))
    active_roles = {"A": ("BASE",), "B": ("DETAIL",), "C": ("BASE", "DETAIL")}[phase]
    names = model.named_roles()
    norm_types = (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm,
                  nn.GroupNorm, nn.LayerNorm, nn.LocalResponseNorm)
    if hasattr(nn, "RMSNorm"):
        norm_types += (nn.RMSNorm,)
    norm_ids = {id(p) for module in model.modules() if isinstance(module, norm_types)
                for p in module.parameters(recurse=False)}
    no_decay = set(getattr(model, "no_decay_names", ()))
    groups, schedules = [], []
    for role in active_roles:
        peak = float(options["peak_lr"] if phase in ("A", "B") else options[role.lower() + "_lr"])
        final = float(options["final_lr"] if phase in ("A", "B") else peak * options["decay_factor"])
        if not math.isfinite(peak) or not math.isfinite(final) or not 0 < final <= peak:
            raise ValueError("each LR schedule must remain finite and positive")
        buckets = {"decay": [], "no_decay": []}
        for name, parameter in names[role]:
            if not parameter.requires_grad:
                raise ValueError(f"{phase} disabled a parameter in the {role} role: {name}")
            exempt = (name.endswith(".bias") or name == "bias" or id(parameter) in norm_ids or name in no_decay)
            buckets["no_decay" if exempt else "decay"].append(parameter)
        for suffix, parameters in buckets.items():
            if parameters:
                groups.append({"params": parameters, "lr": peak, "role": role, "group_name": f"{role}_{suffix}",
                               "weight_decay": 0. if suffix == "no_decay" else weight_decay})
                schedules.append(lambda index, fraction=final / peak: cosine_ratio(index, length, warmup, fraction))
    parameters = [p for group in groups for p in group["params"]]
    active = {id(p) for p in model.parameters() if p.requires_grad}
    if len(parameters) != len(active) or {id(p) for p in parameters} != active or not active:
        raise ValueError("optimizer parameter identities must exactly match the active model partition")
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedules)
    return optimizer, scheduler, parameters, warmup


def _base_state_keys(model):
    if hasattr(model, "base_state_keys"):
        return set(model.base_state_keys())
    detail_names = {name for name, _ in model.named_roles()["DETAIL"]}
    return {key for key in model.state_dict() if key not in detail_names and ".spectral_detail." not in f".{key}."}


def _base_digest(model):
    state = model.state_dict()
    values = {key: state[key] for key in _base_state_keys(model)}
    values.update({name: value for name, value in model.named_buffers()
                   if ".spectral_detail." not in f".{name}."})
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def snapshot_output_affines(model):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        names[id(parameter)]: (index, parameter, parameter.detach().clone())
        for index, layer in enumerate(model.detail_output_layers)
        for parameter in layer.parameters(recurse=False)
    }


def changed_output_affines(snapshot):
    return [name for name, (_, parameter, previous) in snapshot.items()
            if not torch.equal(parameter.detach(), previous)]


@torch.no_grad()
def clear_affine_optimizer_state(optimizer, model, names, *, synchronize=False):
    parameters = dict(model.named_parameters())
    for name in names:
        parameter = parameters[name]
        if synchronize:
            dist.broadcast(parameter, src=0)
        optimizer.state.pop(parameter, None)


def _module_group(role, name):
    return "REFINER" if role == "BASE" and ".refiner." in f".{name}." else role


def _l2_sums(model, snapshots=None):
    sums = dict(BASE=0., REFINER=0., DETAIL=0.)
    for role, named in model.named_roles().items():
        for name, parameter in named:
            if not parameter.requires_grad:
                continue
            if snapshots is None and parameter.grad is None:
                continue
            if snapshots is not None and name not in snapshots:
                continue
            value = parameter.grad.detach() if snapshots is None else parameter.detach() - snapshots[name]
            sums[_module_group(role, name)] += value.abs().double().square().sum().item()
    return {role: math.sqrt(value) for role, value in sums.items()}


def stratified_relative_coefficient_loss(transform, parts, target, strata_sizes, reference_energy, epsilon_E):
    """The pressure objective in fixed orthogonal coordinates for a frozen base."""
    base = parts["base_pressure"]
    if (base.shape != target.shape or target.ndim != 3 or target.shape[0] != FRAMES_PER_RECORD
            or len(strata_sizes) != FRAMES_PER_RECORD or any(size <= 0 for size in strata_sizes)):
        raise ValueError("expected four matched pressure frames and positive stratum sizes")
    if not (math.isfinite(reference_energy) and reference_energy >= 0
            and math.isfinite(epsilon_E) and epsilon_E > 0):
        raise ValueError("the reference energy and its positive floor must be finite")
    with torch.no_grad():
        residual = target.detach().to(base.device, torch.float64) - base.detach().double()
        coarse, packets = transform.analyze(residual[:, None, 1:])
        per_time = coarse.double().square().flatten(1).sum(1) + residual[:, 0].square().sum(1)
    predicted = parts["detail_coefficients"]
    if len(predicted) != len(packets) or len(predicted) != len(transform.levels):
        raise ValueError("expected a coefficient packet for every orthogonal level")
    for level, modes, detail in zip(transform.levels, packets, predicted):
        wanted = modes[..., 16:]
        if detail.shape != wanted.shape or level.valid_detail.shape != wanted.shape[2:]:
            raise ValueError("coefficient packet disagrees with the transform's valid mask")
        residual_detail = (detail.double() - wanted.double()).masked_fill(~level.valid_detail, 0.)
        per_time = per_time + residual_detail.square().flatten(1).sum(1)
    weight = torch.as_tensor(strata_sizes, dtype=torch.float64, device=base.device).detach()
    objective = (per_time * weight).sum() / max(reference_energy, epsilon_E)
    # The zero-amplitude model uses its zero-pressure output as a zero-gradient anchor.
    if not objective.requires_grad and base.requires_grad:
        objective = objective + base.reshape(-1)[0] * 0.
    return objective


def step_microbatches(model, optimizer, scheduler, parameters, dataset, draws, strata_sizes,
                      energy, device, rank=0, world_size=1, diagnostics=False, gradient_clip=1.,
                      phase_b_coefficient_loss=False, local_batch=None):
    """Accumulate globally scaled gradients and update once after all four records."""
    from data_v2 import stratified_relative_pressure_loss

    if len(draws) != GLOBAL_RECORDS or len({key for key, _ in draws}) != GLOBAL_RECORDS:
        raise ValueError("every update must draw four distinct records")
    coefficients_only = phase_b_coefficient_loss and not diagnostics
    if coefficients_only and (model.phase != "B" or any(p.requires_grad for _, p in model.named_roles()["BASE"])):
        raise ValueError("coefficient-only training requires phase B with every BASE parameter frozen")
    local = draws[rank::world_size]
    if local_batch is not None and len(local_batch.samples) != len(local):
        raise ValueError("prepared pressure batch does not match the rank's record count")
    terms = {"data_load_s": 0., "data_wait_s": 0., "forward_s": 0., "backward_s": 0., "optimizer_step_s": 0.}
    if local_batch is not None:
        terms.update(data_load_s=local_batch.read_s, data_wait_s=local_batch.wait_s)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    began = perf_counter()
    optimizer.zero_grad(set_to_none=True)
    parameters_before = {name: p.detach().clone() for role in model.named_roles().values()
                         for name, p in role if p.requires_grad} if diagnostics else None
    errors, record_losses = [], []
    for index, (record_id, slots) in enumerate(local):
        if len(slots) != FRAMES_PER_RECORD:
            raise ValueError("each record must supply four stratum queries")
        if local_batch is None:
            start = perf_counter()
            batch = dataset.read(record_id, slots)
            read_s = perf_counter() - start
            terms["data_load_s"] += read_s
            terms["data_wait_s"] += read_s
        else:
            batch = local_batch.samples[index]
        with model.scopes() if hasattr(model, "scopes") else nullcontext():
            start = perf_counter()
            target = batch["target"].to(device)
            inputs = (batch["velocity"], batch["source"], batch["times"])
            denominator = energy["energies"][record_id], energy["epsilon_E"]
            if coefficients_only:
                result = model(*inputs, return_parts=True, detail_synthesis=False)
                loss_r = stratified_relative_coefficient_loss(model.detail_transform, result, target,
                                                               strata_sizes, *denominator)
            else:
                result = model(*inputs, return_parts=diagnostics)
                prediction = result["prediction"] if diagnostics else result
                loss_r = stratified_relative_pressure_loss(prediction, target, strata_sizes, *denominator)
            sync()
            terms["forward_s"] += perf_counter() - start
            if not torch.isfinite(loss_r).item():
                raise FloatingPointError("the sampled pressure loss is nonfinite")
            record_losses.append(float(loss_r.detach()))
            start = perf_counter()
            (loss_r / GLOBAL_RECORDS).backward()
            sync()
            terms["backward_s"] += perf_counter() - start
        if diagnostics:
            errors.append(orthogonal_errors(model, target, result))
    start = perf_counter()
    if world_size > 1:
        reduce_gradients(parameters, device)
    grad_l2 = _l2_sums(model) if diagnostics else None
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
    learning_rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    optimizer.step()
    scheduler.step()
    if world_size > 1:
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    sync()
    terms["optimizer_step_s"] += perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    elapsed = perf_counter() - began
    if local_batch is not None:
        elapsed += local_batch.wait_s
    param_change = _l2_sums(model, parameters_before) if diagnostics else None
    if world_size > 1:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, (record_losses, errors, terms, elapsed, peak))
        record_losses = [value for rank_values, *_ in gathered for value in rank_values]
        errors = [value for _, rank_values, *_ in gathered for value in rank_values]
        terms = {key: max(item[2][key] for item in gathered) for key in terms}
        elapsed = max(item[3] for item in gathered)
        peak = max(item[4] for item in gathered)
    if len(record_losses) != GLOBAL_RECORDS:
        raise RuntimeError("distributed record aggregation omitted an exposure")
    row = {"data_loss": sum(record_losses) / GLOBAL_RECORDS, "global_gradient_norm": float(gradient_norm),
           "learning_rates": learning_rates, "timing_s": {**terms, "update_s": elapsed},
           "peak_allocated_bytes": peak, "amp_skipped_updates": 0,
           "training_batch_read_in_background": local_batch is not None and local_batch.background,
           "loss_representation": "coefficients" if coefficients_only else "field"}
    if diagnostics:
        field = combine_errors(errors)
        row.update(gradient_l2=grad_l2, parameter_increment_l2=param_change, orthogonal_errors=field,
                   base_output_rms=math.sqrt(field["base_output_sse"] / field["output_elements"]),
                   detail_output_rms=math.sqrt(field["detail_output_sse"] / field["output_elements"]))
    return row


def reduce_gradients(parameters, device):
    """Use sum reduction because every local loss already has the global 1/4 factor."""
    used = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32, device=device)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    buckets = {}
    for parameter, active in zip(parameters, used.tolist()):
        if active:
            buckets.setdefault(parameter.dtype, []).append(parameter)
    for group in buckets.values():
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in group])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        for parameter, value in zip(group, flat.split([p.numel() for p in group])):
            if parameter.grad is None:
                parameter.grad = value.reshape_as(parameter).clone()
            else:
                parameter.grad.copy_(value.reshape_as(parameter))


def capture_rng(device):
    kind, keys, offset, has_gauss, gaussian = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "python": random.getstate(), "numpy": (kind, keys.tolist(), offset, has_gauss, gaussian)}


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    random.setstate(state["python"])
    kind, keys, offset, has_gauss, gaussian = state["numpy"]
    np.random.set_state((kind, np.array(keys, dtype=np.uint32), offset, has_gauss, gaussian))


def trim_log(path, cursor):
    if not path.exists():
        return
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row["completed_updates"] <= cursor:
            rows.append(line)
    temporary = path.with_suffix(".pending")
    temporary.write_text("".join(row + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def writer_lock(path, enabled):
    if not enabled:
        yield
        return
    if os.name == "posix":
        import fcntl

        with path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
    else:
        import msvcrt

        with path.open("a+b") as handle:
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def validation_options(config, dataset):
    section = config.get("validation", {})
    options = {"small_every": 250, "full_every": 3000, "small_records": 4, "times_per_band": 2,
               "batch_times": 8, "receiver_ij": [[2, 100]], **section}
    for key in ("small_every", "full_every", "small_records", "times_per_band", "batch_times"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"validation.{key} must be a positive integer")
    rng = np.random.default_rng(config["seed"] + 61719)
    val_ids = tuple(dataset.validation_ids)
    if not val_ids:
        raise ValueError("a fixed published validation split is required for phase transfer and checkpoint selection")
    chosen = sorted(rng.choice(len(val_ids), size=min(options["small_records"], len(val_ids)), replace=False))
    slots = []
    for band in np.array_split(np.arange(len(dataset.times)), 3):
        indices = np.linspace(0, len(band) - 1, min(options["times_per_band"], len(band)), dtype=np.int64)
        slots.extend(band[indices].tolist())
    return options, val_ids, [val_ids[i] for i in chosen], slots


def _phase_candidate(run, model, phase, cursor, metrics, best, binding, model_metadata):
    score = metrics[SELECTION_METRIC]
    phase_key = f"phase_{phase}"
    update_keys = [key for key in (phase_key, "global") if score < best.get(key, {}).get("score", math.inf)]
    if not update_keys:
        return
    name = f"validation_{phase}_{cursor:07d}.pt"
    save_atomic(run / name, {"schema": CHECKPOINT_SCHEMA, "kind": "full_validation_candidate",
                            "binding": binding, "model": model.state_dict(), "phase": phase, "completed_updates": cursor,
                            "metrics": metrics, **model_metadata})
    for key in update_keys:
        best[key] = {"path": name, "score": score, "update": cursor}


def prepare_training_energy(dataset, destination, external_cache=None):
    from data_v2 import build_training_energy_metadata

    if external_cache is None:
        return build_training_energy_metadata(dataset, destination)
    if destination.exists():
        raise FileExistsError("the run already contains its bound TRAIN energy metadata")
    payload = external_cache.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("complete") is not True:
        raise ValueError("--energy-cache requires complete TRAIN energy metadata")
    # Validate a byte-identical local copy through the normal metadata reader.
    # A complete file returns without reading any pressure frames.
    temporary = destination.with_suffix(".importing")
    created = False
    try:
        with temporary.open("xb") as output:
            created = True
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        receipt = build_training_energy_metadata(dataset, temporary)
        temporary.replace(destination)
    finally:
        if created:
            temporary.unlink(missing_ok=True)
    return receipt


def train(args, config, rank, world_size, device):
    from data_v2 import StratifiedSampler, open_corpus
    from model_v2 import build_model

    main_rank = rank == 0
    output = args.run
    ranges = phase_ranges(config)
    total = config["total_updates"]
    stop = args.stop_after if args.stop_after is not None else total
    checkpoint_every = config.get("checkpoint_every", 1000)
    diagnostic_every = config.get("diagnostics_every", 250)
    if not 0 < stop <= total or any(type(v) is not int or v < 1 for v in (checkpoint_every, diagnostic_every)):
        raise ValueError("stop and reporting intervals must be within the configured positive budget")
    if type(config.get("phase_b_coefficient_loss", False)) is not bool:
        raise ValueError("phase_b_coefficient_loss must be a boolean")
    if args.resume and getattr(args, "energy_cache", None) is not None:
        raise ValueError("resume uses the TRAIN energy metadata already in the run directory")
    init_weights = getattr(args, "init_weights", None)
    init_contract = getattr(args, "init_contract", None)
    if args.resume and (init_weights is not None or init_contract is not None):
        raise ValueError("resume restores the complete run checkpoint")
    if init_contract is not None and init_weights is None:
        raise ValueError("--init-contract requires --init-weights")
    energy_path = output / "TRAIN_ENERGY.json"
    checkpoint = output / "checkpoint.pt"
    if main_rank:
        if args.resume:
            if not checkpoint.is_file() or not energy_path.is_file():
                raise FileNotFoundError("resume requires a durable checkpoint and its TRAIN energy receipt")
        else:
            output.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()
    with writer_lock(output / ".writer.lock", main_rank):
        initialize_run(args, config, rank, world_size, device, output, energy_path, checkpoint,
                       ranges, stop, checkpoint_every, diagnostic_every,
                       open_corpus, StratifiedSampler, build_model, init_weights, init_contract)


def initialize_run(args, config, rank, world_size, device, output, energy_path, checkpoint,
                   ranges, stop, checkpoint_every, diagnostic_every,
                   open_corpus, StratifiedSampler, build_model, init_weights, init_contract):
    from initialization_v2 import initialize_run_weights, semantics, validate_resume_semantics

    main_rank = rank == 0
    dataset = open_corpus(config)
    validation, val_ids, tiny_ids, tiny_slots = validation_options(config, dataset)
    external_energy_cache = getattr(args, "energy_cache", None)
    model = build_model(config, dataset, device)
    if getattr(args, "no_activation_checkpointing", False):
        model.checkpoint_queries = False
    model_semantics = semantics(model, dataset)
    inherited = None
    if not args.resume:
        inherited = initialize_run_weights(model, dataset, path=init_weights, manifest=init_contract)
        if ranges[0][0] == "C" and not inherited.get("rejected"):
            inherited["output_policy"] = "joint_optimizer_from_start"
        if main_rank:
            write_json(output / "INITIALIZATION.json", inherited)
        if inherited.get("rejected"):
            raise ValueError("warm-start tensors were rejected; see INITIALIZATION.json")
        pending = inherited["pending_output_layers"]
        inherited = {**inherited, "initial_pending_output_layers": pending.copy()}
        if ranges[0][0] == "C":
            inherited["pending_output_layers"] = []
        if world_size > 1:
            broadcast_model(model)
    if main_rank:
        if not args.resume and hasattr(dataset, "audit"):
            write_json(output / "DATASET_AUDIT.json", dataset.audit())
        energy = prepare_training_energy(dataset, energy_path, external_energy_cache)
    if world_size > 1:
        dist.barrier()
    if not main_rank:
        energy = json.loads(energy_path.read_text(encoding="utf-8"))
    sampler = StratifiedSampler(dataset.times, dataset.train_ids, config["seed"])
    energy_hash = hashlib.sha256(energy_path.read_bytes()).hexdigest()
    binding = {"config": config, "dataset": dataset.binding, "energy_sha256": energy_hash,
               "device_type": device.type, "world_size": world_size}
    start, best = 0, {}
    saved = None
    if args.resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("schema") != CHECKPOINT_SCHEMA or saved["binding"] != binding:
            raise ValueError("resume requires identical configuration, data, energy metadata, device type and world size")
        validate_resume_semantics(saved, model_semantics)
        start, best, inherited = saved["completed_updates"], saved["best"], saved.get("initializer")
        if not 0 <= start <= stop:
            raise ValueError("requested stop precedes the saved accepted-update cursor")
        model.load_state_dict(saved["model"], strict=True)
        sampler.load_state_dict(saved["sampler"])
        if saved["sampler"].get("updates_drawn", start) != start:
            raise ValueError("checkpoint sampler draw count does not match its accepted-update cursor")

    if main_rank:
        if args.resume:
            for file in ("train.jsonl", "validation.jsonl", "calibration.jsonl"):
                trim_log(output / file, start)
        else:
            write_json(output / "CONFIG.json", config)
            write_json(output / "RUN_BINDING.json", {"binding": binding, "initializer": inherited,
                "model_semantics": model_semantics,
                "train_ids": list(dataset.train_ids), "validation_ids": list(val_ids), "test_ids": list(dataset.test_ids),
                "native_times_s": np.asarray(dataset.times).tolist(), "small_validation_ids": tiny_ids,
                "small_validation_slots": tiny_slots, "global_records_per_update": 4, "frames_per_record": 4,
                "external_train_energy_cache": str(external_energy_cache.resolve()) if external_energy_cache else None,
                "supervised_frame_budget": SUPERVISED_FRAMES * config["total_updates"],
                "paper_example_frame_budget": 120000, "phases": ranges, "amp_enabled": False})

    def run_validation(full):
        ids, slots = (val_ids, None) if full else (tiny_ids, tiny_slots)
        return evaluate(model, dataset, ids, slots, validation["batch_times"], validation["receiver_ij"])

    cursor = start
    try:
        with PressureBatchPrefetch(dataset, sampler, rank, world_size,
                                   enabled=getattr(args, "prefetch", False)) as batches:
            loop(config, rank, world_size, device, output, checkpoint, ranges, stop, checkpoint_every,
                 diagnostic_every, dataset, model, sampler, batches, energy, binding, inherited, model_semantics,
                 validation, run_validation, start, best, saved)
    except BaseException as error:
        if main_rank:
            status_path = output / "STATUS.json"
            if status_path.exists():
                cursor = json.loads(status_path.read_text(encoding="utf-8"))["completed_updates"]
            write_json(status_path, {"state": "FAILED", "completed_updates": cursor,
                                    "error": f"{type(error).__name__}: {error}"})
        raise


def loop(config, rank, world_size, device, output, checkpoint, ranges, stop, checkpoint_every,
         diagnostic_every, dataset, model, sampler, batches, energy, binding, inherited, model_semantics, validation,
         run_validation, start, best, saved):
    main_rank = rank == 0
    started = perf_counter()
    current_phase = None
    saved_phase = saved["phase"] if saved is not None else None
    frozen_digest = None
    total = config["total_updates"]
    stage_b_initialization = saved.get("stage_b_initialization") if saved is not None else None
    previous_cost = saved.get("training_cost", {}) if saved is not None else {}
    elapsed_before_resume = previous_cost.get("training_loop_elapsed_s", 0.)
    boundary_fit_frames = previous_cost.get("boundary_calibration_fit_frames", 0)
    boundary_frame_evaluations = previous_cost.get("boundary_calibration_pressure_frame_evaluations", 0)
    boundary_wall_s = previous_cost.get("boundary_calibration_wall_s", 0.)
    pending = (inherited or {}).get("pending_output_layers", [])
    if saved_phase in ("B", "C") and pending:
        raise ValueError("resumed B/C checkpoint carries unresolved startup readouts")

    def model_metadata():
        return {"model_semantics": model_semantics, "initializer": inherited,
                "stage_b_initialization": stage_b_initialization}

    def cost(cursor):
        startup = stage_b_initialization or {}
        supervised = cursor * SUPERVISED_FRAMES
        startup_evaluations = startup.get("pressure_frame_evaluations", 0)
        return {"completed_updates": cursor, "training_supervised_frames": supervised,
                "stage_b_initialization_fit_frames": startup.get("fit_frames", 0),
                "stage_b_initialization_pressure_frame_evaluations": startup_evaluations,
                "stage_b_initialization_wall_s": startup.get("wall_s", 0.),
                "boundary_calibration_fit_frames": boundary_fit_frames,
                "boundary_calibration_pressure_frame_evaluations": boundary_frame_evaluations,
                "boundary_calibration_wall_s": boundary_wall_s,
                "training_and_calibration_pressure_frames": supervised + startup_evaluations + boundary_frame_evaluations,
                "training_loop_elapsed_s": elapsed_before_resume + perf_counter() - started}

    def status(cursor, phase, state="RUNNING"):
        if main_rank:
            write_json(output / "STATUS.json", {"state": state, "pid": os.getpid(), "phase": phase,
                       "completed_updates": cursor, "target_updates": total,
                       "supervised_frames": cursor * SUPERVISED_FRAMES, "world_size": world_size,
                       "cpu_prefetch_enabled": batches.enabled,
                       "activation_checkpointing_enabled": getattr(model, "checkpoint_queries", None),
                       "elapsed_s": perf_counter() - started, "training_cost": cost(cursor)})

    def save_checkpoint(cursor, phase):
        sampler_state = sampler.state_dict()
        if sampler_state["updates_drawn"] != cursor:
            raise RuntimeError("the training sampler must be committed at the checkpoint cursor")
        rng = [capture_rng(device)]
        if world_size > 1:
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, rng[0])
            rng = gathered
        if main_rank:
            training_cost = cost(cursor)
            save_atomic(checkpoint, {"schema": CHECKPOINT_SCHEMA, "binding": binding, **model_metadata(),
                       "phase": phase, "completed_updates": cursor, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "sampler": sampler_state, "rng_by_rank": rng, "best": best,
                       "training_cost": training_cost})
            write_json(output / "BEST_VALIDATION.json", best)
            write_json(output / "TRAINING_COST.json", training_cost)
        if world_size > 1:
            dist.barrier()

    if saved is not None:
        restore_rng(saved["rng_by_rank"][rank], device)
        if main_rank and stage_b_initialization is not None:
            write_json(output / "STAGE_B_INITIALIZATION.json", stage_b_initialization)
    status(start, saved["phase"] if saved else ranges[0][0])
    for update in range(start, stop):
        phase, begin, end = next(part for part in ranges if part[1] <= update < part[2])
        if phase != current_phase:
            is_resume = saved is not None and saved["phase"] == phase
            if phase == "B" and not is_resume:
                if main_rank:
                    candidate = best.get("phase_A")
                    if candidate is None:
                        raise RuntimeError("stage B requires the stage A best full-time validation state")
                    phase_a = torch.load(output / candidate["path"], map_location="cpu", weights_only=True)
                    model.load_state_dict(phase_a["model"], strict=True)
                if world_size > 1:
                    broadcast_model(model)
            model.set_phase(phase)
            model.zero_grad(set_to_none=True)
            optimizer, scheduler, parameters, warm = build_optimizer(model, phase, end - begin, config)
            if is_resume:
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                if scheduler.last_epoch != start - begin:
                    raise ValueError("checkpoint scheduler does not match the saved phase cursor")
            saved = None
            current_phase = phase
            frozen_digest = _base_digest(model) if phase == "B" else None
            if phase == "B" and not is_resume:
                if main_rank:
                    stage_b_initialization = fit_stage_b_outputs(output, model, dataset, energy, config, pending,
                                                                update, frozen_digest)
                if world_size > 1:
                    shared = [stage_b_initialization]
                    dist.broadcast_object_list(shared, src=0)
                    stage_b_initialization = shared[0]
                if stage_b_initialization["state"] != "complete":
                    if main_rank:
                        write_json(output / "STAGE_B_INITIALIZATION.json", stage_b_initialization)
                    raise RuntimeError(stage_b_initialization["error"])
                rewritten = stage_b_initialization["rewritten_parameters"]
                clear_affine_optimizer_state(optimizer, model, rewritten, synchronize=world_size > 1)
                if pending:
                    inherited = {**inherited, "pending_output_layers": [], "resolved_output_layers": pending.copy()}
                    pending = []
                    # Commit the fitted readout with the sampler and optimizer still at the first B update.
                    save_checkpoint(update, phase)
                if main_rank:
                    write_json(output / "STAGE_B_INITIALIZATION.json", stage_b_initialization)
                    status(update, phase)
            if main_rank:
                event = {"event": "phase_started", "phase": phase, "completed_updates": update,
                         "configured_phase_updates": end - begin, "resumed_current_phase": is_resume,
                         "effective_warmup_updates": warm,
                         "optimizer_parameter_count": sum(p.numel() for p in parameters)}
                append_json(output / "train.jsonl", event)
                print(json.dumps(event), flush=True)
        next_update = update + 1
        boundary = next_update == end
        diagnostics = update == begin or next_update % diagnostic_every == 0 or boundary
        full = boundary or next_update % validation["full_every"] == 0
        small = next_update % validation["small_every"] == 0
        draws, local_batch = batches.take()
        if next_update < stop and not (full or small):
            batches.schedule()
        training = step_microbatches(model, optimizer, scheduler, parameters, dataset, draws,
                                     sampler.strata_sizes, energy, device, rank, world_size, diagnostics,
                                     gradient_clip=config.get("gradient_clip_norm", 1.),
                                     phase_b_coefficient_loss=(phase == "B" and config.get("phase_b_coefficient_loss", False)),
                                     local_batch=local_batch)
        del local_batch
        training.update(completed_updates=next_update, phase=phase, records=[key for key, _ in draws],
                        time_slots=[np.asarray(slots).tolist() for _, slots in draws],
                        stratum_draw_counts=[GLOBAL_RECORDS] * FRAMES_PER_RECORD,
                        supervised_frames=SUPERVISED_FRAMES, cumulative_supervised_frames=next_update * SUPERVISED_FRAMES)
        if phase == "B" and diagnostics:
            if _base_digest(model) != frozen_digest:
                raise RuntimeError("frozen BASE parameters or buffers changed in phase B")
            training["base_frozen_exact"] = True
        if main_rank:
            append_json(output / "train.jsonl", training)
            if diagnostics:
                print(json.dumps(training), flush=True)
        calibrated = boundary and phase in ("B", "C") and config.get("affine_calibration", True)
        if calibrated:
            changed = []
        if main_rank and small:
            small_metrics = run_validation(False)
            append_json(output / "validation.jsonl", {"completed_updates": next_update, "phase": phase,
                        "kind": "small", **small_metrics})
        if main_rank and full:
            metrics = run_validation(True)
            append_json(output / "validation.jsonl", {"completed_updates": next_update, "phase": phase,
                        "kind": "full", **metrics})
            if calibrated:
                metrics, calibration = calibrate_at_boundary(output, model, dataset, energy, config, run_validation,
                                                            phase, next_update, metrics, binding, model_metadata())
                changed = calibration["rewritten_parameters"]
                boundary_fit_frames += calibration.get("fit_frames", 0)
                boundary_frame_evaluations += calibration.get("pressure_frame_evaluations", 0)
                boundary_wall_s += calibration["wall_s"]
            _phase_candidate(output, model, phase, next_update, metrics, best, binding, model_metadata())
        if calibrated:
            if world_size > 1:
                shared = [changed]
                dist.broadcast_object_list(shared, src=0)
                changed = shared[0]
            clear_affine_optimizer_state(optimizer, model, changed, synchronize=world_size > 1)
        if (next_update == stop or next_update % checkpoint_every == 0 or full
                or (next_update == 1 and config.get("save_first_update", False))):
            save_checkpoint(next_update, phase)
        status(next_update, phase)
    status(stop, current_phase or saved_phase, "COMPLETED" if stop == total else "PAUSED")


@torch.no_grad()
def broadcast_model(model):
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor, src=0)


def fit_stage_b_outputs(output, model, dataset, energy, config, pending, cursor, frozen_digest):
    report = {"completed_updates": cursor, "phase": "B", "state": "complete",
              "pending_output_layers": pending, "rewritten_parameters": [],
              "output_policy": "zero_anchor_fit" if pending else "keep_compatible_readouts",
              "fit_frames": 0, "pressure_frame_evaluations": 0, "wall_s": 0.}
    if not pending:
        return report

    from calibration_v2 import calibrate_detail_affines

    started = perf_counter()
    snapshot = snapshot_output_affines(model)
    try:
        if not set(pending) <= {index for index, _, _ in snapshot.values()}:
            raise ValueError("pending B output layers do not match the current pressure hierarchy")
        result = calibrate_detail_affines(model, dataset, energy, config, new_output_layers=pending)
        if not result["accepted"]:
            raise RuntimeError("B entry did not accept a finite fit of the pending readouts")
        changed = changed_output_affines(snapshot)
        if any(snapshot[name][0] not in pending for name in changed):
            raise RuntimeError("B entry calibration changed a compatible readout")
        if _base_digest(model) != frozen_digest:
            raise RuntimeError("B entry calibration changed the frozen BASE state")
        proposed = result.pop("candidate_affines", None)
        if proposed is not None:
            save_atomic(output / "stage_b_initialized_affines.pt", {"affines": proposed, "phase": "B",
                        "completed_updates": cursor})
        report.update(result, rewritten_parameters=changed, wall_s=perf_counter() - started)
    except Exception as error:
        with torch.no_grad():
            for _, parameter, previous in snapshot.values():
                parameter.copy_(previous)
        report.update(state="failed", error=f"{type(error).__name__}: {error}", wall_s=perf_counter() - started)
    return report


def calibrate_at_boundary(output, model, dataset, energy, config, run_validation, phase, cursor, before, binding,
                          model_metadata):
    from calibration_v2 import calibrate_detail_affines

    started = perf_counter()
    name = f"calibration_{phase}_{cursor:07d}"
    common = {"schema": CHECKPOINT_SCHEMA, "binding": binding, "phase": phase, "completed_updates": cursor,
              **model_metadata}
    save_atomic(output / f"{name}_before.pt", {**common, "model": model.state_dict(), "metrics": before})
    snapshot = snapshot_output_affines(model)
    first_evaluation = True

    def evaluate_calibration():
        nonlocal first_evaluation
        if first_evaluation:
            first_evaluation = False
            return before
        return run_validation(True)

    result = calibrate_detail_affines(model, dataset, energy, config, evaluate_calibration)
    proposed = result.pop("candidate_affines", None)
    if proposed is not None:
        save_atomic(output / f"{name}_trial_affines.pt", {"affines": proposed, "phase": phase,
                    "completed_updates": cursor})
    after = result.pop("selected_validation", None)
    if after is None:
        after = result.get("validation_after") if result.get("accepted", True) else before
    if after is None:
        after = run_validation(True)
    save_atomic(output / f"{name}_after.pt", {**common, "model": model.state_dict(), "metrics": after})
    result.update(rewritten_parameters=changed_output_affines(snapshot), wall_s=perf_counter() - started)
    append_json(output / "calibration.jsonl", {"completed_updates": cursor, "phase": phase, **result,
                "validated_before": before, "validated_after": after})
    return after, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--init-weights", "--init-base", type=Path, help="warm-start semantically compatible weights")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--init-contract", type=Path, help="semantic manifest bound to the warm-start checkpoint")
    parser.add_argument("--energy-cache", type=Path, help="import a complete, bound TRAIN_ENERGY.json into a new run")
    parser.add_argument("--prefetch", action="store_true", help="read the next training batch on one CPU thread")
    parser.add_argument("--no-activation-checkpointing", action="store_true",
                        help="retain query activations for backward when the GPU has room")
    parser.add_argument("--stop-after", type=int, help="pause after this absolute accepted-update count")
    args = parser.parse_args()
    if args.init_contract is not None and args.init_weights is None:
        parser.error("--init-contract requires --init-weights")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rank, world_size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 1 <= world_size <= GLOBAL_RECORDS or not 0 <= rank < world_size:
        parser.error("up to four ranks may share the fixed four-record update")
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", local_rank if world_size > 1 else device.index or 0)
        torch.cuda.set_device(device)
        if "cuda_memory_fraction" in config:
            torch.cuda.set_per_process_memory_fraction(float(config["cuda_memory_fraction"]), device)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(config.get("cpu_threads", 2))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=3))
    try:
        train(args, config, rank, world_size, device)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
