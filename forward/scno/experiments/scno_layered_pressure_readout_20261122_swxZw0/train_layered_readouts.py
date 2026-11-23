"""Train and score paired SCNO residual readouts on the sealed layered split."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from pressure_heads import DirectPressurePatches, build_pair, local_inputs, orthogonal_pressure


FRAME_COUNT = 401
STRATA = ((0, 101), (101, 201), (201, 301), (301, 401))
SAMPLES_PER_STRATUM = 16
SEEDS = (261119, 261120)
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
GRID = (201, 201)
RESULT_NAMES = ("base_pressure", "archived_scno", "orthogonal", "direct_pressure")


def atomic_json(path, document):
    working = path.with_suffix(".tmp")
    working.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    working.replace(path)


def atomic_save(path, state):
    working = path.with_suffix(".tmp")
    torch.save(state, working)
    working.replace(path)


def status(out, state, **values):
    document = {"state": state, "pid": os.getpid(), **values}
    atomic_json(out / "STATUS.json", document)
    print(json.dumps(document, allow_nan=False), flush=True)


def cpu_available():
    lines = Path("/proc/meminfo").read_text().splitlines()
    return int(next(line for line in lines if line.startswith("MemAvailable:")).split()[1]) * 1024


def lengths_and_weights(device):
    weights = [(last - first) / SAMPLES_PER_STRATUM for first, last in STRATA]
    return torch.tensor(weights, device=device).repeat_interleave(SAMPLES_PER_STRATUM)


def sample_rows(generator):
    return np.concatenate([
        np.sort(first + generator.choice(last - first, SAMPLES_PER_STRATUM, replace=False))
        for first, last in STRATA
    ])


def pressure_increment(head, features, indices, transform, direct=None, hidden=None):
    if hidden is None:
        hidden = head.hidden(features)
    packets = head.packets(hidden, head.fold(indices), len(indices))
    return orthogonal_pressure(transform, packets) if direct is None else direct(packets)


def init_optimizer(head):
    return torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True)


def learning_rate(passes, epochs):
    warmup = min(WARMUP_EPOCHS, epochs / 2)
    if passes < warmup:
        return LEARNING_RATE * max(0.02, passes / warmup)
    phase = (passes - warmup) / (epochs - warmup)
    return LEARNING_RATE * (0.1 + 0.9 * (1 + math.cos(math.pi * phase)) / 2)


def loss_step(head, optimizer, features, target, energies, rows, weights, gate, transform, direct):
    optimizer.zero_grad(set_to_none=True)
    prediction = pressure_increment(head, features, rows, transform, direct) * gate[rows]
    squared = (prediction - target).square().sum(dim=(-2, -1))
    loss = ((squared @ weights) / energies).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def frozen_fields(model, renderer, row, rows, *, original=False, verify=False):
    from ul_hard_initial_pressure import apply_hard_initial_pressure

    backbone = model.backbone
    velocity, source = backbone._inputs(row["velocity"], row["source"])
    context = backbone.encode(velocity, source)
    decoder = backbone.parent.decoder
    features = local_inputs(decoder, context.features)
    base = backbone.parent.decode(context, rows, normalized=True, detail_enabled=False)
    if verify:
        wrapped_base = backbone.decode(context, rows, normalized=True, detail_enabled=False)
        torch.testing.assert_close(base, wrapped_base, rtol=0, atol=0)
    eligible = bool(model.profiles(velocity)[-1][0])
    arrival = None
    if eligible:
        geometry = model.geometry(velocity, source)
        inherited = context.features[:, :, 1:].flatten(2).transpose(1, 2)
        coefficients = model.readout.coefficients(inherited, geometry.features)
        arrival = renderer._correction(rows, model.full_times_s[rows], coefficients, geometry, True)
        arrival = F.pad(arrival, (0, 0, 1, 0))
        base = base + arrival
    archived = None
    if original:
        archived = backbone.decode(context, rows, normalized=True)
        if arrival is not None:
            archived = archived + arrival
        if verify:
            reference = model(velocity, source, rows, normalized=True)
            ratio = (archived - reference).norm() / reference.norm()
            if float(ratio) > 2e-5:
                raise ValueError(f"The prepared U96 pressure differs from the reviewed path: {float(ratio)}")
        archived = apply_hard_initial_pressure(archived[0], rows)
    return tuple(block[0] for block in features), apply_hard_initial_pressure(base[0], rows), archived, eligible


def build_train_cache(model, renderer, store, keys, rows, out):
    started = perf_counter()
    field_shape = (FRAME_COUNT, GRID[0] - 1, GRID[1])
    residual_bytes = len(keys) * math.prod(field_shape) * torch.float32.itemsize
    if cpu_available() < residual_bytes + (8 << 30):
        raise RuntimeError("Host RAM cannot hold the full training residual cache with 8 GiB free")
    residual = torch.empty(len(keys), *field_shape, dtype=torch.float32)
    energies = torch.empty(len(keys), dtype=torch.float32, device=model.device)
    features = None
    eligible_count = 0
    for record_index, key in enumerate(keys):
        record = store.read(key)
        blocks, base, _, eligible = frozen_fields(model, renderer, record, rows)
        eligible_count += eligible
        target = record["target"].to(model.device)
        energies[record_index] = target.square().sum()
        residual[record_index].copy_((target - base)[:, 1:])
        if features is None:
            features = tuple(value.new_empty((len(keys), *value.shape)) for value in blocks)
        for tensor, value in zip(features, blocks, strict=True):
            tensor[record_index].copy_(value)
        done = record_index + 1
        if done % 50 == 0 or done == len(keys):
            status(out, "CACHING_TRAIN", records=done, total=len(keys), elapsed_s=perf_counter() - started)
    if not torch.isfinite(energies).all() or (energies <= 0).any():
        raise ValueError("Every training record must have finite positive reference energy")
    return features, residual, energies, {
        "elapsed_s": perf_counter() - started, "normalized_residual_bytes": residual_bytes,
        "training_records": len(keys), "u96_eligible_training_records": eligible_count,
    }


def gpu_smoke(model, renderer, store, key, out, rows, gate, direct):
    record = store.read(key)
    blocks, base, original, eligible = frozen_fields(model, renderer, record, rows, original=True, verify=True)
    if not eligible or base[0:3].count_nonzero() or base[:, 0].count_nonzero():
        raise ValueError("The training probe must use the U96 and shared exact pressure constraints")
    decoder = model.backbone.parent.decoder
    features = tuple(block.unsqueeze(0) for block in blocks)
    check_steps = torch.tensor([18, 35, 70, 114, 220], device=model.device)
    target = (record["target"].to(model.device) - base)[None, check_steps, 1:]
    energies = record["target"].square().sum().to(model.device)[None]
    weights = torch.ones(len(check_steps), device=model.device)
    probe = build_pair(decoder, SEEDS[0], model.device)
    losses = {}
    initial = None
    for name, head in probe.items():
        render = direct if name == "direct_pressure" else None
        output = pressure_increment(head, features, check_steps, decoder.detail_transform, render)
        if output.count_nonzero():
            raise ValueError("The paired heads must begin at the identical frozen base pressure")
        optimizer = init_optimizer(head)
        for step in range(2):
            value = loss_step(head, optimizer, features, target, energies, check_steps,
                              weights, gate, decoder.detail_transform, render)
            if step == 0:
                if initial is not None:
                    torch.testing.assert_close(value, initial, rtol=0, atol=0)
                initial = value
        if any(p.grad is None or not p.grad.count_nonzero() for p in head.parameters()):
            raise ValueError(f"The {name} head did not receive gradients in the real-record probe")
        losses[name] = float(value)
    status(out, "GPU_SMOKE_PASS", probe_training_id=key, pressure_shape=list(original.shape),
           equal_initial_loss=float(initial), second_step_loss=losses,
           trainable_parameters_each=sum(p.numel() for p in next(iter(probe.values())).parameters()))


def run_seed(decoder, direct, train_cache, run, seed, out, device, gate):
    features, residual, energies = train_cache
    head_pair = build_pair(decoder, seed, device)
    optimizers = {name: init_optimizer(head) for name, head in head_pair.items()}
    weights = lengths_and_weights(device)
    log_path = out / f"TRAIN_{seed}.jsonl"
    resume_path = out / f"RESUME_{seed}.pt"
    epochs = run["epochs"]
    batch_size = run["batch_size"]
    start_epoch, updates, arm_seconds = 0, 0, dict.fromkeys(head_pair, 0.0)
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location=device, weights_only=True)
        if saved["protocol_sha256"] != run["protocol_sha256"] or saved["seed"] != seed:
            raise ValueError("Resume state differs from the sealed training protocol")
        start_epoch, updates = saved["epoch"], saved["updates"]
        arm_seconds = saved["arm_seconds"]
        for name in head_pair:
            head_pair[name].load_state_dict(saved["heads"][name], strict=True)
            optimizers[name].load_state_dict(saved["optimizer"][name])
    for epoch in range(start_epoch + 1, epochs + 1):
        generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        permutation = generator.permutation(len(residual))
        order_key = hashlib.sha256(permutation.tobytes())
        sums = torch.zeros(len(head_pair), device=device)
        events = dict(zip(head_pair, ([], []), strict=True))
        for first in range(0, len(permutation), batch_size):
            keys = permutation[first:first + batch_size]
            rows_cpu = sample_rows(generator)
            order_key.update(rows_cpu.tobytes())
            idx = torch.as_tensor(keys, device=device)
            rows = torch.as_tensor(rows_cpu, device=device)
            query = tuple(block.index_select(0, idx) for block in features)
            target = residual[keys[:, None], rows_cpu].to(device)
            energy = energies.index_select(0, idx)
            lr = learning_rate(epoch - 1 + (first + len(keys)) / len(permutation), epochs)
            # Alternate launch order. Both arms consume these exact record and time draws.
            sequence = tuple(head_pair)
            for index in (range(2) if updates % 2 == 0 else reversed(range(2))):
                name = sequence[index]
                optimizer = optimizers[name]
                optimizer.param_groups[0]["lr"] = lr
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                loss = loss_step(head_pair[name], optimizer, query, target, energy, rows, weights, gate,
                                 decoder.detail_transform, direct if name == "direct_pressure" else None)
                end.record()
                events[name].append((start, end))
                sums[index] += len(keys) * loss
            updates += 1
        torch.cuda.synchronize(device)
        epoch_seconds = {name: sum(start.elapsed_time(end) for start, end in pairs) / 1000
                         for name, pairs in events.items()}
        for name in head_pair:
            arm_seconds[name] += epoch_seconds[name]
        snapshot = {"protocol_sha256": run["protocol_sha256"], "seed": seed,
                    "epoch": epoch, "updates": updates, "arm_seconds": arm_seconds,
                    "heads": {name: head.state_dict() for name, head in head_pair.items()}}
        atomic_save(out / f"READOUTS_{seed}_E{epoch:03d}.pt", snapshot)
        atomic_save(resume_path, {**snapshot, "optimizer": {name: optimizer.state_dict()
                                                          for name, optimizer in optimizers.items()}})
        report = {"seed": seed, "epoch": epoch, "epochs": epochs, "updates_each": updates,
                  "loss": dict(zip(head_pair, (sums / len(permutation)).tolist(), strict=True)),
                  "query_sha256": order_key.hexdigest(), "gpu_step_seconds": epoch_seconds}
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(report, allow_nan=False) + "\n")
        status(out, "TRAINING", **report)
    return {"updates_each": updates, "gpu_step_seconds": arm_seconds,
            "weights_file": f"READOUTS_{seed}_E{epochs:03d}.pt"}


def score_field(predicted, target, transform, unit):
    if predicted[:3].count_nonzero() or predicted[:, 0].count_nonzero() or not torch.isfinite(predicted).all():
        raise ValueError("A scored field violates the common initial, free-surface, or finite-pressure condition")
    reference_energy = 0.0
    squared_error = 0.0
    coarse_error = 0.0
    for prediction, truth in zip(predicted.split(32), target.split(32), strict=True):
        error = prediction.double() - truth.double()
        reference_energy += float(truth.double().square().sum())
        squared_error += float(error.square().sum())
        coarse, _ = transform.analyze(error[:, None, 1:].float())
        coarse_error += float(coarse.double().square().sum())
    return {"sse_pa2": squared_error * unit**2, "energy_pa2": reference_energy * unit**2,
            "coarsest_sse_pa2": coarse_error * unit**2,
            "detail_sse_pa2": max(0.0, squared_error - coarse_error) * unit**2,
            "relative_l2_percent": 100 * math.sqrt(squared_error / reference_energy)}


@torch.no_grad()
def evaluate(model, renderer, store, validation, rows, gate, direct, training, out):
    decoder = model.backbone.parent.decoder
    transform = decoder.detail_transform
    trained_pairs = {}
    for seed, receipt in training.items():
        trained_pairs[seed] = build_pair(decoder, int(seed), model.device)
        payload = torch.load(out / receipt["weights_file"], map_location=model.device, weights_only=True)
        for name, head in trained_pairs[seed].items():
            head.load_state_dict(payload["heads"][name], strict=True)
            head.eval()
    cases = {}
    for key in validation:
        row = store.read(key)
        feature, base, archived, eligible = frozen_fields(model, renderer, row, rows, original=True)
        if not eligible:
            raise ValueError(f"The layered validation medium does not take the shared U96 branch: {key}")
        units = model.normalizer.metadata.pressure_scale_pa * float(row["source"].reshape(-1)[4])
        target = row["target"].to(model.device)
        case = {"base_pressure": score_field(base, target, transform, units),
                "archived_scno": score_field(archived, target, transform, units), "paired": {}}
        for seed, pair in trained_pairs.items():
            scores = {}
            for name, head in pair.items():
                hidden = head.hidden(tuple(block.unsqueeze(0) for block in feature))
                render = direct if name == "direct_pressure" else None
                correction = torch.cat([pressure_increment(head, None, part, transform, render, hidden)[0]
                                        for part in rows.split(32)])
                prediction = base + F.pad(correction * gate, (0, 0, 1, 0))
                scores[name] = score_field(prediction, target, transform, units)
            case["paired"][seed] = scores
        cases[key] = case
    pooled = {}
    for seed in trained_pairs:
        pooled[seed] = {}
        for name in RESULT_NAMES:
            records = [case.get(name, case["paired"][seed].get(name)) for case in cases.values()]
            moment = {key: sum(record[key] for record in records)
                      for key in ("sse_pa2", "energy_pa2", "coarsest_sse_pa2", "detail_sse_pa2")}
            moment["relative_l2_percent"] = 100 * math.sqrt(moment["sse_pa2"] / moment["energy_pa2"])
            moment["detail_l2_over_total_reference_percent"] = 100 * math.sqrt(
                moment["detail_sse_pa2"] / moment["energy_pa2"])
            pooled[seed][name] = moment
    return cases, pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error("positive epoch and batch counts required")
    import fcntl
    from data_v2 import open_corpus
    from joint_fno_records import Trajectories
    from prepare_ul_warm_protocol import load_ul_warm_protocol
    from ul_current_model_binding import load_current_model
    from ul_fp32_arrival_inference import U96ArrivalInference
    from ul_hard_initial_pressure import apply_hard_initial_pressure

    started = perf_counter()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "RUN.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "LAYERED_READOUT_ABLATION.json").exists():
        raise FileExistsError("The paired results were already finalized")
    try:
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda:0")
        settings = json.loads(args.deployment.read_text(encoding="utf-8"))["ul"]
        model, model_receipt = load_current_model(settings["binding"], settings["training_config"],
                                                  settings["warm_protocol"], settings["completion_receipt"], device)
        source = open_corpus(json.loads(Path(settings["training_config"]).read_text(encoding="utf-8")))
        corpus = load_ul_warm_protocol(source, settings["warm_protocol"])
        store = Trajectories(corpus, args.cache)
        train = [key for key in corpus.train_ids if corpus.records[key]["family"] == "layered"]
        validation = [key for key in corpus.validation_ids if corpus.records[key]["family"] == "layered"]
        if len(train) != 450 or len(validation) != 3:
            raise ValueError("Expected the paper's 450 layered training and three external validation records")
        decoder = model.backbone.parent.decoder
        direct = DirectPressurePatches(decoder.detail_transform).to(device)
        gpu = torch.cuda.get_device_name(device)
        trainable_count = sum(p.numel() for p in build_pair(decoder, SEEDS[0], "cpu")["orthogonal"].parameters())
        context_count = sum(p.numel() for p in model.parameters()) - trainable_count
        run = {"model": model_receipt, "training_ids": train, "validation_ids": validation,
               "warm_protocol_sha256": corpus.binding["warm_protocol_sha256"],
               "grid": GRID, "frames": FRAME_COUNT, "time_step_s": .0025,
               "training_strata": STRATA, "frames_per_stratum": SAMPLES_PER_STRATUM,
               "seeds": SEEDS, "batch_size": args.batch, "epochs": args.epochs, "gpu": gpu,
               "trainable_parameters_each": trainable_count, "frozen_parameters_each": context_count,
               "full_model_parameters_each": context_count + trainable_count,
               "orthogonal_slots": [level.valid_detail.shape[-1] for level in decoder.detail_transform.levels],
               "direct_slots": direct.slots, "time_rank": decoder.detail_latent,
               "optimizer": "AdamW", "peak_learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
               "optimizer_warmup_epochs": WARMUP_EPOCHS, "optimizer_final_lr_fraction": 0.1,
               "precision": "FP32 (no TF32 or autocast)",
               "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ("train_layered_readouts.py", "pressure_heads.py")}}
        run["protocol_sha256"] = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
        if (out / "PROTOCOL.json").is_file():
            previous = json.loads((out / "PROTOCOL.json").read_text(encoding="utf-8"))
            if previous["protocol_sha256"] != run["protocol_sha256"]:
                raise ValueError("This directory contains a different paired training protocol")
        else:
            atomic_json(out / "PROTOCOL.json", run)
        rows = torch.arange(FRAME_COUNT, device=device)
        gate = apply_hard_initial_pressure(torch.ones(FRAME_COUNT, 1, 1, device=device), rows)
        renderer = U96ArrivalInference(model, prepared=True, graph=False)
        gpu_smoke(model, renderer, store, train[0], out, rows, gate, direct)
        features, residual, energies, cache_receipt = build_train_cache(model, renderer, store, train, rows, out)
        training = {str(seed): run_seed(decoder, direct, (features, residual, energies), run,
                                        seed, out, device, gate) for seed in SEEDS}
        del features, residual, energies
        status(out, "SCORING_FULL_401", cases=len(validation), seeds=len(SEEDS))
        cases, pooled = evaluate(model, renderer, store, validation, rows, gate, direct, training, out)
        result = {"state": "COMPLETE", "protocol": run, "cache": cache_receipt, "training": training,
                  "cases": cases, "pooled": pooled, "elapsed_s": perf_counter() - started,
                  "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device)}
        atomic_json(out / "LAYERED_READOUT_ABLATION.json", result)
        status(out, "COMPLETE", elapsed_s=result["elapsed_s"], pooled_percent={
            seed: {name: record["relative_l2_percent"] for name, record in value.items()}
            for seed, value in pooled.items()})
    except BaseException as exc:
        status(out, "FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
