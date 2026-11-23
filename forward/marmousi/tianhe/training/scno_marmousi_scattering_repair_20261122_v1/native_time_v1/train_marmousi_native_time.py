"""Train Marmousi pressure and B5 jointly, retaining the recorded time coordinates."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from time import monotonic
import json
import math
import os

import numpy as np
import torch

from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
from marmousi_native_time_operator import BINS, FRAMES, SIDE, NativeTimeOperator
from score_marmousi_broadband_joint import matrix_lift
from tianhe_observations import CommittedMarmousi
from tools.marmousi_target_dataset_refined.common import atomic_json, read_json, sha_file
from train_joint_pressure import PressureArchive
from train_marmousi_full_joint import PAIRS, cache_contract, relative
from train_marmousi_transport_residual import STRICT_PARENT
from train_no_pde_native import forbid_rollouts


def training_data(args, release, binding, device):
    sources = torch.as_tensor(release.sources, dtype=torch.float32)
    velocities, names, shots = {}, [], []
    pressure = torch.empty((PAIRS, BINS, SIDE, SIDE, 2), dtype=torch.float16, device=device)
    receiver = torch.empty((PAIRS, FRAMES, 201), dtype=torch.float16, device=device)
    pressure_prior, receiver_prior = [], []
    for shot in range(len(sources)):
        shard = torch.load(args.cache / f"SHOT_{shot:02}.pt", weights_only=True, map_location="cpu", mmap=True)
        if shard["shot"] != shot or shard["plan"] != binding:
            raise ValueError("Unexpected training label shard contract")
        start = len(names)
        for key in shard["rows"]:
            row = release.rows[key]
            if row["split"] != "train" or row["is_exact_target"]:
                raise ValueError("Only ordinary TRAIN velocity and waveform pairs are accepted")
            if key not in velocities:
                v = np.load(release.root / row["velocity_path"], allow_pickle=False)
                velocities[key] = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
        names.extend(shard["rows"])
        count = len(names) - start
        shots.extend([shot] * count)
        p, b = shard["pressure_label"], shard["b5_label"]
        pressure[start:len(names)].copy_(p)
        receiver[start:len(names)].copy_(b)
        pressure_prior.append(p.float().mean(0).half())
        mean_b = b.float().mean(0)
        receiver_prior.append(torch.view_as_real(torch.fft.rfft(mean_b, dim=0, norm="ortho")[:BINS]).half())
    counts = Counter(names)
    if len(names) != PAIRS or len(counts) != 800 or set(counts.values()) != {4} or len(set(zip(names, shots))) != PAIRS:
        raise ValueError("Full native-time training requires the complete set of 3,200 ordinary pairs")
    medium = torch.stack(list(velocities.values())).to(device)
    medium_id = {key: index for index, key in enumerate(velocities)}
    rows = torch.as_tensor([medium_id[key] for key in names], device=device)
    data = (medium, rows, torch.as_tensor(shots, device=device), pressure, receiver)
    _, observation, lift = matrix_lift(release)
    model = NativeTimeOperator(sources, torch.stack(pressure_prior), torch.stack(receiver_prior), observation, lift)
    return data, model.to(device)


def objective(model, data, index):
    medium, rows, shots, pressure, receiver = data
    p, b = model(medium[rows[index]], shots[index])
    prediction = torch.view_as_real(p)
    target = pressure[index].float()
    def spectral_energy(value):
        return value[:, 1:].square().sum((1, 2, 3, 4)) + .5 * value[:, 0, :, :, 0].square().sum((1, 2))
    p_error = (spectral_energy(prediction - target) / spectral_energy(target).clamp_min(1.e-7)).mean()
    b_truth = receiver[index].float()
    b_error = relative(b - b_truth, b_truth, (1, 2))
    late = relative(b[:, 100:401] - b_truth[:, 100:401], b_truth[:, 100:401], (1, 2))
    terms = torch.stack((p_error, b_error, late))
    return .6 * p_error + .3 * b_error + .1 * late, terms.detach()


@torch.inference_mode()
def validate(model, archive, device):
    pooled = torch.zeros((2, 4), device=device, dtype=torch.float64)
    closure = 0.
    model.eval()
    cases = archive.items["validation"]
    if len(cases) != 12:
        raise ValueError("Expected the fixed 12-shot ordinary validation panel")
    for i, (row, _, _) in enumerate(cases):
        ref, b5, velocity = archive.arrays("validation", i)
        v = torch.from_numpy(np.array(velocity, dtype=np.float32)).to(device)
        p, b = model(v[None], torch.tensor([row["shot_id"]], device=device))
        wave = model.wavefield(p)
        energy = torch.zeros_like(pooled[0])
        for slots in torch.arange(FRAMES, device=device).split(32):
            pred = model.pressure_at(wave, b, slots)[0]
            indices = slots.cpu().numpy()
            target = torch.from_numpy(ref[indices].copy()).to(device) / archive.release.rms
            target_b5 = torch.from_numpy(b5[indices].copy()).to(device) / archive.release.rms
            energy += torch.stack([value.square().sum(dtype=torch.float64) for value in
                                   (pred - target, target, b[0, slots] - target_b5, target_b5)])
            rows = model.observation.shape[0] // 201
            observed = pred[:, :rows].flatten(1) @ model.observation
            closure = max(closure, float((observed - b[0, slots]).abs().max()))
        pooled[0] += energy
        if row["parent_group_id"] != STRICT_PARENT:
            pooled[1] += energy
    if closure > 2.e-4 or not torch.isfinite(pooled).all():
        raise ValueError(f"Invalid native B5 closure or validation energy, closure={closure}")
    scores = (pooled[:, ::2] / pooled[:, 1::2]).sqrt().tolist()
    return dict(validation=dict(zip(("pressure_l2", "b5_l2"), scores[0], strict=True)),
                strict_validation=dict(zip(("pressure_l2", "b5_l2"), scores[1], strict=True)),
                closure_max_abs=closure)


def save_checkpoint(path, state):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def fit(args, model, data, archive, binding, device):
    rng = torch.Generator().manual_seed(20261122)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.001, fused=True)
    first, steps, best = 1, 0, float("inf")
    last = args.output / "LAST.pt"
    if last.exists():
        saved = torch.load(last, weights_only=True, map_location="cpu")
        if saved["binding"] != binding:
            raise ValueError("Native-time resume requires the identical training contract")
        saved["model"]["pressure_prior"] = model.pressure_prior
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"])
        first, steps, best = saved["epoch"] + 1, saved["steps"], saved["best"]
    started = monotonic()
    prior_shards = tuple(chunk.clone() for chunk in model.pressure_prior.detach().cpu().split(4))
    for epoch in range(first, args.epochs + 1):
        begin_epoch = monotonic()
        model.train()
        progress = (epoch - 1) / max(1, args.epochs - 1)
        optimizer.param_groups[0]["lr"] = args.lr * min(1., epoch / 3) * (.1 + .45 * (1 + math.cos(math.pi * progress)))
        accum = torch.zeros(3, device=device, dtype=torch.float64)
        order = torch.randperm(PAIRS, generator=rng).to(device)
        for outer in order.split(args.batch):
            optimizer.zero_grad(set_to_none=True)
            for index in outer.split(args.microbatch):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, terms = objective(model, data, index)
                (loss * (len(index) / len(outer))).backward()
                accum += terms * len(index)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
        report = dict(epoch=epoch, epochs=args.epochs, steps=steps, pairs=PAIRS, batch=args.batch,
                      microbatch=args.microbatch, epoch_seconds=monotonic() - begin_epoch,
                      loss_pressure_b5_late=(accum / PAIRS).tolist(), seconds=monotonic() - started,
                      peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20),
                      status="COMPLETE" if epoch == args.epochs else "TRAINING")
        selected = False
        if epoch <= 2 or epoch % 5 == 0 or epoch == args.epochs:
            report.update(validate(model, archive, device))
            atomic_json(args.output / f"VAL_{epoch:03}.json", report)
            score = max(report["strict_validation"].values())
            if score < best:
                selected, best = True, score
        weights = {key: value.detach().cpu() for key, value in model.state_dict().items() if key != "pressure_prior"}
        state = dict(model=weights, optimizer=optimizer.state_dict(), rng=rng.get_state(),
                     epoch=epoch, steps=steps, best=best, binding=binding)
        if selected:
            export = dict(model=weights, binding=binding, epoch=epoch, pressure_prior_shards=prior_shards)
            save_checkpoint(args.output / "BEST.pt", export)
        save_checkpoint(last, state)
        atomic_json(args.output / "STATUS.json", report)
        print(json.dumps(report), flush=True)


def run(args):
    if min(args.epochs, args.batch, args.microbatch) < 1 or args.microbatch > args.batch:
        raise ValueError("Use positive epochs and valid batch sizes")
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261122)
    device = torch.device("cuda:0")
    with forbid_rollouts():
        forbid_exact_label_files(args.dataset)
        release = CommittedMarmousi(args.dataset)
        archive = PressureArchive(args.validation, release)
        _, label_binding = cache_contract(args, release)
        binding = dict(schema="marmousi_native_time_joint_v1", labels=label_binding,
                       validation=archive.binding, epochs=args.epochs, batch=args.batch, lr=args.lr,
                       microbatch=args.microbatch, exact_marmousi_labels=0, frames=FRAMES,
                       inference_inputs=["velocity", "registered_source_ids"],
                       operator_sha=sha_file(Path(__file__).with_name("marmousi_native_time_operator.py")),
                       trainer_sha=sha_file(Path(__file__)))
        args.output.mkdir(exist_ok=True, parents=True)
        plan = args.output / "PLAN.json"
        if plan.exists() and read_json(plan) != binding:
            raise ValueError("The native-time output belongs to a different experiment")
        atomic_json(plan, binding)
        data, model = training_data(args, release, label_binding, device)
        print(json.dumps(dict(status="DATA_READY", pairs=PAIRS, epochs=args.epochs,
                              gpu_allocated_mib=round(torch.cuda.memory_allocated() / 2**20))), flush=True)
        fit(args, model, data, archive, binding, device)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("dataset", "cache", "validation", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.0003)
    run(parser.parse_args())
