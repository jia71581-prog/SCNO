"""Full Marmousi pretraining with native 201×201 pressure supervision."""

from argparse import ArgumentParser
from pathlib import Path
from time import monotonic
import json

import numpy as np
import torch

from build_marmousi_native201_cache import SCHEMA as CACHE_SCHEMA
from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
from marmousi_native201_operator import NATIVE, SCHEMA, NativeGridOperator
from marmousi_native_time_operator import BINS, FRAMES
from tianhe_observations import CommittedMarmousi
from tools.marmousi_target_dataset_refined.common import atomic_json, digest, read_json, sha_array, sha_file
from train_joint_pressure import PressureArchive
from train_marmousi_full_joint import PAIRS
from train_marmousi_native_time import fit, objective
from train_no_pde_native import forbid_rollouts


def load_labels(args, release, identity, receiver_identity, device):
    pressure = torch.empty((PAIRS, BINS, NATIVE, NATIVE, 2), dtype=torch.float16, device=device)
    receiver = torch.empty((PAIRS, FRAMES, NATIVE), dtype=torch.float16, device=device)
    velocities, names, shots = {}, [], []
    seen = set()
    for shot in range(len(release.sources)):
        file = args.cache / f"SHOT_{shot:02}.npy"
        receipt = read_json(file.with_suffix(".json"))
        models = receipt["models"]
        shape = (len(models), BINS, NATIVE, NATIVE, 2)
        if receipt != dict(shot=shot, models=models, shape=list(shape), plan=identity):
            raise ValueError(f"Native-grid pressure contract changed at shot {shot}")
        frame = np.load(file, mmap_mode="c", allow_pickle=False)
        b5 = torch.load(args.receiver_cache / f"SHOT_{shot:02}.pt", weights_only=True, map_location="cpu", mmap=True)
        if (frame.shape != shape or frame.dtype != np.float16 or b5["shot"] != shot or b5["plan"] != receiver_identity
                or b5["rows"] != models or b5["b5_label"].shape != (len(models), FRAMES, NATIVE)):
            raise ValueError(f"Native pressure and full B5 roster disagree at shot {shot}")
        start, stop = len(names), len(names) + len(models)
        for key in models:
            record = release.rows[key]
            if key not in release.train_set or record["is_exact_target"] or (key, shot) in seen:
                raise ValueError(f"Invalid or duplicated TRAIN pair: {key} at {shot}")
            seen.add((key, shot))
            if key not in velocities:
                v = np.load(release.root / record["velocity_path"], allow_pickle=False)
                if v.shape != (NATIVE, NATIVE) or v.dtype != np.float32 or sha_array(v) != record["velocity_sha256"]:
                    raise ValueError(f"TRAIN velocity changed: {key}")
                velocities[key] = torch.from_numpy(v)
        names.extend(models)
        shots.extend([shot] * len(models))
        pressure[start:stop].copy_(torch.from_numpy(frame))
        receiver[start:stop].copy_(b5["b5_label"])
    expected = {(key, shot) for key in release.train_ids for shot in release.rosters[key]}
    if seen != expected or len(seen) != PAIRS or len(velocities) != 800:
        raise ValueError("Native-grid supervision must use all 800 models and 3,200 committed pairs")
    medium_id = {key: index for index, key in enumerate(velocities)}
    rows = torch.tensor([medium_id[key] for key in names], device=device)
    return torch.stack(list(velocities.values())).to(device), rows, torch.tensor(shots, device=device), pressure, receiver


def select_microbatch(args, model, data, device):
    saved = read_json(args.output / "CAPACITY.json") if (args.output / "CAPACITY.json").exists() else None
    if saved is not None:
        return saved["microbatch"]
    weights = {name: param.detach().clone() for name, param in model.named_parameters()}
    indices = torch.randperm(PAIRS, generator=torch.Generator().manual_seed(20261122))[:args.batch].to(device)
    attempts = []
    candidates = [args.microbatch, *(s for s in (112, 96, 80, 64, 48, 32, 24, 16, 8, 4, 2, 1) if s < args.microbatch)]
    for size in candidates:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.001, fused=True)
        elapsed = monotonic()
        try:
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                for index in indices.split(size):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, _ = objective(model, data, index)
                    (loss * (len(index) / len(indices))).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_reserved(device)
            total = torch.cuda.mem_get_info(device)[1]
            ready = peak <= .92 * total
            report = dict(microbatch=size, batch=args.batch, updates=3, peak_reserved_mib=peak//2**20,
                          total_mib=total//2**20, seconds=monotonic()-elapsed,
                          result="PASS" if ready else "LOW_HEADROOM")
        except torch.cuda.OutOfMemoryError:
            ready = False
            report = dict(microbatch=size, batch=args.batch, result="OOM")
        finally:
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(weights[name])
            del optimizer
        attempts.append(report)
        print(json.dumps(dict(stage="CAPACITY", **report)), flush=True)
        if ready:
            atomic_json(args.output / "CAPACITY.json", dict(**report, attempts=attempts))
            return size
    raise RuntimeError("No native-grid microbatch met the measured memory headroom")


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
        cache = read_json(args.cache / "PLAN.json")
        b5 = read_json(args.receiver_cache / "PLAN.json")
        if (cache["schema"] != CACHE_SCHEMA or cache["exact_marmousi_labels"] or cache["shape"] != [NATIVE, NATIVE]
                or cache["bins"] != BINS or cache["frames"] != FRAMES or cache["release"] != release.binding
                or b5["release"] != release.binding or b5["pairs"] != cache["pairs"]):
            raise ValueError("Use matched ordinary pressure and B5 caches on the frozen release")
        native_contract = digest(cache)
        receiver_contract = digest(b5)
        binding = dict(schema=SCHEMA, pressure_labels=native_contract, receiver_labels=receiver_contract,
                       validation=archive.binding, epochs=args.epochs, batch=args.batch, lr=args.lr,
                       microbatch_limit=args.microbatch, exact_marmousi_labels=0, frames=FRAMES, bins=BINS,
                       pressure_shape=[NATIVE, NATIVE], pressure_spacing_m=10,
                       seed_sha=sha_file(args.seed), operator_sha=sha_file(Path(__file__).with_name("marmousi_native201_operator.py")),
                       trainer_sha=sha_file(Path(__file__)))
        args.output.mkdir(exist_ok=True, parents=True)
        plan = args.output / "PLAN.json"
        if plan.exists() and read_json(plan) != binding:
            raise ValueError("Native-grid resume requires the original training plan")
        atomic_json(plan, binding)
        model = NativeGridOperator.from_coarse(args.seed).to(device)
        if not np.array_equal(model.sources.cpu().numpy(), release.sources.astype(np.float32)):
            raise ValueError("Seed and data use different source configurations")
        data = load_labels(args, release, native_contract, receiver_contract, device)
        print(json.dumps(dict(stage="DATA_READY", samples=PAIRS, pressure_grid=[NATIVE, NATIVE],
                              frames=FRAMES, gpu_allocated_mib=torch.cuda.memory_allocated()//2**20)), flush=True)
        args.microbatch = select_microbatch(args, model, data, device)
        fit(args, model, data, archive, binding, device)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("dataset", "cache", "receiver-cache", "validation", "seed", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.0003)
    run(parser.parse_args())
