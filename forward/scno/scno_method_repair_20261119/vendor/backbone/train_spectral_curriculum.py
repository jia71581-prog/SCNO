"""Supervised time-window curriculum for direct pressure queries."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import os

import numpy as np
import torch

from spectral_marmousi import build_spectral_marmousi
from full_axis_pressure import FullAxisPressure
from stratified_time import StratifiedTimeSampler


@dataclass(frozen=True)
class Window:
    start_update: int
    end_time_s: float


class TimeWindowCurriculum:
    """At each update, sample distinct frame slots uniformly within its prefix."""

    def __init__(self, times_s, windows, queries_per_record, seed, total_updates):
        self.times_s = np.asarray(times_s, dtype=np.float64)
        self.windows = tuple(Window(**window) for window in windows)
        self.queries = queries_per_record
        self.seed = seed
        starts = [w.start_update for w in self.windows]
        ends = [w.end_time_s for w in self.windows]
        if self.times_s.ndim != 1 or not self.times_s.size:
            raise ValueError("expected a nonempty vector of registered times")
        if not (np.isfinite(self.times_s).all() and (np.diff(self.times_s) > 0).all()):
            raise ValueError("registered times must be finite and strictly increasing")
        if type(total_updates) is not int or total_updates < 1:
            raise ValueError("total_updates must be a positive integer")
        if not starts or any(type(start) is not int for start in starts):
            raise ValueError("stages require integer start_update values")
        if starts[0] != 0 or any(b <= a for a, b in zip(starts, starts[1:])) or starts[-1] >= total_updates:
            raise ValueError("stages must start at update zero and increase within the update budget")
        if not np.isfinite(ends).all() or any(b <= a for a, b in zip(ends, ends[1:])):
            raise ValueError("stage time bounds must be finite and increasing")
        if type(queries_per_record) is not int or queries_per_record < 1:
            raise ValueError("queries_per_record must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        # The model stores its axis in float32. Round decimal bounds to
        # the same representation so a registered boundary frame is included.
        self.prefix_lengths = np.searchsorted(
            self.times_s.astype(np.float32), np.asarray(ends, dtype=np.float32), side="right"
        )
        if self.prefix_lengths[0] < self.queries:
            raise ValueError("the first window has fewer registered times than requested queries")
        if (np.diff(self.prefix_lengths) <= 0).any():
            raise ValueError("each stage must add at least one registered time")
        if self.prefix_lengths[-1] != self.times_s.size:
            raise ValueError("the final window must include every registered time")
        self.starts = np.asarray(starts)
        self.total_updates = total_updates

    def sample(self, update, record):
        if not 0 <= update < self.total_updates:
            raise ValueError("update is outside the configured curriculum")
        stage = int(np.searchsorted(self.starts, update, side="right") - 1)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, update, record, len(self.times_s)]))
        slots = rng.choice(int(self.prefix_lengths[stage]), size=self.queries, replace=False)
        return stage, np.sort(slots)

    def record_for_update(self, update, records):
        epoch, slot = divmod(update, records)
        return int(self.record_order(epoch, records)[slot])

    def record_order(self, epoch, records):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 201]))
        return rng.permutation(records)


class PressureCache:
    """Read R25's registered eight initial labels and 392 later labels."""

    def __init__(self, path, model):
        self.path = Path(path).resolve()
        metadata = self.path / "CACHE_METADATA.json"
        info = json.loads(metadata.read_text())
        if not str(info.get("status", "")).startswith("COMPLETE_VERIFIED_TRAIN"):
            raise ValueError("select a completed, verified TRAIN cache")
        roster_path = self.path.parent / "TRAIN_ROSTER.json"
        roster = json.loads(roster_path.read_text())["records"]
        roster_digest = hashlib.sha256(roster_path.read_bytes()).hexdigest()
        if roster_digest != info.get("TRAIN_roster_SHA"):
            raise ValueError("TRAIN roster digest differs from the cache metadata")
        self.ids = info["ids"]
        if not self.ids or len(self.ids) != len(set(self.ids)):
            raise ValueError("expected a nonempty, unique set of records")
        if len(roster) != len(self.ids) or any(
            item.get("sample_id") != sample_id or item.get("split") != "train" or
            item.get("family") != "marmousi" for item, sample_id in zip(roster, self.ids)
        ):
            raise ValueError("the ordered cache must contain only Marmousi TRAIN records")
        if info["protocol_SHA"] != model.protocol_SHA:
            raise ValueError("the cache and model use different native pressure-frame protocols")
        normalizer_path = self.path.parent / "normalization_ul.json"
        if hashlib.sha256(normalizer_path.read_bytes()).hexdigest() != info.get("normalizer_SHA"):
            raise ValueError("the normalization file does not match the pressure cache")
        cache_scale = float(json.loads(normalizer_path.read_text())["pressure_scale_pa"])
        self.model_scale = float(model.normalizer.metadata.pressure_scale_pa)
        self.cache_to_model = cache_scale / self.model_scale
        if not np.isfinite(self.cache_to_model) or self.cache_to_model <= 0:
            raise ValueError("pressure normalization scales must be positive and finite")
        self.frame_indices = np.asarray(model.full_native_frames, dtype=np.int64)
        self.times = model.full_times_s.detach().cpu().numpy()
        if self.times.size != 400 or tuple(self.frame_indices[:8]) != model.known_native_frames:
            raise ValueError("the model must use the registered 8+392 time layout")

        records = len(self.ids)
        arrays = {
            "velocity": ("known_velocity.npy", (records, 1, 1, 201, 201)),
            "source": ("known_source.npy", (records, 1, 5)),
            "axis": ("known_time_axis.npy", (records, model.native_times_s.numel())),
            "early": ("known_ic.npy", (records, 1, 8, 201, 201)),
            "future": ("target.npy", (records, 1, 392, 201, 201)),
            "energy": ("energy.npy", (records, 392)),
        }
        binding = {
            "path": str(self.path),
            "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            "arrays": {},
        }
        for key, (filename, shape) in arrays.items():
            array_path = self.path / filename
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
                raise ValueError(f"{filename} must contain floating point data of shape {shape}")
            setattr(self, key, array)
            stat = array_path.stat()
            binding["arrays"][filename] = [str(array_path.resolve()), stat.st_size, stat.st_mtime_ns]
        self.binding = binding

    def batch(self, record, slots):
        slots = np.asarray(slots, dtype=np.int64)
        if slots.ndim != 1 or not slots.size or slots[0] < 0 or slots[-1] >= self.times.size:
            raise ValueError("expected valid registered frame slots")
        if not (np.diff(slots) > 0).all():
            raise ValueError("registered frame slots must be unique and ordered")
        source = np.array(self.source[record], dtype=np.float32, copy=True)
        amplitude = float(source[0, 4])
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("expected a positive finite source amplitude")
        record_times = np.array(self.axis[record, self.frame_indices], dtype=np.float32, copy=True)
        if not np.allclose(record_times, self.times, rtol=0, atol=2e-7):
            raise ValueError("cache record times do not align with the model's registered times")
        early_factor = self.model_scale * amplitude
        early = np.asarray(self.early[record, 0], dtype=np.float64) / early_factor
        energy = float(np.square(early).sum() + self.cache_to_model**2 *
                       np.asarray(self.energy[record], dtype=np.float64).sum())
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError("the record's full registered reference energy must be finite and positive")
        early_slots = slots < 8
        target = np.empty((len(slots), 201, 201), dtype=np.float32)
        target[early_slots] = early[slots[early_slots]]
        target[~early_slots] = self.future[record, 0, slots[~early_slots] - 8] * self.cache_to_model
        if not np.isfinite(target).all():
            raise ValueError("selected reference frames contain nonfinite values")
        return {
            "velocity": torch.from_numpy(np.array(self.velocity[record], dtype=np.float32, copy=True)),
            "source": torch.from_numpy(source),
            "times": torch.from_numpy(record_times[slots]),
            "target": torch.from_numpy(target),
            "full_reference_energy": energy,
        }


def supervised_data_loss(model, batch, device, registered_frames, energy_floor):
    """Estimate the current window's mean SSE normalized by full-axis mean energy."""
    prediction = model(batch["velocity"].to(device), batch["source"].to(device),
                       batch["times"].to(device), normalized=True)
    reference = batch["target"].to(device)
    if prediction.shape != reference.shape:
        raise ValueError("each requested time must predict its matched reference pressure field")
    squared_error = (prediction.double() - reference.double()).square()
    energy = max(batch["full_reference_energy"], energy_floor)
    if "frame_weights" not in batch:
        return squared_error.sum() * (registered_frames / len(reference) / energy)
    weights = batch["frame_weights"].to(device=device, dtype=torch.float64)
    return (squared_error.flatten(1).sum(1) * weights).sum() * (registered_frames / energy)


@contextmanager
def run_lock(run, resume):
    import fcntl

    if not resume:
        run.mkdir(parents=True, exist_ok=False)
    elif not (run / "checkpoint.pt").is_file():
        raise FileNotFoundError(f"no resumable checkpoint at {run}")
    with (run / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def checkpoint(run, model, optimizer, binding, next_update, device):
    payload = {
        "binding": binding,
        "next_update": next_update,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
    }
    if device.type == "cuda":
        payload["device_rng"] = torch.cuda.get_rng_state(device)
    temporary = run / ".checkpoint.pending"
    torch.save(payload, temporary)
    os.replace(temporary, run / "checkpoint.pt")


def validate_resume_binding(saved, binding, extend_budget=False, revise_sampling=False):
    previous = saved["binding"]
    if previous == binding:
        return
    old_config = previous["config"]
    new_budget = binding["config"]["total_updates"]
    old_budget = old_config["total_updates"]
    allowed = {}
    if extend_budget and 0 <= saved["next_update"] <= old_budget <= new_budget:
        allowed["total_updates"] = new_budget
    if revise_sampling:
        allowed.update({key: binding["config"][key]
                        for key in ("queries_per_record", "time_sampling", "time_strata")})
    if {**previous, "config": {**old_config, **allowed}} == binding:
        return
    raise ValueError("resume requires the same data, initializer, device and training configuration; "
                     "only explicitly requested budget or time-sampling changes are permitted")


def run_training(args, *, cache=None, model_factory=build_spectral_marmousi, on_update=None):
    config = json.loads(args.config.read_text())
    device = torch.device(args.device)
    torch.manual_seed(config["seed"])
    model = model_factory(config.get("model", {}), device=device).train()
    manifest = getattr(args, "full401_manifest", None)
    slot_key = "frame_slots"
    if cache is None:
        if manifest:
            cache = FullAxisPressure(manifest, model)
        else:
            cache = PressureCache(args.cache, model)
            slot_key = "registered_frame_slots"
    plan = TimeWindowCurriculum(cache.times, config["stages"], config["queries_per_record"],
                                config["seed"], config["total_updates"])
    stratified = config.get("time_sampling") == "stratified_cycle"
    if stratified:
        sampler = StratifiedTimeSampler(plan, len(cache.ids), config["time_strata"])

    def prepare_update(update):
        record = plan.record_for_update(update, len(cache.ids))
        if stratified:
            stage, slots, weights = sampler.sample(update, record)
        else:
            stage, slots = plan.sample(update, record)
        batch = cache.batch(record, slots)
        if stratified:
            batch["frame_weights"] = torch.from_numpy(weights)
        return record, stage, slots, batch
    floor = float(config["energy_floor"])
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("energy_floor must be positive and finite")
    interval = config["save_every_updates"]
    if type(interval) is not int or interval < 1:
        raise ValueError("save_every_updates must be a positive integer")
    groups = model.parameter_groups(config["base_lr"], config["refiner_lr"], config["weight_decay"])
    optimizer = torch.optim.AdamW(groups)
    binding = {
        "config": config,
        "cache": cache.binding,
        "initializer_sha256": model.initialization_receipt["checkpoint_SHA"],
        "device_type": device.type,
    }
    start = 0
    with run_lock(args.run, args.resume):
        if args.resume:
            saved = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=True)
            validate_resume_binding(saved, binding, getattr(args, "extend_budget", False),
                                    getattr(args, "revise_sampling", False))
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            start = saved["next_update"]
            torch.set_rng_state(saved["torch_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(saved["device_rng"], device=device)
            if on_update is not None and start:
                on_update(start, int(np.searchsorted(plan.starts, start - 1, side="right")), True)

        with ThreadPoolExecutor(max_workers=1) as prefetch:
            pending = prefetch.submit(prepare_update, start) if start < plan.total_updates else None
            for update in range(start, plan.total_updates):
                record, stage, slots, batch = pending.result()
                next_update = update + 1
                if next_update < plan.total_updates:
                    pending = prefetch.submit(prepare_update, next_update)
                optimizer.zero_grad(set_to_none=True)
                with model.scopes():
                    loss = supervised_data_loss(model, batch, device, plan.times_s.size, floor)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite pressure data loss")
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"],
                                               error_if_nonfinite=True)
                optimizer.step()
                print(json.dumps({"update": next_update, "record": cache.ids[record],
                                  "stage": stage + 1, "window_end_s": plan.windows[stage].end_time_s,
                                  slot_key: slots.tolist(), "data_loss": float(loss.detach().cpu())}), flush=True)
                first_in_phase = next_update == start + 1
                revised_resume = args.resume and getattr(args, "revise_sampling", False)
                saved = (next_update % interval == 0 or next_update == plan.total_updates
                         or (next_update == 1 and config.get("save_first_update", False))
                         or (first_in_phase and revised_resume))
                if saved:
                    checkpoint(args.run, model, optimizer, binding, next_update, device)
                if on_update is not None:
                    on_update(next_update, stage + 1, saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--full401-manifest", type=Path, help="TRAIN manifest for the complete 0–1 s axis")
    data.add_argument("--cache", type=Path, help="verified legacy registered 400-frame TRAIN cache")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
