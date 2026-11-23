"""TRAIN-only sampled mmap loading with at most two pinned/GPU batch slots.

Lookahead uses a cloned sampler. Only acquire() commits the main sampler;
acquiring the same live plan again retries it without another random draw.
release() fences the last compute use before either slot can be overwritten.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
import math

import torch

from full_model_warm_start import FourStratumFrameSampler, full_reference_energy


def _same_state(a, b):
    return (a["semantics"] == b["semantics"] and a["drawn_records"] == b["drawn_records"]
            and a["sample_calls"] == b["sample_calls"]
            and torch.equal(a["generator_state"], b["generator_state"]))


class MMapTrainSource:
    """Load sealed per-record .pt tensors without materializing full trajectories."""

    def __init__(self, paths, *, train_ids, forbidden_ids=()):
        self.train_ids, self.forbidden_ids = set(map(str, train_ids)), set(map(str, forbidden_ids))
        if not self.train_ids or self.train_ids & self.forbidden_ids:
            raise ValueError("declare disjoint actual TRAIN and VAL/TEST IDs")
        self.paths = {str(key): Path(value) for key, value in paths.items()}
        self.read_log = []

    def __call__(self, key):
        key = str(key)
        if key not in self.train_ids or key in self.forbidden_ids:
            raise ValueError("mmap source refuses non-TRAIN records")
        row = torch.load(self.paths[key], map_location="cpu", weights_only=True, mmap=True)
        if str(row["id"]) != key:
            raise ValueError("mmap record identity differs from the declared TRAIN path")
        self.read_log.append(key)
        return row


@dataclass(frozen=True)
class SamplingPlan:
    serial: int
    record_ids: tuple
    indices: torch.Tensor
    weights: torch.Tensor
    before: dict
    after: dict


@dataclass
class PreparedBatch:
    plan: SamplingPlan
    slot: int
    future: object = None
    acquired: object = None


@dataclass
class ResidentBatch:
    tensors: tuple
    metrics: dict
    plan: SamplingPlan
    slot: int


@dataclass
class _Slot:
    host: tuple
    resident: tuple
    state: str = "free"
    serial: int = -1
    release_event: object = None
    copy_event: object = None
    copy_start: object = None
    owner_stream: object = None


class SampledTrainPrefetch:
    """Single CPU producer, double buffering and asynchronous resident transfer.

    ``row_loader(id)`` must return immutable CPU FP32 sealed TRAIN tensors.
    MMapTrainSource is supplied for per-record torch.save files. Existing
    full-401 reference energies can be supplied; otherwise they are computed
    once, in bounded CPU slabs, and cached without changing their definition.

    Usage: prepare -> acquire -> model/optimizer work -> release. Prepare the
    next batch before compute; after releasing the current slot it can prepare
    the following batch. Returned tensors are borrowed until release.
    """

    def __init__(self, row_loader, sampler, *, train_ids, forbidden_ids=(), max_batch_size,
                 device, energy_table=None, energy_floor=1e-12, buffers=2):
        self.train_ids, self.forbidden_ids = set(map(str, train_ids)), set(map(str, forbidden_ids))
        if not self.train_ids or self.train_ids & self.forbidden_ids:
            raise ValueError("prefetch requires disjoint TRAIN and VAL/TEST membership")
        if (sampler.full_frames != 401 or sampler.samples_per_stratum != 64
                or buffers not in (1, 2) or type(max_batch_size) is not int or max_batch_size < 1
                or not math.isfinite(energy_floor) or energy_floor <= 0):
            raise ValueError("use 401 native frames, four times 64 samples, and at most two positive batch slots")
        self.loader, self.sampler = row_loader, sampler
        self.device, self.max_batch_size = torch.device(device), max_batch_size
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("prefetch supports CPU checking or CUDA resident training")
        self.energy_floor = energy_floor
        self.energy_table = {str(key): float(value) for key, value in (energy_table or {}).items()}
        # Only acquire() changes the committed table. The single worker owns
        # a separate cache, so checkpoint serialization cannot race a producer.
        self._worker_energies = dict(self.energy_table)
        self._initial_energy_keys = frozenset(self.energy_table)
        self._lock, self._queue = RLock(), deque()
        self._planning_state, self._serial, self.closed = deepcopy(sampler.state_dict()), 0, False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ul-selected-mmap")
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._slots = []
        shapes = ((max_batch_size, 1, 201, 201), (max_batch_size, 5),
                  (max_batch_size, 256, 201, 201), (max_batch_size, 256),
                  (max_batch_size, 256), (max_batch_size,))
        dtypes = (torch.float32, torch.float32, torch.float32, torch.long, torch.float32, torch.float64)
        for _ in range(buffers):
            host = tuple(torch.empty(shape, dtype=dtype, pin_memory=self.device.type == "cuda")
                         for shape, dtype in zip(shapes, dtypes))
            resident = (tuple(torch.empty_like(value, device=self.device) for value in host)
                        if self.device.type == "cuda" else host)
            self._slots.append(_Slot(host, resident))
        if self.copy_stream is not None:
            self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))

    def prepare(self, record_ids):
        ids = tuple(map(str, record_ids))
        with self._lock:
            if self.closed or not ids or len(ids) > self.max_batch_size or len(set(ids)) != len(ids):
                raise ValueError("prepare a live, distinct TRAIN batch within the preallocated size")
            if not set(ids) <= self.train_ids or set(ids) & self.forbidden_ids:
                raise ValueError("VAL/TEST records must never enter the prefetch queue")
            index = next((i for i, slot in enumerate(self._slots) if slot.state in ("free", "released")), None)
            if index is None:
                raise RuntimeError("both batch slots are occupied; release a consumed batch before preparing another")
            clone = FourStratumFrameSampler(401, 64)
            clone.load_state_dict(self._planning_state)
            before = deepcopy(self._planning_state)
            sample = clone.sample(len(ids))
            after = deepcopy(clone.state_dict())
            self._serial += 1
            plan = SamplingPlan(self._serial, ids, sample.indices, sample.weights, before, after)
            slot = self._slots[index]
            old_fence = slot.release_event
            slot.state, slot.serial = "preparing", plan.serial
            handle = PreparedBatch(plan, index)
            handle.future = self._executor.submit(self._prepare_slot, slot, plan, old_fence)
            self._queue.append(handle)
            self._planning_state = deepcopy(after)
            return handle

    def _prepare_slot(self, slot, plan, old_fence):
        started, wait_started = perf_counter(), perf_counter()
        if old_fence is not None:
            old_fence.synchronize()
        fence_wait = perf_counter() - wait_started
        batch = len(plan.record_ids)
        v, s, y, indices, weights, energy = (value[:batch] for value in slot.host)
        indices.copy_(plan.indices)
        weights.copy_(plan.weights)
        io_seconds, energy_seconds, energy_hits = 0., 0., 0
        energy_updates = {}
        for row_index, key in enumerate(plan.record_ids):
            tick = perf_counter()
            row = self.loader(key)
            io_seconds += perf_counter() - tick
            if str(row["id"]) != key:
                raise ValueError("TRAIN loader returned the wrong record")
            for name in ("velocity", "source", "target"):
                if row[name].device.type != "cpu" or row[name].dtype != torch.float32:
                    raise ValueError("source mmap data must remain CPU FP32")
            target = row["target"]
            if target.shape != (401, 201, 201) or row["velocity"].numel() != 201 * 201 or row["source"].numel() != 5:
                raise ValueError("TRAIN row changed its complete native pressure or conditioning geometry")
            v[row_index].copy_(row["velocity"].reshape(1, 201, 201))
            s[row_index].copy_(row["source"].reshape(5))
            torch.index_select(target, 0, plan.indices[row_index], out=y[row_index])
            if key not in self._worker_energies:
                tick = perf_counter()
                self._worker_energies[key] = float(full_reference_energy(target[None])[0])
                energy_seconds += perf_counter() - tick
            else:
                energy_hits += 1
            full_energy = self._worker_energies[key]
            if not math.isfinite(full_energy) or full_energy < 0:
                raise ValueError("original complete reference energy must be finite and nonnegative")
            energy[row_index] = max(full_energy, self.energy_floor)
            if key not in self._initial_energy_keys:
                energy_updates[key] = full_energy
        packed_at = perf_counter()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
                slot.copy_start = torch.cuda.Event(enable_timing=True)
                slot.copy_event = torch.cuda.Event(enable_timing=True)
                slot.copy_start.record(self.copy_stream)
                for destination, value in zip(slot.resident, slot.host):
                    destination[:batch].copy_(value[:batch], non_blocking=True)
                slot.copy_event.record(self.copy_stream)
        else:
            slot.copy_event = slot.copy_start = None
        metrics = {"record_ids": list(plan.record_ids), "records": batch, "sampled_frames_per_record": 256,
                   "reference_energy_frames": 401, "cpu_prepare_seconds": packed_at - started,
                   "reuse_fence_wait_seconds": fence_wait, "mmap_loader_seconds": io_seconds,
                   "new_full_energy_seconds": energy_seconds, "energy_cache_hits": energy_hits,
                   "computed_energy_updates": energy_updates,
                   "h2d_enqueue_seconds": perf_counter() - packed_at,
                   "h2d_bytes": sum(value[:batch].numel() * value.element_size() for value in slot.host)
                       if self.device.type == "cuda" else 0, "h2d_ms": None,
                   "plan_serial": plan.serial, "sampler_committed_on_prepare": False}
        return metrics

    def acquire(self, prepared):
        with self._lock:
            if prepared.acquired is None and (not self._queue or self._queue[0] is not prepared):
                raise RuntimeError("consume prepared plans in order; checkpoint RNG follows consumption")
        tick = perf_counter()
        metrics = prepared.future.result()
        with self._lock:
            if self.closed:
                raise RuntimeError("prefetcher is closed")
            slot = self._slots[prepared.slot]
            if slot.serial != prepared.plan.serial or slot.state not in ("preparing", "acquired"):
                raise RuntimeError("this batch was released or its slot has been reused")
            if prepared.acquired is not None:
                if self.device.type == "cuda" and torch.cuda.current_stream(self.device) != slot.owner_stream:
                    raise RuntimeError("retry on the original consuming stream")
                return prepared.acquired
            if not self._queue or self._queue[0] is not prepared:
                raise RuntimeError("consume prepared plans in order; checkpoint RNG follows consumption")
            if not _same_state(self.sampler.state_dict(), prepared.plan.before):
                raise RuntimeError("main sampler changed outside consumption; discard and recreate lookahead")
            if self.device.type == "cuda":
                slot.owner_stream = torch.cuda.current_stream(self.device)
                slot.owner_stream.wait_event(slot.copy_event)
            self.sampler.load_state_dict(prepared.plan.after)
            self.energy_table.update(metrics["computed_energy_updates"])
            self._queue.popleft()
            slot.state = "acquired"
            metrics = {**metrics, "acquire_cpu_wait_seconds": perf_counter() - tick,
                       "sampler_committed_on_acquire": True}
            batch = len(prepared.plan.record_ids)
            result = ResidentBatch(tuple(value[:batch] for value in slot.resident), metrics, prepared.plan, prepared.slot)
            prepared.acquired = result
            return result

    def release(self, batch):
        with self._lock:
            slot = self._slots[batch.slot]
            if slot.serial != batch.plan.serial or slot.state != "acquired":
                raise RuntimeError("release each currently acquired slot exactly once after its last compute use")
            if self.device.type == "cuda":
                if torch.cuda.current_stream(self.device) != slot.owner_stream:
                    raise RuntimeError("release on the consuming stream after its last compute use")
                slot.release_event = torch.cuda.Event()
                slot.release_event.record(torch.cuda.current_stream(self.device))
                if slot.copy_event.query():
                    batch.metrics["h2d_ms"] = slot.copy_start.elapsed_time(slot.copy_event)
            slot.state = "released"

    def consumed_sampler_state(self):
        """Checkpoint this state only at accepted-update boundaries; excludes lookahead."""
        with self._lock:
            return deepcopy(self.sampler.state_dict())

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        if self.device.type == "cuda":
            self.copy_stream.synchronize()
            for slot in self._slots:
                if slot.owner_stream is not None:
                    slot.owner_stream.synchronize()
        self._queue.clear()
        self._slots.clear()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
