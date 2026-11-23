"""Bounded CPU/CUDA caches for full native-resolution training trajectories."""

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
import ctypes
import ctypes.util
from dataclasses import dataclass
import os
from pathlib import Path
from threading import Lock, RLock
from time import monotonic, perf_counter

import torch


_TENSORS = ("source", "velocity", "target")
_GIB = 1 << 30


@dataclass(frozen=True)
class _Packed:
    data: bytes
    shape: tuple
    original_bytes: int


@dataclass(frozen=True)
class _Pending:
    future: Future
    reserved_bytes: int
    cold_read: bool


class _PressureCodec:
    """Blosc byte-shuffle and lossless compression with thread-local codec settings."""

    def __init__(self, name, threads):
        if name not in ("none", "lz4", "zstd") or type(threads) is not int or threads < 1:
            raise ValueError("compression must be none, lz4 or zstd with a positive thread count")
        self.name, self.threads = "none", threads
        if name == "none":
            return
        path = ctypes.util.find_library("blosc")
        if not path:
            return
        try:
            lib = ctypes.CDLL(path)
            lib.blosc_list_compressors.restype = ctypes.c_char_p
            supported = lib.blosc_list_compressors().decode("ascii").split(",")
            self._encode, self._decode = lib.blosc_compress_ctx, lib.blosc_decompress_ctx
        except (OSError, AttributeError):
            return
        pointer, size, integer = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        self._encode.argtypes = [integer, integer, size, size, pointer, pointer, size, ctypes.c_char_p, size, integer]
        self._decode.argtypes = [pointer, pointer, size, integer]
        self._encode.restype = self._decode.restype = integer
        self.name = next((codec for codec in (name, "lz4") if codec in supported), "none")

    def pack(self, raw):
        size = raw.numel() * raw.element_size()
        if size > (1 << 31) - 17:  # Maximum size in the Blosc 1 header.
            return raw
        destination = ctypes.create_string_buffer(size + 16)
        n = self._encode(3, 1, 4, size, raw.data_ptr(), ctypes.addressof(destination), size + 16,
                         self.name.encode("ascii"), 0, self.threads)
        if n <= 0 or n >= size:
            return raw
        return _Packed(ctypes.string_at(destination, n), tuple(raw.shape), size)

    def unpack(self, packed):
        target = torch.empty(packed.shape, dtype=torch.float32, device="cpu")
        n = self._decode(packed.data, target.data_ptr(), packed.original_bytes, self.threads)
        if n != packed.original_bytes:
            raise ValueError("the cached pressure could not be decoded to its original byte count")
        return target


def _cgroup_limits():
    """Resolve both unified and legacy memory limits, including ancestor cgroups."""
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return []
    pairs = []
    for entry in entries:
        _, controllers, relative = entry.split(":", 2)
        if not controllers:
            root = Path("/sys/fs/cgroup")
            maximum, current = "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory")
            maximum, current = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        parts = [part for part in relative.split("/") if part and part not in (".", "..")]
        node = root.joinpath(*parts)
        while True:
            if (node / maximum).is_file() and (node / current).is_file():
                pairs.append((node / maximum, node / current))
            if node == root:
                break
            node = node.parent
    return pairs


def _host_available(limits):
    available = []
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available.append(int(line.split()[1]) * 1024)
                    break
    except OSError:
        try:
            import psutil
            available.append(int(psutil.virtual_memory().available))
        except ImportError:
            if hasattr(os, "sysconf"):
                try:
                    available.append(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
                except (OSError, ValueError):
                    pass
    for maximum, current in limits:
        try:
            limit = maximum.read_text().strip()
            if limit != "max":
                capacity, used = int(limit), int(current.read_text())
                reclaimable = 0
                try:
                    counters = dict(line.split() for line in (current.parent / "memory.stat").read_text().splitlines())
                    reclaimable = int(counters.get("total_inactive_file", counters.get("inactive_file", 0)))
                except (OSError, ValueError):
                    pass
                available.append(max(0, min(capacity, capacity - used + reclaimable)))
        except (OSError, ValueError):
            continue
    return min(available) if available else None


def _bytes(row, *, logical=False):
    size = sum(row[name].numel() * row[name].element_size() for name in ("source", "velocity"))
    target = row["target"]
    if isinstance(target, _Packed):
        return size + (target.original_bytes if logical else len(target.data))
    return size + target.numel() * target.element_size()


def _copy_row(row, device):
    return {**row, **{name: row[name].to(device=device, non_blocking=False) for name in _TENSORS}}


class RecordCache:
    """Values returned by get() are read-only to callers.

    gpu_bytes and cpu_bytes are per-process limits for cached and prefetched rows. The
    caller separately budgets the model, optimizer, activations and live batches.
    prefetch() prepares the next requested rows in order, including cold reads
    and decompression of CPU hits. Only the consumer thread uses CUDA.
    A compressed CPU copy can back a hot GPU row and is evicted first under RAM
    pressure. Raw FP32 rows retain their host source when uploaded, avoiding
    redundant device-to-host copies on GPU eviction. Both tiers stay budgeted.
    No cache files are written.
    """

    def __init__(self, store, device, gpu_bytes, cpu_bytes, host_floor_bytes=2 * _GIB,
                 *, compression="none", compression_threads=2, prefetch_workers=2, prefetch_depth=12,
                 parallel_reads=False):
        for budget in (gpu_bytes, cpu_bytes, host_floor_bytes):
            if type(budget) is not int or budget < 0:
                raise ValueError("cache budgets must be non-negative integer byte counts")
        for count in (prefetch_workers, prefetch_depth):
            if type(count) is not int or count < 1:
                raise ValueError("prefetch workers and depth must be positive integers")
        self.store = store
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("record caching supports CPU or CUDA")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.gpu_budget = gpu_bytes if self.device.type == "cuda" else 0
        self.cpu_budget, self.host_floor = cpu_bytes, host_floor_bytes
        self._codec = _PressureCodec(compression, compression_threads)
        self._requested_codec = compression
        self._limits = _cgroup_limits()
        self._cpu, self._gpu = OrderedDict(), OrderedDict()
        self._cpu_bytes = self._gpu_bytes = 0
        self._slots = getattr(store, "slots", None)
        self._estimate = ((len(self._slots) + 1) * 201 * 201 + 5) * 4 if self._slots is not None else 0
        self._executor, self._pending = None, {}
        self._pending_bytes = 0
        self._workers, self._depth = prefetch_workers, prefetch_depth
        self._lock, self._metric_lock = RLock(), Lock()
        self._io_lock = nullcontext() if parallel_reads else Lock()
        self._last_pressure_check = 0.
        self._metrics = dict(gets=0, gpu_hits=0, cpu_hits=0, store_reads=0, prefetch_reads=0,
                             prefetch_hits=0, prefetch_failures=0, cpu_evictions=0, gpu_evictions=0, gpu_oom_retries=0,
                             prefetch_cold_reads=0, prefetch_cpu_decodes=0, prefetch_high_water=0,
                             compression_s=0., decompression_s=0., compressions=0, decompressions=0,
                             store_read_s=0., get_wall_s=0., prefetch_wait_s=0., h2d_s=0., d2h_s=0.)

    def _read(self, key):
        with self._io_lock:
            start = perf_counter()
            row = self.store.read(key, cache=False)
            if str(row["id"]) != str(key):
                raise ValueError("the sealed store returned the wrong record")
            result = {"id": row["id"], "family": row["family"]}
            for name in _TENSORS:
                value = torch.as_tensor(row[name]).detach()
                if value.device.type != "cpu" or value.dtype != torch.float32:
                    raise ValueError(f"the complete {name} tensor must be stored as CPU float32")
                value = value.contiguous()
                if value.untyped_storage().nbytes() > value.numel() * value.element_size():
                    value = value.clone()  # Avoid retaining an unrelated larger backing array.
                result[name] = value
            if (result["source"].shape not in ((1, 5), (5,)) or result["velocity"].shape[-2:] != (201, 201)
                    or result["velocity"].numel() != 201 * 201):
                raise ValueError("each complete trajectory requires one five-parameter source and one velocity grid")
            target = result["target"]
            if (target.ndim != 3 or target.shape[1:] != (201, 201) or not target.shape[0]
                    or (self._slots is not None and target.shape[0] != len(self._slots))):
                raise ValueError("the cache requires complete native pressure trajectories on the 201×201 grid")
            with self._metric_lock:
                self._metrics["store_reads"] += 1
                self._metrics["store_read_s"] += perf_counter() - start
            return result

    def _pack_row(self, row):
        if self._codec.name == "none" or isinstance(row["target"], _Packed):
            return row
        start = perf_counter()
        target = self._codec.pack(row["target"])
        with self._metric_lock:
            self._metrics["compression_s"] += perf_counter() - start
            self._metrics["compressions"] += 1
        return {**row, "target": target}

    def _unpack_row(self, row):
        start = perf_counter()
        result = {**row, "target": self._codec.unpack(row["target"])}
        with self._metric_lock:
            self._metrics["decompression_s"] += perf_counter() - start
            self._metrics["decompressions"] += 1
        return result

    def _prepare_row(self, key, packed):
        if packed is not None:
            return self._unpack_row(packed), packed
        row = self._read(key)
        if self._codec.name == "none":
            return row, None
        saved = self._pack_row(row)
        return row, saved if isinstance(saved["target"], _Packed) else None

    def _read_workspace(self, additional_reads=0):
        cold = additional_reads + sum(p.cold_read and not p.future.done() for p in self._pending.values())
        # The complete-array reader may retain pressure, float64 normalization
        # scratch and an HDF5 slab. Its transient arrays are bounded by the
        # number of active workers, irrespective of the depth of the queue.
        return min(self._workers, cold) * 4 * self._estimate

    def _drop_cpu(self):
        key = next((key for key in self._cpu if key in self._gpu), None) if self._codec.name != "none" else None
        _, size = self._cpu.pop(key) if key is not None else self._cpu.popitem(last=False)[1]
        self._cpu_bytes -= size
        self._metrics["cpu_evictions"] += 1
        return size

    def _host_room(self, extra):
        available = _host_available(self._limits)
        if available is None or available >= self.host_floor + extra:
            return True
        needed = self.host_floor + extra - available
        while needed > 0 and self._cpu:
            needed -= self._drop_cpu()
        available = _host_available(self._limits)
        return available is None or available >= self.host_floor + extra

    def _reserve_cpu(self, size, *, workspace=None):
        if workspace is None:
            workspace = self._read_workspace()
        committed = size + self._pending_bytes + workspace
        if committed > self.cpu_budget:
            return False
        while self._cpu and self._cpu_bytes + committed > self.cpu_budget:
            self._drop_cpu()
        return self._host_room(0)

    def _keep_cpu(self, key, row, size, *, compressed_only=False):
        if key in self._cpu:
            self._cpu.move_to_end(key)
            return
        if not self.cpu_budget:
            return
        if (self._codec.name != "none" and not isinstance(row["target"], _Packed)
                and self._host_room(2 * size)):
            row = self._pack_row(row)
            size = _bytes(row)
        if compressed_only and not isinstance(row["target"], _Packed):
            return
        if self._reserve_cpu(size):
            self._cpu[key] = (row, size)
            self._cpu_bytes += size

    def _evict_gpu(self, *, demote):
        key, (row, size) = self._gpu.popitem(last=False)
        self._gpu_bytes -= size
        self._metrics["gpu_evictions"] += 1
        if not demote or key in self._cpu or not self.cpu_budget:
            return
        room = self._reserve_cpu(size) and self._host_room(3 * size)
        if room:
            start = perf_counter()
            try:
                host = _copy_row(row, torch.device("cpu"))
            except (torch.OutOfMemoryError, MemoryError):
                return  # A clean, immutable row can always be reloaded from its store.
            self._metrics["d2h_s"] += perf_counter() - start
            self._keep_cpu(key, host, size)

    def _device_room(self, size):
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reusable = max(0, torch.cuda.memory_reserved(self.device) - allocated)
        available = free + reusable
        read_share = getattr(torch.cuda, "get_per_process_memory_fraction", None)
        if read_share is not None:
            available = min(available, int(total * read_share(self.device)) - allocated)
        reserve = min(256 << 20, total // 64)
        return available >= size + reserve

    def _consume_pending(self, key, *, wanted):
        pending = self._pending[key]
        start = perf_counter()
        try:
            row, packed = pending.future.result()
        except Exception:
            self._metrics["prefetch_failures"] += 1
            # A failed speculative read or decode is retried from the sealed
            # source in get(). Any error from the authoritative read propagates.
            return None, None
        finally:
            self._pending_bytes -= self._pending.pop(key).reserved_bytes
            self._metrics["prefetch_wait_s"] += perf_counter() - start
        if wanted:
            self._metrics["prefetch_hits"] += 1
            if not pending.cold_read:
                self._metrics["cpu_hits"] += 1
            return row, packed
        saved = packed if packed is not None else row
        self._keep_cpu(key, saved, _bytes(saved), compressed_only=key in self._gpu)
        return None, None

    def _on_device(self, key, host, size):
        if self.device.type == "cpu":
            self._keep_cpu(key, host, size)
            return host
        cacheable = size <= self.gpu_budget
        while self._gpu and ((cacheable and self._gpu_bytes + size > self.gpu_budget) or not self._device_room(size)):
            self._evict_gpu(demote=True)
        start = perf_counter()
        try:
            result = _copy_row(host, self.device)
        except torch.OutOfMemoryError:
            # Unwind the failed partial copy before clearing allocator blocks and retrying.
            result = None
        if result is None:
            self._metrics["gpu_oom_retries"] += 1
            while self._gpu:
                self._evict_gpu(demote=False)
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
            try:
                result = _copy_row(host, self.device)
            except torch.OutOfMemoryError:
                self._keep_cpu(key, host, size)
                raise
        self._metrics["h2d_s"] += perf_counter() - start
        if cacheable:
            self._gpu[key] = (result, size)
            self._gpu_bytes += size
            if self._codec.name == "none":
                self._keep_cpu(key, host, size)
        else:
            self._keep_cpu(key, host, size)
        return result

    def get(self, key):
        start = perf_counter()
        with self._lock:
            try:
                self._metrics["gets"] += 1
                if monotonic() - self._last_pressure_check >= 1.:
                    self._host_room(0)
                    self._last_pressure_check = monotonic()
                if key in self._gpu:
                    self._metrics["gpu_hits"] += 1
                    self._gpu.move_to_end(key)
                    row = self._gpu[key][0]
                else:
                    host, packed = self._consume_pending(key, wanted=True) if key in self._pending else (None, None)
                    if host is None and key in self._cpu:
                        host, size = self._cpu[key]
                        if isinstance(host["target"], _Packed):
                            self._cpu.move_to_end(key)
                        else:
                            self._cpu.pop(key)
                            self._cpu_bytes -= size
                        self._metrics["cpu_hits"] += 1
                    if host is None:
                        self._host_room(4 * self._estimate)
                        host = self._read(key)
                    if isinstance(host["target"], _Packed):
                        self._host_room(host["target"].original_bytes)
                        host = self._unpack_row(host)
                    size = _bytes(host)
                    self._estimate = max(self._estimate, size)
                    if packed is not None:
                        self._keep_cpu(key, packed, _bytes(packed))
                    elif self.device.type == "cuda" and self._codec.name != "none":
                        # Keep the same CPU source in compressed form before
                        # upload so GPU eviction usually needs no D2H copy.
                        self._keep_cpu(key, host, size, compressed_only=True)
                    row = self._on_device(key, host, size)
                if self.device.type == "cuda":
                    stream = torch.cuda.current_stream(self.device)
                    for name in _TENSORS:
                        row[name].record_stream(stream)
                return row
            finally:
                self._metrics["get_wall_s"] += perf_counter() - start

    def prefetch(self, keys):
        """Queue upcoming host reads and decodes up to the depth and memory budgets."""
        if isinstance(keys, (str, bytes)):
            keys = (keys,)
        ordered = tuple(dict.fromkeys(keys))
        wanted = set(ordered)
        with self._lock:
            for key, pending in tuple(self._pending.items()):
                if key not in wanted and pending.future.done():
                    self._consume_pending(key, wanted=False)
            if not self._estimate:
                return False
            submitted = 0
            for key in ordered:
                if len(self._pending) >= self._depth:
                    break
                if key in self._pending or key in self._gpu:
                    continue
                cached = self._cpu.get(key)
                if cached is not None and not isinstance(cached[0]["target"], _Packed):
                    continue
                packed = None
                if cached is not None:
                    # Move the compressed buffer into the reservation. A
                    # worker may keep it alive even while the LRU is trimmed.
                    packed, size = self._cpu.pop(key)
                    self._cpu_bytes -= size
                else:
                    size = self._estimate if self._codec.name != "none" else 0
                new_bytes = size + self._estimate
                workspace = self._read_workspace(additional_reads=int(packed is None))
                if (not self._host_room(new_bytes + self._pending_bytes + workspace)
                        or not self._reserve_cpu(new_bytes, workspace=workspace)):
                    if cached is not None:
                        self._cpu[key] = cached
                        self._cpu_bytes += size
                    break
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="joint-fno-read")
                future = self._executor.submit(self._prepare_row, key, packed)
                self._pending[key] = _Pending(future, new_bytes, packed is None)
                self._pending_bytes += new_bytes
                metric = "prefetch_cold_reads" if packed is None else "prefetch_cpu_decodes"
                self._metrics[metric] += 1
                submitted += 1
            self._metrics["prefetch_reads"] += submitted
            self._metrics["prefetch_high_water"] = max(self._metrics["prefetch_high_water"], len(self._pending))
            return bool(submitted)

    def warm_cpu(self, keys):
        """Fill spare CPU capacity from the supplied keys without evicting or uploading records."""
        added = 0
        reason = "all_keys_visited"
        for key in keys:
            with self._lock:
                if key in self._cpu or key in self._gpu:
                    continue
                if self._pending:
                    raise RuntimeError("prime the CPU tier before starting background prefetch")
                if self._cpu_bytes >= self.cpu_budget:
                    reason = "cpu_budget"
                    break
                available = _host_available(self._limits)
                if available is not None and available < self.host_floor + 3 * self._estimate:
                    reason = "host_headroom"
                    break
                row = self._read(key)
                self._estimate = max(self._estimate, _bytes(row))
                if self._codec.name != "none":
                    row = self._pack_row(row)
                size = _bytes(row)
                if self._cpu_bytes + size > self.cpu_budget:
                    reason = "cpu_budget"
                    break
                self._cpu[key] = (row, size)
                self._cpu_bytes += size
                added += 1
        return {"added": added, "stop_reason": reason, **self.stats()}

    def stats(self):
        with self._lock, self._metric_lock:
            logical = sum(_bytes(row, logical=True) for row, _ in self._cpu.values())
            return {**self._metrics, "device": str(self.device), "gpu_records": len(self._gpu),
                    "cpu_records": len(self._cpu), "gpu_bytes": self._gpu_bytes, "cpu_bytes": self._cpu_bytes,
                    "cpu_logical_bytes": logical, "cpu_compression_ratio": logical / self._cpu_bytes if self._cpu_bytes else 1.,
                    "compression_requested": self._requested_codec, "compression_used": self._codec.name,
                    "compressed_cpu_records": sum(isinstance(row["target"], _Packed) for row, _ in self._cpu.values()),
                    "cpu_gpu_overlap_records": sum(key in self._gpu for key in self._cpu),
                    "distinct_cached_records": len(self._cpu.keys() | self._gpu.keys()),
                    "gpu_budget_bytes": self.gpu_budget, "cpu_budget_bytes": self.cpu_budget,
                    "host_floor_bytes": self.host_floor, "host_available_bytes": _host_available(self._limits),
                    "prefetch_pending": bool(self._pending), "prefetch_pending_records": len(self._pending),
                    "pending_cpu_bytes": self._pending_bytes, "read_workspace_bytes": self._read_workspace()}

    def clear(self):
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
            self._pending.clear()
            self._pending_bytes = 0
            self._cpu.clear()
            self._gpu.clear()
            self._cpu_bytes = self._gpu_bytes = 0
