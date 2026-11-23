"""Immutable weight-only checkpoint history and acknowledged mirror queue.

record() must complete BEFORE a driver's fixed latest/best path is overwritten.
The full resume checkpoint remains the driver's responsibility. This module
never deletes pending weights or research files: the mirror alone releases a
pending file after its persistent local copy has passed SHA-256 verification.
"""

from collections import OrderedDict
from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
from time import monotonic, sleep, time
import uuid

import torch


SCHEMA = "SCNO_CHECKPOINT_WEIGHT_ARCHIVE_V1"
ACK_SCHEMA = "SCNO_CHECKPOINT_WEIGHT_ARCHIVE_ACK_V1"
_FORBIDDEN = {
    "optimizer", "scheduler", "stage_schedule", "scaler", "grad_scaler",
    "sampler", "frame_sampler", "rng", "rng_cpu", "rng_cuda", "rng_by_rank",
}


class ArchiveBackpressure(TimeoutError):
    """No weight/event was discarded; retry the SAME snapshot after space frees."""


def _snapshot(value, memo):
    if torch.is_tensor(value):
        key = id(value)
        if key not in memo:
            if value.layout != torch.strided:
                raise TypeError("weight archives require the model's dense strided state tensors")
            memo[key] = value.detach().to(device="cpu", copy=True, memory_format=torch.preserve_format)
        return memo[key]
    if isinstance(value, dict):
        result = OrderedDict() if isinstance(value, OrderedDict) else {}
        result.update((key, _snapshot(content, memo)) for key, content in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = _snapshot(value._metadata, memo)
        return result
    if isinstance(value, list):
        return [_snapshot(content, memo) for content in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(content, memo) for content in value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    raise TypeError(f"unsupported non-weight checkpoint metadata type: {type(value).__name__}")


def model_state_sha256(state):
    """Stable name/shape/dtype/logical tensor bytes; no torch.save metadata."""
    if not isinstance(state, dict) or not state:
        raise ValueError("weight_payload['model'] must be a nonempty model state_dict")
    digest = hashlib.sha256(b"SCNO_MODEL_STATE_SHA256_V1\0")
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not torch.is_tensor(value) or value.layout != torch.strided:
            raise TypeError("model state entries must be named dense Tensor parameters/buffers")
        value = value.detach().cpu().resolve_conj().resolve_neg().contiguous()
        header = json.dumps(
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True, separators=(",", ":")).encode()
        data = memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(data).to_bytes(8, "big"))
        for first in range(0, len(data), 8 << 20):
            digest.update(data[first:first + (8 << 20)])
    return digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _fsync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_new(source, destination):
    # Windows rename refuses an existing target and works in managed sandboxes
    # that prohibit hard links. POSIX rename could overwrite, so use link there.
    if os.name == "nt":
        os.rename(source, destination)
    else:
        os.link(source, destination)


class CheckpointWeightArchive:
    """Append every event; deduplicate immutable bytes only at identical update/model SHA.

    Default low-space policy is indefinite backpressure, never dropping weights.
    poll_seconds/wait_timeout_s/disk_usage are injectable for controlled CPU tests.
    A configured timeout raises ArchiveBackpressure before an event is committed.
    """

    def __init__(
            self, run_root, *, minimum_free_bytes=512 << 20, poll_seconds=2.0,
            wait_timeout_s=None, disk_usage=None, sleep_function=None):
        self.run_root = Path(run_root).resolve()
        self.pending = self.run_root / "checkpoint_archive_pending"
        self.acked = self.run_root / "checkpoint_archive_acked"
        self.history = self.run_root / "checkpoint_history"
        self.artifacts = self.history / "artifacts"
        for directory in (self.pending, self.acked, self.history, self.artifacts):
            directory.mkdir(parents=True, exist_ok=True)
        if (type(minimum_free_bytes) is not int or minimum_free_bytes < 0
                or not math.isfinite(poll_seconds) or poll_seconds < 0
                or (wait_timeout_s is not None and (not math.isfinite(wait_timeout_s) or wait_timeout_s < 0))):
            raise ValueError("space reserve, poll interval and optional timeout must be nonnegative")
        self.minimum_free_bytes = minimum_free_bytes
        self.poll_seconds = poll_seconds
        self.wait_timeout_s = wait_timeout_s
        self._disk_usage = disk_usage or shutil.disk_usage
        self._sleep = sleep_function or sleep
        self._thread_lock = threading.RLock()

    @contextmanager
    def _lock(self):
        with self._thread_lock:
            path = self.history / ".archive.lock"
            with path.open("a+b") as handle:
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    while True:
                        try:
                            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                            break
                        except OSError:
                            self._sleep(max(.01, self.poll_seconds))
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    handle.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _json_publish(self, path, value, *, immutable):
        encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        temporary = path.with_name(".archive_json_" + uuid.uuid4().hex + ".part")
        try:
            with temporary.open("xb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            if immutable:
                try:
                    _publish_new(temporary, path)
                except FileExistsError:
                    if _read_json(path) != value:
                        raise RuntimeError(f"refusing to overwrite immutable archive metadata: {path}")
            else:
                os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _append_jsonl(self, path, value):
        line = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(line)
            output.flush()
            os.fsync(output.fileno())

    def _valid_ack(self, metadata):
        marker = _read_json(self.acked / (metadata["filename"] + ".json"))
        return bool(marker and marker.get("schema") == ACK_SCHEMA and marker.get("verified") is True
                    and marker.get("filename") == metadata["filename"]
                    and marker.get("model_sha256") == metadata["model_sha256"]
                    and marker.get("file_sha256") == metadata["file_sha256"]
                    and marker.get("local_sha256") == metadata["file_sha256"]
                    and marker.get("size_bytes") == metadata["size_bytes"]
                    and isinstance(marker.get("local_path"), str) and marker["local_path"])

    def _manifest(self, *, state="READY", extra=None):
        pending, acknowledged_present, unindexed = [], [], []
        for path in sorted(self.pending.glob("weights_u*_*.pt")):
            metadata = _read_json(self.artifacts / (path.name + ".json"))
            if metadata is None:
                unindexed.append({"filename": path.name, "path": str(path), "size_bytes": path.stat().st_size})
            elif self._valid_ack(metadata):
                acknowledged_present.append(metadata)
            else:
                pending.append(metadata)
        result = {
            "schema": SCHEMA, "state": state, "run_root": str(self.run_root),
            "pending_count": len(pending) + len(unindexed),
            "pending": pending, "unindexed_pending": unindexed,
            "acked_pending_files_present": acknowledged_present,
            "pending_files_present": len(pending) + len(unindexed) + len(acknowledged_present),
            "pending_queue_bytes": sum(item["size_bytes"] for item in pending + unindexed + acknowledged_present),
            "artifact_count": sum(1 for _ in self.artifacts.glob("weights_u*_*.pt.json")),
            "events_path": str(self.history / "events.jsonl"),
            "release_owner": "mirror, after persistent local SHA verification",
            **(extra or {}),
        }
        self._json_publish(self.history / "MANIFEST.json", result, immutable=False)
        self._append_jsonl(self.history / "manifest.jsonl", result)
        return result

    def status(self):
        with self._lock():
            return self._manifest()

    def _wait_for_space(self, required):
        started, waits = monotonic(), 0
        while True:
            free = self._disk_usage(self.pending).free
            if free >= required + self.minimum_free_bytes:
                return waits
            self._manifest(state="WAITING_FOR_SPACE", extra={
                "required_weight_bytes_estimate": required, "free_bytes": free,
                "minimum_free_bytes": self.minimum_free_bytes, "backpressure_polls": waits})
            if self.wait_timeout_s is not None and monotonic() - started >= self.wait_timeout_s:
                raise ArchiveBackpressure("archive queue space is insufficient; snapshot not dropped; retry after mirror frees space")
            waits += 1
            self._sleep(self.poll_seconds)

    def record(self, weight_payload, *, event, update, epoch, record_cursor, phase):
        if not isinstance(weight_payload, dict) or _FORBIDDEN & weight_payload.keys():
            raise ValueError("archive model/config/bindings only; exclude optimizer, scheduler, RNG and sampler state")
        if type(update) is not int or update < 0 or not isinstance(event, str) or not event:
            raise ValueError("supply a nonnegative update and a nonempty event name")
        # Validate index fields before writing anything or copying large states.
        json.dumps({"event": event, "update": update, "epoch": epoch,
                    "record_cursor": record_cursor, "phase": phase}, allow_nan=False)
        tensors = {}
        snapshot = _snapshot(weight_payload, tensors)
        model_digest = model_state_sha256(snapshot.get("model"))
        filename = f"weights_u{update:07d}_{model_digest}.pt"
        destination = self.pending / filename
        metadata_path = self.artifacts / (filename + ".json")
        estimate = sum(value.numel() * value.element_size() for value in tensors.values())
        estimate += max(1 << 20, estimate // 10)
        reused = False
        waits = 0
        with self._lock():
            metadata = _read_json(metadata_path)
            if metadata is not None:
                if metadata.get("filename") != filename or metadata.get("model_sha256") != model_digest:
                    raise RuntimeError("immutable artifact identity conflicts with the current weights")
                if self._valid_ack(metadata):
                    reused = True
                elif destination.exists():
                    try:
                        stat = destination.stat()
                        if stat.st_size != metadata["size_bytes"]:
                            raise RuntimeError("immutable pending artifact size changed")
                        if stat.st_mtime_ns != metadata.get("pending_mtime_ns") and _file_sha256(destination) != metadata["file_sha256"]:
                            raise RuntimeError("immutable pending artifact contents changed")
                    except FileNotFoundError:
                        # The mirror may ACK+unlink between exists/stat/hash.
                        if not self._valid_ack(metadata):
                            raise RuntimeError("pending artifact disappeared without its verified ACK")
                    reused = True
                else:
                    raise RuntimeError("pending weights disappeared without a matching verified ACK; refusing silent reconstruction")
            elif destination.exists():
                # Crash recovery after weight publication but before index
                # publication. Adopt only the already immutable matching bytes.
                recovered = torch.load(destination, map_location="cpu", weights_only=True)
                if _FORBIDDEN & recovered.keys():
                    raise RuntimeError("existing filename contains a full training state, not a weight-only archive")
                if model_state_sha256(recovered.get("model")) != model_digest:
                    raise RuntimeError("unindexed existing artifact contains different weights")
                metadata = self._artifact_metadata(destination, model_digest)
                self._json_publish(metadata_path, metadata, immutable=True)
                reused = True
            else:
                waits += self._wait_for_space(estimate)
                while True:
                    temporary = self.pending / (".archive_weight_" + uuid.uuid4().hex + ".part")
                    try:
                        with temporary.open("xb") as output:
                            torch.save(snapshot, output)
                            output.flush()
                            os.fsync(output.fileno())
                        # A hard link publishes without EVER replacing an
                        # existing immutable filename; .part is not mirrorable.
                        _publish_new(temporary, destination)
                        _fsync_directory(self.pending)
                        break
                    except (OSError, RuntimeError) as exc:
                        # PyTorch's zip writer may report ENOSPC as RuntimeError.
                        low_space = self._disk_usage(self.pending).free < estimate + self.minimum_free_bytes
                        retry_space = (
                            isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT)
                        ) or (
                            isinstance(exc, RuntimeError) and low_space and any(
                                marker in str(exc).lower()
                                for marker in ("file write failed", "unexpected pos", "pytorchstreamwriter"))
                        )
                        if not retry_space:
                            raise
                        if temporary.exists():
                            temporary.unlink()
                        waits += self._wait_for_space(estimate)
                    finally:
                        if temporary.exists():
                            temporary.unlink()
                metadata = self._artifact_metadata(destination, model_digest)
                self._json_publish(metadata_path, metadata, immutable=True)
            receipt = {
                "schema": SCHEMA, "event_id": uuid.uuid4().hex,
                "event": event, "update": update, "epoch": epoch,
                "record_cursor": record_cursor, "phase": phase,
                "filename": filename, "model_sha256": model_digest,
                "file_sha256": metadata["file_sha256"], "size_bytes": metadata["size_bytes"],
                "path": str(destination), "artifact_metadata_path": str(metadata_path),
                "ack_path": metadata["ack_path"], "immutable_bytes_reused": reused,
                "already_acknowledged": self._valid_ack(metadata),
                "backpressure_polls": waits, "recorded_unix_time": time(),
                "weight_only": True,
            }
            self._append_jsonl(self.history / "events.jsonl", receipt)
            self._manifest()
            return receipt

    def _artifact_metadata(self, path, model_digest):
        stat = path.stat()
        return {
            "schema": SCHEMA, "filename": path.name, "model_sha256": model_digest,
            "file_sha256": _file_sha256(path), "size_bytes": stat.st_size,
            "pending_mtime_ns": stat.st_mtime_ns, "path": str(path),
            "ack_path": str(self.acked / (path.name + ".json")),
        }


def run_cpu_checks():
    """Real tensor bytes, dedup, changed weights, ACK reuse and space backpressure."""
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory(prefix="cwa_", dir=Path.cwd()) as directory:
        assert Path(directory).resolve().parent == Path.cwd().resolve()
        root = Path(directory)
        archive = CheckpointWeightArchive(root / "normal", minimum_free_bytes=0, poll_seconds=0)
        payload = {"model": OrderedDict([
            ("linear.weight", torch.arange(12, dtype=torch.float32).reshape(3, 4)),
            ("residual_gain", torch.tensor(1.)),
            ("counter", torch.tensor(19, dtype=torch.int64)),
            ("mask", torch.tensor([True, False]))]), "config": {"family": "cpu_test"}}
        payload["model"]["fno.complex_weight"] = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64)
        payload["model"]["empty_buffer"] = torch.empty(0)
        first = archive.record(payload, event="regular", update=12, epoch=1, record_cursor=4, phase="A")
        path = Path(first["path"])
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert all(torch.equal(value, loaded["model"][name]) for name, value in payload["model"].items())
        first_bytes, first_mtime = path.read_bytes(), path.stat().st_mtime_ns
        second = archive.record(payload, event="best_uniform", update=12, epoch=1, record_cursor=4, phase="A")
        assert second["filename"] == first["filename"] and second["immutable_bytes_reused"]
        assert path.read_bytes() == first_bytes and path.stat().st_mtime_ns == first_mtime
        payload["model"]["linear.weight"].add_(.5)
        changed = archive.record(payload, event="best_layered", update=12, epoch=1, record_cursor=4, phase="A")
        assert changed["filename"] != first["filename"] and path.read_bytes() == first_bytes
        # Simulate the mirror's persistent copy, verified ACK, and source release.
        local = root / "persistent_copy.pt"
        shutil.copyfile(changed["path"], local)
        marker = {
            "schema": ACK_SCHEMA, "verified": True, "filename": changed["filename"],
            "model_sha256": changed["model_sha256"], "file_sha256": changed["file_sha256"],
            "local_sha256": _file_sha256(local), "size_bytes": local.stat().st_size,
            "local_path": str(local)}
        Path(changed["ack_path"]).write_text(json.dumps(marker), encoding="utf-8")
        pending_to_release = Path(changed["path"]).resolve()
        assert pending_to_release.parent == archive.pending.resolve()
        pending_to_release.unlink()
        acked = archive.record(payload, event="regular_again", update=12, epoch=1, record_cursor=4, phase="A")
        assert acked["already_acknowledged"] and not Path(acked["path"]).exists()
        assert _file_sha256(local) == changed["file_sha256"]
        events = [json.loads(line) for line in (archive.history / "events.jsonl").read_text().splitlines()]
        assert len(events) == 4 and len({item["event_id"] for item in events}) == 4
        assert len(list(archive.artifacts.glob("*.json"))) == 2
        polls = []

        def space(_):
            polls.append(True)
            return SimpleNamespace(free=0 if len(polls) < 3 else 100 << 20)

        limited = CheckpointWeightArchive(
            root / "wait", minimum_free_bytes=0, poll_seconds=0,
            disk_usage=space, sleep_function=lambda _: None)
        waited = limited.record(payload, event="regular", update=13, epoch=1, record_cursor=8, phase="A")
        assert waited["backpressure_polls"] == 2 and Path(waited["path"]).is_file()
        blocked = CheckpointWeightArchive(
            root / "blocked", minimum_free_bytes=0, poll_seconds=0, wait_timeout_s=0,
            disk_usage=lambda _: SimpleNamespace(free=0))
        try:
            blocked.record(payload, event="regular", update=14, epoch=1, record_cursor=12, phase="A")
        except ArchiveBackpressure:
            pass
        else:
            raise AssertionError("low-space archive falsely committed a checkpoint")
        assert not list(blocked.pending.glob("*.pt")) and not (blocked.history / "events.jsonl").exists()
        assert _read_json(blocked.history / "MANIFEST.json")["state"] == "WAITING_FOR_SPACE"
        try:
            archive.record({**payload, "optimizer": {}},
                           event="bad", update=15, epoch=1, record_cursor=16, phase="A")
        except ValueError:
            pass
        else:
            raise AssertionError("optimizer state entered a weight-only archive")
        return {
            "passed": True, "real_tensor_save_reload_exact": True,
            "scalar_bool_and_integer_state_digest_supported": True,
            "complex_fno_weights_and_empty_buffers_supported": True,
            "same_step_same_weights_shared_immutable_bytes": True,
            "same_step_changed_weights_new_file": True, "one_append_per_event": True,
            "verified_ack_prevents_recreation": True, "backpressure_waits_without_dropping": True,
            "low_space_timeout_commits_no_false_event": True, "optimizer_rejected": True,
            "archive_never_deletes_pending_weights": True,
        }


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--cpu-check"]:
        print(json.dumps(run_cpu_checks(), indent=2))
    else:
        print("Import CheckpointWeightArchive; --cpu-check uses only a temporary CPU archive.")
