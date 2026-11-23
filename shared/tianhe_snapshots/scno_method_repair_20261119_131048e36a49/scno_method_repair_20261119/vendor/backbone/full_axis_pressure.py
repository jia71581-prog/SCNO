"""Full-field supervision on the manuscript's uniform 401-frame axis."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch


GRID = (201, 201)
FULL_TIMES_S = np.arange(401, dtype=np.float64) * .0025


def axis_with_initial_condition(times):
    supplied = np.asarray(times, dtype=np.float64)
    if supplied.shape == FULL_TIMES_S.shape and np.allclose(
            supplied, FULL_TIMES_S, rtol=0, atol=6e-8):
        return FULL_TIMES_S.copy(), False
    positive_times = FULL_TIMES_S[1:]
    if supplied.shape == positive_times.shape and np.allclose(
            supplied, positive_times, rtol=0, atol=6e-8):
        return FULL_TIMES_S.copy(), True
    raise ValueError("expected the 2.5 ms uniform axis from 0 to 1 s, with only t=0 optional")


def validate_pressure_frames(fields, insert_initial):
    expected = len(FULL_TIMES_S) - int(insert_initial)
    if fields.shape != (expected, *GRID) or not np.issubdtype(fields.dtype, np.floating):
        raise ValueError(f"expected {expected} complete 201 x 201 pressure frames in Pa")
    if not insert_initial and np.any(fields[0]):
        raise ValueError("pressure at t=0 must satisfy the zero initial condition")


def pressure_at_slots(fields, slots, insert_initial):
    if not insert_initial:
        return fields[slots]
    selected = np.zeros((len(slots), *GRID), dtype=fields.dtype)
    stored = slots > 0
    selected[stored] = fields[slots[stored] - 1]
    return selected


class FullAxisPressure:
    """Read train-only, complete pressure trajectories in Pa from a manifest."""

    def __init__(self, manifest, model):
        manifest = Path(manifest).resolve()
        serialized = manifest.read_bytes()
        data = json.loads(serialized)
        parent = manifest.parent
        axis = (parent / data["times_s"]).resolve()
        self.times, self.insert_initial = axis_with_initial_condition(
            np.load(axis, allow_pickle=False))
        self.model_scale = float(model.normalizer.metadata.pressure_scale_pa)
        if not np.isfinite(self.model_scale) or self.model_scale <= 0:
            raise ValueError("the model pressure scale must be finite and positive")
        records = data["records"]
        self.ids = [record["sample_id"] for record in records]
        if not self.ids or any(not isinstance(name, str) or not name for name in self.ids):
            raise ValueError("the manifest requires nonempty TRAIN record IDs")
        if len(set(self.ids)) != len(self.ids) or any(record.get("split") != "train" for record in records):
            raise ValueError("the manifest must contain unique TRAIN records")
        self.records = []
        files = []
        for record in records:
            velocity = (parent / record["velocity_mps"]).resolve()
            pressure = (parent / record["pressure_pa"]).resolve()
            source = np.asarray(record["source"], dtype=np.float32)
            if (source.shape != (5,) or not np.isfinite(source).all()
                    or source[2] <= 0 or source[4] == 0):
                raise ValueError("each source requires [x,z,f0,t0,A] with positive f0 and nonzero A")
            self.records.append((velocity, source, pressure))
            files.append([self.file_stamp(velocity), self.file_stamp(pressure)])
        self.binding = {
            "protocol": "uniform401",
            "manifest_path": str(manifest),
            "manifest_sha256": sha256(serialized).hexdigest(),
            "axis_sha256": sha256(axis.read_bytes()).hexdigest(),
            "insert_initial": self.insert_initial,
            "arrays": files,
        }
        self.reference_energies = {}

    @staticmethod
    def file_stamp(path):
        stat = path.stat()
        return [str(path), stat.st_size, stat.st_mtime_ns]

    def batch(self, record, slots):
        slots = np.asarray(slots, dtype=np.int64)
        if (slots.ndim != 1 or not slots.size or slots[0] < 0
                or slots[-1] >= self.times.size or not np.all(np.diff(slots) > 0)):
            raise ValueError("expected unique, ordered, full-axis frame slots")
        velocity_path, source, pressure_path = self.records[record]
        if [self.file_stamp(velocity_path), self.file_stamp(pressure_path)] != self.binding["arrays"][record]:
            raise ValueError("a bound training array changed after the manifest was loaded")
        velocity = np.load(velocity_path, allow_pickle=False)
        fields = np.load(pressure_path, mmap_mode="r", allow_pickle=False)
        if (velocity.shape != GRID or not np.issubdtype(velocity.dtype, np.floating)
                or not np.isfinite(velocity).all() or not np.all(velocity > 0)):
            raise ValueError(f"{velocity_path}: expected a finite positive 201 x 201 velocity grid")
        validate_pressure_frames(fields, self.insert_initial)
        scale = self.model_scale * float(source[4])
        if record not in self.reference_energies:
            # The initial frame contributes zero; use identical sums for both layouts.
            positive_fields = fields if self.insert_initial else fields[1:]
            value = 0.
            for first in range(0, len(positive_fields), 16):
                part = np.asarray(positive_fields[first:first + 16], dtype=np.float64) / scale
                value += float(np.square(part).sum())
            if not np.isfinite(value):
                raise ValueError(f"{pressure_path}: nonfinite full-axis reference energy")
            self.reference_energies[record] = value
        pressure = pressure_at_slots(fields, slots, self.insert_initial)
        targets = (pressure.astype(np.float64) / scale).astype(np.float32)
        if not np.isfinite(targets).all():
            raise ValueError(f"{pressure_path}: selected pressure fields are nonfinite")
        return {
            "velocity": torch.from_numpy(velocity.astype(np.float32))[None, None],
            "source": torch.from_numpy(source.copy())[None],
            "times": torch.from_numpy(self.times[slots].astype(np.float32)),
            "target": torch.from_numpy(targets),
            "full_reference_energy": self.reference_energies[record],
        }
