"""Sealed SCNO data, full-record pressure energy, and V2 stratified supervision."""

from __future__ import annotations
from scno_release_paths import local_asset as _scno_local_asset

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile

import torch
import h5py
import numpy as np


GRID = (201, 201)
SPLITS = ("train", "validation", "test")
ROOTS = {
    "style_marm_joint201": _scno_local_asset('project/data/style_marm_joint201_20261025'),
    "centered_uniform_layered": (
        _scno_local_asset('package/data/uniform_layered_centered1000_20261118')
    ),
}
ENERGY_SCHEMA = "SCNO_V2_FULL_RECORD_NORMALIZED_PRESSURE_SSE"


def _file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    return sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _artifact(root, marker, name):
    path = root / name
    proof = marker["artifacts"][name]
    expected = proof["sha256"] if isinstance(proof, dict) else proof
    if _file_sha(path) != expected:
        raise ValueError(f"published artifact checksum changed: {path}")
    return path, expected


def _check_axis(value, length, endpoint, dt):
    times = np.array(value, dtype=np.float64, copy=True)
    if (times.shape != (length,) or not np.isfinite(times).all() or times[0] != 0.0
            or not math.isclose(times[-1], endpoint, rel_tol=0., abs_tol=5e-8)
            or not np.allclose(np.diff(times), dt, rtol=0., atol=5e-7)):
        raise ValueError("published pressure times do not match the declared physical axis")
    times.flags.writeable = False
    return times


def _source(value):
    source = np.array(value, dtype=np.float32, copy=True)
    if (source.shape != (5,) or not np.isfinite(source).all()
            or not np.all((0 <= source[:2]) & (source[:2] <= 2000))
            or source[2] <= 0 or source[3] < 0):
        raise ValueError("expected physical source [x m, z m, frequency Hz, delay s, amplitude]")
    source.flags.writeable = False
    return source


def normalize_pressure_pa(pressure, amplitude, sigma_u):
    """Apply the fixed reference pressure unit once; the zero-source target is zero."""
    scale = float(amplitude) * float(sigma_u)
    if not (math.isfinite(scale) and math.isfinite(sigma_u) and sigma_u > 0):
        raise ValueError("invalid source-dependent physical pressure unit")
    raw = np.asarray(pressure)
    if not np.isfinite(raw).all():
        raise ValueError("reference pressure contains nonfinite values")
    if amplitude == 0:
        if np.any(raw):
            raise ValueError("zero-amplitude source has nonzero physical pressure labels")
        return np.zeros(raw.shape, dtype=np.float32)
    if scale == 0:
        raise ValueError("nonzero amplitude underflowed the physical pressure unit")
    normalized = (raw.astype(np.float64) / scale).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized pressure is nonfinite")
    return normalized


def restore_pressure_pa(field, amplitude, sigma_u):
    """Recover physical Pa from a normalized prediction."""
    return field * (float(amplitude) * float(sigma_u))


def _source_key(source):
    return tuple(map(float, source))


@dataclass(frozen=True)
class _Record:
    information: dict
    source: np.ndarray
    pressure_path: Path
    pressure_stamp: tuple
    velocity_path: Path | None = None
    velocity_stamp: tuple | None = None


def _id_subset(published, requested, split):
    if requested is None:
        return tuple(published)
    if not isinstance(requested, list):
        raise ValueError(f"{split}_ids must be a list of published record identifiers")
    ids = tuple(map(str, requested))
    if len(set(ids)) != len(ids) or not set(ids).issubset(published):
        raise ValueError(f"{split}_ids must be distinct members of the published {split} split")
    return ids


class Corpus:
    """Read pressure targets by sealed sample ID and physical time slot."""

    def __init__(self, config):
        self.kind = config["kind"]
        if self.kind not in ROOTS:
            raise ValueError(f"unsupported SCNO corpus {self.kind!r}")
        self.root = Path(config.get("root", ROOTS[self.kind])).resolve(strict=True)
        self._bound = {}
        self._view_paths = {}
        self._view_indices = {}
        marker_path = self.root / "DATASET_COMPLETE.json"
        marker = _read_json(marker_path)
        interpolation_key = ("physical_frame_interpolation" if self.kind == "centered_uniform_layered"
                             else "pressure_interpolation")
        if marker["status"] != "COMPLETE" or marker[interpolation_key]:
            raise ValueError("a complete release with exact physical pressure frames is required")
        self._artifacts = {str(marker_path): _file_sha(marker_path)}
        roster = _read_json(self._artifact(marker, "RECORDS.json"))
        if self.kind == "style_marm_joint201":
            if marker["schema"] != "STYLE_MARM_JOINT201_V1" or marker["training_grid"] != list(GRID):
                raise ValueError("the joint corpus publication contract changed")
            normalizer = _read_json(self._artifact(marker, "normalization_stats_train.json"))
            if normalizer["validation_test_labels_used"] is not False:
                raise ValueError("joint normalization must use the published TRAIN records")
            split_manifest = _read_json(self._artifact(marker, "SPLITS.json"))
            for split in SPLITS:
                rows = [r for r in roster if r["split"] == split]
                if len(rows) != marker[f"{split}_records"]:
                    raise ValueError(f"published {split} size disagrees with its roster")
                self._set_view(
                    marker, split, [str(r["record_id"]) for r in rows],
                    length=801, endpoint=4., dt=.005,
                )
                self._check_joint_split_manifest(split_manifest, split, rows)
                for row in rows:
                    self._add_record(str(row["record_id"]), row, self.root / row["source_h5"])
            if config.get("validation_manifest") or config.get("test_manifest"):
                raise ValueError("the joint release supplies its own fixed validation and test splits")
            self.metadata = {
                **normalizer, "pressure_scale_pa": normalizer["pressure_rms"],
                "source_scales": [2000., 2000., 50., float(self.times[-1]), 1.],
                "algorithm": "joint_TRAIN_only_velocity_mean_std_pressure_rms",
            }
        else:
            expected = {"schema": "SCNO_CENTERED_UL1000_V1", "families": {"uniform": 500, "layered": 500}}
            if any(marker[key] != value for key, value in expected.items()):
                raise ValueError("the centered uniform/layered publication contract changed")
            if len(roster) != 1000 or any(row["split"] != "train" for row in roster):
                raise ValueError("the centered release must contain its published 1,000 TRAIN records")
            normalizer = _read_json(self._artifact(marker, "normalization_ul.json"))
            self._set_view(marker, "train", [str(r["record_id"]) for r in roster],
                           length=401, endpoint=1., dt=.0025)
            for row in roster:
                self._add_record(row["sample_id"], row, Path(row["pressure_record"]),
                                 (row["size_bytes"], row["mtime_ns"]))
            for split in ("validation", "test"):
                if config.get(f"{split}_manifest"):
                    self._add_holdout(Path(config[f"{split}_manifest"]), split)
            self.metadata = {
                **normalizer, "velocity_mean_mps": normalizer["velocity_center_mps"],
                "velocity_std_mps": normalizer["velocity_scale_mps"],
            }
        self._check_metadata()
        self.records = {key: rec.information for key, rec in self._bound.items()}
        published = {split: tuple(key for key, row in self.records.items() if row["split"] == split)
                     for split in SPLITS}
        self.train_ids, self.validation_ids, self.test_ids = (
            _id_subset(published[split], config.get(f"{split}_ids"), split) for split in SPLITS
        )
        if len(self.train_ids) < 4:
            raise ValueError("SCNO V2 requires at least four distinct TRAIN records")
        files = {
            key: {"file": str(record.pressure_path), "stamp": list(record.pressure_stamp),
                  "velocity": str(record.velocity_path) if record.velocity_path else None,
                  "velocity_stamp": list(record.velocity_stamp) if record.velocity_stamp else None}
            for key, record in self._bound.items()
        }
        self.binding = {"kind": self.kind, "root": str(self.root), "published_sha256": self._artifacts,
                        "actual_times_sha256": sha256(self.times.astype("<f8").tobytes()).hexdigest(),
                        "source_file_metadata_sha256": _json_sha(files),
                        "selected_ids": dict(zip(SPLITS, map(list, (self.train_ids, self.validation_ids,
                                                                     self.test_ids)))),
                        "normalization_sha256": _json_sha(self.metadata)}

    def _artifact(self, marker, name):
        path, digest = _artifact(self.root, marker, name)
        self._artifacts[str(path)] = digest
        return path

    def _set_view(self, marker, split, expected_ids, *, length, endpoint, dt):
        path = self._artifact(marker, f"dataset_{split}.h5")
        with h5py.File(path, "r") as view:
            if (view["pressure201"].shape != (len(expected_ids), length, *GRID)
                    or view["pressure201"].dtype != np.float32
                    or view["velocity201"].shape != (len(expected_ids), *GRID)
                    or view["source_parameters"].shape != (len(expected_ids), 5)):
                raise ValueError(f"invalid pressure, velocity or source shape in {path}")
            if list(map(str, view["record_ids"][:])) != expected_ids:
                raise ValueError(f"VDS row order disagrees with the published {split} roster")
            for axis in ("x_m", "z_m"):
                np.testing.assert_array_equal(view[axis][:], np.arange(201) * 10)
            times = _check_axis(view["time_field_s"][:], length, endpoint, dt)
        if split == "train":
            self.times = times
        else:
            np.testing.assert_array_equal(times, self.times)
        self._view_paths[split] = path
        self._view_indices[split] = dict(zip(expected_ids, range(len(expected_ids))))

    @staticmethod
    def _check_joint_split_manifest(manifest, split, rows):
        # The release lists each split separately. Check its global IDs if given
        # as a flat list; VDS row and complete-marker checks always apply.
        published = manifest.get(split)
        if isinstance(published, list) and all(isinstance(v, (int, str)) for v in published):
            if list(map(str, published)) != [str(row["record_id"]) for row in rows]:
                raise ValueError(f"the joint release's {split} ID manifest disagrees with its roster")

    def _add_record(self, key, raw, path, expected_stamp=None, velocity_path=None):
        if key in self._bound or raw["split"] not in SPLITS:
            raise ValueError(f"duplicate record or unrecognized split: {key}")
        path = path.resolve(strict=True)
        stamp = _stamp(path)
        if expected_stamp is not None and stamp != expected_stamp:
            raise ValueError(f"published pressure source file changed: {path}")
        if velocity_path is not None:
            velocity_path = velocity_path.resolve(strict=True)
        info = {**raw, "id": key, "family": str(raw["family"])}
        self._bound[key] = _Record(info, _source(raw["source_parameters"]), path, stamp,
                                   velocity_path, _stamp(velocity_path) if velocity_path else None)

    def _add_holdout(self, path, split):
        path = path.resolve(strict=True)
        manifest = _read_json(path)
        if manifest["split"] != split:
            raise ValueError("the explicit held-out manifest must declare its existing split")
        self._artifacts[str(path)] = _file_sha(path)
        time_path = (path.parent / manifest["times_s"]).resolve(strict=True)
        np.testing.assert_array_equal(np.load(time_path, allow_pickle=False), self.times)
        self._artifacts[str(time_path)] = _file_sha(time_path)
        for row in manifest["records"]:
            if row["split"] != split or not row.get("velocity_sha256") or not row.get("medium_id"):
                raise ValueError("held-out records require their published split and medium identity")
            self._add_record(str(row["record_id"]), row, path.parent / row["pressure_pa"],
                             velocity_path=path.parent / row["velocity_mps"])

    def _check_metadata(self):
        stats = self.metadata
        if (not math.isfinite(stats["velocity_mean_mps"])
                or any(not math.isfinite(stats[k]) or stats[k] <= 0
                       for k in ("velocity_std_mps", "pressure_scale_pa"))
                or len(stats["source_scales"]) != 5
                or any(not math.isfinite(value) or value <= 0 for value in stats["source_scales"])):
            raise ValueError("the published velocity, source and pressure normalization is invalid")

    def _check_source_file(self, rec):
        if _stamp(rec.pressure_path) != rec.pressure_stamp:
            raise ValueError(f"bound pressure file changed: {rec.pressure_path}")
        if rec.velocity_path and _stamp(rec.velocity_path) != rec.velocity_stamp:
            raise ValueError(f"bound velocity file changed: {rec.velocity_path}")

    def _native_metadata(self, handle):
        if self.kind == "style_marm_joint201":
            native = handle["pressure401"]
            if native.shape != (len(self.times), 401, 401) or native.dtype != np.float32:
                raise ValueError("invalid source pressure401 shape or dtype")
            np.testing.assert_array_equal(handle["time_field_s"][:], self.times)
            coarse = handle["velocity401"][::2, ::2]
        else:
            native = handle["wavefield"]
            if native.shape != (801, *GRID) or native.dtype != np.float32:
                raise ValueError("invalid native centered pressure shape or dtype")
            coarse = handle["velocity201"][:]
        source = handle["source_parameters"][:].astype(np.float32).reshape(-1)
        return np.asarray(coarse, dtype=np.float32), source, native

    def _read_block_pressure(self, native, first, count):
        """Copy exact registered nodes using contiguous HDF5 reads."""
        # The full Style A/Marmousi source files store contiguous 401² slabs.
        # Decimate in memory to avoid HDF5 point selection over every matrix.
        if self.kind == "style_marm_joint201":
            pressure = np.empty((count, *GRID), dtype=native.dtype)
            for k in range(0, count, 32):
                end = min(k + 32, count)
                pressure[k:end] = native[first + k:first + end, :, :][:, ::2, ::2]
            return pressure
        # The centered records are compressed in one-frame chunks. Read only
        # each registered frame, avoiding the strided HDF5 output selection.
        pressure = np.empty((count, *GRID), dtype=native.dtype)
        for out_slot in range(count):
            pressure[out_slot] = native[2 * (first + out_slot)]
        return pressure

    def _native_pressure(self, native, slots):
        contiguous = len(slots) == slots[-1] - slots[0] + 1
        if contiguous and len(slots) >= 32:
            return self._read_block_pressure(native, int(slots[0]), len(slots))
        if self.kind == "style_marm_joint201":
            if contiguous:
                return native[int(slots[0]):int(slots[-1]) + 1, ::2, ::2]
            # Read each requested frame contiguously before taking the same exact nodes.
            # This avoids h5py's joint sparse selection across time and both spatial axes.
            return np.stack([native[int(slot)][::2, ::2] for slot in slots])
        if contiguous:
            native_slots = slice(2 * int(slots[0]), 2 * int(slots[-1]) + 1, 2)
        else:
            native_slots = 2 * slots
        return native[native_slots]

    @staticmethod
    def _validate_labels(rec, slots, velocity, source, physical):
        np.testing.assert_array_equal(source, rec.source)
        if velocity.shape != GRID or not (np.isfinite(velocity).all() and (velocity > 0).all()):
            raise ValueError(f"invalid velocity grid for record {rec.information['id']}")
        if physical.shape != (len(slots), *GRID) or not np.issubdtype(physical.dtype, np.floating):
            raise ValueError(f"invalid pressure frames for record {rec.information['id']}")
        if np.any(physical[:, 0, :]) or (slots[0] == 0 and np.any(physical[0])):
            raise ValueError("pressure source violates the archived exact free surface or zero initial frame")

    def read(self, record_id, slots):
        slots = np.asarray(slots)
        if (slots.ndim != 1 or not slots.size or not np.issubdtype(slots.dtype, np.integer)
                or slots[0] < 0 or slots[-1] >= len(self.times) or np.any(np.diff(slots) <= 0)):
            raise ValueError("time slots must be nonempty, unique and increasing on the archived axis")
        rec = self._bound[str(record_id)]
        self._check_source_file(rec)
        if rec.velocity_path:
            velocity = np.asarray(np.load(rec.velocity_path, allow_pickle=False), dtype=np.float32)
            source = rec.source
            labels = np.load(rec.pressure_path, mmap_mode="r", allow_pickle=False)
            if labels.shape != (len(self.times), *GRID):
                raise ValueError("held-out pressure must cover the full archived time axis")
            if _array_sha(velocity) != rec.information["velocity_sha256"]:
                raise ValueError("held-out velocity content changed")
            physical = labels[slots]
        else:
            with h5py.File(rec.pressure_path, "r") as handle:
                velocity, source, native = self._native_metadata(handle)
                physical = self._native_pressure(native, slots)
        self._validate_labels(rec, slots, velocity, source, physical)
        target = normalize_pressure_pa(physical, source[4], self.metadata["pressure_scale_pa"])
        return {"velocity": torch.from_numpy(velocity.copy())[None, None],
                "source": torch.from_numpy(source.copy())[None],
                "times": torch.from_numpy(self.times[slots].astype(np.float32)),
                "target": torch.from_numpy(target)}

    def full_energy(self, record_id, chunk=24):
        """Sum the exact training targets over every time, with one source file open."""
        if type(chunk) is not int or chunk < 1:
            raise ValueError("full-energy chunk length must be a positive integer")
        rec = self._bound[str(record_id)]
        self._check_source_file(rec)
        if rec.information["split"] != "train":
            raise ValueError("normalization energy must use TRAIN pressure")
        energy = 0.
        with h5py.File(rec.pressure_path, "r") as handle:
            velocity, source, native = self._native_metadata(handle)
            for first in range(0, len(self.times), chunk):
                slots = np.arange(first, min(first + chunk, len(self.times)))
                physical = self._native_pressure(native, slots)
                self._validate_labels(rec, slots, velocity, source, physical)
                values = normalize_pressure_pa(physical, source[4], self.metadata["pressure_scale_pa"])
                flat = values.astype(np.float64).reshape(-1)
                energy += float(flat @ flat)
        return energy

    def audit(self):
        """Audit split metadata plus one actual non-TEST trajectory per available family."""
        media, source_cases, crops = {}, {}, {}
        summary = {}
        selected = dict(zip(SPLITS, (self.train_ids, self.validation_ids, self.test_ids)))
        for split in SPLITS:
            rows = [row for row in self.records.values() if row["split"] == split]
            media[split] = {row.get("velocity_model_sha256", row.get("velocity_sha256")) for row in rows} - {None}
            source_cases[split] = {
                (row.get("velocity_model_sha256", row.get("velocity_sha256")),
                 _source_key(self._bound[row["id"]].source)) for row in rows
            }
            crops[split] = {
                (rect["raw_file_sha256"], rect["x0_m"], rect["x1_m"], rect["z0_m"], rect["z1_m"])
                for row in rows if (rect := row.get("source_record", {}).get("raw_crop"))
            }
            scopes = {key: sorted({float(self._bound[r["id"]].source[i]) for r in rows})
                      for key, i in (("frequency_hz", 2), ("delay_s", 3), ("amplitude", 4))}
            summary[split] = {"release_records": len(rows), "selected_ids": list(selected[split]),
                              "families": dict(Counter(row["family"] for row in rows)),
                              "unique_velocity_hashes": len(media[split]),
                              "unique_source_positions": len({_source_key(self._bound[r["id"]].source)[:2]
                                                               for r in rows}),
                              **scopes}
        overlaps = {}
        def intersects(a, b):
            return a[0] == b[0] and min(a[2], b[2]) > max(a[1], b[1]) and min(a[4], b[4]) > max(a[3], b[3])

        train_windows = sorted(crops["train"])
        within_train = sum(intersects(a, b) for i, a in enumerate(train_windows) for b in train_windows[i + 1:])
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            crossing = sum(intersects(a, b) for a in crops[first] for b in crops[second] if a != b)
            overlaps[f"{first}_{second}"] = {"shared_velocity_hashes": len(media[first] & media[second]),
                                            "shared_exact_velocity_source_pairs": len(source_cases[first] & source_cases[second]),
                                            "distinct_overlapping_marmousi_crop_pairs": int(crossing)}
        probes = []
        slots = np.array([0, len(self.times) // 2, len(self.times) - 1])
        for split in ("train", "validation"):
            seen = set()
            for key in selected[split]:
                family = self.records[key]["family"]
                if family in seen:
                    continue
                seen.add(family)
                batch = self.read(key, slots)
                rec = self._bound[key]
                amplitude = float(rec.source[4])
                sigma = self.metadata["pressure_scale_pa"]
                recovered = restore_pressure_pa(batch["target"].double(), amplitude, sigma).numpy()
                view_check = False
                if not rec.velocity_path:
                    entry = str(rec.information["record_id"])
                    with h5py.File(self._view_paths[split], "r") as view:
                        j = self._view_indices[split][entry]
                        physical = view["pressure201"][j, slots]
                        np.testing.assert_array_equal(
                            batch["target"].numpy(), normalize_pressure_pa(physical, amplitude, sigma)
                        )
                        np.testing.assert_array_equal(batch["velocity"][0, 0].numpy(),
                                                      view["velocity201"][j].astype(np.float32))
                        np.testing.assert_array_equal(rec.source, view["source_parameters"][j].astype(np.float32))
                    relative = float(np.linalg.norm((recovered - physical).ravel()) /
                                     max(np.linalg.norm(physical.astype(np.float64).ravel()), 1e-300))
                    if relative > 2e-7:
                        raise ValueError("pressure physical recovery disagrees with the archived source")
                    view_check = True
                else:
                    physical = np.load(rec.pressure_path, mmap_mode="r", allow_pickle=False)[slots]
                    relative = float(np.linalg.norm((recovered - physical).ravel()) /
                                     max(np.linalg.norm(physical.astype(np.float64).ravel()), 1e-300))
                probes.append({"split": split, "id": key, "family": family, "time_slots": slots.tolist(),
                               "actual_time_s": self.times[slots].tolist(), "source": rec.source.tolist(),
                               "relative_pressure_roundtrip_l2": relative,
                               "release_vds_equal": view_check, "free_surface_row": 0})
        return {"binding": self.binding, "splits": summary, "overlaps": overlaps,
                "distinct_overlapping_training_marmousi_crop_pairs": int(within_train),
                "spatial_axes": {"order": "Z,X", "spacing_m": [10., 10.], "free_surface_row": 0},
                "time_axis": {"count": len(self.times), "start_s": float(self.times[0]),
                              "end_s": float(self.times[-1]), "time_strata_sizes": [len(s) for s in _strata(self.times)]},
                "normalization": {"sigma_u_pa": self.metadata["pressure_scale_pa"],
                                  "source_scales": self.metadata["source_scales"],
                                  "amplitude_zero_policy": "all-zero reference required; target and physical output zero",
                                  "forward": "y=u_pa/(A*sigma_u)", "inverse": "u_pa=y*A*sigma_u"},
                "probes_exclude_test_pressures": True, "record_probes": probes}


def open_corpus(config):
    settings = dict(config.get("data", config))
    partition_manifest = settings.pop("partition_manifest", None)
    source_assertion = settings.pop("source_geometry_assertion", None)
    corpus = Corpus(settings)
    if partition_manifest is not None:
        from partition_v2 import apply_partition
        corpus = apply_partition(corpus, partition_manifest)
    if source_assertion is not None:
        from source_contract_v2 import assert_source_contract
        corpus = assert_source_contract(corpus, source_assertion)
    return corpus


def _strata(times):
    times = np.asarray(times)
    if times.ndim != 1 or times.size < 4 or not (np.isfinite(times).all() and np.all(np.diff(times) > 0)):
        raise ValueError("time strata require a finite, increasing physical axis with at least four frames")
    return np.array_split(np.arange(times.size, dtype=np.int64), 4)


class StratifiedSampler:
    """Draw four distinct uniformly distributed records and a query from each time quarter."""

    def __init__(self, times, train_ids, seed):
        self.train_ids = tuple(map(str, train_ids))
        if len(self.train_ids) < 4 or len(set(self.train_ids)) != len(self.train_ids):
            raise ValueError("expected at least four distinct training record IDs")
        self.strata = _strata(times)
        self.strata_sizes = tuple(len(group) for group in self.strata)
        self.binding = {"train_ids": list(self.train_ids), "strata_sizes": list(self.strata_sizes),
                        "times_sha256": sha256(np.asarray(times, dtype="<f8").tobytes()).hexdigest()}
        self.rng = np.random.default_rng(seed)
        self.updates_drawn = 0

    def draw(self):
        records = self.rng.choice(len(self.train_ids), size=4, replace=False)
        result = []
        for index in records:
            slots = np.array([int(self.rng.integers(group[0], group[-1] + 1))
                              for group in self.strata], dtype=np.int64)
            result.append((self.train_ids[int(index)], slots))
        self.updates_drawn += 1
        return result

    def state_dict(self):
        return {"binding": self.binding, "rng": self.rng.bit_generator.state,
                "updates_drawn": self.updates_drawn}

    def load_state_dict(self, value):
        if value["binding"] != self.binding or type(value["updates_drawn"]) is not int or value["updates_drawn"] < 0:
            raise ValueError("the checkpoint's sampler belongs to a different data protocol")
        self.rng.bit_generator.state = value["rng"]
        self.updates_drawn = value["updates_drawn"]


def energy_floor(energies):
    values = np.asarray(list(energies), dtype=np.float64)
    if not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("TRAIN record energies must be finite nonnegative numbers")
    positive = values[values > 0]
    return max(1e-12, 1e-6 * float(np.median(positive))) if positive.size else 1e-12


def build_training_energy_metadata(dataset, output):
    """Compute each declared TRAIN record's full-axis normalized pressure SSE once."""
    path = Path(output)
    expected = {"schema": ENERGY_SCHEMA, "corpus_binding_sha256": _json_sha(dataset.binding),
                "train_ids": list(dataset.train_ids), "n_times": len(dataset.times),
                "scope": "selected TRAIN records only", "space_reduction": "sum_over_all_output_nodes"}
    if path.is_file():
        saved = _read_json(path)
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("energy metadata belongs to a different dataset or TRAIN subset")
        energies = saved["energies"]
        if (not set(energies).issubset(dataset.train_ids)
                or any(not math.isfinite(value) or value < 0 for value in energies.values())):
            raise ValueError("saved pressure energy metadata is invalid")
        if saved["complete"]:
            if (set(energies) != set(dataset.train_ids)
                    or saved["epsilon_E"] != energy_floor(energies.values())
                    or saved["floored_records"] != sum(e < saved["epsilon_E"] for e in energies.values())):
                raise ValueError("a complete energy file must contain all declared TRAIN records and the exact floor")
            return saved
    else:
        energies = {}
    for key in dataset.train_ids:
        if key not in energies:
            value = dataset.full_energy(key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid full reference energy for record {key}")
            energies[key] = value
            _write_json(path, {**expected, "complete": False, "energies": energies})
    epsilon = energy_floor(energies.values())
    result = {**expected, "complete": True, "energies": energies, "epsilon_E": epsilon,
              "floored_records": sum(value < epsilon for value in energies.values()),
              "positive_energy_records": sum(value > 0 for value in energies.values())}
    _write_json(path, result)
    return result


def stratified_relative_pressure_loss(prediction, target, strata_sizes, reference_energy, epsilon_E):
    """One-record estimator of its full-axis relative squared pressure error."""
    if (prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != 4
            or len(strata_sizes) != 4 or any(n <= 0 for n in strata_sizes)):
        raise ValueError("expected four matched full pressure frames with four positive stratum sizes")
    reference_energy, epsilon_E = float(reference_energy), float(epsilon_E)
    if not (math.isfinite(reference_energy) and reference_energy >= 0
            and math.isfinite(epsilon_E) and epsilon_E > 0):
        raise ValueError("reference energy and its positive floor must be finite")
    squared = (prediction.double() - target.detach().to(prediction.device, torch.float64)).square()
    weights = torch.as_tensor(strata_sizes, device=prediction.device, dtype=torch.float64).detach()
    return (squared.flatten(1).sum(1) * weights).sum() / max(reference_energy, epsilon_E)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--energy", type=Path)
    args = parser.parse_args()
    dataset = open_corpus(_read_json(args.config))
    _write_json(args.audit, dataset.audit())
    if args.energy is not None:
        build_training_energy_metadata(dataset, args.energy)


if __name__ == "__main__":
    main()
