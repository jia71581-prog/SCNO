"""Prepare an observation-only input and score sealed native-401 predictions."""

from dataclasses import dataclass
from hashlib import sha256 as hash_bytes
import json
from math import sqrt
from pathlib import Path

import numpy as np

from finetune_ul_transfer import sha256
from prepare_ul_paper_instances import OBSERVATION_INDICES, SCHEMA, validated_reference


FIELD_SHAPE = (401, 201, 201)
NATIVE_TIMES = np.arange(401, dtype=np.float64) * .0025
SUPPORTS = {
    "full_401": np.arange(401),
    "early_160": np.arange(160),
    "late_241": np.arange(160, 401),
    "excluding_observations_399": np.setdiff1d(np.arange(401), OBSERVATION_INDICES),
    "early_excluding_observations_158": np.setdiff1d(np.arange(160), OBSERVATION_INDICES),
}


def array_sha(array):
    return hash_bytes(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def tensor_sha(tensor):
    return array_sha(tensor.detach().cpu().contiguous().numpy())


@dataclass(frozen=True)
class ObservationInput:
    velocity_mps: np.ndarray
    source_parameters: np.ndarray
    observed_pa: np.ndarray
    observation_indices: np.ndarray

    def identity(self):
        return {**{f"{k}_sha256": array_sha(v) for k, v in vars(self).items()},
                "time_axis_sha256": array_sha(NATIVE_TIMES)}


@dataclass(frozen=True)
class EvaluationInput:
    key: str
    record_id: str
    primary: bool
    inputs: ObservationInput
    reference: dict


def _observation_input(velocity, source, observed_pa, indices):
    if (velocity.shape != (201, 201) or source.shape != (5,)
            or observed_pa.shape != (2, 201, 201) or not np.array_equal(indices, OBSERVATION_INDICES)
            or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in (velocity, source, observed_pa))
            or velocity.min() <= 0. or source[2] <= 0. or source[4] <= 0.
            or np.any(observed_pa[:, 0]) or np.square(observed_pa.astype(np.float64)).sum() == 0.):
        raise ValueError("Invalid observation-only native-401 input")
    return ObservationInput(velocity, source, observed_pa, indices.copy())


def _open_external(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Validation asset must be inside the validation directory")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def prepare_external_observations(root, roster):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "VALIDATION_MANIFEST.json").read_text(encoding="utf-8"))
    times = _open_external(root, manifest["times_s"])
    if not np.allclose(times, NATIVE_TIMES, atol=1e-7, rtol=0.):
        raise ValueError("External evaluation uses a different registered axis")
    by_index = {r["frozen_roster_index"]: r for r in manifest["records"] if r["family"] == "layered"}
    result = []
    for index in roster:
        row = by_index[index]
        velocity = np.array(_open_external(root, row["velocity_mps"]))
        pressure = _open_external(root, row["pressure_pa"])
        source = np.asarray(row["source_parameters"], dtype=np.float32)
        if (array_sha(velocity) != row["velocity_sha256"] or pressure.shape != FIELD_SHAPE
                or pressure.dtype != np.float32 or row["split"] != "validation"
                or not np.array_equal(source, np.array([1000., 1000., 15., .1, 1.], dtype=np.float32))):
            raise ValueError("The validation record does not bind the recorded geometry or source")
        case = row["record_id"].split(":")[-2]
        inputs = _observation_input(velocity, source, pressure[OBSERVATION_INDICES], OBSERVATION_INDICES)
        result.append(EvaluationInput(case, row["record_id"], True, inputs,
                                      {"root": str(root), "path": row["pressure_pa"],
                                       "sha256": row["pressure_array_sha256"]}))
    return result


def prepare_center_observations(root, cases):
    result = []
    for case in cases:
        directory = Path(root) / case
        src = json.loads((directory / "SOURCE.json").read_text(encoding="utf-8"))
        path = directory / "instance.npz"
        if (src["schema"] != SCHEMA or src["case"] != case or src["instance_sha256"] != sha256(path)
                or src["full_reference_in_instance"]):
            raise ValueError("Center instance provenance mismatch")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(ObservationInput.__dataclass_fields__):
                raise ValueError("A center instance must contain only observation inputs")
            inputs = _observation_input(*(saved[key] for key in ObservationInput.__dataclass_fields__))
        result.append(EvaluationInput(case, case, False, inputs,
                                      {"root": src["reference_root"],
                                       "sha256": src["downstream_reference_sha256"]}))
    return result


def open_reference(case):
    if case.primary:
        ref = _open_external(Path(case.reference["root"]), case.reference["path"])
        if array_sha(ref) != case.reference["sha256"]:
            raise ValueError("External reference array SHA-256 mismatch")
    else:
        ref, v, s, _, verified = validated_reference(case.reference["root"], case.key)
        if (verified["files"]["reference_uniform401.npy"]["sha256"] != case.reference["sha256"]
                or not np.array_equal(v, case.inputs.velocity_mps)
                or not np.array_equal(s, case.inputs.source_parameters)):
            raise ValueError("Scored pressure and center inputs have different bindings")
    if ref.shape != FIELD_SHAPE or not np.array_equal(ref[OBSERVATION_INDICES], case.inputs.observed_pa):
        raise ValueError("The sealed observations do not match the scored reference")
    return ref


def reduce_support(sse, energy, frames):
    if not energy > 0.:
        raise ValueError("Scoring requires positive reference pressure energy")
    return {"frames": int(frames), "sse_pa2": float(sse), "reference_energy_pa2": float(energy),
            "relative_l2_percent": float(100. * sqrt(sse / energy))}


def streamed_window_scores(reference_pa, forecasts_pa, *, frame_batch=16):
    """Accumulate physical Pa precision statistics before pooling cases."""
    if reference_pa.ndim != 3 or len(reference_pa) != 401:
        raise ValueError("Score all 401 native frames, in TZX order")
    sse = {key: np.empty(401, dtype=np.float64) for key in forecasts_pa}
    energy = np.empty(401, dtype=np.float64)
    for first in range(0, 401, frame_batch):
        rows = slice(first, min(first + frame_batch, 401))
        reference = np.asarray(reference_pa[rows], dtype=np.float64)
        if not np.isfinite(reference).all():
            raise FloatingPointError("Nonfinite reference pressure")
        energy[rows] = np.square(reference).sum(axis=(1, 2), dtype=np.float64)
        for name, forecast in forecasts_pa.items():
            predicted = forecast[rows]
            if hasattr(predicted, "detach"):
                predicted = predicted.detach().cpu().numpy()
            if predicted.shape != reference.shape or not np.isfinite(predicted).all() or np.any(predicted[:, 0]):
                raise FloatingPointError("Scored Pa pressure is invalid or violates the top boundary")
            sse[name][rows] = np.square(predicted.astype(np.float64) - reference).sum(axis=(1, 2), dtype=np.float64)
    return {
        name: {k: reduce_support(values[idx].sum(), energy[idx].sum(), len(idx))
               for k, idx in SUPPORTS.items()} for name, values in sse.items()
    }


def pool_case_scores(cases):
    """Pool over independent media using one summed Pa energy per support."""
    entries = list(cases)
    result = {}
    for group in entries[0]:
        result[group] = {}
        for support, index in SUPPORTS.items():
            records = [case[group][support] for case in entries]
            result[group][support] = reduce_support(sum(r["sse_pa2"] for r in records),
                                                   sum(r["reference_energy_pa2"] for r in records), len(index))
    return result
