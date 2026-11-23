"""Fit differentiable shot-wise residuals on released ordinary E12 examples."""

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist


@dataclass(frozen=True, order=True)
class Profile:
    base: str
    geometry: str
    bandwidth: float
    ridge: float
    signed_weight: float


PROFILES = tuple(Profile(*item) for item in product(
    ("native", "plus"), ("global", "depth", "mixed"), (1.0, 2.5, 6.25),
    (.0003, .003, .03), (0.0, 3.0)
))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, allow_nan=False, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


@dataclass
class Shots:
    descriptor: np.ndarray
    parent: np.ndarray
    family: np.ndarray
    energy: np.ndarray
    native: np.ndarray
    plus: np.ndarray
    native_tail: np.ndarray
    plus_tail: np.ndarray


def load_shots(cache, rows, split):
    groups = [[] for _ in range(38)]
    for row in rows:
        name = row["model_id"]
        if Path(name).name != name or "exact" in name:
            raise ValueError(f"Unexpected ordinary medium: {name}")
        with np.load(cache / split / (name + ".npz"), allow_pickle=False) as data:
            ids = data["ids"]
            descriptor = data["descriptor"]
            energy = data["target_energy"]
            native, plus = data["native"], data["plus"]
            native_tail, plus_tail = data["native_tail"], data["plus_tail"]
        if (ids.tolist() != row["shot_ids"] or descriptor.shape != (2601,)
                or native.shape != plus.shape or native.shape != (len(ids), 129, 201)
                or (energy <= 0.0).any()):
            raise ValueError(f"Residual cache differs from the ordinary roster: {name}")
        for slot, shot in enumerate(ids):
            groups[shot].append((descriptor, row["parent"], row["family"], energy[slot],
                                 native[slot], plus[slot], native_tail[slot], plus_tail[slot]))
    if any(not group for group in groups):
        raise ValueError(f"An original physical shot has no support in {split}")
    return [Shots(*[np.asarray(column) for column in zip(*group, strict=True)]) for group in groups]


def features(descriptors, source_x):
    """Track long paths, shallow paths, and impedance contrasts on the same input grid."""
    z = np.arange(51, dtype=np.float64)[:, None] * 40.
    x = z.T
    shallow = .08 + np.exp(-z / 380.) * (.3 + .7 * np.exp(-.5 * ((x - source_x) / 480.)**2))
    local = descriptors * np.sqrt((shallow / shallow.mean()).ravel())
    differences = np.diff(descriptors.reshape(-1, 51, 51), axis=1)
    contrast = differences.reshape(len(descriptors), -1)
    return descriptors, local, contrast


def covariances(train, query, source_x, parent):
    support = features(train, source_x)
    prediction = features(query, source_x)
    distinct = parent[:, None] != parent[None, :]
    gram, cross, lengths = [], [], []
    for source, target in zip(support, prediction, strict=True):
        distances = cdist(source, source, "sqeuclidean")
        nearest = np.sqrt(np.min(np.where(distinct, distances, np.inf), axis=1))
        scale = float(np.median(nearest))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Invalid geometry bandwidth from distinct TRAIN parents")
        gram.append(distances)
        cross.append(cdist(target, source, "sqeuclidean"))
        lengths.append(scale)
    return np.asarray(gram), np.asarray(cross), np.asarray(lengths)


def kernel(distances, lengths, bandwidth, geometry):
    weights = {"global": [1.0, 0.0, 0.0], "depth": [.4, .6, 0.0], "mixed": [.45, .35, .20]}[geometry]
    result = np.zeros(distances.shape[1:], dtype=np.float64)
    for delta, size, weight in zip(distances, lengths, weights, strict=True):
        if weight:
            result += weight * np.exp(-.5 * delta / (size * bandwidth)**2)
    return result


def error_covariance(energy, parents, families, signed_weight):
    """Inverse weight matrix for trace energy and same-parent signed differences."""
    norm = energy / np.median(energy)
    norm = np.clip(norm, .10, 10.)
    diagonal = 1.0 / norm
    result = np.diag(norm)
    if signed_weight:
        for parent in np.unique(parents[families == "F"]):
            indices = np.flatnonzero(parents == parent)
            if len(indices) == 2:
                left, right = indices
                a, b = diagonal[left], diagonal[right]
                s = signed_weight * math.sqrt(a * b)
                determinant = a*b + s*(a+b)
                result[np.ix_(indices, indices)] = np.asarray([[b+s, s], [s, a+s]]) / determinant
    return result


def spectrum_vector(values):
    return np.ascontiguousarray(values).view(np.float32).reshape(len(values), -1)


def validation_quadratic(train, val, val_tail):
    y, q = spectrum_vector(train), spectrum_vector(val)
    return ((y @ y.T).astype(np.float64), (q @ y.T).astype(np.float64),
            np.einsum("ij,ij->i", q, q, dtype=np.float64) + val_tail)


def integrated_error(weights, terms):
    gram, cross, remaining = terms
    variance = np.einsum("ij,ij->i", weights @ gram, weights)
    align = np.einsum("ij,ij->i", weights, cross)
    out = remaining + variance - 2.0 * align
    if np.any(out < -1e-6 * remaining):
        raise FloatingPointError("Residual energy is inconsistent with its Gram matrix")
    return np.maximum(out, 0.0)


def atomic_npz(path, **fields):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez(stream, **fields)
    temp.replace(path)


def fit(cache, output):
    source_plan = cache / "PLAN.json"
    source_status = json.loads((cache / "STATUS.json").read_text())
    plan = json.loads(source_plan.read_text())
    if (source_status["status"] != "COMPLETE" or plan["exact_waveform_reads"] != 0
            or source_status["models"] != sum(len(rows) for rows in plan["roster"].values())):
        raise ValueError("The ordinary-only cache is incomplete")
    training = load_shots(cache, plan["roster"]["train"], "train")
    validation = load_shots(cache, plan["roster"]["validation"], "validation")
    train_parents = set(np.concatenate([shot.parent for shot in training]))
    val_parents = set(np.concatenate([shot.parent for shot in validation]))
    if not train_parents.isdisjoint(val_parents):
        raise ValueError("Residual fits require disjoint TRAIN and VAL parents")
    val_order = sorted(val_parents, key=lambda name: hashlib.sha256(
        ("marmousi_b5_realign_20261121:" + name).encode()).hexdigest())
    audit = set(val_order[::4])
    energy = np.zeros(2)
    baseline = np.zeros((2, 2))
    scores = np.zeros((len(PROFILES), 2))
    output.mkdir(exist_ok=False, parents=True)
    start = monotonic()
    write_json(output / "PLAN.json", dict(
        ordinary_cache_sha256=sha(source_plan), train_parents=len(train_parents),
        calibration_parent_ids=sorted(val_parents - audit),
        audit_parent_ids=sorted(audit), test_data_reads=0,
        configurations=[asdict(p) for p in PROFILES], candidate_inputs="velocity and source only"
    ))
    all_covariances = []
    for sid, (ts, vs, source) in enumerate(zip(training, validation, plan["sources"], strict=True)):
        test_index = np.isin(vs.parent, list(audit))
        selectors = (~test_index, test_index)
        for ix, select in enumerate(selectors):
            energy[ix] += vs.energy[select].sum()
        gram, cross, lengths = covariances(ts.descriptor, vs.descriptor, float(source[0]), ts.parent)
        all_covariances.append((gram, lengths))
        terms = {
            base: validation_quadratic(getattr(ts, base), getattr(vs, base), getattr(vs, base+"_tail"))
            for base in ("native", "plus")
        }
        for base_index, base in enumerate(("native", "plus")):
            for ix, select in enumerate(selectors):
                baseline[base_index, ix] += terms[base][-1][select].sum()
        regularizers = {eta: error_covariance(ts.energy, ts.parent, ts.family, eta) for eta in (0.0, 3.0)}
        # Different residual targets share identical interpolation weights.
        by_geometry = {}
        for i, profile in enumerate(PROFILES):
            shape = (profile.geometry, profile.bandwidth, profile.ridge, profile.signed_weight)
            if shape not in by_geometry:
                k = kernel(gram, lengths, profile.bandwidth, profile.geometry)
                q = kernel(cross, lengths, profile.bandwidth, profile.geometry)
                factor = cho_factor(k + profile.ridge * regularizers[profile.signed_weight], lower=True)
                by_geometry[shape] = cho_solve(factor, q.T).T
            err = integrated_error(by_geometry[shape], terms[profile.base])
            for ix, select in enumerate(selectors):
                scores[i, ix] += err[select].sum()
        if (sid + 1) % 5 == 0 or sid == 37:
            print(json.dumps(dict(shot_groups=sid+1, total=38, seconds=monotonic()-start)), flush=True)
    metrics = np.sqrt(scores / energy)
    old = np.sqrt(baseline / energy)
    best_id = int(np.argmin(metrics[:, 0]))
    selected = PROFILES[best_id]
    report = dict(
        objective="pooled full-time B5 relative L2 by Parseval",
        old_native=old[0].tolist(), old_native_plus_kernel=old[1].tolist(),
        chosen=asdict(selected), relative_l2=metrics[best_id].tolist(),
        improved_both_ordinary_groups=bool(np.all(metrics[best_id] < old[1])),
        score_order=["validation_selection", "validation_audit"],
        profile_scores=[dict(profile=asdict(p), relative_l2=m.tolist())
                        for p, m in zip(PROFILES, metrics, strict=True)],
        readout_identity=plan["native"], cache_plan_sha256=sha(source_plan),
        fitting_code_sha256=sha(__file__), used_exact_marmousi_for_training_or_selection=False,
    )
    files = []
    if report["improved_both_ordinary_groups"]:
        for sid, (ts, (distances, lengths)) in enumerate(zip(training, all_covariances, strict=True)):
            k = kernel(distances, lengths, selected.bandwidth, selected.geometry)
            r = error_covariance(ts.energy, ts.parent, ts.family, selected.signed_weight)
            chol = np.linalg.cholesky(k + selected.ridge * r)
            rhs = spectrum_vector(getattr(ts, selected.base)).astype(np.float64)
            weights = solve_triangular(chol, rhs, lower=True).astype(np.float32, order="C")
            path = output / f"shot_{sid:02d}.npz"
            atomic_npz(path, train_descriptor=ts.descriptor, lengths=lengths, chol=chol,
                       residual_coordinates=weights.view(np.complex64).reshape(len(weights), 129, 201))
            files.append(dict(shot=sid, path=path.name, sha256=sha(path)))
    report["frozen_shot_blocks"] = files
    report["status"] = "SEALED" if files else "ORDINARY_IMPROVEMENT_NOT_CONFIRMED"
    report["seconds"] = monotonic() - start
    write_json(output / "SELECTED.json", report)
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ("profile_scores", "frozen_shot_blocks", "readout_identity")}), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fit(args.cache, args.output)


if __name__ == "__main__":
    main()
