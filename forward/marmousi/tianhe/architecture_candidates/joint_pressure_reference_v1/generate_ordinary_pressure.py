"""Materialize the frozen 24 TRAIN and 12 VAL 801-frame 5 m pressure records."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools.marmousi_target_dataset_refined.common import atomic_json, digest, read_json, save_array, sha_file, sync_directory


SCHEMA = "marmousi_lwc5_ordinary_full_pressure_v1"
TRAIN_SHA = "facac4bab3f679d845d19d8cb6eb0d8d02655cd5ee9910295cd4dbf704cff6cf"
VALIDATION_SHA = "d67aad52b30f72751d1eb5187c3ee8c400cba4cdc1d36abe2a86e59c308191d2"
VALIDATION_RECEIPT_SHA = "a2adf259d15caf42cb60c0fe3fe30c2006bf4c11a9c3205e4d39097f12c4ec5c"
SIGMA150_SHA = "fdc8d68d276699016cabc5dd135c817cae6b30ac80d263af063f4152e2ce8dcf"
SPLITS = {"train": 24, "validation": 12}
FRAMES = tuple(range(801))
PRESSURE_SHAPE = (801, 201, 201)
B5_SHAPE = (801, 201)
DTYPE = np.dtype("<f4")
UNITS = "raw_numeric_lwc_pressure"


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def array_sha(value):
    return sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def val_rows(release, source):
    checksum = sha_file(source)
    data = read_json(source)
    if checksum == VALIDATION_SHA:
        require(data["split"] == "validation" and data["release_binding"] == release.binding
                and data["receiver_rms"] == release.rms and data["full_pressure_shape_per_shot"] == list(PRESSURE_SHAPE),
                "Original VAL pressure manifest changed release or axes")
        return data["models"], checksum
    require(checksum == VALIDATION_RECEIPT_SHA and data["validation_manifest_sha256"] == VALIDATION_SHA
            and data["used_true_test_labels"] is False and data["parents"] == 6 and data["paired_shots"] == 12
            and len(data["per_shot"]) == 12, "Use the original VAL manifest or its immutable 12-shot evaluation receipt")
    members = {}
    for shot in data["per_shot"]:
        identifier = shot["model_id"]
        row = release.rows[identifier]
        require(row["family"] == shot["family"], "Archived VAL family changed")
        if identifier not in members:
            members[identifier] = {key: row[key] for key in ("model_id", "family", "parent_group_id", "velocity_sha256")}
            members[identifier].update(shot_ids=[], parent_used_for_e12_val_b5_selection=(
                row["parent_group_id"] in data["val_parents_previously_exposed_to_e12_b5_selection"]))
        members[identifier]["shot_ids"].append(shot["shot_id"])
    return list(members.values()), checksum


def build_plan(release, train_path, validation_path):
    from make_direction_labels import EXCLUDED_MODELS
    import score_fullfield_e12 as teacher

    require(sha_file(train_path) == TRAIN_SHA, "The existing 24-shot TRAIN manifest checksum changed")
    train = read_json(train_path)
    require(train["split"] == "train" and train["binding"]["release_binding"] == release.binding
            and train["binding"]["receiver_rms"] == release.rms and train["frames"] == len(FRAMES)
            and train["native_shape_zx"] == [201, 201] and train["model_selection_independent_of_scores"] is True,
            "Original TRAIN pressure manifest changed release or axes")
    val, val_checksum = val_rows(release, validation_path)
    models = {"train": [], "validation": []}
    parents = set()
    for split, candidates in (("train", train["models"]), ("validation", val)):
        require(Counter(row["family"] for row in candidates) == {key: SPLITS[split]//12 for key in "ABCDEF"},
                "The fixed ordinary split must retain every registered family and parent count")
        for proposed in candidates:
            model = proposed["model_id"]
            row = release.rows[model]
            parent = row["parent_group_id"]
            shots, roster = proposed["shot_ids"], release.roster[model]
            require(model in release.splits[split] and row["split"] == split and model not in EXCLUDED_MODELS
                    and not row["is_exact_target"] and not row["is_exact_initial"] and not release.is_exact_anchor(row["velocity_sha256"])
                    and row["velocity_sha256"] != SIGMA150_SHA and parent not in parents
                    and row["waveform_used_for_supervised_training"] == (split == "train")
                    and all(proposed[key] == row[key] for key in ("family", "parent_group_id", "velocity_sha256"))
                    and row["velocity_path"] == f"velocities/{model}.npy"
                    and len(roster) == 4 and roster == sorted(set(roster)) and len(shots) == 2
                    and shots[0] in roster[:2] and shots[1] in roster[2:],
                    f"Invalid archived ordinary pressure row: {model}")
            was_exposed = split == "validation" and parent == "validation_F_parent_0000"
            require(proposed.get("parent_used_for_e12_val_b5_selection", False) is was_exposed,
                    f"The existing E12 B5 validation exposure proof changed: {parent}")
            record = {key: row[key] for key in ("model_id", "family", "parent_group_id", "velocity_sha256", "velocity_path")}
            record.update(split=split, shot_ids=shots, parent_used_for_e12_val_b5_selection=was_exposed)
            models[split].append(record)
            parents.add(parent)
    require(np.array_equal(release.times, np.arange(len(FRAMES)) * .005) and release.sources.shape == (38, 5)
            and np.all(release.sources[:, 1:] == (40., 8., .1875, 1.)), "Expected the registered 38 shallow Ricker sources and 801 times")
    return {"schema": SCHEMA, "release_binding": release.binding, "dataset": str(release.root),
            "geometry_array_sha256": release.geometry, "normalization_sha256": release.norm_digest,
            "roster_sha256": release.roster_digest, "receiver_rms": release.rms,
            "ordinary_train_manifest_sha256": TRAIN_SHA, "ordinary_validation_manifest_sha256": VALIDATION_SHA,
            "ordinary_validation_input_sha256": val_checksum, "models": models, "shots_by_split": SPLITS,
            "frames": len(FRAMES), "time_axis_s": {"first": 0., "step": .005, "last": 4.},
            "native_grid_m": {"first_zx": [0., 0.], "step_zx": [10., 10.], "last_zx": [2000., 2000.]},
            "pressure_axes": ["time", "z", "x"], "pressure_shape": list(PRESSURE_SHAPE), "b5_shape": list(B5_SHAPE),
            "dtype": DTYPE.str, "stored_units": UNITS, "pressure_excludes_cpml_exterior": True,
            "reference": {"scheme": "LWC8-4", "grid_spacing_m": 5., "dt_s": .0003125, "steps": 12800,
                          "pressure_sampling": "Coincident 10 m physical nodes without spatial interpolation",
                          "b5_sampling": "Frozen five-depth observation applied to original 5 m solver state"},
            "generator_source_sha256": sha_file(__file__), "reference_extractor_sha256": sha_file(teacher.__file__),
            "validation_used_for_ordinary_field_training": False, "exact_or_initial_or_fwi_media_used": False}


def shot_folder(root, row, shot):
    return root / row["split"] / row["model_id"] / f"shot_{shot:02d}"


def released_b5(release, row):
    key = row["model_id"]
    directory = release.root / "models" / key
    path = directory / "receivers.npy"
    info, roster = read_json(directory / "meta.json"), release.roster[key]
    require(info["status"] == "COMPLETE" and info["model_id"] == key and info["split"] == row["split"]
            and info["shot_ids"] == roster and info["completed_shots"] == len(roster)
            and info["geometry_time_hashes"] == release.geometry and info["velocity_sha256"] == row["velocity_sha256"]
            and all(info[key] == value for key, value in release.binding.items())
            and sha_file(path) == info["receivers_file_sha256"], "The published B5 record changed")
    b5 = np.load(path, allow_pickle=False)
    require(b5.shape == (4, *B5_SHAPE) and b5.dtype == DTYPE and np.isfinite(b5).all(), "Invalid B5 release array")
    return {shot: b5[roster.index(shot)] for shot in row["shot_ids"]}, info["receivers_file_sha256"]


def check_shot(pressure, b5, published, tolerance):
    require(pressure.shape == PRESSURE_SHAPE and pressure.dtype == DTYPE and np.isfinite(pressure).all()
            and np.any(pressure) and not np.any(pressure[:, 0]) and not np.any(pressure[0]),
            "Expected complete nonzero physical pressure with exact zero free surface and initial frame")
    require(b5.shape == B5_SHAPE and b5.dtype == DTYPE and np.isfinite(b5).all() and not np.any(b5[0]),
            "Invalid 5 m B5 replay or initial receiver frame")
    truth = published.astype(np.float64)
    energy = float(np.square(truth).sum())
    require(energy > 0., "Published B5 reference is entirely zero")
    relative = float(np.sqrt(np.square(b5.astype(np.float64) - truth).sum() / energy))
    require(np.isfinite(relative) and relative <= tolerance, f"B5 replay differs from published record: {relative:.8g} > {tolerance:.8g}")
    return relative, energy


def commit_shot(root, row, shot, plan_sha, release, pressure, b5, published, b5_file, statistics):
    tol = release.contract["solver"]["cpu_reference_equivalence_tolerance"]
    relative, energy = check_shot(pressure, b5, published, tol)
    target = shot_folder(root, row, shot)
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), "Refuse to replace a committed pressure shot")
    with TemporaryDirectory(prefix=f".{target.name}.partial-", dir=target.parent) as scratch:
        stage = Path(scratch)
        label_path, b5_path = stage / "pressure.npy", stage / "b5.npy"
        save_array(label_path, pressure)
        save_array(b5_path, b5)
        result = {"schema": SCHEMA, "status": "COMMITTED", "plan_sha256": plan_sha, "row": row, "shot_id": shot,
                  "source_x_z_f0_delay_amp": release.sources[shot].tolist(), "relative_directory": target.relative_to(root).as_posix(),
                  "pressure_file_sha256": sha_file(label_path), "b5_file_sha256": sha_file(b5_path),
                  "pressure_array_sha256": array_sha(pressure), "b5_array_sha256": array_sha(b5),
                  "pressure_shape": list(PRESSURE_SHAPE), "b5_shape": list(B5_SHAPE), "dtype": DTYPE.str, "stored_units": UNITS,
                  "published_b5_sha256": b5_file, "published_b5_shot_array_sha256": array_sha(published),
                  "b5_replay_relative_l2": relative, "b5_parity_tolerance": tol,
                  "qc": {"top_zero_all_times": True, "pressure_initial_zero": True, "b5_initial_zero": True,
                         "published_b5_energy": energy, "pressure_finite": True, "b5_parity_passed": True},
                  "solver": statistics}
        atomic_json(stage / "RECEIPT.json", result)
        sync_directory(stage)
        stage.rename(target)
        sync_directory(target.parent)
    return result


def read_shot(root, row, shot, plan_sha, release):
    target = shot_folder(root, row, shot)
    require(target.is_dir() and not target.is_symlink() and target.resolve().is_relative_to(root), "Committed pressure directory changed")
    result = read_json(target / "RECEIPT.json")
    require(result["schema"] == SCHEMA and result["status"] == "COMMITTED" and result["plan_sha256"] == plan_sha
            and result["row"] == row and result["shot_id"] == shot and result["source_x_z_f0_delay_amp"] == release.sources[shot].tolist()
            and result["relative_directory"] == target.relative_to(root).as_posix()
            and 0 <= result["b5_replay_relative_l2"] <= result["b5_parity_tolerance"] == release.contract["solver"]["cpu_reference_equivalence_tolerance"],
            "A committed shot differs from its frozen split, plan, geometry or B5 parity")
    for name, shape in (("pressure", PRESSURE_SHAPE), ("b5", B5_SHAPE)):
        path = target / f"{name}.npy"
        require(not path.is_symlink() and sha_file(path) == result[f"{name}_file_sha256"], "A committed pressure or B5 file changed")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        require(value.shape == shape and value.dtype == DTYPE and array_sha(value) == result[f"{name}_array_sha256"],
                "A committed pressure or B5 array changed")
    return result


def publish(output, plan_sha, receipts):
    counts = {split: sum(item["row"]["split"] == split for item in receipts) for split in SPLITS}
    status = {"schema": SCHEMA, "status": "COMPLETE" if counts == SPLITS else "PARTIAL", "plan_sha256": plan_sha,
              "shots_by_split": counts, "shots": sum(counts.values()), "shards": receipts,
              "raw_pressure_bytes": sum(counts.values()) * int(np.prod(PRESSURE_SHAPE)) * DTYPE.itemsize,
              "ordinary_val_not_used_for_training": True, "exact_initial_and_fwi_excluded": True}
    atomic_json(output / "STATUS.json", status)
    return status


def run(args):
    from make_direction_labels import Release
    from score_fullfield_e12 import LWCReference, NativeFinePressure, run_lock

    release = Release(args.dataset)
    output = args.output.resolve()
    require(not output.is_relative_to(release.root), "Write offline labels in a separate directory outside the release")
    plan = build_plan(release, args.train_manifest, args.validation_manifest)
    plan_sha = digest(plan)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        path = output / "PLAN.json"
        if path.exists():
            require(read_json(path) == plan, "Resume must preserve the exact ordinary pressure plan")
        else:
            atomic_json(path, plan)
        if args.plan_only:
            print(json.dumps({"status": "PLAN_VERIFIED", "plan_sha256": plan_sha, "shots_by_split": SPLITS}), flush=True)
            return
        import torch

        receipts, completed = [], set()
        for rows in plan["models"].values():
            for row in rows:
                for shot in row["shot_ids"]:
                    if shot_folder(output, row, shot).exists():
                        receipts.append(read_shot(output, row, shot, plan_sha, release))
                        completed.add((row["model_id"], shot))
        engines, new = {}, 0
        with torch.no_grad():
            for rows in plan["models"].values():
                for row in rows:
                    pending = [shot for shot in row["shot_ids"] if (row["model_id"], shot) not in completed]
                    if args.max_new is not None:
                        pending = pending[:max(0, args.max_new - new)]
                    if not pending:
                        continue
                    velocity = release.velocity(row["model_id"], row["split"], row["family"])
                    observed, published_file = released_b5(release, row)
                    batch = len(pending)
                    if batch not in engines:
                        require(torch.cuda.is_available() and torch.device(args.device).type == "cuda", "Use the assigned CUDA card for offline 5 m replay")
                        reference = LWCReference(release.root, device=args.device, build_dir=output / "fine_cuda")
                        engines[batch] = NativeFinePressure(reference.forward_contract, reference.cache, device=str(reference.device), batch_size=batch)
                    pressure, b5, solver = engines[batch].replay(velocity, pending, FRAMES)
                    pressure = pressure.cpu().numpy()
                    require(pressure.shape == (batch, *PRESSURE_SHAPE) and b5.shape == (batch, *B5_SHAPE), "Unexpected offline pressure batch shape")
                    for index, shot in enumerate(pending):
                        receipt = commit_shot(output, row, shot, plan_sha, release, pressure[index], b5[index], observed[shot], published_file, solver)
                        receipts.append(receipt)
                        new += 1
                        status = publish(output, plan_sha, receipts)
                        print(json.dumps({"stage": "SHOT_COMMITTED", "model_id": row["model_id"], "shot_id": shot,
                                          "b5_replay_relative_l2": receipt["b5_replay_relative_l2"], "shots_by_split": status["shots_by_split"]}), flush=True)
        status = publish(output, plan_sha, receipts)
        print(json.dumps({key: status[key] for key in ("status", "plan_sha256", "shots_by_split", "raw_pressure_bytes")}), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    for key in ("dataset", "train-manifest", "validation-manifest", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-new", type=int, help="Maximum number of missing shots to commit in this invocation")
    args = parser.parse_args()
    if args.max_new is not None and args.max_new < 1:
        parser.error("--max-new must be positive")
    run(args)


if __name__ == "__main__":
    main()
