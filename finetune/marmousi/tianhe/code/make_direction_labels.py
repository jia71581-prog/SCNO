#!/usr/bin/env python3
"""Commit released TRAIN or VAL F-family secants and check them at half-step with LWC8-4."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from tools.marmousi_target_dataset_refined.common import (
    atomic_json, digest, read_json, sha_array, sha_file, sync_directory,
)


BINDING = ("contract_hash", "dataset_recipe_hash", "manifest_hash")
EXPECTED_CONTRACT = "a8902f32c1ffe9a568f2d9370c78ccc761ffe948202658b9e0b2735c3ab5b047"
RECORD_SHAPE = (801, 201)
VELOCITY_SHAPE = (201, 201)
RECORD_UNITS = "raw_lwc"
SECANT_UNITS = "frozen_train_rms_normalized_per_unit_bounded_sigmoid_latent"
SCHEMA = "marmousi_lwc_direction_bank_v1"
EXCLUDED_MODELS = {"train_A_exact_initial", "test_E_exact_marmousi"}
GEOMETRY_NAMES = ("source_parameters", "receiver_coordinates", "observation_times",
                  "receiver_support_indices", "receiver_support_weights")


def require(ok, message):
    if not ok:
        raise ValueError(message)


@contextmanager
def split_lock(path):
    import fcntl

    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"A writer already owns the split at {path.parent}") from exc
        yield


def atomic_npz(path, **arrays):
    temporary = NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    pending = Path(temporary.name)
    try:
        with temporary:
            np.savez(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(pending, path)
        sync_directory(path.parent)
    finally:
        pending.unlink(missing_ok=True)


class Release:
    """Bind exact physical release metadata without opening test or opposite-split labels."""

    def __init__(self, root):
        self.root = root.resolve(strict=True)
        self.contract = read_json(root / "acquisition_contract.json")
        self.recipe = read_json(root / "dataset_recipe.json")
        receipt = read_json(root / "manifest_receipt.json")
        rows = [json.loads(line) for line in (root / "generation_manifest.jsonl").read_text().splitlines() if line]
        self.rows = {row["model_id"]: row for row in rows}
        self.splits = read_json(root / "splits.json")
        self.binding = {key: receipt[key] for key in BINDING}
        require(self.binding == {
            "contract_hash": digest({k: v for k, v in self.contract.items() if k != "contract_hash"}),
            "dataset_recipe_hash": digest({k: v for k, v in self.recipe.items() if k != "dataset_recipe_hash"}),
            "manifest_hash": digest(rows),
        } and self.contract["contract_hash"] == self.recipe["contract_hash"] == self.binding["contract_hash"] == EXPECTED_CONTRACT
            and self.recipe["dataset_recipe_hash"] == self.binding["dataset_recipe_hash"]
            and len(self.rows) == len(rows) == receipt["models"] == 1000, "Released contract, recipe or model manifest has changed")
        split_index = {model_id: split for split in ("train", "validation", "test") for model_id in self.splits[split]}
        require(sum(len(self.splits[s]) for s in ("train", "validation", "test")) == len(rows)
                and split_index == {k: row["split"] for k, row in self.rows.items()}, "Inconsistent model splits")
        self.parents = Counter(row["parent_group_id"] for row in rows)
        self.roster = read_json(root / "shot_roster.json")
        roster_spec = self.contract["dataset"]["shot_roster"]
        require(self.roster.keys() == self.rows.keys() and roster_spec == self.recipe["shot_roster"]
                and digest(self.roster) == roster_spec["canonical_sha256"]
                and sha_file(root / "shot_roster.json") == roster_spec["file_sha256"], "Changed shot roster")
        self.roster_digest = roster_spec["canonical_sha256"]
        report_path = root / "reports" / "dataset_check.json"
        norm_path = root / "normalization.json"
        report, norm = read_json(report_path), read_json(norm_path)
        require(all(report[key] == norm[key] == value for key, value in self.binding.items())
                and report["status"] == "PASS" and report["stage"] == "full1000"
                and report["all_receiver_shot_hashes_verified"] and report["no_parent_split_leakage"]
                and report["pair_sources_identical"] and norm["exact_target_included"] is False
                and set(norm["training_ids"]) <= set(self.splits["train"]), "Unverified release or non-TRAIN normalization")
        self.rms = float(norm["receiver_rms"])
        require(np.isfinite(self.rms) and self.rms > 0., "Invalid frozen TRAIN receiver RMS")
        self.norm_digest = sha_file(norm_path)
        self.validation_digest = sha_file(report_path)
        self.geometry = {key: self.contract["shared_arrays"][key]["array_sha256"] for key in GEOMETRY_NAMES}
        self.sources = self.shared("source_parameters")
        self.times = self.shared("observation_times")
        self.receivers = self.shared("receiver_coordinates")
        require(self.sources.shape == (38, 5) and self.receivers.shape == (201, 2)
                and np.array_equal(self.times, np.arange(801) * .005), "Unexpected frozen acquisition")
        self.bounds = tuple(self.contract["grid"]["velocity_bounds_m_s"])
        self.h_levels = tuple(self.contract["parameterization"]["pair_h_levels"])
        require(self.recipe["family_F"]["parameterization"] == self.contract["parameterization"]
                and self.contract["parameterization"]["type"] == "bounded_sigmoid", "Inconsistent F-family latent space")

    def shared(self, name):
        spec = self.contract["shared_arrays"][name]
        path = self.root / spec["path"]
        require(sha_file(path) == spec["file_sha256"], f"Changed acquisition file: {name}")
        values = np.load(path, allow_pickle=False)
        require(sha_array(values) == spec["array_sha256"], f"Changed acquisition array: {name}")
        return values

    def pair_plan(self, split):
        groups = defaultdict(list)
        for model_id in self.splits[split]:
            row = self.rows[model_id]
            if row["family"] == "F":
                groups[row["pair_id"]].append(row)
        by_h = defaultdict(list)
        for key, rows in sorted(groups.items()):
            require(len(rows) == self.parents[key] == 2 and {row["pair_sign"] for row in rows} == {-1, 1},
                    f"Invalid family-F split, group or sign: {key}")
            minus, plus = sorted(rows, key=lambda row: row["pair_sign"])
            params, h = plus["parameters"], plus["parameters"]["h_actual"]
            require(all(row["parent_group_id"] == key and row["split"] == split and not row["is_exact_target"]
                        and not row["is_exact_initial"] and row["model_id"] not in EXCLUDED_MODELS
                        and row["waveform_used_for_supervised_training"] == (split == "train") for row in rows)
                    and minus["parameters"] == params and minus["template_initial_sha256"] == plus["template_initial_sha256"]
                    and h == params["h_requested"] and h in self.h_levels and params["direction_rms"] == 1.
                    and params["symmetric_parameter_construction"], f"Invalid held-out or TRAIN recipe: {key}")
            minus_id, plus_id = minus["model_id"], plus["model_id"]
            shots = self.roster[minus_id]
            require(shots == self.roster[plus_id] == sorted(set(shots)) and len(shots) == 4
                    and all(type(i) is int and 0 <= i < len(self.sources) for i in shots), f"Invalid F-family roster: {key}")
            by_h[h].append({
                "pair_id": key, "parent_group_id": key, "split": split, "minus_model_id": minus_id, "plus_model_id": plus_id,
                "minus_velocity_sha256": minus["velocity_sha256"], "plus_velocity_sha256": plus["velocity_sha256"],
                "direction_type": "archived_family_F_multiscale_gaussian", "direction_seed": params["direction_seed"],
                "h": h, "shot_ids": shots, "bridge_alpha": params["alpha"],
                "initial_template_sha256": plus["template_initial_sha256"],
            })
        require(set(by_h) == set(self.h_levels), "Missing released F-family latent steps")
        # Round-robin order gives every registered step size a place in a short prefix.
        plan = [group[i] for i in range(max(map(len, by_h.values()))) for _, group in sorted(by_h.items()) if i < len(group)]
        require(len(plan) == {"train": 50, "validation": 6}[split]
                and len({digest(row["direction_seed"]) for row in plan}) == len(plan), "Inconsistent split parent or direction count")
        return plan

    def endpoint(self, model_id, split, shot_ids):
        row = self.rows[model_id]
        media = self.velocity(model_id, split, "F")
        directory = self.root / "models" / model_id
        raw_path = directory / "receivers.npy"
        meta = read_json(directory / "meta.json")
        require(meta["status"] == "COMPLETE" and meta["model_id"] == model_id and meta["split"] == split
                and meta["completed_shots"] == len(shot_ids) and meta["shot_ids"] == shot_ids
                and meta["velocity_sha256"] == row["velocity_sha256"] and meta["geometry_time_hashes"] == self.geometry
                and all(meta[k] == value for k, value in self.binding.items())
                and sha_file(raw_path) == meta["receivers_file_sha256"], f"Changed F-family receiver gather: {model_id}")
        raw = np.load(raw_path, allow_pickle=False)
        require(raw.dtype == np.float32 and raw.shape == (len(shot_ids), *RECORD_SHAPE)
                and np.isfinite(raw).all() and np.any(raw), f"Invalid F-family receiver gather: {model_id}")
        return media, raw, meta["receivers_file_sha256"]

    def velocity(self, model_id, split, family):
        row = self.rows[model_id]
        path = f"velocities/{model_id}.npy"
        require(model_id not in EXCLUDED_MODELS and row["split"] == split and row["family"] == family
                and row["velocity_path"] == path and not self.is_exact_anchor(row["velocity_sha256"]),
                "Only frozen, authorized non-anchor split media may supply labels")
        media = np.load(self.root / path, allow_pickle=False)
        require(media.shape == VELOCITY_SHAPE and media.dtype == np.float32 and np.isfinite(media).all()
                and sha_array(media) == row["velocity_sha256"], f"Changed released velocity: {model_id}")
        return media

    def is_exact_anchor(self, velocity_sha):
        return any(self.contract["templates"][f"{name}_array_sha256"] == velocity_sha for name in ("initial", "truth"))


def logit(velocity, bounds):
    lo, hi = bounds
    v = velocity.astype(np.float64)
    require(np.isfinite(v).all() and (v > lo).all() and (v < hi).all(), "Velocity leaves the bounded-logit interior")
    return np.log(v - lo) - np.log(hi - v)


def velocity_from_logit(latent, bounds):
    lo, hi = bounds
    return lo + (hi - lo) * np.exp(-np.logaddexp(0., -latent))


def centered_secant(minus, plus, h, scale):
    return (plus.astype(np.float64) - minus.astype(np.float64)) / (2 * h * scale)


def record_archive(release, directory, pair):
    shard = directory / f'{pair["pair_id"]}.npz'
    receipt_path = shard.with_suffix(".json")
    if receipt_path.is_file():
        record = read_json(receipt_path)
        require(all(record[k] == value for k, value in pair.items()) and record["release_binding"] == release.binding
                and record["shard"] == shard.name and sha_file(shard) == record["shard_sha256"],
                f"Cannot reuse a mismatched or corrupt label shard: {shard}")
        return record, True
    (minus, raw_minus, b5_minus_sha), (plus, raw_plus, b5_plus_sha) = [
        release.endpoint(pair[f"{sign}_model_id"], pair["split"], pair["shot_ids"]) for sign in ("minus", "plus")]
    lo_z, hi_z = (logit(v, release.bounds) for v in (minus, plus))
    h = pair["h"]
    center = velocity_from_logit((lo_z + hi_z) / 2, release.bounds)
    direction = (hi_z - lo_z) / (2 * h)
    rms = float(np.sqrt(np.mean(direction**2)))
    secant = centered_secant(raw_minus, raw_plus, h, release.rms)
    require(abs(rms - 1.) <= .001 and np.isfinite(secant).all() and np.any(secant)
            and not release.is_exact_anchor(sha_array(center.astype(np.float32))),
            f"Invalid latent direction or zero secant: {pair['pair_id']}")
    atomic_npz(shard, velocity_minus=minus, velocity_plus=plus, center_velocity_mps=center,
               direction_latent=direction, h=np.float64(h), shot_ids=np.asarray(pair["shot_ids"], dtype=np.int64),
               b5_minus=raw_minus, b5_plus=raw_plus, b5_secant_normalized=secant.astype(np.float32))
    record = {
        **pair, "shard": shard.name, "shard_sha256": sha_file(shard), "release_binding": release.binding,
        "center_model_id": pair["parent_group_id"], "center_velocity_sha256": sha_array(center),
        "direction_latent_sha256": sha_array(direction), "direction_latent_rms": rms,
        "minus_b5_release_file_sha256": b5_minus_sha, "plus_b5_release_file_sha256": b5_plus_sha,
        "physical_half_step_rms_m_s": float(np.sqrt(np.mean(((plus.astype(np.float64) - minus) / 2)**2))),
    }
    atomic_json(receipt_path, record)
    return record, False


def existing_manifest(directory, split, plan, selected):
    manifest_path = directory / "manifest.json"
    old = read_json(manifest_path) if manifest_path.exists() else None
    if old is not None:
        require(old["schema"] == SCHEMA and old["split"] == split
                and old["plan_sha256"] == digest(plan) and len(old["records"]) <= len(selected)
                and digest({k: v for k, v in old.items() if k != "bank_content_sha256"}) == old["bank_content_sha256"],
                "Preserve the existing split bank; do not shrink or rebind a committed prefix")
    return old


def commit_manifest(release, directory, split, plan, records, old, **provenance):
    manifest_path = directory / "manifest.json"
    manifest = {
        **release.binding, "schema": SCHEMA, "split": split, "status": "COMPLETE" if len(records) == len(plan) else "PREFIX",
        "units": RECORD_UNITS, "secant_units": SECANT_UNITS, "receiver_train_rms": release.rms,
        "shot_roster_hash": release.roster_digest, "normalization_file_sha256": release.norm_digest,
        "dataset_validation_file_sha256": release.validation_digest, "observation_shape": RECORD_SHAPE,
        "velocity_bounds_m_s": release.bounds, "parent_count": len({r["parent_group_id"] for r in records}),
        "pair_count": len(records), "pair_shot_count": sum(len(r["shot_ids"]) for r in records),
        "full_parent_count": len({r["parent_group_id"] for r in plan}), "plan_sha256": digest(plan), "records": records,
        "ordinary_splits_only": True, "exact_targets_or_fwi_checkpoints_used": False,
        **provenance,
    }
    manifest = {**manifest, "bank_content_sha256": digest(manifest)}
    if old is None or old["bank_content_sha256"] != manifest["bank_content_sha256"]:
        atomic_json(manifest_path, manifest)
    return manifest


def archive(release, directory, split, max_parents):
    plan = release.pair_plan(split)
    require(max_parents is None or 1 <= max_parents <= len(plan), "Invalid parent limit")
    selected = plan[:max_parents]
    old = existing_manifest(directory, split, plan, selected)
    records, reused = [], 0
    for pair in selected:
        record, cached = record_archive(release, directory, pair)
        records.append(record)
        reused += cached
    manifest = commit_manifest(release, directory, split, plan, records, old)
    print(json.dumps({"manifest": str(directory / "manifest.json"), "split": split, "parents": len(records), "reused_parents": reused,
                      "pair_shots": manifest["pair_shot_count"], "bank_content_sha256": manifest["bank_content_sha256"]}), flush=True)


def l2(value):
    return float(np.linalg.norm(value.reshape(-1)))


def check_half_steps(release, directory, split, count, batch, max_error, *, grouping="h", levels=None):
    import torch
    from lwc_reference import LWCReference

    torch.set_num_threads(4)
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest["split"] == split and manifest["schema"] == SCHEMA
            and all(manifest[k] == value for k, value in release.binding.items())
            and digest({k: v for k, v in manifest.items() if k != "bank_content_sha256"}) == manifest["bank_content_sha256"],
            "Use only the verified bank from the chosen split")
    levels = release.h_levels if levels is None else levels
    by_group = defaultdict(list)
    for row in manifest["records"]:
        by_group[row[grouping]].append(row)
    selected = [row for level in levels for row in by_group[level][:count]]
    require(all(by_group[level] for level in levels), f"QC requires at least one parent per registered {grouping}")
    quality = directory / "quality"
    quality.mkdir(exist_ok=True)
    checks, engine = [], None
    for record in selected:
        shard = directory / record["shard"]
        require(sha_file(shard) == record["shard_sha256"], f"Changed QC label shard {shard}")
        saved = quality / f'{record["pair_id"]}.json'
        if saved.is_file():
            prior = read_json(saved)
            require(prior["shard_sha256"] == record["shard_sha256"] and prior["split"] == split,
                    f"Changed half-step QC binding: {saved}")
            checks.append(prior)
            continue
        if engine is None:
            engine = LWCReference(release.root, device="cuda:0", build_dir=quality / "cuda_build")
        with np.load(shard, allow_pickle=False) as data:
            require(sha_array(data["center_velocity_mps"]) == record["center_velocity_sha256"]
                    and sha_array(data["direction_latent"]) == record["direction_latent_sha256"], f"Changed half-step probe: {shard}")
            z = logit(data["center_velocity_mps"], release.bounds)
            d = data["direction_latent"]
            shots = data["shot_ids"].tolist()
            coarse = centered_secant(data["b5_minus"], data["b5_plus"], record["h"], release.rms)
        h = record["h"] / 2
        refined = []
        for sign in (-1, 1):
            v = velocity_from_logit(z + sign * h * d, release.bounds)
            refined.append(np.concatenate([engine.predict(v, shots[i:i + batch]).cpu().numpy()
                                           for i in range(0, len(shots), batch)]))
        fine = centered_secant(*refined, h, release.rms)
        require(np.isfinite(fine).all() and l2(fine) > 0, "Invalid refined LWC secant")
        errors = [l2(a - b) / l2(b) for a, b in zip(coarse, fine, strict=True) if l2(b) > 0.]
        require(len(errors) == len(shots), "Zero refined directional record")
        row = {"pair_id": record["pair_id"], "split": split, "h": record["h"], "h_refined": h, "shot_ids": shots,
               "relative_l2": l2(coarse - fine) / l2(fine), "per_shot_relative_l2": errors,
               "signed_cosine": float(np.vdot(coarse, fine)) / (l2(coarse) * l2(fine)),
               "shard_sha256": record["shard_sha256"], "fine_solver": "frozen_LWC8-4_CUDA_801_B5"}
        atomic_json(saved, row)
        checks.append(row)
        print(json.dumps(row), flush=True)
    worst = max(max(row["per_shot_relative_l2"]) for row in checks)
    pooled_worst = max(row["relative_l2"] for row in checks)
    result = {"schema": SCHEMA, "split": split, "status": "PASS" if pooled_worst <= max_error else "FAIL",
              "bank_content_sha256": manifest["bank_content_sha256"], "h_half_h_checks": checks,
              "maximum_pooled_pair_relative_l2": pooled_worst, "maximum_per_shot_relative_l2": worst,
              "pooled_pair_relative_l2_limit": max_error}
    atomic_json(directory / "QC.json", result)
    print(json.dumps({"qc": str(directory / "QC.json"), "status": result["status"], "split": split,
                      "parent_checks": len(checks), "maximum_pooled_pair_relative_l2": pooled_worst}), flush=True)
    return result["status"] == "PASS"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Bank root; use distinct train/validation subdirectories")
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--stage", choices=("archive", "qc"), default="archive")
    parser.add_argument("--max-parents", type=int)
    parser.add_argument("--qc-parents-per-h", type=int, default=1)
    parser.add_argument("--shot-batch", type=int, default=4)
    parser.add_argument("--qc-max-relative-l2", type=float, default=.05)
    args = parser.parse_args()
    release = Release(args.dataset)
    directory = (args.output / args.split).resolve()
    require(not directory.is_relative_to(release.root), "Place the directional bank outside the frozen release")
    require(1 <= args.shot_batch <= 4 and args.qc_parents_per_h > 0 and 0 < args.qc_max_relative_l2 < 1,
            "Invalid shot batch or half-step quality policy")
    directory.mkdir(parents=True, exist_ok=True)
    with split_lock(directory / ".writer.lock"):
        if args.stage == "archive":
            archive(release, directory, args.split, args.max_parents)
        else:
            require(args.max_parents is None, "Use --max-parents only while archiving")
            return 0 if check_half_steps(release, directory, args.split, args.qc_parents_per_h,
                                        args.shot_batch, args.qc_max_relative_l2) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
