"""Freeze and materialize 64 ordinary plus 32 signed TRAIN-only pressure records."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from generate_ordinary_pressure import (
    B5_SHAPE, DTYPE, FRAMES, PRESSURE_SHAPE, SCHEMA as ORDINARY_SCHEMA, SIGMA150_SHA,
    UNITS, array_sha, check_shot, commit_shot, read_shot, released_b5, require, shot_folder,
)
from make_direction_labels import EXCLUDED_MODELS, RECORD_UNITS, SCHEMA as BANK_SCHEMA, Release
from tools.marmousi_target_dataset_refined.common import (
    atomic_json, digest, read_json, save_array, sha_file, sync_directory,
)


SCHEMA = "marmousi_lwc5_expanded_train_full_pressure_v1"
POLICY = "ordinary64_signed32_parent_and_source_stratified_v1"
QUOTAS = dict(zip("ABCDEF", (11, 11, 11, 11, 10, 10)))
EXPECTED = {"train": 64, "train_signed": 32}
OLD_PLAN_SHA = "c118f0400f44064bc3217acbad6d94829f02fd0bc69fa885cdbd8673d34f0b3a"
RESERVE_SHA = {
    "signed_reserve": "a8512ac83de10ab7a1b3067f525e4e7708b58917682a0215114229249ee97c84",
    "coupled_plan": "f8b0a147b092439af27f413f1a33b04d6ed011ad9751dc06e6b7024ce3803864",
    "coupled_audit": "fbd83eead0049a2e96f2252f72058f97ce96e14da443b9fca835cf3d4338e0b0",
}
BANK_SHA = {
    "A": "c6b94cebd04eba2f4267a6937e1b7582b44684c36d1ea4f6326e8ecff06c593c",
    "F": "02fc775e9537fe529ddacd4e44cb32e77a813f81174ed6f7862842124154deb9",
}
SIDES = ("minus", "plus")


def ordinary_row(release, row):
    model = row["model_id"]
    velocity_sha = row["velocity_sha256"]
    require(row["split"] == "train" and model in release.splits["train"] and model not in EXCLUDED_MODELS
            and row["waveform_used_for_supervised_training"] is True and not row["is_exact_initial"]
            and not row["is_exact_target"] and not release.is_exact_anchor(velocity_sha) and velocity_sha != SIGMA150_SHA
            and row["velocity_path"] == f"velocities/{model}.npy",
            f"Ineligible model in expanded TRAIN bank: {model}")
    return {k: row[k] for k in ("model_id", "family", "parent_group_id", "velocity_sha256", "velocity_path", "split")}


def read_bank(release, path, family, forbidden):
    require(path.name == "manifest.json" and path.parent.name == "train" and sha_file(path) == BANK_SHA[family],
            f"Use the exact frozen ordinary {family} TRAIN direction manifest")
    manifest = read_json(path)
    records = manifest["records"]
    expected_pairs, expected_parents = (16, 8) if family == "A" else (50, 50)
    require(manifest["schema"] == BANK_SCHEMA and manifest["split"] == "train" and manifest["units"] == RECORD_UNITS
            and manifest["ordinary_splits_only"] is True and manifest["exact_targets_or_fwi_checkpoints_used"] is False
            and manifest["receiver_train_rms"] == release.rms and manifest["shot_roster_hash"] == release.roster_digest
            and manifest["normalization_file_sha256"] == release.norm_digest
            and all(manifest[k] == v for k, v in release.binding.items())
            and manifest["bank_content_sha256"] == digest({k: v for k, v in manifest.items() if k != "bank_content_sha256"})
            and expected_pairs == len(records) == len({r["pair_id"] for r in records})
            and len({r["parent_group_id"] for r in records}) == expected_parents, f"Changed {family} bank")
    declared_f = {r["pair_id"]: r for r in release.pair_plan("train")} if family == "F" else {}
    for record in records:
        key, parent = record["pair_id"], record["parent_group_id"]
        require(record["split"] == "train" and parent not in forbidden and record["release_binding"] == release.binding
                and record["shard"] == f"{key}.npz" and Path(key).name == key, f"Invalid TRAIN signed provenance: {key}")
        if family == "A":
            center = release.rows[record["center_model_id"]]
            ordinary_row(release, center)
            require(center["family"] == "A" and center["parent_group_id"] == parent
                    and center["velocity_sha256"] == record["center_model_release_sha256"]
                    and record["origin"] == "generated_from_released_A_parent" and record["h"] == .003
                    and record["direction_type"] in ("deep_tapered", "smooth_multiscale")
                    and release.roster[center["model_id"]] == record["shot_ids"]
                    and all(record[f"{side}_model_id"] == f"{key}_{side}" for side in SIDES), f"Changed A pair {key}")
        else:
            require(all(record[k] == v for k, v in declared_f[key].items()), f"Changed F pair {key}")
            for side in SIDES:
                ordinary_row(release, release.rows[record[f"{side}_model_id"]])
    return records


def stratum(release, shot):
    return min(5, int(6 * float(release.sources[shot, 0]) / 2000.))


def rank_pair(release, record, shots, chosen_bins, chosen_sources):
    left, right = shots
    return (sum(chosen_bins[stratum(release, shot)] for shot in shots), stratum(release, left) == stratum(release, right),
            sum(chosen_sources[shot] for shot in shots), digest([POLICY, record["pair_id"], shots]))


def choose_signed(release, banks, occupied):
    signed, bins, sources = [], Counter(), Counter()
    for family, controls in (("A", (True, False, True, False)), ("F", tuple(zip(release.h_levels, (True, False, True, False))))):
        for condition in controls:
            pool = []
            for r in banks[family]:
                if r["parent_group_id"] in occupied:
                    continue
                if family == "A":
                    initial = release.rows[r["center_model_id"]]["template_initial_sha256"]
                    eligible = r["direction_type"] == "deep_tapered" and (initial == SIGMA150_SHA) == condition
                else:
                    initial = r["initial_template_sha256"]
                    eligible = (r["h"], initial == SIGMA150_SHA) == condition
                if eligible:
                    pool.extend((r, (left, right), initial) for left in r["shot_ids"][:2] for right in r["shot_ids"][2:])
            require(pool, f"Insufficient fresh directional parents for {family} {condition}")
            record, shots, initial = min(pool, key=lambda x: rank_pair(release, x[0], x[1], bins, sources))
            key, parent = record["pair_id"], record["parent_group_id"]
            fields = ("pair_id", "parent_group_id", "h", "direction_type", "direction_seed")
            pair = {k: record[k] for k in fields}
            pair.update(family=family, split="train", shot_ids=list(shots), initial_template_sha256=initial,
                        full_bank_shot_ids=record["shot_ids"], bank_manifest_sha256=BANK_SHA[family], bank_shard=record["shard"],
                        bank_shard_sha256=record["shard_sha256"], endpoints={})
            if family == "A":
                pair["center_model_id"] = record["center_model_id"]
                pair["center_velocity_release_sha256"] = record["center_model_release_sha256"]
            for side in SIDES:
                relative = f"train_signed/{key}/{side}"
                pair["endpoints"][side] = {"model_id": record[f"{side}_model_id"],
                                           "velocity_sha256": record[f"{side}_velocity_sha256"],
                                           "relative_directory": relative, "velocity_path": f"{relative}/velocity.npy"}
            signed.append(pair)
            occupied.add(parent)
            bins.update(stratum(release, shot) for shot in shots)
            sources.update(shots)
    return signed


def choose_ordinary(release, occupied):
    pools, selected = {family: [] for family in QUOTAS}, []
    for model in release.splits["train"]:
        row = release.rows[model]
        if row["is_exact_initial"] or row["is_exact_target"] or row["parent_group_id"] in occupied:
            continue
        item = ordinary_row(release, row)
        pools[item["family"]].append((item, row["template_initial_sha256"], row.get("pair_sign", 0)))
    global_bins, sources, signs = Counter(), Counter(), Counter()
    for i, (family, quota) in enumerate(QUOTAS.items()):
        bins = Counter()
        for slot in range(quota):
            want_new = (i + slot) % 2 == 0
            options = ((row, seed, sign, shot) for row, seed, sign in pools[family]
                       if (seed == SIGMA150_SHA) == want_new and row["parent_group_id"] not in occupied
                       for shot in release.roster[row["model_id"]])

            def rank(option):
                row, _, sign, shot = option
                group = stratum(release, shot)
                return bins[group], global_bins[group], sources[shot], signs[sign], digest([POLICY, row["model_id"], shot])

            chosen = min(options, key=rank, default=None)
            require(chosen is not None, f"Insufficient coverage for ordinary family {family}, cohort={want_new}")
            row, seed, sign, shot = chosen
            selected.append(dict(row, shot_ids=[shot], initial_template_sha256=seed, parent_used_for_e12_val_b5_selection=False))
            occupied.add(row["parent_group_id"])
            bins[stratum(release, shot)] += 1
            global_bins[stratum(release, shot)] += 1
            sources[shot] += 1
            signs[sign] += 1
    return selected


def build_plan(release, old_path, reserve_path, bank_paths, extractor_sha):
    reserve_sha = sha_file(reserve_path)
    require(sha_file(old_path) == OLD_PLAN_SHA and reserve_sha in RESERVE_SHA.values(),
            "Use the frozen original 24/12 pressure PLAN and six-parent reserve receipt")
    old, reserve = read_json(old_path), read_json(reserve_path)
    if reserve_sha == RESERVE_SHA["coupled_plan"]:
        archive = reserve["ordinary_signed_archive"]
        require(all(archive["bank_manifest_sha256"][f"bank_{k.lower()}"]["train"] == v for k, v in BANK_SHA.items())
                and all(archive["release_binding"][k] == v for k, v in release.binding.items())
                and archive["reserved_signed_shard_files_opened"] == 0, "Reserve metadata disagrees with actual TRAIN banks")
        reserve_parents = archive["reserved_parent_groups"]
    elif reserve_sha == RESERVE_SHA["coupled_audit"]:
        reserve_parents = reserve["val_direction_roster"]["reserve_parents"]
    else:
        reserve_parents = reserve["reserved_parent_groups"]
    require(old["schema"] == ORDINARY_SCHEMA and old["release_binding"] == release.binding
            and old["shots_by_split"] == {"train": 24, "validation": 12} and len(set(reserve_parents)) == 6
            and old["reference_extractor_sha256"] == extractor_sha
            and old["receiver_rms"] == release.rms and old["roster_sha256"] == release.roster_digest
            and old["normalization_sha256"] == release.norm_digest and old["geometry_array_sha256"] == release.geometry,
            "Reviewed original pressure or reserve metadata changed")
    forbidden = {r["parent_group_id"] for r in release.rows.values() if r["split"] != "train"}
    require(set(reserve_parents) <= forbidden, "One of the frozen reserve parents entered TRAIN")
    old_parents = {r["parent_group_id"] for r in old["models"]["train"]}
    require(len(old_parents) == 12 and not old_parents & forbidden, "Previous ordinary TRAIN parents changed")
    banks = {family: read_bank(release, path, family, forbidden) for family, path in bank_paths.items()}
    occupied = forbidden | old_parents
    signed = choose_signed(release, banks, occupied)
    occupied.update(r["parent_group_id"] for r in banks["A"])
    ordinary = choose_ordinary(release, occupied)
    require(len(ordinary) == 64 and len(signed) == 8 and Counter(p["family"] for p in signed) == {"A": 4, "F": 4},
            "Expanded membership is incomplete")
    axes = ("frames", "time_axis_s", "native_grid_m", "pressure_axes", "pressure_shape", "b5_shape", "dtype",
            "stored_units", "pressure_excludes_cpml_exterior", "reference")
    return {"schema": SCHEMA, "policy": POLICY, "dataset": str(release.root), **{k: old[k] for k in axes},
            "release_binding": release.binding, "receiver_rms": release.rms, "geometry_array_sha256": release.geometry,
            "normalization_sha256": release.norm_digest, "roster_sha256": release.roster_digest,
            "previous_ordinary_plan_sha256": OLD_PLAN_SHA, "reserve_receipt_sha256": reserve_sha,
            "bank_manifest_sha256": BANK_SHA, "shots_by_split": EXPECTED, "models": {"train": ordinary}, "signed_pairs": signed,
            "ordinary_family_quotas": QUOTAS, "source_stratum_edges_m": [2000 * i / 6 for i in range(7)],
            "generator_source_sha256": sha_file(__file__), "reference_extractor_sha256": extractor_sha,
            "ordinary_parent_count": len(ordinary), "signed_parent_count": len(signed),
            "exact_or_initial_or_fwi_media_used": False, "validation_or_reserved_waveforms_opened": False}


def read_signed_endpoints(release, pair, bank_paths):
    path = bank_paths[pair["family"]].parent / pair["bank_shard"]
    require(not path.is_symlink() and sha_file(path) == pair["bank_shard_sha256"], f"Changed signed shard: {path}")
    with np.load(path, allow_pickle=False) as data:
        require(data["h"].shape == () and float(data["h"]) == pair["h"]
                and data["shot_ids"].dtype == np.int64 and data["shot_ids"].tolist() == pair["full_bank_shot_ids"],
                "Signed shard geometry or h changed")
        endpoints = {}
        for side in SIDES:
            member = pair["endpoints"][side]
            velocity, raw = data[f"velocity_{side}"], data[f"b5_{side}"]
            require(velocity.shape == (201, 201) and velocity.dtype == DTYPE and np.isfinite(velocity).all()
                    and array_sha(velocity) == member["velocity_sha256"] and not release.is_exact_anchor(member["velocity_sha256"])
                    and member["velocity_sha256"] != SIGMA150_SHA and raw.shape == (4, *B5_SHAPE) and raw.dtype == DTYPE
                    and np.isfinite(raw).all(), f"Wrong frozen {side} input or archived B5")
            shots = {s: raw[pair["full_bank_shot_ids"].index(s)] for s in pair["shot_ids"]}
            if pair["family"] == "F":
                row = ordinary_row(release, release.rows[member["model_id"]])
                observed, source_sha = released_b5(release, dict(row, shot_ids=pair["shot_ids"]))
                require(all(np.array_equal(shots[s], observed[s]) for s in shots), "F bank and release B5 disagree")
            else:
                source_sha = pair["bank_shard_sha256"]
            endpoints[side] = (velocity, shots, source_sha)
    require(pair["endpoints"]["minus"]["velocity_sha256"] != pair["endpoints"]["plus"]["velocity_sha256"], "Signed velocities are identical")
    return endpoints


def save_signed_velocity(root, pair, side, velocity):
    member = pair["endpoints"][side]
    path = root / member["velocity_path"]
    if not path.exists():
        save_array(path, velocity)
    saved = np.load(path, allow_pickle=False)
    require(not path.is_symlink() and saved.shape == (201, 201) and saved.dtype == DTYPE
            and array_sha(saved) == member["velocity_sha256"], "Committed signed velocity changed")
    return sha_file(path)


def signed_receipt(target, row, shot, base, pressure, b5, observed, source_sha, solver, tolerance):
    relative, energy = check_shot(pressure, b5, observed, tolerance)
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), "Refuse to replace a committed signed pressure shot")
    with TemporaryDirectory(prefix=f".{target.name}.partial-", dir=target.parent) as directory:
        stage = Path(directory)
        arrays = {}
        for key, value in (("pressure", pressure), ("b5", b5)):
            file = stage / f"{key}.npy"
            save_array(file, value)
            arrays.update({f"{key}_file_sha256": sha_file(file), f"{key}_array_sha256": array_sha(value)})
        result = dict(base, **arrays, schema=SCHEMA, status="COMMITTED", row=row, shot_id=shot, stored_units=UNITS,
                      pressure_shape=list(PRESSURE_SHAPE), b5_shape=list(B5_SHAPE), dtype=DTYPE.str,
                      published_b5_sha256=source_sha, published_b5_shot_array_sha256=array_sha(observed),
                      b5_replay_relative_l2=relative, b5_parity_tolerance=tolerance, solver=solver,
                      qc={"top_zero_all_times": True, "pressure_initial_zero": True, "b5_initial_zero": True,
                          "published_b5_energy": energy, "pressure_finite": True, "b5_parity_passed": True})
        atomic_json(stage / "RECEIPT.json", result)
        sync_directory(stage)
        stage.rename(target)
        sync_directory(target.parent)
    return result


def read_committed_signed(target, row, shot, base, tolerance):
    require(not target.is_symlink() and target.is_dir(), "Unexpected signed shard path")
    result = read_json(target / "RECEIPT.json")
    require(result["schema"] == SCHEMA and result["status"] == "COMMITTED" and result["row"] == row
            and result["shot_id"] == shot and all(result[k] == v for k, v in base.items())
            and 0 <= result["b5_replay_relative_l2"] <= result["b5_parity_tolerance"] == tolerance,
            "Committed signed receipt changed provenance or parity")
    for key, shape in (("pressure", PRESSURE_SHAPE), ("b5", B5_SHAPE)):
        file = target / f"{key}.npy"
        require(not file.is_symlink() and sha_file(file) == result[f"{key}_file_sha256"], "Changed signed pressure or B5 file")
        value = np.load(file, mmap_mode="r", allow_pickle=False)
        require(value.shape == shape and value.dtype == DTYPE and array_sha(value) == result[f"{key}_array_sha256"],
                "Changed signed pressure or B5 values")
    return result


def entries(plan):
    for pair in plan["signed_pairs"]:
        for side in SIDES:
            member = pair["endpoints"][side]
            row = dict(member, split="train_signed", origin_split="train", family=pair["family"],
                       pair_id=pair["pair_id"], parent_group_id=pair["parent_group_id"], h=pair["h"], side=side,
                       pair_sign=-1 if side == "minus" else 1, shot_ids=pair["shot_ids"],
                       bank_manifest_sha256=pair["bank_manifest_sha256"], bank_shard_sha256=pair["bank_shard_sha256"])
            yield row, pair
    for row in plan["models"]["train"]:
        yield row, None


def publish(root, plan_sha, receipts):
    counts = {split: sum(r["row"]["split"] == split for r in receipts) for split in EXPECTED}
    status = {"schema": SCHEMA, "status": "COMPLETE" if counts == EXPECTED else "PARTIAL", "plan_sha256": plan_sha,
              "shots_by_split": counts, "shots": sum(counts.values()), "shards": receipts,
              "raw_pressure_bytes": sum(counts.values()) * int(np.prod(PRESSURE_SHAPE)) * DTYPE.itemsize,
              "validation_or_reserved_waveforms_opened": False, "exact_initial_and_fwi_excluded": True}
    atomic_json(root / "STATUS.json", status)
    return status


def run(args):
    from score_fullfield_e12 import LWCReference, NativeFinePressure, run_lock
    import score_fullfield_e12 as core

    release = Release(args.dataset)
    root = args.output.resolve()
    banks = {"A": args.bank_a.resolve(strict=True), "F": args.bank_f.resolve(strict=True)}
    protected = [release.root, args.ordinary_plan.resolve(strict=True).parent, *(p.parent.parent for p in banks.values())]
    require(not any(root.is_relative_to(path) for path in protected), "Write the expansion outside the frozen release and earlier banks")
    plan = build_plan(release, args.ordinary_plan, args.reserve_receipt, banks, sha_file(core.__file__))
    plan_sha = digest(plan)
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        plan_path = root / "PLAN.json"
        if plan_path.exists():
            require(read_json(plan_path) == plan, "Resume requires exactly the frozen expanded plan")
        else:
            require(not any(root.glob("train*")), "Cannot start a new plan over existing shot files")
            atomic_json(plan_path, plan)
        print(json.dumps({"status": "PLAN_VERIFIED", "plan_sha256": plan_sha, "shots_by_split": EXPECTED}), flush=True)
        if args.plan_only:
            return

        import torch

        tolerance = release.contract["solver"]["cpu_reference_equivalence_tolerance"]
        receipts, completed, endpoints, engines = [], set(), {}, {}
        for row, pair in entries(plan):
            for shot in row["shot_ids"]:
                if pair is None:
                    target = shot_folder(root, row, shot)
                    if target.exists():
                        receipts.append(read_shot(root, row, shot, plan_sha, release))
                    else:
                        continue
                else:
                    target = root / row["relative_directory"] / f"shot_{shot:02d}"
                    if not target.exists():
                        continue
                    saved = root / row["velocity_path"]
                    require(not saved.is_symlink() and array_sha(np.load(saved, allow_pickle=False)) == row["velocity_sha256"],
                            "Committed signed velocity file changed")
                    base = {"plan_sha256": plan_sha, "relative_directory": target.relative_to(root).as_posix(),
                            "source_x_z_f0_delay_amp": release.sources[shot].tolist(), "velocity_file_sha256": sha_file(saved)}
                    receipts.append(read_committed_signed(target, row, shot, base, tolerance))
                completed.add((row["model_id"], shot))
        new, reference = 0, None
        with torch.no_grad():
            for row, pair in entries(plan):
                pending = [s for s in row["shot_ids"] if (row["model_id"], s) not in completed]
                if args.max_new is not None:
                    pending = pending[:max(0, args.max_new - new)]
                if not pending:
                    continue
                if pair is None:
                    velocity = release.velocity(row["model_id"], "train", row["family"])
                    observed, source_sha = released_b5(release, row)
                else:
                    key = pair["pair_id"]
                    if key not in endpoints:
                        endpoints[key] = read_signed_endpoints(release, pair, banks)
                    velocity, observed, source_sha = endpoints[key][row["side"]]
                    velocity_file_sha = save_signed_velocity(root, pair, row["side"], velocity)
                if reference is None:
                    require(torch.cuda.is_available() and torch.device(args.device).type == "cuda", "Offline replay requires the assigned CUDA card")
                    reference = LWCReference(release.root, device=args.device, build_dir=root / "fine_cuda")
                batch = len(pending)
                if batch not in engines:
                    engines[batch] = NativeFinePressure(reference.forward_contract, reference.cache, device=str(reference.device), batch_size=batch)
                pressure, b5, stats = engines[batch].replay(velocity, pending, FRAMES)
                pressure = pressure.cpu().numpy()
                require(pressure.shape == (batch, *PRESSURE_SHAPE) and b5.shape == (batch, *B5_SHAPE), "Wrong pressure batch shape")
                for i, shot in enumerate(pending):
                    if pair is None:
                        receipt = commit_shot(root, row, shot, plan_sha, release, pressure[i], b5[i], observed[shot], source_sha, stats)
                    else:
                        target = root / row["relative_directory"] / f"shot_{shot:02d}"
                        base = {"plan_sha256": plan_sha, "relative_directory": target.relative_to(root).as_posix(),
                                "source_x_z_f0_delay_amp": release.sources[shot].tolist(), "velocity_file_sha256": velocity_file_sha}
                        receipt = signed_receipt(target, row, shot, base, pressure[i], b5[i], observed[shot], source_sha, stats, tolerance)
                    receipts.append(receipt)
                    new += 1
                    status = publish(root, plan_sha, receipts)
                    print(json.dumps({"stage": "SHOT_COMMITTED", "model_id": row["model_id"], "shot_id": shot,
                                      "b5_replay_relative_l2": receipt["b5_replay_relative_l2"],
                                      "shots_by_split": status["shots_by_split"]}), flush=True)
        status = publish(root, plan_sha, receipts)
        print(json.dumps({k: status[k] for k in ("status", "plan_sha256", "shots_by_split", "raw_pressure_bytes")}), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    for key in ("dataset", "ordinary-plan", "reserve-receipt", "bank-a", "bank-f", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    if args.max_new is not None and args.max_new <= 0:
        parser.error("--max-new must be positive")
    run(args)


if __name__ == "__main__":
    main()
