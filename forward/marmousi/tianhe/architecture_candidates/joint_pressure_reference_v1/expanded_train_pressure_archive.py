"""Stream the committed TRAIN-only pressure expansion beside the old fixed VAL bank."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from pathlib import Path

import numpy as np
import torch

import generate_expanded_train_pressure as generator
from make_direction_labels import Release
from signed_full_pressure_loss import b5_signed_energies, body_signed_energies
import train_joint_pressure as joint
from tools.marmousi_target_dataset_refined.common import digest, read_json, sha_file


SCHEMA = "marmousi_expanded_train_pressure_reader_v1"
PLAN_FILE_SHA = "93c0223270e2da42d18a933ab4d1ff16f0fb87c942e7ec83c9fc39c375da939c"
require = joint.require


@dataclass(frozen=True)
class SignedPressureShot:
    pair_id: str
    parent_group_id: str
    family: str
    h: float
    shot_id: int
    frames: tuple
    receiver_rms: float
    velocity_endpoints: torch.Tensor
    raw_pressure_endpoints: torch.Tensor
    raw_b5_endpoints: torch.Tensor

    def reference_secants(self):
        # Subtract the archived minus/plus in float64 before scaling.
        factor = 2.*self.h*self.receiver_rms
        return tuple((labels[1].double()-labels[0].double())/factor
                     for labels in (self.raw_pressure_endpoints, self.raw_b5_endpoints))


def safe_path(root, relative):
    current = root/relative
    path = current.resolve(strict=True)
    require(path == current and path.is_relative_to(root), "Pressure arrays must stay in the registered partition")
    return path


def frozen_metadata(root, release, old):
    plan_file, status_file = root/"PLAN.json", root/"STATUS.json"
    plan, status = read_json(plan_file), read_json(status_file)
    plan_stamp = sha_file(plan_file)
    plan_sha = digest(plan)
    ordinary_hashes = {key: release.binding[key] for key in ("contract_hash", "dataset_recipe_hash", "manifest_hash")}
    old_plan_sha = sha_file(old.root/"PLAN.json")
    require(plan_stamp == PLAN_FILE_SHA and plan["schema"] == status["schema"] == generator.SCHEMA
            and status["status"] == "COMPLETE" and status["plan_sha256"] == plan_sha
            and plan["shots_by_split"] == status["shots_by_split"] == generator.EXPECTED
            and status["shots"] == len(status["shards"]) == 96
            and plan["release_binding"] == ordinary_hashes and plan["receiver_rms"] == release.rms
            and plan["normalization_sha256"] == release.binding["normalization_file_sha256"]
            and plan["roster_sha256"] == release.binding["shot_roster_hash"]
            and plan["geometry_array_sha256"] == old.binding["geometry_array_sha256"]
            and plan["previous_ordinary_plan_sha256"] == old_plan_sha == generator.OLD_PLAN_SHA
            and plan["generator_source_sha256"] == sha_file(generator.__file__)
            and plan["reserve_receipt_sha256"] in generator.RESERVE_SHA.values()
            and plan["bank_manifest_sha256"] == generator.BANK_SHA
            and plan["frames"] == joint.FRAMES and plan["pressure_shape"] == list(generator.PRESSURE_SHAPE)
            and plan["b5_shape"] == list(generator.B5_SHAPE) and plan["dtype"] == generator.DTYPE.str
            and plan["stored_units"] == generator.UNITS and plan["pressure_excludes_cpml_exterior"] is True
            and plan["exact_or_initial_or_fwi_media_used"] is False and status["exact_initial_and_fwi_excluded"] is True
            and plan["validation_or_reserved_waveforms_opened"] is status["validation_or_reserved_waveforms_opened"] is False
            and set(plan["models"]) == {"train"} and len(plan["models"]["train"]) == plan["ordinary_parent_count"] == 64
            and len(plan["signed_pairs"]) == plan["signed_parent_count"] == 8,
            "Use the complete 64 ordinary +32 signed TRAIN expansion bound to the original ordinary bank")
    receipts = {row["relative_directory"]: row for row in status["shards"]}
    require(len(receipts) == 96, "Every expanded label must have a distinct committed receipt")
    return plan, receipts, {"plan_sha256": plan_sha, "plan_file_sha256": plan_stamp,
                            "status_file_sha256": sha_file(status_file)}


class ExpandedPressureArchive:
    """Give existing trainers ordinary .items/.take and a separate matched signed API."""

    def __init__(self, old, root, *, bank_a, bank_f):
        self.old, self.release = old, old.release
        self.root = Path(root).resolve(strict=True)
        release = self.release
        plan, self.receipts, stamp = frozen_metadata(self.root, release, old)
        bank_release = Release(release.root)
        forbidden = {row["parent_group_id"] for row in release.rows.values() if row["split"] != "train"}
        require(bank_release.binding == plan["release_binding"] and bank_release.rms == release.rms
                and old.parents["validation"] <= forbidden, "The frozen TRAIN roles no longer match the ordinary release")
        self.bank_paths = {"A": Path(bank_a).resolve(strict=True), "F": Path(bank_f).resolve(strict=True)}
        bank_records = {family: {row["pair_id"]: row for row in generator.read_bank(bank_release, path, family, forbidden)}
                        for family, path in self.bank_paths.items()}
        self.pairs = tuple(plan["signed_pairs"])
        require(Counter(pair["family"] for pair in self.pairs) == {"A": 4, "F": 4} and
                Counter(row["family"] for row in plan["models"]["train"]) == generator.QUOTAS,
                "Expansion must keep its frozen family allocation")
        for pair in self.pairs:
            bank = bank_records[pair["family"]][pair["pair_id"]]
            require(pair["split"] == "train" and pair["parent_group_id"] not in forbidden
                    and all(pair[key] == bank[key] for key in ("h", "pair_id", "parent_group_id", "direction_type", "direction_seed"))
                    and isfinite(pair["h"]) and pair["h"] > 0.
                    and pair["bank_manifest_sha256"] == generator.BANK_SHA[pair["family"]]
                    and pair["bank_shard"] == bank["shard"] and pair["bank_shard_sha256"] == bank["shard_sha256"]
                    and pair["full_bank_shot_ids"] == bank["shot_ids"]
                    and len(pair["shot_ids"]) == len(set(pair["shot_ids"])) == 2 and set(pair["shot_ids"]) <= set(bank["shot_ids"])
                    and set(pair["endpoints"]) == set(generator.SIDES), "The signed expansion must use the actual frozen TRAIN pair")
            for side, endpoint in pair["endpoints"].items():
                directory = f"train_signed/{pair['pair_id']}/{side}"
                require(endpoint["relative_directory"] == directory and endpoint["velocity_path"] == f"{directory}/velocity.npy"
                        and endpoint["model_id"] == bank[f"{side}_model_id"]
                        and endpoint["velocity_sha256"] == bank[f"{side}_velocity_sha256"],
                        "A signed endpoint changed physical medium or subtraction sign")
            require(pair["endpoints"]["minus"]["velocity_sha256"] != pair["endpoints"]["plus"]["velocity_sha256"],
                    "Signed endpoint velocities must differ")
        ordinary_parents = [row["parent_group_id"] for row in plan["models"]["train"]]
        signed_parents = [pair["parent_group_id"] for pair in self.pairs]
        require(len(set(ordinary_parents+signed_parents)) == 72
                and not set(ordinary_parents+signed_parents) & (old.parents["train"] | forbidden),
                "New ordinary, signed and legacy parent groups cannot overlap")
        self._checked = set()
        extra, signed = [], {}
        planned = set()
        for row, pair in generator.entries(plan):
            if pair is None:
                source = generator.ordinary_row(bank_release, release.rows[row["model_id"]])
                require(all(row[key] == value for key, value in source.items()) and len(row["shot_ids"]) == 1
                        and set(row["shot_ids"]) <= set(release.rosters[row["model_id"]]), "Invalid expanded ordinary media or source")
            for shot in row["shot_ids"]:
                if pair is None:
                    relative = generator.shot_folder(self.root, row, shot).relative_to(self.root).as_posix()
                    velocity = safe_path(release.root, row["velocity_path"])
                else:
                    relative = f"{row['relative_directory']}/shot_{shot:02d}"
                    velocity = safe_path(self.root, row["velocity_path"])
                folder = safe_path(self.root, relative)
                registered = self.receipts[relative]
                require(relative not in planned and registered["row"] == row and registered["shot_id"] == shot
                        and registered["source_x_z_f0_delay_amp"] == release.sources[shot].tolist()
                        and registered["plan_sha256"] == stamp["plan_sha256"] and registered["status"] == "COMMITTED"
                        and registered["schema"] == (generator.ORDINARY_SCHEMA if pair is None else generator.SCHEMA)
                        and registered["qc"]["b5_parity_passed"] is True and registered["qc"]["top_zero_all_times"] is True,
                        "The committed pressure shot identity, source, or label QC changed")
                item = ({**row, "shot_id": shot}, folder, velocity)
                if pair is None:
                    extra.append(item)
                else:
                    signed[(pair["pair_id"], row["side"], shot)] = item
                planned.add(relative)
        require(len(extra) == 64 and len(signed) == 32 and planned == set(self.receipts),
                "Missing a registered ordinary or signed pressure shot")
        self._signed = signed
        self.items = {"train": [*old.items["train"], *extra], "validation": old.items["validation"]}
        self.parents = {"train": old.parents["train"] | set(ordinary_parents), "validation": old.parents["validation"]}
        self.signed_parents = frozenset(signed_parents)
        self.binding = {"schema": SCHEMA, "old": old.binding, "expanded": stamp, "banks": generator.BANK_SHA,
                        "shots": {"train": len(self.items["train"]), "validation": len(self.items["validation"]), "signed_endpoints": 32},
                        "ordinary_training_parent_count": len(self.parents["train"]), "signed_parent_count": len(self.signed_parents),
                        "reserve_or_test_values_used": False}

    @lru_cache(maxsize=6)
    def _new_arrays(self, folder, velocity_file):
        relative = folder.relative_to(self.root).as_posix()
        receipt = self.receipts[relative]
        pressure = safe_path(self.root, f"{relative}/pressure.npy")
        b5 = safe_path(self.root, f"{relative}/b5.npy")
        if folder not in self._checked:
            require(receipt == read_json(safe_path(self.root, f"{relative}/RECEIPT.json"))
                    and receipt["pressure_file_sha256"] == sha_file(pressure) and receipt["b5_file_sha256"] == sha_file(b5)
                    and ("velocity_file_sha256" not in receipt or receipt["velocity_file_sha256"] == sha_file(velocity_file)),
                    "An expanded pressure, B5, or signed velocity payload changed")
            self._checked.add(folder)
        p, trace = (np.load(path, mmap_mode="r", allow_pickle=False) for path in (pressure, b5))
        velocity = np.load(velocity_file, allow_pickle=False)
        require(p.shape == generator.PRESSURE_SHAPE and trace.shape == generator.B5_SHAPE and velocity.shape == joint.SPACE
                and all(arr.dtype == generator.DTYPE for arr in (p, trace, velocity))
                and bool(np.isfinite(velocity).all()) and bool((velocity > 0.).all())
                and generator.array_sha(velocity) == receipt["row"]["velocity_sha256"], "Wrong committed expanded pressure axes or velocity")
        return p, trace, velocity

    def arrays(self, split, index):
        _, folder, velocity = self.items[split][index]
        if split == "validation" or index < len(self.old.items["train"]):
            return self.old.arrays(split, index)
        return self._new_arrays(folder, velocity)

    def take(self, split, index, slots, device):
        pressure, b5, _ = self.arrays(split, index)
        return tuple(torch.as_tensor(values[slots].copy(), device=device).div_(self.release.rms) for values in (pressure, b5))

    @lru_cache(maxsize=8)
    def _signed_b5(self, pair_index):
        pair = self.pairs[pair_index]
        path = safe_path(self.bank_paths[pair["family"]].parent, pair["bank_shard"])
        require(sha_file(path) == pair["bank_shard_sha256"], "The selected released signed B5 shard changed")
        with np.load(path, allow_pickle=False) as data:
            require(float(data["h"]) == pair["h"] and data["shot_ids"].tolist() == pair["full_bank_shot_ids"],
                    "The released signed B5 scale or source axis changed")
            traces = np.stack([data[f"b5_{side}"] for side in generator.SIDES])
        require(traces.shape == (2, len(pair["full_bank_shot_ids"]), *generator.B5_SHAPE)
                and traces.dtype == generator.DTYPE and bool(np.isfinite(traces).all()), "Invalid signed B5 endpoint arrays")
        for i, side in enumerate(generator.SIDES):
            for shot in pair["shot_ids"]:
                _, folder, _ = self._signed[(pair["pair_id"], side, shot)]
                stamp = self.receipts[folder.relative_to(self.root).as_posix()]["published_b5_shot_array_sha256"]
                require(generator.array_sha(traces[i, pair["full_bank_shot_ids"].index(shot)]) == stamp,
                        "The published signed B5 differs from the full-pressure generator's original reference")
        return traces

    def signed_shot(self, pair_index, shot, slots, device):
        pair = self.pairs[pair_index]
        require(shot in pair["shot_ids"], "Use the same committed physical source for both signs")
        slots = list(slots)
        require(slots and all(type(slot) is int and 0 <= slot < joint.FRAMES for slot in slots)
                and all(a < b for a, b in zip(slots, slots[1:])), "Signed endpoints need matching sorted native frame indices")
        endpoints = []
        for side in generator.SIDES:
            _, folder, velocity = self._signed[(pair["pair_id"], side, shot)]
            pressure, _, medium = self._new_arrays(folder, velocity)
            endpoints.append((medium, pressure[slots].copy()))
        arrays = [torch.as_tensor(np.stack(values), device=device) for values in zip(*endpoints, strict=True)]
        source_slot = pair["full_bank_shot_ids"].index(shot)
        released_b5 = torch.as_tensor(self._signed_b5(pair_index)[:, source_slot, slots].copy(), device=device)
        return SignedPressureShot(pair["pair_id"], pair["parent_group_id"], pair["family"], pair["h"],
                                  shot, tuple(slots), self.release.rms, *arrays, released_b5)


@torch.no_grad()
def signed_training_energies(archive, lifter, device, *, block=16):
    """Freeze TRAIN-only pooled sign-direction energies before selecting any weights."""
    body, b5, count = 0., 0., 0
    for index, pair in enumerate(archive.pairs):
        for shot in pair["shot_ids"]:
            for slots in joint._blocks(joint.FRAMES, block):
                label = archive.signed_shot(index, shot, slots, device)
                pressure, trace = label.reference_secants()
                body += float(lifter.project_null(pressure).square().sum())
                b5 += float(trace.square().sum())
            count += 1
    require(count == 16 and isfinite(body) and isfinite(b5) and min(body, b5) > 0.,
            "Need valid pooled energy from 16 matched TRAIN directions")
    return {"mean_signed_body_shot_energy": body/count, "mean_signed_b5_shot_energy": b5/count,
            "mean_signed_body_point_energy": body/(count*int(np.prod(generator.PRESSURE_SHAPE))),
            "mean_signed_b5_sample_energy": b5/(count*int(np.prod(generator.B5_SHAPE))),
            "physical_shot_pairs": count, "signed_endpoint_count": 2*count, "validation_used": False}


def separated_secant_losses(lifter, label, normalized_pressure_endpoints, normalized_b5_endpoints, count, energies):
    """Compute independent null-space pressure and prescribed B5 secant losses."""
    require(normalized_pressure_endpoints.shape == label.raw_pressure_endpoints.shape
            and normalized_b5_endpoints.shape == label.raw_b5_endpoints.shape
            and len(label.frames) == count*len(joint.WINDOWS)
            and all(first <= slot < stop for index, (first, stop) in enumerate(joint.WINDOWS)
                    for slot in label.frames[index*count:(index+1)*count]),
            "A signed update must use the same stratified times, physical source and minus/plus order")
    ref_p = label.raw_pressure_endpoints.double()/label.receiver_rms
    ref_b5 = label.raw_b5_endpoints.double()/label.receiver_rms
    body, receiver = None, None
    for index, (first, stop) in enumerate(joint.WINDOWS):
        selected = slice(index*count, (index+1)*count)
        weight = (stop-first)/count
        p, y = normalized_pressure_endpoints[:, selected], ref_p[:, selected]
        b, d = normalized_b5_endpoints[:, selected], ref_b5[:, selected]
        local_body = body_signed_energies(lifter, *p, *y, label.h, sample_weights=weight)
        local_receiver = b5_signed_energies(*b, *d, label.h, sample_weights=weight)
        body = local_body if body is None else body+local_body
        receiver = local_receiver if receiver is None else receiver+local_receiver
    return {"signed_body": body.normalized(energies["mean_signed_body_point_energy"]),
            "signed_b5": receiver.normalized(energies["mean_signed_b5_sample_energy"])}
