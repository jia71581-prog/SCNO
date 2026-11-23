"""Helpers used by the public training entry points."""
from functools import lru_cache
import numpy as np
import torch
from run_sourced_cases import require

ARCHIVE_SCHEMA = "marmousi_lwc5_ordinary_full_pressure_v1"

FRAMES, SPACE = 801, (201, 201)

SPLITS = ("train", "validation")

WINDOWS = ((0, 100), (100, 200), (200, 400), (400, 600), (600, 801))

DEPTHS = ((0, 60), (60, 120), (120, 201))

class PressureArchive:
    """The ordinary-only, atomically completed pressure bank. Read labels lazily."""

    def __init__(self, folder, release):
        from tools.marmousi_target_dataset_refined.common import digest, read_json, sha_file

        self.root, self.release = folder.resolve(strict=True), release
        plan = read_json(self.root/"PLAN.json")
        status = read_json(self.root/"STATUS.json")
        self.plan_sha = digest(plan)
        scientific = {key: release.binding[key] for key in ("contract_hash", "dataset_recipe_hash", "manifest_hash")}
        require(plan["schema"] == status["schema"] == ARCHIVE_SCHEMA and status["status"] == "COMPLETE"
                and status["plan_sha256"] == self.plan_sha and plan["release_binding"] == scientific
                and plan["normalization_sha256"] == release.binding["normalization_file_sha256"]
                and plan["roster_sha256"] == release.binding["shot_roster_hash"]
                and plan["frames"] == FRAMES and plan["receiver_rms"] == release.rms
                and plan["stored_units"] == "raw_numeric_lwc_pressure" and plan["dtype"] == "<f4"
                and plan["pressure_shape"] == [FRAMES, *SPACE] and plan["b5_shape"] == [FRAMES, SPACE[1]]
                and plan["pressure_excludes_cpml_exterior"] is True and plan["exact_or_initial_or_fwi_media_used"] is False
                and status["ordinary_val_not_used_for_training"] is True and status["exact_initial_and_fwi_excluded"] is True
                and all(release.contract["shared_arrays"][key]["array_sha256"] == stamp for key, stamp in plan["geometry_array_sha256"].items()),
                "Use the committed ordinary 5 m pressure archive bound to this release")
        require(set(plan["models"]) == set(SPLITS), "Only ordinary TRAIN and VAL models may enter pressure selection")
        self.items, self.parents = {}, {}
        self._status_receipts = {item["relative_directory"]: item for item in status["shards"]}
        require(len(status["shards"]) == len(self._status_receipts) == 36, "The complete archived shot receipts differ")
        for split, expected in zip(SPLITS, (24, 12), strict=True):
            models = plan["models"][split]
            parents = self.parents[split] = set()
            records = self.items[split] = []
            for row in models:
                media = release.rows[row["model_id"]]
                parent = row["parent_group_id"]
                require(row["split"] == split and media["split"] == split and not media["is_exact_initial"] and not media["is_exact_target"]
                        and all(row[key] == media[key] for key in ("parent_group_id", "family", "velocity_sha256")) and parent not in parents
                        and len(row["shot_ids"]) == 2 and sorted(set(row["shot_ids"])) == sorted(row["shot_ids"])
                        and set(row["shot_ids"]) <= set(release.rosters[row["model_id"]]), "Invalid ordinary model, parent or source roster")
                parents.add(parent)
                for shot in row["shot_ids"]:
                    path = self.root/split/row["model_id"]/f"shot_{shot:02d}"
                    velocity = (release.root/media["velocity_path"]).resolve(strict=True)
                    require(path.resolve() == path and velocity.is_relative_to(release.root), "Pressure or velocity path leaves its committed source")
                    records.append(({**row, "shot_id": shot}, path, velocity))
            require(len(models) == len(parents) == expected//2 and status["shots_by_split"][split] == len(records) == expected,
                    "Need the fixed 24 ordinary TRAIN and 12 ordinary VAL pressure shots")
        require(self.parents["train"].isdisjoint(self.parents["validation"]), "A pressure parent crosses TRAIN and VAL")
        self.binding = {"schema": ARCHIVE_SCHEMA, "plan_sha256": self.plan_sha, "status_sha256": sha_file(self.root/"STATUS.json"),
                        "geometry_array_sha256": plan["geometry_array_sha256"], "shots": {k: len(v) for k, v in self.items.items()}}
        self._checked, self._energy = set(), {}

    def authorized_arrays(self):
        return {file for entries in self.items.values() for _, folder, velocity in entries
                for file in (folder/"pressure.npy", folder/"b5.npy", velocity)}

    @lru_cache(maxsize=3)
    def arrays(self, split, index):
        from tools.marmousi_target_dataset_refined.common import read_json, sha_file
        from evaluate_no_pde_sensitivity_v2 import array_sha

        row, folder, velocity_path = self.items[split][index]
        pressure_path, b5_path, receipt = folder/"pressure.npy", folder/"b5.npy", read_json(folder/"RECEIPT.json")
        if folder not in self._checked:
            require(receipt == self._status_receipts[folder.relative_to(self.root).as_posix()]
                    and receipt["schema"] == ARCHIVE_SCHEMA and receipt["status"] == "COMMITTED" and receipt["plan_sha256"] == self.plan_sha
                    and receipt["shot_id"] == row["shot_id"] and receipt["source_x_z_f0_delay_amp"] == self.release.sources[row["shot_id"]].tolist()
                    and all(receipt["row"][key] == row[key] for key in ("model_id", "parent_group_id", "split", "velocity_sha256"))
                    and receipt["qc"]["b5_parity_passed"] is True and receipt["qc"]["top_zero_all_times"] is True
                    and receipt["pressure_file_sha256"] == sha_file(pressure_path) and receipt["b5_file_sha256"] == sha_file(b5_path),
                    "Archived ordinary pressure shard or frozen source changed")
            self._checked.add(folder)
        pressure, b5 = (np.load(path, mmap_mode="r", allow_pickle=False) for path in (pressure_path, b5_path))
        velocity = np.load(velocity_path, allow_pickle=False)
        require(pressure.shape == (FRAMES, *SPACE) and b5.shape == (FRAMES, SPACE[1]) and velocity.shape == SPACE
                and all(value.dtype == np.float32 for value in (pressure, b5, velocity))
                and array_sha(velocity) == row["velocity_sha256"] and np.all(velocity > 0.),
                "Pressure labels or physical velocity axes changed")
        return pressure, b5, velocity

    def take(self, split, index, slots, device):
        pressure, b5, _ = self.arrays(split, index)
        return tuple(torch.as_tensor(values[slots].copy(), device=device).div_(self.release.rms) for values in (pressure, b5))

    def train_energy(self, index):
        if index not in self._energy:
            pressure, b5, _ = self.arrays("train", index)
            field = [[sum(np.square(pressure[lo:min(lo+16, last), a:b], dtype=np.float64).sum()
                          for lo in range(first, last, 16))/((last-first)*(b-a)*SPACE[1]*self.release.rms**2)
                      for a, b in DEPTHS] for first, last in WINDOWS]
            receiver = [np.square(b5[first:last], dtype=np.float64).mean()/self.release.rms**2 for first, last in WINDOWS]
            self._energy[index] = (np.maximum(field, .015*np.max(field)), np.maximum(receiver, .015*np.max(receiver)))
            require(all(np.isfinite(value).all() and np.min(value) > 0. for value in self._energy[index]), "Invalid ordinary TRAIN pressure energy")
        return self._energy[index]

def _blocks(total, size):
    return (list(range(first, min(first+size, total))) for first in range(0, total, size))
