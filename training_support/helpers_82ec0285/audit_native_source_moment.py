"""Helpers used by the public training entry points."""
import numpy as np
from evaluate_no_pde_sensitivity_v2 import array_sha, require

def select_cases(data):
    a, e = ([row for row in data.rows.values() if row["split"] == "train" and row["family"] == family
             and not row["is_exact_initial"] and not row["is_exact_target"]] for family in ("A", "E"))
    require(a and e, "The source diagnostic needs an ordinary TRAIN A and E medium")
    by_family = [{shot: sorted((row for row in records if shot in data.rosters[row["model_id"]]), key=lambda row: row["model_id"])
                  for shot in range(len(data.sources))} for records in (a, e)]
    center = .5 * (len(data.sources) - 1)
    shared = [shot for shot in range(len(data.sources)) if all(group[shot] for group in by_family)]
    order = lambda shot: (abs(shot - center), shot)
    shots = [min(shared, key=order)] * 2 if shared else [min((shot for shot, rows in group.items() if rows), key=order) for group in by_family]
    cases = [(group[shot][0], shot) for group, shot in zip(by_family, shots)]
    require(len({row["parent_group_id"] for row, _ in cases}) == 2, "A and E must be distinct TRAIN parents")
    return cases

def validate_velocity(data, row):
    path = (data.root / row["velocity_path"]).resolve(strict=True)
    require(path.is_relative_to(data.root) and row["split"] == "train" and not row["is_exact_initial"]
            and not row["is_exact_target"], "Use only a registered ordinary TRAIN velocity")
    v = np.load(path, allow_pickle=False)
    stamp = array_sha(v)
    templates = data.contract["templates"]
    require(v.shape == (201, 201) and v.dtype == np.float32 and np.isfinite(v).all() and (v > 0.).all()
            and stamp == row["velocity_sha256"] and stamp not in {templates["truth_array_sha256"], templates["initial_array_sha256"]},
            "Velocity changed or coincides with a reserved template")
    return v
