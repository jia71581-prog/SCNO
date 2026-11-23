"""Read exact 401-frame pressure samples from the centered homogeneous/two-layer release."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from full_axis_pressure import FULL_TIMES_S, GRID, axis_with_initial_condition
from native400_runtime.feature_enhanced_objective_20261022 import SampledLabels
from native400_runtime.style_marm_joint_data_20261025 import sha


class CenteredUniformLayeredData:
    def __init__(self, root):
        self.root = Path(root).resolve()
        marker = self.root / "DATASET_COMPLETE.json"
        complete = json.loads(marker.read_text())
        if (complete["schema"] != "SCNO_CENTERED_UL1000_V1" or complete["status"] != "COMPLETE"
                or complete["families"] != {"uniform": 500, "layered": 500}
                or complete["shape"] != [1000, 401, 201, 201]):
            raise ValueError("expected the balanced centered TRAIN release")
        artifacts = complete["artifacts"]
        for name, expected in artifacts.items():
            if sha(self.root / name) != expected:
                raise ValueError(f"published artifact changed: {name}")
        self.records = json.loads((self.root / "RECORDS.json").read_text())
        if not (len(self.records) == len({r["sample_id"] for r in self.records}) == 1000 and
                all(r["record_id"] == i and r["split"] == "train" for i, r in enumerate(self.records))):
            raise ValueError("the publication requires 1,000 ordered, distinct TRAIN records")
        self.source = np.asarray(complete["source_parameters"], np.float32)
        expected_source = np.array([1000., 1000., 15., .1, 1.], dtype=np.float32)
        np.testing.assert_array_equal(self.source, expected_source)
        with h5py.File(self.root / "dataset_train.h5", "r") as h5:
            times, insert = axis_with_initial_condition(h5["time_field_s"][:])
            if insert or h5["pressure201"].shape != tuple(complete["shape"]):
                raise ValueError("the published physical pressure axis differs")
            np.testing.assert_array_equal(h5["record_ids"][:], np.arange(1000))
        self.times = torch.from_numpy(times.astype(np.float32))
        norm = json.loads((self.root / "normalization_ul.json").read_text())
        self.normalization = {**norm, "velocity_mean_mps": norm["velocity_center_mps"],
                              "velocity_std_mps": norm["velocity_scale_mps"], "record_count": len(self.records)}
        self.binding = {"root": str(self.root), "complete_sha256": sha(marker), "file_hashes": artifacts,
                        "train_count": len(self.records), "labels_normalized": False,
                        "val_test_labels_read": False, "native_selection": "0:801:2"}

    def batch(self, ids, frames):
        index, = ids
        record = self.records[index]
        path = Path(record["pressure_record"])
        info = path.stat()
        if info.st_size != record["size_bytes"] or info.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"sealed TRAIN source changed: {path}")
        with h5py.File(path, "r") as h5:
            if h5["wavefield"].shape != (801, *GRID) or h5["wavefield"].dtype != np.float32:
                raise ValueError(f"invalid archived pressure storage: {path}")
            velocity = h5["velocity201"][:].astype(np.float32)
            source = h5["source_parameters"][:].astype(np.float32)
            selected = h5["wavefield"][np.asarray(frames, dtype=np.int64) * 2]
        np.testing.assert_array_equal(source, self.source)
        if not (velocity.shape == GRID and np.isfinite(velocity).all() and (velocity > 0).all()):
            raise ValueError("expected one finite positive velocity grid")
        return {"c": torch.from_numpy(velocity)[None, None], "source": torch.from_numpy(source)[None],
                "labels": SampledLabels(frames, torch.from_numpy(selected)[None])}
