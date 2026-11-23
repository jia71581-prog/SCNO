"""Read the committed Tianhe single-shot data and apply its frozen B5 observation."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tools.marmousi_target_dataset_refined.common import (
    EXACT_TARGET, digest, load_manifest, read_json, sha_array, sha_file,
)


class B5Readout(nn.Module):
    """Compose linear 10 m to 5 m prolongation with the solver's recorded B5."""

    def __init__(self, contract, support_indices, support_weights):
        super().__init__()
        self.coarse_shape = tuple(contract["grid"]["shape_zx"])
        self.fine_shape = tuple(contract["solver"]["reference_grid"]["shape_zx"])
        self.fine_spacing = contract["solver"]["reference_grid"]["spacing_xz_m"][0]
        self.pad = round(contract["solver"]["physical_absorbing_width_m"] / self.fine_spacing)
        ratio = self.fine_spacing / contract["grid"]["spacing_xz_m"][0]
        nz, nx = self.coarse_shape
        fine_nz, fine_nx = self.fine_shape
        if (self.coarse_shape != (201, 201) or self.fine_shape != (401, 401)
                or ratio != .5 or support_indices.shape != support_weights.shape
                or support_indices.shape[0] != contract["acquisition"]["receivers"]
                or np.any(support_weights < 0) or not np.all(np.isfinite(support_weights))
                or not np.allclose(support_weights.sum(axis=1), 1., rtol=0, atol=1e-14)):
            raise ValueError("B5 support does not match the certified 10 m/5 m slice")
        row, col = np.where(support_weights != 0)
        z, x = np.divmod(support_indices[row, col], fine_nx + 2 * self.pad)
        x = x - self.pad
        if (np.any(x < 0) or np.any(x >= fine_nx) or np.any(z >= fine_nz)
                or np.any(z * self.fine_spacing < 10.) or np.any(z * self.fine_spacing > 30.)):
            raise ValueError("B5 nonzero taps leave the physical receiver strip")
        fx, fz = x * ratio, z * ratio
        ix, iz = np.minimum(fx.astype(int), nx - 2), np.minimum(fz.astype(int), nz - 2)
        dx, dz = fx - ix, fz - iz
        weights = support_weights[row, col]
        # Keep only the four near-surface rows the measurement can depend on.
        self.support_depth_rows = 4
        matrix = np.zeros((support_indices.shape[0], self.support_depth_rows * nx), dtype=np.float64)
        for oz, wz in ((0, 1 - dz), (1, dz)):
            for ox, wx in ((0, 1 - dx), (1, dx)):
                value = weights * wz * wx
                nonzero = value != 0
                dof = (iz + oz) * nx + ix + ox
                np.add.at(matrix, (row[nonzero], dof[nonzero]), value[nonzero])
        self.register_buffer("matrix", torch.from_numpy(matrix.T.copy()), persistent=False)
        self.register_buffer("reference_indices", torch.from_numpy(support_indices.copy()).long(), persistent=False)
        self.register_buffer("reference_weights", torch.from_numpy(support_weights.copy()), persistent=False)

    def forward(self, pressure):
        if pressure.shape[-2:] != self.coarse_shape:
            raise ValueError("SCNO pressure must have 201 × 201 spatial shape")
        slab = pressure[..., :self.support_depth_rows, :].flatten(-2)
        return slab @ self.matrix.to(slab)

    def explicit(self, pressure):
        """Independent comparison: build the entire prolonged pressure plane."""
        spatial = pressure.reshape(-1, 1, *self.coarse_shape)
        fine = nn.functional.interpolate(spatial, size=self.fine_shape, mode="bilinear", align_corners=True)
        padded = nn.functional.pad(fine, (self.pad, self.pad, 0, self.pad))
        observed = (padded.flatten(1)[:, self.reference_indices] * self.reference_weights.to(padded)).sum(-1)
        return observed.reshape(*pressure.shape[:-2], -1)


class CommittedMarmousi:
    """Use the frozen split and committed shot energies without bulk-loading waveforms."""

    def __init__(self, root, *, prefix_probe=False):
        self.root = Path(root).resolve()
        self.contract, _, rows, receipt = load_manifest(self.root)
        self.rows = {row["model_id"]: row for row in rows}
        self.binding = {k: receipt[k] for k in ("contract_hash", "dataset_recipe_hash", "manifest_hash")}
        self.splits = read_json(self.root / "splits.json")
        model_count = receipt["models"]
        declared_counts = self.contract["dataset"]["counts"]
        if len(self.rows) != model_count or model_count != sum(declared_counts.values()):
            raise ValueError("Manifest model count does not match the acquisition contract")
        stage = "prefix160" if prefix_probe else f"full{model_count}"
        report_name = "prefix160_check.json" if prefix_probe else "dataset_check.json"
        check_path = self.root / "reports" / report_name
        if not check_path.is_file():
            raise ValueError(f"{stage} validation is required: {check_path}")
        report = read_json(check_path)
        required_models = 160 if prefix_probe else model_count
        if (report.get("status") != "PASS" or report.get("stage") != stage
                or not report.get("normalization_frozen") or not report.get("no_parent_split_leakage")
                or not report.get("exact_target_identity") or not report.get("exact_target_in_first16_test")
                or report.get("checked_complete_models") != required_models
                or report.get("checked_complete_shots") != required_models * 38
                or any(report.get(k) != v for k, v in self.binding.items())):
            raise ValueError(f"{stage} validation receipt does not match this dataset")
        self.binding["dataset_validation_file_sha256"] = sha_file(check_path)
        self.shots = self.contract["acquisition"]["shots"]
        self.times = self._shared("observation_times")
        self.sources = self._shared("source_parameters")
        self.observe = B5Readout(self.contract, self._shared("receiver_support_indices"),
                                 self._shared("receiver_support_weights"))
        self.record_shape = (self.shots, len(self.times), self.contract["acquisition"]["receivers"])
        if (self.record_shape != (38, 3201, 201) or not np.array_equal(
                self.times, np.arange(3201, dtype=np.float64) * .00125)
                or self.sources.shape != (38, 5) or not np.all(self.sources[:, 4] == 1.)):
            raise ValueError("Unexpected acquisition time/source/receiver layout")
        split_pairs = [(split, key) for split in ("train", "validation", "test")
                       for key in self.splits[split]]
        split_ids = [key for _, key in split_pairs]
        if (len(split_ids) != model_count or len(set(split_ids)) != model_count
                or any(len(self.splits[split]) != count for split, count in declared_counts.items())
                or set(split_ids) != self.rows.keys()
                or any(self.rows[key]["split"] != split for split, key in split_pairs)
                or self.splits["test_exact_target"] != [EXACT_TARGET]
                or EXACT_TARGET not in self.splits["test"]):
            raise ValueError("Split metadata does not match the frozen model manifest")
        allowed = set(self.splits[stage])
        self.train_ids = [key for key in self.splits["train"] if key in allowed]
        self.validation_ids = [key for key in self.splits["validation"] if key in allowed]
        for keys, wanted in ((self.train_ids, 128 if prefix_probe else declared_counts["train"]),
                             (self.validation_ids, 16 if prefix_probe else declared_counts["validation"])):
            if len(keys) != wanted or any(self.rows[key]["is_exact_target"] for key in keys):
                raise ValueError("Invalid train/validation membership")
        self.train_set, self.validation_set = set(self.train_ids), set(self.validation_ids)
        norm_path = self.root / "normalization.json"
        normal = read_json(norm_path)
        normal_ids = self.splits["normalization_training_ids"]
        if (len(normal_ids) != 128 or len(set(normal_ids)) != 128 or not set(normal_ids) <= self.train_set
                or normal.get("training_ids") != normal_ids or normal.get("exact_target_included") is not False
                or any(normal.get(k) != v for k, v in self.binding.items()
                       if k != "dataset_validation_file_sha256")):
            raise ValueError("Training-only normalization receipt does not match this dataset")
        self.rms = float(normal["receiver_rms"])
        if not np.isfinite(self.rms) or self.rms <= 0:
            raise ValueError("Invalid shared training receiver RMS")
        self.metadata = {
            "velocity_mean_mps": normal["velocity_mean_m_s"],
            "velocity_std_mps": normal["velocity_std_m_s"],
            # The model API calls this key pressure_rms. The labels use LWC's
            # numerical pressure units. All predictions here stay normalized.
            "pressure_rms": self.rms,
            "source_scales": (2000., 2000., 50., 4., 1.),
            "train_manifest_sha256": digest([self.rows[key] for key in self.train_ids]),
            "record_count": len(self.train_ids) * self.shots,
            "algorithm": "frozen_train128_velocity_mean_std_and_numerical_B5_receiver_rms",
        }
        self.binding["normalization_file_sha256"] = sha_file(norm_path)
        contents = {}
        self.energies = {}
        for key in self.train_ids + self.validation_ids:
            meta = read_json(self.root / "models" / key / "meta.json")
            if (meta.get("status") != "COMPLETE" or meta.get("model_id") != key
                    or meta.get("completed_shots") != self.shots
                    or meta.get("split") != self.rows[key]["split"]
                    or meta.get("shape") != list(self.record_shape)
                    or meta.get("velocity_sha256") != self.rows[key]["velocity_sha256"]
                    or any(meta.get(k) != receipt[k] for k in ("contract_hash", "dataset_recipe_hash", "manifest_hash"))):
                raise ValueError(f"Invalid complete model receipt: {key}")
            energies = np.asarray(meta["shot_energy"], dtype=np.float64) / self.rms**2
            if energies.shape != (self.shots,) or not np.isfinite(energies).all() or not (energies > 0).all():
                raise ValueError(f"Invalid shot energy: {key}")
            self.energies[key] = energies
            if key in normal_ids:
                contents[key] = {name: meta[name] for name in ("velocity_sha256", "receivers_file_sha256")}
        if digest(contents) != normal["training_content_hash"]:
            raise ValueError("Normalization inputs no longer match the complete model receipts")
        self.binding["train_model_count"] = len(self.train_ids)
        self.binding["validation_model_count"] = len(self.validation_ids)

    def _shared(self, name):
        dtype = self.contract["shared_arrays"][name]
        value = np.load(self.root / dtype["path"], allow_pickle=False)
        if sha_array(value) != dtype["array_sha256"]:
            raise ValueError(f"Corrupt contract array: {name}")
        return value

    @lru_cache(maxsize=64)
    def _model_files(self, model):
        if model not in self.train_set and model not in self.validation_set:
            raise ValueError("Training view does not open test models")
        root = self.root / "models" / model
        velocity = np.load(root / "velocity.npy", allow_pickle=False)
        records = np.load(root / "receivers.npy", mmap_mode="r", allow_pickle=False)
        if (velocity.shape != (201, 201) or velocity.dtype != np.float32
                or sha_array(velocity) != self.rows[model]["velocity_sha256"]
                or records.shape != self.record_shape or records.dtype != np.float32):
            raise ValueError(f"Invalid committed model arrays: {model}")
        return velocity, records

    def sample(self, split, index, slots):
        if split not in ("train", "validation"):
            raise ValueError("Training/validation loader does not open the held-out test split")
        model_index, shot = divmod(int(index), self.shots)
        keys = self.train_ids if split == "train" else self.validation_ids
        if not 0 <= model_index < len(keys):
            raise IndexError(index)
        model = keys[model_index]
        velocity, records = self._model_files(model)
        labels = records[shot, slots] / np.float32(self.rms)
        if not np.isfinite(labels).all():
            raise ValueError(f"Nonfinite receiver labels in {model} shot {shot}")
        return {
            "model_id": model, "shot_id": shot, "velocity": torch.from_numpy(velocity),
            "source": torch.from_numpy(self.sources[shot].astype(np.float32)),
            "times": torch.from_numpy(self.times[slots].astype(np.float32)),
            "target": torch.from_numpy(labels),
            "full_energy": float(self.energies[model][shot]),
        }
