"""Read-only model/input interface check for a new, early-label-only probe."""
from scno_release_paths import local_asset as _scno_local_asset
from pathlib import Path
import hashlib
import json
import sys
import types

import numpy as np
import torch

PROJECT = Path(_scno_local_asset('project'))
ASSET = PROJECT / "paper_assets_main_fresh_20261013_marmousi_20261019"
HERE = Path(__file__).resolve().parent
SUPPORT = Path(_scno_local_asset('runtime/final_algorithm_revision/evidence_code/support'))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(SUPPORT))

from model_adapter import _local_protocol_path
from native400_runtime import r24_cf_preserved_operator as preserved
from native400_runtime.r25_cf_full400_operator import build_r25_cf_full400_operator


def load_marmousi(device="cpu"):
    initializer = ASSET / "models/initializers/marmousi.pt"
    checkpoint = ASSET / "models/marmousi_epoch65.pt"
    config = {
        "parent_checkpoint": str(initializer),
        "parent_checkpoint_SHA": hashlib.sha256(initializer.read_bytes()).hexdigest(),
        "dense_time_block": 1,
        "refiner_time_block": 1,
        "checkpoint_queries": False,
    }
    # DGFeatureBlock's module imports ResidualRefiner for an unused convenience
    # subclass. Bind that name to the exact canonical refiner rather than load
    # an old bootstrap's machine-specific training paths. No model operations
    # or parameters are replaced by this import-only alias.
    refiner = preserved._module("marm_probe_canonical_refiner", preserved.COMPAT / "r22/refiner.py")
    bootstrap = types.ModuleType("bootstrap24")
    bootstrap.__file__ = str(preserved.COMPAT / "r24/bootstrap24.py")
    bootstrap.ResidualRefiner = refiner.ResidualRefiner
    sys.modules["bootstrap24"] = bootstrap
    with _local_protocol_path():
        model = build_r25_cf_full400_operator(config, device="cpu", kind="marmousi")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    assert saved["schema"] == "MARM_INDEPENDENT_DG_PHYSICS100_V1"
    model.load_state_dict(saved["model_state"], strict=True)
    del saved
    model.eval().requires_grad_(False).to(device)
    velocity_path = PROJECT / "runs/paper_marmousi201_single_source_fullfit_20261020/velocity401.npy"
    reference_path = PROJECT / "runs/paper_marmousi201_single_source_fullfit_20261020/case_0/reference_pa.npy"
    velocity = np.array(np.load(velocity_path, mmap_mode="r")[::2, ::2], copy=True)
    # Only the eight permitted fields are touched; later values stay unopened.
    reference_mmap = np.load(reference_path, mmap_mode="r")
    assert reference_mmap.shape == (400, 201, 201)
    early8 = np.array(reference_mmap[:8], copy=True)
    del reference_mmap
    source = torch.tensor([868.0423, 688.824, 15.0, 0.1, 1.0], dtype=torch.float32, device=device)
    inputs = {
        "velocity": torch.from_numpy(velocity).to(device),
        "source": source,
        "early8_pa": torch.from_numpy(early8).to(device),
        "early_times": model.full_times_s[:8].detach().clone(),
        "full_times": model.full_times_s.detach().clone(),
        "reference_path": str(reference_path),
        "pressure_scale_pa": float(model.normalizer.metadata.pressure_scale_pa),
        "checkpoint": str(checkpoint),
    }
    return model, inputs


if __name__ == "__main__":
    torch.set_num_threads(2)
    model, inputs = load_marmousi("cpu")
    receipt = {
        "status": "STRICT_CPU_INTERFACE_LOAD_PASS",
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "readout_module": "refiner.output",
        "readout_weight_shape": list(model.refiner.output.weight.shape),
        "readout_bias_shape": list(model.refiner.output.bias.shape),
        "affine_increment_dimensions": model.refiner.output.weight.numel() + model.refiner.output.bias.numel(),
        "dg_enabled": bool(model.refiner.dg.enabled),
        "dg_faces_enabled": bool(model.refiner.dg.face_enabled),
        "source": inputs["source"].tolist(),
        "velocity_shape": list(inputs["velocity"].shape),
        "early_pressure_shape": list(inputs["early8_pa"].shape),
        "early_times": inputs["early_times"].tolist(),
        "pressure_scale_pa": inputs["pressure_scale_pa"],
        "future_reference_values_opened": False,
        "forward_calls": 0,
        "optimizer_steps": 0,
        "checkpoint": inputs["checkpoint"],
        "reference_path": inputs["reference_path"],
    }
    (HERE / "INTERFACE_CHECK.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
