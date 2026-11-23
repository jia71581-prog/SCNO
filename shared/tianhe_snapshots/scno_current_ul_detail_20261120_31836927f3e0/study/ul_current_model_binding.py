"""Strict restoration of the adopted E200 plus U96 pressure operator."""

import json
from pathlib import Path

import torch

from finetune_ul_transfer import load_pretrained, sha256
from ul_arrival_geometry_general import one_interface_profiles
from ul_general_arrival_readout import GeneralLayeredArrivalReadout, generalized_geometry
from ul_layered_arrival_readout import ArrivalSCNO


READOUT_KEY = ("boundary_only", "model")
E200_SHA = "8ad2a020787314067c7ec84f658203428e1ab2260b7d7322c5f925573f6b7c93"
U96_SHA = "b3a57619803fb4d4899ee591047b7afb91a8469c9ba6a02f2181d5ab6c44aab7"


def verify_current_binding(binding_path):
    binding = json.loads(Path(binding_path).read_text(encoding="utf-8"))
    checkpoints = binding["checkpoints"]
    e200, u96 = checkpoints["pretraining"], checkpoints["side_bottom_u0096"]
    composition = binding["composition"]
    if (e200["sha256"], u96["sha256"], tuple(composition["state_dict_path"]),
            composition["head_width"], composition["wrapper"], composition["geometry"],
            composition["profiles"]) != (E200_SHA, U96_SHA, READOUT_KEY, 96, "ArrivalSCNO",
                                         "generalized_geometry", "one_interface_profiles"):
        raise ValueError("The binding must identify the adopted E200 and U96 model")
    if binding["axis"] != {"frames": 401, "first_s": 0, "last_s": 1, "step_s": .0025, "grid": [201, 201]}:
        raise ValueError("Invalid physical pressure grid or registered time axis")
    return (Path(binding["results_root"]) / e200["relative_path"],
            Path(binding["results_root"]) / u96["relative_path"])


def load_current_model(binding_path, training_config, warm_protocol, completion_receipt, device):
    e200, u96 = verify_current_binding(binding_path)
    if sha256(u96) != U96_SHA:
        raise ValueError("U96 checkpoint SHA-256 mismatch")
    warm_protocol = Path(warm_protocol)
    if warm_protocol.name == "WARM_PROTOCOL.json":
        warm_protocol = warm_protocol.parent
    backbone, receipt = load_pretrained(e200, E200_SHA, training_config, warm_protocol, device,
                                        completion_receipt=completion_receipt)
    saved = torch.load(u96, map_location="cpu", weights_only=True)
    head = GeneralLayeredArrivalReadout(backbone.parent.decoder.in_channels, width=96)
    head.load_state_dict(saved[READOUT_KEY[0]][READOUT_KEY[1]], strict=True)
    model = ArrivalSCNO(backbone, head.to(device), geometry=generalized_geometry,
                        profiles=one_interface_profiles).requires_grad_(False).eval()
    if sum(p.numel() for p in model.parameters()) != 16044650:
        raise ValueError("Unexpected number of parameters after composing E200 and U96")
    return model, {
        "e200_sha256": receipt["checkpoint_sha256"], "u96_sha256": U96_SHA,
        "state_dict_path": list(READOUT_KEY), "composition": "ArrivalSCNO",
        "e200_run_binding_sha256": receipt["run_binding_sha256"],
        "normalizer": repr(model.normalizer.metadata),
    }
