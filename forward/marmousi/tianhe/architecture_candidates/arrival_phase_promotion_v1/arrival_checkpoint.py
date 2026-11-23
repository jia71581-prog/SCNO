"""Restore the paired UL and arrival weights for receiver or full-field inference."""

from copy import deepcopy
from pathlib import Path

import torch

import arrival_phase
import train_tianhe_ul_b5 as legacy
from run_arrival_trial import require


def load_arrival_readout(dataset, checkpoint_path, *, device="cpu"):
    """Return the checkpoint's pressure and B5 readout on its registered grid."""
    device = torch.device(device)
    data = legacy.CommittedMarmousi(dataset)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    binding, trial, extra = (saved[key] for key in ("binding", "trial", "arrival"))
    require(trial["variant"] == "arrival" and extra is not None
            and binding["dataset_binding"] == data.binding
            and binding["trainer_sha256"] == legacy.sha_file(legacy.__file__)
            and binding["model_adapter_sha256"] == legacy.sha_file(legacy.build_marmousi_ul.__code__.co_filename)
            and binding["b5_adapter_sha256"] == legacy.sha_file(Path(legacy.__file__).with_name("tianhe_observations.py"))
            and trial["phase_source_sha256"] == legacy.sha_file(arrival_phase.__file__),
            "Checkpoint, arrival source, model and B5 bindings differ")
    model = legacy.build_marmousi_ul(data, binding["seed"], device, binding["query_block"],
                                     interface_correction=binding["velocity_interface_correction"])
    require(saved["model_semantics"] == model.parent.checkpoint_semantics()
            and saved["repair_settings"] == model.checkpoint_payload()["repair_settings"],
            "Original checkpoint architecture changed")
    model.load_state_dict(saved["model"], strict=True)
    residual = arrival_phase.LocalArrivalPhase(model.parent.decoder.in_channels).to(device)
    require(extra["semantics"] == residual.semantics(), "Arrival checkpoint architecture changed")
    residual.load_state_dict(extra["model"], strict=True)
    result = arrival_phase.ArrivalReceivers(model, residual, deepcopy(data.observe).to(device=device, dtype=torch.float32))
    result.eval()
    return result
