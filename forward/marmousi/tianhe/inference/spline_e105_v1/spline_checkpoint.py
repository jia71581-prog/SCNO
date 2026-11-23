"""Restore an approved receiver-spline checkpoint for physical inputs."""

import hashlib
from pathlib import Path

import torch

import arrival_checkpoint
import cached_receiver_spline


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_spline_readout(dataset, checkpoint, *, device="cpu"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    binding = saved["binding"]
    anchor = Path(binding["anchor_checkpoint"])
    if (binding["pilot"] is not False or binding["anchor_sha256"] != digest(anchor)
            or binding["head_sha256"] != digest(cached_receiver_spline.__file__)):
        raise ValueError("Receiver-spline checkpoint, exact anchor or decoder source differs")
    base = arrival_checkpoint.load_arrival_readout(dataset, anchor, device=device)
    native = base.model.full_times_s
    head = cached_receiver_spline.ReceiverSplineResidual(native, seed=binding["head_seed"]).to(device)
    if head.semantics() != binding["semantics"]:
        raise ValueError("The receiver-spline checkpoint describes a different architecture")
    head.load_state_dict(saved["model"], strict=True)
    view = cached_receiver_spline.SplineCorrectedReceivers(base, head)
    view.eval()
    return view
