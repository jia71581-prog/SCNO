"""Opt in to a validation-approved derivative-trained Marmousi B5 readout."""

import hashlib
import json
from pathlib import Path
import re

import torch


SELECTION_SCHEMA = "marmousi_b5_sensitivity_selection_v1"
SHARED_FRAMEWORK = "scno_orthogonal_detail_propagation_v1"
KERNEL_SHA256 = "3862f4f2c486c9b6b023c4a0842b3562810eca4a8bb19b3a1c1a2ecaca51b7da"
_HASH_FIELDS = ("candidate_checkpoint_sha256", "base_checkpoint_sha256", "dataset_contract_sha256",
                "dataset_manifest_sha256", "held_out_val_pairs_sha256")


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_selection(path):
    """Process a validation approval before importing or restoring a candidate."""
    with Path(path).open("r", encoding="utf-8") as stream:
        selected = json.load(stream)
    if (not isinstance(selected, dict) or selected.get("schema") != SELECTION_SCHEMA
            or selected.get("status") != "APPROVED" or selected.get("selection_split") != "validation"
            or selected.get("training_used_exact_marmousi") is not False
            or selected.get("selection_used_exact_marmousi") is not False
            or selected.get("framework") not in (None, SHARED_FRAMEWORK)
            or any(not isinstance(selected.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", selected[key])
                   for key in _HASH_FIELDS)
            or selected.get("base_checkpoint_sha256") != KERNEL_SHA256):
        raise ValueError("A validation approval with frozen pair and checkpoint hashes is required")
    return selected


def _restore_candidate(dataset, base, checkpoint, *, expected_sha256, device, framework):
    if framework == SHARED_FRAMEWORK:
        from train_shared_framework import load_sensitivity_readout
    else:
        from train_sensitivity import load_sensitivity_readout

    return load_sensitivity_readout(dataset, base, checkpoint, expected_sha256=expected_sha256, device=device)


def load_selected_readout(dataset, base_checkpoint, selected_checkpoint, selection_json, *, device="cuda:0"):
    """Restore an approved differentiable readout on the physical 38 × 801 × 201 geometry."""
    from tianhe_observations import CommittedMarmousi

    selected = read_selection(selection_json)
    base_path = Path(base_checkpoint).resolve(strict=True)
    path = Path(selected_checkpoint).resolve(strict=True)
    expected = selected["candidate_checkpoint_sha256"]
    if base_path == path or _sha256(base_path) != selected["base_checkpoint_sha256"] or _sha256(path) != expected:
        raise ValueError("Candidate or baseline bytes differ from their validation approval")
    release = CommittedMarmousi(dataset)
    if (selected["dataset_contract_sha256"] != release.binding["contract_hash"]
            or selected["dataset_manifest_sha256"] != release.binding["manifest_hash"]):
        raise ValueError("Validation approval belongs to a different Marmousi release")
    archive = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    binding = archive.get("binding") if isinstance(archive, dict) else None
    if not isinstance(binding, dict) or binding.get("training_used_exact_marmousi") is not False:
        raise ValueError("Adapter checkpoint must record that its fit excludes exact Marmousi")
    framework = selected.get("framework")
    if framework == SHARED_FRAMEWORK and binding.get("framework") != framework:
        raise ValueError("Approved framework differs from the recorded adapter implementation")
    del archive
    readout = _restore_candidate(release.root, base_path, path, expected_sha256=expected, device=device,
                                 framework=framework)
    compulsory = ("eval", "encode", "decode", "__call__")
    if not all(callable(getattr(readout, name, None)) for name in compulsory):
        raise TypeError("The approved candidate does not implement the B5 FWI interface")
    requested, actual = torch.device(device), torch.device(readout.device)
    sources = torch.as_tensor(release.sources, dtype=torch.float32)
    times = torch.as_tensor(release.times, dtype=torch.float64)
    if (sources.shape != (38, 5) or times.shape != (801,) or tuple(release.trace_shape) != (801, 201)
            or readout.receiver_rms != release.rms or not torch.equal(readout.sources.detach().cpu(), sources)
            or not torch.equal(readout.times.detach().cpu(), times) or actual.type != requested.type
            or (requested.index is not None and actual.index != requested.index)):
        raise ValueError("Approved candidate altered the source, time, receiver or device binding")
    # Eval mode retains the velocity derivative used by the all-shot objective.
    readout.eval()
    return readout
