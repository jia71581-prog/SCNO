"""Helpers used by the public training entry points."""
from pathlib import Path
import torch
from torch import nn
from tools.marmousi_target_dataset_refined.common import sha_file

SCHEMA = "marmousi_e12_native_pressure_pilot_v1"

E12 = "caf8fe9f1d92ce97592df084117243efc292ee4f97f7ecf65d7b6f9edfa6d779"

NATIVE_SHA = "56b8fe5aaf1a1ccefc5f125f013cdfc2777327a056cdef6574b4a83c1452dca8"

KERNEL_SHA = "3862f4f2c486c9b6b023c4a0842b3562810eca4a8bb19b3a1c1a2ecaca51b7da"

SPLINE_LOADER_SHA = "0c154e9519817476b39e383f5e7a3594255ea0ca2b185c2d1db167b23c96b308"

MODES = ("lora_only", "detail_output")

def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)

class NativePressurePilot(nn.Module):
    """Train archived detail roles and project their actual pressure via frozen B5."""

    def __init__(self, core, data, detail_state, mode):
        super().__init__()
        from shallow_b5 import ShallowArrivalReceivers
        from ul_current_lora import CurrentCompensatedLoRA

        require(mode in MODES, "Choose one recorded native detail adaptation mode")
        core.anchor = ShallowArrivalReceivers.from_reference(core.anchor)
        self.core = core
        self.device = core.device
        self.receiver_rms = data.rms
        self.mode = mode
        self.sources = torch.as_tensor(data.sources, dtype=torch.float32, device=self.device)
        self.times = torch.as_tensor(data.times, dtype=torch.float64, device=self.device)
        model = core.anchor.model
        for module in (model, core.anchor.residual, core.residual):
            module.requires_grad_(False)
        model.compiled_readout = None  # Use the dynamically differentiable coefficient-first decoder.
        heads = model.parent.decoder.detail_heads
        require(len(heads) == 3 and all(tuple(h[2].weight.shape) == (32, 32) and isinstance(h[-1], nn.Linear) for h in heads),
                "The exact three E12 detail affine and spatial output roles are required")
        for head in heads:
            head[2] = CurrentCompensatedLoRA(head[2])
        self.detail = nn.ModuleList(head[2] for head in heads)
        self.detail.load_state_dict(detail_state, strict=True)
        require(sum(p.numel() for p in self.detail.parameters() if p.requires_grad) == 768,
                "E12 requires the frozen affine plus the original rank-four factors")
        self.outputs = nn.ModuleList(head[-1] for head in heads)
        self.outputs.requires_grad_(mode == "detail_output")
        self.core.eval()

    def encode(self, velocity, source):
        source = torch.as_tensor(source, dtype=torch.float32, device=self.device)
        if source.ndim == 1:
            source = source[None]
        require(source.ndim == 2 and source.shape[1] == 5
                and bool(((source[:, None] == self.sources).all(-1).sum(-1) == 1).all()),
                "Use original frozen physical source coordinates and Ricker parameters")
        return self.core.encode(velocity, source)

    def decode_pressure(self, context, frame_indices=None, *, normalized=True):
        return self.core.decode_pressure(context, frame_indices, normalized=normalized)

    def forward(self, velocity, source, frame_indices=None, *, normalized=True):
        return self.decode_pressure(self.encode(velocity, source), frame_indices, normalized=normalized)

    def observe(self, pressure):
        """Use frozen 10-to-5 m prolongation and original B5 geometric weights."""
        return self.core.anchor.readout(pressure)

    def decode_b5(self, context, frame_indices=None, *, normalized=True):
        """Evaluate the exact shallow dependency of the same neural pressure."""
        return self.core.decode(context, frame_indices, normalized=normalized)

def checkpoint_identity(dataset, base_checkpoint, e12_checkpoint, selection, native_checkpoint):
    from sensitivity_readout import read_selection
    from tianhe_observations import CommittedMarmousi
    from train_shared_framework import FRAMEWORK, source_binding
    import spline_checkpoint

    data = CommittedMarmousi(dataset)
    selected = read_selection(selection)
    require(sha_file(e12_checkpoint) == selected["candidate_checkpoint_sha256"] == E12
            and sha_file(base_checkpoint) == selected["base_checkpoint_sha256"] == KERNEL_SHA
            and selected["epoch"] == 12 and selected["framework"] == FRAMEWORK
            and selected["dataset_contract_sha256"] == data.binding["contract_hash"]
            and selected["dataset_manifest_sha256"] == data.binding["manifest_hash"]
            and selected["training_or_selection_used_fwi_audits"] is False
            and sha_file(native_checkpoint) == NATIVE_SHA
            and sha_file(spline_checkpoint.__file__) == SPLINE_LOADER_SHA, "Frozen E12/UL release or weights differ")
    saved = torch.load(e12_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    binding = saved["binding"]
    require(saved["selected"] is True and saved["epoch"] == 12 and binding["framework"] == FRAMEWORK
            and binding["dataset_binding"] == data.binding and binding["receiver_rms"] == data.rms
            and binding["kernel_sha256"] == KERNEL_SHA and binding["native_sha256"] == NATIVE_SHA
            and binding["training_used_exact_marmousi"] is False and binding["selection_used_exact_marmousi"] is False
            and binding["training_or_selection_used_fwi_audits"] is False and binding["sources"] == source_binding(),
            "Selected E12 pressure adaptation provenance changed")
    require(all(key == "gain" or key.startswith("detail.") for key in saved["model"]), "Unexpected archived E12 parameters")
    identity = {"schema": SCHEMA, "dataset_binding": data.binding, "receiver_rms": data.rms,
                "e12_checkpoint_sha256": E12, "original_base_checkpoint_sha256": KERNEL_SHA,
                "native_checkpoint_sha256": NATIVE_SHA, "original_selection_sha256": sha_file(selection),
                "pure_core_code_sha256": sha_file(__file__), "spline_loader_sha256": SPLINE_LOADER_SHA}
    detail = {key.removeprefix("detail."): value for key, value in saved["model"].items() if key.startswith("detail.")}
    return data, identity, detail, binding["anchors"]

def load_neural_pressure_core(dataset, base_checkpoint, e12_checkpoint, selection, native_checkpoint,
                              *, device="cuda:0", mode="detail_output"):
    """Restore only the archived native neural E12 pressure branch."""
    from spline_checkpoint import load_spline_readout

    data, identity, detail, anchors = checkpoint_identity(dataset, base_checkpoint, e12_checkpoint, selection, native_checkpoint)
    core = load_spline_readout(data.root, Path(native_checkpoint), device=device)
    require(float(core.anchor.model.normalizer.metadata.pressure_scale_pa) == data.rms,
            "Native pressure units differ from the released numerical RMS")
    model = NativePressurePilot(core, data, detail, mode).eval()
    model.identity = identity
    model.archived_anchors = anchors
    return model
