"""Compensated orthogonal detail and acoustic propagation, with a frozen Marmousi arrival readout."""

from contextlib import contextmanager
from functools import lru_cache
from typing import NamedTuple

import torch
from torch import nn

from coarse_propagation import CoarsePropagation
from kernel_corrected_readout import load_kernel_corrected_readout
from shallow_b5 import ShallowArrivalReceivers
from ul_current_lora import CurrentCompensatedLoRA


FRAMEWORK = "scno_orthogonal_detail_propagation_v1"


class DetailAffine(CurrentCompensatedLoRA):
    """The shared compensated update, with an explicit frozen-anchor query."""

    enabled = True

    def forward(self, values):
        return super().forward(values) if self.enabled else self.base(values)


class Encoded(NamedTuple):
    native: object
    propagation_difference: torch.Tensor


class SharedFrameworkReadout(nn.Module):
    """Native pressure plus detail LoRA and a bounded scalar propagation direction."""

    def __init__(self, dataset, checkpoint, device, build_dir):
        super().__init__()
        self.base = load_kernel_corrected_readout(dataset, checkpoint, device=device)
        self.device = self.base.device
        self.sources, self.times, self.receiver_rms = self.base.sources, self.base.times, self.base.receiver_rms
        self.core = self.base.native_core
        self.core.anchor = ShallowArrivalReceivers.from_reference(self.core.anchor)
        self.core.anchor.model.compiled_readout = None
        heads = self.core.anchor.model.parent.decoder.detail_heads
        if [tuple(h[2].weight.shape) for h in heads] != [(32, 32)] * 3:
            raise ValueError("The shared training scheme requires the three original 32×32 detail affines")
        for head in heads:
            head[2] = DetailAffine(head[2])
        self.detail = nn.ModuleList(h[2] for h in heads)
        self.gain = nn.Parameter(torch.zeros((), device=device))
        if sum(p.numel() for p in self.parameters() if p.requires_grad) != 769:
            raise ValueError("The detail-only scheme must expose 768 LoRA scalars and one gain")
        self.propagator = CoarsePropagation(dataset, device=device, build_dir=build_dir)
        self.base.eval()

    @contextmanager
    def use_frozen_detail(self):
        saved = [layer.enabled for layer in self.detail]
        for layer in self.detail:
            layer.enabled = False
        try:
            yield
        finally:
            for layer, enabled in zip(self.detail, saved):
                layer.enabled = enabled

    def registered_sources(self, shots):
        sources = torch.as_tensor(shots, dtype=torch.float32, device=self.device).reshape(-1, 5)
        matched = (sources[:, None] == self.sources[None]).all(-1)
        if not bool((matched.sum(-1) == 1).all()):
            raise ValueError("Use only frozen physical sources")
        indices = matched.nonzero(as_tuple=False)[:, 1].cpu().tolist()
        return sources, indices

    def make_proposal(self, velocity, sources, snapshot_slots=None):
        _, ids = self.registered_sources(sources)
        if snapshot_slots is None:
            return self.propagator.predict(velocity, ids, grad=velocity.requires_grad) / self.receiver_rms
        pressure, b5 = self.propagator.predict_snapshots(velocity, ids, snapshot_slots, include_b5=True)
        return b5 / self.receiver_rms, pressure / self.receiver_rms

    def encode(self, velocity, source, proposal=None):
        velocity = torch.as_tensor(velocity, dtype=torch.float32, device=self.device).reshape(201, 201)
        source, _ = self.registered_sources(source)
        base = self.base.encode(velocity, source)
        if proposal is None:
            proposal = self.make_proposal(velocity, source)
        with self.use_frozen_detail():
            frozen = self.base.decode(base, normalized=True)
        return Encoded(base, proposal - frozen)

    def decode(self, context, frame_indices=None, *, normalized=True):
        baseline = self.base.decode(context.native, frame_indices, normalized=True)
        direction = context.propagation_difference
        if frame_indices is not None:
            slots = torch.as_tensor(frame_indices, device=self.device)
            direction = direction[:, slots] if slots.ndim == 1 else direction.gather(1, slots[..., None].expand(-1, -1, 201))
        predicted = baseline + self.gain * direction
        return predicted if normalized else predicted * self.receiver_rms

    def forward(self, velocity, source, frame_indices=None, *, normalized=True):
        return self.decode(self.encode(velocity, source), frame_indices, normalized=normalized)

    def full_pressure(self, context, frame_indices, proposal):
        native = context.native.native_context
        candidate = self.core.decode_pressure(native, frame_indices, normalized=True)
        with self.use_frozen_detail():
            frozen = self.core.decode_pressure(native, frame_indices, normalized=True)
        return candidate + self.gain * (proposal - frozen)

    def trust(self):
        return torch.stack([layer.delta_weight().square().sum() / layer.base.weight.square().sum().clamp_min(1e-12)
                            for layer in self.detail]).mean()


@lru_cache(maxsize=1)
def native_residual_operators():
    from finetune_ul_transfer import residual_operators

    return residual_operators()


def defect_loss(readout, normalized_pressure, proposal, velocity, sources, sample_times, observation):
    op = native_residual_operators()
    difference = (normalized_pressure - proposal).double()
    minus, center, plus = difference[:, :sample_times], difference[:, sample_times:2*sample_times], difference[:, 2*sample_times:]
    dt = float(readout.times[1] - readout.times[0])
    terms = []
    for i, source in enumerate(sources):
        valid = op.make_valid_mask(source=source[:2].tolist(), halo=8, radius=90., device=velocity.device)
        residual = op.lwc_residual(minus[i], center[i], plus[i], velocity, h=10., delta=dt)
        scale = op.residual_scale_squared(observation[i, :201], source[2].item())
        terms.append(residual[:, valid].square().mean() / scale)
    return torch.stack(terms).mean()
