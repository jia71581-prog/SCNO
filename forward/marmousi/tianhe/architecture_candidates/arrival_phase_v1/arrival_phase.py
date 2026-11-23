"""Add local arrival phase to the archived UL without replacing its parameters."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LocalArrivalPhase(nn.Module):
    """Independently modulate direct and free-surface pulse quadratures."""

    def __init__(self, feature_channels, hidden_channels=32, seed=261120):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.hidden = nn.Conv2d(feature_channels + 2, hidden_channels, 1)
            self.amplitude = nn.Conv2d(hidden_channels, 2, 1)
            self.phase = nn.Conv2d(hidden_channels, 2, 1)
            for head in (self.amplitude, self.phase):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def semantics(self):
        return {
            "version": "marmousi_local_arrival_quadrature_v1",
            "carrier_delays": ["direct", "free_surface_reflected"],
            "spatial_condition": "existing UL FNO and interface features plus physical travel times",
            "spatial_hidden_channels": self.hidden.out_channels,
            "time_condition": "pi*f0*(t-t0-travel_time), evaluated at frozen physical times",
            "quadratures": ["Ricker", "half Ricker derivative with respect to phase"],
            "gate": "zero at t=0, free surface and inactive source",
            "initial_output": "exactly zero",
        }

    def forward(self, context, times, *, depth_rows):
        features, source = context[:2]
        travel = context.propagation.travel_times[:, :, 1:depth_rows]
        features = features[:, :, 1:depth_rows]
        hidden = F.silu(self.hidden(torch.cat((features, travel * .25), dim=1)))
        amplitude, quadrature = self.amplitude(hidden), self.phase(hidden)
        delay = times[:, :, None, None] - source[:, 3, None, None, None]
        frequency = math.pi * source[:, 2, None, None, None]
        correction = None
        for path in range(2):
            phase = (delay - travel[:, None, path]) * frequency
            phase2 = phase.square()
            envelope = (-phase2).exp()
            pulse = (1. - 2. * phase2) * envelope
            derivative = (2. * phase2 - 3.) * phase * envelope
            local = amplitude[:, None, path] * pulse + quadrature[:, None, path] * derivative
            correction = local if correction is None else correction + local
        active = (times > 0) & (source[:, 4:5] != 0)
        return F.pad(correction * active[:, :, None, None], (0, 0, 1, 0))


class ArrivalReceivers:
    """Use the same phase correction for full pressure and B5 receiver queries."""

    def __init__(self, model, residual, readout):
        self.model = model
        self.residual = residual
        self.readout = readout
        self.device = model.device

    def encode(self, velocity, source):
        return self.model.encode(velocity, source)

    def eval(self):
        self.model.eval()
        self.residual.eval()

    def _times(self, context, slots):
        if slots is not None:
            slots = torch.as_tensor(slots, device=self.device)
        return self.model.parent._selected_times(slots, len(context.source))

    def _correction(self, context, slots, depth_rows, normalized):
        correction = self.residual(context, self._times(context, slots), depth_rows=depth_rows)
        if not normalized:
            scale = self.model.normalizer.metadata.pressure_scale_pa * context.source[:, 4]
            correction = correction * scale[:, None, None, None]
        return correction

    def decode_pressure(self, context, frame_indices=None, *, normalized=True):
        pressure = self.model.decode(context, frame_indices=frame_indices, normalized=normalized)
        correction = self._correction(context, frame_indices, pressure.shape[-2], normalized)
        return pressure + correction

    def decode(self, context, frame_indices=None, *, normalized=True):
        pressure = self.model.decode(context, frame_indices=frame_indices, normalized=normalized)
        depth_rows = self.readout.support_depth_rows
        correction = self._correction(context, frame_indices, depth_rows, normalized)
        slab = (pressure[..., :depth_rows, :] + correction).flatten(-2)
        return slab @ self.readout.matrix.to(slab)

    def __call__(self, velocity, source, frame_indices=None, *, normalized=True):
        return self.decode(self.encode(velocity, source), frame_indices, normalized=normalized)
