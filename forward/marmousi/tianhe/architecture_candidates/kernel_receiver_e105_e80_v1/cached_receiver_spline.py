"""A frozen-context, source-conditioned spline correction on the B5 support."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def native_cubic_splines(times):
    """Exact native-axis rows of clamped cubic splines, dense in the first 2 s."""
    times = torch.as_tensor(times, dtype=torch.float64, device="cpu")
    expected = torch.arange(801, dtype=torch.float64) * .005
    if times.shape != expected.shape or not torch.allclose(times, expected, atol=5e-7, rtol=0.):
        raise ValueError("Use the certified 801 native times at 5 ms")
    breaks = torch.cat((torch.linspace(0., 2., 129, dtype=torch.float64),
                        torch.linspace(2., 4., 33, dtype=torch.float64)[1:]))
    knots = torch.cat((breaks[:1].repeat(4), breaks[1:-1], breaks[-1:].repeat(4)))
    column = expected[:, None]
    basis = ((column >= knots[:-1]) & (column < knots[1:])).to(column)
    for degree in range(1, 4):
        n = len(knots) - degree - 1
        left = knots[degree:degree + n] - knots[:n]
        right = knots[degree + 1:degree + 1 + n] - knots[1:n + 1]
        first = (column - knots[:n]) / left.clamp_min(1e-12) * (left > 0.)
        second = (knots[degree + 1:degree + 1 + n] - column) / right.clamp_min(1e-12) * (right > 0.)
        basis = first * basis[:, :n] + second * basis[:, 1:n + 1]
    basis[-1, -1] = 1.
    basis[0].zero_()
    return basis.float()


class HorizontalFeatureBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.stencil = nn.Conv2d(width, width, (1, 5), dilation=(1, dilation),
                                 padding=(0, 2 * dilation), padding_mode="replicate", groups=width)
        self.mix = nn.Conv2d(width, width, 1)

    def forward(self, features):
        normalized = self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return features + self.mix(F.silu(self.stencil(normalized)))


class ReceiverSplineResidual(nn.Module):
    """Predict depth-local time coefficients with a horizontal receptive field."""

    def __init__(self, native_times, feature_channels=136, width=96, seed=261122):
        super().__init__()
        self.register_buffer("time_rows", native_cubic_splines(native_times))
        self.feature_channels = feature_channels
        self.width = width
        self.dilations = (1, 4, 16)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.input = nn.Conv2d(feature_channels + 8, width, 1)
            self.blocks = nn.Sequential(*(HorizontalFeatureBlock(width, value) for value in self.dilations))
            self.coefficient = nn.Conv2d(width, self.time_rows.shape[1], 1)
            nn.init.zeros_(self.coefficient.weight)
            nn.init.zeros_(self.coefficient.bias)

    def semantics(self):
        return {"version": "marmousi_cached_receiver_spline_v1", "native_frames": 801,
                "feature_channels": self.feature_channels, "latent_width": self.width,
                "basis_columns": self.time_rows.shape[1], "horizontal_dilation": list(self.dilations),
                "source_inputs": "physical source parameters, direct/reflected travel delays, existing UL features",
                "knots": "clamped cubic, 15.625 ms spacing through 2 s and 62.5 ms thereafter",
                "normalization": "inherited numerical B5 RMS", "exactly_zero_at_attachment": True,
                "vertical_receptive_field": "each depth row independent, including normalization"}

    def spatial_coefficients(self, features, travel_times, source):
        if (features.ndim != 4 or features.shape[1] != self.feature_channels
                or travel_times.shape != (features.shape[0], 2, *features.shape[-2:])
                or source.shape != (features.shape[0], 5)):
            raise ValueError("Expected frozen UL features, two travel delays and five physical source entries")
        frequency = source[:, 2, None, None, None]
        angle = (2. * math.pi) * frequency * travel_times
        fields = (source[:, 2:4] / source.new_tensor((8., 4.)))[:, :, None, None]
        scalars = fields.expand(-1, -1, *features.shape[-2:])
        values = torch.cat((features, travel_times * .25, angle.sin(), angle.cos(), scalars), dim=1)
        learned = self.blocks(F.silu(self.input(values)))
        return self.coefficient(learned) * (source[:, 4, None, None, None] != 0)

    def _contract(self, coefficients, slots):
        rows = self.time_rows
        if slots is not None:
            slots = torch.as_tensor(slots, device=rows.device)
            if slots.dtype not in (torch.int32, torch.int64) or slots.ndim not in (1, 2):
                raise ValueError("Use integer registered frame indices")
            if slots.ndim == 2 and slots.shape[0] != coefficients.shape[0]:
                raise ValueError("Each per-shot index row must correspond to one input source")
            if slots.numel() and (slots.min() < 0 or slots.max() >= len(rows)):
                raise ValueError("Query indices are outside the registered time axis")
            rows = rows[slots]
        return rows.to(coefficients) @ coefficients

    def read_receivers(self, features, travel_times, source, readout, frame_indices=None):
        """Apply the fixed B5 linear projector before the time contraction."""
        if features.shape[-2:] != (readout.support_depth_rows - 1, readout.coarse_shape[1]):
            raise ValueError("Supply exactly the receiver strip's active spatial rows")
        coeff = self.spatial_coefficients(features, travel_times, source)
        top = readout.coarse_shape[1]
        support = readout.matrix[top:].to(coeff)
        projected = coeff.flatten(-2) @ support
        return self._contract(projected, frame_indices)

    def read_pressure(self, features, travel_times, source, frame_indices=None):
        """Return full pressure correction including the exactly zero free surface."""
        size = features.shape[-2:]
        coeff = self.spatial_coefficients(features, travel_times, source).flatten(-2)
        pressure = self._contract(coeff, frame_indices)
        return F.pad(pressure.unflatten(-1, size), (0, 0, 1, 0))


def weighted_training_objective(prediction, target, full_energy, frame_weights, energy_floor):
    """Use all 801 samples to evaluate the exact mean of the original stratified loss."""
    denominator = torch.as_tensor(full_energy, device=prediction.device, dtype=torch.float64).clamp_min(energy_floor)
    weights = torch.as_tensor(frame_weights, device=prediction.device, dtype=torch.float64)
    frame_sse = (prediction - target).square().sum(-1).double()
    return ((frame_sse * weights).sum(1) / denominator).mean()


class SplineCorrectedReceivers:
    """Recompute context from physical inputs for pressure queries and inversion."""

    def __init__(self, anchor, residual):
        self.anchor = anchor
        self.residual = residual
        self.device = anchor.device

    def encode(self, velocity, source):
        return self.anchor.encode(velocity, source)

    def eval(self):
        self.anchor.eval()
        self.residual.eval()

    def _scale(self, value, context, normalized):
        if normalized:
            return value
        scale = self.anchor.model.normalizer.metadata.pressure_scale_pa * context.source[:, 4]
        return value * scale.reshape(-1, *([1] * (value.ndim - 1)))

    def decode(self, context, frame_indices=None, *, normalized=True):
        depth = self.anchor.readout.support_depth_rows
        value = self.residual.read_receivers(context.features[:, :, 1:depth],
            context.propagation.travel_times[:, :, 1:depth], context.source, self.anchor.readout, frame_indices)
        return self.anchor.decode(context, frame_indices, normalized=normalized) + self._scale(value, context, normalized)

    def decode_pressure(self, context, frame_indices=None, *, normalized=True):
        value = self.residual.read_pressure(context.features[:, :, 1:],
            context.propagation.travel_times[:, :, 1:], context.source, frame_indices)
        return self.anchor.decode_pressure(context, frame_indices, normalized=normalized) + self._scale(value, context, normalized)

    def __call__(self, velocity, source, frame_indices=None, *, normalized=True):
        return self.decode(self.encode(velocity, source), frame_indices, normalized=normalized)
