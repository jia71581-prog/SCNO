"""Nonlinear time-conditioned detail prediction with fixed orthogonal pressure synthesis."""

from copy import deepcopy
from functools import partial
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class PreparedDetailFeatures(NamedTuple):
    spatial: tuple


class DetailHiddenChunk(NamedTuple):
    start: int
    stop: int
    second_hidden: tuple
    valid_masks: tuple


class DetailCoefficientChunk(NamedTuple):
    start: int
    stop: int
    coefficients: tuple


class TimeConditionedDetailHead(nn.Module):
    """Predict each detail coefficient from a spatial block and physical-time features.

    Shared spatial work is independent of the number of queried times. Each
    time chunk uses batched affine maps on the block grids. Output affine
    layers become available after the caller fits residual coefficients by QR
    and commits an accepted, non-increasing calibration receipt.
    """

    def __init__(self, in_channels, time_features, transform, *, hidden_width=32,
                 projection_channels=4, spatial_refiner=False, time_chunk=32,
                 checkpoint_hidden=False):
        super().__init__()
        if min(in_channels, time_features, hidden_width, projection_channels, time_chunk) < 1:
            raise ValueError("channel widths and time chunk size must be positive")
        if not len(transform.levels) or any(level.modes <= 16 for level in transform.levels):
            raise ValueError("each requested level must contain active detail slots")
        self.in_channels = int(in_channels)
        self.time_features = int(time_features)
        self.hidden_width = int(hidden_width)
        self.time_chunk = int(time_chunk)
        self.checkpoint_hidden = bool(checkpoint_hidden)
        self.transform = deepcopy(transform)
        self.spatial_refiner = None
        if spatial_refiner:
            self.spatial_refiner = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                nn.SiLU(), nn.Conv2d(in_channels, in_channels, 1))
        self.feature_projection = nn.Conv2d(in_channels, projection_channels, 1)
        self.spatial_first = nn.ModuleList()
        self.temporal_first = nn.ModuleList()
        self.second_hidden = nn.ModuleList()
        self.readouts = nn.ModuleList()
        for level in self.transform.levels:
            slots = level.valid_detail.shape[-1]
            self.spatial_first.append(nn.Linear(projection_channels * level.modes + slots, hidden_width))
            self.temporal_first.append(nn.Linear(time_features, hidden_width))
            self.second_hidden.append(nn.Linear(hidden_width, hidden_width))
            self.readouts.append(nn.Linear(hidden_width, slots))
        # Every newly introduced affine, including the output readouts, receives
        # nonzero Xavier initialization. QR calibration supplies deployed readouts.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight, gain=1.)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.register_buffer("readout_calibrated", torch.tensor(False))
        self._readout_ready = False
        self.calibration_receipt = None
        self.register_load_state_dict_post_hook(self._after_state_load)

    @property
    def valid_masks(self):
        return tuple(level.valid_detail for level in self.transform.levels)

    @property
    def output_layers(self):
        return tuple(self.readouts)

    @property
    def calibration_required(self):
        return not self._readout_ready

    def _after_state_load(self, module, incompatible_keys):
        self._readout_ready = bool(self.readout_calibrated.detach().cpu())

    @torch.no_grad()
    def mark_calibrated(self, receipt, *, optimizer=None):
        """Commit the caller's residual-QR result and clear old output moments."""
        before, after = receipt.get("before"), receipt.get("after")
        if (receipt.get("accepted") is not True or before is None or after is None
                or not all(math.isfinite(float(v)) and float(v) >= 0 for v in (before, after))
                or float(after) > float(before) + 1e-8 * max(1., float(before))):
            raise ValueError("commit an accepted residual-QR receipt with non-increasing calibration error")
        if optimizer is not None:
            for layer in self.readouts:
                for parameter in layer.parameters():
                    optimizer.state.pop(parameter, None)
        self.readout_calibrated.fill_(True)
        self._readout_ready = True
        self.calibration_receipt = dict(receipt)

    def _require_calibrated(self):
        if not self._readout_ready:
            raise RuntimeError("fit the new output affines to training residual details by QR before deployment")

    def prepare_spatial(self, features):
        """Prepare one record batch; retain this context only for its current forward."""
        if features.ndim != 4 or features.shape[1] != self.in_channels or not features.shape[0]:
            raise ValueError("expected source-conditioned spatial features [B,C,Z,X]")
        height, width = self.transform.shape
        if features.shape[-2:] == (height + 1, width):
            features = features[:, :, 1:]
        elif features.shape[-2:] != (height, width):
            raise ValueError("spatial features must cover the active grid or its zero-extended pressure grid")
        if self.spatial_refiner is not None:
            features = features + self.spatial_refiner(features)
        _, packets = self.transform.analyze(self.feature_projection(features))
        prepared = []
        for layer, packet, mask in zip(self.spatial_first, packets, self.valid_masks):
            block = packet.permute(0, 2, 3, 1, 4).flatten(-2)
            incoming = torch.cat((block, mask.to(block).expand(features.shape[0], -1, -1, -1)), dim=-1)
            prepared.append(layer(incoming))
        return PreparedDetailFeatures(tuple(prepared))

    def _select_times(self, prepared, temporal, frame_indices):
        batch = prepared.spatial[0].shape[0]
        if (temporal.ndim != 3 or temporal.shape[0] != batch
                or temporal.shape[-1] != self.time_features or temporal.shape[1] == 0):
            raise ValueError("supply source-conditioned physical-time features [B,T,Q]")
        if temporal.device != prepared.spatial[0].device:
            raise ValueError("place physical-time and spatial features on the same device")
        if frame_indices is None:
            return temporal
        indices = torch.as_tensor(frame_indices, device=temporal.device)
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("time-row indices must be integers")
        indices = indices.long()
        if indices.ndim == 1 and indices.numel():
            return temporal.index_select(1, indices)
        if indices.ndim == 2 and indices.shape[0] == batch and indices.shape[1]:
            return temporal.gather(1, indices[..., None].expand(-1, -1, temporal.shape[-1]))
        raise ValueError("use shared [K] or per-record [B,K] nonempty time-row indices")

    def _hidden_scale(self, scale, spatial, temporal):
        time = self.temporal_first[scale](temporal)
        first = F.silu(spatial[:, None] + time[:, :, None, None])
        return F.silu(self.second_hidden[scale](first))

    def iter_hidden(self, prepared, temporal, frame_indices=None, *, time_chunk=None):
        """Yield second-SiLU features and valid slots, including before calibration."""
        if not isinstance(prepared, PreparedDetailFeatures):
            prepared = self.prepare_spatial(prepared)
        selected = self._select_times(prepared, temporal, frame_indices)
        chunk = self.time_chunk if time_chunk is None else int(time_chunk)
        if chunk < 1:
            raise ValueError("the query chunk size must be positive")
        for start in range(0, selected.shape[1], chunk):
            stop = min(start + chunk, selected.shape[1])
            times = selected[:, start:stop]
            hidden = []
            for scale, spatial in enumerate(prepared.spatial):
                function = partial(self._hidden_scale, scale)
                if self.checkpoint_hidden and self.training and torch.is_grad_enabled():
                    value = checkpoint(function, spatial, times, use_reentrant=False)
                else:
                    value = function(spatial, times)
                hidden.append(value)
            yield DetailHiddenChunk(start, stop, tuple(hidden), self.valid_masks)

    def calibration_features(self, prepared, temporal, frame_indices=None, *, time_chunk=None):
        """Return complete second-hidden tensors for an explicitly bounded calibration batch."""
        chunks = list(self.iter_hidden(prepared, temporal, frame_indices, time_chunk=time_chunk))
        return tuple(torch.cat([chunk.second_hidden[level] for chunk in chunks], dim=1)
                     for level in range(len(self.readouts)))

    def iter_coefficients(self, prepared, temporal, frame_indices=None, *, time_chunk=None):
        """Yield [B,K,GZ,GX,S] coefficient packets, with all padded slots zero."""
        self._require_calibrated()
        for chunk in self.iter_hidden(prepared, temporal, frame_indices, time_chunk=time_chunk):
            coefficients = tuple(layer(hidden).masked_fill(~mask, 0)
                                 for layer, hidden, mask in zip(
                                     self.readouts, chunk.second_hidden, chunk.valid_masks))
            yield DetailCoefficientChunk(chunk.start, chunk.stop, coefficients)

    def forward(self, features, temporal, frame_indices=None, *, time_chunk=None):
        """Return one complete [B,T,GZ,GX,S] coefficient tensor per scale."""
        chunks = list(self.iter_coefficients(features, temporal, frame_indices, time_chunk=time_chunk))
        return tuple(torch.cat([chunk.coefficients[level] for chunk in chunks], dim=1)
                     for level in range(len(self.readouts)))

    def synthesize(self, coefficients):
        """Apply the inherited fixed transform and append the exact zero pressure surface."""
        if len(coefficients) != len(self.transform.levels):
            raise ValueError("one coefficient packet is required for every detail scale")
        batch, count = coefficients[0].shape[:2]
        packets = []
        for value, level in zip(coefficients, self.transform.levels):
            if value.shape != (batch, count, level.gz, level.gx, level.valid_detail.shape[-1]):
                raise ValueError("a coefficient packet changed its registered level geometry")
            packets.append(value.reshape(batch * count, 1, level.gz, level.gx, -1))
        last = self.transform.levels[-1]
        coarse = coefficients[0].new_zeros((batch * count, 1, last.gz * 4, last.gx * 4))
        interior = self.transform.synthesize(coarse, packets)
        pressure = interior.reshape(batch, count, *self.transform.shape)
        return F.pad(pressure, (0, 0, 1, 0))

    def iter_pressure(self, features, temporal, frame_indices=None, *, time_chunk=None):
        """Stream pressure chunks while reusing one shared spatial preparation."""
        prepared = features if isinstance(features, PreparedDetailFeatures) else self.prepare_spatial(features)
        for chunk in self.iter_coefficients(prepared, temporal, frame_indices, time_chunk=time_chunk):
            yield chunk.start, chunk.stop, self.synthesize(chunk.coefficients)

    def pressure_correction(self, features, temporal, frame_indices=None, *, time_chunk=None):
        """Return [B,T,Z+1,X] fixed-synthesis pressure corrections."""
        return torch.cat([pressure for _, _, pressure in self.iter_pressure(
            features, temporal, frame_indices, time_chunk=time_chunk)], dim=1)


class TimeConditionedCoarsestBaseHead(TimeConditionedDetailHead):
    """Separate base-field correction in the terminal sixteen SS slots.

    This optional branch changes the base field's coarsest component. It is
    deliberately separate from the detail head, whose fixed synthesis still
    leaves every base coarsest coefficient unchanged. Its readout requires
    its own residual-QR calibration before it can contribute to predictions.
    """

    def __init__(self, in_channels, time_features, transform, **kwargs):
        super().__init__(in_channels, time_features, transform, **kwargs)
        last = self.transform.levels[-1]
        projection_channels = self.feature_projection.out_channels
        spatial = nn.Linear(projection_channels * last.modes + 16, self.hidden_width)
        output = nn.Linear(self.hidden_width, 16)
        for module in (spatial, output):
            nn.init.xavier_uniform_(module.weight, gain=1.)
            nn.init.zeros_(module.bias)
        self.spatial_first = nn.ModuleList([spatial])
        self.temporal_first = nn.ModuleList([self.temporal_first[-1]])
        self.second_hidden = nn.ModuleList([self.second_hidden[-1]])
        self.readouts = nn.ModuleList([output])
        self.register_buffer("valid_coarsest", torch.ones(last.gz, last.gx, 16, dtype=torch.bool), persistent=False)

    @property
    def valid_masks(self):
        return (self.valid_coarsest,)

    def prepare_spatial(self, features):
        if features.ndim != 4 or features.shape[1] != self.in_channels or not features.shape[0]:
            raise ValueError("expected source-conditioned spatial features [B,C,Z,X]")
        height, width = self.transform.shape
        if features.shape[-2:] == (height + 1, width):
            features = features[:, :, 1:]
        elif features.shape[-2:] != (height, width):
            raise ValueError("spatial features must cover the active grid or its zero-extended pressure grid")
        if self.spatial_refiner is not None:
            features = features + self.spatial_refiner(features)
        _, packets = self.transform.analyze(self.feature_projection(features))
        block = packets[-1].permute(0, 2, 3, 1, 4).flatten(-2)
        mask = self.valid_coarsest.to(block).expand(features.shape[0], -1, -1, -1)
        return PreparedDetailFeatures((self.spatial_first[0](torch.cat((block, mask), dim=-1)),))

    def targets_from_residual(self, residual):
        """Pack the terminal coarse field of normalized [B,T,Z,X] residuals."""
        height, width = self.transform.shape
        if residual.ndim != 4:
            raise ValueError("expected normalized pressure residuals [B,T,Z,X]")
        if residual.shape[-2:] == (height + 1, width):
            residual = residual[:, :, 1:]
        elif residual.shape[-2:] != (height, width):
            raise ValueError("residual geometry must match the inherited active grid")
        batch, count = residual.shape[:2]
        coarse, _ = self.transform.analyze(residual.reshape(batch * count, 1, height, width))
        last = self.transform.levels[-1]
        packed = coarse.reshape(batch, count, last.gz, 4, last.gx, 4).transpose(-3, -2).flatten(-2)
        return (packed,)

    def synthesize(self, coefficients):
        if len(coefficients) != 1:
            raise ValueError("the base-coarse branch has exactly one terminal SS packet")
        values = coefficients[0]
        batch, count = values.shape[:2]
        last = self.transform.levels[-1]
        if values.shape != (batch, count, last.gz, last.gx, 16):
            raise ValueError("terminal SS packet geometry changed")
        coarse = values.reshape(batch * count, 1, last.gz, last.gx, 4, 4)
        coarse = coarse.transpose(-3, -2).reshape(batch * count, 1, last.gz * 4, last.gx * 4)
        zeros = [values.new_zeros(batch * count, 1, level.gz, level.gx, level.valid_detail.shape[-1])
                 for level in self.transform.levels]
        interior = self.transform.synthesize(coarse, zeros)
        return F.pad(interior.reshape(batch, count, *self.transform.shape), (0, 0, 1, 0))
