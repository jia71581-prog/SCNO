"""Differentiable B5-only decoder for the frozen 801-row Marmousi UL.

The same encoder, propagation context, basis weights, and arrival module are
used. The full-pressure method remains available on the reference interface.
"""

import torch
from torch import nn
from torch.nn import functional as F

from arrival_phase import ArrivalReceivers
from joint_fno_decoder import _frame_indices
from joint_fno_propagation import JointFNOPropagation
from marmousi_ul_model_801 import MarmousiTravelTimeFNOSCNO
from method_repair_model import CorrectedSCNOReadout
from spectral_detail import PolynomialPyramid
from wide_base_readout import base_readout_semantics
from wide_time_basis import decoder_identity


_WIDTH = 201
_SUPPORT_ROWS = 4
_ANALYSIS_ROWS = (32, 16, 8)
_ANALYSIS_GROUPS = (4, 2, 1)
_SYNTHESIS_ROWS = (3, 4, 4)


def _pointwise(layer):
    return (type(layer) is nn.Conv2d and layer.kernel_size == (1, 1)
            and layer.stride == (1, 1) and layer.padding == (0, 0)
            and layer.dilation == (1, 1) and layer.groups == 1)


def _validate_configuration(model, readout):
    if (type(model) is not CorrectedSCNOReadout
            or type(model.parent) is not MarmousiTravelTimeFNOSCNO
            or type(model.parent.propagation) is not JointFNOPropagation
            or model.nonlinear_details or model.coarsest_base or model.local_phase
            or model.basis_size != 800 or model.base_hidden_width != 800
            or model.detail_head is not None or model.coarse_head is not None
            or readout.support_depth_rows != _SUPPORT_ROWS
            or readout.matrix.shape != (_SUPPORT_ROWS * _WIDTH, _WIDTH)):
        raise ValueError("B5 reduction requires the fixed Marmousi UL/arrival layout")
    parent, decoder = model.parent, model.parent.decoder
    base = base_readout_semantics(decoder)
    detail = decoder_identity(decoder)
    if (base["profile"] != "marmousi" or base["base_hidden_width"] != 800
            or base["native_frames"] != 801 or detail["time_basis"] != 800
            or not _pointwise(parent.gabor_spatial) or not _pointwise(decoder.detail_projection)
            or type(decoder.detail_transform) is not PolynomialPyramid
            or parent.propagation.delay_bins != 512
            or not torch.equal(parent.propagation.native_times, parent.full_times_s)):
        raise ValueError("Marmousi decoder or propagation structure differs from the reviewed source")
    levels = decoder.detail_transform.levels
    if tuple((level.nz, level.nx, level.gz, level.gx) for level in levels) != (
            (200, 201, 25, 25), (100, 100, 13, 13), (52, 52, 7, 7)):
        raise ValueError("Polynomial transform grid differs from the reviewed stencil")
    for depth, groups, level in zip(_ANALYSIS_ROWS, _ANALYSIS_GROUPS, levels):
        gather = _gather_rows(level, groups)
        if not (int(gather.min()) == 0 and int(gather.max()) == depth * level.nx - 1):
            raise ValueError("Polynomial analysis has an unexpected shallow dependency")
    for count, level in zip(_SYNTHESIS_ROWS, levels):
        unpad = _unpad_rows(level, count)
        if int(unpad.max()) >= level.gz_modes * level.gx * level.gx_modes:
            raise ValueError("Polynomial synthesis reaches past the first depth block")


def _gather_rows(level, groups):
    shape = (level.gz, level.gx, level.gz_modes, level.gx_modes)
    return level.gather.reshape(shape)[:groups].flatten()


def _unpad_rows(level, count):
    return level.unpad.reshape(level.nz, level.nx)[:count].flatten()


def _detail_features(decoder, features):
    projection = decoder.detail_projection(features[:, :, 1:1 + _ANALYSIS_ROWS[0]])
    hidden = []
    with torch.autocast(device_type=projection.device.type, enabled=False):
        field = projection.float()
        batch, channels = field.shape[:2]
        for groups, level in zip(_ANALYSIS_GROUPS, decoder.detail_transform.levels):
            blocks = field.flatten(-2).index_select(-1, _gather_rows(level, groups))
            blocks = blocks.reshape(batch, channels, groups, level.gx, level.gz_modes, level.gx_modes)
            modes = level.qz[:groups].transpose(-2, -1) @ (blocks @ level.qx)
            modes = modes.flatten(-2).index_select(-1, level.order)
            top = modes[:, :, :1].permute(0, 2, 3, 1, 4).flatten(-2)
            mask = level.valid_detail[:1].to(top).expand(batch, -1, -1, -1)
            hidden.append(torch.cat((top, mask), dim=-1))
            ss = modes[..., :16].reshape(batch, channels, groups, level.gx, 4, 4)
            field = ss.transpose(-3, -2).reshape(batch, channels, groups * 4, level.gx * 4)
    # The learned head follows the caller's precision policy, as in wide_time_basis.
    for index, head in enumerate(decoder.detail_heads):
        for layer in tuple(head)[:-1]:
            hidden[index] = layer(hidden[index])
    return hidden


def _detail_packets(model, features, slots):
    decoder = model.parent.decoder
    hidden = _detail_features(decoder, features)
    batch = features.shape[0]
    if slots is None:
        slots = torch.arange(decoder.native_times.numel(), device=features.device)
    packets = []
    for value, head, temporal, level in zip(hidden, decoder.detail_heads, decoder.detail_time,
                                             decoder.detail_transform.levels):
        rank, width = decoder.detail_latent, head[-1].in_features
        n_detail = level.valid_detail.shape[-1]
        weights = head[-1].weight.reshape(rank, n_detail * width)
        bias = head[-1].bias
        chunks = []
        for selected in slots.split(model.time_chunk, dim=-1):
            if not selected.numel():
                continue
            rows = temporal.weight[selected]
            folded = (rows @ weights).reshape(*rows.shape[:-1], n_detail, width)
            folded_bias = rows @ bias.reshape(rank, n_detail) if bias is not None else None
            if selected.ndim == 1:
                coefficient = torch.einsum("bxyh,ksh->bkxys", value, folded)
                if folded_bias is not None:
                    coefficient = coefficient + folded_bias[None, :, None, None, :]
            else:
                coefficient = torch.einsum("bxyh,bksh->bkxys", value, folded)
                if folded_bias is not None:
                    coefficient = coefficient + folded_bias[:, :, None, None, :]
            chunks.append(coefficient.masked_fill(~level.valid_detail[:1], 0))
        packets.append(torch.cat(chunks, dim=1) if chunks else
                       value.new_empty((batch, 0, 1, level.gx, n_detail)))
    return packets


def _synthesize_detail(model, features, slots, count):
    levels = model.detail_transform.levels
    details = _detail_packets(model, features, slots)
    batch = features.shape[0]
    field = features.new_zeros((batch, count, 4, levels[-1].gx * 4))
    with torch.autocast(device_type=features.device.type, enabled=False):
        field = field.float()
        for depth, level, detail in reversed(tuple(zip(_SYNTHESIS_ROWS, levels, details))):
            coarse = field.reshape(batch, count, 1, 4, level.gx, 4)
            coarse = coarse.transpose(-3, -2).flatten(-2)
            detail = detail.float().masked_fill(~level.valid_detail[:1], 0)
            modes = torch.cat((coarse, detail), dim=-1).index_select(-1, level.inverse_order)
            modes = modes.reshape(batch, count, 1, level.gx, level.gz_modes, level.gx_modes)
            blocks = level.qz[:1] @ (modes @ level.qx.transpose(-2, -1))
            field = blocks.transpose(-3, -2).flatten(-4)
            field = field.index_select(-1, _unpad_rows(level, depth)).reshape(batch, count, depth, level.nx)
    return F.pad(field, (0, 0, 1, 0))


def _inherited_pressure(parent, context, slots, times):
    features, source, physics = context
    shallow = features[:, :, 1:_SUPPORT_ROWS]
    base = shallow
    for layer in tuple(parent.decoder.proj)[:-1]:
        base = layer(base)
    final = parent.decoder.proj[-1]
    weights = final.weight[:, :, 0, 0]
    bias = final.bias
    if slots is not None:
        weights, bias = weights[slots], bias[slots]
    if weights.ndim == 2:
        weights = weights.unsqueeze(0).expand(len(source), -1, -1)
        bias = bias.unsqueeze(0).expand(len(source), -1)
    bank = parent.gabor(times, source[:, 3:4], source[:, 2:3], parent.duration_s)
    temporal = parent.gabor_embedding(bank)
    gabor = parent.gabor_spatial(shallow)

    # Delay bins, pulse rows, and the span are still set by the full encoder.
    rows = parent.propagation.time_rows(physics, slots)
    nodes = slice(_WIDTH, _SUPPORT_ROWS * _WIDTH)
    index = physics.indices[..., nodes].flatten(1, 2)
    values = physics.weights[..., nodes].flatten(1, 2)
    carrier = values.new_zeros((len(source), parent.propagation.delay_bins, values.shape[-1]))
    carrier.scatter_add_(1, index, values)
    carrier = carrier.reshape(len(source), parent.propagation.delay_bins, _SUPPORT_ROWS - 1, _WIDTH)
    basis = torch.cat((base, gabor, torch.ones_like(gabor[:, :1]), carrier), dim=1)
    row_groups = tuple(term * parent.residual_gain for term in (weights, temporal, bias[..., None]))
    factors = torch.cat((*row_groups, rows), dim=-1)
    factors = factors * (source[:, 4] != 0).to(features.dtype)[:, None, None]
    basis = F.pad(basis, (0, 0, 1, 0))
    return torch.bmm(factors, basis.flatten(2)).reshape(len(source), times.shape[1], _SUPPORT_ROWS, _WIDTH)


class ShallowArrivalReceivers(ArrivalReceivers):
    """Same checkpoint and training graph, with only B5 dependency rows decoded."""

    def __init__(self, model, residual, readout):
        _validate_configuration(model, readout)
        super().__init__(model, residual, readout)

    @classmethod
    def from_reference(cls, reference):
        return cls(reference.model, reference.residual, reference.readout)

    def decode_shallow_pressure(self, context, frame_indices=None, *, normalized=True):
        slots = _frame_indices(frame_indices, len(context.source), context.features.device)
        if slots is not None and slots.numel() and (slots.min() < 0 or slots.max() >= len(self.model.full_times_s)):
            raise ValueError("frame indices must be within the native time axis")
        times = self.model.parent._selected_times(slots, len(context.source))
        inherited = _inherited_pressure(self.model.parent, context, slots, times)
        detail = _synthesize_detail(self.model, context.features, slots, times.shape[1])
        source_amp = context.source[:, 4]
        pressure = inherited + detail * (source_amp != 0)[:, None, None, None]
        if not normalized:
            scale = self.model.normalizer.metadata.pressure_scale_pa * source_amp
            pressure = pressure * scale[:, None, None, None]
        correction = self._correction(context, slots, _SUPPORT_ROWS, normalized)
        return pressure + correction

    def decode(self, context, frame_indices=None, *, normalized=True):
        slab = self.decode_shallow_pressure(context, frame_indices, normalized=normalized).flatten(-2)
        return slab @ self.readout.matrix.to(slab)
