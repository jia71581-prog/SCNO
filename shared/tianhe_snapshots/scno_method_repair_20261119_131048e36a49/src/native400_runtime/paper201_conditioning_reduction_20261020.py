"""Affine conditioning reduction for a private original-paper Marmousi model.

The source frequency/delay tolerance and arbitrary positive source amplitude
are retained. These are real-arithmetic parameter transformations; FP32
reduction-order changes need numerical admission, not a bitwise claim.
"""
from __future__ import annotations

import copy
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'PAPER201_CONDITIONING_AFFINE_REDUCTION_V1'
RAW_FEATURES = 63
RAW_KEEP = tuple(i for i in range(RAW_FEATURES) if i != 8)


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _parameter(value, reference):
    return nn.Parameter(value.detach().clone(), requires_grad=reference.requires_grad)


class SelectedInputLinear(nn.Linear):
    """Store 62 effective columns while consuming the original 63 features."""

    def __init__(self, source, weight, keep=RAW_KEEP):
        # No random initialization: checkpoint compaction must preserve RNG.
        nn.Module.__init__(self)
        self.original_in_features = source.in_features
        self.in_features = len(keep)
        self.out_features = source.out_features
        self.weight = _parameter(weight, source.weight)
        self.bias = None if source.bias is None else _parameter(source.bias, source.bias)
        self.register_buffer('_conditioning_columns',
            torch.tensor(keep, dtype=torch.long, device=weight.device), persistent=False)

    def forward(self, value):
        _require(value.shape[-1] == self.original_in_features, 'original63 coordinate features required')
        return F.linear(value.index_select(-1, self._conditioning_columns), self.weight, self.bias)


class SelectedKappaConv(nn.Conv2d):
    """Store c and alpha=1 kappa columns; still accept the original five."""

    def __init__(self, source, weight):
        nn.Module.__init__(self)
        self.in_channels, self.out_channels = 2, source.out_channels
        self.kernel_size = (1, 1)
        self.stride, self.padding, self.dilation = (1, 1), (0, 0), (1, 1)
        self.groups, self.padding_mode = 1, 'zeros'
        self.output_padding = (0, 0)
        self.transposed = False
        self._reversed_padding_repeated_twice = (0, 0, 0, 0)
        self.weight = _parameter(weight, source.weight)
        self.bias = None if source.bias is None else _parameter(source.bias, source.bias)
        self.register_buffer('_conditioning_columns',
            torch.tensor([0, 2], dtype=torch.long, device=weight.device), persistent=False)

    def forward(self, value):
        _require(value.ndim == 4 and value.shape[1] == 5, 'original five c/kappa channels required')
        return F.conv2d(value.index_select(1, self._conditioning_columns), self.weight, self.bias)


def _check_anchor(module, anchor, expected_shape):
    _require(anchor is not None, 'fixed registered frame88 anchor must be supplied')
    reference = module._conditioning_fixed_anchor
    value = torch.as_tensor(anchor, device=reference.device, dtype=reference.dtype)
    _require(value.numel() > 0 and tuple(value.shape) == expected_shape, 'registered anchor shape changed')
    _require(bool(torch.isfinite(value).all()) and bool((value == reference).all()),
             'conditioning reduction only admits the original fixed frame88 anchor')


def _source_forward(self, source_normalized, source_map, medium, record_to_medium,
                    ic_snapshots=None, anchor_normalized=None):
    _check_anchor(self, anchor_normalized, (source_normalized.shape[0], 1))
    return self._conditioning_original_forward(self, source_normalized, source_map, medium,
        record_to_medium, ic_snapshots=ic_snapshots, anchor_normalized=None)


def _coordinate_forward(self, coords_xyz_m, source_parameters, travel, *,
                        domain_x_m, domain_z_m, domain_t_s, anchor_time_s=None):
    _check_anchor(self, anchor_time_s, (coords_xyz_m.shape[0],))
    _require(domain_t_s == self._conditioning_domain_t_s, 'coordinate physical time scale changed')
    return self._conditioning_original_forward(self, coords_xyz_m, source_parameters, travel,
        domain_x_m=domain_x_m, domain_z_m=domain_z_m, domain_t_s=domain_t_s, anchor_time_s=None)


def _linear(layer, inputs, outputs=None):
    _require(type(layer) is nn.Linear and layer.in_features == inputs and layer.bias is not None,
             'expected original biased linear conditioning layer')
    if outputs is not None:
        _require(layer.out_features == outputs, 'conditioning hidden width mismatch')


def reduce_conditioning(model):
    """Mutate one private CPU model once and return its auditable receipt.

    Build the same original model and call this function before strict loading
    a reduced state_dict. No optimizer state is transformed by this function.
    """
    _require(not hasattr(model, '_conditioning_reduction_receipt'), 'conditioning already reduced')
    _require(all(p.device.type == 'cpu' for p in model.parameters()), 'reduce a private CPU model before device transfer')
    source, coordinate = model.base.source_encoder, model.base.coordinate_encoder
    dense, cycle = model.base.dense_decoder, model.base.scno_cycle
    _require(model.native_times_s.ndim == 1 and model.native_times_s.numel() == 801,
             'original801 time authority required')
    T = float(model.base.domain_t_s)
    _require(T == 1., 'original one-second physical domain required')
    anchor_time = model.native_times_s[88].detach().clone()
    anchor = anchor_time / T
    _require(float(anchor_time) == float(torch.tensor(.11, dtype=anchor_time.dtype)), 'registered frame88 time changed')
    hidden = source.parameter_mlp[2].out_features
    _linear(source.parameter_mlp[0], 5, hidden)
    _linear(source.parameter_mlp[2], hidden, hidden)
    _linear(source.anchor_projection, 1, hidden)
    for name in ['local_projection', 'map_projection', 'ic_projection']:
        _linear(getattr(source, name), hidden, hidden)
    _linear(coordinate.raw_projection, RAW_FEATURES)
    _linear(coordinate.periodic_in.linear, RAW_FEATURES, coordinate.raw_projection.out_features)
    _linear(coordinate.anchor_raw_projection, 2, coordinate.raw_projection.out_features)
    _require(type(cycle.La[0]) is nn.Conv2d and cycle.La[0].in_channels == 5
             and cycle.La[0].kernel_size == (1, 1) and cycle.La[0].stride == (1, 1)
             and cycle.La[0].padding == (0, 0) and cycle.La[0].dilation == (1, 1)
             and cycle.La[0].groups == 1, 'expected original pointwise SCNO coefficient lift')
    _require(torch.equal(cycle.alpha.cpu(), torch.tensor([.5, 1., 1.5, 2.], dtype=cycle.alpha.dtype)),
             'fixed SCNO kappa alpha values changed')
    _require(dense.multiscale_fuse.bias is not None and dense.source_projection.bias is not None
             and dense.map_projection.bias is not None
             and dense.multiscale_fuse.kernel_size == dense.map_projection.kernel_size == (1, 1)
             and dense.multiscale_fuse.bias.shape == dense.source_projection.bias.shape == dense.map_projection.bias.shape,
             'three same-width additive dense spatial biases required')
    _require(hasattr(source.forward, '__func__') and hasattr(coordinate.forward, '__func__'),
             'original bound source/coordinate forwards required')
    before = sum(p.numel() for p in model.parameters())
    tensors_before = sum(1 for _ in model.parameters())
    rng = torch.get_rng_state().clone()
    raw_old_shape = tuple(coordinate.raw_projection.weight.shape)
    periodic_old_shape = tuple(coordinate.periodic_in.linear.weight.shape)
    kappa_old_shape = tuple(cycle.La[0].weight.shape)

    with torch.no_grad():
        constant = F.linear(anchor.reshape(1, 1).to(source.anchor_projection.weight),
                            source.anchor_projection.weight, source.anchor_projection.bias).reshape(-1)
        bias = source.parameter_mlp[2].bias.detach().clone()
        for name in ['local_projection', 'map_projection', 'ic_projection']:
            bias.add_(getattr(source, name).bias)
        bias.add_(constant)
        source.parameter_mlp[2].bias.copy_(bias)
        for name in ['local_projection', 'map_projection', 'ic_projection']:
            getattr(source, name).register_parameter('bias', None)
        del source.anchor_projection
        source.register_buffer('_conditioning_fixed_anchor', anchor.to(source.parameter_mlp[2].bias), persistent=False)
        source.__dict__['_conditioning_original_forward'] = source.forward.__func__
        source.__dict__['forward'] = MethodType(_source_forward, source)

        raw, periodic, anchored = coordinate.raw_projection, coordinate.periodic_in.linear, coordinate.anchor_raw_projection
        raw_weight = raw.weight.detach().clone()
        raw_weight[:, 2].add_(anchored.weight[:, 0] - anchored.weight[:, 1])
        raw_weight[:, 6].add_(anchored.weight[:, 1])
        raw_bias = raw.bias + anchored.bias + anchor.to(raw.bias) * (anchored.weight[:, 1] - anchored.weight[:, 0])
        # raw[8] = raw[6] - raw[7] for every source delay, including tolerance.
        raw_weight[:, 6].add_(raw_weight[:, 8])
        raw_weight[:, 7].sub_(raw_weight[:, 8])
        periodic_weight = periodic.weight.detach().clone()
        periodic_weight[:, 6].add_(periodic_weight[:, 8])
        periodic_weight[:, 7].sub_(periodic_weight[:, 8])
        coordinate.raw_projection = SelectedInputLinear(raw, raw_weight[:, RAW_KEEP])
        coordinate.raw_projection.bias.copy_(raw_bias)
        coordinate.periodic_in.linear = SelectedInputLinear(periodic, periodic_weight[:, RAW_KEEP])
        del coordinate.anchor_raw_projection
        coordinate.register_buffer('_conditioning_fixed_anchor', anchor_time.to(raw.bias), persistent=False)
        coordinate.__dict__['_conditioning_domain_t_s'] = T
        coordinate.__dict__['_conditioning_original_forward'] = coordinate.forward.__func__
        coordinate.__dict__['forward'] = MethodType(_coordinate_forward, coordinate)

        old = cycle.La[0]
        coefficients = cycle.alpha.to(old.weight)
        kappa_weight = torch.stack((old.weight[:, 0],
            (old.weight[:, 1:] * coefficients[None, :, None, None]).sum(1)), dim=1)
        cycle.La[0] = SelectedKappaConv(old, kappa_weight)
        dense.multiscale_fuse.bias.add_(dense.source_projection.bias).add_(dense.map_projection.bias)
        dense.source_projection.register_parameter('bias', None)
        dense.map_projection.register_parameter('bias', None)

    _require(torch.equal(rng, torch.get_rng_state()), 'conditioning reduction must preserve CPU RNG')
    after = sum(p.numel() for p in model.parameters())
    receipt = dict(schema=SCHEMA, anchor_time_s=float(anchor_time), normalized_anchor=float(anchor),
        domain_t_s=T, parameters_before=before, parameters_after=after, parameters_removed=before-after,
        parameter_tensors_removed=tensors_before-sum(1 for _ in model.parameters()),
        source_anchor_fold_target='base.source_encoder.parameter_mlp.2.bias',
        source_additive_bias_names=['local_projection.bias', 'map_projection.bias', 'ic_projection.bias'],
        coordinate_anchor_fold_target='base.coordinate_encoder.raw_projection',
        coordinate_removed_input_column=8, original_coordinate_features=RAW_FEATURES,
        effective_coordinate_features=len(RAW_KEEP), scno_kept_input_columns=[0, 2],
        scno_kappa_coefficients=cycle.alpha.cpu().tolist(), dense_spatial_biases_merged=True,
        original_source_parameter_columns_retained=True, original_frequency_delay_tolerance_retained=True,
        arbitrary_positive_amplitude_retained=True, no_medium_constant_folding=True,
        original_parameter_shapes=dict(raw_weight=list(raw_old_shape), periodic_weight=list(periodic_old_shape),
                                       kappa_weight=list(kappa_old_shape)),
        CPU_RNG_preserved=True, exact_arithmetic_equivalence=True, bitwise_FP32_equivalence_claim=False,
        optimizer_state_transformed=False)
    model.__dict__['_conditioning_reduction_receipt'] = copy.deepcopy(receipt)
    return receipt


def _gradient(parameter, name):
    _require(isinstance(parameter, nn.Parameter) and parameter.grad is not None,
             'complete reduced conditioning gradients required: ' + name)
    gradient = parameter.grad.detach()
    _require(gradient.shape == parameter.shape and bool(torch.isfinite(gradient).all()),
             'invalid reduced conditioning gradient: ' + name)
    return gradient


def conditioning_metadata(model):
    """Fixed, JSON-safe structure for the enclosing adapter's extra-state bind.

    Read actual buffers and guards, never learned weights or gradients. The
    enclosing adapter also binds its complete native axis and normalizer.
    """
    receipt = getattr(model, '_conditioning_reduction_receipt', None)
    _require(isinstance(receipt, dict) and receipt.get('schema') == SCHEMA, 'reduced conditioning model required')
    source, coordinate = model.base.source_encoder, model.base.coordinate_encoder
    cycle, dense = model.base.scno_cycle, model.base.dense_decoder
    T = float(model.base.domain_t_s)
    _require(T == coordinate._conditioning_domain_t_s == 1., 'conditioning time scale changed')
    a = model.native_times_s[88].detach().cpu()
    _require(torch.equal(source._conditioning_fixed_anchor.detach().cpu(), a/T)
             and torch.equal(coordinate._conditioning_fixed_anchor.detach().cpu(), a), 'fixed registered anchor changed')
    for layer in [coordinate.raw_projection, coordinate.periodic_in.linear]:
        _require(isinstance(layer, SelectedInputLinear) and layer.original_in_features == RAW_FEATURES
                 and layer.in_features == len(RAW_KEEP)
                 and layer._conditioning_columns.detach().cpu().tolist() == list(RAW_KEEP),
                 'effective coordinate input selection changed')
    _require(isinstance(cycle.La[0], SelectedKappaConv) and cycle.La[0].in_channels == 2
             and cycle.La[0]._conditioning_columns.detach().cpu().tolist() == [0, 2]
             and cycle.alpha.detach().cpu().tolist() == [.5, 1., 1.5, 2.], 'effective kappa selection changed')
    _require(not hasattr(source, 'anchor_projection') and not hasattr(coordinate, 'anchor_raw_projection')
             and all(getattr(source, name).bias is None for name in ['local_projection', 'map_projection', 'ic_projection'])
             and dense.source_projection.bias is None and dense.map_projection.bias is None,
             'removed conditioning parameter returned')
    return dict(schema=SCHEMA, anchor_native_index=88, anchor_time_s=float(a), normalized_anchor=float(a/T),
        domain_t_s=T, original_coordinate_features=RAW_FEATURES, coordinate_keep=list(RAW_KEEP),
        scno_input_keep=[0, 2], scno_kappa_coefficients=[.5, 1., 1.5, 2.],
        source_anchor_fold_target='base.source_encoder.parameter_mlp.2.bias',
        coordinate_anchor_fold_target='base.coordinate_encoder.raw_projection',
        source_biases_folded=['local_projection.bias', 'map_projection.bias', 'ic_projection.bias'],
        dense_biases_folded=['source_projection.bias', 'map_projection.bias'],
        original_source_five_columns_preserved=True)


def _expand_raw(gradient):
    old = gradient.new_zeros((gradient.shape[0], RAW_FEATURES))
    old[:, RAW_KEEP] = gradient
    old[:, 8] = old[:, 6] - old[:, 7]
    return old


def legacy_conditioning_gradients(model):
    """Only affected original names -> g_old=A^T g_new, never None entries.

    The caller combines these 15 overrides with its other parameter maps.
    No original AdamW moments or historical optimizer trajectory are implied.
    """
    receipt = getattr(model, '_conditioning_reduction_receipt', None)
    _require(isinstance(receipt, dict) and receipt.get('schema') == SCHEMA, 'reduced conditioning model required')
    source, coordinate = model.base.source_encoder, model.base.coordinate_encoder
    dense, cycle = model.base.dense_decoder, model.base.scno_cycle
    result = {}
    g = _gradient(source.parameter_mlp[2].bias, 'source merged bias')
    anchor = source._conditioning_fixed_anchor.to(g)
    for suffix in ['parameter_mlp.2.bias', 'local_projection.bias', 'map_projection.bias',
                   'ic_projection.bias', 'anchor_projection.bias']:
        result['base.source_encoder.' + suffix] = g.clone()
    result['base.source_encoder.anchor_projection.weight'] = (anchor * g[:, None]).clone()
    raw = _expand_raw(_gradient(coordinate.raw_projection.weight, 'raw projection weight'))
    bias = _gradient(coordinate.raw_projection.bias, 'raw projection bias')
    a = coordinate._conditioning_fixed_anchor.to(bias) / coordinate._conditioning_domain_t_s
    result['base.coordinate_encoder.raw_projection.weight'] = raw
    result['base.coordinate_encoder.raw_projection.bias'] = bias.clone()
    result['base.coordinate_encoder.periodic_in.linear.weight'] = _expand_raw(
        _gradient(coordinate.periodic_in.linear.weight, 'periodic first weight'))
    result['base.coordinate_encoder.anchor_raw_projection.weight'] = torch.stack(
        (raw[:, 2] - a*bias, -raw[:, 2] + raw[:, 6] + a*bias), dim=1)
    result['base.coordinate_encoder.anchor_raw_projection.bias'] = bias.clone()
    g = _gradient(cycle.La[0].weight, 'SCNO coefficient weight')
    old = g.new_zeros((g.shape[0], 5, 1, 1))
    old[:, 0] = g[:, 0]
    old[:, 1:] = g[:, 1:2] * cycle.alpha.to(g)[None, :, None, None]
    result['base.scno_cycle.La.0.weight'] = old
    g = _gradient(dense.multiscale_fuse.bias, 'dense merged spatial bias')
    for suffix in ['multiscale_fuse.bias', 'source_projection.bias', 'map_projection.bias']:
        result['base.dense_decoder.' + suffix] = g.clone()
    return result
