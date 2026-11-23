"""Full inherited training scope for one original paper Marmousi201 instance.

Every inherited requires_grad flag and the native zero-IC gradient hook are
retained. Use make_optimizer: its original parameter groups give the masked
refiner input tensor zero weight decay. Losses, complete400 supervision,
residual AIS and assessment are the unchanged canonical paper201 functions.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import torch

from .r25_cf_full400_operator import build_r25_cf_full400_operator
from .paper_marmousi_fullfield_20261020 import (
    sha, immutable_digest, trainable_digest, cpu_tree, equal_tree,
    dg_geometry, dg_energy, fixed_supervision, pressure_metrics, assess, backward,
)

SCHEMA = 'PAPER_MARMOUSI201_SINGLE_SOURCE_FULL_INHERITED_FIT_AIS_V1'
PAPER_CHECKPOINT_SHA256 = 'bc3b955472b59a3d26cdacfbeff5641e07deccfa0b5c7422dee36ad00e9210ff'
MASKED_INPUT_NAME = 'refiner.static.0.conv1.weight'
PERMANENT_FROZEN_NAME = 'base.source_encoder.ic_encoder.0.weight'
ROOT = Path(__file__).resolve().parents[2]


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def build(config, checkpoint, device='cpu'):
    """Strict epoch65 restore; do not change any inherited trainability flag."""
    checkpoint = Path(checkpoint)
    _require(sha(checkpoint) == PAPER_CHECKPOINT_SHA256,
        'original paper Marmousi epoch65 checkpoint required')
    model = build_r25_cf_full400_operator(config, device='cpu', kind='marmousi')
    inherited = {n: p.requires_grad for n, p in model.named_parameters()}
    saved = torch.load(str(checkpoint), map_location='cpu', mmap=True, weights_only=False)
    _require(isinstance(saved, dict) and 'model_state' in saved, 'complete paper model_state required')
    model.load_state_dict(saved['model_state'], strict=True)
    _require(inherited == {n: p.requires_grad for n, p in model.named_parameters()},
        'strict state load changed inherited trainability')
    model.__dict__['_paper_fullfit_inherited_flags'] = inherited
    model.to(device)
    selection(model)
    return model


def selection(model):
    """Complete inherited-active roster, permanent freezes and module counts."""
    inherited = model.__dict__.get('_paper_fullfit_inherited_flags')
    named = dict(model.named_parameters())
    _require(isinstance(inherited, dict) and set(inherited) == set(named),
        'model must come from this complete paper201 initializer')
    _require(inherited == {n: p.requires_grad for n, p in named.items()},
        'inherited trainability flags changed')
    groups, active, frozen = {}, [], []
    for name, p in named.items():
        group = '.'.join(name.split('.')[:2])
        row = groups.setdefault(group, dict(trainable_parameters=0, frozen_parameters=0,
            trainable_tensors=0, frozen_tensors=0))
        kind = 'trainable' if p.requires_grad else 'frozen'
        row[kind+'_parameters'] += p.numel()
        row[kind+'_tensors'] += 1
        (active if p.requires_grad else frozen).append(name)
    count = sum(p.numel() for p in named.values() if p.requires_grad)
    total = sum(p.numel() for p in named.values())
    _require(frozen == [PERMANENT_FROZEN_NAME], 'unexpected permanent-freeze roster')
    _require(named[MASKED_INPUT_NAME].requires_grad, 'native masked input tensor must remain active')
    return dict(schema=SCHEMA, names=active, tensors=len(active), parameters=count,
        total_parameters=total, trainable_fraction=count/total, groups=groups,
        frozen_names=frozen, permanent_frozen_names=frozen, frozen_tensors=len(frozen),
        frozen_parameters=total-count, inherited_trainability_preserved=True,
        masked_input_parameter=MASKED_INPUT_NAME, masked_IC_input_channels=[1, 9],
        masked_input_weight_decay=0., zero_IC_gradient_hook_retained=True,
        policy='all inherited-active paper201 parameters; no additional freezing',
        pressure_shape=[400, 201, 201], measured_pressure_inputs=0,
        gradient_contract='every inherited-active tensor must receive a finite gradient; connected zero gradients are permitted')


def make_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """Fresh original grouped AdamW, preserving masked-input weight_decay=0."""
    _require(type(lr) in (int, float) and math.isfinite(lr) and lr > 0., 'positive finite learning rate required')
    _require(type(weight_decay) in (int, float) and math.isfinite(weight_decay)
        and weight_decay >= 0., 'finite nonnegative weight decay required')
    selection(model)
    groups = model.parameter_groups(base_lr=lr, refiner_lr=lr, weight_decay=weight_decay)
    ids = [id(p) for g in groups for p in g['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    _require(len(ids) == len(set(ids)) and set(ids) == expected,
        'original optimizer groups must cover each active tensor exactly once')
    masked = [g for g in groups if g.get('group_name') == 'refiner_masked_input']
    parameter = dict(model.named_parameters())[MASKED_INPUT_NAME]
    _require(len(masked) == 1 and len(masked[0]['params']) == 1
        and masked[0]['params'][0] is parameter and masked[0]['weight_decay'] == 0.,
        'native masked input requires its own zero-decay group')
    _require(all(g['weight_decay'] == weight_decay for g in groups
        if g.get('group_name') != 'refiner_masked_input'), 'ordinary inherited group decay changed')
    return torch.optim.AdamW(groups, foreach=False)


def optimizer_selection(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [dict(name=g.get('group_name'), learning_rate=float(g['lr']),
        weight_decay=float(g['weight_decay']), tensors=len(g['params']),
        parameters=sum(p.numel() for p in g['params']),
        parameter_names=[names[id(p)] for p in g['params']]) for g in optimizer.param_groups]


def cpu_contract(config_path=None, output_path=None, times=(.25, .6, .95)):
    """Actual three-time CPU gradient probe, zero optimizer/GPU updates."""
    import numpy as np
    from label_quality_fd.data_generation.raw_dataset import MediumFactory
    started = time.monotonic()
    config_path = Path(config_path or ROOT/'configs/native400/paper_marmousi_two_sources_fullfield_ais_20261020.json')
    output_path = Path(output_path or ROOT/'runs/paper_marmousi201_single_source_fullfit_20261020/FULLFIT_CPU_CHECKS.json')
    cfg = json.loads(config_path.read_text())
    checkpoint = Path(cfg['checkpoint'])
    model = build(cfg['native_config'], checkpoint, device='cpu')
    optimizer = make_optimizer(model)
    roster = selection(model)
    record = cfg['records'][0]
    fine = MediumFactory(cfg['physics_config']).make(record)
    c = torch.from_numpy(np.array(fine[::2, ::2], copy=True))[None, None]
    f = torch.tensor(record['source_model_fp32'], dtype=torch.float32)[None]
    t = torch.tensor(times, dtype=torch.float32)
    _require(c.shape == (1, 1, 201, 201) and bool(((t >= 0) & (t <= 1)).all()),
        'original sharp201 and physical-time inputs required')
    immutable_before, active_before = immutable_digest(model), trainable_digest(model)
    with model.scopes():
        prediction = model(c, f, t, normalized=True)
        _require(prediction.shape == (len(t), 201, 201) and prediction.dtype == torch.float32,
            'native FP32 pressure output required')
        _require(torch.isfinite(prediction).all() and (prediction[:, 0] == 0).all(),
            'finite field and exact top boundary required')
        loss = prediction.square().mean(dtype=torch.float32)
        loss.backward()
    named = dict(model.named_parameters())
    missing = [n for n, p in named.items() if p.requires_grad and p.grad is None]
    nonfinite = [n for n, p in named.items() if p.requires_grad and p.grad is not None
        and not bool(torch.isfinite(p.grad).all())]
    zero = [n for n, p in named.items() if p.requires_grad and p.grad is not None
        and not bool(p.grad.abs().any())]
    frozen_grads = [n for n, p in named.items() if not p.requires_grad and p.grad is not None]
    masked = named[MASKED_INPUT_NAME]
    masked_zero = masked.grad is not None and bool((masked.grad[:, 1:9] == 0).all())
    group_rows = optimizer_selection(model, optimizer)
    actual_ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    coverage = len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == {
        id(p) for p in model.parameters() if p.requires_grad}
    immutable_same = immutable_digest(model) == immutable_before
    active_same = trainable_digest(model) == active_before
    passed = (not missing and not nonfinite and not frozen_grads and masked_zero
        and coverage and immutable_same and active_same and not torch.cuda.is_initialized())
    statistics = {n: dict(shape=list(p.shape), dtype=str(p.grad.dtype),
        norm=float(torch.linalg.vector_norm(p.grad)), finite=bool(torch.isfinite(p.grad).all()))
        for n, p in named.items() if p.requires_grad and p.grad is not None}
    report = dict(schema=SCHEMA, status='PASS' if passed else 'FAIL',
        module_SHA256=sha(Path(__file__)), config_SHA256=sha(config_path),
        paper_checkpoint_SHA256=sha(checkpoint), selection=roster, optimizer_groups=group_rows,
        actual_CPU_native_forward_backward=True, physical_times_s=list(times),
        pressure_shape=list(prediction.shape), source=record['source'], loss=float(loss.detach()),
        all_selected_gradients_present_and_finite=not missing and not nonfinite,
        missing_gradient_names=missing, nonfinite_gradient_names=nonfinite,
        connected_zero_gradient_names=zero, frozen_parameter_gradient_names=frozen_grads,
        masked_IC_input_slice_gradient_exactly_zero=masked_zero,
        masked_input_weight_decay_zero=True, optimizer_covers_active_once=coverage,
        gradient_statistics=statistics, inherited_trainability_preserved=True,
        permanently_frozen_parameters_and_all_buffers_bitwise_unchanged=immutable_same,
        active_parameters_unchanged_without_optimizer=active_same,
        optimizer_updates=0, scientific_FIT_updates=0, CUDA_initialized=torch.cuda.is_initialized(),
        seconds=time.monotonic()-started,
        interpretation='Actual native CPU gradient/group contract; no optimizer update, accuracy or convergence claim.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name('.'+output_path.name+'.tmp')
    with temporary.open('w') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    _require(passed, 'full inherited-scope CPU contract failed; inspect explicit parameter lists')
    return report
