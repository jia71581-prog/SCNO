"""Paper epoch65 Marmousi: full pressure supervision and joint residual AIS.

This is the paper's 15 Hz/201-node operator, not the later 8 Hz model.
Only small source projections and refiner affine FiLM layers are adapted.
Spatial source-map kernels, zero-IC channels, anchor projection, propagation
and medium kernels stay frozen. All 400 outputs,
including the first eight, are predictions; no reference is copied to output.
The SIPG objective remains a coarse source-free regularizer, not a CPML-state
teacher or an exact residual of the fine-grid reference solver.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

from .r25_cf_full400_operator import build_r25_cf_full400_operator
from paso.dg_weak import get_dg_operator
from .paper_marmousi_joint_ais_20261020 import stopping_summary

PREFIXES = ('base.source_encoder.parameter_mlp.',
            'base.source_encoder.local_projection.',
            'base.source_encoder.map_projection.',
            'base.source_encoder.hidden_norm.',
            'base.source_encoder.rank_projection.',
            'base.dense_decoder.source_projection.',
            'base.scno_cycle.source_proj.')
SCHEMA = 'PAPER_MARMOUSI_FULLFIELD_SMALL_CONDITIONAL_INSTANCE_AIS_V2'


def allowed_parameter(name):
    return name.startswith(PREFIXES) or (
        name.startswith(('refiner.dynamic.', 'refiner.up.')) and '.film.' in name)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build(config, checkpoint, device='cpu'):
    model = build_r25_cf_full400_operator(config, device='cpu', kind='marmousi')
    saved = torch.load(str(checkpoint), map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(saved['model_state'], strict=True)
    inherited = {n: p.requires_grad for n, p in model.named_parameters()}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(inherited[name] and allowed_parameter(name))
        parameter.grad = None
    model.to(device)
    model.train()
    return model


def selection(model):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or any(not allowed_parameter(n) for n, _ in named):
        raise ValueError('Exact conditional-projection and FiLM whitelist required')
    total = sum(p.numel() for p in model.parameters())
    return dict(names=[n for n, _ in named], tensors=len(named),
                parameters=sum(p.numel() for _, p in named), prefixes=list(PREFIXES),
                refiner_policy='only dynamic/up .film. affine maps; all convolution kernels frozen',
                total_parameters=total,trainable_fraction=sum(p.numel() for _, p in named)/total)


def immutable_digest(model):
    """Includes every frozen parameter and all registered persistent/nonpersistent buffers."""
    h = hashlib.sha256()
    values = [('parameter:' + n, p) for n, p in model.named_parameters() if not p.requires_grad]
    values += [('buffer:' + n, b) for n, b in model.named_buffers()]
    for name, value in values:
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str((a.dtype, a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def trainable_digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict): return isinstance(b, dict) and set(a) == set(b) and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) == type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray): return isinstance(b, np.ndarray) and np.array_equal(a, b)
    return a == b


def dg_geometry(model, source):
    op = get_dg_operator(201, 201, 10., 10., 4, 3, 4, model.device, torch.float32)
    mask = op.valid_mask(source[0, :2], 0., 0., 4)
    if not bool(mask.any()): raise ValueError('Nonempty source-free DG domain required')
    return op, mask


def dg_energy(model, c, f, slots, op, *, time_block=None):
    """Graph-connected per-element mean over the sixteen inherited Q3 modes.

    Last registered 1.0s center moves to 0.9995s for an in-domain central
    stencil. This physics grid is explicitly separate from the data time grid.
    """
    slots = slots.to(device=model.device, dtype=torch.long)
    centers = model.full_times_s.index_select(0, slots).clamp(.0005, .9995)
    requested = torch.stack((centers - .0005, centers, centers + .0005), 1)
    p = model(c, f, requested.flatten(), normalized=True).reshape(-1, 3, 201, 201)
    comp = op.components(p[:, 0], p[:, 1], p[:, 2], c[0, 0], .0005, 2.)
    r = sum(comp.values())
    scale = torch.stack([v.detach() for v in comp.values()]).square().mean(0)
    scale = torch.sqrt(scale + r.new_tensor(1e-8))
    return (r / scale).square().mean(-1).flatten(1)


def fixed_supervision(truth):
    if truth.dtype != torch.float32 or truth.shape != (400, 201, 201):
        raise ValueError('Complete normalized FP32 pressure labels required')
    frame_energy = truth.square().mean((-2, -1))
    total = frame_energy.mean()
    if not bool(torch.isfinite(total)) or not bool(total > 0): raise ValueError('Positive target energy required')
    shares = frame_energy.reshape(4, 100).sum(-1) / frame_energy.sum()
    weights = .9 + .025 / shares.clamp_min(.001)
    return dict(energy=total.detach(), frame_weights=weights.repeat_interleave(100).detach(),
                window_shares=shares.cpu().tolist(), window_weights=weights.cpu().tolist(),
                failure_frame_rms=frame_energy.sqrt().detach())


def pressure_metrics(prediction, truth, supervision):
    """All error arithmetic FP32, including subtraction, square and reduction."""
    if prediction.shape != truth.shape or prediction.dtype != truth.dtype or truth.dtype != torch.float32:
        raise ValueError('Matched FP32 full fields required')
    error = (prediction - truth).square()
    energy = truth.square()
    def rel(a, b): return float(torch.sqrt(a.sum(dtype=torch.float32) / b.sum(dtype=torch.float32)))
    windows = [rel(error[j*100:(j+1)*100], energy[j*100:(j+1)*100]) for j in range(4)]
    threshold = (.1 * supervision['failure_frame_rms'])[:, None, None]
    fail = (prediction-truth).abs() > threshold
    # Complete uniform grid counts: independent of AIS training distribution.
    data_pf = float(fail.to(torch.float32).mean(dtype=torch.float32))
    return dict(full_relative_l2=rel(error, energy), future_relative_l2=rel(error[8:], energy[8:]),
                first8_relative_l2=rel(error[:8], energy[:8]), late_relative_l2=rel(error[270:], energy[270:]),
                window_relative_l2=windows, data_failure_probability=data_pf,
                data_failure_epsilon_in_frame_reference_rms=.1,
                top_max_abs=float(prediction[:, 0].abs().max()),
                receiver_relative_l2=rel(error[:, 2, [50, 100, 150]], energy[:, 2, [50, 100, 150]]))


@torch.no_grad()
def assess(model, c, f, truth, supervision, op, valid, guard=None):
    was = model.training; model.eval()
    try:
        prediction = model(c, f, model.full_times_s, normalized=True)
        metrics = pressure_metrics(prediction, truth, supervision)
        energies = []
        for start in range(0, 400, 16):
            if guard: guard()
            indices = torch.arange(start, min(start+16, 400), device=model.device)
            energies.append(dg_energy(model, c, f, indices, op).detach().cpu())
        energies = torch.cat(energies)
        values = energies[:, valid.cpu().flatten()]
        failure = stopping_summary(energies, valid.cpu().flatten(), epsilon=.1,
                                   failure_probability_tolerance=.01)
        metrics.update(physics_mean=float(values.mean(dtype=torch.float32)),
                       physics_failure_probability=failure['failure_probability'],
                       physics_failure_summary=failure,
                       physics_failure_epsilon=.1,
                       failure_probability_measure='uniform complete400 x valid DG elements; exact enumeration',
                       physics_grid_center_last_s=.9995,
                       CPML_states_supervised=False)
        return prediction.detach(), metrics, energies
    finally:
        model.train(was)


def backward(model, c, f, truth, supervision, op, valid, plan, receiver_weight=.01,
             physics_weight=.001, ic_weight=.01):
    """Full400 pressure/receiver every step, plus unbiased stratified AIS PDE."""
    with model.scopes():
        prediction = model(c, f, model.full_times_s, normalized=True)
        e2 = (prediction-truth).square()
        field = (e2 * supervision['frame_weights'][:, None, None]).mean(dtype=torch.float32) / supervision['energy']
        target_receiver = truth[:, 2, [50, 100, 150]]
        receiver = (prediction[:, 2, [50, 100, 150]]-target_receiver).square().mean(dtype=torch.float32)
        receiver = receiver / target_receiver.square().mean(dtype=torch.float32).detach()
        sup = field + receiver_weight*receiver
        sup.backward()
        metrics = pressure_metrics(prediction.detach(), truth, supervision)
        metrics.update(field_loss=float(field.detach()), receiver_loss=float(receiver.detach()))
        del prediction, e2, field, receiver, sup
        selected = plan['selected_times']
        energies = dg_energy(model, c, f, selected, op)
        weights = plan['weights'].index_select(0, selected).to(model.device)
        physics = (energies * weights).sum(dtype=torch.float32) / (400*int(valid.sum()))
        initial = model(c, f, torch.tensor([0., .0005, .001], device=model.device), normalized=True)
        derivative = (-3*initial[0]+4*initial[1]-initial[2])/.001
        ic = initial[0].square().mean()+(derivative/(2*torch.pi*f[0, 2])).square().mean()
        total_physics = physics_weight*physics+ic_weight*ic
        total_physics.backward()
        metrics.update(physics_loss=float(physics.detach()), ic_loss=float(ic.detach()),
                       loss=metrics['field_loss']+receiver_weight*metrics['receiver_loss']+float(total_physics.detach()),
                       data_time_count=400, data_spatial_nodes=201*201,
                       physics_unique_times=len(selected), physics_sampled_draws=128)
        return metrics, selected, energies.detach().cpu()
