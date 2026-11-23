"""TRAIN-only 4 s field objective with a fixed teacher-defect SIPG target.

The receiver term is raw native201 row 2, NOT the fine401 B5 measurement.
R_h(prediction)-R_h(TRAIN truth) is evaluated with the same coarse SIPG
operator and actual saved-time spacing. It is a discrete consistency penalty,
not a claim that this operator equals the fine-grid Deepwave teacher.

No file, TEST, QC, optimizer or CUDA work occurs on import. Reference statistics
are explicit checkpointable inputs, computed once from TRAIN records. A leaf
VJP bridge lets time chunks release their decoder graphs while encoding once.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import math

import numpy as np
import torch

from paso.dg_weak import get_dg_operator

SCHEMA = 'PAPER4S_TRAIN_TEACHER_DEFECT_OBJECTIVE_V1'


@dataclass(frozen=True)
class ObjectiveConfig:
    frames: int = 801
    grid_size: int = 201
    dt_s: float = .005
    spacing_m: float = 10.
    pressure_rms: float = 9.09277469201749e-10
    receiver_rms: float = 8.68226568436776e-10
    receiver_row: int = 2
    receiver_weight: float = .01
    physics_weight: float = .001
    ic_weight: float = .01
    window_fraction: float = .1
    window_share_floor: float = .001
    physics_relative_floor: float = 1e-6
    physics_absolute_floor: float = 1e-12
    element_cells: int = 4
    dg_order: int = 3
    quadrature_order: int = 4
    penalty: float = 2.
    source_exclusion_cells: int = 4

    def __post_init__(self):
        for name in ('frames', 'grid_size', 'element_cells', 'dg_order', 'quadrature_order'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('positive integer '+name+' required')
        if self.frames < 4 or (self.grid_size-1) % self.element_cells:
            raise ValueError('at least four times and divisible DG spatial grid required')
        if not 0 <= self.receiver_row < self.grid_size or self.source_exclusion_cells < 0:
            raise ValueError('receiver row/source exclusion invalid')
        for name in ('dt_s', 'spacing_m', 'pressure_rms', 'receiver_rms',
                     'window_share_floor', 'physics_relative_floor', 'physics_absolute_floor', 'penalty'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('positive finite '+name+' required')
        for name in ('receiver_weight', 'physics_weight', 'ic_weight'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError('nonnegative finite '+name+' required')
        if not 0 <= self.window_fraction <= 1:
            raise ValueError('window fraction outside [0,1]')


def time_axis(config=ObjectiveConfig(), device='cpu'):
    """Teacher FP64 multiplication followed by one FP32 cast, on CPU."""
    return torch.from_numpy(np.asarray(np.arange(config.frames, dtype=np.float64)*config.dt_s,
                                       dtype=np.float32)).to(device)


def _sha(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((value.dtype, value.shape)).encode()+value.tobytes()).hexdigest()


def _inputs(c, source, truth_pa, times, config):
    if c.ndim == 3:
        c = c[:, None]
    b, n, nt = c.shape[0], config.grid_size, config.frames
    if (c.shape != (b, 1, n, n) or source.shape != (b, 5)
            or truth_pa.shape != (b, nt, n, n) or b < 1):
        raise ValueError('batched c/source/complete native field shape differs')
    if any(x.dtype != torch.float32 for x in (c, source, truth_pa)):
        raise ValueError('FP32 velocity, source and physical pressure required')
    if any(x.device != c.device for x in (source, truth_pa)):
        raise ValueError('all numerical inputs must be on one device')
    times = torch.as_tensor(times, device=c.device, dtype=torch.float32)
    if not torch.equal(times, time_axis(config, c.device)):
        raise ValueError('exact uniform physical teacher times required')
    if not bool(torch.isfinite(c).all()) or not bool((c > 0).all()):
        raise ValueError('finite positive velocity required')
    if not bool(torch.isfinite(source).all()) or not bool((source[:, 2] > 0).all() & (source[:, 4] > 0).all()):
        raise ValueError('finite source with positive frequency/amplitude required')
    return c, source, truth_pa, times


def _geometry(c, source, config):
    op = get_dg_operator(config.grid_size, config.grid_size, config.spacing_m, config.spacing_m,
                         config.element_cells, config.dg_order, config.quadrature_order,
                         c.device, torch.float32)
    valid = torch.stack([op.valid_mask(f[:2], 0., 0., config.source_exclusion_cells) for f in source])
    if not bool(valid.flatten(1).any(1).all()):
        raise ValueError('nonempty source-free interior required for every record')
    return op, valid


@torch.no_grad()
def prepare_reference(c, source, truth_pa, times, config=ObjectiveConfig(), *, chunk_size=16):
    """Compute immutable per-TRAIN-record scales; returned tensors are on CPU.

    physics_scale2 = mean(teacher component squared) + relative_floor *
    mean(normalized field squared) + absolute_floor. All terms are fixed before
    optimization. The reported teacher defect is deliberately not set to zero.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('positive integer reference chunk size required')
    c, source, truth_pa, times = _inputs(c, source, truth_pa, times, config)
    if not bool(torch.isfinite(truth_pa).all()) or bool(truth_pa[:, 0].ne(0).any()):
        raise ValueError('finite TRAIN fields with actual zero initial field required')
    op, valid = _geometry(c, source, config)
    frame_energy = (truth_pa/config.pressure_rms).square().mean((-2, -1))
    energy = frame_energy.mean(1)
    receiver_energy = (truth_pa[:, :, config.receiver_row]/config.receiver_rms).square().mean((1, 2))
    if not bool(torch.isfinite(energy).all() & torch.isfinite(receiver_energy).all()
                & (energy > 0).all() & (receiver_energy > 0).all()):
        raise ValueError('positive TRAIN field and raw receiver energy required')
    frame_weights = torch.empty_like(frame_energy)
    for window in torch.tensor_split(torch.arange(config.frames, device=c.device), 4):
        share = frame_energy[:, window].sum(1)/frame_energy.sum(1)
        frame_weights[:, window] = (1-config.window_fraction +
            config.window_fraction/4/share.clamp_min(config.window_share_floor))[:, None]
    component_sums = c.new_zeros((len(c), 5))
    defect_sums = c.new_zeros(len(c))
    for start in range(1, config.frames-1, chunk_size):
        end = min(start+chunk_size, config.frames-1)
        target = truth_pa[:, start-1:end+1]/config.pressure_rms
        for i in range(len(c)):
            comp = op.components(target[i, :-2], target[i, 1:-1], target[i, 2:], c[i, 0],
                                 config.dt_s, config.penalty)
            pieces = torch.stack(list(comp.values()))[:, :, valid[i], :]
            component_sums[i] += pieces.square().sum((1, 2, 3))
            defect_sums[i] += pieces.sum(0).square().sum()
    denominator = (config.frames-2)*valid.flatten(1).sum(1)*op.nb
    component_energy = component_sums/denominator[:, None]
    scale2 = component_energy.mean(1)+config.physics_relative_floor*energy+config.physics_absolute_floor
    if not bool(torch.isfinite(scale2).all() & torch.isfinite(defect_sums).all()):
        raise FloatingPointError('nonfinite TRAIN teacher-defect statistics')
    return dict(schema=SCHEMA, config=asdict(config), time_sha256=_sha(times),
        velocity_sha256=[_sha(v) for v in c], source=source.detach().cpu().clone(),
        field_shape=list(truth_pa.shape[1:]), frame_weights=frame_weights.cpu(),
        field_energy=energy.cpu(), receiver_energy=receiver_energy.cpu(), valid_mask=valid.cpu(),
        teacher_component_energy=component_energy.cpu(), physics_scale2=scale2.cpu(),
        teacher_defect_mean_square=(defect_sums/denominator).cpu(),
        pressure_target='raw native field', receiver_target='raw native row; NOT fine-grid B5',
        physics_target='coarse SIPG R_h(pred)-R_h(TRAIN truth), actual saved dt')


def stack_references(references):
    """Combine single-record cached references in exact minibatch order."""
    if not references:
        raise ValueError('nonempty reference list required')
    first = references[0]
    output = {}
    for key, value in first.items():
        values = [r[key] for r in references]
        if torch.is_tensor(value):
            output[key] = torch.cat(values, dim=0)
        elif key == 'velocity_sha256':
            output[key] = [item for group in values for item in group]
        else:
            if any(item != value for item in values):
                raise ValueError('reference protocols differ: '+key)
            output[key] = copy.deepcopy(value)
    return output


def _reference(reference, c, source, times, config):
    if (reference['schema'] != SCHEMA or reference['config'] != asdict(config)
            or reference['time_sha256'] != _sha(times)
            or reference['velocity_sha256'] != [_sha(v) for v in c]
            or not torch.equal(reference['source'], source.detach().cpu())):
        raise ValueError('fixed TRAIN reference binding differs')
    out = {k: v.to(c.device) if torch.is_tensor(v) else v for k, v in reference.items()}
    for key in ('frame_weights', 'physics_scale2', 'field_energy', 'receiver_energy'):
        if not bool(torch.isfinite(out[key]).all()) or not bool((out[key] > 0).all()):
            raise ValueError('invalid reference statistic: '+key)
    return out


def uniform_physics_plan(config, count, generator):
    """Without-replacement time sampling; weights estimate the complete mean.

    Caller owns and checkpoints the CPU generator and reuses this exact plan in
    the pre/post objectives. This helper is uniform sampling, not residual AIS.
    """
    n = config.frames-2
    if type(count) is not int or not 1 <= count <= n:
        raise ValueError('physics count outside internal time support')
    centers = torch.randperm(n, generator=generator)[:count].sort().values+1
    return dict(centers=centers, weights=torch.full((count,), n/count, dtype=torch.float32))


def _physics_plan(plan, config, device):
    if plan is None:
        return torch.arange(1, config.frames-1, device=device), torch.ones(config.frames-2, device=device)
    centers = torch.as_tensor(plan['centers'], device=device)
    weights = torch.as_tensor(plan['weights'], device=device)
    if (centers.dtype != torch.int64 or centers.ndim != 1 or centers.numel() == 0
            or weights.shape != centers.shape or weights.dtype != torch.float32
            or not bool((centers >= 1).all() & (centers < config.frames-1).all())
            or centers.unique().numel() != centers.numel()
            or not bool(torch.isfinite(weights).all() & (weights > 0).all())):
        raise ValueError('unique interior physics centers and positive FP32 weights required')
    return centers, weights


class EncodingBridge:
    """Detach only encoded tensors; return their accumulated VJP once at end."""
    def __init__(self, encoded):
        self.pairs, memo = [], {}
        def clone(value):
            if torch.is_tensor(value):
                if id(value) not in memo:
                    leaf = value.detach().requires_grad_(value.requires_grad)
                    memo[id(value)] = leaf
                    if value.requires_grad:
                        self.pairs.append((value, leaf))
                return memo[id(value)]
            if is_dataclass(value) and not isinstance(value, type):
                result = copy.copy(value)
                for field in fields(value):
                    object.__setattr__(result, field.name, clone(getattr(value, field.name)))
                return result
            if isinstance(value, dict):
                return {k: clone(v) for k, v in value.items()}
            if isinstance(value, tuple):
                values = [clone(v) for v in value]
                return type(value)(*values) if hasattr(value, '_fields') else tuple(values)
            if isinstance(value, list):
                return [clone(v) for v in value]
            return value
        self.encoded = clone(encoded)
        self.finished = False

    def backward(self):
        if self.finished:
            raise ValueError('encoding VJP already consumed')
        connected = [(original, leaf.grad) for original, leaf in self.pairs if leaf.grad is not None]
        if connected:
            torch.autograd.backward([v for v, _ in connected], [g for _, g in connected])
        self.finished = True


def run_objective(model, c, source, truth_pa, times, reference, *, config=ObjectiveConfig(),
                  chunk_size=16, physics_plan=None, backward=True, task_weight=1.):
    """Accumulate a batch-mean gradient, or evaluate without changing gradients.

    Model API: encode(c,source), query(encoded,times,normalized=False) -> B,T,Z,X,
    and optional coefficient_regularization(encoded). Caller owns zero_grad,
    clipping, optimizer/controller updates, curriculum and checkpointing.
    All field times are supervised once. Halo queries only supply PDE stencils.
    """
    if type(chunk_size) is not int or chunk_size < 1 or not math.isfinite(task_weight) or task_weight <= 0:
        raise ValueError('positive chunk size and task weight required')
    c, source, truth_pa, times = _inputs(c, source, truth_pa, times, config)
    ref = _reference(reference, c, source, times, config)
    op, valid = _geometry(c, source, config)
    if not torch.equal(valid, ref['valid_mask']):
        raise ValueError('cached source-free spatial mask differs')
    centers, sample_weights = _physics_plan(physics_plan, config, c.device)
    batch, n, nt = len(c), config.grid_size, config.frames
    valid_counts = valid.flatten(1).sum(1)
    totals = {key: c.new_zeros(()) for key in ('field_loss', 'receiver_loss', 'physics_loss', 'ic_loss', 'coefficient_loss')}
    errors = c.new_zeros(batch)
    late_errors, late_targets = c.new_zeros(batch), c.new_zeros(batch)
    receiver_errors = c.new_zeros(batch)
    maximum = c.new_zeros(())
    with torch.set_grad_enabled(backward):
        encoded = model.encode(c, source)
        bridge = EncodingBridge(encoded) if backward else None
        query_encoding = bridge.encoded if bridge else encoded
        for start in range(0, nt, chunk_size):
            end = min(start+chunk_size, nt)
            lo, hi = max(0, start-1), min(nt, end+1)
            if start == 0:
                hi = max(hi, 3)
            prediction_pa = model.query(query_encoding, times[lo:hi], normalized=False)
            if prediction_pa.shape != (batch, hi-lo, n, n) or prediction_pa.dtype != torch.float32:
                raise ValueError('query must return physical FP32 [B,T,Z,X]')
            if not bool(torch.isfinite(prediction_pa).all()):
                raise FloatingPointError('nonfinite physical prediction')
            error = (prediction_pa-truth_pa[:, lo:hi])/config.pressure_rms
            central = error[:, start-lo:end-lo]
            field = (central.square()*ref['frame_weights'][:, start:end, None, None]).sum()/(batch*nt*n*n)
            receiver_error = central[:, :, config.receiver_row]*(config.pressure_rms/config.receiver_rms)
            receiver = receiver_error.square().sum()/(batch*nt*n)
            physics, initial = c.new_zeros(()), c.new_zeros(())
            selection = (centers >= max(1, start)) & (centers < min(nt-1, end))
            selected = centers[selection]-lo
            if config.physics_weight and selected.numel():
                for i in range(batch):
                    comp = op.components(error[i, selected-1], error[i, selected], error[i, selected+1],
                                         c[i, 0], config.dt_s, config.penalty)
                    residual = sum(comp.values())[:, valid[i], :]
                    physics = physics + (residual.square()*sample_weights[selection, None, None]).sum()/(
                        batch*(nt-2)*valid_counts[i]*op.nb*ref['physics_scale2'][i])
            if start == 0:
                derivative = (-3*error[:, 0]+4*error[:, 1]-error[:, 2])/(2*config.dt_s)
                initial = error[:, 0].square().mean()+(derivative/(2*torch.pi*source[:, 2, None, None])).square().mean()
            loss = field+config.receiver_weight*receiver+config.physics_weight*physics+config.ic_weight*initial
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite training objective chunk')
            if backward:
                (task_weight*loss).backward()
            for key, value in (('field_loss', field), ('receiver_loss', receiver), ('physics_loss', physics), ('ic_loss', initial)):
                totals[key] += value.detach()
            with torch.no_grad():
                errors += central.square().sum((1, 2, 3))
                receiver_errors += receiver_error.square().sum((1, 2))
                late_start = max(start, 3*nt//4)
                if late_start < end:
                    late_errors += error[:, late_start-lo:end-lo].square().sum((1, 2, 3))
                    late_targets += (truth_pa[:, late_start:end]/config.pressure_rms).square().sum((1, 2, 3))
                maximum = torch.maximum(maximum, prediction_pa.detach().abs().max())
        penalty = (model.coefficient_regularization(query_encoding)
                   if hasattr(model, 'coefficient_regularization') else c.new_zeros(()))
        if penalty.ndim != 0 or not bool(torch.isfinite(penalty)):
            raise ValueError('finite scalar coefficient regularization required')
        if backward and penalty.requires_grad:
            (task_weight*penalty).backward()
        totals['coefficient_loss'] += penalty.detach()
        if bridge:
            bridge.backward()
    loss = (totals['field_loss']+config.receiver_weight*totals['receiver_loss']+
            config.physics_weight*totals['physics_loss']+config.ic_weight*totals['ic_loss']+totals['coefficient_loss'])
    def relative(numerator, denominator):
        return torch.sqrt(numerator/denominator.clamp_min(torch.finfo(torch.float32).tiny)).cpu().tolist()
    values = {k: float(v) for k, v in totals.items()}
    values.update(loss=float(loss), weighted_loss=task_weight*float(loss), batch_size=batch,
        relative_l2_per_record=relative(errors, ref['field_energy']*nt*n*n),
        late_relative_l2_per_record=relative(late_errors, late_targets),
        raw_receiver_relative_l2_per_record=relative(receiver_errors, ref['receiver_energy']*nt*n),
        full_relative_l2=float(torch.sqrt(errors.sum()/(ref['field_energy'].sum()*nt*n*n))),
        teacher_defect_normalized_mean=float((ref['teacher_defect_mean_square']/ref['physics_scale2']).mean()),
        maximum_absolute_physical_pressure=float(maximum), field_times=nt,
        physics_centers=int(centers.numel()), receiver_kind='raw native row; NOT fine-grid B5',
        physics_kind='fixed TRAIN teacher-defect consistency; NOT zero teacher residual',
        ic_kind='zero initial pressure and teacher-matched saved-dt one-sided derivative')
    return values


__all__ = ['ObjectiveConfig', 'time_axis', 'prepare_reference', 'stack_references',
           'uniform_physics_plan', 'EncodingBridge', 'run_objective']
