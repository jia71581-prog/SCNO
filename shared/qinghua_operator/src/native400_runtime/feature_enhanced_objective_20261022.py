"""Positive-support, stratified HT estimator of the original 4 s objective.

Field and receiver terms estimate all saved times; the SIPG teacher-defect
term estimates every interior time center. The four strata are sampled
uniformly WITHOUT replacement, with explicit checkpointable inclusion
weights. Original TRAIN reference statistics and all objective coefficients
remain unchanged. IC3 and coefficient regularization are exact.

Only requested native-time labels need be resident on CPU. Decoder chunks
accumulate VJPs into the existing EncodingBridge leaves, including the
optional velocity-autoencoder auxiliary loss, before one encoder backward.
No final pressure interpolation, file I/O, RNG mutation, CUDA initialization
or optimizer work occurs on import. Reported relative errors are sampled
diagnostics, not unbiased relative-L2 estimates or full-panel evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from . import paper4s_training_objective_20261021 as base

ObjectiveConfig = base.ObjectiveConfig
EncodingBridge = base.EncodingBridge
prepare_reference = base.prepare_reference
stack_references = base.stack_references
time_axis = base.time_axis
SCHEMA = 'FEATURE_ENHANCED_STRATIFIED_HT_OBJECTIVE_V1'


def _windows(first, stop):
    """Integer intervals with the same boundaries as torch.tensor_split(.,4)."""
    size, extra = divmod(stop-first, 4)
    bounds, start = [], first
    for window in range(4):
        end = start+size+(window < extra)
        bounds.append((start, end))
        start = end
    return bounds


def _positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer at least {minimum}')


def make_plan(frames=801, field_per_window=12, physics_per_window=2, seed=0):
    """JSON plan for four independent uniform-without-replacement strata.

    Every item has positive inclusion probability. A requested count larger
    than its stratum means a census of that stratum. Weights already contain
    1/frames or 1/(frames-2), respectively; callers must NOT divide again.
    A private CPU generator leaves global training RNG untouched.
    """
    _positive_integer(frames, 'frames', 6)
    _positive_integer(field_per_window, 'field_per_window')
    _positive_integer(physics_per_window, 'physics_per_window')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be a nonnegative signed-64-bit integer')
    generator = torch.Generator(device='cpu').manual_seed(seed)

    def sample(bounds, count, denominator):
        indices, weights = [], []
        for lo, hi in bounds:
            size, take = hi-lo, min(count, hi-lo)
            picks = (torch.randperm(size, generator=generator)[:take]+lo).sort().values.tolist()
            indices.extend(picks)
            weights.extend([size/(take*denominator)]*take)
        return indices, weights

    indices, field_weights = sample(_windows(0, frames), field_per_window, frames)
    centers, physics_weights = sample(_windows(1, frames-1), physics_per_window, frames-2)
    return dict(schema=SCHEMA, frames=frames, seed=seed,
        field_indices=indices, field_weights=field_weights,
        physics_centers=centers, physics_weights=physics_weights)


def _validate_plan(plan, frames=None):
    if not isinstance(plan, dict) or plan.get('schema') != SCHEMA:
        raise ValueError('explicit stratified HT plan schema required')
    nt = plan.get('frames')
    _positive_integer(nt, 'plan frames', 6)
    if frames is not None and nt != frames:
        raise ValueError('plan and objective frame support differ')
    for index_key, weight_key, lo, hi, denominator in (
        ('field_indices', 'field_weights', 0, nt, nt),
        ('physics_centers', 'physics_weights', 1, nt-1, nt-2),
    ):
        indices, weights = plan.get(index_key), plan.get(weight_key)
        if (not isinstance(indices, list) or not isinstance(weights, list)
                or len(indices) != len(weights) or not indices
                or any(type(i) is not int or not lo <= i < hi for i in indices)
                or indices != sorted(set(indices))
                or any(type(w) not in (float, int) or not math.isfinite(w) or w <= 0 for w in weights)):
            raise ValueError('sorted unique supported indices and positive JSON HT weights required: '+index_key)
        for start, end in _windows(lo, hi):
            selected = [j for j, i in enumerate(indices) if start <= i < end]
            if not selected:
                raise ValueError('every time stratum needs positive sampling support: '+index_key)
            expected = (end-start)/(len(selected)*denominator)
            if any(not math.isclose(weights[j], expected, rel_tol=1e-12, abs_tol=1e-15) for j in selected):
                raise ValueError('HT weights must be N/k divided by complete support: '+weight_key)
    return nt


def required_frames(plan):
    """Sorted union of field labels, PDE three-point halos and exact IC3."""
    _validate_plan(plan)
    indices = set(plan['field_indices']) | {0, 1, 2}
    for center in plan['physics_centers']:
        indices.update((center-1, center, center+1))
    return sorted(indices)


@dataclass(frozen=True, eq=False)
class SampledLabels:
    """CPU physical FP32 labels indexed by their original native time slots."""
    indices: tuple[int, ...] | list[int]
    values: torch.Tensor
    _positions: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self):
        indices = tuple(self.indices)
        if (not indices or any(type(i) is not int or i < 0 for i in indices)
                or len(indices) != len(set(indices))):
            raise ValueError('unique nonnegative integer label frame indices required')
        if (not torch.is_tensor(self.values) or self.values.ndim != 4
                or self.values.dtype != torch.float32 or self.values.device.type != 'cpu'
                or self.values.shape[0] < 1 or self.values.shape[1] != len(indices)
                or min(self.values.shape[-2:]) < 1 or self.values.requires_grad):
            raise ValueError('constant CPU FP32 sampled labels with shape [B,S,Z,X] required')
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError('finite sampled physical pressure labels required')
        object.__setattr__(self, 'indices', indices)
        object.__setattr__(self, '_positions', {slot: i for i, slot in enumerate(indices)})

    def take(self, indices, device):
        try:
            positions = [self._positions[i] for i in indices]
        except KeyError as error:
            raise ValueError(f'missing sampled label frame {error.args[0]}') from error
        return self.values.index_select(1, torch.tensor(positions, dtype=torch.int64)).to(device)


def _inputs(c, source, labels, times, config, plan):
    _validate_plan(plan, config.frames)
    if c.ndim == 3:
        c = c[:, None]
    b, n = c.shape[0], config.grid_size
    if (c.shape != (b, 1, n, n) or source.shape != (b, 5) or b < 1
            or c.dtype != torch.float32 or source.dtype != torch.float32
            or source.device != c.device):
        raise ValueError('physical FP32 batched velocity/source on one device required')
    if not isinstance(labels, SampledLabels) or labels.values.shape != (b, len(labels.indices), n, n):
        raise ValueError('sampled labels must match velocity batch and native spatial grid')
    if any(i >= config.frames for i in labels.indices):
        raise ValueError('label frame index outside complete teacher support')
    missing = sorted(set(required_frames(plan))-set(labels.indices))
    if missing:
        raise ValueError('missing sampled label frames: '+str(missing))
    if bool(labels.values[:, labels._positions[0]].ne(0).any()):
        raise ValueError('actual zero initial teacher field required')
    times = torch.as_tensor(times, device=c.device, dtype=torch.float32)
    if not torch.equal(times, base.time_axis(config, c.device)):
        raise ValueError('exact uniform physical teacher times required')
    if not bool(torch.isfinite(c).all() & (c > 0).all()):
        raise ValueError('finite positive velocity required')
    if not bool(torch.isfinite(source).all() & (source[:, 2] > 0).all() & (source[:, 4] > 0).all()):
        raise ValueError('finite source with positive frequency/amplitude required')
    return c, source, labels, times


def run_objective(model, c, source, labels, times, reference, *, config=ObjectiveConfig(),
                  plan, chunk_size=4, backward=True, auxiliary_weight=.001, task_weight=1.):
    """Estimate the original batch mean with exact HT gradient weights.

    Model API is the existing encode/query/coefficient_regularization API,
    plus optional auxiliary_loss(query_encoding)->scalar. The caller owns
    zero_grad and every optimizer, sampling-cursor and checkpoint operation.
    Sampling plans must be reused between comparison arms / pre-post probes.
    ``backward=False`` preserves existing gradients. Auxiliary weight is
    independent of the unchanged coefficient regularizer.
    """
    _positive_integer(chunk_size, 'chunk_size')
    if (type(backward) is not bool or not math.isfinite(task_weight) or task_weight <= 0
            or not math.isfinite(auxiliary_weight) or auxiliary_weight < 0):
        raise ValueError('boolean backward, positive task weight and nonnegative auxiliary weight required')
    c, source, labels, times = _inputs(c, source, labels, times, config, plan)
    ref = base._reference(reference, c, source, times, config)
    op, valid = base._geometry(c, source, config)
    if not torch.equal(valid, ref['valid_mask']):
        raise ValueError('cached source-free spatial mask differs')
    batch, n, nt = len(c), config.grid_size, config.frames
    if ref['frame_weights'].shape != (batch, nt):
        raise ValueError('complete TRAIN frame weights required')
    totals = {key: c.new_zeros(()) for key in
              ('field_loss', 'receiver_loss', 'physics_loss', 'ic_loss', 'coefficient_loss', 'auxiliary_loss')}
    errors, receiver_errors = c.new_zeros(batch), c.new_zeros(batch)
    late_errors, late_targets = c.new_zeros(batch), c.new_zeros(batch)
    maximum = c.new_zeros(())
    field_indices, centers = plan['field_indices'], plan['physics_centers']
    field_weights = c.new_tensor(plan['field_weights'])
    physics_weights = c.new_tensor(plan['physics_weights'])
    valid_counts = valid.flatten(1).sum(1)

    def add_backward(loss):
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite sampled training objective term')
        if backward and loss.requires_grad:
            (task_weight*loss).backward()

    with torch.set_grad_enabled(backward):
        encoded = model.encode(c, source)
        bridge = EncodingBridge(encoded) if backward else None
        query_encoding = bridge.encoded if bridge else encoded

        def error_at(indices):
            nonlocal maximum
            slots = torch.tensor(indices, device=c.device, dtype=torch.int64)
            prediction = model.query(query_encoding, times.index_select(0, slots), normalized=False)
            if (prediction.shape != (batch, len(indices), n, n)
                    or prediction.dtype != torch.float32 or prediction.device != c.device):
                raise ValueError('query must return physical FP32 [B,T,Z,X] on input device')
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError('nonfinite sampled physical prediction')
            maximum = torch.maximum(maximum, prediction.detach().abs().max())
            target = labels.take(indices, c.device)
            return (prediction-target)/config.pressure_rms, target

        for start in range(0, len(field_indices), chunk_size):
            indices = field_indices[start:start+chunk_size]
            slots = torch.tensor(indices, device=c.device, dtype=torch.int64)
            weights = field_weights[start:start+len(indices)]
            error, target = error_at(indices)
            weighted = weights[None, :, None, None]
            field_loss = (error.square()*ref['frame_weights'].index_select(1, slots)[:, :, None, None]
                          *weighted).sum()/(batch*n*n)
            receiver_error = error[:, :, config.receiver_row]*(config.pressure_rms/config.receiver_rms)
            receiver_loss = (receiver_error.square()*weights[None, :, None]).sum()/(batch*n)
            add_backward(field_loss+config.receiver_weight*receiver_loss)
            totals['field_loss'] += field_loss.detach()
            totals['receiver_loss'] += receiver_loss.detach()
            with torch.no_grad():
                errors += (error.square()*weighted).sum((1, 2, 3))
                receiver_errors += (receiver_error.square()*weights[None, :, None]).sum((1, 2))
                late = slots >= 3*nt//4
                late_errors += (error[:, late].square()*weighted[:, late]).sum((1, 2, 3))
                late_targets += ((target[:, late]/config.pressure_rms).square()*weighted[:, late]).sum((1, 2, 3))

        if config.physics_weight:
            for start in range(0, len(centers), chunk_size):
                selected = centers[start:start+chunk_size]
                indices = sorted({slot for center in selected for slot in (center-1, center, center+1)})
                error, _ = error_at(indices)
                positions = {slot: i for i, slot in enumerate(indices)}
                left = [positions[center-1] for center in selected]
                middle = [positions[center] for center in selected]
                right = [positions[center+1] for center in selected]
                weights = physics_weights[start:start+len(selected)]
                physics_loss = c.new_zeros(())
                for record in range(batch):
                    components = op.components(error[record, left], error[record, middle], error[record, right],
                        c[record, 0], config.dt_s, config.penalty)
                    residual = sum(components.values())[:, valid[record], :]
                    physics_loss = physics_loss+(residual.square()*weights[:, None, None]).sum()/(
                        batch*valid_counts[record]*op.nb*ref['physics_scale2'][record])
                add_backward(config.physics_weight*physics_loss)
                totals['physics_loss'] += physics_loss.detach()

        initial_error, _ = error_at([0, 1, 2])
        derivative = (-3*initial_error[:, 0]+4*initial_error[:, 1]-initial_error[:, 2])/(2*config.dt_s)
        initial = (initial_error[:, 0].square().mean()+
            (derivative/(2*torch.pi*source[:, 2, None, None])).square().mean())
        add_backward(config.ic_weight*initial)
        totals['ic_loss'] += initial.detach()

        for method, key, weight in (('coefficient_regularization', 'coefficient_loss', 1.),
                                    ('auxiliary_loss', 'auxiliary_loss', auxiliary_weight)):
            penalty = getattr(model, method)(query_encoding) if hasattr(model, method) else c.new_zeros(())
            if not torch.is_tensor(penalty) or penalty.ndim != 0 or not bool(torch.isfinite(penalty)):
                raise ValueError('finite scalar '+method+' required')
            add_backward(weight*penalty)
            totals[key] += penalty.detach()
        if bridge:
            bridge.backward()

    loss = (totals['field_loss']+config.receiver_weight*totals['receiver_loss']+
        config.physics_weight*totals['physics_loss']+config.ic_weight*totals['ic_loss']+
        totals['coefficient_loss']+auxiliary_weight*totals['auxiliary_loss'])

    def relative(numerator, denominator):
        return torch.sqrt(numerator/denominator.clamp_min(torch.finfo(torch.float32).tiny)).cpu().tolist()

    result = {key: float(value) for key, value in totals.items()}
    result.update(loss=float(loss), weighted_loss=task_weight*float(loss), batch_size=batch,
        auxiliary_weight=auxiliary_weight, auxiliary_weighted_loss=auxiliary_weight*result['auxiliary_loss'],
        relative_l2_per_record=relative(errors, ref['field_energy']*n*n),
        late_relative_l2_per_record=relative(late_errors, late_targets),
        raw_receiver_relative_l2_per_record=relative(receiver_errors, ref['receiver_energy']*n),
        full_relative_l2=float(torch.sqrt(errors.sum()/(ref['field_energy'].sum()*n*n))),
        teacher_defect_normalized_mean=float((ref['teacher_defect_mean_square']/ref['physics_scale2']).mean()),
        maximum_absolute_physical_pressure=float(maximum), field_times=len(field_indices),
        field_time_support=nt, physics_centers=len(centers), physics_time_support=nt-2,
        sampled_label_frames=len(labels.indices), metrics_are_sample_estimates=True,
        estimator='four-stratum uniform-without-replacement Horvitz-Thompson batch mean',
        receiver_kind='raw native row; NOT fine-grid B5',
        physics_kind='fixed TRAIN teacher-defect consistency; NOT zero teacher residual',
        ic_kind='exact IC3: zero initial pressure and teacher-matched saved-dt one-sided derivative')
    return result


__all__ = ['SCHEMA', 'ObjectiveConfig', 'time_axis', 'prepare_reference', 'stack_references',
           'make_plan', 'required_frames', 'SampledLabels', 'run_objective']
