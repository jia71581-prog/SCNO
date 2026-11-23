"""FP32 joint residual AIS for the paper's 400-time, 50x50 DG elements.

DATA supervision is external and can cover the complete wavefield each step.
This sampler only changes the proposal for a fixed, uniform-in-time/element
physics objective. Each of four 100-slot windows draws two times and sixteen
elements per time, with replacement. The joint proposal is 25% uniform plus
75% normalized sqrt(previous committed residual-energy EMA).

``plan`` uses a private copy of the committed CPU generator. Only ``commit``
advances its generator, EMA and version; callers must commit after a successful
optimizer update. Uncommitted plans are intentionally absent from checkpoints.
All spatial residuals at every computed selected time enter the next EMA, once
per distinct time, irrespective of sampling multiplicities. Spatial decoding
and residual calculation remain dense at those times.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import torch

SCHEMA = 'PAPER_MARMOUSI_JOINT_DG_AIS_FP32_V1_20261020'
PLAN_SCHEMA = SCHEMA + '_PLAN'
STOPPING_SCHEMA = SCHEMA + '_UNIFORM_FULL_GRID_FAILURE_STATISTIC'
N_TIMES = 400
N_ELEMENTS = 2500
WINDOWS = ((0, 100), (100, 200), (200, 300), (300, 400))
TIME_DRAWS = 2
ELEMENT_DRAWS = 16
UNIFORM_MIX = .25
EMA_BETA = .9
CONFIG = dict(times=N_TIMES, elements=N_ELEMENTS,
    windows=[list(w) for w in WINDOWS], time_draws_per_window=TIME_DRAWS,
    element_draws_per_time=ELEMENT_DRAWS, replacement=True,
    uniform_mix=UNIFORM_MIX, ema_beta=EMA_BETA,
    proposal='joint .25 uniform + .75 sqrt(EMA)/sum(sqrt(EMA))',
    first_observation='initialize_to_measured_energy', residual_floor=None,
    arithmetic='FP32', feedback='all valid elements at unique computed times',
    loss_denominator='400 * number_of_valid_DG_elements',
    importance='multiplicity / (32 * actual_window_joint_q)',
    generator='private committed CPU torch.Generator')


def _require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def _integer(value, name):
    _require(type(value) is int and 0 <= value < 2**63, name + ' must be a nonnegative int64')
    return value


def _tensor(value, dtype, shape, name):
    _require(isinstance(value, torch.Tensor), name + ' must be a tensor')
    _require(value.device.type == 'cpu' and value.dtype == dtype
        and tuple(value.shape) == tuple(shape) and not value.requires_grad,
        name + ' must be detached CPU ' + str(dtype) + ' ' + str(tuple(shape)))
    return value


def _valid_mask(value):
    _require(isinstance(value, torch.Tensor), 'valid_mask must be a tensor')
    _require(tuple(value.shape) in ((N_ELEMENTS,), (50, 50)),
        'fixed 2500-element or 50x50 valid mask required')
    value = value.reshape(N_ELEMENTS)
    _tensor(value, torch.bool, (N_ELEMENTS,), 'valid_mask')
    _require(value.any(), 'at least one valid DG element required')
    return value.detach().clone()


def _digest(value):
    """Canonical digest, including exact CPU tensor bytes and dtypes."""
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            _require(item.device.type == 'cpu' and not item.requires_grad,
                'only detached CPU tensors can be fingerprinted')
            item = item.contiguous()
            h.update(str(item.dtype).encode())
            h.update(str(tuple(item.shape)).encode())
            h.update(item.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                h.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(str(len(item)).encode())
            for member in item:
                visit(member)
        else:
            h.update(json.dumps(item, sort_keys=True, separators=(',', ':'),
                allow_nan=False).encode())

    visit(value)
    return h.hexdigest()


@torch.no_grad()
def joint_proposal32(energy, valid_mask):
    """Return window joint q, time marginal qt and element conditional qe.

    This pure helper accepts small shapes for finite-sum checks. Production
    shapes are enforced by PaperMarmousiJointAIS. q is the product of the
    normalized FP32 vectors passed to the two categorical draws, so inverse
    weights use the same hierarchical probabilities. FP32 roundoff remains.
    """
    _require(isinstance(energy, torch.Tensor) and energy.ndim == 2
        and min(energy.shape) > 0, 'nonempty energy[time,element] required')
    _tensor(energy, torch.float32, energy.shape, 'energy')
    _tensor(valid_mask, torch.bool, (energy.shape[1],), 'window valid_mask')
    _require(valid_mask.any(), 'window needs valid elements')
    _require(torch.isfinite(energy).all() and (energy >= 0).all(),
        'finite nonnegative residual energies required')
    valid = valid_mask[None].expand_as(energy)
    uniform = valid.to(torch.float32) / float(energy.shape[0] * int(valid_mask.sum()))
    measured = energy.masked_fill(~valid, 0.)
    maximum = measured.max()
    if bool(maximum > 0):
        # Scaling before sqrt keeps summation finite even for FP32 max energy.
        mass = (measured / maximum).sqrt()
        adaptive = mass / mass.sum(dtype=torch.float32)
        joint = UNIFORM_MIX * uniform + (1. - UNIFORM_MIX) * adaptive
    else:
        joint = uniform
    joint = joint / joint.sum(dtype=torch.float32)
    qt = joint.sum(1, dtype=torch.float32)
    qt = qt / qt.sum(dtype=torch.float32)
    qe = joint / qt[:, None]
    qe = qe / qe.sum(1, keepdim=True, dtype=torch.float32)
    q = qt[:, None] * qe
    _require(torch.isfinite(q).all() and (q[:, valid_mask] > 0).all()
        and not q[:, ~valid_mask].any(), 'joint proposal lost valid support')
    return q, qt, qe


@torch.no_grad()
def draw_window32(energy, valid_mask, generator):
    """Actual two-time/sixteen-element hierarchical categorical draw."""
    _require(isinstance(generator, torch.Generator)
        and generator.device.type == 'cpu', 'private CPU generator required')
    q, qt, qe = joint_proposal32(energy, valid_mask)
    time_draws = torch.multinomial(qt, TIME_DRAWS, replacement=True, generator=generator)
    counts = torch.zeros(energy.shape, dtype=torch.long)
    elements, time_q, element_q, joint_q = [], [], [], []
    for t in time_draws.tolist():
        e = torch.multinomial(qe[t], ELEMENT_DRAWS, replacement=True, generator=generator)
        counts[t] += torch.bincount(e, minlength=energy.shape[1])
        elements.append(e)
        time_q.append(qt[t].expand(ELEMENT_DRAWS))
        element_q.append(qe[t, e])
        joint_q.append(q[t, e])
    weights = torch.zeros_like(energy)
    weights[:, valid_mask] = counts[:, valid_mask].to(torch.float32) / (
        float(TIME_DRAWS * ELEMENT_DRAWS) * q[:, valid_mask])
    _require(torch.isfinite(weights).all(), 'nonfinite inverse probability weights')
    return dict(q=q, qt=qt, qe=qe, time_draws=time_draws,
        element_draws=torch.stack(elements), counts=counts, weights=weights,
        draw_time_q=torch.cat(time_q), draw_element_q=torch.cat(element_q),
        draw_q=torch.cat(joint_q))


class PaperMarmousiJointAIS:
    """One independent source/medium instance and one committed RNG stream."""

    def __init__(self, valid_mask, seed=2026102001, *, binding=None):
        self.valid_mask = _valid_mask(valid_mask)
        self.seed = _integer(seed, 'seed')
        # Bind c/f/normalizer/checkpoint identifiers when supplied by the caller.
        self.binding = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
        self.commits = 0
        self.ema = torch.zeros((N_TIMES, N_ELEMENTS), dtype=torch.float32)
        self.seen = torch.zeros((N_TIMES, N_ELEMENTS), dtype=torch.bool)
        self.count = torch.zeros((N_TIMES, N_ELEMENTS), dtype=torch.long)
        self._generator = torch.Generator(device='cpu').manual_seed(self.seed)
        self._pending = None

    @property
    def n_valid(self):
        return int(self.valid_mask.sum())

    def state_dict(self):
        return dict(schema=SCHEMA, config=copy.deepcopy(CONFIG), seed=self.seed,
            binding=copy.deepcopy(self.binding), commits=self.commits,
            valid_mask=self.valid_mask.clone(), ema=self.ema.clone(),
            seen=self.seen.clone(), count=self.count.clone(),
            generator_state=self._generator.get_state().clone())

    def fingerprint(self):
        return _digest(self.state_dict())

    def load_state_dict(self, state):
        expected = {'schema', 'config', 'seed', 'binding', 'commits', 'valid_mask',
            'ema', 'seen', 'count', 'generator_state'}
        _require(isinstance(state, dict) and set(state) == expected, 'strict AIS state keys required')
        _require(state['schema'] == SCHEMA and state['config'] == CONFIG,
            'AIS schema/config changed')
        _require(type(state['seed']) is int and state['seed'] == self.seed
            and state['binding'] == self.binding, 'AIS seed/instance binding changed')
        mask = _valid_mask(state['valid_mask'])
        _require(torch.equal(mask, self.valid_mask), 'fixed valid DG mask changed')
        commits = _integer(state['commits'], 'commits')
        shape = (N_TIMES, N_ELEMENTS)
        ema = _tensor(state['ema'], torch.float32, shape, 'EMA').clone()
        seen = _tensor(state['seen'], torch.bool, shape, 'seen').clone()
        count = _tensor(state['count'], torch.long, shape, 'count').clone()
        _require(torch.isfinite(ema).all() and (ema >= 0).all(), 'invalid EMA energy')
        _require((count >= 0).all() and (count <= commits).all()
            and torch.equal(seen, count > 0), 'invalid committed observation counts')
        _require(not ema[~seen].any() and not seen[:, ~mask].any()
            and not count[:, ~mask].any(), 'unseen/invalid elements must retain zero history')
        # Full spatial feedback means every valid element at a time has one age.
        row_count = count[:, mask]
        _require((row_count == row_count[:, :1]).all(), 'partial-element feedback history rejected')
        gs = state['generator_state']
        _require(isinstance(gs, torch.Tensor) and gs.ndim == 1,
            'one-dimensional CPU generator state required')
        _tensor(gs, torch.uint8, gs.shape, 'generator_state')
        generator = torch.Generator(device='cpu')
        generator.set_state(gs.clone())
        if commits == 0:
            initial = torch.Generator(device='cpu').manual_seed(self.seed).get_state()
            _require(not seen.any() and not count.any() and not ema.any()
                and torch.equal(gs, initial), 'zero-version state must be a true cold start')
        self.ema, self.seen, self.count = ema, seen, count
        self.commits, self._generator, self._pending = commits, generator, None

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state['valid_mask'], state['seed'], binding=state['binding'])
        obj.load_state_dict(state)
        return obj

    from_state = from_state_dict

    @torch.no_grad()
    def plan(self, step=None):
        if step is not None:
            _require(_integer(step, 'step') == self.commits, 'plan version must equal committed updates')
        before = self._generator.get_state().clone()
        generator = torch.Generator(device='cpu')
        generator.set_state(before)
        weights = torch.zeros_like(self.ema)
        counts = torch.zeros_like(self.count)
        times, all_draws, all_q, receipts = [], [], [], []
        for window, (first, stop) in enumerate(WINDOWS):
            d = draw_window32(self.ema[first:stop], self.valid_mask, generator)
            weights[first:stop], counts[first:stop] = d['weights'], d['counts']
            td = d['time_draws'] + first
            times.append(td)
            draw_times = td.repeat_interleave(ELEMENT_DRAWS)
            draw_elements = d['element_draws'].reshape(-1)
            all_draws.append(torch.stack((draw_times, draw_elements), 1))
            all_q.append(d['draw_q'])
            # Retain every occurrence, including repeated time/element pairs.
            rows = torch.stack((draw_times.to(torch.float32), draw_elements.to(torch.float32),
                d['draw_time_q'], d['draw_element_q'], d['draw_q']), 1).tolist()
            receipts.append(dict(window=window, slots=[first, stop],
                joint_probability_sha256=_digest(d['q']),
                joint_probability_sum=float(d['q'].sum(dtype=torch.float32)),
                minimum_valid_joint_q=float(d['q'][:, self.valid_mask].min()),
                time_draws=td.tolist(), draw_columns=['time_slot', 'DG_element',
                    'q_time', 'q_element_given_time', 'actual_window_joint_q'],
                draws=[[int(t), int(e), qt, qe, q] for t, e, qt, qe, q in rows]))
        time_draws = torch.cat(times)
        selected = torch.unique(time_draws, sorted=True)
        plan = dict(schema=PLAN_SCHEMA, version=self.commits,
            state_fingerprint=self.fingerprint(), valid_mask_sha256=_digest(self.valid_mask),
            selected_times=selected, time_draws=time_draws,
            time_counts=torch.bincount(time_draws, minlength=N_TIMES),
            counts=counts, weights=weights, draws=torch.cat(all_draws),
            draw_q=torch.cat(all_q), generator_state_before=before,
            generator_state_after=generator.get_state().clone(),
            denominator=N_TIMES*self.n_valid,
            receipt=dict(policy=SCHEMA, proposal_version=self.commits,
                proposal_source='previous committed detached FP32 residual-energy EMA',
                windows=receipts, uniform_mix=UNIFORM_MIX, ema_beta=EMA_BETA,
                total_time_draws=8, total_element_draws=128,
                selected_unique_times=selected.tolist(), n_valid_elements=self.n_valid,
                weight='count / (32 * actual_window_joint_q)',
                loss='sum_all_four_windows(weight * element_energy)/(400*Nvalid)',
                duplicate_draws_retained=True, no_weight_clipping=True,
                no_self_normalization=True, committed_rng_unchanged_by_plan=True,
                full_spatial_decode_at_selected_times=True,
                feedback_all_valid_elements_at_each_computed_unique_time=True))
        plan['plan_sha256'] = _digest(plan)
        self._pending = (self.commits, plan['state_fingerprint'], plan['plan_sha256'])
        return plan

    def _validate_plan(self, plan):
        _require(isinstance(plan, dict) and plan.get('schema') == PLAN_SCHEMA
            and type(plan.get('version')) is int and plan['version'] == self.commits,
            'stale, duplicate or foreign plan version')
        if self._pending is None:
            # Reconstruct a transaction after restoring a committed checkpoint.
            self.plan()
        fingerprint = self.fingerprint()
        digest = _digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
        _require(plan.get('plan_sha256') == digest and self._pending == (
            self.commits, fingerprint, digest), 'plan payload or committed proposal state changed')
        _require(torch.equal(plan['generator_state_before'], self._generator.get_state()),
            'plan RNG origin changed')

    @torch.no_grad()
    def commit(self, plan, time_indices, element_energy):
        """Commit after optimizer.step; average duplicate feedback once per time.

        ``time_indices`` must cover exactly plan['selected_times']; repeated
        feedback rows are allowed and averaged. Each row supplies all 2500
        detached CPU FP32 energies; invalid elements are ignored in the EMA.
        No accepted state changes if validation fails.
        """
        self._validate_plan(plan)
        _require(isinstance(time_indices, torch.Tensor) and time_indices.ndim == 1,
            'time_indices must be a one-dimensional CPU int64 tensor')
        _tensor(time_indices, torch.long, time_indices.shape, 'time_indices')
        _tensor(element_energy, torch.float32, (len(time_indices), N_ELEMENTS), 'element_energy')
        _require(torch.isfinite(element_energy).all() and (element_energy >= 0).all(),
            'finite nonnegative full-element feedback required')
        unique = torch.unique(time_indices, sorted=True)
        _require(torch.equal(unique, plan['selected_times']),
            'feedback must cover all and only computed selected times')
        values = torch.stack([(element_energy[time_indices == t] /
            float(int((time_indices == t).sum()))).sum(0, dtype=torch.float32)
            for t in unique.tolist()])
        _require(torch.isfinite(values).all(), 'duplicate feedback average overflow')
        valid = self.valid_mask
        old = self.ema[unique]
        observed = self.seen[unique]
        updated = torch.where(observed, old*EMA_BETA + values*(1.-EMA_BETA), values)
        updated[:, ~valid] = 0.
        _require(torch.isfinite(updated).all(), 'EMA overflow')
        next_ema, next_seen, next_count = self.ema.clone(), self.seen.clone(), self.count.clone()
        next_ema[unique] = updated
        next_seen[unique] = valid[None].expand(len(unique), -1)
        next_count[unique] += valid.to(torch.long)[None]
        _require(self.commits < 2**63-1, 'sampler version overflow')
        next_generator = torch.Generator(device='cpu')
        next_generator.set_state(plan['generator_state_after'].clone())
        self.ema, self.seen, self.count = next_ema, next_seen, next_count
        self._generator = next_generator
        self.commits += 1
        self._pending = None

    def abort(self, plan=None):
        """Discard an uncommitted transaction without changing RNG/history."""
        if plan is not None:
            self._validate_plan(plan)
        self._pending = None


def importance_loss32(element_energy, plan, time_indices=None):
    """Graph-connected FP32 physics mean for all or selected computed times."""
    _require(isinstance(element_energy, torch.Tensor) and element_energy.dtype == torch.float32
        and element_energy.ndim == 2 and element_energy.shape[1] == N_ELEMENTS,
        'graph-connected FP32 element_energy[T,2500] required')
    _require(torch.isfinite(element_energy).all() and (element_energy >= 0).all(),
        'finite nonnegative element energies required')
    if time_indices is None:
        time_indices = (torch.arange(N_TIMES) if element_energy.shape[0] == N_TIMES
            else plan['selected_times'])
    _tensor(time_indices, torch.long, (element_energy.shape[0],), 'loss time_indices')
    _require(torch.equal(torch.unique(time_indices, sorted=True), time_indices),
        'loss must use unique sorted time rows; multiplicity is already in weights')
    _require(bool(((time_indices >= 0) & (time_indices < N_TIMES)).all())
        and bool(torch.isin(plan['selected_times'], time_indices).all()),
        'loss rows must include all computed sampled times')
    weights = plan['weights'].index_select(0, time_indices).to(device=element_energy.device)
    return (element_energy*weights).sum(dtype=torch.float32) / float(plan['denominator'])


@torch.no_grad()
def stopping_summary(normalized_residual_energies, valid_mask, epsilon=.1,
                     failure_probability_tolerance=.01):
    """Uniform finite-grid P_F, analogous to DeepLWC equations 14 and 15.

    Input must be the already calculated COMPLETE 400x2500 energy grid,
    never an adaptive sample or EMA. Failure is sqrt(energy)>epsilon. The
    discrete prior is uniform over all 400 slots and fixed valid DG elements.
    This function reports a statistic and does not define a training policy.
    """
    mask = _valid_mask(valid_mask)
    energy = normalized_residual_energies
    _require(isinstance(energy, torch.Tensor)
        and tuple(energy.shape) in ((N_TIMES, N_ELEMENTS), (N_TIMES, 50, 50)),
        'failure statistics require the complete 400-time DG residual grid')
    energy = energy.reshape(N_TIMES, N_ELEMENTS)
    _tensor(energy, torch.float32, (N_TIMES, N_ELEMENTS), 'full-grid residual energies')
    _require(torch.isfinite(energy).all() and (energy >= 0).all(),
        'finite nonnegative full-grid residual energies required')
    for value, name in ((epsilon, 'epsilon'),
            (failure_probability_tolerance, 'failure_probability_tolerance')):
        _require(type(value) in (int, float) and math.isfinite(value) and value > 0.,
            name + ' must be finite and positive')
    _require(failure_probability_tolerance <= 1., 'probability tolerance must be at most one')
    residual = energy[:, mask].sqrt()
    failure = residual > torch.tensor(epsilon, dtype=torch.float32)
    count, total = int(failure.sum()), failure.numel()
    probability = float(torch.tensor(count, dtype=torch.float32) / float(total))
    windows = [dict(slots=[first, stop], failures=int(failure[first:stop].sum()),
        cells=(stop-first)*int(mask.sum()),
        failure_probability=float(failure[first:stop].to(torch.float32).mean(dtype=torch.float32)))
        for first, stop in WINDOWS]
    return dict(schema=STOPPING_SCHEMA, full_grid=True, uniform_prior=True,
        epsilon=float(epsilon), failure_probability_tolerance=float(failure_probability_tolerance),
        failure_definition='sqrt(normalized residual energy) > epsilon',
        failure_probability=probability, failure_count=count, total_cells=total,
        below_tolerance=probability < failure_probability_tolerance,
        residual_max=float(residual.max()), residual_energy_mean=float(
            energy[:, mask].mean(dtype=torch.float32)), windows=windows,
        probability_from_complete_uniform_grid=True, adaptive_sample_fraction_used=False,
        Monte_Carlo_used=False, training_stopping_policy_defined=False,
        arithmetic='FP32 residual square root, threshold and probability; exact integer counts')


full_grid_stopping_summary = stopping_summary


def self_check():
    """Finite-sum/gradient and real CPU transaction checks; no model or GPU."""
    import io
    import itertools

    checks, numbers = {}, {}
    global_rng = torch.get_rng_state().clone()

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(name)

    def reject(name, function):
        try:
            function()
        except (ValueError, TypeError, RuntimeError, KeyError):
            check(name, True)
            return
        raise AssertionError(name + ' did not reject')

    mask = torch.tensor([True, True, False, True])
    energy = torch.tensor([[1., 4., 999., 2.], [9., .2, 999., 1.],
        [.01, 3., 999., 8.]], dtype=torch.float32)
    q, qt, qe = joint_proposal32(energy, mask)
    uniform = mask[None].expand_as(energy).to(torch.float32)/9.
    mass = energy.masked_fill(~mask[None], 0.).sqrt()
    exact_q = .25*uniform + .75*mass/mass.sum(dtype=torch.float32)
    check('joint_is_global_sqrt_EMA_mixture', torch.allclose(q, exact_q, atol=2e-8, rtol=1e-6))
    check('all_proposal_arithmetic_FP32', all(x.dtype == torch.float32 for x in (q, qt, qe)))
    check('actual_joint_is_hierarchical_product', torch.equal(q, qt[:, None]*qe))
    check('joint_and_marginals_sum_to_one', torch.allclose(q.sum(), torch.tensor(1.))
        and torch.allclose(qt.sum(), torch.tensor(1.))
        and torch.allclose(qe.sum(1), torch.ones(3)))
    check('positive_uniform_exploration_floor', bool((q >= .25*uniform*(1.-1e-6)).all()))
    check('invalid_elements_have_no_support', not bool(q[:, ~mask].any()))
    cold, _, _ = joint_proposal32(torch.zeros_like(energy), mask)
    check('cold_start_uniform_without_residual_floor', torch.allclose(cold, uniform, rtol=1e-6, atol=1e-8))
    huge, _, _ = joint_proposal32(torch.full_like(energy, torch.finfo(torch.float32).max), mask)
    check('FP32_max_energy_probability_is_finite', torch.isfinite(huge).all())
    reject('attached_proposal_energy_rejected', lambda: joint_proposal32(energy.requires_grad_(), mask))
    energy = energy.detach()
    reject('FP64_proposal_rejected', lambda: joint_proposal32(energy.double(), mask))

    # Integrate the true clustered draw expectation: 2 time draws, each with
    # 16 conditional element draws, including all multiplicities.
    theta = torch.tensor([.4, -.2, .7], dtype=torch.float32, requires_grad=True)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)/7
    values = (theta[0]*x.sin() + theta[1]*x + theta[2]).square()*mask[None]
    denominator = 400*int(mask.sum())
    expected = (2*qt*16*(qe[:, mask]*values[:, mask]/(32*q[:, mask])).sum(1)).sum()/denominator
    exact = values.sum(dtype=torch.float32)/denominator
    g_exact = torch.autograd.grad(exact, theta, retain_graph=True)[0]
    g_expected = torch.autograd.grad(expected, theta)[0]
    numbers['HT_expectation_absolute_error'] = float((expected-exact).abs())
    numbers['HT_gradient_max_absolute_error'] = float((g_expected-g_exact).abs().max())
    check('exact_expected_HT_loss_matches_full_grid', torch.allclose(expected, exact, rtol=2e-6, atol=1e-9))
    check('exact_expected_HT_gradient_matches_full_grid', torch.allclose(g_expected, g_exact, rtol=2e-6, atol=1e-9))
    exhaustive = torch.zeros((), dtype=torch.float32)
    for pair in itertools.product(range(3), repeat=2):
        estimate = sum((16*qe[t, mask]*values[t, mask].detach()/(32*q[t, mask])).sum()
            for t in pair)
        exhaustive += qt[list(pair)].prod()*estimate/denominator
    check('enumerated_ordered_time_pairs_include_duplicates',
        torch.allclose(exhaustive, exact.detach(), rtol=2e-6, atol=1e-9))
    repeated = draw_window32(torch.tensor([[2.]]), torch.tensor([True]),
        torch.Generator(device='cpu').manual_seed(7))
    check('all32_duplicate_draws_retained', int(repeated['counts'][0, 0]) == 32)
    check('duplicate_counts_give_unit_HH_weight', float(repeated['weights'][0, 0]) == 1.)

    valid = torch.zeros(N_ELEMENTS, dtype=torch.bool)
    valid[[0, 1, 3, N_ELEMENTS-1]] = True
    binding = dict(checkpoint_SHA='syntheticCPU', source=[120., 1000., 15., .1, 1.])
    sampler = PaperMarmousiJointAIS(valid, seed=20261020, binding=binding)
    fingerprint = sampler.fingerprint()
    state_before = sampler.state_dict()
    plan = sampler.plan(0)
    check('plan_does_not_advance_committed_state_or_rng', sampler.fingerprint() == fingerprint)
    same = sampler.plan(0)
    check('repeated_uncommitted_plan_is_deterministic', same['plan_sha256'] == plan['plan_sha256'])
    check('four_windows_each_draw32_cells', all(int(plan['counts'][a:b].sum()) == 32 for a, b in WINDOWS))
    check('eight_time_draws_and128_elements', int(plan['time_counts'].sum()) == 8
        and int(plan['counts'].sum()) == 128)
    check('hierarchical_time_multiplicity_matches_element_counts',
        torch.equal(plan['counts'].sum(1), plan['time_counts']*16))
    check('selected_times_are_unique_draw_union', torch.equal(plan['selected_times'],
        torch.unique(plan['draws'][:, 0], sorted=True)))
    check('invalid_elements_never_drawn_or_weighted', not bool(plan['counts'][:, ~valid].any())
        and not bool(plan['weights'][:, ~valid].any()))
    correct_weights = True
    for t, e in torch.nonzero(plan['counts']).tolist():
        draw_q = plan['draw_q'][(plan['draws'][:, 0] == t) & (plan['draws'][:, 1] == e)][0]
        correct_weights &= bool(plan['weights'][t, e] == plan['counts'][t, e].float()/(32*draw_q))
    check('weights_use_actual_joint_q_and_duplicate_counts', correct_weights)
    check('receipts_record_actual_q_for_every_draw', sum(len(r['draws'])
        for r in plan['receipt']['windows']) == 128)

    # CPU torch.save/load round trip of the actual complete sampler state.
    memory = io.BytesIO()
    torch.save(state_before, memory)
    memory.seek(0)
    restored = PaperMarmousiJointAIS.from_state_dict(torch.load(memory, map_location='cpu', weights_only=False))
    check('cold_complete_checkpoint_roundtrip', restored.fingerprint() == fingerprint)
    check('restore_reproduces_next_draw_and_weights', restored.plan()['plan_sha256'] == plan['plan_sha256'])
    original_copy = copy.deepcopy(plan)
    bad = copy.deepcopy(plan)
    bad['weights'][0, 0] += 1.
    feedback = torch.full((len(plan['selected_times']), N_ELEMENTS), 4., dtype=torch.float32)
    reject('tampered_plan_rejected', lambda: sampler.commit(bad, plan['selected_times'], feedback))
    reject('missing_computed_time_feedback_rejected', lambda: sampler.commit(plan,
        plan['selected_times'][:-1], feedback[:-1]))
    reject('attached_feedback_rejected', lambda: sampler.commit(plan,
        plan['selected_times'], feedback.requires_grad_()))
    feedback = feedback.detach()
    check('failed_commits_do_not_change_accepted_state', sampler.fingerprint() == fingerprint)
    tids = plan['selected_times'].repeat_interleave(2)
    duplicated = torch.empty((len(tids), N_ELEMENTS), dtype=torch.float32)
    duplicated[::2] = 2.
    duplicated[1::2] = 6.
    duplicated[:, ~valid] = 999.
    sampler.commit(plan, tids, duplicated)
    restored.commit(original_copy, tids, duplicated)
    selected = plan['selected_times']
    check('first_observation_initializes_to_unweighted_average', bool((sampler.ema[selected][:, valid] == 4.).all()))
    check('duplicate_feedback_counts_once_per_distinct_cell', bool((sampler.count[selected][:, valid] == 1).all()))
    check('invalid_elements_keep_zero_unseen_history', not bool(sampler.ema[:, ~valid].any())
        and not bool(sampler.seen[:, ~valid].any()) and not bool(sampler.count[:, ~valid].any()))
    check('only_successful_commit_advances_rng_and_version', sampler.commits == 1
        and torch.equal(sampler.state_dict()['generator_state'], plan['generator_state_after']))
    check('restored_transaction_commits_identically', sampler.fingerprint() == restored.fingerprint())
    reject('duplicate_old_version_commit_rejected', lambda: sampler.commit(plan, tids, duplicated))
    reject('future_plan_step_rejected', lambda: sampler.plan(2))

    state = sampler.state_dict()
    restored = PaperMarmousiJointAIS.from_state_dict(state)
    next_plan, restored_plan = sampler.plan(), restored.plan()
    check('nonempty_EMA_restore_next_plan_identical', next_plan['plan_sha256'] == restored_plan['plan_sha256'])
    before_ema, before_seen, before_count = sampler.ema.clone(), sampler.seen.clone(), sampler.count.clone()
    old_fingerprint = sampler.fingerprint()
    sampler.abort(next_plan)
    check('abort_preserves_rng_EMA_and_version', sampler.fingerprint() == old_fingerprint)
    next_plan = sampler.plan()
    fresh = torch.full((len(next_plan['selected_times']), N_ELEMENTS), 20., dtype=torch.float32)
    sampler.commit(next_plan, next_plan['selected_times'], fresh)
    ix = next_plan['selected_times']
    expected_ema = torch.where(before_seen[ix][:, valid], before_ema[ix][:, valid]*.9 + 20.*.1,
        torch.full((len(ix), int(valid.sum())), 20.))
    check('subsequent_EMA_is_beta_point9_once', torch.equal(sampler.ema[ix][:, valid], expected_ema))
    check('full_spatial_feedback_increments_count_once',
        torch.equal(sampler.count[ix][:, valid], before_count[ix][:, valid]+1))
    corrupt = copy.deepcopy(state)
    corrupt['count'][int(selected[0]), 0] += 1
    reject('partial_element_history_restore_rejected', lambda: restored.load_state_dict(corrupt))
    wrong_source = PaperMarmousiJointAIS(valid, seed=20261020,
        binding=dict(binding, source=[130., 1000., 15., .1, 1.]))
    reject('wrong_instance_binding_restore_rejected', lambda: wrong_source.load_state_dict(state))
    reject('changed_seed_restore_rejected', lambda: PaperMarmousiJointAIS(valid, seed=1).load_state_dict(state))
    alias = sampler.state_dict()
    alias['ema'].fill_(100.)
    check('state_dict_tensors_do_not_alias_live_history', not bool((sampler.ema == 100.).all()))

    # The graph helper uses unique time rows; multiplicities are already in W.
    e = torch.ones((len(ix), N_ELEMENTS), dtype=torch.float32, requires_grad=True)
    loss = importance_loss32(e, next_plan)
    loss.backward()
    check('physics_loss_and_gradient_are_FP32', loss.dtype == e.grad.dtype == torch.float32)
    check('physics_loss_gradient_equals_W_over_fixed_denominator', torch.equal(e.grad,
        next_plan['weights'][ix]/float(400*int(valid.sum()))))
    reject('duplicated_loss_rows_cannot_double_count', lambda: importance_loss32(
        e.detach().repeat_interleave(2, 0), next_plan, ix.repeat_interleave(2)))

    full = torch.zeros((N_TIMES, N_ELEMENTS), dtype=torch.float32)
    full[:, ~valid] = 100.
    full[0, 0] = .04
    full[1, 0] = torch.tensor(.1, dtype=torch.float32).square()
    report = stopping_summary(full, valid)
    check('failure_probability_uses_complete_uniform_grid', report['failure_count'] == 1
        and report['total_cells'] == 400*int(valid.sum())
        and report['failure_probability'] == float(torch.tensor(1., dtype=torch.float32)/(400*int(valid.sum()))))
    check('epsilon_boundary_is_not_strict_failure', report['failure_count'] == 1)
    check('failure_statistic_defines_no_training_stop_policy', not report['training_stopping_policy_defined']
        and not report['adaptive_sample_fraction_used'] and not report['Monte_Carlo_used'])
    reject('adaptive_subset_rejected_for_failure_probability', lambda: stopping_summary(full[:8], valid))
    reject('attached_fullgrid_statistic_rejected', lambda: stopping_summary(full.requires_grad_(), valid))
    check('global_torch_rng_unchanged', torch.equal(global_rng, torch.get_rng_state()))
    return dict(schema=SCHEMA, status='PASS', checks_passed=len(checks), checks=checks,
        numbers=numbers, CUDA_initialized=torch.cuda.is_initialized(), model_loaded=False,
        optimizer_updates=0, GPU_calls=0,
        scope='Exact finite-sum and gradient expectation, replacement multiplicities, real CPU torch checkpoint roundtrip, committed EMA/RNG transactions, complete-grid uniform failure probability.')
