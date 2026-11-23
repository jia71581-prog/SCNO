"""Joint full400 operator/learned-optimizer durable state and lossless AIS codec.

This independent module copies the previously verified journal transaction
implementation; it does not change or import the frozen PGLO state validator.
"""
from __future__ import annotations
import datetime
import scno_posix_lock as fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import uuid
import copy
import numpy as np
import torch
from .paper_marmousi_joint_ais_20261020 import PaperMarmousiJointAIS, CONFIG as AIS_CONFIG, SCHEMA as AIS_SCHEMA

SCHEMA = 'MARMOUSI_JOINT400_LEARNED_OPTIMIZER_V1'
COMPACT_SCHEMA = SCHEMA + '_LOSSLESS_AIS_COMPACT_V1'
BOUNDARY = 'after complete joint theta/phi batch; no pending gradient or sampler'
FROZEN_NAME = 'base.source_encoder.ic_encoder.0.weight'
MASKED_NAME = 'refiner.static.0.conv1.weight'

def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, name + ' must be a genuine nonnegative integer')
    return value


def _digest(value, name):
    _require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
             name + ' must be a lowercase SHA256')
    return value


def _same(a, b):
    """Unlike Python equality, reject bool/int and tuple/list substitutions."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


def _cpu_finite_tensor(value, name, *, floating=False):
    _require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.numel() > 0,
             name + ' must be a nonempty CPU tensor')
    _require(not value.requires_grad, name + ' must be detached')
    if floating:
        _require(value.is_floating_point() or value.is_complex(), name + ' must be floating/complex')
    _require(bool(torch.isfinite(value).all()), name + ' contains nonfinite values')


def _check_rng(rng, *, allow_cpu_rng=False):
    _require(type(allow_cpu_rng) is bool, 'allow_cpu_rng must be bool')
    _require(type(rng) is dict and set(rng) == {'python', 'numpy', 'torch', 'cuda'}, 'complete RNG state required')
    try:
        # Private generator instances: no global random state changes.
        random.Random().setstate(rng['python'])
        np.random.RandomState().set_state(rng['numpy'])
        for name in ['torch', 'cuda']:
            value = rng[name]
            if name == 'cuda' and value is None and allow_cpu_rng:
                continue
            _cpu_finite_tensor(value, name + ' RNG')
            _require(value.dtype == torch.uint8 and value.ndim == 1, name + ' RNG byte vector required')
        torch.Generator(device='cpu').set_state(rng['torch'])
    except (TypeError, RuntimeError, IndexError) as exc:
        raise ValueError('invalid checkpoint RNG state') from exc


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sampler_binding(row, cfg, parent_sha):
    _integer(row, 'sampler row')
    _require(row < cfg['tasks_per_epoch'], 'sampler row outside training roster')
    return dict(row=row, sample_id=f'train_marmousi_n15_{row:04d}', parent_sha256=parent_sha,
                dataset_sha256=cfg['dataset_sha256'])


def _validate_compact(value):
    fields = {'schema', 'ais_schema', 'config', 'seed', 'binding', 'commits', 'valid_mask',
              'row_counts', 'seen_times', 'ema_seen', 'generator_state'}
    _require(type(value) is dict and set(value) == fields, 'complete lossless AIS compact fields required')
    _require(value['schema'] == COMPACT_SCHEMA and value['ais_schema'] == AIS_SCHEMA and
             _same(value['config'], AIS_CONFIG), 'compact AIS numerical protocol mismatch')
    _integer(value['seed'], 'AIS seed'); commits = _integer(value['commits'], 'AIS commits')
    mask, counts, seen, ema = (value[k] for k in ['valid_mask', 'row_counts', 'seen_times', 'ema_seen'])
    for name, tensor, dtype, shape in [('valid_mask', mask, torch.bool, (2500,)),
                                     ('row_counts', counts, torch.int64, (400,))]:
        _cpu_finite_tensor(tensor, name)
        _require(tensor.dtype == dtype and tuple(tensor.shape) == shape, 'compact AIS '+name+' shape/dtype')
    _require(bool(mask.any()) and bool((counts >= 0).all()) and bool((counts <= commits).all()),
             'compact AIS valid domain/row ages')
    expected_seen = torch.nonzero(counts > 0, as_tuple=False).flatten()
    _require(isinstance(seen, torch.Tensor) and seen.device.type == 'cpu' and seen.dtype == torch.int64
             and not seen.requires_grad and torch.equal(seen, expected_seen), 'compact AIS seen_times must be exact sorted rows')
    _require(isinstance(ema, torch.Tensor) and ema.device.type == 'cpu' and ema.dtype == torch.float32 and
             not ema.requires_grad and tuple(ema.shape) == (seen.numel(), 2500), 'compact AIS EMA shape/dtype')
    _require(bool(torch.isfinite(ema).all()) and bool((ema >= 0).all()) and not bool(ema[:, ~mask].any()),
             'compact AIS finite nonnegative EMA and exact invalid zeros')
    gs = value['generator_state']
    _cpu_finite_tensor(gs, 'AIS generator')
    _require(gs.dtype == torch.uint8 and gs.ndim == 1, 'AIS generator byte vector')
    torch.Generator().set_state(gs)
    if commits == 0:
        _require(not bool(counts.any()) and torch.equal(gs, torch.Generator().manual_seed(value['seed']).get_state()),
                 'zero-commit AIS must be true cold state')
    _canonical(value['binding'])
    return value


def pack_sampler(sampler):
    """Losslessly remove repeated spatial counts/seen flags and unseen EMA rows."""
    state = sampler.state_dict() if isinstance(sampler, PaperMarmousiJointAIS) else sampler
    # Canonical loader rejects partial spatial feedback, invalid cells or RNG.
    canonical = PaperMarmousiJointAIS.from_state_dict(state).state_dict()
    mask = canonical['valid_mask']
    first = int(torch.nonzero(mask, as_tuple=False)[0])
    counts = canonical['count'][:, first].clone()
    seen = torch.nonzero(counts > 0, as_tuple=False).flatten()
    result = dict(schema=COMPACT_SCHEMA, ais_schema=canonical['schema'],
        config=copy.deepcopy(canonical['config']), seed=canonical['seed'], binding=copy.deepcopy(canonical['binding']),
        commits=canonical['commits'], valid_mask=mask.clone(), row_counts=counts, seen_times=seen,
        ema_seen=canonical['ema'].index_select(0, seen).clone(), generator_state=canonical['generator_state'].clone())
    return _validate_compact(result)


def unpack_sampler(compact):
    """Return a fresh canonical AIS object with identical next-plan randomness."""
    c = _validate_compact(compact)
    count = c['row_counts'][:, None].expand(400, 2500).clone()
    count[:, ~c['valid_mask']] = 0
    ema = torch.zeros((400, 2500), dtype=torch.float32)
    ema.index_copy_(0, c['seen_times'], c['ema_seen'])
    state = dict(schema=c['ais_schema'], config=copy.deepcopy(c['config']), seed=c['seed'],
        binding=copy.deepcopy(c['binding']), commits=c['commits'], valid_mask=c['valid_mask'].clone(),
        count=count, seen=count > 0, ema=ema, generator_state=c['generator_state'].clone())
    return PaperMarmousiJointAIS.from_state_dict(state)


def _optimizer_state(optimizer, names_by_group, settings, model_state, age, *, label):
    _require(type(optimizer) is dict and set(optimizer) == {'state', 'param_groups'}, label+' complete optimizer state')
    groups = optimizer['param_groups']; moments = optimizer['state']
    _require(type(groups) is list and type(moments) is dict and len(groups) == len(names_by_group) == len(settings),
             label+' group layout differs')
    _require(all(type(identifier) is int and identifier >= 0 for identifier in moments),
             label+' moment IDs must be genuine nonnegative integers')
    all_ids = []
    for group, names, expected in zip(groups, names_by_group, settings):
        ids = group.get('params')
        _require(type(ids) is list and len(ids) == len(names) and all(type(i) is int and i >= 0 for i in ids),
                 label+' parameter IDs')
        all_ids.extend(ids)
        for key in ['lr', 'weight_decay']:
            _require(type(group.get(key)) in (int, float) and group[key] == expected[key], label+' '+key+' changed')
        if 'group_name' in expected:
            _require(group.get('group_name') == expected['group_name'], label+' group identity changed')
        _require(_same(group.get('betas'), (.9, .999)) and group.get('eps') == 1e-8 and
                 group.get('amsgrad') is False and group.get('maximize') is False and group.get('foreach') is False
                 and group.get('capturable') is False and group.get('differentiable') is False and group.get('fused') is None,
                 label+' Adam numerical settings changed')
        for identifier, name in zip(ids, names):
            if age == 0:
                continue
            value = moments.get(identifier)
            _require(type(value) is dict and set(value) == {'step', 'exp_avg', 'exp_avg_sq'}, label+' complete populated moments')
            for field, tensor in value.items(): _cpu_finite_tensor(tensor, label+' '+name+' '+field, floating=True)
            _require(value['step'].numel() == 1 and float(value['step']) == age, label+' moment age/cursor mismatch')
            parameter = model_state[name]
            _require(value['exp_avg'].shape == value['exp_avg_sq'].shape == parameter.shape and
                     value['exp_avg'].dtype == value['exp_avg_sq'].dtype == parameter.dtype,
                     label+' moment shape/dtype mismatch')
            squared = torch.view_as_real(value['exp_avg_sq']) if value['exp_avg_sq'].is_complex() else value['exp_avg_sq']
            _require(bool((squared >= 0).all()), label+' negative second moment')
    _require(len(all_ids) == len(set(all_ids)), label+' duplicated parameter IDs')
    _require(set(moments) == set(all_ids) if age else not moments, label+' missing/extra optimizer states')


def validate_checkpoint(state, cfg, config_sha, dataset_complete_sha, parent_sha,
                        expected_names, expected_immutable, *, allow_cpu_rng=False):
    """Pure validation of one joint theta/phi commit, without reading labels."""
    _require(type(state) is dict and type(cfg) is dict and state.get('schema') == cfg.get('schema') == SCHEMA,
             'joint400 checkpoint schema')
    for name, expected in [('config_sha256', config_sha), ('dataset_sha256', cfg['dataset_sha256']),
        ('dataset_complete_sha256', dataset_complete_sha), ('parent_sha256', parent_sha), ('immutable_digest', expected_immutable)]:
        _digest(expected, name)
        _require(state.get(name) == expected, 'joint checkpoint binding mismatch: '+name)
    for name, expected in [('epochs', 20), ('tasks_per_epoch', 500), ('task_batch_size', 4), ('maximum_outer_updates', 2500)]:
        _require(type(cfg.get(name)) is int and cfg[name] == expected, 'authorized joint400 budget: '+name)
    seed = _integer(cfg['seed'], 'training seed')
    names = list(expected_names)
    _require(len(names) == len(set(names)) == 398 and all(type(n) is str for n in names) and FROZEN_NAME not in names,
             'exact398 inherited-active main parameters required')
    _require(type(state.get('active_parameter_names')) is list and state['active_parameter_names'] == names
             and cfg['active_parameter_names'] == names, 'main parameter roster/order changed')
    epoch, offset, updates, exposures = (_integer(state.get(k), k) for k in ['epoch', 'offset', 'updates', 'task_exposures'])
    _require(epoch <= 20 and offset < 500 and offset % 4 == 0 and (epoch < 20 or offset == 0), 'joint epoch/offset')
    _require(exposures == epoch*500+offset and updates*4 == exposures and updates <= 2500, 'joint cursor/exposure mismatch')
    _require(type(state.get('inherited_optimizer_updates')) is int and state['inherited_optimizer_updates'] == 2000,
             'original2000 main moment history required')
    _require(type(state.get('total_optimizer_updates')) is int and state['total_optimizer_updates'] == 2000+updates,
             'main optimizer cumulative age must be2000+new joint updates')
    order = state.get('current_epoch_order')
    _require(type(order) is list and all(type(i) is int for i in order) and
             order == torch.randperm(500, generator=torch.Generator().manual_seed(seed+epoch)).tolist(),
             'joint deterministic500 source order changed')
    _require(state.get('checkpoint_boundary') == BOUNDARY, 'only complete joint theta/phi commits can be saved')
    _require(type(state.get('CUDA_restore8')) is bool and (not state['CUDA_restore8'] or updates >= 8)
             and (updates <= 8 or state['CUDA_restore8']), 'joint CUDA8 restore boundary')
    elapsed = state.get('elapsed_worker_seconds')
    _require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, 'finite elapsed worker time')
    history = state.get('separate_pglo_history')
    _require(type(history) is dict and _canonical(history) == _canonical(cfg['separate_pglo_history'])
             and type(history.get('updates')) is int and history['updates'] == 6
             and type(history.get('wall_seconds')) in (int, float) and abs(history['wall_seconds']-100.165) < .01
             and history.get('counted_in_joint_updates') is False, 'old6 PGLO updates/cost must remain separate')
    _require(type(state.get('monitor_history')) is list, 'monitor history list required')
    for monitor in state['monitor_history']:
        _require(type(monitor) is dict and type(monitor.get('epoch')) is int and 1 <= monitor['epoch'] <= epoch and
                 type(monitor.get('updates')) is int and monitor['updates'] == monitor['epoch']*125, 'monitor cursor mismatch')

    model = state.get('model_state')
    _require(type(model) is dict and set(model) == set(cfg['model_state_keys']) and FROZEN_NAME in model
             and set(names).issubset(model), 'full original main model state required')
    for name, value in model.items(): _cpu_finite_tensor(value, 'main model '+name)
    groups = cfg['optimizer_groups']
    _require(type(groups) is list and len(groups) == 10, 'original10 main AdamW groups')
    grouped_names = [g['parameter_names'] for g in groups]
    flattened = [name for group in grouped_names for name in group]
    _require(len(flattened) == len(set(flattened)) == 398 and set(flattened) == set(names), 'main optimizer coverage')
    masked = [g for g in groups if g['group_name'] == 'refiner_masked_input']
    _require(len(masked) == 1 and masked[0]['parameter_names'] == [MASKED_NAME] and masked[0]['weight_decay'] == 0.,
             'masked-IC input group must retain zero decay')
    _optimizer_state(state.get('optimizer_state'), grouped_names, groups, model, 2000+updates, label='mainAdamW')
    _require(_canonical(state.get('normalizer')) == _canonical(cfg['normalizer']) and
             state['normalizer']['pressure_scale_pa'] == 1.252277344221455e-8, 'frozen parent normalizer changed')
    authority = torch.from_numpy(np.asarray(np.arange(801, dtype=np.float64)*.00125, np.float32))
    native, registered = state.get('native_times_s'), state.get('registered_times_s')
    _cpu_finite_tensor(native, 'native times', floating=True)
    _cpu_finite_tensor(registered, 'registered times', floating=True)
    frames = cfg['full_native_frames']
    _require(type(frames) is list and len(frames) == 400 and all(type(i) is int and 0 <= i < 801 for i in frames)
             and frames == sorted(set(frames)), 'fixed full400 native frame IDs')
    _require(native.dtype == registered.dtype == torch.float32 and torch.equal(native, authority)
             and torch.equal(registered, authority[frames]), 'canonical native/registered times changed')

    learned = state.get('learned_optimizer_state')
    _require(isinstance(learned, dict) and _canonical(state.get('learned_optimizer_config')) ==
             _canonical(cfg['block_learned_optimizer']), 'joint learned optimizer config mismatch')
    phi_names = cfg['meta_parameter_names']
    _require(type(phi_names) is list and len(phi_names) == len(set(phi_names)) > 0
             and all(type(n) is str and n in learned for n in phi_names), 'learned parameter roster')
    extra = learned.get('_extra_state')
    _require(type(extra) is dict and _canonical(extra.get('config')) == _canonical(cfg['block_learned_optimizer']),
             'learned optimizer extra config identity')
    extra_names = extra.get('parameter_names', extra.get('block_names'))
    _require(type(extra_names) in (tuple, list) and list(extra_names) == names, 'learned main block identities')
    _require(extra.get('schema') == cfg.get('learned_optimizer_schema',
        'MARMOUSI_DIRECT400_ALLPARAM_CONTINUE_FIT2000_V1_BLOCK_CONTROLLER')
        and extra.get('conditioning_dimensions') == [8, 16, 4, 2], 'learned controller architecture identity')
    signature = cfg['learned_signature']
    _require(type(signature) is dict and set(learned) == set(signature) | {'_extra_state'}, 'complete learned tensor signature')
    for name, spec in signature.items():
        value = learned[name]
        _cpu_finite_tensor(value, 'learned optimizer '+name, floating=True)
        _require(list(value.shape) == spec['shape'] and str(value.dtype) == spec['dtype'], 'learned tensor signature drift: '+name)
    meta_settings = [dict(lr=cfg['meta_learning_rate'], weight_decay=cfg.get('meta_weight_decay', 0.))]
    _optimizer_state(state.get('meta_optimizer_state'), [phi_names], meta_settings, learned, updates, label='phiAdam')

    samplers = state.get('samplers')
    _require(type(samplers) is dict and all(type(row) is int and 0 <= row < 500 for row in samplers),
             'per-record compact AIS integer roster')
    visited_current = set(order[:offset]); total_commits = 0
    for row in range(500):
        expected_commits = epoch + int(row in visited_current)
        if row not in samplers:
            _require(expected_commits == 0, 'visited record missing its durable AIS history')
            continue
        compact = _validate_compact(samplers[row])
        _require(compact['commits'] == expected_commits and compact['seed'] == seed+row
                 and _same(compact['binding'], sampler_binding(row, cfg, parent_sha)), 'AIS source identity/committed clock drift')
        total_commits += compact['commits']
    _require(total_commits == exposures, 'sum of per-record AIS commits must equal new source exposures')
    _check_rng(state.get('rng'), allow_cpu_rng=allow_cpu_rng)
    return dict(status='PASS', epoch=epoch, offset=offset, updates=updates, task_exposures=exposures,
        main_optimizer_age=2000+updates, meta_optimizer_age=updates, per_record_AIS_commits=total_commits,
        old_pglo_updates_counted=False, no_label_reads=True, global_RNG_unchanged=True,
        actual_restored_immutable_digest_still_required=True)

def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def load_verified_latest(run, cfg, config_sha, dataset_complete_sha, parent_sha,
                         expected_names, expected_immutable):
    """Recover an interrupted commit, then hash-before-load its durable state."""
    run = Path(run)
    args = (cfg, config_sha, dataset_complete_sha, parent_sha, expected_names, expected_immutable)
    with (run / '.checkpoint_transaction.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'CHECKPOINT_PENDING.json').exists():
            _reconcile_pending(run, args)
        return _load_receipted_latest(run, args)


def _load_bound_file(path, digest, args):
    """Keep one descriptor from SHA verification through deserialization."""
    path = Path(path)
    _digest(digest, 'checkpoint file SHA')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        actual = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            actual.update(block)
        _require(actual.hexdigest() == digest, 'latest checkpoint SHA differs from durable receipt')
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=False)
        after = os.fstat(stream.fileno())
        at_path = path.stat()
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    _require(signature(before) == signature(after) == signature(at_path),
             'checkpoint changed during verified loading')
    validate_checkpoint(state, *args)
    return state


def _receipt_matches_state(receipt, state):
    for key in ['epoch', 'offset', 'updates', 'task_exposures']:
        _require(type(receipt.get(key)) is int and receipt[key] == state[key], 'receipt cursor mismatch: ' + key)


def _load_receipted_latest(run, args):
    path = run / 'latest.pt'
    receipt = json.loads((run / 'CHECKPOINT.json').read_text())
    _require(Path(receipt['path']).resolve() == path.resolve(), 'checkpoint receipt path mismatch')
    digest = _digest(receipt.get('sha256'), 'checkpoint receipt SHA')
    state = _load_bound_file(path, digest, args)
    _receipt_matches_state(receipt, state)
    return state


def _checkpoint_fault(stage):
    """No-op injection point for CPU crash-window tests."""


def _record_commit_recovery(run, pending, outcome, state, *, candidate_validated=None):
    moment = datetime.datetime.now(datetime.timezone.utc)
    directory = run / 'checkpoint_recovery_history' / (moment.strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    _write_json(directory / 'CHECKPOINT_PENDING.json', pending)
    candidate = run / pending['candidate_basename']
    if outcome == 'ROLLED_BACK_UNPUBLISHED_CANDIDATE' and candidate.exists():
        os.rename(candidate, directory / candidate.name)
        _sync_directory(run)
    abandoned_exposures = None
    if candidate_validated:
        abandoned_exposures = pending['new_receipt']['task_exposures'] - state['task_exposures']
        _require(abandoned_exposures >= 0, 'unpublished candidate predates durable checkpoint')
    report = dict(schema='JOINT400_CHECKPOINT_COMMIT_RECOVERY_V1', status=outcome, utc=moment.isoformat(),
        recovered_durable_boundary={key: state[key] for key in ['epoch', 'offset', 'updates', 'task_exposures']},
        candidate_validated=candidate_validated, unpublished_candidate_task_exposures_to_replay=abandoned_exposures,
        additional_unlogged_inflight_exposures='unknown', prior_costs_preserved=True,
        no_optimizer_update_performed=True, evidence_directory=str(directory))
    _write_json(directory / 'RECOVERY.json', report)
    _sync_directory(directory)
    return report


def _reconcile_pending(run, args):
    """Only publish a hash-bound, fully validated candidate or keep valid old state."""
    pending_path = run / 'CHECKPOINT_PENDING.json'
    pending = json.loads(pending_path.read_text())
    _require(type(pending) is dict and pending.get('schema') == 'JOINT400_CHECKPOINT_PENDING_V1', 'invalid commit journal')
    candidate_name = pending.get('candidate_basename')
    _require(type(candidate_name) is str and candidate_name.startswith('.latest.') and candidate_name.endswith('.tmp')
             and Path(candidate_name).name == candidate_name, 'invalid checkpoint candidate path')
    candidate = run / candidate_name
    latest = run / 'latest.pt'
    receipt_path = run / 'CHECKPOINT.json'
    new_receipt = pending.get('new_receipt')
    _require(type(new_receipt) is dict and Path(new_receipt.get('path', '')).resolve() == latest.resolve(),
             'pending final path mismatch')
    digest = _digest(new_receipt.get('sha256'), 'pending candidate SHA')
    expected_binding = dict(config_sha256=args[1], dataset_sha256=args[0]['dataset_sha256'],
        dataset_complete_sha256=args[2], parent_sha256=args[3], immutable_digest=args[5])
    _require(_same(pending.get('binding'), expected_binding), 'pending checkpoint input binding mismatch')
    actual_latest_sha = _sha(latest) if latest.exists() else None
    if actual_latest_sha == digest:
        state = _load_bound_file(latest, digest, args)
        _receipt_matches_state(new_receipt, state)
        _write_json(receipt_path, new_receipt)
        _record_commit_recovery(run, pending, 'COMPLETED_PUBLISHED_LATEST_RECEIPT', state, candidate_validated=True)
    elif latest.exists():
        # A candidate not yet renamed must not silently supersede an intact
        # formally committed state, even when it contains additional updates.
        formal = json.loads(receipt_path.read_text())
        _require(_same(formal, pending.get('previous_receipt')) and formal.get('sha256') == actual_latest_sha,
                 'latest matches neither pending candidate nor previous durable receipt')
        state = _load_receipted_latest(run, args)
        candidate_validated = False
        if candidate.exists() and _sha(candidate) == digest:
            try:
                proposed = _load_bound_file(candidate, digest, args)
                _receipt_matches_state(new_receipt, proposed)
                _require(proposed['updates'] >= state['updates'], 'candidate cursor moves backwards')
                candidate_validated = True
            except (ValueError, RuntimeError, TypeError):
                candidate_validated = False
        _record_commit_recovery(run, pending, 'ROLLED_BACK_UNPUBLISHED_CANDIDATE', state,
                                candidate_validated=candidate_validated)
    else:
        _require(pending.get('previous_receipt') is None and not receipt_path.exists(),
                 'missing latest may only recover the first checkpoint commit')
        _require(candidate.exists(), 'first pending checkpoint candidate is missing')
        state = _load_bound_file(candidate, digest, args)
        _receipt_matches_state(new_receipt, state)
        os.replace(candidate, latest)
        _sync_directory(run)
        _write_json(receipt_path, new_receipt)
        _record_commit_recovery(run, pending, 'COMPLETED_FIRST_CHECKPOINT_PUBLICATION', state, candidate_validated=True)
    pending_path.unlink()
    _sync_directory(run)


def save_verified_latest(run, state, cfg, config_sha, dataset_complete_sha, parent_sha,
                         expected_names, expected_immutable):
    """Validate, serialize/read back, and journal the only durable latest.

Returns the complete CPU readback state. A small pending journal bridges the
otherwise non-atomic latest.pt/CHECKPOINT.json pair. No failed validation can
replace a prior valid checkpoint; crash leftovers are reconciled by load.
"""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    args = (cfg, config_sha, dataset_complete_sha, parent_sha, expected_names, expected_immutable)
    validate_checkpoint(state, *args)
    with (run / '.checkpoint_transaction.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _require(not (run / 'CHECKPOINT_PENDING.json').exists(), 'pending commit must be recovered before a new save')
        previous = None
        if (run / 'latest.pt').exists() or (run / 'CHECKPOINT.json').exists():
            prior = _load_receipted_latest(run, args)
            _require(state['updates'] >= prior['updates'], 'new checkpoint cannot rewind durable updates')
            previous = json.loads((run / 'CHECKPOINT.json').read_text())
        candidate = run / ('.latest.' + uuid.uuid4().hex + '.tmp')
        with candidate.open('wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(run)
        digest = _sha(candidate)
        loaded = _load_bound_file(candidate, digest, args)
        _require(_same(state, loaded), 'full checkpoint serialized readback differs')
        _checkpoint_fault('after_candidate_readback_before_pending')
        receipt = dict(path=str((run / 'latest.pt').resolve()), sha256=digest,
            **{key: state[key] for key in ['epoch', 'offset', 'updates', 'task_exposures']},
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        pending = dict(schema='JOINT400_CHECKPOINT_PENDING_V1', candidate_basename=candidate.name,
            new_receipt=receipt, previous_receipt=previous,
            binding=dict(config_sha256=config_sha, dataset_sha256=cfg['dataset_sha256'],
                         dataset_complete_sha256=dataset_complete_sha, parent_sha256=parent_sha,
                         immutable_digest=expected_immutable), candidate_bytes=candidate.stat().st_size)
        _write_json(run / 'CHECKPOINT_PENDING.json', pending)
        _checkpoint_fault('after_pending_before_replace')
        os.replace(candidate, run / 'latest.pt')
        _sync_directory(run)
        _checkpoint_fault('after_replace_before_receipt')
        _write_json(run / 'CHECKPOINT.json', receipt)
        _checkpoint_fault('after_receipt_before_pending_cleanup')
        (run / 'CHECKPOINT_PENDING.json').unlink()
        _sync_directory(run)
        return loaded


def _sync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _journal_metrics(raw, durable_update):
    parsed, invalid, partial = [], [], False
    chunks = raw.splitlines(keepends=True)
    for number, line in enumerate(chunks, 1):
        if not line.endswith(b'\n'):
            partial = True
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            _integer(row.get('updates'), 'logged updates')
            _integer(row.get('task_exposures'), 'logged exposures')
            _require(type(row.get('tasks')) is list and len(row['tasks']) > 0, 'logged nonempty task transaction')
            parsed.append(row)
        except (ValueError, TypeError, AttributeError, UnicodeDecodeError):
            invalid.append(number)
    beyond = [row for row in parsed if row['updates'] > durable_update]
    return dict(parsed_transaction_rows=len(parsed), invalid_or_incomplete_lines=invalid,
        non_newline_terminated_final_line=partial,
        maximum_recorded_updates=max((row['updates'] for row in parsed), default=None),
        maximum_recorded_task_exposures=max((row['task_exposures'] for row in parsed), default=None),
        beyond_durable_logged_updates=[row['updates'] for row in beyond],
        beyond_durable_logged_transactions=len(beyond),
        logged_replayed_task_exposures_lower_bound=sum(len(row['tasks']) for row in beyond),
        duplicate_update_ids=len(parsed) - len({row['updates'] for row in parsed}),
        unlogged_inflight_task_exposures='unknown', unlogged_inflight_outer_updates='unknown',
        counts_are_lower_bounds=True)


def archive_resume_metrics(run, state, *, checkpoint_sha=None, reason='explicit resume'):
    """Retire an old metrics segment without deleting cost/exposure history.

The returned lower bound counts already-computed tasks whose outer updates
will be replayed after the durable checkpoint. Unknown work after the final
logged transaction is explicitly unquantified. It never changes ledger files.
"""
    run = Path(run)
    _require(type(reason) is str and reason.strip(), 'explicit resume reason required')
    for name in ['epoch', 'offset', 'updates', 'task_exposures']:
        _integer(state[name], 'durable ' + name)
    if checkpoint_sha is None:
        checkpoint_sha = json.loads((run / 'CHECKPOINT.json').read_text())['sha256']
    _digest(checkpoint_sha, 'resume checkpoint SHA')
    history = run / 'resume_history'
    history.mkdir(parents=True, exist_ok=True)
    with (run / '.resume_metrics.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        metrics = run / 'metrics.jsonl'
        # Complete a previous rename transaction interrupted after its journal
        # was durably prepared. The original bytes remain authoritative.
        for pending in sorted(history.glob('*/RECEIPT.json')):
            old = json.loads(pending.read_text())
            if old.get('status') != 'PREPARED':
                continue
            archived = pending.parent / 'metrics.jsonl'
            if not archived.exists() and old['metrics_present']:
                _require(metrics.exists() and _sha(metrics) == old['metrics_sha256'],
                         'interrupted resume journal cannot locate its original metrics')
                os.rename(metrics, archived)
                _sync_directory(run)
            if old['metrics_present']:
                _require(_sha(archived) == old['metrics_sha256'], 'interrupted resume archive changed')
            old['status'] = 'ARCHIVED_RECOVERED'
            old['recovered_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            _write_json(pending, old)
        moment = datetime.datetime.now(datetime.timezone.utc)
        segment = history / (moment.strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex)
        segment.mkdir()
        raw = metrics.read_bytes() if metrics.exists() else b''
        present = metrics.exists()
        summaries = _journal_metrics(raw, state['updates'])
        costs = {}
        for name in ['STATUS.json', 'CHECKPOINT.json', 'WORKER.json', 'WORKER_ERROR.json',
                     'LEDGER_START.json', 'LEDGER_FINISH.json', 'ERROR.json']:
            source = run / name
            if source.exists():
                content = source.read_bytes()
                with (segment / name).open('wb') as stream:
                    stream.write(content); stream.flush(); os.fsync(stream.fileno())
                costs[name] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
        previous_lower_bound = sum(json.loads(p.read_text()).get('logged_replayed_task_exposures_lower_bound', 0)
            for p in history.glob('*/RECEIPT.json'))
        receipt = dict(schema='JOINT400_RESUME_METRICS_SEGMENT_V1', status='PREPARED',
            utc=moment.isoformat(), reason=reason, archive=str(segment), checkpoint_sha256=checkpoint_sha,
            durable_boundary={name: state[name] for name in ['epoch', 'offset', 'updates', 'task_exposures']},
            prior_worker_seconds_at_checkpoint=state.get('elapsed_worker_seconds'),
            metrics_present=present, metrics_sha256=hashlib.sha256(raw).hexdigest(), metrics_bytes=len(raw),
            evidence_snapshots=costs, **summaries,
            cumulative_logged_replayed_task_exposures_lower_bound=previous_lower_bound + summaries['logged_replayed_task_exposures_lower_bound'],
            old_metric_bytes_preserved=True, scientific_cursor_restarts_at_durable_boundary=True,
            physical_replayed_work_is_not_new_unique_roster_coverage=True,
            ledger_cost_policy='All prior ledger attempts/costs remain untouched; unknown in-flight work is not reset to zero.')
        _write_json(segment / 'RECEIPT.json', receipt)
        if present:
            _require(metrics.exists() and _sha(metrics) == receipt['metrics_sha256'],
                     'metrics changed during recovery; old worker must be stopped')
            os.rename(metrics, segment / 'metrics.jsonl')
            _sync_directory(run)
            _sync_directory(segment)
        receipt['status'] = 'ARCHIVED'
        _write_json(segment / 'RECEIPT.json', receipt)
        return receipt


__all__ = ['SCHEMA', 'BOUNDARY', 'validate_checkpoint', 'load_verified_latest', 'save_verified_latest', 'archive_resume_metrics', 'pack_sampler', 'unpack_sampler', 'sampler_binding']
