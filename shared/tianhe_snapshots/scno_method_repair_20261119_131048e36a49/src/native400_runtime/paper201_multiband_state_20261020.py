"""Pinned V2 -> temporal-band pilot migration and complete independent restart.

The paused parent's file, cursor, AIS and RNG remain immutable provenance. The
pilot has its own progress. Expanded controller embeddings retain the parent's
Adam bias clock: new rows start with zero moments, not a fresh per-row clock.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import scno_posix_lock as fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

import torch

from . import marmousi_reduced400_continue_20261020 as rt
from . import marmousi_reduced400_state_20261020 as st
from .paper201_multiband_fno_20261020 import (
    CurriculumAdamW, MultiBandConfig, MultiBandPaperOperator, set_curriculum,
)

SCHEMA = 'PAPER201_MULTIBAND_COMPLETE_PILOT_STATE_V1'
ROOT = Path(__file__).resolve().parents[2]
PARENT_DIR = ROOT/'runs/marmousi_joint400_reduced_prefetch_20261020'
PARENT_SHA = '2f758ffdfc06a3b0c627d94983094285a7a0bf0bba934bfaa2b6123485166aeb'
require, cpu_tree, same = rt.require, rt.cpu_tree, st.old._same


def immutable_digest(model):
    """Only the explicit committed curriculum clock is mutable among buffers."""
    digest = hashlib.sha256()
    values = [('parameter:'+n, p) for n, p in model.named_parameters() if not p.requires_grad]
    values += [('buffer:'+n, b) for n, b in model.named_buffers() if n != 'curriculum_step']
    for name, value in values:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _strict(actual, expected, label):
    require(same(cpu_tree(actual), cpu_tree(expected)), label+' strict roundtrip differs')


def _options(group):
    return copy.deepcopy({k: v for k, v in group.items() if k not in ('params', 'parameter_names')})


def _groups(module, specs):
    named = dict(module.named_parameters())
    return [dict(_options(g), params=[named[n] for n in g['parameter_names']]) for g in specs]


def _specs(module, optimizer):
    names = {id(p): n for n, p in module.named_parameters()}
    return [dict(_options(g), parameter_names=[names[id(p)] for p in g['params']])
            for g in optimizer.param_groups]


def _check_pins(lineage):
    require(lineage['parent_sha256'] == PARENT_SHA, 'paused V2 parent SHA required')
    for path_key, sha_key in (('parent_checkpoint', 'parent_sha256'),
                              ('parent_config_path', 'parent_config_sha256')):
        require(rt.sha(lineage[path_key]) == lineage[sha_key], 'pinned parent changed: '+path_key)
    require(same(json.loads(Path(lineage['parent_config_path']).read_text()), lineage['parent_config']),
            'parent configuration differs from pinned file')


def load_parent(parent_dir=PARENT_DIR, device='cpu'):
    """Validate the complete paused V2 checkpoint and strict-load all four states."""
    parent_dir = Path(parent_dir).resolve()
    cfg_path, path = parent_dir/'TRAINING_CONFIG.json', parent_dir/'latest.pt'
    require(not (parent_dir/'CHECKPOINT_PENDING.json').exists(), 'parent has unresolved transaction')
    cfg_sha = rt.sha(cfg_path)
    cfg = json.loads(cfg_path.read_text())
    require(rt.sha(path) == PARENT_SHA, 'latest must be the pinned paused V2 checkpoint')
    payload = st.LatestStore(parent_dir, cfg, cfg_sha).load()
    require(rt.sha(path) == PARENT_SHA and rt.sha(cfg_path) == cfg_sha, 'parent changed during load')
    # Architecture construction must not consume caller RNG or silently resume it.
    with torch.random.fork_rng(devices=[]):
        model, optimizer, _ = rt.create_operator(cfg, 'cpu')
        model.load_state_dict(payload['model_state'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state'])
        controller = rt.make_learned(model, cfg)
        controller.load_state_dict(payload['learned_optimizer_state'], strict=True)
        meta = torch.optim.Adam(controller.parameters(), lr=cfg['meta_learning_rate'],
                                weight_decay=cfg['meta_weight_decay'], foreach=False)
        meta.load_state_dict(payload['meta_optimizer_state'])
    for object_, key in ((model, 'model_state'), (optimizer, 'optimizer_state'),
                         (controller, 'learned_optimizer_state'), (meta, 'meta_optimizer_state')):
        _strict(object_.state_dict(), payload[key], 'parent '+key)
    require(rt.immutable_digest(model) == cfg['immutable_digest'], 'parent immutable values differ')
    # JSON serializes tuple metadata as arrays. Canonical JSON compares every
    # scalar exactly while preserving the runtime tuple types in our checkpoint.
    require(st.canonical(vars(model.normalizer.metadata)) == st.canonical(cfg['normalizer']),
            'parent fixed normalizer differs')
    require(torch.equal(model.native_times_s.cpu(), payload['native_times_s']) and
            torch.equal(model.full_times_s.cpu(), payload['registered_times_s']), 'parent exact time axis differs')
    if torch.device(device).type != 'cpu':
        model.to(device); controller.to(device)
        optimizer.load_state_dict(payload['optimizer_state'])
        meta.load_state_dict(payload['meta_optimizer_state'])
    return payload, cfg, model, optimizer, controller, meta


def _expand_controller(model, old_controller, old_meta):
    names = tuple(n for n, p in model.named_parameters() if p.requires_grad)
    old_names = tuple('backbone.'+n for n in old_controller.parameter_names)
    require(names[:len(old_names)] == old_names, 'parent block order changed')
    controller = rt.BlockLearnedOptimizer(names, old_controller.config).to(model.device)
    state = controller.state_dict()
    old = old_controller.state_dict()
    for name in state:
        if name not in ('_extra_state', 'embedding.weight'):
            state[name] = old[name].clone()
    state['embedding.weight'].zero_()
    state['embedding.weight'][:len(old_names)].copy_(old['embedding.weight'])
    controller.load_state_dict(state, strict=True)
    specs = _specs(old_controller, old_meta)
    meta = torch.optim.Adam(_groups(controller, specs), foreach=False)
    new_params = dict(controller.named_parameters())
    for name, old_param in old_controller.named_parameters():
        if old_param not in old_meta.state:
            continue
        moment = copy.deepcopy(old_meta.state[old_param])
        if name == 'embedding.weight':
            for field in ('exp_avg', 'exp_avg_sq'):
                expanded = torch.zeros_like(new_params[name])
                expanded[:len(old_names)].copy_(moment[field])
                moment[field] = expanded
        meta.state[new_params[name]] = moment
    return controller, meta


def migrate(parent_dir=PARENT_DIR, band_config=None, device='cpu'):
    """Wrap every original parameter object and preserve all inherited moments."""
    payload, cfg, backbone, old_opt, old_controller, old_meta = load_parent(parent_dir, device)
    band_config = MultiBandConfig() if band_config is None else band_config
    if isinstance(band_config, dict):
        band_config = MultiBandConfig(**band_config)
    original_ids = {n: id(p) for n, p in backbone.named_parameters()}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026102041)
        model = MultiBandPaperOperator(backbone, band_config).to(device)
        groups = [dict(_options(g), params=list(g['params'])) for g in old_opt.param_groups]
        require(len(groups) == 10, 'all ten original AdamW groups required')
        for i, head in enumerate(model.bands.heads):
            groups.append(dict(_options(old_opt.param_groups[0]), params=list(head.parameters()),
                               group_name='temporal_band_'+str(i), temporal_band=i,
                               band_active=i < band_config.active_bands(0)))
        optimizer = CurriculumAdamW(groups, foreach=False)
        for parameter, moment in old_opt.state.items():
            optimizer.state[parameter] = copy.deepcopy(moment)
        controller, meta = _expand_controller(model, old_controller, old_meta)
    require(all(id(dict(model.backbone.named_parameters())[n]) == i for n, i in original_ids.items()),
            'wrapper replaced a parent parameter')
    set_curriculum(model, optimizer, 0)
    parent_dir = Path(parent_dir).resolve()
    progress_keys = ('updates', 'stage_updates', 'epoch', 'offset', 'task_exposures',
                    'current_epoch_order', 'total_optimizer_updates', 'elapsed_worker_seconds',
                    'CUDA_restore8', 'prefetch_updates')
    lineage = dict(schema=SCHEMA+'_LINEAGE', parent_checkpoint=str(parent_dir/'latest.pt'),
        parent_sha256=PARENT_SHA, parent_config_path=str(parent_dir/'TRAINING_CONFIG.json'),
        parent_config_sha256=rt.sha(parent_dir/'TRAINING_CONFIG.json'), parent_config=copy.deepcopy(cfg),
        parent_progress={k: copy.deepcopy(payload[k]) for k in progress_keys if k in payload},
        parent_history_reference=dict(path=str(parent_dir/'latest.pt'), sha256=PARENT_SHA,
            keys=['samplers', 'rng', 'monitor_history', 'normalizer', 'native_times_s', 'registered_times_s'],
            policy='complete original state retained in pinned parent; pilot progress is separate'),
        band_config=asdict(band_config), controller_config=asdict(controller.config),
        controller_names=list(controller.parameter_names),
        controller_migration='old shared network and embedding rows copied; new embedding rows zero',
        meta_migration='all old moments and step inherited; expanded embedding new moments zero; '
                       'new rows share inherited bias clock, no old-trajectory equivalence claim',
        base_age_by_name={n: int(optimizer.state[p]['step']) if p in optimizer.state else 0
                          for n, p in model.named_parameters() if p.requires_grad},
        meta_base_age_by_name={n: int(meta.state[p]['step']) if p in meta.state else 0
                               for n, p in controller.named_parameters()},
        optimizer_groups=_specs(model, optimizer), meta_optimizer_groups=_specs(controller, meta),
        model_signature=st.signature(model.state_dict()), learned_signature=st.signature(controller.state_dict()),
        immutable_digest=immutable_digest(model), normalizer=copy.deepcopy(vars(model.normalizer.metadata)),
        native_times_s=cpu_tree(model.native_times_s), registered_times_s=cpu_tree(model.full_times_s))
    return model, optimizer, controller, meta, lineage


def _finite_state(state, signature, label):
    require(set(state) == set(signature), label+' complete keys required')
    for name, spec in signature.items():
        value = state[name]
        if 'extra_state' in spec:
            require(same(value, spec['extra_state']), label+' semantic state differs: '+name)
        else:
            st.old._cpu_finite_tensor(value, label+' '+name)
            require(list(value.shape) == spec['shape'] and str(value.dtype) == spec['dtype'],
                    label+' shape/dtype differs: '+name)


def _optimizer_state(state, specs, parameters, ages, config=None, step=0):
    require(type(state) is dict and set(state) == {'state', 'param_groups'}, 'complete optimizer state required')
    require(len(state['param_groups']) == len(specs), 'optimizer group count differs')
    all_ids, all_names = [], []
    require(all(type(k) is int and k >= 0 for k in state['state']), 'optimizer moment ID type')
    for actual, spec in zip(state['param_groups'], specs):
        expected = _options(spec)
        band = expected.get('temporal_band')
        if band is not None:
            expected['band_active'] = band < config.active_bands(step)
        require(same(_options(actual), expected), 'optimizer group settings differ')
        require(not actual.get('amsgrad') and not actual.get('capturable') and
                not actual.get('differentiable') and not actual.get('maximize') and
                not actual.get('fused'), 'unsupported Adam settings')
        ids, names = actual['params'], spec['parameter_names']
        require(len(ids) == len(names) and all(type(k) is int and k >= 0 for k in ids), 'optimizer group roster')
        all_ids.extend(ids); all_names.extend(names)
        for identifier, name in zip(ids, names):
            age = ages[name]
            moment = state['state'].get(identifier)
            if moment is None:
                require(age == 0, 'missing populated Adam moment: '+name)
                continue
            require(type(moment) is dict and set(moment) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Adam moment fields')
            for field, value in moment.items():
                st.old._cpu_finite_tensor(value, 'Adam '+name+' '+field, floating=True)
            require(moment['step'].numel() == 1 and float(moment['step']) == age, 'Adam age differs: '+name)
            parameter = parameters[name]
            require(all(moment[k].shape == parameter.shape and moment[k].dtype == parameter.dtype
                        for k in ('exp_avg', 'exp_avg_sq')), 'Adam moment shape/dtype: '+name)
            require(bool((moment['exp_avg_sq'] >= 0).all()), 'negative Adam variance: '+name)
            if age == 0:
                require(not bool(moment['exp_avg'].any()) and not bool(moment['exp_avg_sq'].any()),
                        'dormant band must have zero moments')
    require(len(all_ids) == len(set(all_ids)) and set(state['state']) <= set(all_ids), 'optimizer ID coverage')
    require(len(all_names) == len(set(all_names)) and set(all_names) == set(ages), 'optimizer parameter coverage')


def validate_checkpoint(payload):
    """Validate numerics, exact schedules, complete coverage and fixed contracts."""
    require(type(payload) is dict and payload.get('schema') == SCHEMA, 'multiband checkpoint schema')
    lineage, progress = payload['lineage'], payload['progress']
    require(lineage.get('schema') == SCHEMA+'_LINEAGE', 'lineage schema')
    config = MultiBandConfig(**lineage['band_config'])
    updates = st.old._integer(progress.get('updates'), 'pilot updates')
    require('sampler' in progress and 'rng' in progress, 'complete independent sampler/RNG required')
    st.old._check_rng(progress['rng'], allow_cpu_rng=True)
    sampler = progress['sampler']
    if sampler is None:
        require(updates == 0, 'committed pilot requires sampler state')
    else:
        st.old._validate_compact(sampler)
        require(sampler['commits'] == updates, 'pilot sampler commits differ from update cursor')
    _finite_state(payload['model_state'], lineage['model_signature'], 'model')
    _finite_state(payload['learned_optimizer_state'], lineage['learned_signature'], 'controller')
    clock = payload['model_state']['curriculum_step']
    require(clock.shape == () and clock.dtype == torch.int64 and int(clock) == updates,
            'curriculum clock must equal committed pilot updates')
    require(payload['immutable_digest'] == lineage['immutable_digest'], 'immutable binding changed')
    require(same(payload['normalizer'], lineage['normalizer']) and
            st.canonical(lineage['normalizer']) == st.canonical(lineage['parent_config']['normalizer']),
            'fixed normalization changed')
    for key in ('native_times_s', 'registered_times_s'):
        require(same(payload[key], lineage[key]), 'fixed physical time axis changed')
    base_ages = {}
    for group in lineage['optimizer_groups']:
        band = group.get('temporal_band')
        added = updates if band is None else max(0, updates-config.activation_steps[band])
        for name in group['parameter_names']:
            base_ages[name] = lineage['base_age_by_name'][name]+added
    _optimizer_state(payload['optimizer_state'], lineage['optimizer_groups'], payload['model_state'],
                     base_ages, config, updates)
    meta_ages = {n: a+updates for n, a in lineage['meta_base_age_by_name'].items()}
    _optimizer_state(payload['meta_optimizer_state'], lineage['meta_optimizer_groups'],
                     payload['learned_optimizer_state'], meta_ages)
    require(payload['learned_optimizer_state']['_extra_state']['parameter_names'] ==
            tuple(lineage['controller_names']) == tuple(lineage['base_age_by_name']), 'fixed controller block identities')
    return dict(status='PASS', updates=updates, active_bands=config.active_bands(updates))


def _restore(payload, device='cpu'):
    validate_checkpoint(payload)
    lineage = payload['lineage']
    with torch.random.fork_rng(devices=[]):
        backbone, _, _ = rt.create_operator(lineage['parent_config'], 'cpu')
        model = MultiBandPaperOperator(backbone, MultiBandConfig(**lineage['band_config'])).to(device)
        model.load_state_dict(payload['model_state'], strict=True)
        optimizer = CurriculumAdamW(_groups(model, lineage['optimizer_groups']), foreach=False)
        optimizer.load_state_dict(payload['optimizer_state'])
        controller = rt.BlockLearnedOptimizer(lineage['controller_names'],
                         rt.BlockOptimizerConfig(**lineage['controller_config'])).to(device)
        controller.load_state_dict(payload['learned_optimizer_state'], strict=True)
        meta = torch.optim.Adam(_groups(controller, lineage['meta_optimizer_groups']), foreach=False)
        meta.load_state_dict(payload['meta_optimizer_state'])
    for object_, key in ((model, 'model_state'), (optimizer, 'optimizer_state'),
                         (controller, 'learned_optimizer_state'), (meta, 'meta_optimizer_state')):
        _strict(object_.state_dict(), payload[key], key)
    require(immutable_digest(model) == lineage['immutable_digest'], 'restored frozen parameters/buffers changed')
    require(same(vars(model.normalizer.metadata), lineage['normalizer']), 'restored RMS normalization differs')
    require(torch.equal(model.native_times_s.cpu(), lineage['native_times_s']) and
            torch.equal(model.full_times_s.cpu(), lineage['registered_times_s']), 'restored physical time differs')
    return model, optimizer, controller, meta, copy.deepcopy(lineage), cpu_tree(payload['progress'])


def _read(path, receipt):
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
        require(digest.hexdigest() == receipt['sha256'], 'checkpoint SHA differs from receipt')
        stream.seek(0)
        payload = torch.load(stream, map_location='cpu', weights_only=False)
        after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    require(identity(before) == identity(after) == identity(Path(path).stat()), 'checkpoint changed while reading')
    validate_checkpoint(payload)
    require(payload['progress']['updates'] == receipt['updates'], 'receipt update cursor differs')
    return payload


def _paths(path):
    path = Path(path).resolve()
    return path, path.with_suffix('.receipt.json'), path.with_suffix('.pending.json'), path.with_suffix('.lock')


def _recover(path, receipt_path, pending_path):
    if not pending_path.exists():
        return
    journal = json.loads(pending_path.read_text())
    require(journal['schema'] == SCHEMA and journal['receipt']['path'] == str(path), 'pending journal binding')
    candidate = path.parent/journal['candidate']
    require(candidate.parent == path.parent and candidate.name.startswith('.'+path.name+'.')
            and candidate.suffix == '.tmp', 'pending candidate path')
    receipt = journal['receipt']
    if path.exists() and rt.sha(path) == receipt['sha256']:
        _read(path, receipt)
    else:
        _read(candidate, receipt)
        os.replace(candidate, path)
        st.old._sync_directory(path.parent)
    st.old._write_json(receipt_path, receipt)
    pending_path.unlink()
    st.old._sync_directory(path.parent)


def save_checkpoint(path, model, opt, controller, meta, lineage, progress):
    """Validate and actual-load a candidate, then atomically publish this latest."""
    path, receipt_path, pending_path, lock_path = _paths(path)
    require(path != Path(lineage['parent_checkpoint']).resolve(), 'cannot overwrite paused parent')
    _check_pins(lineage)
    require(immutable_digest(model) == lineage['immutable_digest'], 'fixed parameters/buffers changed before save')
    require(all(p.grad is None for p in model.parameters()), 'checkpoint requires completed main transaction')
    # Meta gradients are consumed but may remain after Adam.step; state_dict excludes them.
    payload = cpu_tree(dict(schema=SCHEMA, model_state=model.state_dict(), optimizer_state=opt.state_dict(),
        learned_optimizer_state=controller.state_dict(), meta_optimizer_state=meta.state_dict(),
        normalizer=vars(model.normalizer.metadata), native_times_s=model.native_times_s,
        registered_times_s=model.full_times_s, immutable_digest=immutable_digest(model),
        lineage=lineage, progress=progress))
    validate_checkpoint(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _recover(path, receipt_path, pending_path)
        if path.exists():
            previous = _read(path, json.loads(receipt_path.read_text()))
            require(progress['updates'] >= previous['progress']['updates'], 'checkpoint rewind')
            require(same(lineage, previous['lineage']), 'checkpoint lineage changed')
        candidate = path.parent/('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
        with candidate.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush(); os.fsync(stream.fileno())
        receipt = dict(schema=SCHEMA, path=str(path), sha256=rt.sha(candidate), updates=progress['updates'])
        loaded = _read(candidate, receipt)
        _strict(loaded, payload, 'serialized checkpoint')
        _restore(loaded, 'cpu')
        st.old._write_json(pending_path, dict(schema=SCHEMA, candidate=candidate.name, receipt=receipt))
        os.replace(candidate, path)
        st.old._sync_directory(path.parent)
        st.old._write_json(receipt_path, receipt)
        pending_path.unlink(); st.old._sync_directory(path.parent)
    return receipt


def restore_checkpoint(path, device='cpu'):
    path, receipt_path, pending_path, lock_path = _paths(path)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _recover(path, receipt_path, pending_path)
        receipt = json.loads(receipt_path.read_text())
        require(receipt['schema'] == SCHEMA and receipt['path'] == str(path), 'checkpoint receipt binding')
        payload = _read(path, receipt)
    _check_pins(payload['lineage'])
    return _restore(payload, device)
