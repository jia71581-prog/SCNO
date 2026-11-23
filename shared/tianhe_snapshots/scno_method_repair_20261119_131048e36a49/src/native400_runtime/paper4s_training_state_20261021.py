"""Complete state transactions for the independent 4 s, 500-epoch stage.

All parent weights transfer; this new dataset/measurement objective explicitly
starts fresh main/controller optimizer clocks. The paused parent is immutable.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import uuid

import numpy as np
import torch

from . import marmousi_direct400_continue_20261020 as joint
from .paper201_multiband_fno_20261020 import CurriculumAdamW

SCHEMA = 'PAPER4S_MULTIBAND10_FULL_JOINT_TRAINING_V1'
RNG_LOCK = threading.RLock()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def freeze(value):
    if torch.is_tensor(value):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: freeze(v) for k, v in value.items()}
    if isinstance(value, list):
        return [freeze(v) for v in value]
    if isinstance(value, tuple):
        return tuple(freeze(v) for v in value)
    return copy.deepcopy(value)


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def finite_tree(value):
    if torch.is_tensor(value):
        if value.device.type != 'cpu' or not bool(torch.isfinite(value).all()):
            raise ValueError('checkpoint tensors must be finite CPU values')
    elif isinstance(value, dict):
        for v in value.values():
            finite_tree(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            finite_tree(v)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite checkpoint scalar')


def rng_state():
    with RNG_LOCK:
        return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def set_rng(value):
    with RNG_LOCK:
        random.setstate(value['python'])
        np.random.set_state(value['numpy'])
        torch.set_rng_state(value['torch'])
        if value['cuda'] is not None:
            torch.cuda.set_rng_state_all(value['cuda'])


def order_for(cfg, epoch):
    return torch.randperm(cfg['records'], generator=torch.Generator().manual_seed(cfg['seed'] + epoch)).tolist()


def set_curriculum(model, optimizer, exposures):
    model.set_training_exposures(exposures)
    active = sum(exposures >= epoch * model.config.records_per_epoch for epoch in model.config.activation_epochs)
    for group in optimizer.param_groups:
        if 'temporal_band' in group:
            group['band_active'] = group['temporal_band'] < active
    return active


def objects(cfg, device='cpu'):
    from .paper4s_multiband_operator_20261021 import Paper4sConfig, build_from_parent
    with RNG_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg['seed'])
        model, lineage = build_from_parent(config=Paper4sConfig(**cfg['model']), device='cpu')
        model.to(device)
        groups = model.backbone.parameter_groups(base_lr=cfg['learning_rate'],
            refiner_lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        for i, head in enumerate(model.heads):
            groups.append(dict(params=list(head.parameters()), lr=cfg['learning_rate'],
                weight_decay=cfg['weight_decay'], temporal_band=i, band_active=i == 0,
                group_name=f'temporal_band_{i}'))
        optimizer = CurriculumAdamW(groups, foreach=False)
        controller = joint.make_learned(model, dict(block_learned_optimizer=cfg['controller']))
        meta = torch.optim.Adam(controller.parameters(), lr=cfg['meta_learning_rate'], foreach=False)
    set_curriculum(model, optimizer, 0)
    return model, optimizer, controller, meta, lineage


def initial_progress(cfg):
    return dict(epoch=0, offset=0, updates=0, exposures=0, order=order_for(cfg, 0),
                cuda_restore8=False, elapsed_seconds=0., epochs_finished=0)


def validate_progress(progress, cfg, runtime):
    n, target, batch = cfg['records'], cfg['epochs'], runtime['batch_size']
    for key in ('epoch', 'offset', 'updates', 'exposures', 'epochs_finished'):
        if type(progress.get(key)) is not int or progress[key] < 0:
            raise ValueError('nonnegative integer cursor required: ' + key)
    epoch, offset = progress['epoch'], progress['offset']
    if epoch > target or not 0 <= offset < n or (epoch == target and offset != 0):
        raise ValueError('epoch/offset exceeds authorized 500-epoch protocol')
    if progress['epochs_finished'] != epoch or progress['exposures'] != epoch * n + offset:
        raise ValueError('exact complete-roster exposure accounting differs')
    if offset % batch != 0:
        raise ValueError('partial epoch is not on a committed batch boundary')
    if progress['updates'] != epoch * math.ceil(n / batch) + offset // batch:
        raise ValueError('optimizer updates disagree with complete and partial epochs')
    if progress['order'] != order_for(cfg, epoch):
        raise ValueError('epoch shuffle/cursor mismatch')
    if (type(progress.get('cuda_restore8')) is not bool
            or (progress['updates'] > 8 and not progress['cuda_restore8'])
            or (progress['updates'] < 8 and progress['cuda_restore8'])):
        raise ValueError('actual CUDA8 restore must precede update9 checkpoint')
    if not math.isfinite(progress['elapsed_seconds']) or progress['elapsed_seconds'] < 0:
        raise ValueError('finite nonnegative elapsed time required')


def commit_progress(progress, cfg, runtime, batch_ids):
    validate_progress(progress, cfg, runtime)
    offset = progress['offset']
    expected = progress['order'][offset:offset + runtime['batch_size']]
    if list(batch_ids) != expected or not expected:
        raise ValueError('commit does not match frozen epoch order; no dropped/repeated tails')
    progress['updates'] += 1
    progress['exposures'] += len(expected)
    progress['offset'] += len(expected)
    if progress['offset'] == cfg['records']:
        progress['epoch'] += 1
        progress['epochs_finished'] = progress['epoch']
        progress['offset'] = 0
        progress['order'] = order_for(cfg, progress['epoch'])
    validate_progress(progress, cfg, runtime)


def optimizer_roster(module, optimizer):
    names = {id(p): n for n, p in module.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def immutable_digest(model):
    h = hashlib.sha256()
    values = [('buffer:' + n, b) for n, b in model.named_buffers() if n != 'training_exposures']
    values += [('frozen:' + n, p) for n, p in model.named_parameters() if not p.requires_grad]
    for name, value in values:
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str((array.dtype, array.shape)).encode()); h.update(array.tobytes())
    return h.hexdigest()


def payload(cfg, runtime, model, optimizer, controller, meta, lineage, progress, scales):
    validate_progress(progress, cfg, runtime)
    if immutable_digest(model) != cfg['immutable_digest']:
        raise ValueError('fixed model features/buffers changed')
    return dict(schema=SCHEMA, config=copy.deepcopy(cfg), config_digest=digest(cfg),
        runtime=copy.deepcopy(runtime), runtime_digest=digest(runtime),
        model=freeze(model.state_dict()), optimizer=freeze(optimizer.state_dict()),
        controller=freeze(controller.state_dict()), meta=freeze(meta.state_dict()),
        optimizer_roster=optimizer_roster(model, optimizer), meta_roster=optimizer_roster(controller, meta),
        lineage=freeze(lineage), progress=copy.deepcopy(progress), rng=rng_state(),
        teacher_scales=freeze(scales), field_times=freeze(model.full_times_s),
        immutable_digest=cfg['immutable_digest'],
        normalizer=copy.deepcopy(vars(model.normalizer.metadata)),
        optimizer_origin='fresh for the independently authorized 4s dataset and objective',
        transaction='pre-gradient, Adam proposal, learned scale, post-gradient, controller update, cursor committed')


def validate(value, cfg, runtime):
    if value.get('schema') != SCHEMA or value['config_digest'] != digest(cfg) or not same(value['config'], cfg):
        raise ValueError('checkpoint configuration identity mismatch')
    if value['runtime_digest'] != digest(runtime) or not same(value['runtime'], runtime):
        raise ValueError('selected batch/time-block identity mismatch')
    validate_progress(value['progress'], cfg, runtime)
    finite_tree(value)
    if int(value['model']['training_exposures']) != value['progress']['exposures']:
        raise ValueError('model exposure clock differs from committed cursor')
    if value['immutable_digest'] != cfg['immutable_digest']:
        raise ValueError('fixed-feature digest differs from registered configuration')
    updates = value['progress']['updates']
    epoch_updates = math.ceil(cfg['records'] / runtime['batch_size'])
    for label, roster in (('optimizer', value['optimizer_roster']), ('meta', value['meta_roster'])):
        state = value[label]
        if set(state) != {'state', 'param_groups'} or len(state['param_groups']) != len(roster):
            raise ValueError('complete optimizer group state required')
        covered = []
        for group, names in zip(state['param_groups'], roster):
            if len(group['params']) != len(names):
                raise ValueError('optimizer named roster mismatch')
            covered.extend(group['params'])
            index = group.get('temporal_band')
            age = updates if index is None else max(0, updates - cfg['model']['activation_epochs'][index] * epoch_updates)
            if index is not None:
                expected_active = value['progress']['exposures'] >= cfg['model']['activation_epochs'][index] * cfg['records']
                if group['band_active'] != expected_active:
                    raise ValueError('saved dormant-band group differs from exposure clock')
            parameters = value['model'] if label == 'optimizer' else value['controller']
            for param_id, name in zip(group['params'], names):
                parameter = parameters.get(name)
                if not torch.is_tensor(parameter):
                    raise ValueError('optimizer roster names an absent tensor')
                moment = state['state'].get(param_id)
                if moment is None:
                    if age:
                        raise ValueError('populated optimizer moment is missing')
                    continue
                if set(moment) != {'step', 'exp_avg', 'exp_avg_sq'} or float(moment['step']) != age:
                    raise ValueError('Adam moment fields or coordinate age differ')
                if any(moment[k].shape != parameter.shape or moment[k].dtype != parameter.dtype
                       for k in ('exp_avg', 'exp_avg_sq')):
                    raise ValueError('Adam moment shape/dtype differs from its named parameter')
                if bool((moment['exp_avg_sq'] < 0).any()):
                    raise ValueError('negative Adam second moment')
                if age == 0 and (bool(moment['exp_avg'].any()) or bool(moment['exp_avg_sq'].any())):
                    raise ValueError('dormant head acquired optimizer history')
        if len(set(covered)) != len(covered) or not set(state['state']) <= set(covered):
            raise ValueError('duplicated/unknown optimizer parameter ID')
    if not set(value['teacher_scales']) <= set(range(cfg['records'])):
        raise ValueError('teacher scales contain non-TRAIN records')
    visited = set(range(cfg['records'])) if value['progress']['epoch'] else set(value['progress']['order'][:value['progress']['offset']])
    if not visited <= set(value['teacher_scales']):
        raise ValueError('a committed TRAIN record lost its fixed teacher reference')
    if value['field_times'].shape != (801,) or value['field_times'].dtype != torch.float32:
        raise ValueError('exact raw801 physical time axis required')
    rng = value['rng']
    if set(rng) != {'python', 'numpy', 'torch', 'cuda'} or rng['torch'].dtype != torch.uint8:
        raise ValueError('complete RNG state required')
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    torch.Generator().set_state(rng['torch'])
    if rng['cuda'] is not None and (not isinstance(rng['cuda'], list) or
            any(v.dtype != torch.uint8 or v.ndim != 1 for v in rng['cuda'])):
        raise ValueError('complete CUDA RNG list required')


def restore(value, cfg, runtime, device='cpu'):
    validate(value, cfg, runtime)
    model, optimizer, controller, meta, lineage = objects(cfg, device)
    if not same(lineage, value['lineage']):
        raise ValueError('saved transfer lineage differs from the pinned parent')
    set_curriculum(model, optimizer, value['progress']['exposures'])
    for label, expected in (('optimizer', optimizer), ('meta', meta)):
        options = lambda g: {k: v for k, v in g.items() if k != 'params'}
        if not same([options(g) for g in expected.state_dict()['param_groups']],
                    [options(g) for g in value[label]['param_groups']]):
            raise ValueError('optimizer hyperparameters differ from registered configuration')
    model.load_state_dict(value['model'], strict=True)
    optimizer.load_state_dict(value['optimizer'])
    controller.load_state_dict(value['controller'], strict=True)
    meta.load_state_dict(value['meta'])
    set_curriculum(model, optimizer, value['progress']['exposures'])
    for object_, label in ((model, 'model'), (optimizer, 'optimizer'), (controller, 'controller'), (meta, 'meta')):
        if not same(freeze(object_.state_dict()), value[label]):
            raise ValueError('actual new-object restoration differs: ' + label)
    if optimizer_roster(model, optimizer) != value['optimizer_roster'] or optimizer_roster(controller, meta) != value['meta_roster']:
        raise ValueError('restored optimizer parameter identities differ')
    if not torch.equal(model.full_times_s.cpu(), value['field_times']):
        raise ValueError('physical time axis differs after restore')
    if not same(vars(model.normalizer.metadata), value['normalizer']):
        raise ValueError('inherited physical codec differs after restore')
    if immutable_digest(model) != cfg['immutable_digest']:
        raise ValueError('fixed features differ after complete restoration')
    return model, optimizer, controller, meta, lineage, copy.deepcopy(value['progress']), copy.deepcopy(value['teacher_scales'])


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class LatestStore:
    """CPU snapshot writes/readback occur off the GPU transaction path."""
    def __init__(self, run, cfg, runtime):
        self.run, self.cfg, self.runtime = Path(run), cfg, runtime
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='complete-checkpoint')
        self.pending = None

    def _save(self, value, actual_cpu_restore):
        validate(value, self.cfg, self.runtime)
        path = self.run / 'latest.pt'
        temporary = self.run / ('.latest.' + uuid.uuid4().hex + '.pt')
        torch.save(value, temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        check = torch.load(temporary, map_location='cpu', weights_only=False)
        validate(check, self.cfg, self.runtime)
        if not same(value, check):
            raise ValueError('complete checkpoint serialization/readback differs')
        if actual_cpu_restore:
            # Historical constructor factories seed global Python/NumPy/CUDA
            # generators. A separate CPU process isolates those side effects
            # from the live GPU worker and its scopes/module contexts.
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                PYTHONPATH=os.pathsep.join(sys.path), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            result = subprocess.run([sys.executable, '-B', '-m',
                'native400_runtime.paper4s_training_state_20261021', str(temporary)],
                env=environment, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('isolated CPU restoration failed: ' + result.stdout + result.stderr)
        receipt = dict(schema=SCHEMA, sha256=sha(temporary), updates=value['progress']['updates'],
            epoch=value['progress']['epoch'], exposures=value['progress']['exposures'],
            all_state_readback=True, actual_cpu_restore=actual_cpu_restore)
        write_json(self.run / 'CHECKPOINT_PENDING.json', receipt)
        os.replace(temporary, path)
        write_json(self.run / 'CHECKPOINT.json', receipt)
        (self.run / 'CHECKPOINT_PENDING.json').unlink()
        descriptor = os.open(str(self.run), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return receipt

    def save(self, value, actual_cpu_restore=False):
        self.flush()
        self.pending = self.pool.submit(self._save, value, actual_cpu_restore)

    def flush(self):
        if self.pending is None:
            return None
        result = self.pending.result()
        self.pending = None
        return result

    def load(self):
        self.flush()
        if (self.run / 'CHECKPOINT_PENDING.json').exists():
            raise ValueError('unresolved checkpoint publish transaction; explicit recovery required')
        receipt = json.loads((self.run / 'CHECKPOINT.json').read_text())
        path = self.run / 'latest.pt'
        if sha(path) != receipt['sha256']:
            raise ValueError('checkpoint bytes differ from committed receipt')
        value = torch.load(path, map_location='cpu', weights_only=False)
        validate(value, self.cfg, self.runtime)
        if value['progress']['updates'] != receipt['updates'] or value['progress']['exposures'] != receipt['exposures']:
            raise ValueError('checkpoint receipt cursor mismatch')
        return value

    def close(self):
        try:
            self.flush()
        finally:
            self.pool.shutdown(wait=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    restored = restore(checkpoint, checkpoint['config'], checkpoint['runtime'], device='cpu')
    if torch.cuda.is_initialized():
        raise AssertionError('isolated CPU checkpoint validation initialized CUDA')
    print(json.dumps(dict(status='PASS', updates=checkpoint['progress']['updates'], cuda_initialized=False)))
