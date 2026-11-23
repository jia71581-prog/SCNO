#!/usr/bin/env python3
"""Isolated, complete-field matched-parameter FNO/DeepONet instance workers.

Only the supplied run directory is writable. No controller, research task, or
GPU work is started by importing this module. The parent controller owns the
single scientific ledger. This worker never automatically retries or resumes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.dont_write_bytecode = True
SCHEMA = 'MARMOUSI_MATCHED_BASELINES_COMPLETE_STATE_V1'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 << 20), b''):
            result.update(part)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name('.' + path.name + '.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def identity():
    return dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]),
                start_time_utc=utc())


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def equal_tree(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def get_rng(device):
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None)


def set_rng(state, device):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if str(device).startswith('cuda') and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def verify_bindings(cfg, config_path=None):
    """A frozen small manifest binds data and the actual production closure."""
    paths = {}
    for key in ('manifest', 'inputs', 'reference'):
        path = Path(cfg[key + '_path']).resolve()
        expected = cfg[key + '_sha256']
        if sha(path) != expected:
            raise ValueError(f'{key} SHA mismatch: {path}')
        paths[str(path)] = expected
    manifest = json.loads(Path(cfg['manifest_path']).read_text())
    for name, expected in {**manifest['files'], **cfg.get('binding_files', {})}.items():
        path = Path(name)
        path = path if path.is_absolute() else ROOT / path
        if sha(path) != expected:
            raise ValueError(f'Frozen dependency changed: {path}')
        paths[str(path.resolve())] = expected
    if config_path is not None:
        paths[str(Path(config_path).resolve())] = sha(config_path)
    return paths


def load_data(cfg, device):
    import numpy as np
    import torch
    with np.load(cfg['inputs_path'], allow_pickle=False) as arrays:
        velocity = arrays['velocity_mps'].copy()
        source = arrays['source_model_fp32'].copy()
        times = arrays['time_s'].copy()
        scale = float(arrays['scale'])
    target = np.load(cfg['reference_path'], allow_pickle=False)
    if velocity.shape != (201, 201) or source.shape != (5,) or times.shape != (400,) or target.shape != (400, 201, 201):
        raise ValueError('Exact 201x201 / source5 / native400 paper data required')
    for name, array in [('velocity', velocity), ('source', source), ('time', times), ('reference', target)]:
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f'{name} must be finite, original FP32')
    if not (scale > 0 and math.isfinite(scale) and scale == cfg['pressure_rms_pa']):
        raise ValueError('Frozen pressure normalization differs')
    if not np.all(np.diff(times) > 0):
        raise ValueError('Strict original time order required')
    norm = cfg['normalization']
    if norm['velocity_std'] <= 0 or min(norm['source_std']) <= 0 or norm['time_scale_s'] <= 0:
        raise ValueError('Positive fixed feature normalization required')
    # The original 10 m grid spans 0..2000 m. No field interpolation.
    z = np.arange(201, dtype=np.float32) * np.float32(10. / norm['coord_extent_m'][1])
    x = np.arange(201, dtype=np.float32) * np.float32(10. / norm['coord_extent_m'][0])
    zz, xx = np.meshgrid(z, x, indexing='ij')
    normalized_source = (source - np.asarray(norm['source_mean'], dtype=np.float32)) / np.asarray(norm['source_std'], dtype=np.float32)
    channels = np.concatenate([((velocity - np.float32(norm['velocity_mean'])) / np.float32(norm['velocity_std']))[None],
                               np.broadcast_to(normalized_source[:, None, None], (5, 201, 201)), xx[None], zz[None]], axis=0)
    reference = torch.from_numpy(target.copy()).to(device)
    truth = reference / scale
    return dict(inputs=torch.from_numpy(channels.copy())[None].to(device), truth=truth,
                reference_pa=target, time_s=times, times=torch.from_numpy(times.copy()).to(device),
                scale=scale, shape=target.shape, norm=norm)


def fixed_supervision(truth, boundaries, receiver_indices):
    """Same .9 global + .1 relative-window DATA objective as paper fullfit."""
    import torch
    if truth.dtype != torch.float32 or boundaries[0] != 0 or boundaries[-1] != truth.shape[0]:
        raise ValueError('FP32 complete contiguous temporal coverage required')
    lengths = [b-a for a, b in zip(boundaries[:-1], boundaries[1:])]
    if len(lengths) != 4 or len(set(lengths)) != 1 or min(lengths) <= 0:
        raise ValueError('Original four equal windows required')
    frame_energy = truth.square().mean((-2, -1), dtype=torch.float32)
    energy = frame_energy.mean(dtype=torch.float32)
    shares = torch.stack([frame_energy[a:b].sum(dtype=torch.float32) for a, b in zip(boundaries[:-1], boundaries[1:])]) / frame_energy.sum(dtype=torch.float32)
    weights = .9 + .025 / shares.clamp_min(.001)
    frame_weights = weights.repeat_interleave(lengths[0])
    receiver = torch.tensor(receiver_indices, device=truth.device, dtype=torch.long)
    if receiver.ndim != 2 or receiver.shape[1] != 2 or receiver.shape[0] == 0:
        raise ValueError('Nonempty receiver [z,x] pairs required')
    if len(set(tuple(pair) for pair in receiver_indices)) != len(receiver_indices):
        raise ValueError('Receiver indices must be unique')
    receiver_energy = truth[:, receiver[:, 0], receiver[:, 1]].square().sum(dtype=torch.float32)
    if not bool(torch.isfinite(energy) & (energy > 0) & torch.isfinite(receiver_energy) & (receiver_energy > 0)):
        raise ValueError('Positive finite frozen field/receiver energy required')
    return dict(frame_weights=frame_weights.detach(), energy=energy.detach(), receiver=receiver,
                receiver_energy=receiver_energy.detach(), window_shares=shares.cpu().tolist(),
                window_weights=weights.cpu().tolist())


def full_loss(prediction, truth, supervision, receiver_weight):
    import torch
    error = (prediction - truth).square()
    field = (error * supervision['frame_weights'][:, None, None]).mean(dtype=torch.float32) / supervision['energy']
    rec = supervision['receiver']
    receiver = error[:, rec[:, 0], rec[:, 1]].sum(dtype=torch.float32) / supervision['receiver_energy']
    return field + receiver_weight * receiver, field, receiver


def point_coordinates(start, stop, shape, times, normalization):
    import torch
    _, height, width = shape
    ids = torch.arange(start, stop, dtype=torch.long, device=times.device)
    ti = torch.div(ids, height * width, rounding_mode='floor')
    spatial = ids.remainder(height * width)
    zi = torch.div(spatial, width, rounding_mode='floor')
    xi = spatial.remainder(width)
    coordinates = torch.stack((times[ti] / normalization['time_scale_s'],
                               xi.float() * (10. / normalization['coord_extent_m'][0]),
                               zi.float() * (10. / normalization['coord_extent_m'][1])), dim=-1)
    return coordinates, ti, zi, xi


def backward_complete(model, kind, inputs, truth, times, normalization, supervision, receiver_weight, chunk_points):
    """Every spacetime point contributes once; no point or time subsampling.

    DeepONet branch graph is evaluated once. A detached latent accumulates the
    exact sum of all chunk derivatives before one VJP through the branch.
    Each chunk has the original fixed full-domain denominator.
    """
    import torch
    if kind == 'fno':
        prediction = model(inputs)[0]
        loss, field, receiver = full_loss(prediction, truth, supervision, receiver_weight)
        loss.backward()
        return dict(loss=float(loss.detach()), field_loss=float(field.detach()), receiver_loss=float(receiver.detach()))
    branch = model.encode_branch(inputs)
    surrogate = branch.detach().requires_grad_(True)
    totals = torch.zeros(3, device=truth.device, dtype=torch.float32)
    rec = supervision['receiver']
    total = truth.numel()
    for start in range(0, total, chunk_points):
        stop = min(start + chunk_points, total)
        coordinates, ti, zi, xi = point_coordinates(start, stop, truth.shape, times, normalization)
        prediction = model.decode_trunk(surrogate, coordinates)[0]
        error = (prediction - truth.reshape(-1)[start:stop]).square()
        field = (error * supervision['frame_weights'][ti]).sum(dtype=torch.float32) / (supervision['energy'] * total)
        receiver_mask = ((zi[:, None] == rec[None, :, 0]) & (xi[:, None] == rec[None, :, 1])).any(dim=-1)
        receiver = error[receiver_mask].sum(dtype=torch.float32) / supervision['receiver_energy']
        loss = field + receiver_weight * receiver
        loss.backward()
        totals += torch.stack((loss.detach(), field.detach(), receiver.detach()))
    if surrogate.grad is None:
        raise ValueError('Missing complete branch gradient')
    branch.backward(surrogate.grad)
    return dict(zip(('loss', 'field_loss', 'receiver_loss'), (float(value) for value in totals)))


def predict_complete(model, kind, data, chunk_points):
    import torch
    old_training = model.training
    model.eval()
    with torch.no_grad():
        if kind == 'fno':
            prediction = model(data['inputs'])[0]
        else:
            branch = model.encode_branch(data['inputs'])
            prediction = torch.empty(data['shape'], device=data['inputs'].device, dtype=torch.float32)
            for start in range(0, prediction.numel(), chunk_points):
                stop = min(start + chunk_points, prediction.numel())
                coords, _, _, _ = point_coordinates(start, stop, data['shape'], data['times'], data['norm'])
                prediction.reshape(-1)[start:stop] = model.decode_trunk(branch, coords)[0]
        # Physical scaling itself is FP32; FP64 compatibility starts from Pa.
        result = (prediction * data['scale']).cpu().numpy()
    model.train(old_training)
    return result


def score_pa(prediction, reference, receiver_indices):
    import numpy as np
    if prediction.dtype != np.float32 or reference.dtype != np.float32 or prediction.shape != reference.shape:
        raise ValueError('Matched FP32 physical Pa fields required')
    result = {}
    z, x = np.asarray(receiver_indices, dtype=np.int64).T
    for label, dtype in [('fp32', np.float32), ('fp64', np.float64)]:
        p, y = prediction.astype(dtype), reference.astype(dtype)
        if not np.isfinite(p).all():
            raise ValueError('Nonfinite prediction')
        error, energy = (p-y)**2, y*y
        result[label] = {}
        for name, selection in [('full400', slice(None)), ('forecast392', slice(8, None)),
                                ('late130', slice(270, None)), ('late98', slice(302, None))]:
            sse, den = error[selection].sum(dtype=dtype), energy[selection].sum(dtype=dtype)
            if not (np.isfinite(sse) and np.isfinite(den) and den > 0):
                raise ValueError(f'Invalid {label}/{name} metric')
            result[label][name] = dict(sse_Pa2=float(sse), truth_energy_Pa2=float(den), relative_L2=float(np.sqrt(sse / den)))
        sse, den = error[:, z, x].sum(dtype=dtype), energy[:, z, x].sum(dtype=dtype)
        if not (np.isfinite(sse) and np.isfinite(den) and den > 0):
            raise ValueError('Invalid receiver metric')
        result[label]['receiver400'] = dict(sse_Pa2=float(sse), truth_energy_Pa2=float(den), relative_L2=float(np.sqrt(sse / den)))
    return result


def build_objects(cfg, kind, device):
    import torch
    from baselines.marmousi_matched_20261020.models import build_model
    settings = dict(cfg['model'][kind])
    settings.setdefault('target_parameters', cfg.get('target_parameters', 14552439))
    model = build_model(kind, **settings).to(device)
    train = cfg['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=train['lr'], weight_decay=train['weight_decay'])
    def factor(step):
        warmup = train['warmup_steps']
        if warmup and step < warmup:
            return float(step + 1) / warmup
        fraction = min(1., max(0., (step - warmup) / max(1, train['steps'] - warmup)))
        return train['min_lr_ratio'] + (1. - train['min_lr_ratio']) * .5 * (1. + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return model, optimizer, scheduler


def restore_objects(cfg, kind, payload, device, restore_rng=True):
    if payload['schema'] != SCHEMA or payload['kind'] != kind or payload['config'] != cfg:
        raise ValueError('Complete checkpoint configuration identity differs')
    model, optimizer, scheduler = build_objects(cfg, kind, device)
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    if not (equal_tree(payload['model'], model.state_dict()) and equal_tree(payload['optimizer'], optimizer.state_dict())
            and equal_tree(payload['scheduler'], scheduler.state_dict())):
        raise ValueError('Reconstructed complete model/optimizer/scheduler state differs')
    if restore_rng:
        set_rng(payload['rng'], device)
    return model, optimizer, scheduler


def atomic_checkpoint(path, payload):
    import torch
    temp = path.with_name('.' + path.name + '.tmp')
    with temp.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    if not equal_tree(payload, loaded):
        raise ValueError('Complete atomic checkpoint readback differs')
    return loaded


def worker(config_path, kind):
    import numpy as np
    import torch
    from baselines.marmousi_matched_20261020.models import parameter_report
    cfg = json.loads(Path(config_path).read_text())
    run = Path(cfg['run_dir']).resolve() / kind
    if not run.is_relative_to(ROOT / 'runs'):
        raise ValueError('Worker writes only below this project runs/')
    run.mkdir(parents=True, exist_ok=True)
    # The exclusive marker also rejects a failed prior attempt. Resumption is a
    # separate explicitly reviewed operation; not a silent new start.
    with (run / 'WORKER.json').open('x') as handle:
        json.dump(dict(**identity(), kind=kind, config_path=str(Path(config_path).resolve())), handle, indent=2)
    birth = time.monotonic()
    committed = 0
    model = optimizer = scheduler = None
    data = supervision = None
    latest_loss = None
    train_seconds = 0.
    config_sha = sha(config_path)
    train = cfg['training']
    device = cfg.get('device', 'cuda:0')
    if not str(device).startswith('cuda'):
        raise ValueError('Production worker requires CUDA; CPU correctness checks use imported functions')
    info = json.loads((run / 'WORKER.json').read_text())
    def status(state, **extra):
        write_json(run / 'STATUS.json', dict(status=state, utc=utc(), **info, committed_updates=committed,
                   target_updates=train['steps'], elapsed_seconds=time.monotonic()-birth,
                   training_seconds=train_seconds, loss=latest_loss, **extra))
    def save():
        if sha(config_path) != config_sha:
            raise ValueError('Frozen config changed')
        verify_bindings(cfg)
        state = dict(schema=SCHEMA, kind=kind, config=cfg, config_sha256=config_sha,
                     model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                     scheduler=cpu_tree(scheduler.state_dict()), rng=get_rng(device),
                     committed_updates=committed, next_update=committed+1, training_seconds=train_seconds,
                     elapsed_seconds=time.monotonic()-birth, pressure_rms_pa=data['scale'],
                     normalization=cfg['normalization'], original_time_s=data['time_s'].copy(),
                     supervision=cpu_tree(supervision), data_bindings=bindings)
        loaded = atomic_checkpoint(run / 'latest.pt', state)
        write_json(run / 'CHECKPOINT.json', dict(utc=utc(), sha256=sha(run / 'latest.pt'),
                   committed_updates=committed, next_update=committed+1, complete_readback=True))
        return loaded
    try:
        status('VALIDATING')
        bindings = verify_bindings(cfg, config_path)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(train.get('cpu_threads', 4))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        random.seed(train['seed'])
        np.random.seed(train['seed'])
        torch.manual_seed(train['seed'])
        torch.cuda.manual_seed_all(train['seed'])
        model, optimizer, scheduler = build_objects(cfg, kind, device)
        data = load_data(cfg, device)
        supervision = fixed_supervision(data['truth'], train['data_window_boundaries'], train['receiver_indices'])
        selection = parameter_report(model, cfg['target_parameters'])
        if not selection['exact_match'] or selection['total_real_scalars'] != selection['trainable_real_scalars']:
            raise ValueError('Every baseline parameter must be trainable and exactly match the frozen target')
        write_json(run / 'SELECTION.json', selection)
        save()
        prediction = predict_complete(model, kind, data, train['chunk_points'])
        initial = score_pa(prediction, data['reference_pa'], train['receiver_indices'])
        write_json(run / 'BASELINE.json', dict(step=0, utc=utc(), metrics=initial))
        del prediction
        restore8 = None
        stop_reason = None
        with (run / 'metrics.jsonl').open('x', buffering=1) as log:
            for next_update in range(1, train['steps'] + 1):
                if any(path.exists() for path in (run/'STOP', run/'STOP_REQUEST.json', run.parent/'STOP_REQUEST.json')):
                    stop_reason = 'own_run_stop_requested'
                    break
                if time.monotonic() - birth >= train['wall_cap_seconds']:
                    stop_reason = 'wall_cap_reached'
                    break
                tick = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                loss = backward_complete(model, kind, data['inputs'], data['truth'], data['times'], data['norm'],
                                         supervision, train['receiver_weight'], train['chunk_points'])
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                if any(g is None for g in gradients):
                    raise ValueError('Missing trainable parameter gradient')
                if not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
                    raise ValueError('Nonfinite trainable gradient')
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
                    raise ValueError('Nonfinite optimizer endpoint')
                torch.cuda.synchronize(device)
                committed = next_update
                elapsed = time.monotonic() - tick
                train_seconds += elapsed
                latest_loss = loss['loss']
                row = dict(utc=utc(), step=committed, seconds=elapsed, training_seconds=train_seconds,
                           gradient_norm=float(norm), learning_rate_next=optimizer.param_groups[0]['lr'],
                           all400_frames_supervised=True, all201x201_nodes_supervised=True, **loss)
                log.write(json.dumps(row, allow_nan=False) + '\n')
                status('RUNNING')
                if committed == 1 or committed % 20 == 0:
                    print(json.dumps(row), flush=True)
                if committed == 8:
                    saved = save()
                    witness = predict_complete(model, kind, data, train['chunk_points'])
                    model = optimizer = scheduler = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    model, optimizer, scheduler = restore_objects(cfg, kind, saved, device)
                    restored_data = load_data(cfg, device)
                    restored_supervision = fixed_supervision(restored_data['truth'], train['data_window_boundaries'], train['receiver_indices'])
                    if not (equal_tree(saved['original_time_s'], restored_data['time_s'])
                            and equal_tree(saved['supervision'], restored_supervision)
                            and saved['normalization'] == restored_data['norm']
                            and saved['pressure_rms_pa'] == restored_data['scale']
                            and saved['data_bindings'] == verify_bindings(cfg, config_path)):
                        raise ValueError('CUDA8 restored data/time/normalizer/supervision bindings differ')
                    data, supervision = restored_data, restored_supervision
                    after = predict_complete(model, kind, data, train['chunk_points'])
                    if not np.array_equal(witness, after) or saved['next_update'] != 9:
                        raise ValueError('Actual CUDA8 new-object pressure/next-update restore differs')
                    if not equal_tree(get_rng(device), saved['rng']):
                        raise ValueError('CUDA8 restored RNG differs')
                    restore8 = dict(status='RESTORED_AWAITING_UPDATE9', checkpoint_sha256=sha(run/'latest.pt'),
                                    committed_updates=8, next_update=9, pressure_bitwise_equal=True,
                                    model_optimizer_scheduler_rng_exact=True,
                                    data_normalizer_original_times_supervision_bindings_exact=True)
                    write_json(run / 'CUDA_RESTORE8.json', restore8)
                    del saved, witness, after
                if committed == 9 and restore8 is not None:
                    restore8.update(status='PASS', restored_objects_committed_update9=True, utc=utc())
                    write_json(run / 'CUDA_RESTORE8.json', restore8)
                if committed % train['save_every'] == 0:
                    save()
                if committed % train['eval_every'] == 0:
                    prediction = predict_complete(model, kind, data, train['chunk_points'])
                    assessment = dict(utc=utc(), step=committed, metrics=score_pa(prediction, data['reference_pa'], train['receiver_indices']))
                    with (run / 'assessments.jsonl').open('a') as handle:
                        handle.write(json.dumps(assessment, allow_nan=False) + '\n')
                    write_json(run / 'LATEST_ASSESSMENT.json', assessment)
                    del prediction
        saved = save()
        prediction = predict_complete(model, kind, data, train['chunk_points'])
        final_metrics = score_pa(prediction, data['reference_pa'], train['receiver_indices'])
        prediction_temp = run / '.prediction_pa.npy.tmp'
        with prediction_temp.open('wb') as handle:
            np.save(handle, prediction, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(prediction_temp, run / 'prediction_pa.npy')
        cpu_model, cpu_opt, cpu_scheduler = restore_objects(cfg, kind, saved, 'cpu', restore_rng=False)
        cpu_restore = dict(status='PASS', complete_model_optimizer_scheduler=True,
                           populated_optimizer=len(cpu_opt.state) > 0, committed_updates=committed,
                           checkpoint_sha256=sha(run / 'latest.pt'))
        write_json(run / 'FINAL_CPU_RESTORE.json', cpu_restore)
        del cpu_model, cpu_opt, cpu_scheduler
        finished = committed == train['steps']
        result = dict(status='COMPLETE' if finished else 'INTERRUPTED', utc=utc(), kind=kind,
                      committed_updates=committed, target_updates=train['steps'], stop_reason=stop_reason,
                      elapsed_seconds=time.monotonic()-birth, training_seconds=train_seconds,
                      initialization='independent_random', input_pressure_frames=0,
                      exact_time_axis_sha256=hashlib.sha256(data['time_s'].tobytes()).hexdigest(),
                      DATA_objective='same paper .9 global + .1 four-window relative energy mix',
                      receiver_weight=train['receiver_weight'], PDE_or_IC_training_terms=False,
                      comparison_scope='supervised single-instance fit; not heldout generalization; pretraining histories differ',
                      initial_metrics=initial, final_metrics=final_metrics, CUDA_restore8=restore8,
                      final_CPU_restore=cpu_restore, parameters=parameter_report(model, cfg['target_parameters']),
                      checkpoint_sha256=sha(run / 'latest.pt'), prediction_sha256=sha(run / 'prediction_pa.npy'))
        write_json(run / 'RESULT.json', result)
        status(result['status'])
        return 0 if finished else 2
    except BaseException as error:
        write_json(run / 'ERROR.json', dict(status='FAILED', utc=utc(), **info, committed_updates=committed,
                   elapsed_seconds=time.monotonic()-birth, training_seconds=train_seconds,
                   error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                   automatic_retry=False, checkpoint_may_lag_committed_updates=True))
        status('FAILED')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--kind', required=True, choices=('fno', 'deeponet'))
    args = parser.parse_args()
    return worker(args.config, args.kind)


if __name__ == '__main__':
    raise SystemExit(main())
