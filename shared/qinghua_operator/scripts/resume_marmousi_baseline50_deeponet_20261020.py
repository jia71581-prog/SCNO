#!/usr/bin/env python3
"""Explicit bounded-host-memory continuation of the durable DeepONet step8.

Original mathematics/configuration/checkpoints are read-only. The controller
must reopen the same scientific task after FNO has naturally completed. CPU
preflight is available independently and never initializes CUDA.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
sys.dont_write_bytecode = True
import train_marmousi_baseline50_20261020 as w
old = w.old
PARENT_SHA = '203959e420276b7b20e379bb9b80a0934f8073020cac85a2ec6d7ecf9e392af6'
HOST_POLICY = dict(cache_records=64, prefetch_depth=1, prefetch_workers=1)


def process_matches(record):
    path = Path('/proc')/str(record['pid'])/'stat'
    try:
        parts = path.read_text().split(') ', 1)[1].split()
        return int(parts[19]) == int(record['start_ticks']) and parts[0] != 'Z'
    except FileNotFoundError:
        return False


def load_parent(spec):
    import torch
    if spec['parent_checkpoint_sha256'] != PARENT_SHA or old.sha(spec['parent_checkpoint_path']) != PARENT_SHA:
        raise ValueError('Only the explicitly verified complete parent step8 is authorized')
    for prefix in ('parent_config', 'parent_worker', 'parent_baseline', 'parent_epochs'):
        if old.sha(spec[prefix+'_path']) != spec[prefix+'_sha256']:
            raise ValueError(f'Immutable {prefix} changed')
    cfg = json.loads(Path(spec['parent_config_path']).read_text())
    saved = torch.load(spec['parent_checkpoint_path'], map_location='cpu', weights_only=False)
    if saved['config'] != cfg or saved['schema'] != w.SCHEMA or saved['kind'] != 'deeponet':
        raise ValueError('Parent complete-state original configuration identity mismatch')
    p = saved['progress']
    if (p['committed_updates'], p['completed_epochs'], p['active_epoch'], p['offset'], p['training_exposures']) != (8, 1, 2, 500, 1000):
        raise ValueError('Parent is not the durable end-of-epoch2 pre-bookkeeping cursor')
    if cfg['training']['epochs'] != 50 or cfg['training']['records_per_epoch'] != 500 or w.selection(cfg, 'deeponet') != (128, 524288):
        raise ValueError('No changes to authorized epochs, real batch, or complete-field chunking')
    if spec['runtime_policy'] != HOST_POLICY:
        raise ValueError('Only the reviewed bounded host-memory policy is authorized')
    w.validate_progress(p, cfg, 'deeponet')
    if w.bindings(cfg, spec['parent_config_path']) != saved['bindings']:
        raise ValueError('Original frozen source/data/probe bindings differ')
    recorded_identity = json.loads((Path(spec['parent_checkpoint_path']).parent/'WORKER.json').read_text())
    if recorded_identity != spec['parent_worker_identity']:
        raise ValueError('Original worker identity receipt changed')
    if process_matches(recorded_identity):
        raise ValueError('Original DeepONet worker still alive; concurrent writers forbidden')
    return cfg, saved


def resume_bindings(spec, config_path):
    if old.sha(spec['manifest_path']) != spec['manifest_sha256']:
        raise ValueError('Resume source manifest changed')
    files = json.loads(Path(spec['manifest_path']).read_text())['files']
    for name, expected in files.items():
        path = Path(name)
        if old.sha(path if path.is_absolute() else ROOT/path) != expected:
            raise ValueError(f'Resume source dependency changed: {name}')
    return dict(manifest_sha256=spec['manifest_sha256'], config_sha256=old.sha(config_path), files=files)


def dataset_with_policy(cfg, policy):
    from marmousi_baseline50_data_20261020 import Marmousi500Dataset
    data = dict(cfg['data'], cache_records=policy['cache_records'])
    return Marmousi500Dataset(dict(data, normalization=cfg['normalization'], pressure_rms_pa=cfg['pressure_rms_pa']))


def commit_finished_epoch(progress, cfg):
    """Bookkeeping only: no update, no exposure, and no tail-batch replay."""
    from marmousi_baseline50_data_20261020 import epoch_order
    w.validate_progress(progress, cfg, 'deeponet')
    if progress['offset'] != cfg['training']['records_per_epoch']:
        raise ValueError('Cannot commit an incompletely exposed epoch')
    result = copy.deepcopy(progress)
    result['completed_epochs'] += 1
    result['active_epoch'] = result['completed_epochs']+1
    result['offset'] = 0
    result['roster'] = [] if result['completed_epochs'] == 50 else list(map(int, epoch_order(500, cfg['training']['seed'], result['completed_epochs'])))
    w.validate_progress(result, cfg, 'deeponet')
    return result


def tensor_stream_sha(tensor, block_elements=1 << 20):
    """At most one small contiguous device-to-host block; no full CPU clone."""
    if not tensor.is_contiguous() or block_elements < 1:
        raise ValueError('Streaming witness requires contiguous tensors and positive chunks')
    digest = hashlib.sha256(json.dumps(dict(shape=list(tensor.shape), dtype=str(tensor.dtype)), sort_keys=True).encode())
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), block_elements):
        block = flat[start:start+block_elements].cpu()
        digest.update(block.numpy().tobytes())
        del block
    return digest.hexdigest()


def full_tail_witness(dataset, indices, cfg, device, trained_times):
    """All116 parent-tail records, every native400 field point, one at a time."""
    import torch
    result = []
    for index in indices:
        cpu = dataset.load_batch([index])
        data = w.to_device(cpu, cfg, device)
        if not torch.equal(data['times'], trained_times):
            raise ValueError('Restored native training times differ')
        sup = w.supervision(data['truth'], cfg)
        result.append(dict(index=int(index), inputs=tensor_stream_sha(data['inputs']),
                           truth=tensor_stream_sha(data['truth']), times=tensor_stream_sha(data['times']),
                           supervision={key:tensor_stream_sha(value) for key, value in sup.items()},
                           scale=float(cpu['scale'])))
        del cpu, data, sup
    return result


def cpu_preflight(config_path):
    import torch
    started = time.monotonic()
    torch.set_num_threads(2)
    spec = json.loads(Path(config_path).read_text())
    cfg, saved = load_parent(spec)
    model, optimizer, scheduler = w.restore_objects(cfg, 'deeponet', saved, 'cpu', restore_rng=False)
    dataset = dataset_with_policy(cfg, spec['runtime_policy'])
    if (not old.equal_tree(dataset.state_dict(), saved['data_state'])
            or not torch.equal(torch.from_numpy(dataset.times.copy()), saved['trained_times'])):
        raise ValueError('Host-only policy changed mathematical dataset state')
    if sorted({int(value['step']) for value in optimizer.state.values()}) != [8]:
        raise ValueError('Original Adam step ages not preserved')
    committed = commit_finished_epoch(saved['progress'], cfg)
    if (committed['committed_updates'], committed['training_exposures'], committed['completed_epochs']) != (8, 1000, 2):
        raise ValueError('Epoch2 bookkeeping replayed training')
    if torch.cuda.is_initialized():
        raise ValueError('CPU preflight unexpectedly initialized CUDA')
    out = Path(spec['output_dir']).resolve()
    if not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Only project run outputs allowed')
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='PASS', parent_checkpoint_sha256=PARENT_SHA,
                  parent_config_sha256=spec['parent_config_sha256'], resume_worker_sha256=old.sha(__file__),
                  original_model_Adam_scheduler_exact=True, optimizer_step_ages=[8], complete_roster_exact=True,
                  data_state_normalizer_native_time_binding_exact=True, original_RNG_payload_preserved=True,
                  epoch2_commit_without_optimizer_update=True, CUDA_initialized=False,
                  progress_before=saved['progress'], progress_after_epoch_bookkeeping=committed,
                  runtime_policy=spec['runtime_policy'], seconds=time.monotonic()-started)
    old.write_json(out/'RESUME_CPU_PREFLIGHT.json', report)
    return report


def require_fno_complete(cfg):
    run = Path(cfg['run_dir'])/'fno'
    result = json.loads((run/'RESULT.json').read_text())
    worker = json.loads((run/'WORKER.json').read_text())
    if process_matches(worker) or result['status'] != 'COMPLETE' or result['completed_epochs'] != 50 or result['training_exposures'] != 25000:
        raise ValueError('FNO must naturally complete and exit before the authorized continuation')


def worker(config_path):
    import math
    import numpy as np
    import torch
    from marmousi_baseline50_data_20261020 import Prefetch, epoch_batches
    from baselines.marmousi_matched_20261020.models import parameter_report
    spec = json.loads(Path(config_path).read_text())
    cfg, parent = load_parent(spec)
    require_fno_complete(cfg)
    pins = resume_bindings(spec, config_path)
    out = Path(spec['output_dir']).resolve()
    if not out.is_relative_to(ROOT/'runs') or out == Path(spec['parent_checkpoint_path']).resolve().parent:
        raise ValueError('Only the distinct authorized resume output directory is writable')
    preflight = json.loads((out/'RESUME_CPU_PREFLIGHT.json').read_text())
    if preflight['status'] != 'PASS' or preflight['parent_checkpoint_sha256'] != PARENT_SHA or preflight['resume_worker_sha256'] != old.sha(__file__):
        raise ValueError('Final-source actual complete CPU recovery preflight required')
    identity = dict(**old.identity(), kind='deeponet')
    with (out/'WORKER.json').open('x') as handle:
        json.dump(identity, handle, indent=2)
    started = time.monotonic()
    progress = old.cpu_tree(parent['progress'])
    original_training_seconds = progress['training_seconds']
    previous_elapsed = float(spec['prior_elapsed_seconds'])
    if not math.isfinite(previous_elapsed) or previous_elapsed < 0:
        raise ValueError('Finite last reported parent elapsed time required')
    device = cfg.get('device', 'cuda:0')
    train = cfg['training']
    batch_size, chunk = w.selection(cfg, 'deeponet')
    baseline = json.loads(Path(spec['parent_baseline_path']).read_text())
    inherited_epochs = Path(spec['parent_epochs_path']).read_bytes()
    epoch_rows = [json.loads(row) for row in inherited_epochs.splitlines() if row.strip()]
    if len(epoch_rows) != 1 or epoch_rows[0]['epoch'] != 1 or epoch_rows[0]['training_exposures'] != 500:
        raise ValueError('Exactly the original completed epoch1 must be inherited')
    latest_loss = None
    lineage = dict(source_checkpoint_sha256=PARENT_SHA, resumed_from_updates=8, resumed_from_exposures=1000,
                   source_run=str(Path(spec['parent_checkpoint_path']).parent), durable_cursor_end_epoch2=True,
                   inherited_epoch1_sha256=spec['parent_epochs_sha256'], original_config_sha256=spec['parent_config_sha256'],
                   source_config_sha256=spec['parent_config_sha256'],
                   failure_cause='SIGKILL observed; cause unproven', same_scientific_task=True,
                   prior_last_reported_elapsed_seconds=previous_elapsed, prior_elapsed_is_lower_bound=True,
                   original_training_seconds=original_training_seconds)
    def status(state):
        old.write_json(out/'STATUS.json', dict(status=state, utc=old.utc(), **identity, **progress,
                       target_epochs=50, target_exposures=25000, loss=latest_loss, batch_size=batch_size, chunk_points=chunk,
                       prior_last_reported_elapsed_seconds=previous_elapsed, elapsed_seconds_current_segment=time.monotonic()-started,
                       elapsed_seconds_lower_bound=previous_elapsed+time.monotonic()-started))
    def save():
        if resume_bindings(spec, config_path) != pins or w.bindings(cfg, spec['parent_config_path']) != parent['bindings']:
            raise ValueError('Original or continuation source/data/config drifted')
        w.validate_progress(progress, cfg, 'deeponet')
        payload = dict(schema=w.SCHEMA, kind='deeponet', config=cfg, progress=old.cpu_tree(progress),
                       model=old.cpu_tree(model.state_dict()), optimizer=old.cpu_tree(optimizer.state_dict()),
                       scheduler=old.cpu_tree(scheduler.state_dict()), rng=old.get_rng(device), data_state=dataset.state_dict(),
                       bindings=parent['bindings'], trained_times=trained_times.cpu().clone(), normalization=cfg['normalization'],
                       pressure_rms_pa=cfg['pressure_rms_pa'], resume_bindings=pins, runtime_policy=spec['runtime_policy'],
                       recovery_lineage=lineage)
        loaded = old.atomic_checkpoint(out/'latest.pt', payload)
        receipt = dict(utc=old.utc(), sha256=old.sha(out/'latest.pt'), committed_updates=progress['committed_updates'],
                       completed_epochs=progress['completed_epochs'], training_exposures=progress['training_exposures'], complete_readback=True)
        old.write_json(out/'CHECKPOINT.json', receipt)
        return loaded, receipt
    def cpu_restore(saved, receipt):
        cpu_model, cpu_opt, cpu_scheduler = w.restore_objects(cfg, 'deeponet', saved, 'cpu', restore_rng=False)
        result = dict(status='PASS', checkpoint_sha256=receipt['sha256'], committed_updates=progress['committed_updates'],
                      completed_epochs=progress['completed_epochs'], training_exposures=progress['training_exposures'],
                      model_populated_Adam_scheduler_cursor_roster_rng=True)
        del cpu_model, cpu_opt, cpu_scheduler
        old.set_rng(saved['rng'], device)
        return result
    def complete_epoch(epochs):
        nonlocal progress
        old_steps = progress['committed_updates']
        progress = commit_finished_epoch(progress, cfg)
        saved, receipt = save()
        prediction, metrics, protocol = w.assess_paper(model, 'deeponet', cfg, paper, trained_times, chunk)
        restore = cpu_restore(saved, receipt)
        old.write_json(out/'LATEST_CPU_RESTORE.json', restore)
        row = dict(epoch=progress['completed_epochs'], epoch_exposures=500,
                   training_exposures=progress['training_exposures'], committed_updates=progress['committed_updates'],
                   metrics=metrics, evaluation_time_protocol=protocol, checkpoint_sha256=receipt['sha256'], CPU_restore=restore,
                   inherited_precheckpoint_exposure_bookkeeping=(progress['completed_epochs'] == 2),
                   bookkeeping_optimizer_updates=progress['committed_updates']-old_steps)
        epochs.write(json.dumps(row, allow_nan=False)+'\n')
        epochs.flush()
        old.write_json(out/'LATEST_ASSESSMENT.json', row)
        del saved, prediction
    try:
        w.configure_runtime(cfg)
        status('RESTORING_PARENT')
        model, optimizer, scheduler = w.restore_objects(cfg, 'deeponet', parent, device)
        dataset = dataset_with_policy(cfg, spec['runtime_policy'])
        trained_times = parent['trained_times'].to(device)
        if not old.equal_tree(dataset.state_dict(), parent['data_state']):
            raise ValueError('Full original data bindings changed')
        paper = w.paper_data(cfg, device)
        old.write_json(out/'BASELINE.json', baseline)
        old.write_json(out/'SELECTION.json', parameter_report(model, cfg['target_parameters']))
        old.write_json(out/'RECOVERY_LINEAGE.json', lineage)
        # Stream all116 prior-tail records; release every one before the next.
        tail = progress['roster'][(progress['offset']-1)//batch_size*batch_size:progress['offset']]
        if len(tail) != 116:
            raise ValueError('Unexpected parent tail batch identity')
        before_data = full_tail_witness(dataset, tail, cfg, device, trained_times)
        before_prediction = w.predict(model, 'deeponet', paper['inputs'], paper['shape'], paper['times'], paper['norm'], paper['scale'], chunk)[0]
        saved8, receipt8 = save()
        model = optimizer = scheduler = None
        del dataset
        gc.collect()
        torch.cuda.empty_cache()
        model, optimizer, scheduler = w.restore_objects(cfg, 'deeponet', saved8, device)
        dataset = dataset_with_policy(cfg, spec['runtime_policy'])
        if not old.equal_tree(dataset.state_dict(), saved8['data_state']):
            raise ValueError('Actual CUDA8 new dataset state differs')
        after_data = full_tail_witness(dataset, tail, cfg, device, trained_times)
        after_prediction = w.predict(model, 'deeponet', paper['inputs'], paper['shape'], paper['times'], paper['norm'], paper['scale'], chunk)[0]
        if before_data != after_data or not np.array_equal(before_prediction, after_prediction) or not old.equal_tree(old.get_rng(device), saved8['rng']):
            raise ValueError('Actual CUDA8 complete data/prediction/RNG restore differs')
        restore8 = dict(status='RESTORED_AWAITING_UPDATE9', committed_updates=8, next_update=9,
                        parent_checkpoint_sha256=PARENT_SHA, checkpoint_sha256=receipt8['sha256'],
                        complete_state_exact=True, pressure_bitwise_equal=True, all116_tail_records_verified=True,
                        all400_frames_and201x201_nodes_streamed=True, per_record_witness=before_data,
                        maximum_host_tensor_witness_block_elements=1 << 20, old_model_and_dataset_released_before_rebuild=True)
        old.write_json(out/'CUDA_RESTORE8.json', restore8)
        del saved8, before_data, after_data, before_prediction, after_prediction
        # Original epoch1 is inherited byte-for-byte; the durable epoch2 tail is
        # already trained. Commit it now, without another optimizer call.
        with (out/'EPOCHS.jsonl').open('xb') as handle:
            handle.write(inherited_epochs)
        parent_metrics = Path(spec['parent_checkpoint_path']).parent/'metrics.jsonl'
        inherited_metrics = [json.loads(line) for line in parent_metrics.read_text().splitlines() if line.strip()]
        if [row['step'] for row in inherited_metrics] != list(range(1, 9)):
            raise ValueError('Parent metric history does not prove exactly the original8 updates')
        lineage.update(inherited_metrics_path=str(parent_metrics), inherited_metrics_sha256=old.sha(parent_metrics))
        old.write_json(out/'RECOVERY_LINEAGE.json', lineage)
        with (out/'metrics.jsonl').open('xb') as handle:
            pass  # New segment logs only9..200; immutable prior1..8 are SHA-bound.
        stopped = None
        with (out/'EPOCHS.jsonl').open('a', buffering=1) as epochs, (out/'metrics.jsonl').open('a', buffering=1) as log:
            complete_epoch(epochs)
            if progress['committed_updates'] != 8 or progress['training_exposures'] != 1000:
                raise ValueError('Recovery epoch2 bookkeeping replayed updates')
            while progress['completed_epochs'] < 50 and stopped is None:
                epoch = progress['completed_epochs']+1
                batches = list(epoch_batches(progress['roster'], batch_size, offset=progress['offset']))
                with Prefetch(dataset, batches, depth=1, workers=1) as prefetch:
                    for ids, cpu in zip(batches, prefetch):
                        if any(path.exists() for path in (out/'STOP', out/'STOP_REQUEST.json')):
                            stopped = 'explicit_resume_stop_requested'
                            break
                        if cpu['ids'] != ids:
                            raise ValueError('Bounded prefetch order differs from restored roster')
                        tick = time.monotonic()
                        data = w.to_device(cpu, cfg, device)
                        if not torch.equal(data['times'], trained_times):
                            raise ValueError('Native time axis changed')
                        sup = w.supervision(data['truth'], cfg)
                        optimizer.zero_grad(set_to_none=True)
                        losses = w.backward_complete(model, 'deeponet', data, sup, train['receiver_weight'], chunk)
                        if not w.gradients_finite(model) or not all(math.isfinite(value) for value in losses.values()):
                            raise ValueError('Nonfinite complete-batch loss/gradient')
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train['grad_clip'], error_if_nonfinite=True)
                        optimizer.step()
                        scheduler.step()
                        if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
                            raise ValueError('Nonfinite optimizer endpoint')
                        torch.cuda.synchronize(device)
                        seconds = time.monotonic()-tick
                        progress['offset'] += len(ids)
                        progress['committed_updates'] += 1
                        progress['training_exposures'] += len(ids)
                        progress['training_seconds'] += seconds
                        latest_loss = losses['loss']
                        row = dict(utc=old.utc(), epoch=epoch, offset=progress['offset'], step=progress['committed_updates'],
                                   training_exposures=progress['training_exposures'], batch_size=len(ids), seconds=seconds,
                                   samples_per_second=len(ids)/seconds, gradient_norm=float(norm),
                                   prefetch=cpu.get('prefetch'), resumed_segment=True, **losses)
                        log.write(json.dumps(row, allow_nan=False)+'\n')
                        print(json.dumps(row), flush=True)
                        del data, sup, cpu
                        if progress['committed_updates'] == 9:
                            restore8.update(status='PASS', restored_objects_committed_update9=True, utc=old.utc())
                            old.write_json(out/'CUDA_RESTORE8.json', restore8)
                        if progress['committed_updates'] % train['save_every_updates'] == 0:
                            save()
                        status('RUNNING')
                del prefetch
                if stopped is None:
                    complete_epoch(epochs)
                    status('RUNNING')
        if progress['completed_epochs'] == 50:
            saved = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
            receipt = json.loads((out/'CHECKPOINT.json').read_text())
        else:
            saved, receipt = save()
        prediction, metrics, protocol = w.assess_paper(model, 'deeponet', cfg, paper, trained_times, chunk)
        final_restore = cpu_restore(saved, receipt)
        old.write_json(out/'FINAL_CPU_RESTORE.json', final_restore)
        with (out/'.prediction_pa.npy.tmp').open('wb') as handle:
            np.save(handle, prediction, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(out/'.prediction_pa.npy.tmp', out/'prediction_pa.npy')
        complete = progress['completed_epochs'] == 50 and progress['training_exposures'] == 25000 and progress['committed_updates'] == 200
        lineage.update(additional_updates=progress['committed_updates']-8, additional_exposures=progress['training_exposures']-1000)
        result = dict(status='COMPLETE' if complete else 'INTERRUPTED', kind='deeponet', utc=old.utc(), **progress,
                      target_epochs=50, exposures=progress['training_exposures'], initialization='independent_random',
                      input_pressure_frames=0, batch_size=batch_size, chunk_points=chunk,
                      parameters=parameter_report(model, cfg['target_parameters']), initial_metrics=baseline['metrics'],
                      final_metrics=metrics, evaluation_time_protocol=protocol, checkpoint_sha256=receipt['sha256'],
                      prediction_sha256=old.sha(out/'prediction_pa.npy'), final_CPU_restore=final_restore,
                      CUDA_restore8=restore8, recovery_lineage=lineage, runtime_policy=spec['runtime_policy'], stop_reason=stopped,
                      elapsed_seconds_current_segment=time.monotonic()-started, prior_last_reported_elapsed_seconds=previous_elapsed,
                      elapsed_seconds_lower_bound=previous_elapsed+time.monotonic()-started,
                      training_objective='unchanged per-sample four-window DATA plus receiver, batch mean; no PDE/IC')
        old.write_json(out/'RESULT.json', result)
        status(result['status'])
        return 0 if complete else 2
    except BaseException as error:
        old.write_json(out/'ERROR.json', dict(status='FAILED', utc=old.utc(), **identity, **progress,
                       error=str(error), traceback=traceback.format_exc(), automatic_retry=False,
                       elapsed_seconds_current_segment=time.monotonic()-started))
        status('FAILED')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--prepare-cpu', action='store_true')
    args = parser.parse_args()
    if args.prepare_cpu:
        result = cpu_preflight(args.config)
        print(json.dumps({key:value for key, value in result.items() if not key.startswith('progress_')}))
        return 0
    return worker(args.config)


if __name__ == '__main__':
    raise SystemExit(main())
