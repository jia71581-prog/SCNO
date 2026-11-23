"""Warm-started 200-epoch homogeneous/layered SCNO with sealed full-axis validation."""

import argparse
from collections import deque
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import sys
from time import perf_counter, sleep, time

import numpy as np
import torch

from data_v2 import open_corpus, _write_json
from checkpoint_weight_archive import CheckpointWeightArchive
from full_model_warm_start import FourStratumFrameSampler
from joint_fno_cache import RecordCache
from joint_fno_records import Trajectories
from method_repair_model import CorrectedSCNOReadout
from model_v2 import build_model
from predict_v2 import load_model
from readout_conditioning_repair import absorb_neural_gain
from repair_stage_schedule import StageController, capture_rng, restore_rng
from ul_backbone_lr_repair import BackboneRateController, migrate_stage_controller


def emit(stage, **fields):
    print(json.dumps({'stage': stage, 'time_unix': time(), **fields}, allow_nan=False), flush=True)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_atomic(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def role_partition(model):
    roles = {'BASE': [], 'DETAIL': [], 'FROZEN': []}
    fixed = {'parent.residual_gain', 'parent.source_encoder.ic_encoder.0.weight'}
    for name, parameter in model.named_parameters():
        if name in fixed:
            role = 'FROZEN'
        elif name.startswith(('parent.decoder.detail_', 'detail_head.')):
            role = 'DETAIL'
        else:
            role = 'BASE'
        roles[role].append((name, parameter))
    return roles


def resident_pressure_loss(model, inputs, frozen_details=None):
    velocity, source, reference, indices, weights, denominator = inputs
    if frozen_details is None:
        prediction = model(velocity, source, indices, normalized=True)
    else:
        context = model.encode(velocity, source)
        base = model.parent.decode(context, indices, normalized=True, detail_enabled=False)
        details = frozen_details.packets(context.features, indices)
        correction = model.synthesize_correction(None, details, batch=len(source),
            count=indices.shape[-1], device_tensor=context.features)
        prediction = base + correction * (source[:, 4] != 0)[:, None, None, None]
    frame_sse = (prediction - reference).square().flatten(2).sum(2)
    return ((frame_sse.double() * weights.double()).sum(1) / denominator).mean()


def batch_loss(model, rows, sampler, energy_table, frozen_details=None):
    velocity = torch.cat([r['velocity'] for r in rows])
    source = torch.cat([r['source'] for r in rows])
    sample = sampler.sample(len(rows), device=model.device)
    reference = torch.stack([r['target'].index_select(0, indices)
                             for r, indices in zip(rows, sample.indices)])
    for row in rows:
        if row['id'] not in energy_table:
            energy_table[row['id']] = float(row['target'].double().square().sum())
    denominator = torch.tensor([energy_table[r['id']] for r in rows], device=model.device,
                               dtype=torch.float64).clamp_min(1e-12)
    return resident_pressure_loss(model, (velocity, source, reference, sample.indices, sample.weights, denominator),
                                  frozen_details)


def backward_attempt(model, cache, keys, microbatch, sampler, energy_table, frozen_details=None):
    """Release failed graphs before returning; optimizer updates are outside."""
    try:
        value = 0.
        for offset in range(0, len(keys), microbatch):
            subset = keys[offset:offset + microbatch]
            rows = [cache.get(key) for key in subset]
            loss = batch_loss(model, rows, sampler, energy_table, frozen_details)
            (loss * (len(subset) / len(keys))).backward()
            value += float(loss.detach()) * len(subset) / len(keys)
            del loss, rows
        return value
    except torch.cuda.OutOfMemoryError:
        return None


def backward_resident_attempt(model, inputs, microbatch, frozen_details=None):
    """Reuse exactly the consumed sampling plan on every OOM attempt."""
    try:
        batch = len(inputs[1])
        value = 0.
        for offset in range(0, batch, microbatch):
            selected = tuple(tensor[offset:offset + microbatch] for tensor in inputs)
            weight = len(selected[1]) / batch
            loss = resident_pressure_loss(model, selected, frozen_details)
            (loss * weight).backward()
            value += float(loss.detach()) * weight
            del loss
        return value
    except torch.cuda.OutOfMemoryError:
        return None


def trim_gpu_cache(cache, budget):
    cache.gpu_budget = max(0, int(budget))
    with cache._lock:
        while cache._gpu and cache._gpu_bytes > cache.gpu_budget:
            cache._evict_gpu(demote=True)


def tune_batch(model, cache, train_ids, sampler, energy_table, candidates, out):
    """Probe real joint updates, then restore all weights and random state."""
    saved_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    rng, sampler_state = capture_rng(), deepcopy(sampler.state_dict())
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    roles = role_partition(model)
    for role, pairs in roles.items():
        for _, parameter in pairs:
            parameter.requires_grad_(role != 'FROZEN')
    model.train()
    trials = []
    free, total = torch.cuda.mem_get_info(model.device)
    process_budget = min(int(total * .95), free + torch.cuda.memory_reserved(model.device))
    started = perf_counter()
    for count in candidates:
        trial = {'records': count, 'fits': False}
        optimizer = None
        try:
            rows = [cache.get(key) for key in train_ids[:count]]
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            torch.cuda.reset_peak_memory_stats(model.device)
            seconds = []
            for _ in range(2):
                torch.cuda.synchronize(model.device)
                tick = perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = batch_loss(model, rows, sampler, energy_table)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize(model.device)
                seconds.append(perf_counter() - tick)
                del loss
            peak = torch.cuda.max_memory_allocated(model.device)
            total = torch.cuda.get_device_properties(model.device).total_memory
            trial.update(fits=peak < process_budget - (1 << 30), peak_bytes=peak,
                         steady_update_seconds=seconds[-1], records_per_second=count / seconds[-1])
        except torch.cuda.OutOfMemoryError:
            trial['out_of_memory'] = True
        finally:
            if 'loss' in locals():
                del loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            optimizer = None
            if 'rows' in locals():
                del rows
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            model.load_state_dict(saved_state, strict=True)
        trials.append(trial)
        emit('batch_probe', **trial)
        if trial.get('out_of_memory'):
            break
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(flags[name])
    sampler.load_state_dict(sampler_state)
    restore_rng(rng)
    valid = [r for r in trials if r['fits']]
    if not valid:
        raise RuntimeError('no joint-training batch passed the measured memory budget')
    best_rate = max(r['records_per_second'] for r in valid)
    selected = max((r for r in valid if r['records_per_second'] >= .96 * best_rate),
                   key=lambda r: r['records'])
    total = torch.cuda.get_device_properties(model.device).total_memory
    resident_rows = selected['records'] * ((401 + 1) * 201 * 201 + 5) * 4
    gpu_cache_bytes = max(0, process_budget - selected['peak_bytes'] - (2 << 30) + resident_rows)
    result = {'trials': trials, 'selected': selected, 'gpu_cache_bytes': gpu_cache_bytes,
              'seconds': perf_counter() - started, 'model_and_rng_restored': True,
              'observed_process_memory_budget_bytes': process_budget,
              'precision': 'strict_fp32', 'sampled_frames_per_record': 256}
    _write_json(out / 'BATCH_TUNING.json', result)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--parent-config', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--candidates', nargs='+', type=int, default=[1, 2, 4, 8, 12, 16])
    p.add_argument('--cpu-cache-gib', type=float, default=72.)
    p.add_argument('--eval-every', type=int, default=2)
    p.add_argument('--seed', type=int, default=261119)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-updates', type=int, default=0)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--backbone-lr-factor', type=float, default=.1)
    p.add_argument('--repair-learning-rate', action='store_true')
    p.add_argument('--frozen-detail-fold', action='store_true')
    p.add_argument('--sampled-prefetch', action='store_true')
    p.add_argument('--fast-polynomial', action='store_true')
    p.add_argument('--joint-from-epoch', type=int, choices=(76,))
    p.add_argument('--until-epoch', type=int, help='stop a bound short trial after this complete epoch')
    return p.parse_args()


def main():
    args = parse_args()
    if args.epochs != 200 or args.eval_every < 1 or any(n < 1 for n in args.candidates):
        raise ValueError('this run uses exactly 200 additional epochs and positive batch/evaluation intervals')
    if args.until_epoch is not None and (not args.resume or not 1 <= args.until_epoch < args.epochs):
        raise ValueError('only a bootstrapped short trial can specify an earlier complete epoch')
    end_epoch = args.until_epoch or args.epochs
    if not torch.cuda.is_available():
        raise RuntimeError('the long run requires CUDA')
    if args.sampled_prefetch and args.cache_dir is None:
        raise ValueError('sampled prefetch requires the sealed completed mmap cache')
    import fcntl
    args.out.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.out / 'RUN.lock').open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint_path = args.out / 'checkpoint.pt'
    if checkpoint_path.exists() != args.resume:
        raise ValueError('resume the existing bound run or choose an unused output directory')
    if args.joint_from_epoch is not None and (not args.resume or args.repair_learning_rate):
        raise ValueError('bootstrapped epoch-76 branches use normal resume')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.96)
    device = torch.device('cuda:0')
    old_config = json.loads(args.parent_config.read_text())
    parent_corpus = open_corpus(old_config)
    from prepare_ul_warm_protocol import load_ul_warm_protocol
    corpus = load_ul_warm_protocol(parent_corpus, args.protocol)
    if (len(corpus.train_ids) != 900 or len(corpus.test_ids) != 100 or len(corpus.validation_ids) < 6
            or len(corpus.times) != 401):
        raise ValueError('require sealed 900 TRAIN / 100 prior-exposed reserved records and >=6 new VAL trajectories')
    run_settings = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
                    if name not in ('resume', 'smoke_updates', 'cache_dir', 'repair_learning_rate',
                                    'frozen_detail_fold', 'sampled_prefetch', 'fast_polynomial') and value is not None}
    binding = {'settings': run_settings, 'parent_sha256': file_hash(args.parent),
               'dataset_binding': corpus.binding}
    binding_sha = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    start_time = time()
    if args.resume:
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if saved['run_binding_sha256'] != binding_sha:
            legacy_settings = dict(run_settings)
            legacy_settings.pop('backbone_lr_factor')
            legacy_binding = {**binding, 'settings': legacy_settings}
            legacy_sha = hashlib.sha256(json.dumps(legacy_binding, sort_keys=True).encode()).hexdigest()
            if not args.repair_learning_rate or saved['run_binding_sha256'] != legacy_sha:
                raise ValueError('resume data, initialization checkpoint or execution settings changed')
        elif args.repair_learning_rate:
            raise ValueError('this checkpoint already contains the LR intervention; use normal resume')
        model = CorrectedSCNOReadout.from_checkpoint(build_model(old_config, parent_corpus, torch.device('cpu')),
                                                    saved).to(device)
        start_time = saved['started_at_unix']
    else:
        parent = load_model(old_config, args.parent, parent_corpus, device, False).eval()
        store0 = Trajectories(corpus)
        row = store0.read(corpus.train_ids[0])
        velocity, source = row['velocity'].to(device), row['source'].to(device)
        with torch.no_grad():
            reference = parent(velocity, source)
            gain_receipt = absorb_neural_gain(parent, startup_mode='warm_start')
            model = CorrectedSCNOReadout(parent, nonlinear_details=False, coarsest_base=False,
                                         basis_size=800, base_hidden_width=800, time_chunk=401).eval()
            candidate = model(velocity, source)
            relative = float((candidate - reference).double().norm() / reference.double().norm())
        if relative > 2e-5:
            raise ValueError('warm expansion changed the inherited complete pressure prediction')
        _write_json(args.out / 'INITIALIZATION.json', {
            'parent': str(args.parent), 'parent_sha256': binding['parent_sha256'],
            'transition': 'compatible_warm_start_new_optimizer_200_additional_epochs',
            'loaded_original_parameters': True, 'original_base_columns_and_8_detail_bases_preserved': True,
            'gain_absorption': gain_receipt, 'basis_expansion': model.basis_expansion_receipt,
            'base_expansion': model.base_expansion_receipt,
            'complete_401_frame_prediction_relative_difference': relative})
        del reference, candidate, row, velocity, source, store0
    model.train()
    transform_policy = None
    if args.fast_polynomial:
        from fast_polynomial_transform import install_fast_polynomial
        transform_policy = install_fast_polynomial(model.detail_transform)
    store = Trajectories(corpus, args.cache_dir)
    cache = RecordCache(store, device, 0, int(args.cpu_cache_gib * (1 << 30)), compression='none',
                        prefetch_workers=4, prefetch_depth=32, parallel_reads=True, host_floor_bytes=4 << 30)
    sampler = FourStratumFrameSampler(401, 64, seed=args.seed)
    energy_table = dict(saved.get('energy_table', {})) if args.resume else {}
    if args.resume:
        tuning = saved['batch_tuning']
        sampler.load_state_dict(saved['frame_sampler'])
    else:
        tuning = tune_batch(model, cache, list(corpus.train_ids), sampler, energy_table, args.candidates, args.out)
    batch_size = tuning['selected']['records']
    microbatch = saved.get('microbatch', batch_size) if args.resume else batch_size
    cache.gpu_budget = tuning['gpu_cache_bytes']
    config = deepcopy(old_config)
    config.update(seed=args.seed, protocol='staged', affine_calibration=True,
                  affine_calibration_records=32, affine_calibration_times_per_record=8)
    control_kwargs = dict(epochs=200, updates_per_epoch=math.ceil(900 / batch_size),
                          named_roles=role_partition(model), train_ids=corpus.train_ids,
                          test_ids=corpus.test_ids, validation_ids=corpus.validation_ids,
                          native_times=model.full_times_s, startup_mode='resume' if args.resume else 'warm_start')
    controller = BackboneRateController
    if args.joint_from_epoch is not None:
        from ul_epoch76_joint import Epoch76JointController
        controller = Epoch76JointController
    control = controller(model, config, backbone_factor=args.backbone_lr_factor, **control_kwargs)
    weight_archive = CheckpointWeightArchive(args.out)
    last_archive_receipt = None
    epoch = saved['epochs_completed'] if args.resume else 0
    record_cursor = saved['records_completed_in_epoch'] if args.resume else 0
    best = dict(saved.get('best_scores', {})) if args.resume else {}
    history = list(saved.get('validation_history', [])) if args.resume else []
    c76_transition = saved.get('c76_transition') if args.resume else None
    base_lr_pilot = saved.get('base_lr_pilot') if args.resume else None
    if args.until_epoch is not None and (not base_lr_pilot or epoch >= end_epoch):
        raise ValueError('the short trial must have an archived LR change and remaining updates')
    if args.joint_from_epoch is not None and not c76_transition:
        raise ValueError('missing original epoch-75 archive provenance')
    if args.resume:
        if args.repair_learning_rate:
            legacy_control = StageController(model, config, **control_kwargs)
            legacy_control.load_state_dict(saved['stage_schedule'])
            intervention = migrate_stage_controller(legacy_control, control)
            intervention.update(source_run_binding_sha256=saved['run_binding_sha256'],
                                target_run_binding_sha256=binding_sha,
                                epochs_completed=epoch, records_completed_in_epoch=record_cursor)
            control.lr_intervention_receipt = intervention
            _write_json(args.out / 'LR_INTERVENTION.json', intervention)
            del legacy_control
        else:
            control.load_state_dict(saved['stage_schedule'])
            control.lr_intervention_receipt = saved.get('lr_intervention')
        del saved
        failure = args.out / 'FAILURE.json'
        if failure.exists():
            os.replace(failure, args.out / ('INTERRUPTION_RECORD_%d.json' % time()))
    _write_json(args.out / 'RUN.json', {'run_binding_sha256': binding_sha, **binding,
                                      'worker_pid': os.getpid(), 'started_at_unix': start_time,
                                      'execution': {'cache_dir': str(args.cache_dir),
                                                    'frozen_detail_fold': args.frozen_detail_fold,
                                                    'sampled_prefetch': args.sampled_prefetch,
                                                    'fast_polynomial': args.fast_polynomial},
                                      'stage_plan': control.plan, 'precision': 'strict_fp32',
                                      **({'c76_transition': c76_transition} if c76_transition else {}),
                                      **({'base_lr_pilot': base_lr_pilot} if base_lr_pilot else {}),
                                      'checkpoint_retention': {
                                          'all_regular_and_best_weight_events': True,
                                          'history': str(weight_archive.history),
                                          'pending': str(weight_archive.pending),
                                          'latest_complete_resume': str(checkpoint_path),
                                          'release_pending_only_after_verified_durable_mirror': True},
                                      'train_frames_per_record_per_update': 256, 'evaluation_frames': 401,
                                      'quality_targets_percent': {'uniform': 2., 'layered': 3.},
                                      'warm_A_policy': 'BASE updates with inherited detail predictor active and frozen',
                                      'test_policy': '100 reserved records were exposed to the initialization CP; no independent-test claim'})
    def payload():
        return {**model.checkpoint_payload(), 'config': old_config, 'dataset_binding': corpus.binding,
                'run_binding_sha256': binding_sha, 'run_settings': run_settings,
                'started_at_unix': start_time, 'epochs_completed': epoch,
                'records_completed_in_epoch': record_cursor, 'microbatch': microbatch,
                'stage_schedule': control.state_dict(), 'frame_sampler': sampler.state_dict(),
                'batch_tuning': tuning, 'energy_table': energy_table,
                'lr_intervention': control.lr_intervention_receipt,
                **({'c76_transition': c76_transition} if c76_transition else {}),
                **({'base_lr_pilot': base_lr_pilot} if base_lr_pilot else {}),
                'best_scores': best, 'validation_history': history}

    def archive_weights(snapshot, event):
        nonlocal last_archive_receipt
        # Keep the normal model/normalizer/architecture payload independently
        # loadable, while the latest checkpoint retains complete resume state.
        omitted = {'stage_schedule', 'frame_sampler', 'batch_tuning', 'energy_table',
                   'best_scores', 'validation_history'}
        weights = {key: value for key, value in snapshot.items() if key not in omitted}
        last_archive_receipt = weight_archive.record(weights, event=event,
            update=control.completed_updates, epoch=epoch, record_cursor=record_cursor,
            phase=control.current_phase or 'INITIAL')
        return last_archive_receipt

    def save_preserved(path, snapshot, event):
        # Never overwrite the mutable convenience file before its weights
        # and save-event identity have entered the immutable archive.
        receipt = archive_weights(snapshot, event)
        save_atomic(path, snapshot)
        return receipt

    def persist():
        waiting = False
        while shutil.disk_usage(args.out).free < (700 << 20):
            if not waiting:
                emit('checkpoint_archive_storage_wait', free_bytes=shutil.disk_usage(args.out).free,
                     recovery='wait for verified durable mirror acknowledgements; no weights discarded')
                _write_json(args.out / 'STATUS.json', {'state': 'WAITING_FOR_CHECKPOINT_STORAGE',
                    'pid': os.getpid(), 'epochs_completed': epoch,
                    'accepted_updates': control.completed_updates, 'time_unix': time()})
                waiting = True
            sleep(1.)
        save_preserved(checkpoint_path, payload(), 'regular_checkpoint')

    def calibrate(_, specification):
        from wide_stage_b_calibration import calibrate_wide_stage_b
        def records():
            for item in specification['draws']:
                row = cache.get(item['record_id'])
                if row['id'] not in energy_table:
                    energy_table[row['id']] = float(row['target'].double().square().sum())
                yield {**row, 'record_energy': energy_table[row['id']]}
        result = calibrate_wide_stage_b(model, records, calibration_spec=specification,
                                        train_ids=corpus.train_ids, test_ids=corpus.test_ids,
                                        optimizer=control.optimizer)
        _write_json(args.out / 'STAGE_B_CALIBRATION.json', result)
        emit('stage_B_calibrated', receipt=result)
        return result

    def validate(label):
        from ul_training_quality import evaluate_ul_quality
        rows = [cache.get(key) for key in corpus.validation_ids]
        quality_history = [{'epoch': item['epoch'], **item['quality']} for item in history]
        result = evaluate_ul_quality(model, rows, history=quality_history, epoch=epoch)
        _write_json(args.out / 'VALIDATION_LATEST.json', result)
        summary = result.get('by_family', result.get('families', result.get('summary', {})))
        history.append({'epoch': epoch, 'updates': control.completed_updates, 'label': label,
                        'quality': result})
        for family, values in summary.items():
            value = values.get('pooled_relative_l2_percent', values.get('pooled_field_relative_l2_percent'))
            if value is None:
                continue
            if float(value) < best.get(family, float('inf')):
                best[family] = float(value)
                save_preserved(args.out / f'best_{family}.pt', {
                    **model.checkpoint_payload(), 'config': old_config,
                    'dataset_binding': corpus.binding, 'epochs_completed': epoch,
                    'selection': {'role': 'VALIDATION', 'family': family, 'metric': 'pooled_relative_l2_percent',
                                  'value': float(value), 'records': list(corpus.validation_ids)},
                    'run_binding_sha256': binding_sha}, 'best_' + family)
        if all(name in summary and summary[name]['pooled_relative_l2_percent'] is not None
               and math.isfinite(summary[name]['pooled_relative_l2_percent'])
               for name in ('uniform', 'layered')):
            ratio = max(summary['uniform']['pooled_relative_l2_percent'] / 2.,
                        summary['layered']['pooled_relative_l2_percent'] / 3.)
            if ratio < best.get('joint_target_ratio', float('inf')):
                best['joint_target_ratio'] = float(ratio)
                save_preserved(args.out / 'best_joint.pt', {
                    **model.checkpoint_payload(), 'config': old_config,
                    'dataset_binding': corpus.binding, 'epochs_completed': epoch,
                    'selection': {'role': 'VALIDATION', 'metric': 'max(uniform_error/2,layered_error/3)',
                                  'value': float(ratio), 'quality': result,
                                  'records': list(corpus.validation_ids)},
                    'run_binding_sha256': binding_sha}, 'best_joint')
        emit('validation', label=label, epoch=epoch, updates=control.completed_updates, quality=result)
        return result

    if not args.resume:
        validate('before_new_updates')
        persist()
    elif args.repair_learning_rate:
        persist()
        emit('learning_rate_repaired', receipt=control.lr_intervention_receipt)
    else:
        archive_weights(payload(), 'resume_start')
    stop_requested = []
    def request_stop(signum, frame):
        stop_requested.append(signum)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    accepted_run = 0
    rolling = []
    frozen_details = None
    emit('training_started', pid=os.getpid(), epochs=200, start_epoch=epoch, stop_after_epoch=end_epoch, train_records=900,
         validation_records=len(corpus.validation_ids), batch_size=batch_size, microbatch=microbatch)
    while epoch < end_epoch:
        epoch_started = perf_counter()
        order = np.random.default_rng(args.seed + epoch).permutation(corpus.train_ids).tolist()
        prefetcher, pending = None, deque()
        remaining_batches = [order[first:first + batch_size]
                             for first in range(record_cursor, len(order), batch_size)]
        next_prepared = 0
        if args.sampled_prefetch:
            from ul_sampled_prefetch import MMapTrainSource, SampledTrainPrefetch
            paths = {key: store.root / (hashlib.sha256(str(key).encode()).hexdigest() + '.pt')
                     for key in corpus.train_ids}
            source = MMapTrainSource(paths, train_ids=corpus.train_ids,
                                    forbidden_ids=(*corpus.test_ids, *corpus.validation_ids))
            prefetcher = SampledTrainPrefetch(source, sampler, train_ids=corpus.train_ids,
                forbidden_ids=(*corpus.test_ids, *corpus.validation_ids), max_batch_size=batch_size,
                device=device, energy_table=energy_table)
            for batch_keys in remaining_batches[:2]:
                pending.append(prefetcher.prepare(batch_keys))
                next_prepared += 1
        for first in range(record_cursor, len(order), batch_size):
            tick = perf_counter()
            keys = order[first:first + batch_size]
            entry = control.begin_update(calibrate)
            model.parent.encoder_trainable = entry['phase'] != 'B'
            if entry['phase_changed']:
                emit('phase_entered', phase=entry['phase'], updates=control.completed_updates,
                     learning_rates=entry['learning_rates'])
            if args.frozen_detail_fold and entry['phase'] == 'A':
                if frozen_details is None:
                    from frozen_detail_factors import FrozenDetailFactors
                    frozen_details = FrozenDetailFactors(model.parent.decoder)
                    emit('frozen_detail_factors_compiled', phase=entry['phase'])
            else:
                frozen_details = None
            if prefetcher is None:
                cache.prefetch(order[first:first + batch_size + 2 * microbatch])
                resident = None
            else:
                resident = prefetcher.acquire(pending.popleft())
                if tuple(keys) != resident.plan.record_ids:
                    raise RuntimeError('prefetched record order differs from the bound epoch cursor')
                energy_table.update(prefetcher.energy_table)
            sampler_before = deepcopy(sampler.state_dict())
            retry_rng = capture_rng()
            retry_buffers = {name: value.detach().clone() for name, value in model.named_buffers()
                             if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
            while True:
                control.optimizer.zero_grad(set_to_none=True)
                if resident is None:
                    total_loss = backward_attempt(model, cache, keys, microbatch, sampler, energy_table, frozen_details)
                else:
                    total_loss = backward_resident_attempt(model, resident.tensors, microbatch, frozen_details)
                if total_loss is not None:
                    break
                control.optimizer.zero_grad(set_to_none=True)
                sampler.load_state_dict(sampler_before)
                restore_rng(retry_rng)
                with torch.no_grad():
                    for name, value in model.named_buffers():
                        if name in retry_buffers:
                            value.copy_(retry_buffers[name])
                trim_gpu_cache(cache, cache.gpu_budget - (2 << 30))
                gc.collect()
                torch.cuda.empty_cache()
                if microbatch <= 1:
                    raise torch.cuda.OutOfMemoryError('one-record pressure update does not fit available device memory')
                microbatch = max(1, microbatch // 2)
                emit('reduce_microbatch', microbatch=microbatch, effective_batch=batch_size)
            norm = control.clip_gradients()
            first_joint_step = c76_transition and control.completed_updates == control.joint_start
            step_audit = {}
            if first_joint_step:
                for role in ('BASE', 'DETAIL'):
                    outputs = [(name, parameter, parameter.grad.detach().norm())
                               for name, parameter in control.roles[role] if parameter.grad is not None]
                    if not outputs:
                        raise RuntimeError(f'first C step lacked gradients for {role}')
                    magnitudes = torch.stack([norm_value for _, _, norm_value in outputs]).cpu()
                    index = int(magnitudes.argmax())
                    if not torch.isfinite(magnitudes).all() or magnitudes[index] <= 0:
                        raise RuntimeError(f'first C step has no finite nonzero gradient for {role}')
                    name, parameter, _ = outputs[index]
                    step_audit[role] = (name, parameter, parameter.detach().clone(), float(magnitudes[index]))
            control.optimizer.step()
            control.finish_update(optimizer_was_stepped=True)
            if step_audit:
                changes = {role: {'parameter': name, 'gradient_norm': magnitude,
                                  'max_abs_parameter_change': float((parameter.detach() - before).abs().max())}
                           for role, (name, parameter, before, magnitude) in step_audit.items()}
                prior_width = model.basis_expansion_receipt['old_basis']
                opened_columns = [int(weight[:, prior_width:].count_nonzero())
                                  for name, weight in control.roles['DETAIL']
                                  if name.startswith('parent.decoder.detail_time.')]
                first_step_receipt = {'epoch': epoch + 1, 'accepted_updates': control.completed_updates,
                                      'phase': entry['phase'], 'changes': changes,
                                      'new_detail_time_entries_nonzero_per_scale': opened_columns}
                _write_json(args.out / 'C76_FIRST_STEP.json', first_step_receipt)
                emit('c76_first_joint_update', **first_step_receipt)
            if prefetcher is not None:
                prefetcher.release(resident)
                if next_prepared < len(remaining_batches):
                    pending.append(prefetcher.prepare(remaining_batches[next_prepared]))
                    next_prepared += 1
            duration = perf_counter() - tick
            record_cursor = first + len(keys)
            accepted_run += 1
            rolling.append((len(keys), duration))
            rolling = rolling[-50:]
            rate = sum(n for n, _ in rolling) / sum(t for _, t in rolling)
            left = (end_epoch - epoch) * 900 - record_cursor
            status = {'state': 'TRAINING', 'pid': os.getpid(), 'time_unix': time(),
                      'epochs_completed': epoch, 'epoch_in_progress': epoch + 1,
                      'records_completed_in_epoch': record_cursor, 'epochs_total': 200,
                      'accepted_updates': control.completed_updates, 'phase': entry['phase'],
                      'last_record_mean_relative_squared_error': total_loss,
                      'recent_records_per_second': rate, 'eta_seconds': left / rate,
                      'eta_timestamp_unix': time() + left / rate, 'effective_batch': batch_size,
                      'microbatch': microbatch, 'learning_rates': entry['learning_rates'],
                      'best_validation_scores': best, 'gpu_allocated_bytes': torch.cuda.memory_allocated(),
                      'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'gpu_reserved_bytes': torch.cuda.memory_reserved(),
                      'disk_free_bytes': shutil.disk_usage(args.out).free}
            status['last_weight_archive'] = last_archive_receipt
            if resident is not None:
                status['input_pipeline'] = resident.metrics
            _write_json(args.out / 'STATUS.json', status)
            if control.completed_updates <= 3 or control.completed_updates % 10 == 0:
                emit('update', **status, gradient_norm=float(norm) if norm is not None else None,
                     cache=cache.stats())
            resident = None
            if control.completed_updates % 100 == 0 or first_joint_step:
                persist()
            if stop_requested:
                persist()
                _write_json(args.out / 'STATUS.json', {**status, 'state': 'PAUSED_AT_UPDATE_BOUNDARY',
                            'signal': stop_requested[-1], 'checkpoint_committed': True})
                emit('paused_at_update_boundary', updates=control.completed_updates)
                if prefetcher is not None:
                    prefetcher.close()
                return
            if args.smoke_updates and accepted_run >= args.smoke_updates:
                persist()
                emit('smoke_complete', accepted_updates=control.completed_updates)
                if prefetcher is not None:
                    prefetcher.close()
                return
        if prefetcher is not None:
            prefetcher.close()
            prefetcher = None
        epoch += 1
        record_cursor = 0
        emit('epoch_complete', epoch=epoch, seconds=perf_counter() - epoch_started)
        if epoch % args.eval_every == 0 or epoch in (1, end_epoch):
            validate('complete_401_frame_validation')
        persist()
    if args.until_epoch is not None:
        result = history[-1]
        if result['epoch'] != end_epoch or result['updates'] != control.completed_updates:
            raise RuntimeError('pilot endpoint has no complete-epoch validation')
        _write_json(args.out / 'STATUS.json', {'state': 'COMPLETE', 'pid': os.getpid(), 'phase': control.current_phase,
                    'epochs_completed': epoch, 'epochs_total': args.epochs,
                    'accepted_updates': control.completed_updates, 'trial_end_epoch': end_epoch,
                    'final_validation': result['quality'], 'time_unix': time()})
        emit('pilot_complete', epochs_completed=epoch, accepted_updates=control.completed_updates)
        return
    result = validate('final')
    _write_json(args.out / 'STATUS.json', {'state': 'COMPLETE', 'epochs_completed': 200,
                'accepted_updates': control.completed_updates, 'best_validation_percent': best,
                'final_validation': result, 'time_unix': time()})
    emit('complete', epochs_completed=200, best_validation_percent=best)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if '--out' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--out') + 1])
            try:
                _write_json(output / 'FAILURE.json', {'state': 'FAILED', 'pid': os.getpid(),
                            'time_unix': time(), 'error': repr(error),
                            'recovery': 'restore the last atomically saved checkpoint; never reuse partial optimizer updates'})
            except Exception:
                pass
        raise
