#!/usr/bin/env python3
"""True-batch 500-source/50-epoch FNO and DeepONet workers; no auto-resume.

The controller owns the scientific ledger. --probe performs complete real-data
forward/backward passes, without constructing or stepping an optimizer. Every
training batch includes every native400 x 201 x 201 label. The shared DeepONet
trunk is computed once per point chunk, then contracted with all B branches.
"""
from __future__ import annotations
import argparse
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
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
sys.dont_write_bytecode = True
import train_marmousi_matched_baselines_20261020 as old

SCHEMA = 'MARMOUSI_BASELINE_TRUEB_500SOURCE_50EPOCH_V1'


def configure_runtime(cfg):
    import numpy as np
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(cfg['training'].get('cpu_threads', 4))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    seed = cfg['training']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bindings(cfg, config_path):
    path = Path(cfg['manifest_path'])
    if old.sha(path) != cfg['manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    files = {**json.loads(path.read_text())['files'], **cfg.get('binding_files', {}), **cfg.get('probe_results', {})}
    for key, value in files.items():
        filename = Path(key)
        filename = filename if filename.is_absolute() else ROOT/filename
        if old.sha(filename) != value:
            raise ValueError(f'Frozen dependency changed: {filename}')
    return dict(source_manifest_sha256=cfg['manifest_sha256'], config_sha256=old.sha(config_path), files=files)


def make_dataset(cfg):
    from marmousi_baseline50_data_20261020 import Marmousi500Dataset
    return Marmousi500Dataset(dict(cfg['data'], normalization=cfg['normalization'], pressure_rms_pa=cfg['pressure_rms_pa']))


def selection(cfg, kind):
    train = cfg['training']
    return int(train['batch_size'][kind]), int(train['chunk_points'][kind])


def verify_probe(cfg, kind):
    """Training uses only the frozen, measured zero-update admission result."""
    expected = (Path(cfg['run_dir']).resolve()/kind/'PROBE.json')
    proofs = cfg.get('probe_results', {})
    if str(expected) not in proofs or old.sha(expected) != proofs[str(expected)]:
        raise ValueError('Missing or changed actual probe proof')
    proof = json.loads(expected.read_text())
    probe_config = Path(cfg['run_dir']).resolve()/'PROBE_CONFIG.json'
    batch, chunk = selection(cfg, kind)
    chosen = proof.get('selected') or {}
    if not (proof['status'] == 'PASS' and proof['kind'] == kind and proof['optimizer_updates'] == 0
            and proof['source_manifest_sha256'] == cfg['manifest_sha256']
            and proof['config_sha256'] == old.sha(probe_config)
            and proof['actual_GPU_B2_vs_B1']['status'] == 'PASS'
            and chosen.get('batch_size') == batch and chosen.get('chunk_points') == chunk
            and chosen['peak_reserved_bytes'] <= cfg['probe']['max_reserved_gib']*(1 << 30)):
        raise ValueError('Production choice is not the actual verified zero-update probe')
    if not any(row.get('status') == 'PASS' and row.get('gradients_finite') is True
               and all(row.get(key) == chosen[key] for key in chosen)
               for row in proof['candidates']):
        raise ValueError('Selected throughput/memory is not an actual successful probe candidate')
    return proof


def build_objects(cfg, kind, device, with_optimizer=True):
    import torch
    from baselines.marmousi_matched_20261020.models import build_model
    model = build_model(kind, target_parameters=cfg['target_parameters'], **cfg['model'][kind]).to(device)
    if not with_optimizer:
        return model
    train = cfg['training']
    batch_size, _ = selection(cfg, kind)
    total = train['epochs'] * math.ceil(train['records_per_epoch']/batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train['lr'], weight_decay=train['weight_decay'])
    def factor(step):
        warmup = train['warmup_updates']
        if warmup and step < warmup:
            return float(step+1)/warmup
        fraction = min(1., max(0., (step-warmup)/max(1, total-warmup)))
        return train['min_lr_ratio'] + (1-train['min_lr_ratio'])*.5*(1+math.cos(math.pi*fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return model, optimizer, scheduler


def to_device(cpu, cfg, device):
    """Original raw Pa is divided by the fixed RMS on the GPU, in FP32."""
    import torch
    raw = cpu['pressure_pa'].to(device=device, dtype=torch.float32, non_blocking=True)
    scales = torch.as_tensor(cpu['scale'], device=device, dtype=torch.float32)
    divisor = scales if scales.ndim == 0 else scales[:, None, None, None]
    return dict(inputs=cpu['inputs'].to(device, non_blocking=True), truth=raw/divisor,
                times=cpu['times'].to(device, non_blocking=True), scale=scales, ids=cpu['ids'],
                norm=cfg['normalization'], shape=tuple(raw.shape[1:]))


def supervision(truth, cfg):
    from marmousi_baseline50_data_20261020 import fixed_supervision
    train = cfg['training']
    return fixed_supervision(truth, train['data_window_boundaries'], train['receiver_indices'])


def backward_complete(model, kind, data, sup, receiver_weight, chunk_points, loss_multiplier=1.):
    import torch
    from marmousi_baseline50_data_20261020 import weighted_loss
    inputs, truth = data['inputs'], data['truth']
    if kind == 'fno':
        prediction = model(inputs)
        loss, field, receiver = weighted_loss(prediction, truth, sup, receiver_weight)
        (loss*loss_multiplier).backward()
        return dict(loss=float(loss.detach()), field_loss=float(field.detach()), receiver_loss=float(receiver.detach()))
    batch = truth.shape[0]
    points = truth[0].numel()
    branch = model.encode_branch(inputs)
    surrogate = branch.detach().requires_grad_(True)
    totals = torch.zeros(3, device=truth.device, dtype=torch.float32)
    rec = sup['receiver']
    for start in range(0, points, chunk_points):
        stop = min(start+chunk_points, points)
        coords, ti, zi, xi = old.point_coordinates(start, stop, truth.shape[1:], data['times'], data['norm'])
        # Exactly one shared trunk evaluation for all B records in this chunk.
        prediction = model.decode_trunk(surrogate, coords)
        error = (prediction-truth.reshape(batch, -1)[:, start:stop]).square()
        field_per_sample = (error*sup['frame_weights'][:, ti]).sum(dim=1, dtype=torch.float32)/(sup['energy']*points)
        mask = ((zi[:, None] == rec[None, :, 0]) & (xi[:, None] == rec[None, :, 1])).any(dim=-1)
        receiver_per_sample = error[:, mask].sum(dim=1, dtype=torch.float32)/sup['receiver_energy']
        field = field_per_sample.mean(dtype=torch.float32)
        receiver = receiver_per_sample.mean(dtype=torch.float32)
        loss = field+receiver_weight*receiver
        (loss*loss_multiplier).backward()
        totals += torch.stack([loss.detach(), field.detach(), receiver.detach()])
    if surrogate.grad is None:
        raise ValueError('Missing complete-batch branch derivative')
    branch.backward(surrogate.grad)
    return dict(zip(('loss', 'field_loss', 'receiver_loss'), map(float, totals)))


def gradients_finite(model):
    import torch
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    return bool(gradients) and all(g is not None for g in gradients) and bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())


def predict(model, kind, inputs, shape, times, norm, scales, chunk_points):
    import torch
    training = model.training
    model.eval()
    with torch.no_grad():
        if kind == 'fno':
            prediction = model(inputs)
        else:
            branch = model.encode_branch(inputs)
            prediction = torch.empty((len(inputs), *shape), dtype=torch.float32, device=inputs.device)
            points = math.prod(shape)
            for start in range(0, points, chunk_points):
                stop = min(start+chunk_points, points)
                coords, _, _, _ = old.point_coordinates(start, stop, shape, times, norm)
                prediction.reshape(len(inputs), -1)[:, start:stop] = model.decode_trunk(branch, coords)
        scale = torch.as_tensor(scales, device=inputs.device, dtype=torch.float32)
        result = (prediction*(scale if scale.ndim == 0 else scale[:, None, None, None])).cpu().numpy()
    model.train(training)
    return result


def paper_data(cfg, device):
    evaluation = dict(cfg['evaluation'], normalization=cfg['normalization'], pressure_rms_pa=cfg['pressure_rms_pa'])
    for key in ('inputs', 'reference'):
        if old.sha(evaluation[key+'_path']) != evaluation[key+'_sha256']:
            raise ValueError('Paper evaluation input changed')
    return old.load_data(evaluation, device)


def assess_paper(model, kind, cfg, paper, trained_times, chunk_points):
    import numpy as np
    times = trained_times if kind == 'fno' else paper['times']
    prediction = predict(model, kind, paper['inputs'], paper['shape'], times, paper['norm'], paper['scale'], chunk_points)[0]
    metrics = old.score_pa(prediction, paper['reference_pa'], cfg['training']['receiver_indices'])
    trained = trained_times.detach().cpu().numpy()
    evaluated = times.detach().cpu().numpy()
    exact = paper['time_s']
    digest = lambda array: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    protocol = dict(trained_axis_sha256=digest(trained), evaluated_axis_sha256=digest(evaluated),
                    paper_axis_sha256=digest(exact), exact_to_paper=bool(np.array_equal(evaluated, exact)),
                    ulp_differences=int(np.count_nonzero(evaluated.view(np.uint32) != exact.view(np.uint32))),
                    max_abs_time_difference_s=float(np.max(np.abs(evaluated.astype(np.float64)-exact.astype(np.float64)))),
                    training_to_paper_ulp_differences=int(np.count_nonzero(trained.view(np.uint32) != exact.view(np.uint32))),
                    same_native_frame_indices=True, output_interpolation=False,
                    method='fixed400_head_on_training_axis' if kind == 'fno' else 'explicit_original_paper_time_queries')
    return prediction, metrics, protocol


def initial_progress(count, seed):
    from marmousi_baseline50_data_20261020 import epoch_order
    return dict(completed_epochs=0, active_epoch=1, offset=0, roster=list(map(int, epoch_order(count, seed, 0))),
                committed_updates=0, training_exposures=0, training_seconds=0.)


def validate_progress(progress, cfg, kind):
    from marmousi_baseline50_data_20261020 import epoch_order
    count = cfg['training']['records_per_epoch']
    batch, _ = selection(cfg, kind)
    done, offset = progress['completed_epochs'], progress['offset']
    if progress['training_exposures'] != done*count+offset or not 0 <= offset <= count:
        raise ValueError('Checkpoint exposure/epoch cursor inconsistent')
    expected_updates = done*math.ceil(count/batch) + math.ceil(offset/batch)
    if progress['committed_updates'] != expected_updates or (offset < count and offset % batch):
        raise ValueError('Checkpoint complete-batch cursor inconsistent')
    expected = [] if done == cfg['training']['epochs'] else list(map(int, epoch_order(count, cfg['training']['seed'], done)))
    if progress['roster'] != expected or progress['active_epoch'] != done+1:
        raise ValueError('Checkpoint deterministic complete epoch roster differs')


def restore_objects(cfg, kind, saved, device, restore_rng=True):
    if saved['schema'] != SCHEMA or saved['config'] != cfg or saved['kind'] != kind:
        raise ValueError('Complete checkpoint identity mismatch')
    validate_progress(saved['progress'], cfg, kind)
    model, optimizer, scheduler = build_objects(cfg, kind, device)
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    if not all((old.equal_tree(model.state_dict(), saved['model']), old.equal_tree(optimizer.state_dict(), saved['optimizer']),
                old.equal_tree(scheduler.state_dict(), saved['scheduler']))):
        raise ValueError('Complete rebuilt model/Adam/scheduler differs')
    if restore_rng:
        old.set_rng(saved['rng'], device)
    return model, optimizer, scheduler


def probe(config_path, kind, output=None):
    """Measured resource admission only. Never calls optimizer.step()."""
    import numpy as np
    import torch
    cfg = json.loads(Path(config_path).read_text())
    run = Path(cfg['run_dir']).resolve()/kind
    if not run.is_relative_to(ROOT/'runs'):
        raise ValueError('Only project run outputs allowed')
    run.mkdir(parents=True, exist_ok=True)
    output = Path(output).resolve() if output else run/'PROBE.json'
    if not output.is_relative_to(Path(cfg['run_dir']).resolve()):
        raise ValueError('Probe output must stay inside this run')
    if output.exists():
        raise ValueError('Existing probe evidence retained; no implicit retry')
    pins = bindings(cfg, config_path)
    with (run/'PROBE_WORKER.json').open('x') as handle:
        json.dump(dict(**old.identity(), kind=kind, optimizer_updates=0), handle, indent=2)
    started = time.monotonic()
    configure_runtime(cfg)
    device = cfg.get('device', 'cuda:0')
    dataset = make_dataset(cfg)
    model = build_objects(cfg, kind, device, with_optimizer=False)
    records = []
    selected = None
    try:
        # Actual GPU batch2 versus the arithmetic average of two batch1 gradients.
        cpu = dataset.load_batch([0, 1])
        data = to_device(cpu, cfg, device)
        cpu_truth = cpu['pressure_pa']/float(cfg['pressure_rms_pa'])
        normalization = dict(cpu_gpu_bitwise_equal=bool(torch.equal(cpu_truth, data['truth'].cpu())),
                             max_abs_normalized_difference=float((cpu_truth-data['truth'].cpu()).abs().max()),
                             production_normalization='rawPa_to_GPU_then_FP32_divide_fixed_RMS')
        chunk0 = min(cfg['probe']['chunk_candidates'])
        model.zero_grad(set_to_none=True)
        batched = backward_complete(model, kind, data, supervision(data['truth'], cfg), cfg['training']['receiver_weight'], chunk0)
        if not gradients_finite(model) or not all(math.isfinite(value) for value in batched.values()):
            raise ValueError('Nonfinite initial trueB2 loss/gradient')
        expected = [p.grad.detach().cpu().clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        singles = []
        for i in range(2):
            one = dict(data, inputs=data['inputs'][i:i+1], truth=data['truth'][i:i+1])
            singles.append(backward_complete(model, kind, one, supervision(one['truth'], cfg), cfg['training']['receiver_weight'], chunk0, .5)['loss'])
        numerator = denominator = 0.
        max_abs = 0.
        for expected_grad, parameter in zip(expected, model.parameters()):
            actual = parameter.grad.detach().cpu()
            delta = (actual-expected_grad).double()
            numerator += float(delta.square().sum())
            denominator += float(expected_grad.double().square().sum())
            max_abs = max(max_abs, float(delta.abs().max()))
        gradient_relative_l2 = math.sqrt(numerator/max(denominator, 1e-300))
        loss_relative_error = abs(batched['loss']-sum(singles)/2)/max(abs(batched['loss']), 1e-30)
        if (not all(math.isfinite(value) for value in [gradient_relative_l2, loss_relative_error, *singles])
                or gradient_relative_l2 > 1e-4 or loss_relative_error > 1e-5 or not gradients_finite(model)):
            raise ValueError(f'GPU trueB2/averageB1 numerical check failed: {gradient_relative_l2}, {loss_relative_error}')
        numerical = dict(status='PASS', gradient_relative_l2=gradient_relative_l2, gradient_max_abs=max_abs,
                         loss_relative_error=loss_relative_error, batch_loss=batched['loss'],
                         mean_single_loss=sum(singles)/2, optimizer_updates=0)
        del cpu, data, one, expected, cpu_truth, expected_grad, actual, delta
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        limit = cfg['probe']['max_reserved_gib']*(1 << 30)
        previous = None
        def measure(batch, chunk):
            nonlocal selected, previous
            free, _ = torch.cuda.mem_get_info(device)
            current = torch.cuda.memory_reserved(device)
            predicted = None if previous is None else previous['peak_reserved_bytes']*batch/previous['batch_size']*max(1., chunk/previous['chunk_points'])
            if predicted is not None and (predicted > limit or predicted > free+current):
                records.append(dict(status='SKIPPED_PREDICTED_MEMORY', batch_size=batch, chunk_points=chunk,
                                    predicted_reserved_bytes=predicted, actual_free_bytes=free))
                return
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            local_cpu = dataset.load_batch(list(range(batch)))
            local_data = to_device(local_cpu, cfg, device)
            local_sup = supervision(local_data['truth'], cfg)
            torch.cuda.synchronize(device)
            begin = time.monotonic()
            model.zero_grad(set_to_none=True)
            try:
                loss = backward_complete(model, kind, local_data, local_sup, cfg['training']['receiver_weight'], chunk)
                finite = gradients_finite(model)
                if not finite or not all(math.isfinite(value) for value in loss.values()):
                    raise ValueError('Nonfinite/missing probe gradient')
                torch.cuda.synchronize(device)
                seconds = time.monotonic()-begin
                row = dict(status='PASS', batch_size=batch, chunk_points=chunk, seconds=seconds,
                           samples_per_second=batch/seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(device), gradients_finite=finite, **loss)
                if row['peak_reserved_bytes'] > limit:
                    row['status'] = 'EXCEEDS_CONCURRENT_ADMISSION'
                records.append(row)
                previous = row
                if row['status'] == 'PASS' and (selected is None or row['samples_per_second'] > selected['samples_per_second']):
                    selected = {key: row[key] for key in ('batch_size', 'chunk_points', 'samples_per_second', 'peak_allocated_bytes', 'peak_reserved_bytes')}
            except torch.cuda.OutOfMemoryError as error:
                records.append(dict(status='OOM', batch_size=batch, chunk_points=chunk, error=str(error)))
                raise
            finally:
                model.zero_grad(set_to_none=True)
                del local_cpu, local_data, local_sup
                gc.collect()
                torch.cuda.empty_cache()
        for batch in cfg['probe']['batch_candidates']:
            if time.monotonic()-started > cfg['probe'].get('maximum_seconds', 1800):
                break
            measure(batch, chunk0)
        if selected is not None and kind == 'deeponet':
            selected_batch = selected['batch_size']
            for chunk in cfg['probe']['chunk_candidates']:
                if chunk != chunk0 and time.monotonic()-started <= cfg['probe'].get('maximum_seconds', 1800):
                    measure(selected_batch, chunk)
        if selected is None:
            raise ValueError('No measured complete-batch candidate fits concurrent admission')
        if bindings(cfg, config_path) != pins:
            raise ValueError('Probe dependencies drifted')
        result = dict(status='PASS', kind=kind, optimizer_updates=0, selected=selected, candidates=records,
                      normalization_check=normalization, actual_GPU_B2_vs_B1=numerical,
                      elapsed_seconds=time.monotonic()-started, **pins)
        old.write_json(output, result)
        return 0
    except BaseException as error:
        old.write_json(output, dict(status='FAILED', kind=kind, selected=None, candidates=records, optimizer_updates=0,
                                   error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started, **pins))
        raise


def worker(config_path, kind):
    import numpy as np
    import torch
    from baselines.marmousi_matched_20261020.models import parameter_report
    from marmousi_baseline50_data_20261020 import Prefetch, epoch_order
    cfg = json.loads(Path(config_path).read_text())
    run = Path(cfg['run_dir']).resolve()/kind
    if not run.is_relative_to(ROOT/'runs'):
        raise ValueError('Only project run outputs allowed')
    verify_probe(cfg, kind)
    run.mkdir(parents=True, exist_ok=True)
    identity = dict(**old.identity(), kind=kind)
    with (run/'WORKER.json').open('x') as handle:
        json.dump(identity, handle, indent=2)
    started = time.monotonic()
    train = cfg['training']
    if train['epochs'] != 50 or train['records_per_epoch'] != 500 or cfg['target_parameters'] != 14552439:
        raise ValueError('This authorization is exactly two independent 500-source/50-epoch matched baselines')
    device = cfg.get('device', 'cuda:0')
    batch_size, chunk = selection(cfg, kind)
    progress = initial_progress(500, train['seed'])
    pins = bindings(cfg, config_path)
    latest_loss = None
    restore8 = None
    stopped = None
    def status(state):
        old.write_json(run/'STATUS.json', dict(status=state, utc=old.utc(), **identity, **progress,
                       target_epochs=50, target_exposures=25000, batch_size=batch_size, chunk_points=chunk,
                       loss=latest_loss, elapsed_seconds=time.monotonic()-started))
    def save():
        if bindings(cfg, config_path) != pins:
            raise ValueError('Frozen training configuration/source changed')
        validate_progress(progress, cfg, kind)
        payload = dict(schema=SCHEMA, kind=kind, config=cfg, progress=old.cpu_tree(progress),
                       model=old.cpu_tree(model.state_dict()), optimizer=old.cpu_tree(optimizer.state_dict()),
                       scheduler=old.cpu_tree(scheduler.state_dict()), rng=old.get_rng(device), data_state=dataset.state_dict(),
                       bindings=pins, trained_times=trained_times.cpu().clone(), normalization=cfg['normalization'],
                       pressure_rms_pa=cfg['pressure_rms_pa'])
        saved = old.atomic_checkpoint(run/'latest.pt', payload)
        receipt = dict(utc=old.utc(), sha256=old.sha(run/'latest.pt'), committed_updates=progress['committed_updates'],
                       completed_epochs=progress['completed_epochs'], training_exposures=progress['training_exposures'], complete_readback=True)
        old.write_json(run/'CHECKPOINT.json', receipt)
        return saved, receipt
    def cpu_restore(saved, receipt):
        cpu_model, cpu_opt, cpu_scheduler = restore_objects(cfg, kind, saved, 'cpu', restore_rng=False)
        result = dict(status='PASS', checkpoint_sha256=receipt['sha256'], committed_updates=progress['committed_updates'],
                      completed_epochs=progress['completed_epochs'], training_exposures=progress['training_exposures'],
                      model_populated_Adam_scheduler_cursor_roster_rng=True)
        del cpu_model, cpu_opt, cpu_scheduler
        old.set_rng(saved['rng'], device)
        return result
    try:
        configure_runtime(cfg)
        status('PREPARING')
        dataset = make_dataset(cfg)
        if len(dataset) != 500:
            raise ValueError('Complete frozen 500-source training roster required')
        first = dataset.load_batch([0])
        trained_times = first['times'].to(device)
        del first
        model, optimizer, scheduler = build_objects(cfg, kind, device)
        report = parameter_report(model, cfg['target_parameters'])
        if not report['exact_match'] or report['frozen_real_scalars']:
            raise ValueError('Exactly all-active matched parameters required')
        old.write_json(run/'SELECTION.json', report)
        paper = paper_data(cfg, device)
        save()
        initial_pred, initial_metrics, protocol = assess_paper(model, kind, cfg, paper, trained_times, chunk)
        old.write_json(run/'BASELINE.json', dict(epoch=0, metrics=initial_metrics, evaluation_time_protocol=protocol))
        del initial_pred
        with (run/'metrics.jsonl').open('x', buffering=1) as log, (run/'EPOCHS.jsonl').open('x', buffering=1) as epochs:
            while progress['completed_epochs'] < 50 and stopped is None:
                epoch = progress['completed_epochs']+1
                while progress['offset'] < 500 and stopped is None:
                    remaining = progress['roster'][progress['offset']:]
                    batches = [remaining[i:i+batch_size] for i in range(0, len(remaining), batch_size)]
                    prefetch = Prefetch(dataset, batches, depth=cfg['data'].get('prefetch_depth', 2), workers=cfg['data'].get('prefetch_workers', 2))
                    try:
                        for batch_indices, cpu in zip(batches, prefetch):
                            if any(path.exists() for path in (run/'STOP', run/'STOP_REQUEST.json', run.parent/'STOP_REQUEST.json')):
                                stopped = 'explicit_run_stop_requested'
                                break
                            if list(map(int, cpu['ids'])) != batch_indices:
                                raise ValueError('Prefetch record order differs from durable cursor')
                            tick = time.monotonic()
                            data = to_device(cpu, cfg, device)
                            if not torch.equal(data['times'], trained_times):
                                raise ValueError('Batch time axis changed')
                            sup = supervision(data['truth'], cfg)
                            optimizer.zero_grad(set_to_none=True)
                            losses = backward_complete(model, kind, data, sup, train['receiver_weight'], chunk)
                            if not gradients_finite(model) or not all(math.isfinite(value) for value in losses.values()):
                                raise ValueError('Missing/nonfinite complete-batch gradient')
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train['grad_clip'], error_if_nonfinite=True)
                            optimizer.step()
                            scheduler.step()
                            if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
                                raise ValueError('Nonfinite optimizer endpoint')
                            torch.cuda.synchronize(device)
                            seconds = time.monotonic()-tick
                            progress['offset'] += len(batch_indices)
                            progress['committed_updates'] += 1
                            progress['training_exposures'] += len(batch_indices)
                            progress['training_seconds'] += seconds
                            latest_loss = losses['loss']
                            row = dict(utc=old.utc(), epoch=epoch, offset=progress['offset'], step=progress['committed_updates'],
                                       training_exposures=progress['training_exposures'], batch_size=len(batch_indices),
                                       seconds=seconds, samples_per_second=len(batch_indices)/seconds,
                                       gradient_norm=float(norm), prefetch=cpu.get('prefetch'), **losses)
                            log.write(json.dumps(row, allow_nan=False)+'\n')
                            print(json.dumps(row), flush=True)
                            status('RUNNING')
                            if progress['committed_updates'] == 8:
                                prefetch.close()
                                saved, receipt = save()
                                witness, _, _ = assess_paper(model, kind, cfg, paper, trained_times, chunk)
                                data_witness = old.cpu_tree(dict(inputs=data['inputs'], truth=data['truth'],
                                                               times=data['times'], supervision=sup))
                                del data, sup, cpu
                                model = optimizer = scheduler = None
                                gc.collect()
                                torch.cuda.empty_cache()
                                model, optimizer, scheduler = restore_objects(cfg, kind, saved, device)
                                rebuilt_dataset = make_dataset(cfg)
                                if not old.equal_tree(saved['data_state'], rebuilt_dataset.state_dict()):
                                    raise ValueError('CUDA8 dataset bindings changed')
                                reloaded = to_device(rebuilt_dataset.load_batch(batch_indices), cfg, device)
                                if not (torch.equal(reloaded['inputs'].cpu(), data_witness['inputs'])
                                        and torch.equal(reloaded['truth'].cpu(), data_witness['truth'])
                                        and torch.equal(reloaded['times'], saved['trained_times'].to(device))
                                        and old.equal_tree(supervision(reloaded['truth'], cfg), data_witness['supervision'])):
                                    raise ValueError('CUDA8 actual data/normalizer/time/supervision reconstruction differs')
                                dataset = rebuilt_dataset
                                progress = old.cpu_tree(saved['progress'])
                                after, _, _ = assess_paper(model, kind, cfg, paper, trained_times, chunk)
                                if not np.array_equal(witness, after) or not old.equal_tree(old.get_rng(device), saved['rng']):
                                    raise ValueError('CUDA8 exact prediction/RNG restore differs')
                                restore8 = dict(status='RESTORED_AWAITING_UPDATE9', checkpoint_sha256=receipt['sha256'],
                                                committed_updates=8, next_update=9, complete_state_exact=True, pressure_bitwise_equal=True)
                                old.write_json(run/'CUDA_RESTORE8.json', restore8)
                                del reloaded, saved, witness, after, data_witness
                                break  # Fresh prefetch reconstructs remaining roster from durable cursor.
                            if progress['committed_updates'] == 9 and restore8 is not None:
                                restore8.update(status='PASS', restored_objects_committed_update9=True, utc=old.utc())
                                old.write_json(run/'CUDA_RESTORE8.json', restore8)
                            if progress['committed_updates'] % train['save_every_updates'] == 0:
                                save()
                            del data, sup, cpu
                    finally:
                        prefetch.close()
                if stopped is not None:
                    break
                if progress['offset'] != 500:
                    raise ValueError('Epoch ended without every original record')
                progress['completed_epochs'] = epoch
                progress['active_epoch'] = epoch+1
                progress['offset'] = 0
                progress['roster'] = [] if epoch == 50 else list(map(int, epoch_order(500, train['seed'], epoch)))
                saved, receipt = save()
                prediction, metrics, protocol = assess_paper(model, kind, cfg, paper, trained_times, chunk)
                restore = cpu_restore(saved, receipt)
                old.write_json(run/'LATEST_CPU_RESTORE.json', restore)
                epoch_row = dict(epoch=epoch, epoch_exposures=500, training_exposures=progress['training_exposures'],
                                 committed_updates=progress['committed_updates'], metrics=metrics,
                                 evaluation_time_protocol=protocol, checkpoint_sha256=receipt['sha256'], CPU_restore=restore)
                epochs.write(json.dumps(epoch_row, allow_nan=False)+'\n')
                old.write_json(run/'LATEST_ASSESSMENT.json', epoch_row)
                del saved, prediction
                status('RUNNING')
        if progress['completed_epochs'] == 50:
            saved = torch.load(run/'latest.pt', map_location='cpu', weights_only=False)
            receipt = json.loads((run/'CHECKPOINT.json').read_text())
        else:
            saved, receipt = save()
        final_prediction, final_metrics, protocol = assess_paper(model, kind, cfg, paper, trained_times, chunk)
        restore = cpu_restore(saved, receipt)
        old.write_json(run/'FINAL_CPU_RESTORE.json', restore)
        with (run/'.prediction_pa.npy.tmp').open('wb') as handle:
            np.save(handle, final_prediction, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(run/'.prediction_pa.npy.tmp', run/'prediction_pa.npy')
        finished = progress['completed_epochs'] == 50 and progress['training_exposures'] == 25000
        result = dict(status='COMPLETE' if finished else 'INTERRUPTED', kind=kind, utc=old.utc(),
                      initialization='independent_random', input_pressure_frames=0, target_epochs=50,
                      **progress, exposures=progress['training_exposures'], parameters=report, batch_size=batch_size, chunk_points=chunk,
                      initial_metrics=initial_metrics, final_metrics=final_metrics, evaluation_time_protocol=protocol,
                      checkpoint_sha256=receipt['sha256'], prediction_sha256=old.sha(run/'prediction_pa.npy'),
                      final_CPU_restore=restore, CUDA_restore8=restore8, stop_reason=stopped,
                      training_objective='per-sample original four-window DATA plus .01 receiver, then batch mean; no PDE/IC terms',
                      elapsed_seconds=time.monotonic()-started)
        old.write_json(run/'RESULT.json', result)
        status(result['status'])
        return 0 if finished else 2
    except BaseException as error:
        old.write_json(run/'ERROR.json', dict(status='FAILED', utc=old.utc(), **identity, **progress,
                       error=str(error), traceback=traceback.format_exc(), automatic_retry=False,
                       elapsed_seconds=time.monotonic()-started))
        status('FAILED')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--kind', choices=('fno', 'deeponet'), required=True)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--probe-output')
    args = parser.parse_args()
    return probe(args.config, args.kind, args.probe_output) if args.probe else worker(args.config, args.kind)


if __name__ == '__main__':
    raise SystemExit(main())
