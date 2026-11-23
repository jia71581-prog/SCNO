#!/usr/bin/env python3
"""Explicit continuation of the two completed50 models with known8 and SIPG.

The controller owns authorization/ledger. Probe runs full forward/backward only,
with zero optimizer updates. All400 pressure frames are network outputs. Deep's
complete output adjoint is computed before chunked shared-trunk recomputation;
physical time halos therefore contribute once to every required output gradient.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
sys.dont_write_bytecode = True
import train_marmousi_baseline50_20261020 as base
import train_marmousi_matched_baselines_20261020 as old
from resume_marmousi_baseline50_deeponet_20261020 import tensor_stream_sha

SCHEMA = 'MARMOUSI_BASELINE_PDE_CONTINUATION_V1'
PARENT_UPDATES = {'fno': 400, 'deeponet': 200}


def load_parent(cfg, kind):
    import torch
    info = cfg['parents'][kind]
    for stem in ('checkpoint', 'config'):
        if old.sha(info[stem+'_path']) != info[stem+'_sha256']:
            raise ValueError('Frozen parent '+stem+' changed')
    saved = torch.load(info['checkpoint_path'], map_location='cpu', weights_only=False)
    parent_cfg = json.loads(Path(info['config_path']).read_text())
    if saved['schema'] != base.SCHEMA or saved['kind'] != kind or saved['config'] != parent_cfg:
        raise ValueError('Parent complete checkpoint identity differs')
    base.validate_progress(saved['progress'], parent_cfg, kind)
    progress = saved['progress']
    if (progress['completed_epochs'], progress['training_exposures'], progress['committed_updates']) != (50, 25000, PARENT_UPDATES[kind]):
        raise ValueError('Only completed original50 terminal parents are authorized')
    for key in ('normalization', 'pressure_rms_pa'):
        if cfg[key] != saved[key]:
            raise ValueError('Continuation changes original '+key)
    if cfg['training']['seed'] != parent_cfg['training']['seed']:
        raise ValueError('Continuation must preserve deterministic roster seed')
    return saved


def stage_schedule(optimizer, cfg, kind):
    import torch
    train = cfg['training']
    batch, _ = base.selection(cfg, kind)
    total = train['epochs']*math.ceil(train['records_per_epoch']/batch)
    # An explicitly new stage LR clock. Old Adam ages/moments remain untouched.
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] = train['lr']
    def factor(step):
        warm = train['warmup_updates']
        if warm and step < warm:
            return float(step+1)/warm
        fraction = min(1., max(0., (step-warm)/max(1, total-warm)))
        return train['min_lr_ratio']+(1-train['min_lr_ratio'])*.5*(1+math.cos(math.pi*fraction))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def migrate_parent(cfg, kind, saved, device='cpu', restore_rng=True):
    import torch
    from baselines.marmousi_pde_20261020.models import build_model
    # New adapter initialization is reproducible from the parent's CPU RNG;
    # the private initialization fork does not consume the inherited train RNG.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(saved['rng']['torch'])
        model = build_model(kind, parent_state=saved['model'], architecture=saved['config']['model'][kind]).to(device)
    optimizer = torch.optim.AdamW(model.parent.parameters(), lr=saved['config']['training']['lr'],
                                  weight_decay=saved['config']['training']['weight_decay'])
    optimizer.load_state_dict(saved['optimizer'])
    if not old.equal_tree(optimizer.state_dict(), saved['optimizer']):
        raise ValueError('Original complete Adam state did not migrate exactly')
    if not old.equal_tree(model.parent.state_dict(), saved['model']):
        raise ValueError('Original model tensors did not migrate exactly')
    if len(optimizer.param_groups) != 1:
        raise ValueError('Expected original single Adam group')
    optimizer.param_groups[0]['params'].extend(list(model.known8_adapter.parameters()))
    ages = sorted({int(state['step']) for state in optimizer.state.values()})
    if ages != [saved['progress']['committed_updates']] or any(p in optimizer.state for p in model.known8_adapter.parameters()):
        raise ValueError('Old Adam ages or new adapter age are incorrect')
    scheduler = stage_schedule(optimizer, cfg, kind)
    lineage = dict(parent_checkpoint_sha256=cfg['parents'][kind]['checkpoint_sha256'],
                   parent_config_sha256=cfg['parents'][kind]['config_sha256'],
                   parent_progress=old.cpu_tree(saved['progress']), parent_scheduler=old.cpu_tree(saved['scheduler']),
                   parent_rng=old.cpu_tree(saved['rng']), parent_Adam_step_ages=ages,
                   parent_model_and_Adam_tensors_exact=True,
                   adapter_parameters=sum(p.numel() for p in model.known8_adapter.parameters()), adapter_initial_Adam_age=0,
                   stage_LR_policy='explicit_new_cosine_warmup_clock; inherited_Adam_moments_and_step_ages',
                   adapter_initialization='parent_CPU_torch_RNG_fork; zero_last_convolution; inherited_RNG_unconsumed',
                   stage_roster_epoch_offset=50, parent_initialization='independent_random',
                   initial_function='zero_adapter_preserves_parent_function; future_updates_change_objective_and_LR')
    if restore_rng:
        old.set_rng(saved['rng'], device)
    return model, optimizer, scheduler, lineage


def initial_progress(cfg):
    from marmousi_baseline50_data_20261020 import epoch_order
    return dict(completed_epochs=0, active_epoch=1, offset=0, committed_updates=0,
                training_exposures=0, training_seconds=0.,
                roster=list(map(int, epoch_order(500, cfg['training']['seed'], 50))))


def validate_progress(progress, cfg, kind):
    from marmousi_baseline50_data_20261020 import epoch_order
    done, offset = progress['completed_epochs'], progress['offset']
    batch, _ = base.selection(cfg, kind)
    if not (0 <= done <= cfg['training']['epochs'] and 0 <= offset <= 500):
        raise ValueError('Stage cursor outside authorized budget')
    expected = [] if done == cfg['training']['epochs'] else list(map(int, epoch_order(500, cfg['training']['seed'], 50+done)))
    if (progress['training_exposures'] != done*500+offset or progress['active_epoch'] != done+1
            or progress['committed_updates'] != done*math.ceil(500/batch)+math.ceil(offset/batch)
            or (offset < 500 and offset % batch) or progress['roster'] != expected):
        raise ValueError('Stage complete-batch cursor/roster differs')


def restore_stage(cfg, kind, saved, device='cpu', restore_rng=True):
    import torch
    from baselines.marmousi_pde_20261020.models import build_model
    if saved['schema'] != SCHEMA or saved['config'] != cfg or saved['kind'] != kind:
        raise ValueError('Continuation complete checkpoint identity differs')
    validate_progress(saved['progress'], cfg, kind)
    model = build_model(kind, architecture=saved['parent_architecture']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    scheduler = stage_schedule(optimizer, cfg, kind)
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    for actual, expected in ((model.state_dict(), saved['model']), (optimizer.state_dict(), saved['optimizer']),
                             (scheduler.state_dict(), saved['scheduler'])):
        if not old.equal_tree(actual, expected):
            raise ValueError('Restored model/Adam/stage scheduler differs')
    if restore_rng:
        old.set_rng(saved['rng'], device)
    return model, optimizer, scheduler


def to_device(cpu, cfg, device):
    import torch
    data = base.to_device(cpu, cfg, device)
    data['inputs'] = torch.cat((data['inputs'], data['truth'][:, :8]), dim=1)
    data['sources'] = torch.as_tensor(cpu['source_model_fp32'], device=device, dtype=torch.float32)
    return data


def make_physics(dataset, cfg, device):
    import torch
    from baselines.marmousi_pde_20261020.physics import NonuniformSIPG
    return NonuniformSIPG(torch.from_numpy(dataset.times.copy()).to(device),
                          torch.from_numpy(dataset.velocity.copy()).to(device), **cfg['physics'])


def centers(count, chunk_frames, device):
    import torch
    for start in range(1, count-1, chunk_frames):
        yield torch.arange(start, min(start+chunk_frames, count-1), device=device)


def triplets(prediction, center):
    import torch
    indices = center[:, None]+torch.tensor([-1, 0, 1], device=center.device)[None]
    return prediction[:, indices]


def predict_normalized(model, kind, data, chunk):
    import torch
    with torch.no_grad():
        if kind == 'fno':
            return model(data['inputs'])
        branch = model.encode_branch(data['inputs'])
        result = torch.empty_like(data['truth'])
        points = math.prod(data['shape'])
        for start in range(0, points, chunk):
            stop = min(start+chunk, points)
            coords, _, _, _ = old.point_coordinates(start, stop, data['shape'], data['times'], data['norm'])
            result.reshape(len(result), -1)[:, start:stop] = model.decode_trunk(branch, coords)
        return result


def output_objective(prediction, data, sup, physics, scales, cfg, backward=False, multiplier=1.):
    """Physical halos share one global output leaf; each interior center appears once."""
    import torch
    from marmousi_baseline50_data_20261020 import weighted_loss
    train = cfg['training']
    loss, field, receiver = weighted_loss(prediction, data['truth'], sup, train['receiver_weight'])
    if backward:
        (loss*multiplier).backward()
    pde = torch.zeros((), device=prediction.device)
    for center in centers(prediction.shape[1], train['physics_chunk_frames'], prediction.device):
        term = physics.loss_chunk(triplets(prediction, center), center, data['sources'], scales)
        if backward:
            (term*train['pde_weight']*multiplier).backward()
        pde += term.detach()
    return dict(loss=float(loss.detach()+train['pde_weight']*pde), field_loss=float(field.detach()),
                receiver_loss=float(receiver.detach()), pde_loss=float(pde))


def backward_complete(model, kind, data, sup, physics, scales, cfg, chunk, loss_multiplier=1.):
    """Exact first derivative, without keeping all398 SIPG/trunk graphs alive."""
    import torch
    if kind == 'fno':
        actual = model(data['inputs'])
        leaf = actual.detach().requires_grad_(True)
        losses = output_objective(leaf, data, sup, physics, scales, cfg, True, loss_multiplier)
        actual.backward(leaf.grad)
        return losses
    branch = model.encode_branch(data['inputs'])
    surrogate = branch.detach().requires_grad_(True)
    points = math.prod(data['shape'])
    with torch.no_grad():
        leaf = torch.empty_like(data['truth'])
        for start in range(0, points, chunk):
            stop = min(start+chunk, points)
            coords, _, _, _ = old.point_coordinates(start, stop, data['shape'], data['times'], data['norm'])
            leaf.reshape(len(leaf), -1)[:, start:stop] = model.decode_trunk(surrogate, coords)
    leaf.requires_grad_(True)
    losses = output_objective(leaf, data, sup, physics, scales, cfg, True, loss_multiplier)
    adjoint = leaf.grad.reshape(len(leaf), -1)
    for start in range(0, points, chunk):
        stop = min(start+chunk, points)
        coords, _, _, _ = old.point_coordinates(start, stop, data['shape'], data['times'], data['norm'])
        model.decode_trunk(surrogate, coords).backward(adjoint[:, start:stop])
    if surrogate.grad is None:
        raise ValueError('Missing complete output-to-branch derivative')
    branch.backward(surrogate.grad)
    return losses


def gate_decision(initial, current, cfg):
    rule = cfg['gate']
    if initial['indices'] != rule['panel'] or current['indices'] != rule['panel']:
        raise ValueError('Gate panel identities changed')
    before, after = initial['cases'], current['cases']
    if any(len(rows)!=len(rule['panel']) or [row['index'] for row in rows]!=rule['panel'] for rows in (before,after)):
        raise ValueError('Gate requires all four original cases in their frozen order')
    start_mean = sum(row['future_relative_l2'] for row in before)/len(before)
    final_mean = sum(row['future_relative_l2'] for row in after)/len(after)
    checks = dict(finite=all(math.isfinite(v) for row in after for v in row.values() if isinstance(v, (int, float))),
                  future_mean=final_mean <= min(rule['future_mean_absolute_max'], rule['future_mean_relative_max']*start_mean),
                  every_case_future=all(b['future_relative_l2'] <= rule['per_case_relative_max']*a['future_relative_l2'] for a,b in zip(before,after)),
                  every_case_amplitude=all(rule['norm_ratio_min'] <= row['future_norm_ratio'] <= rule['norm_ratio_max'] for row in after),
                  every_case_pde=all(row['pde_loss'] <= rule['pde_max'] for row in after),
                  data_improved=sum(row['field_loss'] for row in after) <= rule['data_mean_relative_max']*sum(row['field_loss'] for row in before),
                  full_field_not_degraded=sum(row['full_relative_l2'] for row in after) <= rule['full_mean_relative_max']*sum(row['full_relative_l2'] for row in before))
    return dict(status='PASS' if all(checks.values()) else 'FAILED_EFFECTIVENESS', checks=checks,
                fixed_rule=rule, before=initial, after=current,
                rule_frozen_before_training=True, paper_heldout_used_for_gate=False)


def assess_panel(model, kind, dataset, physics, cfg, device, chunk):
    import torch
    cases = []
    for index in cfg['gate']['panel']:
        cpu = dataset.load_batch([index])
        data = to_device(cpu, cfg, device)
        with torch.no_grad():
            sup = base.supervision(data['truth'], cfg)
            scales = physics.fixed_scales(data['truth'], data['sources'], cfg['training']['physics_chunk_frames'])
            prediction = predict_normalized(model, kind, data, chunk)
            metrics = output_objective(prediction, data, sup, physics, scales, cfg)
            reference = data['truth'][:, 8:].double()
            future = prediction[:, 8:].double()
            energy = reference.square().sum()
            pred_energy = future.square().sum()
            metrics.update(index=index, future_relative_l2=float(((future-reference).square().sum()/energy).sqrt()),
                           full_relative_l2=float(((prediction.double()-data['truth'].double()).square().sum()/data['truth'].double().square().sum()).sqrt()),
                           future_norm_ratio=float((pred_energy/energy).sqrt()),
                           future_cosine=float((future*reference).sum()/(energy*pred_energy).sqrt().clamp_min(1e-30)),
                           known8_relative_l2=float(((prediction[:, :8].double()-data['truth'][:, :8].double()).square().sum()/data['truth'][:, :8].double().square().sum()).sqrt()))
            altered = dict(data, inputs=data['inputs'].clone())
            altered['inputs'][:, 8:] = 0
            zero = predict_normalized(model, kind, altered, chunk)
            metrics['known8_zero_change_relative_l2'] = float(((prediction[:, 8:]-zero[:, 8:]).double().square().sum()/energy).sqrt())
            del zero
            other_index = cfg['gate']['panel'][(cfg['gate']['panel'].index(index)+1)%len(cfg['gate']['panel'])]
            other = dataset.load_batch([other_index])
            altered['inputs'][:, 8:] = other['pressure_pa'][:, :8].to(device)/cfg['pressure_rms_pa']
            changed = predict_normalized(model, kind, altered, chunk)
            metrics['known8_swapped_change_relative_l2'] = float(((prediction[:, 8:]-changed[:, 8:]).double().square().sum()/energy).sqrt())
        cases.append(metrics)
        del cpu, data, sup, scales, prediction, reference, future, altered, other, changed
    return dict(indices=cfg['gate']['panel'], cases=cases, input_pressure_frames=8, output_frames=400,
                primary_frames=[8,400], known8_is_input_only=True, utc=old.utc())


def assess_paper(model, kind, cfg, device, trained_times, chunk):
    import numpy as np
    import torch
    paper = base.paper_data(cfg, device)
    paper['inputs'] = torch.cat((paper['inputs'], paper['truth'][None,:8]), dim=1)
    return base.assess_paper(model, kind, cfg, paper, trained_times, chunk)


def train_prediction_witness(model, kind, dataset, cfg, device, chunk):
    """A fixed TRAIN case, with all400 predicted frames; no intermediate heldout use."""
    data=to_device(dataset.load_batch([cfg['gate']['panel'][0]]),cfg,device)
    prediction=predict_normalized(model,kind,data,chunk)
    return (prediction*cfg['pressure_rms_pa']).cpu().numpy()


def cpu_preflight(config_path, kind):
    import torch
    started = time.monotonic()
    torch.set_num_threads(2)
    cfg = json.loads(Path(config_path).read_text())
    saved = load_parent(cfg, kind)
    model, optimizer, scheduler, lineage = migrate_parent(cfg, kind, saved, 'cpu', restore_rng=False)
    dataset = base.make_dataset(cfg)
    if not old.equal_tree(dataset.state_dict(), saved['data_state']) or not torch.equal(torch.from_numpy(dataset.times.copy()), saved['trained_times']):
        raise ValueError('Continuation mathematical data/normalizer/time changed')
    with torch.no_grad():
        maps = torch.randn(2,8,33,33)
        combined = torch.cat((maps, torch.randn_like(maps)),dim=1)
        if kind == 'fno':
            initial_equal = torch.equal(model(combined), model.parent(maps))
        else:
            coords = torch.rand(19,3)
            initial_equal = torch.equal(model.decode_trunk(model.encode_branch(combined),coords),
                                         model.parent.decode_trunk(model.parent.encode_branch(maps),coords))
    if not initial_equal or torch.cuda.is_initialized():
        raise ValueError('CPU parent function preservation/CUDA isolation failed')
    old.set_rng(saved['rng'],'cpu')
    actual_rng=old.get_rng('cpu')
    if not all(old.equal_tree(actual_rng[key],saved['rng'][key]) for key in ('python','numpy','torch')):
        raise ValueError('Parent actual CPU RNG restoration differs')
    report = dict(status='PASS', kind=kind, parent_checkpoint_sha256=cfg['parents'][kind]['checkpoint_sha256'],
                  parent_model_Adam_and_age_preserved=True, parent_function_bitwise_equal=True,
                  data_state_and_time_bitwise_equal=True, new_adapter_parameters=lineage['adapter_parameters'], new_adapter_Adam_state='absent_age0',
                  CUDA_initialized=False, optimizer_updates=0, parent_CPU_RNG_actual_restore=True, lineage=lineage,
                  worker_sha256=old.sha(__file__), seconds=time.monotonic()-started)
    # RNG arrays are checkpoint payload, never JSON. The lineage records its hash below.
    report['lineage'] = {k:v for k,v in lineage.items() if k != 'parent_rng'}
    report['parent_rng_preserved_for_actual_launch'] = True
    output = Path(cfg['run_dir'])/kind
    output.mkdir(parents=True, exist_ok=True)
    old.write_json(output/'CPU_PREFLIGHT.json', report)
    return report


def scales_for(data, physics, cache, cfg):
    """Each immutable teacher record supplies a fixed PDE scale, never prediction RMS."""
    import torch
    missing = [i for i, index in enumerate(data['ids']) if str(int(index)) not in cache]
    if missing:
        values = physics.fixed_scales(data['truth'][missing], data['sources'][missing], cfg['training']['physics_chunk_frames'])
        for position, value in zip(missing, values):
            cache[str(int(data['ids'][position]))] = float(value)
    return torch.tensor([cache[str(int(index))] for index in data['ids']], device=data['truth'].device, dtype=torch.float32)


def data_witness(data, sup):
    return dict(inputs=tensor_stream_sha(data['inputs']), truth=tensor_stream_sha(data['truth']),
                times=tensor_stream_sha(data['times']), sources=tensor_stream_sha(data['sources']),
                supervision={key:tensor_stream_sha(value) for key,value in sup.items()}, ids=list(map(int,data['ids'])))


def public_lineage(lineage):
    return {key:value for key,value in lineage.items() if key != 'parent_rng'}


def checked_bindings(cfg, config_path):
    result = base.bindings(cfg, config_path)
    for info in cfg['parents'].values():
        for stem in ('checkpoint', 'config'):
            path = info[stem+'_path']
            if old.sha(path) != info[stem+'_sha256']:
                raise ValueError('Frozen parent binding changed: '+path)
            result['files'][path] = info[stem+'_sha256']
    return result


def probe(config_path, kind, output=None):
    """Full real DATA+PDE gradients; no optimizer step, no candidate retries."""
    import torch
    cfg = json.loads(Path(config_path).read_text())
    run = Path(cfg['run_dir']).resolve()/kind
    if not run.is_relative_to(ROOT/'runs'):
        raise ValueError('Probe output outside project runs')
    run.mkdir(parents=True, exist_ok=True)
    output = Path(output).resolve() if output else run/'PROBE.json'
    if not output.is_relative_to(Path(cfg['run_dir']).resolve()) or output.exists():
        raise ValueError('Probe output must be new evidence within this run')
    pins = checked_bindings(cfg, config_path)
    with (run/'PROBE_WORKER.json').open('x') as handle:
        json.dump(dict(**old.identity(), kind=kind, optimizer_updates=0),handle,indent=2)
    started = time.monotonic()
    base.configure_runtime(cfg)
    device = cfg.get('device','cuda:0')
    parent = load_parent(cfg,kind)
    model, optimizer, scheduler, lineage = migrate_parent(cfg,kind,parent,device)
    # Retain restored Adam tensors so measured residency includes production state.
    del parent
    dataset = base.make_dataset(cfg)
    physics = make_physics(dataset,cfg,device)
    cache, records, selected = {}, [], None
    try:
        cpu = dataset.load_batch([0,1])
        data = to_device(cpu,cfg,device)
        scale = scales_for(data,physics,cache,cfg)
        chunk0 = min(cfg['probe']['chunk_candidates'])
        model.zero_grad(set_to_none=True)
        batched = backward_complete(model,kind,data,base.supervision(data['truth'],cfg),physics,scale,cfg,chunk0)
        if not base.gradients_finite(model) or not all(math.isfinite(v) for v in batched.values()):
            raise ValueError('Initial B2 full DATA+PDE gradients are not finite')
        expected = [p.grad.detach().cpu().clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        singles = []
        for i in range(2):
            one = dict(data, inputs=data['inputs'][i:i+1], truth=data['truth'][i:i+1], sources=data['sources'][i:i+1], ids=data['ids'][i:i+1])
            singles.append(backward_complete(model,kind,one,base.supervision(one['truth'],cfg),physics,scale[i:i+1],cfg,chunk0,.5)['loss'])
        numerator = denominator = 0.
        for expected_grad, parameter in zip(expected,model.parameters()):
            actual = parameter.grad.detach().cpu()
            numerator += float((actual.double()-expected_grad.double()).square().sum())
            denominator += float(expected_grad.double().square().sum())
        gradient_error = math.sqrt(numerator/max(denominator,1e-300))
        loss_error = abs(batched['loss']-sum(singles)/2)/max(abs(batched['loss']),1e-30)
        if (not all(math.isfinite(v) for v in (gradient_error,loss_error,*singles))
                or gradient_error > 2e-4 or loss_error > 2e-5 or not base.gradients_finite(model)):
            raise ValueError(f'TrueB2/meanB1 DATA+PDE mismatch: {gradient_error}, {loss_error}')
        numerical = dict(status='PASS',gradient_relative_l2=gradient_error,loss_relative_error=loss_error,
                         batch_loss=batched['loss'],mean_single_loss=sum(singles)/2,optimizer_updates=0)
        del cpu,data,scale,one,expected,actual,expected_grad
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        limit = cfg['probe']['max_reserved_gib']*(1<<30)
        previous = None
        def measure(batch,chunk):
            nonlocal previous,selected
            free,_ = torch.cuda.mem_get_info(device)
            current = torch.cuda.memory_reserved(device)
            predicted = None if previous is None else previous['peak_reserved_bytes']*batch/previous['batch_size']*max(1.,chunk/previous['chunk_points'])
            if predicted is not None and (predicted > limit or predicted > free+current):
                records.append(dict(status='SKIPPED_PREDICTED_MEMORY',batch_size=batch,chunk_points=chunk,predicted_reserved_bytes=predicted,actual_free_bytes=free))
                return
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            cpu_batch = dataset.load_batch(list(range(batch)))
            batch_data = to_device(cpu_batch,cfg,device)
            sup = base.supervision(batch_data['truth'],cfg)
            scale_start = time.monotonic()
            scales = scales_for(batch_data,physics,cache,cfg)
            torch.cuda.synchronize(device)
            scale_seconds = time.monotonic()-scale_start
            tick = time.monotonic()
            losses = backward_complete(model,kind,batch_data,sup,physics,scales,cfg,chunk)
            if not base.gradients_finite(model) or not all(math.isfinite(v) for v in losses.values()):
                raise ValueError('Nonfinite real probe gradients')
            torch.cuda.synchronize(device)
            seconds = time.monotonic()-tick
            row = dict(status='PASS',batch_size=batch,chunk_points=chunk,seconds=seconds,samples_per_second=batch/seconds,
                       scale_prepare_seconds=scale_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device),gradients_finite=True,**losses)
            if row['peak_reserved_bytes'] > limit:
                row['status'] = 'EXCEEDS_CONCURRENT_ADMISSION'
            records.append(row)
            previous = row
            if row['status']=='PASS' and (selected is None or row['samples_per_second']>selected['samples_per_second']):
                selected = {key:row[key] for key in ('batch_size','chunk_points','samples_per_second','peak_allocated_bytes','peak_reserved_bytes')}
            del cpu_batch,batch_data,sup,scales
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        for batch in cfg['probe']['batch_candidates']:
            if time.monotonic()-started > cfg['probe'].get('maximum_seconds',1800):
                break
            measure(batch,chunk0)
        if selected is not None and kind=='deeponet':
            batch=selected['batch_size']
            for chunk in cfg['probe']['chunk_candidates']:
                if chunk != chunk0 and time.monotonic()-started <= cfg['probe'].get('maximum_seconds',1800):
                    measure(batch,chunk)
        if selected is None or checked_bindings(cfg,config_path)!=pins:
            raise ValueError('No actual admissible probe or bindings drifted')
        old.write_json(output,dict(status='PASS',kind=kind,optimizer_updates=0,selected=selected,candidates=records,
                                  actual_GPU_B2_vs_B1=numerical,physics=physics.metadata(),
                                  physical_scale_records_cached=len(cache),elapsed_seconds=time.monotonic()-started,**pins))
        return 0
    except BaseException as error:
        old.write_json(output,dict(status='FAILED',kind=kind,optimizer_updates=0,selected=None,candidates=records,
                                  error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started,**pins))
        raise


def worker(config_path, kind):
    import numpy as np
    import torch
    from baselines.marmousi_pde_20261020.models import parameter_report, CONTINUED_PARAMETERS
    from marmousi_baseline50_data_20261020 import Prefetch, epoch_order
    cfg=json.loads(Path(config_path).read_text())
    run=Path(cfg['run_dir']).resolve()/kind
    if not run.is_relative_to(ROOT/'runs'):
        raise ValueError('Only new project run outputs authorized')
    base.verify_probe(cfg,kind)
    train=cfg['training']
    if (train['epochs']!=50 or train['records_per_epoch']!=500 or cfg['gate']['epoch']!=5
            or cfg['target_parameters']!=CONTINUED_PARAMETERS):
        raise ValueError('Expected authorized50-stage budget with gate5 and current parameter floor')
    pins=checked_bindings(cfg,config_path)
    run.mkdir(parents=True,exist_ok=True)
    identity=dict(**old.identity(),kind=kind)
    with (run/'WORKER.json').open('x') as handle:
        json.dump(identity,handle,indent=2)
    started=time.monotonic()
    device=cfg.get('device','cuda:0')
    batch,chunk=base.selection(cfg,kind)
    progress=initial_progress(cfg)
    scales_cache={}
    restore8=None
    gate=None
    stopped=None
    latest_loss=None
    def status(state):
        old.write_json(run/'STATUS.json',dict(status=state,utc=old.utc(),**identity,**progress,
                       parent_completed_epochs=50,parent_updates=PARENT_UPDATES[kind],parent_exposures=25000,
                       cumulative_epochs=50+progress['completed_epochs'],cumulative_updates=PARENT_UPDATES[kind]+progress['committed_updates'],
                       cumulative_exposures=25000+progress['training_exposures'],target_additional_epochs=50,
                       batch_size=batch,chunk_points=chunk,loss=latest_loss,elapsed_seconds=time.monotonic()-started))
    def save():
        if checked_bindings(cfg,config_path)!=pins:
            raise ValueError('Frozen new source/config/parent changed')
        validate_progress(progress,cfg,kind)
        payload=dict(schema=SCHEMA,kind=kind,config=cfg,progress=old.cpu_tree(progress),
                     model=old.cpu_tree(model.state_dict()),optimizer=old.cpu_tree(optimizer.state_dict()),
                     scheduler=old.cpu_tree(scheduler.state_dict()),rng=old.get_rng(device),
                     data_state=dataset.state_dict(),trained_times=trained_times.cpu().clone(),
                     normalization=cfg['normalization'],pressure_rms_pa=cfg['pressure_rms_pa'],
                     parent_architecture=parent_architecture,lineage=lineage,bindings=pins,
                     physics_metadata=physics.metadata(),physics_scales=dict(scales_cache),
                     initial_train_panel=initial_panel,effectiveness_gate=gate)
        saved=old.atomic_checkpoint(run/'latest.pt',payload)
        receipt=dict(utc=old.utc(),sha256=old.sha(run/'latest.pt'),committed_updates=progress['committed_updates'],
                     completed_epochs=progress['completed_epochs'],training_exposures=progress['training_exposures'],complete_readback=True)
        old.write_json(run/'CHECKPOINT.json',receipt)
        return saved,receipt
    def cpu_restore(saved,receipt):
        restored=restore_stage(cfg,kind,saved,'cpu',restore_rng=False)
        rebuilt_dataset=base.make_dataset(cfg)
        rebuilt_physics=make_physics(rebuilt_dataset,cfg,'cpu')
        if (not old.equal_tree(rebuilt_dataset.state_dict(),saved['data_state'])
                or not torch.equal(torch.from_numpy(rebuilt_dataset.times.copy()),saved['trained_times'])
                or rebuilt_physics.metadata()!=saved['physics_metadata']
                or saved['normalization']!=cfg['normalization'] or saved['pressure_rms_pa']!=cfg['pressure_rms_pa']
                or not old.equal_tree(saved['initial_train_panel'],initial_panel)
                or not old.equal_tree(saved['effectiveness_gate'],gate)
                or not old.equal_tree(saved['lineage'],lineage)):
            raise ValueError('CPU restored data/physics/normalizer/initial-panel/lineage differs')
        restored_cache={str(int(key)):float(torch.tensor(value,dtype=torch.float32)) for key,value in saved['physics_scales'].items()}
        if (restored_cache!=saved['physics_scales'] or restored_cache!=scales_cache
                or not all(0<=int(key)<500 and math.isfinite(value) and value>0 for key,value in restored_cache.items())):
            raise ValueError('CPU restored full physical scale cache differs')
        old.set_rng(saved['rng'],'cpu')
        actual_rng=old.get_rng('cpu')
        if not all(old.equal_tree(actual_rng[key],saved['rng'][key]) for key in ('python','numpy','torch')):
            raise ValueError('Actual CPU RNG restoration differs')
        result=dict(status='PASS',checkpoint_sha256=receipt['sha256'],committed_updates=progress['committed_updates'],
                    completed_epochs=progress['completed_epochs'],training_exposures=progress['training_exposures'],
                    complete_model_populated_Adam_stage_scheduler_RNG_roster=True,
                    actual_CPU_RNG_restore=True,CUDA_RNG_payload_preserved=True,
                    data_state_physics_scale_cache_normalizer_time_and_panel_chain=True,
                    parent_Adam_step_age=PARENT_UPDATES[kind]+progress['committed_updates'],
                    adapter_Adam_step_age=progress['committed_updates'])
        del restored,rebuilt_dataset,rebuilt_physics
        old.set_rng(saved['rng'],device)
        return result
    try:
        base.configure_runtime(cfg)
        status('RESTORING_PARENT')
        parent=load_parent(cfg,kind)
        dataset=base.make_dataset(cfg)
        if not old.equal_tree(dataset.state_dict(),parent['data_state']):
            raise ValueError('Original dataset/normalizer bindings differ')
        trained_times=torch.from_numpy(dataset.times.copy()).to(device)
        if not torch.equal(trained_times.cpu(),parent['trained_times']):
            raise ValueError('Original complete400 time axis differs')
        model,optimizer,scheduler,lineage=migrate_parent(cfg,kind,parent,device)
        parent_architecture=parent['config']['model'][kind]
        del parent
        physics=make_physics(dataset,cfg,device)
        report=parameter_report(model,cfg['target_parameters'])
        if not report['exact_match'] or report['frozen_real_scalars']:
            raise ValueError('All active real parameters must satisfy agreed count')
        old.write_json(run/'SELECTION.json',report)
        old.write_json(run/'MIGRATION.json',dict(status='PASS',**public_lineage(lineage)))
        initial_panel=assess_panel(model,kind,dataset,physics,cfg,device,chunk)
        old.write_json(run/'STAGE0_TRAIN_PANEL.json',initial_panel)
        save()
        with (run/'metrics.jsonl').open('x',buffering=1) as metrics_log,(run/'EPOCHS.jsonl').open('x',buffering=1) as epoch_log:
            while progress['completed_epochs']<train['epochs'] and stopped is None:
                epoch=progress['completed_epochs']+1
                while progress['offset']<500 and stopped is None:
                    remaining=progress['roster'][progress['offset']:]
                    batches=[remaining[i:i+batch] for i in range(0,len(remaining),batch)]
                    prefetch=Prefetch(dataset,batches,depth=cfg['data'].get('prefetch_depth',1),workers=cfg['data'].get('prefetch_workers',1))
                    need_restore=False
                    try:
                        for indices,cpu in zip(batches,prefetch):
                            if any(path.exists() for path in (run/'STOP',run/'STOP_REQUEST.json',run.parent/'STOP_REQUEST.json')):
                                stopped='explicit_run_stop_requested'
                                break
                            if list(map(int,cpu['ids']))!=indices:
                                raise ValueError('Prefetch differs from durable complete-batch order')
                            tick=time.monotonic()
                            data=to_device(cpu,cfg,device)
                            if not torch.equal(data['times'],trained_times):
                                raise ValueError('Training axis changed')
                            sup=base.supervision(data['truth'],cfg)
                            scales=scales_for(data,physics,scales_cache,cfg)
                            optimizer.zero_grad(set_to_none=True)
                            losses=backward_complete(model,kind,data,sup,physics,scales,cfg,chunk)
                            if not base.gradients_finite(model) or not all(math.isfinite(v) for v in losses.values()):
                                raise ValueError('Incomplete/nonfinite full DATA+PDE gradient')
                            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),train['grad_clip'],error_if_nonfinite=True)
                            optimizer.step()
                            scheduler.step()
                            if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
                                raise ValueError('Nonfinite optimizer endpoint')
                            torch.cuda.synchronize(device)
                            seconds=time.monotonic()-tick
                            progress['offset']+=len(indices)
                            progress['committed_updates']+=1
                            progress['training_exposures']+=len(indices)
                            progress['training_seconds']+=seconds
                            latest_loss=losses['loss']
                            row=dict(utc=old.utc(),epoch=epoch,offset=progress['offset'],step=progress['committed_updates'],
                                     training_exposures=progress['training_exposures'],batch_size=len(indices),seconds=seconds,
                                     samples_per_second=len(indices)/seconds,gradient_norm=float(norm),
                                     parent_Adam_age=PARENT_UPDATES[kind]+progress['committed_updates'],adapter_Adam_age=progress['committed_updates'],
                                     stage_lr=optimizer.param_groups[0]['lr'],prefetch=cpu.get('prefetch'),**losses)
                            metrics_log.write(json.dumps(row,allow_nan=False)+'\n')
                            print(json.dumps(row),flush=True)
                            status('RUNNING')
                            if progress['committed_updates']==8:
                                prefetch.close()
                                witness=data_witness(data,sup)
                                saved,receipt=save()
                                prediction_before=train_prediction_witness(model,kind,dataset,cfg,device,chunk)
                                restore_indices=list(indices)
                                del data,sup,scales,cpu
                                need_restore=True
                                break
                            if progress['committed_updates']==9 and restore8 is not None:
                                restore8.update(status='PASS',restored_objects_committed_update9=True,utc=old.utc())
                                old.write_json(run/'CUDA_RESTORE8.json',restore8)
                            if progress['committed_updates']%train['save_every_updates']==0:
                                save()
                            del data,sup,scales,cpu
                    finally:
                        prefetch.close()
                        del prefetch
                    if need_restore:
                        # Drop every reference to old cache/model before allocating replacements.
                        del dataset,physics,model,optimizer,scheduler
                        gc.collect()
                        torch.cuda.empty_cache()
                        model,optimizer,scheduler=restore_stage(cfg,kind,saved,device)
                        dataset=base.make_dataset(cfg)
                        physics=make_physics(dataset,cfg,device)
                        if not old.equal_tree(dataset.state_dict(),saved['data_state']) or physics.metadata()!=saved['physics_metadata']:
                            raise ValueError('Restored dataset or physical discretization differs')
                        reloaded=to_device(dataset.load_batch(restore_indices),cfg,device)
                        if data_witness(reloaded,base.supervision(reloaded['truth'],cfg))!=witness:
                            raise ValueError('Full native batch data/known8/normalizer/time reconstruction differs')
                        scales_cache=dict(saved['physics_scales'])
                        progress=old.cpu_tree(saved['progress'])
                        prediction_after=train_prediction_witness(model,kind,dataset,cfg,device,chunk)
                        if not np.array_equal(prediction_before,prediction_after) or not old.equal_tree(old.get_rng(device),saved['rng']):
                            raise ValueError('New-object CUDA prediction or RNG differs')
                        restore8=dict(status='RESTORED_AWAITING_UPDATE9',checkpoint_sha256=receipt['sha256'],committed_updates=8,
                                      next_update=9,complete_state_exact=True,pressure_bitwise_equal=True,
                                      full_batch_streaming_witness_equal=True,restored_complete_physics_scale_cache=True,
                                      witness_train_source=cfg['gate']['panel'][0],paper_heldout_used=False)
                        old.write_json(run/'CUDA_RESTORE8.json',restore8)
                        del reloaded,saved,prediction_before,prediction_after,witness
                if stopped is not None:
                    break
                if progress['offset']!=500:
                    raise ValueError('Epoch ended without all500 original sources')
                progress['completed_epochs']=epoch
                progress['active_epoch']=epoch+1
                progress['offset']=0
                progress['roster']=[] if epoch==train['epochs'] else list(map(int,epoch_order(500,train['seed'],50+epoch)))
                panel=assess_panel(model,kind,dataset,physics,cfg,device,chunk)
                if epoch==cfg['gate']['epoch']:
                    gate=gate_decision(initial_panel,panel,cfg)
                    gate.update(completed_epochs=epoch,committed_updates=progress['committed_updates'],utc=old.utc())
                    old.write_json(run/'WORKER_GATE.json',gate)
                    if gate['status']!='PASS':
                        stopped='pre_registered_train_effectiveness_gate_failed'
                saved,receipt=save()
                restore=cpu_restore(saved,receipt)
                old.write_json(run/'LATEST_CPU_RESTORE.json',restore)
                epoch_row=dict(epoch=epoch,epoch_exposures=500,training_exposures=progress['training_exposures'],
                               committed_updates=progress['committed_updates'],train_panel=panel,
                               checkpoint_sha256=receipt['sha256'],CPU_restore=restore,effectiveness_gate=None if gate is None else gate['status'])
                epoch_log.write(json.dumps(epoch_row,allow_nan=False)+'\n')
                old.write_json(run/'LATEST_ASSESSMENT.json',epoch_row)
                del saved
                status('RUNNING' if stopped is None else 'FAILED_EFFECTIVENESS')
        saved,receipt=save()
        final_prediction,final_metrics,protocol=assess_paper(model,kind,cfg,device,trained_times,chunk)
        restore=cpu_restore(saved,receipt)
        old.write_json(run/'FINAL_CPU_RESTORE.json',restore)
        with (run/'.prediction_pa.npy.tmp').open('wb') as handle:
            np.save(handle,final_prediction,allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(run/'.prediction_pa.npy.tmp',run/'prediction_pa.npy')
        finished=progress['completed_epochs']==50 and progress['training_exposures']==25000 and gate is not None and gate['status']=='PASS'
        result=dict(status='COMPLETE' if finished else ('FAILED_EFFECTIVENESS' if gate is not None and gate['status']!='PASS' else 'INTERRUPTED'),
                    kind=kind,utc=old.utc(),**progress,target_epochs=50,exposures=progress['training_exposures'],
                    cumulative_epochs=50+progress['completed_epochs'],cumulative_exposures=25000+progress['training_exposures'],
                    cumulative_updates=PARENT_UPDATES[kind]+progress['committed_updates'],initialization='continued_original50_terminal',
                    input_pressure_frames=8,output_frames=400,output_true_frames_copied=0,parameters=report,batch_size=batch,chunk_points=chunk,
                    lineage=public_lineage(lineage),final_metrics=final_metrics,evaluation_time_protocol=protocol,
                    checkpoint_sha256=receipt['sha256'],prediction_sha256=old.sha(run/'prediction_pa.npy'),
                    final_CPU_restore=restore,CUDA_restore8=restore8,effectiveness_gate=gate,stop_reason=stopped,
                    training_objective='per-example original full400 four-window DATA + .01 receiver + fixed .001 nonuniform SIPG; batch mean; no false zero IC at first observed time',
                    elapsed_seconds=time.monotonic()-started)
        old.write_json(run/'RESULT.json',result)
        status(result['status'])
        return 0 if finished else 2
    except BaseException as error:
        old.write_json(run/'ERROR.json',dict(status='FAILED',utc=old.utc(),**identity,**progress,error=str(error),
                       traceback=traceback.format_exc(),automatic_retry=False,elapsed_seconds=time.monotonic()-started))
        status('FAILED')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--kind',choices=('fno','deeponet'),required=True)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--probe-output')
    parser.add_argument('--cpu-preflight',action='store_true')
    args=parser.parse_args()
    if args.cpu_preflight:
        cpu_preflight(args.config,args.kind)
        return 0
    return probe(args.config,args.kind,args.probe_output) if args.probe else worker(args.config,args.kind)


if __name__=='__main__':
    raise SystemExit(main())
