"""Run matched Trace-Spectral sampling and reflection experiments."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from contextlib import nullcontext
import subprocess
import time

import h5py
import numpy as np
import torch

from marmousi_symmetry import ReflectionTraceSpectral, reflect_inputs
from trace_spectral_v1 import full_train
from trace_spectral_v1.basis import SpaceBasis, TimeBasis
from trace_spectral_v1.input_pack import pack_input
from trace_spectral_v1.marmousi_data import atomic_json, load_bound, sha256
from trace_spectral_v1.mass_loss import WINDOWS
from trace_spectral_v1.prefetch_train import Prefetch, load_arrays
from trace_spectral_v1.train import code_digest, input_kwargs, save_state

ARMS = ('uniform', 'mixed', 'mixed_reflection')
SEED = 20261114


class Sampling:
    def __init__(self, rows):
        marmousi = defaultdict(list)
        families = defaultdict(list)
        for index, row in enumerate(rows):
            if 'family' in row:
                families[row['family']].append(index)
            else:
                marmousi[row['medium_id']].append(index)
        self.marmousi = list(marmousi.values())
        self.families = [families[name] for name in sorted(families)]
        self.count = len(rows)
        if len(self.marmousi) not in (55, 79) or list(map(len, self.families)) != [131, 130, 130]:
            raise ValueError('Unexpected dataset composition')

    def draw(self, step, batch, arm):
        rng = np.random.default_rng([SEED, step])
        if arm == 'uniform':
            return rng.choice(self.count, batch, replace=False)
        n_marm = round(batch / 3)
        ids = [rng.choice(self.marmousi[i]) for i in rng.choice(len(self.marmousi), n_marm, replace=False)]
        quotas = np.full(3, (batch - n_marm) // 3)
        remainder = (batch - n_marm) % 3
        quotas[(step + np.arange(remainder)) % 3] += 1
        for n, pool in zip(quotas, self.families, strict=True):
            ids.extend(rng.choice(pool, n, replace=False))
        return rng.permutation(ids)


@torch.inference_mode()
def evaluate(model, data, normalization, scale, device):
    model.eval()
    errors = {'direct': [], 'equivariant': []}
    rows = []
    with h5py.File(data, 'r') as file:
        media = file['medium_id'].asstr()[:]
        for i, name in enumerate(file['sample_id'].asstr()[:]):
            velocity = torch.from_numpy(file['velocity401'][i:i+1]).to(device)
            source = torch.from_numpy(file['source_solver'][i:i+1]).to(device)
            target = torch.from_numpy(file['wavefield'][i]).to(device, dtype=torch.float64)
            direct = model.predict_coefficients(pack_input(velocity, source, **normalization))
            vr, sr = reflect_inputs(velocity, source, model.space_x.length)
            reverse = model.predict_coefficients(pack_input(vr, sr, **normalization))
            symmetric = .5 * (direct + model.reflect_coefficients(reverse))
            row = dict(sample_id=name, medium_id=media[i])
            for key, coefficients in (('direct', direct), ('equivariant', symmetric)):
                pressure = model.decode_full(coefficients, pressure_scale=scale)[0]
                residual = pressure.double() - target
                row[key] = [float(residual[a:b].norm() / target[a:b].norm())
                            for a,b in ((0, len(target)), *WINDOWS)]
                errors[key].append(row[key])
            rows.append(row)
    if not all(np.isfinite(values).all() for values in errors.values()):
        raise ValueError('Nonfinite development metric')
    model.train()
    return dict(mean={key:np.mean(values,axis=0).tolist() for key,values in errors.items()}, rows=rows)


def batch_tensors(arrays, ids, device):
    return tuple(torch.from_numpy(array[ids]).to(device) for array in arrays)


def train_step(model, objective, optimizer, tensors, device, mirror, normalization):
    velocity, source, adjoint, statistics = tensors
    if mirror:
        velocity, source = reflect_inputs(velocity, source, model.space_x.length)
        adjoint = model.reflect_coefficients(adjoint)
    optimizer.zero_grad(set_to_none=True)
    context = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
    with context:
        coefficients = model.predict_coefficients(pack_input(velocity, source, **normalization))
    loss, rss = objective(coefficients, adjoint, statistics)
    if (rss < 0).any():
        loss, rss = objective(coefficients, adjoint, statistics, precision=torch.float64)
        if (rss < -1e-6 * statistics[...,0]).any():
            raise ValueError('Invalid negative pressure residual')
    loss = loss.mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)


@torch.no_grad()
def verify_reflection(model, objective, row, arrays, index, scale, normalization, device):
    with h5py.File(row['file'], 'r') as file:
        wave = file['wavefield'][:]
    direct, stats = objective.statistics(wave, scale)
    reflected, reflected_stats = objective.statistics(wave[..., ::-1].copy(), scale)
    torch.testing.assert_close(stats, reflected_stats, rtol=1e-10, atol=0)
    cached = torch.from_numpy(arrays[2][index].copy())
    if float((direct - cached).norm() / direct.norm()) >= 2e-6:
        raise ValueError('Existing cached adjoint disagrees with training label')
    mapped = model.reflect_coefficients(direct.to(device)).cpu()
    relative = float((reflected - mapped).norm() / direct.norm())
    if relative >= 2e-6:
        raise ValueError(f'Reflected adjoint discrepancy: {relative}')
    v,s,_,_ = batch_tensors(arrays, [index], device)
    vr,sr = reflect_inputs(v,s,model.space_x.length)
    direct_pred = model.predict_equivariant_coefficients(v,s,normalization)
    mirrored_pred = model.predict_equivariant_coefficients(vr,sr,normalization)
    torch.testing.assert_close(direct_pred, model.reflect_coefficients(mirrored_pred), rtol=1e-6, atol=1e-7)
    print(json.dumps(dict(event='reflection_verified', relative_adjoint=relative)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--interval', type=int, required=True)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    if args.steps <= 0 or args.interval <= 0 or not 3 <= args.batch <= 32 or args.lr <= 0:
        parser.error('Invalid training settings')
    if args.device == 'cuda':
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'],text=True).strip()
        if gpu:
            raise RuntimeError(f'GPU is already occupied by {gpu}')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    args.out.mkdir(exist_ok=False)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    manifest = load_bound(args.manifest, ck['digests']['manifest'])
    rows = [r for r in manifest['records'] if r['split']=='train']
    sampler = Sampling(rows)
    contract = manifest['split_indexes']['development']
    dev = Path(contract['file'])
    if sha256(dev) != contract['sha256'] or {r['medium_id'] for r in rows} & {
            r['medium_id'] for r in manifest['records'] if r['split']=='development'}:
        raise ValueError('Development index or medium split mismatch')
    space = SpaceBasis(length=ck['extent_m'])
    temporal = TimeBasis(bubbles=ck['bubbles'])
    model = ReflectionTraceSpectral(space, temporal).to(args.device)
    model.load_state_dict(ck['model'])
    objective = full_train.PressureObjective(space.matrix, space.matrix[:,1:],temporal.matrix).to(args.device)
    digests = dict(manifest=ck['digests']['manifest'], code=sha256(Path(full_train.__file__)),
                   base_code=code_digest(), basis=objective.basis_signature)
    if ck['digests'] != digests:
        raise ValueError('Model or data provenance mismatch')
    arrays = load_arrays(args.cache, digests['manifest'], digests['basis'], rows, digests['code'])
    meta = {key:ck[key] for key in ('extent_m','bubbles','pressure_scale_pa','normalization','digests')}
    normalization, scale = input_kwargs(ck), ck['pressure_scale_pa']
    optimizer = torch.optim.AdamW(model.parameters(), fused=args.device=='cuda')
    optimizer.load_state_dict(ck['optimizer'])
    for group in optimizer.param_groups:
        group.update(lr=args.lr, fused=args.device=='cuda', foreach=False, capturable=False)
    origin = ck['step']
    first = ck.get('sampling_updates', 0)
    del ck
    start = time.monotonic()
    config = dict(arm=args.arm, parent=str(args.checkpoint), parent_sha256=sha256(args.checkpoint),
                  parent_step=origin, first_sampling_update=first, device=args.device, lr=args.lr,
                  batch=args.batch, steps=args.steps, interval=args.interval, seed=SEED,
                  script_sha256=sha256(Path(__file__)),
                  reflection_sha256=sha256(Path(__file__).with_name('marmousi_symmetry.py')),
                  manifest_sha256=digests['manifest'])
    atomic_json(args.out/'RUN_CONFIG.json',config)
    verify_reflection(model, objective, rows[0], arrays, 0, scale, normalization, args.device)
    def status(stage, update, **fields):
        from os import getpid
        atomic_json(args.out/'STATUS.json', dict(stage=stage, pid=getpid(), update=update, target=args.steps,
                                               time=datetime.now(timezone.utc).isoformat(), **fields))
    baseline = evaluate(model,dev,normalization,scale,args.device)
    atomic_json(args.out/'DEV_0000.json',baseline)
    print(json.dumps(dict(event='development',update=0,mean=baseline['mean'])),flush=True)
    # Track the sum of full and late errors. The external selection also checks each metric.
    best = sum(baseline['mean']['equivariant'][i] for i in (0,3))
    pipe = Prefetch(arrays,args.batch) if args.device=='cuda' else None
    ids = sampler.draw(first+1,args.batch,args.arm)
    future = pipe.schedule(0,ids) if pipe else None
    slot = 0
    try:
        for update in range(1, args.steps + 1):
            step = first + update
            if pipe:
                tensors = pipe.take(slot,future).tensors
                if update < args.steps:
                    ids = sampler.draw(step+1,args.batch,args.arm)
                    future = pipe.schedule(1-slot,ids)
            else:
                tensors = batch_tensors(arrays,sampler.draw(step,args.batch,args.arm),args.device)
            flip = args.arm=='mixed_reflection' and np.random.default_rng([SEED,step,1]).random() < .5
            loss,norm = train_step(model,objective,optimizer,tensors,args.device,flip,normalization)
            if pipe:
                pipe.release(slot)
                slot = 1-slot
            if update == 1 or update % 10 == 0:
                status('TRAINING',update,loss=loss)
                print(json.dumps(dict(event='update',update=update,loss=loss,grad_norm=norm,
                                      seconds=time.monotonic()-start)),flush=True)
            if update % args.interval == 0 or update == args.steps:
                status('VALIDATING',update)
                metrics = evaluate(model,dev,normalization,scale,args.device)
                atomic_json(args.out/f'DEV_{update:04d}.json',metrics)
                p = dict(**meta,model=model.state_dict(),optimizer=optimizer.state_dict(),
                         step=origin+update,sampling_updates=step,arm=args.arm,inference='equivariant')
                path=args.out/f'step_{update:04d}.pt'
                save_state(path,p)
                full,_,_,late = metrics['mean']['equivariant']
                if full+late < best:
                    best=full+late
                    save_state(args.out/'best.pt',p)
                print(json.dumps(dict(event='development',update=update,mean=metrics['mean'])),flush=True)
        status('COMPLETE',args.steps)
    finally:
        if pipe:
            pipe.close()


if __name__ == '__main__':
    main()
