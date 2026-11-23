"""Fine-tune Trace-Spectral heads against reflection-equivariant pressure."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
import torch
from torch import nn

from marmousi_symmetry import ReflectionTraceSpectral
from trace_spectral_v1 import full_train
from trace_spectral_v1.basis import SpaceBasis, TimeBasis
from trace_spectral_v1.marmousi_data import atomic_json, load_bound, sha256
from trace_spectral_v1.mass_loss import WINDOWS
from extra_cache import join
from trace_spectral_v1.train import code_digest, input_kwargs, save_state
from train_marmousi import Sampling

SEED = 20261114
MIN_LATE_GAIN = .005


class EquivariantFineTune(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.base.encoder.requires_grad_(False)

    def forward(self, velocity, source, normalization):
        return self.base.predict_equivariant_coefficients(velocity, source, normalization)


@torch.inference_mode()
def evaluate(model, index, normalization, scale, device):
    rows = []
    with h5py.File(index, 'r') as f:
        for i, sample in enumerate(f['sample_id'].asstr()[:]):
            v=torch.from_numpy(f['velocity401'][i:i+1]).to(device)
            s=torch.from_numpy(f['source_solver'][i:i+1]).to(device)
            y=torch.from_numpy(f['wavefield'][i]).to(device,dtype=torch.float64)
            predicted=model.base.decode_full(model(v,s,normalization),pressure_scale=scale)[0]
            error=predicted.double()-y
            values=[float(error[lo:hi].norm()/y[lo:hi].norm()) for lo,hi in ((0,len(y)),*WINDOWS)]
            rows.append(dict(sample_id=sample, medium_id=f['medium_id'].asstr()[i], relative_l2=values))
    values=np.asarray([row['relative_l2'] for row in rows])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite validation metrics')
    return dict(mean=values.mean(axis=0).tolist(), records=rows)


def train_step(model, objective, optimizer, arrays, ids, normalization, device):
    tensors=(torch.from_numpy(array[ids]).to(device) for array in arrays)
    velocity, source, adjoint, statistics=tensors
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device=='cuda'):
        coefficients=model(velocity,source,normalization)
    loss,rss=objective(coefficients,adjoint,statistics)
    if (rss < 0).any():
        loss,rss=objective(coefficients,adjoint,statistics,precision=torch.float64)
        if (rss < -1e-6*statistics[...,0]).any():
            raise ValueError('Negative pressure residual')
    loss=loss.mean()
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.base.entity_heads.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True,type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--cache',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--extra-cache',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--interval',type=int,required=True)
    parser.add_argument('--batch',type=int,default=6)
    parser.add_argument('--lr',type=float,default=2e-4)
    args=parser.parse_args()
    if not 3 <= args.batch <= 12 or args.steps <= 0 or args.interval <= 0 or args.lr <= 0:
        parser.error('Invalid batch, step count, interval or learning rate')
    if args.device=='cuda':
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if pids:
            raise RuntimeError(f'GPU already has compute processes: {pids}')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    original=torch.load(args.parent,map_location='cpu',weights_only=True)
    manifest=load_bound(args.manifest,original['digests']['manifest'])
    rows=[row for row in manifest['records'] if row['split']=='train']
    contract=manifest['split_indexes']['development']
    dev=Path(contract['file'])
    if sha256(dev) != contract['sha256']:
        raise ValueError('Development index hash mismatch')
    space=SpaceBasis(length=original['extent_m'])
    temporal=TimeBasis(bubbles=original['bubbles'])
    model=EquivariantFineTune(ReflectionTraceSpectral(space,temporal)).to(args.device)
    model.base.load_state_dict(original['model'])
    objective=full_train.PressureObjective(space.matrix,space.matrix[:,1:],temporal.matrix).to(args.device)
    digests=dict(manifest=original['digests']['manifest'], code=sha256(Path(full_train.__file__)),
                 base_code=code_digest(), basis=objective.basis_signature)
    if original['digests']!=digests:
        raise ValueError('Parent objective or model code hash mismatch')
    arrays, rows, addendum = join(args.cache, rows, digests, args.extra_cache)
    sampler=Sampling(rows)
    optimizer=torch.optim.AdamW(model.base.entity_heads.parameters(), lr=args.lr, weight_decay=1e-6, fused=args.device=='cuda')
    normalization,scale=input_kwargs(original),original['pressure_scale_pa']
    lineage=dict(parent_sha256=sha256(args.parent), manifest_sha256=digests['manifest'],
                 code_sha256=sha256(Path(__file__)),
                 reflection_sha256=sha256(Path(__file__).with_name('marmousi_symmetry.py')),
                 sampler_sha256=sha256(Path(__file__).with_name('train_marmousi.py')),
                 parent_step=original['step'], batch=args.batch, lr=args.lr, seed=SEED,
                 addendum_sha256=addendum, training_samples=len(rows),
                 cache_code_sha256=sha256(Path(__file__).with_name('extra_cache.py')))
    offset=0
    if args.resume:
        previous=torch.load(args.resume,map_location='cpu',weights_only=True)
        if previous['lineage'] != lineage:
            raise ValueError('Resume settings or model lineage mismatch')
        model.base.entity_heads.load_state_dict(previous['heads'])
        optimizer.load_state_dict(previous['optimizer'])
        offset=previous['step']
        reference=previous['reference']
        del previous
    del original
    args.out.mkdir(exist_ok=False)
    atomic_json(args.out/'RUN_CONFIG.json',dict(**lineage, device=args.device, steps=args.steps,
                                              offset=offset,parent=str(args.parent),
                                              resume=str(args.resume) if args.resume else None,
                                              training_objective='pressure_loss_on_equivariant_prediction',
                                              baseline_late_gain_required=MIN_LATE_GAIN))

    def status(stage, update, **fields):
        atomic_json(args.out/'STATUS.json',dict(stage=stage,pid=os.getpid(),step=update,target=offset+args.steps,
                                               time=datetime.now(timezone.utc).isoformat(),**fields))

    def checkpoint(step, metrics):
        return dict(lineage=lineage, step=step, heads=model.base.entity_heads.state_dict(),
                    optimizer=optimizer.state_dict(), metrics=metrics['mean'], reference=reference)

    status('VALIDATING',offset)
    baseline=evaluate(model,dev,normalization,scale,args.device)
    if len(baseline['records']) != contract['samples']:
        raise ValueError('Unexpected development sample count')
    if not args.resume:
        reference=baseline
    atomic_json(args.out/f'DEV_{offset:04d}.json',baseline)
    print(json.dumps(dict(event='development',step=offset,mean=baseline['mean'])),flush=True)
    baseline_full, baseline_late = reference['mean'][0], reference['mean'][-1]
    best = baseline['mean'][-1] if args.resume else baseline_late
    start=time.monotonic()
    for step in range(offset+1,offset+args.steps+1):
        ids=sampler.draw(step,args.batch,'mixed')
        loss,norm=train_step(model,objective,optimizer,arrays,ids,normalization,args.device)
        if step==offset+1 or step%20==0:
            status('TRAINING',step,loss=loss)
            print(json.dumps(dict(event='update',step=step,loss=loss,grad_norm=norm,
                                  elapsed=time.monotonic()-start)),flush=True)
        if (step-offset)%args.interval==0 or step==offset+args.steps:
            status('VALIDATING',step)
            metrics=evaluate(model,dev,normalization,scale,args.device)
            for old, new in zip(reference['records'],metrics['records'],strict=True):
                if old['sample_id']!=new['sample_id']:
                    raise ValueError('Development sample order changed')
            atomic_json(args.out/f'DEV_{step:04d}.json',metrics)
            candidate=checkpoint(step,metrics)
            save_state(args.out/'last.pt',candidate)
            full,_,_,late=metrics['mean']
            baseline_media={}
            trained_media={}
            for item in reference['records']:
                baseline_media.setdefault(item['medium_id'],[]).append(item['relative_l2'][-1])
            for item in metrics['records']:
                trained_media.setdefault(item['medium_id'],[]).append(item['relative_l2'][-1])
            media_improved=sum(bool(np.mean(trained_media[m]) < np.mean(values))
                               for m,values in baseline_media.items())
            early_ok = metrics['mean'][1] <= reference['mean'][1] + .005
            accepted = full <= baseline_full and late <= baseline_late - MIN_LATE_GAIN and media_improved >= 2 and early_ok
            if accepted and late<best:
                best=late
                save_state(args.out/'best.pt',candidate)
            print(json.dumps(dict(event='development',step=step,mean=metrics['mean'],
                                  media_improved=media_improved,accepted=accepted)),flush=True)
    status('COMPLETE',offset+args.steps,passed=(args.out/'best.pt').exists())


if __name__=='__main__':
    main()
