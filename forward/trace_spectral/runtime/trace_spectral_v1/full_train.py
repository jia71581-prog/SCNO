"""Train on all 500 single-source shots with exact pressure loss and a RAM cache."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time
import traceback

import h5py
import numpy as np
import torch
from torch import nn

from .basis import SpaceBasis, TimeBasis
from .input_pack import pack_input
from .marmousi_data import atomic_json, load_bound, sha256
from .mass_loss import MassLoss, spatial_action, time_action, window_weights
from .model import TraceSpectralWave
from .train import code_digest, input_kwargs, save_state


class PressureObjective(MassLoss):
    """Use the zero coefficient vector as reference and retain each adjoint."""

    def __init__(self, sx, sz, t):
        super().__init__(sx,sz,t)
        self.register_buffer('sx',torch.as_tensor(sx,dtype=torch.float64))
        self.register_buffer('sz',torch.as_tensor(sz,dtype=torch.float64))
        for j,(lo,hi) in enumerate(self.windows):
            support=getattr(self,f'time_support_{j}')
            self.register_buffer(f'adjoint_{j}',torch.as_tensor(t[lo:hi],dtype=torch.float64)[:,support].T.contiguous())

    @torch.no_grad()
    def statistics(self, native, scale):
        pressure=torch.from_numpy(native).to(self.sx.device,dtype=torch.float64).div_(scale)
        energy, adjoint=[], []
        for j,(lo,hi) in enumerate(self.windows):
            p=pressure[lo:hi]
            energy.append(p.square().sum())
            b=time_action(p,getattr(self,f'adjoint_{j}'))
            adjoint.append(spatial_action(b,self.sz.T,self.sx.T).float())
        weights=window_weights(torch.stack(energy)[None])[0]
        energy=torch.stack(energy)
        return torch.cat(adjoint,dim=0).cpu(), torch.stack((energy,weights),dim=-1).cpu()

    def forward(self, coefficients, packed_adjoint, statistics, precision=torch.float32):
        a=coefficients.to(precision)
        v=spatial_action(a,self.gz.to(precision),self.gx.to(precision))
        energy,weight=statistics.to(precision).unbind(-1)
        start=0
        terms=[]
        for j in range(3):
            support=getattr(self,f'time_support_{j}')
            local=a.index_select(-3,support)
            temporal=time_action(v.index_select(-3,support),getattr(self,f'time_gram_{j}').to(precision))
            b=packed_adjoint[:,start:start+len(support)].to(precision)
            terms.append((local*(temporal-2*b)).flatten(1).sum(-1)+energy[:,j])
            start+=len(support)
        rss=torch.stack(terms,dim=-1)
        return (weight*rss).sum(-1),rss


def new_model():
    space=SpaceBasis(length=1000.)
    temporal=TimeBasis(bubbles=8)
    objective=PressureObjective(space.matrix,space.matrix[:,1:],temporal.matrix).cuda()
    return TraceSpectralWave(space_x=space,space_z=space,time_basis=temporal).cuda(),objective


class RamShots:
    """A TRAIN-only data reader. Cached adjoints are rebuilt after a restart."""

    def __init__(self, manifest, objective, pressure_scale):
        self.rows=[row for row in manifest['records'] if row['split']=='train']
        if len(self.rows)!=500 or len({row['sample_id'] for row in self.rows})!=500:
            raise ValueError('Expected the registered 500 TRAIN sample IDs')
        self.objective=objective
        self.pressure_scale=pressure_scale
        self.entries={}

    def read(self, indexes):
        pending={}
        for i in indexes:
            if i in self.entries:
                continue
            row=self.rows[i]
            with h5py.File(row['file'],'r') as file:
                if (file.attrs['split']!='train' or file.attrs['sample_id']!=row['sample_id']
                        or file.attrs['physical_sources']!=1 or file.attrs['units']!='Pa'):
                    raise ValueError('Unexpected TRAIN shard provenance')
                wave=file['wavefield'][:]
                velocity=file['velocity401'][:]
                source=file['source_solver'][:]
            if (wave.shape!=(801,201,201) or wave.dtype!=np.float32 or velocity.shape!=(401,401)
                    or source.shape!=(5,) or not np.array_equal(source,row['source_solver'])):
                raise ValueError(f'Invalid TRAIN wavefield {row["sample_id"]}')
            if 'pressure_sha256' in row:
                valid=(hashlib.sha256(memoryview(wave)).hexdigest()==row['pressure_sha256']
                       and hashlib.sha256(memoryview(velocity)).hexdigest()==row['velocity401_sha256'])
            else:
                valid=sha256(Path(row['file']))==row['sha256']
            if not valid:
                raise ValueError(f'Invalid TRAIN content hash {row["sample_id"]}')
            pending[i]=(wave,velocity,source)
        return pending

    def commit(self, pending):
        for i,(wave,velocity,source) in pending.items():
            adjoint,statistics=self.objective.statistics(wave,self.pressure_scale)
            self.entries[i]=(torch.from_numpy(velocity),torch.from_numpy(source),adjoint,statistics)

    def batch(self, indexes):
        return tuple(torch.stack(items).pin_memory().cuda(non_blocking=True)
                     for items in zip(*(self.entries[i] for i in indexes)))


def step(model,objective,optim,data,indexes,micro,normalization):
    optim.zero_grad(set_to_none=True)
    total=torch.zeros((),device='cuda')
    for start in range(0,len(indexes),micro):
        ids=indexes[start:start+micro]
        velocity,source,adjoint,stats=data.batch(ids)
        packed=pack_input(velocity,source,**normalization)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            coefficients=model.predict_coefficients(packed)
        losses,rss=objective(coefficients,adjoint,stats)
        if bool((rss<0).any()):
            losses,rss=objective(coefficients,adjoint,stats,precision=torch.float64)
            torch._assert_async((rss>=-1e-6*stats[...,0]).all(),'Invalid negative window loss')
        loss=losses.sum()/len(indexes)
        loss.backward()
        total+=loss.detach()
    norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optim.step()
    return float(total),float(norm)


@torch.inference_mode()
def evaluate(model,rows,normalization,scale):
    model.eval()
    result=[]
    for row in rows:
        with h5py.File(row['file'],'r') as file:
            if file.attrs['split']!='development':
                raise ValueError('Evaluation requires DEVELOPMENT')
            velocity=torch.from_numpy(file['velocity401'][:])[None].cuda()
            source=file['source_solver'][:][None]
            truth=torch.from_numpy(file['wavefield'][:]).cuda()
        prediction=model.predict_pressure(pack_input(velocity,source,**normalization),pressure_scale=scale)[0]
        error=prediction-truth
        result.append([float(error.norm()/truth.norm()),float(error[480:].norm()/truth[480:].norm())])
    model.train()
    return dict(mean_full_late=np.mean(result,axis=0).tolist(),p90_full_late=np.percentile(result,90,axis=0).tolist())


def choose_batch(model,objective,data,optim,normalization,candidates):
    prefix=list(range(max(candidates)))
    data.commit(data.read(prefix))
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    measurements=[]
    for n in candidates:
        samples=[]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for repeat in range(3):
                started=time.perf_counter()
                step(model,objective,optim,data,prefix[:n],n,normalization)
                if repeat:
                    samples.append(time.perf_counter()-started)
            peak=torch.cuda.max_memory_reserved()
            measurement=dict(micro=n,seconds=median(samples),samples_per_s=n/median(samples),reserved_bytes=peak)
            measurements.append(measurement)
            print(json.dumps(dict(event='batch_probe',**measurement)),flush=True)
        except torch.OutOfMemoryError:
            print(json.dumps(dict(event='batch_probe_oom',micro=n)),flush=True)
            optim.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
        finally:
            model.load_state_dict(initial)
            optim.state.clear()
            optim.zero_grad(set_to_none=True)
    viable=[r for r in measurements if r['reserved_bytes']<=.92*torch.cuda.get_device_properties(0).total_memory]
    if not viable:
        raise RuntimeError('No batch candidate has CUDA headroom')
    best=max(viable,key=lambda row:row['samples_per_s'])
    return best,measurements


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--manifest',type=Path,required=True)
    cli.add_argument('--manifest-sha',required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--steps',type=int,default=12600)
    cli.add_argument('--candidates',type=int,nargs='+',default=[8,16,24,32,40,48])
    cli.add_argument('--resume',action='store_true')
    args=cli.parse_args()
    if args.steps<1 or not args.candidates or sorted(set(args.candidates))!=args.candidates or not 1<=args.candidates[0]<=args.candidates[-1]<=64:
        cli.error('Use positive steps and ascending distinct batch candidates in [1,64]')
    import fcntl
    args.out.mkdir(exist_ok=True)
    with (args.out/'TRAIN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(args)
        except Exception:
            path=args.out/'STATUS.json'
            previous=json.loads(path.read_text()) if path.exists() else {}
            atomic_json(path,{**previous,'stage':'FAILED','traceback':traceback.format_exc()})
            raise


def run(args):
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise RuntimeError('CUDA device occupied before startup')
    manifest=load_bound(args.manifest,args.manifest_sha)
    if manifest['status']!='COMPLETE':
        raise ValueError('Expected a complete data manifest')
    torch.set_num_threads(4)
    torch.manual_seed(2026111205)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.95)
    rng=np.random.default_rng(2026111301)
    model,objective=new_model()
    scale=1.252277344221455e-8
    meta=dict(extent_m=1000,bubbles=8,pressure_scale_pa=scale,
              normalization=dict(velocity_center=3728.63427734375,velocity_scale=2141.111083984375))
    normalization=input_kwargs(meta)
    data=RamShots(manifest,objective,scale)
    dev=[row for row in manifest['records'] if row['split']=='development']
    optim=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-6,fused=True)
    digests=dict(manifest=args.manifest_sha,code=sha256(Path(__file__)),base_code=code_digest(),basis=objective.basis_signature)
    checkpoint=args.out/'last.pt'
    status=args.out/'STATUS.json'
    settings=args.out/'RUN_CONFIG.json'
    update,cursor,epoch,best=0,0,0,None
    if args.resume:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['digests']!=digests:
            raise ValueError('Checkpoint is from a different manifest, objective or implementation')
        model.load_state_dict(state['model'])
        optim.load_state_dict(state['optimizer'])
        update,cursor,epoch,best,batch,micro,order=(state[k] for k in ('step','cursor','epoch','best','batch','micro','order'))
        rng.bit_generator.state=state['rng']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
    else:
        if settings.exists() or checkpoint.exists():
            raise ValueError('Use a new output directory or --resume')
        atomic_json(status,dict(stage='BATCH_PROBE',pid=os.getpid(),target_steps=args.steps))
        selected,measurements=choose_batch(model,objective,data,optim,normalization,args.candidates)
        batch=micro=selected['micro']
        order=rng.permutation(500).tolist()
        atomic_json(settings,dict(**meta,digests=digests,batch=batch,micro=micro,target_steps=args.steps,
                                   dataset_size=500,dev_size=len(dev),batch_probe=measurements,
                                   start='fresh',cache='TRAIN-only adjoints in process RAM',optimizer='AdamW',
                                   learning_rate=1e-4,warmup_steps=250,seed=2026111205,sampler_seed=2026111301))
    def report(stage,**extra):
        atomic_json(status,dict(stage=stage,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),
                               step=update,target_steps=args.steps,epoch=epoch,cursor=cursor,
                               batch=batch,micro=micro,ram_cache=len(data.entries),**extra))
    def persist():
        save_state(checkpoint,dict(**meta,digests=digests,model=model.state_dict(),optimizer=optim.state_dict(),
                    step=update,cursor=cursor,epoch=epoch,best=best,batch=batch,micro=micro,order=order,
                    rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
    report('TRAINING')
    with ThreadPoolExecutor(max_workers=1) as reader,(args.out/'metrics.jsonl').open('a',buffering=1) as log:
        upcoming=reader.submit(data.read,order[cursor:cursor+batch])
        while update<args.steps:
            started=time.perf_counter()
            indices=order[cursor:cursor+batch]
            pending=upcoming.result()
            if cursor+len(indices)==500:
                next_order,next_cursor=rng.permutation(500).tolist(),0
            else:
                next_order,next_cursor=order,cursor+len(indices)
            upcoming=reader.submit(data.read,next_order[next_cursor:next_cursor+batch])
            data.commit(pending)
            del pending
            optim.param_groups[0]['lr']=1e-4*min((update+1)/250,1.)
            value,norm=step(model,objective,optim,data,indices,micro,normalization)
            update+=1
            epoch+=int(next_cursor==0)
            order,cursor=next_order,next_cursor
            item=dict(event='update',step=update,loss=value,grad_norm=norm,actual_batch=len(indices),
                      seconds=time.perf_counter()-started,ram_cache=len(data.entries),
                      cuda_allocated_peak=torch.cuda.max_memory_allocated())
            log.write(json.dumps(item)+'\n')
            if update==1 or update%50==0 or update==args.steps:
                persist()
                report('TRAINING',loss=value,seconds=item['seconds'])
                print(json.dumps(item),flush=True)
            if update%250==0 or update==args.steps:
                metrics=evaluate(model,dev,normalization,scale)
                log.write(json.dumps(dict(event='development',step=update,**metrics))+'\n')
                atomic_json(args.out/'DEVELOPMENT.json',dict(step=update,**metrics))
                if best is None or sum(metrics['mean_full_late'])<best:
                    best=sum(metrics['mean_full_late'])
                    persist()
                    import shutil
                    temporary=args.out/'best.tmp'
                    shutil.copyfile(checkpoint,temporary)
                    temporary.replace(args.out/'best.pt')
    report('COMPLETE')


if __name__=='__main__':
    main()
