"""Continue an existing full training run with double-buffered input."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch

from .full_train import RamShots, evaluate, new_model, step
from .marmousi_data import atomic_json, load_bound, sha256
from .train import code_digest, input_kwargs, save_state


class Slot:
    def __init__(self, arrays, capacity):
        self.host = [torch.empty((capacity, *a.shape[1:]),
                                 dtype=torch.from_numpy(np.empty((), dtype=a.dtype)).dtype,
                                 pin_memory=True) for a in arrays]
        self.views = [t.numpy() for t in self.host]
        self.device = [torch.empty_like(t, device='cuda', pin_memory=False) for t in self.host]
        self.uploaded = torch.cuda.Event()
        self.consumed = torch.cuda.Event()


class Batch:
    def __init__(self, tensors):
        self.tensors = tensors

    def batch(self, indexes):
        return self.tensors


class Prefetch:
    def __init__(self, arrays, capacity):
        self.arrays = arrays
        self.slots = [Slot(arrays, capacity), Slot(arrays, capacity)]
        self.stream = torch.cuda.Stream()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def close(self):
        self.executor.shutdown()
        self.stream.synchronize()

    def fill(self, slot, ids):
        if slot.uploaded.query() is False:
            slot.uploaded.synchronize()
        count = len(ids)
        for values, view in zip(self.arrays, slot.views, strict=True):
            # All IDs are validated at the dataset boundary. "clip" avoids
            # the extra temporary that NumPy uses for bounds-error recovery.
            np.take(values, ids, axis=0, out=view[:count], mode='clip')
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(slot.consumed)
            for source, target in zip(slot.host, slot.device, strict=True):
                target[:count].copy_(source[:count], non_blocking=True)
            slot.uploaded.record(self.stream)
        return Batch(tuple(t[:count] for t in slot.device))

    def schedule(self, index, ids):
        return self.executor.submit(self.fill, self.slots[index], ids)

    def take(self, index, future):
        batch = future.result()
        torch.cuda.current_stream().wait_event(self.slots[index].uploaded)
        return batch

    def release(self, index):
        self.slots[index].consumed.record(torch.cuda.current_stream())


def load_arrays(directory, manifest_hash, basis, rows, original_code):
    receipt = json.loads((directory / 'COMPLETE.json').read_text())
    ids = [row['sample_id'] for row in rows]
    if (receipt['status'] != 'COMPLETE' or receipt['manifest'] != manifest_hash or receipt['basis'] != basis
            or receipt['records'] != 500 or receipt['ids'] != ids or receipt['teacher_objective'] != original_code):
        raise ValueError('RAM cache differs from the saved data or teacher objective')
    arrays = []
    for key in ('velocity','source','adjoint','stats'):
        item = receipt['files'][key]
        path = directory / item['file']
        if path.stat().st_size != item['bytes']:
            raise ValueError('Incomplete RAM cache')
        arrays.append(np.load(path, mmap_mode='r', allow_pickle=False))
    return arrays


def verify_cache(arrays, manifest, objective, scale, order):
    reference = RamShots(manifest,objective,scale)
    for i,(native,velocity,source) in reference.read(order[:2]).items():
        adjoint,stats=objective.statistics(native,scale)
        b=torch.from_numpy(arrays[2][i].copy())
        weight=torch.from_numpy(arrays[3][i].copy())
        torch.testing.assert_close(weight,stats,rtol=1e-10,atol=0.)
        relative=float((b-adjoint).norm()/adjoint.norm())
        if relative>2e-6 or not np.array_equal(arrays[0][i],velocity) or not np.array_equal(arrays[1][i],source):
            raise ValueError(f'CPU cache differs from the prior CUDA statistics for row {i}: {relative}')
        print(json.dumps(dict(event='cache_check',row=i,adjoint_relative_l2=relative)),flush=True)


def verify_transfer(pipe, arrays, order):
    for slot,ids in ((0,order[:32]),(1,order[32:52]),(0,order[52:84])):
        batch=pipe.take(slot,pipe.schedule(slot,ids))
        for staged,array in zip(batch.tensors,arrays,strict=True):
            if not np.array_equal(staged.cpu().numpy(),array[ids]):
                raise ValueError('Prefetch does not preserve values or sample order')
        pipe.release(slot)


def run(args):
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise RuntimeError('CUDA device occupied before startup')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.95)
    parent=args.resume or args.parent
    state=torch.load(parent,map_location='cpu',weights_only=True)
    manifest=load_bound(args.manifest,state['digests']['manifest'])
    model,objective=new_model()
    original=sha256(Path(__file__).with_name('full_train.py'))
    expected=dict(manifest=state['digests']['manifest'],code=original,base_code=code_digest(),basis=objective.basis_signature)
    version=sha256(Path(__file__))
    if state['digests']!=expected or (args.resume and state.get('prefetch_version')!=version):
        raise ValueError('The checkpoint is incompatible with this objective or implementation')
    model.load_state_dict(state['model'])
    optim=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-6,fused=True)
    optim.load_state_dict(state['optimizer'])
    update,cursor,epoch,best,batch,micro,order=(state[k] for k in ('step','cursor','epoch','best','batch','micro','order'))
    if micro!=batch or len(order)!=500 or sorted(order)!=list(range(500)) or not 0<=cursor<500 or not update<args.steps:
        raise ValueError('Invalid parent batch, sampler cursor or budget')
    origin=update
    rng=np.random.default_rng()
    rng.bit_generator.state=state['rng']
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state(state['cuda_rng'])
    scale=state['pressure_scale_pa']
    meta={k:state[k] for k in ('extent_m','bubbles','pressure_scale_pa','normalization')}
    normalization=input_kwargs(meta)
    train=[row for row in manifest['records'] if row['split']=='train']
    dev=[row for row in manifest['records'] if row['split']=='development']
    arrays=load_arrays(args.cache,expected['manifest'],expected['basis'],train,original)
    status=args.out/'STATUS.json'

    def report(stage,**extra):
        atomic_json(status,dict(stage=stage,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),
                                step=update,target_steps=args.steps,epoch=epoch,cursor=cursor,batch=batch,micro=micro,
                                ram_cache=len(train),**extra))

    checkpoint=args.out/'last.pt'

    def persist():
        save_state(checkpoint,dict(**meta,digests=expected,model=model.state_dict(),optimizer=optim.state_dict(),
                                  step=update,cursor=cursor,epoch=epoch,best=best,batch=batch,micro=micro,order=order,
                                  rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                                  prefetch_version=version))

    report('VERIFYING')
    verify_cache(arrays,manifest,objective,scale,order)
    pipe=Prefetch(arrays,batch)
    try:
        verify_transfer(pipe,arrays,order)
        if not args.resume:
            atomic_json(args.out/'RUN_CONFIG.json',dict(**meta,digests=expected,prefetch_version=version,
                        parent=str(parent),parent_sha256=sha256(parent),parent_step=origin,
                        target_steps=args.steps,batch=batch,micro=micro,dataset_size=len(train),
                        dev_size=len(dev),cache=str(args.cache),optimizer='AdamW',learning_rate=1e-4,warmup_steps=250))
        report('TRAINING')
        torch.cuda.reset_peak_memory_stats()
        slot=0
        incoming=pipe.schedule(slot,order[cursor:cursor+batch])
        with (args.out/'metrics.jsonl').open('a',buffering=1) as log:
            while update<args.steps:
                started=time.perf_counter()
                ids=order[cursor:cursor+batch]
                packed=pipe.take(slot,incoming)
                acquired=time.perf_counter()
                if cursor+len(ids)==500:
                    next_order,next_cursor=rng.permutation(500).tolist(),0
                else:
                    next_order,next_cursor=order,cursor+len(ids)
                next_slot=1-slot
                incoming=pipe.schedule(next_slot,next_order[next_cursor:next_cursor+batch])
                optim.param_groups[0]['lr']=1e-4*min((update+1)/250,1.)
                value,norm=step(model,objective,optim,packed,ids,micro,normalization)
                pipe.release(slot)
                slot=next_slot
                update+=1
                epoch+=int(next_cursor==0)
                order,cursor=next_order,next_cursor
                item=dict(event='update',step=update,loss=value,grad_norm=norm,actual_batch=len(ids),
                          seconds=time.perf_counter()-started,host_wait_seconds=acquired-started,
                          ram_cache=len(train),cuda_allocated_peak=torch.cuda.max_memory_allocated())
                log.write(json.dumps(item)+'\n')
                if update==origin+1 or update%50==0 or update==args.steps:
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
                        temporary=args.out/'best.tmp'
                        shutil.copyfile(checkpoint,temporary)
                        temporary.replace(args.out/'best.pt')
        report('COMPLETE')
    finally:
        pipe.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=12600)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--parent',type=Path)
    source.add_argument('--resume',type=Path)
    args=parser.parse_args()
    import fcntl
    args.out.mkdir(exist_ok=True)
    with (args.out/'TRAIN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.parent and (args.out/'RUN_CONFIG.json').exists():
            raise ValueError('Use --resume for an existing prefetch run')
        try:
            run(args)
        except Exception:
            path=args.out/'STATUS.json'
            previous=json.loads(path.read_text()) if path.exists() else {}
            atomic_json(path,{**previous,'stage':'FAILED','traceback':traceback.format_exc()})
            raise


if __name__=='__main__':
    main()
