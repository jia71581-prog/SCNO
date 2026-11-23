"""Checkpointed FP64 GPU normalizers with bounded, parallel HDF5 readers."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .core import atomic_json,digest,source_input


def statistics_index(folder,contract):
    rows=[]
    for path in sorted(Path(folder).glob('*.h5')):
        raw=path.with_suffix('.json').read_bytes();receipt=json.loads(raw)
        if receipt['contract']!=contract:raise ValueError('statistics receipt contract mismatch')
        stat=path.stat()
        rows.append(dict(path=path.name,split=receipt['split'],
                         identity=[stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns],
                         receipt_sha256=hashlib.sha256(raw).hexdigest(),receipt=receipt))
    return rows


def read_statistics_record(path,contract):
    with h5py.File(path,'r') as f:
        if f.attrs['contract']!=contract or f.attrs['split']!='train':
            raise ValueError('GPU statistics may only read compatible training records')
        return f['velocity_km_s'][:],f['source_zx'][:],f['displacement_m'][:]


def reduce_statistics_batch(rows,cfg):
    """Return one row per medium: input mean/std and output mean/std."""
    velocities=np.stack([x[0] for x in rows]);locations=np.stack([x[1] for x in rows])
    fields=np.stack([x[2] for x in rows]);batch=len(rows)
    # Retain the original CPU FP32 encoder exactly; only reductions move to GPU.
    source=source_input(locations,cfg)
    v=torch.as_tensor(velocities,device='cuda',dtype=torch.float64).reshape(batch,2,-1)
    u=torch.as_tensor(fields,device='cuda',dtype=torch.float64).reshape(batch,-1,2)
    s=source.to(device='cuda',dtype=torch.float64).reshape(batch,-1)
    vm,vs=v.mean(2),v.std(2,unbiased=True)
    sm,ss=s.mean(1,keepdim=True),s.std(1,unbiased=True,keepdim=True)
    result=torch.cat([vm,sm,vs,ss,u.mean(1),u.std(1,unbiased=True)],dim=1).cpu().numpy()
    if not np.isfinite(result).all():raise FloatingPointError('non-finite GPU normalization moments')
    return result


def moments_to_statistics(moments,contract):
    return dict(input_mean=moments[:,:3].mean(0).tolist(),
                input_std=(moments[:,3:6].mean(0)+1e-5).tolist(),
                output_mean=moments[:,6:8].mean(0).tolist(),
                output_std=(moments[:,8:10].mean(0)+1e-12).tolist(),
                count=len(moments),fitted_on='train_only',
                method='average_per_medium_mean_and_sample_std',contract=contract)


def fit_statistics_gpu(folder,cfg,contract,workdir,batch_size=4,readers=4,
                       stop_after_records=None,records=None):
    if batch_size<1 or readers<0:raise ValueError('invalid GPU statistics runtime')
    torch.set_num_threads(1)
    folder=Path(folder);work=Path(workdir);work.mkdir(parents=True,exist_ok=True)
    records=statistics_index(folder,contract) if records is None else records
    training=[row for row in records if row['split']=='train']
    if not training:raise ValueError('no training records')
    fingerprint=digest(dict(contract=contract,cfg=cfg,records=records))
    checkpoint=work/'moments.npz';moments=np.zeros((len(training),10),dtype='float64');completed=0
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as saved:
            if str(saved['fingerprint'])!=fingerprint:raise ValueError('statistics input changed since checkpoint')
            completed=int(saved['completed']);moments[:]=saved['moments']
        if not 0<=completed<=len(training):raise ValueError('invalid statistics checkpoint position')
    reused=completed;started=time.monotonic();last_saved=completed;last_status=started
    def status(stage):
        elapsed=time.monotonic()-started
        obj=dict(stage=stage,device='cuda',precision='float64',completed=completed,total=len(training),
                 reused=reused,newly_processed=completed-reused,batch_size=batch_size,readers=readers,
                 elapsed_s=elapsed,records_per_s=(completed-reused)/elapsed if elapsed else 0,
                 updated_at=time.time())
        atomic_json(work/'STATUS.json',obj);print(json.dumps(dict(stage='normalization',**{k:v for k,v in obj.items() if k!='stage'})),flush=True)
    def save():
        tmp=checkpoint.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:np.savez(f,moments=moments,completed=completed,fingerprint=fingerprint)
        os.replace(tmp,checkpoint)
    pool=None;pending=deque();next_submit=completed
    try:
        if readers:pool=ProcessPoolExecutor(max_workers=readers,mp_context=mp.get_context('spawn'))
        def fill():
            nonlocal next_submit
            while pool and len(pending)<2*batch_size and next_submit<len(training):
                path=folder/training[next_submit]['path']
                pending.append(pool.submit(read_statistics_record,str(path),contract));next_submit+=1
        fill();status('RUNNING')
        while completed<len(training):
            count=min(batch_size,len(training)-completed)
            if pool:
                rows=[pending.popleft().result() for _ in range(count)];fill()
            else:
                rows=[read_statistics_record(str(folder/training[i]['path']),contract)
                      for i in range(completed,completed+count)]
            moments[completed:completed+count]=reduce_statistics_batch(rows,cfg);completed+=count
            del rows
            stopping=stop_after_records is not None and completed-reused>=stop_after_records
            if completed-last_saved>=128 or completed==len(training) or stopping:
                save();last_saved=completed
            if time.monotonic()-last_status>=5 or stopping:
                status('PARTIAL' if stopping else 'RUNNING');last_status=time.monotonic()
            if stopping:return None
        result=moments_to_statistics(moments,contract)
        atomic_json(work/'normalization.json',result);status('COMPLETE')
        return result
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)
        torch.cuda.empty_cache()
