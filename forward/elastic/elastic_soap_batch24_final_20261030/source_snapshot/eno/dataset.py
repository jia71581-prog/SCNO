"""Atomic per-medium elastic records, independent split and train-only scaling."""
import json,os,time,shutil
from pathlib import Path
import h5py
import numpy as np
import torch
from .core import atomic_json,sha,digest,random_medium,source_input,quarantine_orphan


def write_record(path,medium,location,wavefield,split,contract,runtime=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not np.isfinite(wavefield).all():raise ValueError('non-finite labels')
    tmp=path.with_suffix('.h5.tmp')
    with h5py.File(tmp,'w') as f:
        f.create_dataset('velocity_km_s',data=medium)
        f.create_dataset('source_zx',data=np.asarray(location,dtype='int64'))
        f.create_dataset('displacement_m',data=wavefield,chunks=True,compression='lzf',shuffle=True)
        f.attrs.update(split=split,contract=contract,components='ux,uz',axes='z,x,t,component',backend='devito')
    os.replace(tmp,path)
    atomic_json(path.with_suffix('.json'),dict(sha256=sha(path),split=split,contract=contract,
                runtime=runtime or {},
                velocity_sha256=__import__('hashlib').sha256(np.ascontiguousarray(medium).tobytes()).hexdigest()))


def read_record(path,contract):
    with h5py.File(path,'r') as f:
        if f.attrs['contract']!=contract:raise ValueError('record configuration mismatch')
        return f['velocity_km_s'][:],f['source_zx'][:],f['displacement_m'][:],str(f.attrs['split'])


def fit_statistics(folder,cfg,contract):
    am=[];ast=[];ym=[];yst=[]
    for path in sorted(Path(folder).glob('*.h5')):
        with h5py.File(path,'r') as f:
            if f.attrs['contract']!=contract:raise ValueError('statistics contract mismatch')
            if f.attrs['split']!='train':continue
            v=f['velocity_km_s'][:].astype('float64');loc=f['source_zx'][:];u=f['displacement_m'][:].astype('float64')
        s=source_input(loc[None],cfg).numpy().astype('float64')
        am.append([v[0].mean(),v[1].mean(),s.mean()])
        ast.append([v[0].std(ddof=1),v[1].std(ddof=1),s.std(ddof=1)])
        ym.append(u.mean((0,1,2)));yst.append(u.reshape(-1,2).std(0,ddof=1))
    if not am:raise ValueError('no training records')
    return dict(input_mean=np.mean(am,axis=0).tolist(),input_std=(np.mean(ast,axis=0)+1e-5).tolist(),
                output_mean=np.mean(ym,axis=0).tolist(),output_std=(np.mean(yst,axis=0)+1e-12).tolist(),
                count=len(am),fitted_on='train_only',method='average_per_medium_mean_and_sample_std',contract=contract)


def data_contract(cfg):return digest({k:cfg[k] for k in ['grid','source','fd','data']})


def generation_runtime(cfg,overrides=None):
    runtime=dict(workers=cfg['data'].get('workers',1),threads=cfg['fd'].get('threads',4),dispatch_batch=4,direct_kernel=False,
                 device='cpu',batch_size=1)
    overrides=overrides or {}
    if set(overrides)-set(runtime):raise ValueError('unknown generation runtime option')
    runtime.update(overrides)
    if any(type(runtime[k]) is not int or runtime[k]<1 for k in ['workers','threads','dispatch_batch','batch_size']):raise ValueError('generation runtime values must be positive integers')
    if type(runtime['direct_kernel']) is not bool:raise ValueError('direct_kernel must be boolean')
    if runtime['device'] not in ['cpu','cuda']:raise ValueError('generation device must be cpu or cuda')
    if runtime['device']=='cuda' and (runtime['direct_kernel'] or runtime['workers']!=1):
        raise ValueError('batched GPU generation uses one controller and direct_kernel=false')
    return runtime


def _generate_one(task):
    folder,i,cfg=task[:3]
    runtime=generation_runtime(cfg,task[3] if len(task)>3 else None)
    from .physics import simulate,diagnostics
    torch.set_num_threads(1)
    contract=data_contract(cfg);path=Path(folder)/f'{i:06d}.h5'
    quarantine_orphan(path,path.with_suffix('.json'))
    if path.exists():
        receipt=json.loads(path.with_suffix('.json').read_text())
        if receipt['contract']!=contract or receipt['sha256']!=sha(path):raise ValueError(f'corrupt/incompatible existing record {path}')
        return i,0.
    if shutil.disk_usage(folder).free<cfg['data'].get('reserve_gb',24)*1024**3:raise RuntimeError('disk reserve reached; resumable stop')
    t=time.time();v=random_medium(cfg['data']['seed']+i,cfg['grid']['n'])
    rng=np.random.default_rng(cfg['data']['seed']+i+1000000)
    loc=rng.integers(2,cfg['grid']['n']-2,size=2)
    u=simulate(v,loc[None],cfg,runtime['device'],threads=runtime['threads'],direct_kernel=runtime['direct_kernel'])[0]
    if np.linalg.norm(u)==0:raise ValueError('zero generated observation')
    split='train' if i<cfg['data']['train_count'] else 'validation'
    write_record(path,v,loc,u,split,contract,runtime)
    return i,time.time()-t


def _generate_gpu_batch(folder,indices,cfg,runtime):
    from .physics import simulate_batch,diagnostics
    runtime=generation_runtime(cfg,runtime)
    contract=data_contract(cfg);pending=[];velocities=[];locations=[]
    folder=Path(folder)
    for i in indices:
        path=folder/f'{i:06d}.h5';receipt=path.with_suffix('.json')
        quarantine_orphan(path,receipt)
        if path.exists():
            r=json.loads(receipt.read_text())
            if r['contract']!=contract or r['sha256']!=sha(path):raise ValueError(f'corrupt/incompatible existing record {path}')
            continue
        pending.append(i);velocities.append(random_medium(cfg['data']['seed']+i,cfg['grid']['n']))
        rng=np.random.default_rng(cfg['data']['seed']+i+1000000)
        locations.append(rng.integers(2,cfg['grid']['n']-2,size=2))
    if not pending:return [(i,0.) for i in indices]
    reserve=cfg['data'].get('reserve_gb',24)*1024**3
    uncompressed=len(pending)*cfg['grid']['n']**2*cfg['grid']['frames']*2*4
    if shutil.disk_usage(folder).free<reserve+uncompressed:raise RuntimeError('disk reserve reached; resumable stop')
    begin=time.time()
    wavefields=simulate_batch(np.stack(velocities),np.stack(locations),cfg)
    audit=diagnostics(cfg,'cuda')
    if audit['top_normal_stress_max']!=0 or audit['top_shear_antisymmetry_error']!=0:
        raise ValueError('GPU free-surface audit failed')
    for i,v,loc,u in zip(pending,velocities,locations,wavefields):
        if np.linalg.norm(u)==0:raise ValueError('zero generated observation')
        split='train' if i<cfg['data']['train_count'] else 'validation'
        recorded_runtime=dict(runtime,execution=audit['execution'],actual_batch_size=len(pending),
                              devito_source_sha256=audit['devito_source_sha256'],cuda_source_sha256=audit['cuda_source_sha256'])
        write_record(folder/f'{i:06d}.h5',v,loc,u,split,contract,recorded_runtime)
    per_record=(time.time()-begin)/len(pending)
    generated=set(pending)
    return [(i,per_record if i in generated else 0.) for i in indices]


def _gpu_iterator(folder,count,cfg,runtime):
    batch=runtime['batch_size']
    for start in range(0,count,batch):
        yield from _generate_gpu_batch(folder,range(start,min(start+batch,count)),cfg,runtime)


def generate(cfg,run):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    folder=Path(run)/'data';folder.mkdir(parents=True,exist_ok=True)
    contract=data_contract(cfg);count=cfg['data']['count'];t=time.time()
    old=folder/'COMPLETE.json'
    if old.exists():
        marker=json.loads(old.read_text())
        if marker['contract']!=contract:raise ValueError('completed dataset config mismatch')
    runtime_path=Path(run)/'DATA_RUNTIME.json'
    runtime=generation_runtime(cfg,json.loads(runtime_path.read_text()) if runtime_path.exists() else None)
    atomic_json(Path(run)/'ACTIVE_DATA_RUNTIME.json',runtime)
    tasks=((str(folder),i,cfg,runtime) for i in range(count))
    workers=runtime['workers']
    if runtime['device']=='cuda':
        pool=None;iterator=_gpu_iterator(folder,count,cfg,runtime)
    elif workers>1:
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'))
        iterator=pool.map(_generate_one,tasks,chunksize=runtime['dispatch_batch'])
    else:pool=None;iterator=map(_generate_one,tasks)
    reused=0;new=0;new_start=None
    try:
        for done,(i,seconds) in enumerate(iterator,1):
            if seconds==0:reused+=1
            else:
                new+=1
                if new_start is None:new_start=time.time()
            status=dict(stage='GENERATING',completed=done,total=count,last_id=i,seconds=time.time()-t,
                        last_record_seconds=seconds,reused=reused,newly_generated=new,
                        new_generation_elapsed_s=time.time()-new_start if new_start else 0.,runtime=runtime)
            atomic_json(folder/'STATUS.json',status)
            if done%10==0 or done<=2:print(json.dumps(dict(stage='data',**{k:v for k,v in status.items() if k!='stage'})),flush=True)
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)
        if runtime['device']=='cuda':
            from .physics import release_gpu_buffers
            release_gpu_buffers()
    manifest=[];seen=set()
    for i in range(count):
        p=folder/f'{i:06d}.h5';r=json.loads(p.with_suffix('.json').read_text())
        if r['sha256']!=sha(p) or r['contract']!=contract:raise ValueError(f'audit failed {p}')
        if r['velocity_sha256'] in seen:raise ValueError('duplicate medium across dataset')
        seen.add(r['velocity_sha256']);manifest.append(dict(path=p.name,**r))
    if runtime['device']=='cuda':
        from .gpu_statistics import fit_statistics_gpu
        stats=fit_statistics_gpu(folder,cfg,contract,Path(run)/'normalization_work',batch_size=4,readers=4)
    else:stats=fit_statistics(folder,cfg,contract)
    atomic_json(folder/'normalization.json',stats)
    atomic_json(folder/'manifest.json',dict(contract=contract,records=manifest))
    atomic_json(folder/'COMPLETE.json',dict(contract=contract,count=count,train=stats['count'],validation=count-stats['count'],
                manifest_sha256=sha(folder/'manifest.json'),stats_sha256=sha(folder/'normalization.json'),audit='PASS'))
    return folder


class Records(torch.utils.data.Dataset):
    def __init__(self,folder,contract,split):
        self.folder=Path(folder);self.contract=contract
        m=json.loads((self.folder/'manifest.json').read_text())
        if m['contract']!=contract:raise ValueError('training manifest mismatch')
        self.paths=[self.folder/r['path'] for r in m['records'] if r['split']==split]
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        v,loc,u,_=read_record(self.paths[i],self.contract)
        return torch.from_numpy(v),torch.from_numpy(loc),torch.from_numpy(u)
