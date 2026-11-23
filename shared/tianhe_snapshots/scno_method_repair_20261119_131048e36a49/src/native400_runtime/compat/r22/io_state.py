"""Atomic complete base/refiner state and exact per-rank RNG restoration."""
from pathlib import Path
import os,json,hashlib,random
import numpy as np
import torch
import torch.distributed as dist
from sampler import RecordCursor,TimeSampler

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
    return h.hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch']);torch.cuda.set_rng_state(x['cuda'])
def save(path,model,opt,cursor,sampler,engine,config_sha,stage,baseline,last_metrics,strict=False):
    rank=dist.get_rank();digest=engine.core.tree_digest
    identities=dict(model=digest(model.state_dict()),opt=digest(opt.state_dict()),cursor=digest(cursor.state_dict()),sampler=digest(sampler.state_dict()))
    all_id=[None]*dist.get_world_size();all_rng=[None]*dist.get_world_size()
    dist.all_gather_object(all_id,identities);dist.all_gather_object(all_rng,rng_state())
    if any(x!=identities for x in all_id):raise AssertionError('all-rank state differs')
    path=Path(path);error=None
    if rank==0:
        try:
            payload=dict(schema='R22_COMPLETE_BASE_AND_REFINER_WORLD2_V1',family=engine.family,config_sha=config_sha,stage=stage,
                base_source=engine.config['family'][engine.family],base_model_digest=engine.base_model_digest,
                base_source_clocks=engine.base_source_clocks,base_source_optimizer_digest=engine.base_source_optimizer_digest,
                base_model_state={k:v.detach().cpu() for k,v in engine.model.state_dict().items()},refiner_state=model.state_dict(),optimizer_state=opt.state_dict(),
                cursor=cursor.state_dict(),sampler=sampler.state_dict(),rng_by_rank=all_rng,full_state_digests=all_id,baseline=baseline,last_metrics=last_metrics,
                parent_weights_frozen=True,new_optimizer_only_new_parameters=True,output_shape=[201,201,100],score_only_unknown92=True)
            tmp=path.with_suffix('.pt.tmp');torch.save(payload,tmp)
            with tmp.open('rb') as f:os.fsync(f.fileno())
            tmp.replace(path)
            write_json(path.parent/'LATEST.json',dict(path=str(path),SHA=sha(path),step=cursor.steps,stage=stage,records=cursor.exposures,record_epochs=cursor.exposures/len(cursor.ids),digests=identities))
        except BaseException as ex:error=repr(ex)
    errors=[None]*dist.get_world_size();dist.all_gather_object(errors,error)
    if any(errors):raise RuntimeError(str(errors))
    if strict:
        old_rng=digest(rng_state());rc,ts,_,_=load(path,model,opt,engine,config_sha)
        after=dict(model=digest(model.state_dict()),opt=digest(opt.state_dict()),cursor=digest(rc.state_dict()),sampler=digest(ts.state_dict()))
        if after!=identities or digest(rng_state())!=old_rng:raise AssertionError('strict save/reload')
        write_json(path.parent/f'RELOAD_STEP{cursor.steps:06d}_RANK{rank}.json',dict(exact=True,digests=after,RNG_SHA=old_rng))
    return identities

def load(path,model,opt,engine,config_sha):
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p['schema']!='R22_COMPLETE_BASE_AND_REFINER_WORLD2_V1' or p['family']!=engine.family or p['config_sha']!=config_sha or p['base_model_digest']!=engine.base_model_digest or p['base_source']['base_SHA']!=engine.config['family'][engine.family]['base_SHA']:raise PermissionError('matching R22 recipe/base')
    if engine.core.tree_digest(p['base_model_state'])!=engine.base_model_digest:raise PermissionError('saved full base digest')
    model.load_state_dict(p['refiner_state'],strict=True);opt.load_state_dict(p['optimizer_state'])
    cursor=RecordCursor(engine.ids,state=p['cursor']);sampler=TimeSampler(len(engine.ids),p['sampler'])
    current=dict(model=engine.core.tree_digest(model.state_dict()),opt=engine.core.tree_digest(opt.state_dict()),cursor=engine.core.tree_digest(cursor.state_dict()),sampler=engine.core.tree_digest(sampler.state_dict()))
    if any(x!=current for x in p['full_state_digests']):raise AssertionError('complete tuple mismatch')
    restore_rng(p['rng_by_rank'][dist.get_rank()])
    return cursor,sampler,p['stage'],p
