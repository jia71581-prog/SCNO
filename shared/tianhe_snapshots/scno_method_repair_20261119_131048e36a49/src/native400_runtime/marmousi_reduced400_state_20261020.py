"""V2 joint checkpoint validation and a small, journaled latest-only store.

AIS codecs and numerical checks are shared with the original joint run. Only
the reduced model signature and rebased optimizer clocks differ.
"""
from __future__ import annotations
import copy
import datetime
import scno_posix_lock as fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

import torch

from . import marmousi_direct400_state_20261020 as old
from .paper201_time_axis_20261020 import native_time_axis

SCHEMA = 'MARMOUSI_REDUCED400_JOINT_CONTINUATION_V2'
BOUNDARY = old.BOUNDARY
pack_sampler, unpack_sampler, sampler_binding = old.pack_sampler, old.unpack_sampler, old.sampler_binding
require, canonical = old._require, old._canonical


def signature(state):
    return {n:dict(shape=list(v.shape),dtype=str(v.dtype)) if torch.is_tensor(v)
            else dict(extra_state=copy.deepcopy(v)) for n,v in state.items()}


def _optimizer(value, groups, model, ages, label):
    require(type(value)==dict and set(value)=={'state','param_groups'}, label+' complete state')
    require(len(value['param_groups'])==len(groups), label+' group count')
    all_ids=[]
    for actual,spec in zip(value['param_groups'],groups):
        require(len(actual['params'])==len(spec['parameter_names']),label+' group roster')
        for identifier,name in zip(actual['params'],spec['parameter_names']):
            all_ids.append(identifier)
            moment=value['state'].get(identifier)
            require(identifier not in value['state'] or type(moment)==dict,label+' moment entry type')
            # Share the original tested Adam numerical/group/moment validator;
            # each reduced tensor has its own explicitly inherited base age.
            group=dict(actual,params=[identifier])
            moments={} if not moment else {identifier:moment}
            old._optimizer_state(dict(state=moments,param_groups=[group]),[[name]],[spec],
                                 model,ages[name],label=label+' '+name)
    require(len(all_ids)==len(set(all_ids)) and set(value['state'])<=set(all_ids),label+' optimizer IDs')


def validate(state,cfg,config_sha,*,allow_cpu_rng=False):
    require(state.get('schema')==cfg['schema']==SCHEMA,'reduced joint schema')
    for key,expected in [('config_sha256',config_sha),('dataset_sha256',cfg['dataset_sha256']),
            ('dataset_complete_sha256',cfg['dataset_complete_sha256']),
            ('parent_sha256',cfg['parent_sha256']),('migration_parent_sha256',cfg['migration_parent_sha256']),
            ('immutable_digest',cfg['immutable_digest'])]:
        require(state.get(key)==expected,'checkpoint binding '+key)
    epoch,offset,updates=(old._integer(state.get(k),k) for k in ('epoch','offset','updates'))
    local=updates-cfg['inherited_joint_updates']
    require(0<=epoch<=20 and 0<=offset<500 and offset%4==0 and (epoch<20 or offset==0),'cursor range')
    require(local>=0 and updates<=2500 and state.get('stage_updates')==local
            and state['task_exposures']==epoch*500+offset==updates*4,'global/stage clocks')
    require(state['current_epoch_order']==torch.randperm(500,generator=torch.Generator().manual_seed(cfg['seed']+epoch)).tolist(),
            'source order')
    require(state['checkpoint_boundary']==BOUNDARY,'complete joint transaction required')
    require(type(state['CUDA_restore8']) is bool and (not state['CUDA_restore8'] or local>=8)
            and (local<=8 or state['CUDA_restore8']),'new-stage CUDA restore boundary')
    require(state['active_parameter_names']==cfg['active_parameter_names'] and len(state['active_parameter_names'])==393,
            'reduced393 block identities')
    model=state['model_state'];require(set(model)==set(cfg['model_signature']),'complete reduced model keys')
    for name,spec in cfg['model_signature'].items():
        value=model[name]
        if 'extra_state' in spec:
            require(canonical(value)==canonical(spec['extra_state']),'model semantic state '+name)
        else:
            old._cpu_finite_tensor(value,name)
            require(list(value.shape)==spec['shape'] and str(value.dtype)==spec['dtype'],'model signature '+name)
    names=cfg['active_parameter_names'];groups=cfg['optimizer_groups']
    coverage=[n for g in groups for n in g['parameter_names']]
    require(len(coverage)==len(set(coverage))==393 and set(coverage)==set(names),'main optimizer coverage')
    ages={n:cfg['base_age_by_name'][n]+local for n in names}
    _optimizer(state['optimizer_state'],groups,model,ages,'mainAdamW')
    require(canonical(state['normalizer'])==canonical(cfg['normalizer']),'fixed normalizer')
    authority=native_time_axis()
    require(torch.equal(state['native_times_s'],authority) and
            torch.equal(state['registered_times_s'],authority[cfg['full_native_frames']]),'exact teacher time axis')
    learned=state['learned_optimizer_state']
    require(signature(learned).keys()==cfg['learned_signature'].keys(),'complete learned state')
    for name,spec in cfg['learned_signature'].items():
        value=learned[name]
        if 'extra_state' in spec:require(canonical(value)==canonical(spec['extra_state']),'controller semantic state')
        else:
            old._cpu_finite_tensor(value,'controller '+name,floating=True)
            require(list(value.shape)==spec['shape'] and str(value.dtype)==spec['dtype'],'controller signature '+name)
    meta_group=dict(lr=cfg['meta_learning_rate'],weight_decay=cfg['meta_weight_decay'],
                    parameter_names=cfg['meta_parameter_names'])
    _optimizer(state['meta_optimizer_state'],[meta_group],learned,
               {n:local for n in cfg['meta_parameter_names']},'phiAdam')
    samplers=state['samplers'];require(type(samplers)==dict and all(type(i) is int and 0<=i<500 for i in samplers),'AIS roster')
    visited=set(state['current_epoch_order'][:offset]);commits=0
    for i in range(500):
        expected=epoch+int(i in visited)
        if i not in samplers:require(expected==0,'missing visited AIS history');continue
        sampler=old._validate_compact(samplers[i])
        require(sampler['commits']==expected and sampler['seed']==cfg['seed']+i
                and old._same(sampler['binding'],sampler_binding(i,cfg,cfg['parent_sha256'])),'AIS history/binding')
        commits+=sampler['commits']
    require(commits==state['task_exposures'],'AIS total commits')
    old._check_rng(state['rng'],allow_cpu_rng=allow_cpu_rng)
    return dict(status='PASS',updates=updates,stage_updates=local,task_exposures=commits)


class LatestStore:
    """A pending journal bridges the atomic model file and its SHA receipt."""
    def __init__(self,run,cfg,config_sha):
        self.run=Path(run);self.cfg=cfg;self.config_sha=config_sha
        self.run.mkdir(parents=True,exist_ok=True)

    def _read(self,path,receipt):
        require(Path(receipt['path']).resolve()==(self.run/'latest.pt').resolve(),'receipt final path')
        old._digest(receipt['sha256'],'checkpoint receipt SHA')
        with Path(path).open('rb') as stream:
            before=os.fstat(stream.fileno());digest=hashlib.sha256()
            for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
            require(digest.hexdigest()==receipt['sha256'],'checkpoint SHA differs')
            stream.seek(0);state=torch.load(stream,map_location='cpu',weights_only=False)
            after=os.fstat(stream.fileno());current=Path(path).stat()
        identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)
        require(identity(before)==identity(after)==identity(current),'checkpoint changed while reading')
        validate(state,self.cfg,self.config_sha,allow_cpu_rng=True)
        require(all(type(receipt.get(k)) is int and state[k]==receipt[k]
                    for k in ('updates','stage_updates','epoch','offset','task_exposures')),'receipt cursor')
        return state

    def _recover(self):
        pending=self.run/'CHECKPOINT_PENDING.json'
        if not pending.exists():return
        value=json.loads(pending.read_text())
        require(value['schema']==SCHEMA and value['config_sha256']==self.config_sha,'pending journal binding')
        name=value['candidate'];require(Path(name).name==name and name.startswith('.latest.') and name.endswith('.tmp'),'candidate path')
        latest=self.run/'latest.pt';candidate=self.run/name;new=value['receipt']
        if latest.exists() and old._sha(latest)==new['sha256']:
            self._read(latest,new);old._write_json(self.run/'CHECKPOINT.json',new)
        elif latest.exists():
            previous=value['previous'];require(previous is not None,'unexpected existing latest')
            require(json.loads((self.run/'CHECKPOINT.json').read_text())==previous,'previous formal receipt changed')
            self._read(latest,previous)
            if candidate.exists():
                history=self.run/'checkpoint_recovery';history.mkdir(exist_ok=True)
                os.replace(candidate,history/candidate.name)
        else:
            require(value['previous'] is None,'missing prior latest')
            self._read(candidate,new);os.replace(candidate,latest);old._write_json(self.run/'CHECKPOINT.json',new)
        pending.unlink();old._sync_directory(self.run)

    def load(self):
        with (self.run/'.checkpoint_transaction.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);self._recover()
            return self._read(self.run/'latest.pt',json.loads((self.run/'CHECKPOINT.json').read_text()))

    def save(self,state):
        validate(state,self.cfg,self.config_sha,allow_cpu_rng=True)
        with (self.run/'.checkpoint_transaction.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);self._recover()
            previous=json.loads((self.run/'CHECKPOINT.json').read_text()) if (self.run/'latest.pt').exists() else None
            if previous:require(state['updates']>=self._read(self.run/'latest.pt',previous)['updates'],'checkpoint rewind')
            path=self.run/('.latest.'+uuid.uuid4().hex+'.tmp')
            with path.open('wb') as handle:
                torch.save(state,handle);handle.flush();os.fsync(handle.fileno())
            receipt=dict(path=str((self.run/'latest.pt').resolve()),sha256=old._sha(path),
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **{k:state[k] for k in ('updates','stage_updates','epoch','offset','task_exposures')})
            loaded=self._read(path,receipt);require(old._same(state,loaded),'serialized checkpoint differs')
            old._write_json(self.run/'CHECKPOINT_PENDING.json',dict(schema=SCHEMA,config_sha256=self.config_sha,
                candidate=path.name,receipt=receipt,previous=previous))
            os.replace(path,self.run/'latest.pt');old._sync_directory(self.run)
            old._write_json(self.run/'CHECKPOINT.json',receipt)
            (self.run/'CHECKPOINT_PENDING.json').unlink();old._sync_directory(self.run)
            return loaded
