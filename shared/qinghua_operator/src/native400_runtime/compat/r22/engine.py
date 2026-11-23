"""Frozen original101 inference plus true201 known conditions, TRAIN-only."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
from torch.nn import functional as F
class FrozenBase:
    def __init__(self,config,family,device):
        self.config=config;self.family=family;self.device=device
        r21=Path(config['r21_root']);sys.path.insert(0,str(r21))
        import bootstrap,reader92,known_inputs,data_views201
        self.core=bootstrap.upstream('shorttrain_core_r2');self.bridge=bootstrap.upstream('model_bridge')
        from boundary_exact import install_exact_boundary
        from protocol92 import FULL_FRAMES,FORECAST_FRAMES
        self.full_frames=tuple(FULL_FRAMES);self.frames=tuple(FORECAST_FRAMES);self.known_inputs=known_inputs;self.view=data_views201
        binding=json.loads((r21/'INPUT_BINDINGS.json').read_text())['families'][family];old=binding['original_code_and_inputs']
        self.loader=reader92.Forecast92Loader(family,old['parent']['code'],old['parent']['config']);self.ids=list(self.loader.ids)
        if self.ids!=old['TRAIN_record_ids']:raise PermissionError('original TRAIN roster')
        self.helpers={k:self.bridge.module('r22_'+family+'_'+k,v['path']) for k,v in old['helpers'].items()}
        self.model,self.norm,_,self.provider=self.bridge.released_parent(family,str(device),gpu_authorized=True)
        parent=config['family'][family];path=Path(parent['base_checkpoint']);h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
        if h.hexdigest()!=parent['base_SHA']:raise PermissionError('frozen base SHA')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['family']!=family or not saved.get('no_optimizer_reset'):raise PermissionError('complete R20 baseline')
        self.model.load_state_dict(saved['model_state'],strict=True);self.model.scno_depth=2;install_exact_boundary(self.model)
        self.model.requires_grad_(False);self.model.eval();self.base_model_digest=self.core.tree_digest(self.model.state_dict())
        self.base_source_clocks=dict(native=saved['updates_completed'],Adam_age=saved['optimizer_age'])
        self.base_source_optimizer_digest=self.core.tree_digest(saved['optimizer_state'])
        if torch.count_nonzero(self.norm.decode_pressure(torch.zeros(1,1,1,1,device=device),torch.ones(1,device=device))):raise ValueError('zero mean linear normalization required')
        del saved
    def read_train(self,sid):return self.loader.read(sid)
    def read_known(self,sid):return self.known_inputs.known_only(self.loader,sid,self.view,self.full_frames)
    def prepare(self,data,packet):
        if packet['split']!='train' or not packet.get('IC_physical_Pa'):raise PermissionError('known input provenance')
        coarse=self.bridge.restrict_sample(data,packet['source_FP64'])
        coarse={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in coarse.items()}
        fine={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in data.items()}
        with torch.no_grad(),self.core.scopes(self.helpers,{k:[] for k in ('padding','sampling','pooling')}):prepared,grid=self.provider.prepare(self.model,self.norm,coarse)
        return fine,(prepared,grid,coarse)
    @torch.no_grad()
    def predict_base(self,context,indices):
        prepared,grid,coarse=context;native=[self.frames[int(j)] for j in indices];times=coarse['time_axis'][native][None]
        with self.core.scopes(self.helpers,{k:[] for k in ('padding','sampling','pooling')}):p=self.model.dense_normalized(prepared,times,dense_grid=grid,time_block=16,apply_correction=True)
        if p.shape!=(1,len(indices),101,101) or not torch.isfinite(p).all():raise FloatingPointError('base output')
        pa=self.norm.decode_pressure(p,coarse['source'][:,4]);pa201=F.interpolate(pa[0,:,None],size=(201,201),mode='bicubic',align_corners=True)
        normalized=self.norm.encode_pressure(pa201[:,0][None],coarse['source'][:,4])[0,:,None]
        if torch.count_nonzero(pa201[:,:,:1,:]):raise AssertionError('exact base top')
        return normalized,pa201
    def truth_after_seal(self,sid,seal):return self.known_inputs.truth_after_seal(self.loader,sid,self.view,self.full_frames,seal)[0]
    def close(self):self.loader.close()
