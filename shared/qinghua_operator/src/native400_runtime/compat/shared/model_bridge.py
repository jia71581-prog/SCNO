"""Original released factory with explicit101 inputs; no bank equality shortcuts."""
from pathlib import Path
import importlib.util,sys
from data_views import pressure_view,velocity_view,source_map101,FUTURE_FRAMES
ROOT=Path(__file__).resolve().parents[1]
PROVIDER=ROOT/'shared/provider.py'
INPUT_IO=None  # unused legacy source locator
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);v=importlib.util.module_from_spec(s);sys.modules[name]=v;s.loader.exec_module(v);return v

def released_parent(family,device,gpu_authorized=False):
 if family not in ('layered','marmousi'):raise ValueError('one fixed family per process')
 provider=module('fixed100_original_provider_'+family,PROVIDER);model,norm,identity=provider.load_parent(family,device,gpu_authorized=gpu_authorized)
 if identity['step']!=50000:raise PermissionError('original family50000 only')
 model.scno_depth=2;return model,norm,identity,provider

def restrict_sample(sample,source_FP64):
 import torch
 if sample['velocity'].shape!=(1,1,201,201) or sample['ic'].shape!=(1,8,201,201):raise ValueError('original knowninput201 IC8')
 out=dict(sample);out['velocity']=velocity_view(sample['velocity']);out['ic']=pressure_view(sample['ic']);out['source_map']=torch.tensor(source_map101(source_FP64),device=sample['velocity'].device)[None,None];out['source']=torch.tensor(source_FP64,dtype=torch.float32,device=sample['velocity'].device)[None];out['x']=sample['x'][::2];out['z']=sample['z'][::2]
 if out['velocity'].shape!=(1,1,101,101) or out['ic'].shape!=(1,8,101,101) or not bool(torch.all(out['ic'][...,0,:]==0)) or not torch.allclose(out['source_map'].sum(),torch.tensor(1.,device=out['source_map'].device),atol=2e-7,rtol=0):raise AssertionError('101 viewshape/top/unitmass')
 return out

def forward100(model,norm,sample,provider,time_block=100,physical=False):
 if len(FUTURE_FRAMES)!=100 or time_block<=0:raise ValueError('exactfull100 API')
 # Fresh parameter-dependent preparation every call; no detached encoder cache.
 value=provider.dense_forward(model,norm,sample,sample['time_axis'][list(FUTURE_FRAMES)][None],time_block=time_block,physical=physical)
 if value.shape!=(1,100,101,101):raise AssertionError('oneAPI exact100 output in originalfixedorder')
 return value
