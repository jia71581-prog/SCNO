"""Native bilinear forward, verified Cartesian/time-repeat field transpose only."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

def axis_matrix(axis,extent):
 # Actual incoming normalized coordinates, identical alignTrue border mapping.
 pixel=(((axis+1)/2)*(extent-1)).clamp(0,extent-1)
 low=pixel.floor().long();high=(low+1).clamp_max(extent-1);fraction=pixel-low.to(pixel.dtype)
 return F.one_hot(low,num_classes=extent).to(axis.dtype)*(1-fraction)[...,None]+F.one_hot(high,num_classes=extent).to(axis.dtype)*fraction[...,None]

def verify_cartesian(field,grid,target_shape):
 if field.ndim!=4 or grid.ndim!=4 or grid.shape[-1]!=2 or grid.shape[0]!=field.shape[0] or grid.shape[1]!=1:return None
 if grid.requires_grad or not field.requires_grad or field.dtype not in (torch.float32,torch.float64) or grid.dtype!=field.dtype or grid.device!=field.device:return None
 hq,wq=target_shape;p=grid.shape[2]
 if hq*wq<=1 or p%(hq*wq) or p<=1:return None
 t=p//(hq*wq);values=grid.reshape(grid.shape[0],t,hq,wq,2)
 if not bool(torch.isfinite(values).all()):return None
 x=values[:,0,0,:,0];z=values[:,0,:,0,1]
 # Exact validation across EVERY time/row/column; dimensions alone never suffice.
 if not torch.equal(values[...,0],x[:,None,None,:].expand(-1,t,hq,-1)) or not torch.equal(values[...,1],z[:,None,:,None].expand(-1,t,-1,wq)):return None
 return t,x,z

class _NativeForwardFastFieldAdjoint(torch.autograd.Function):
 @staticmethod
 def forward(ctx,field,grid,original,pz,px,t,hq,wq):
  ctx.save_for_backward(pz,px);ctx.t=t;ctx.hq=hq;ctx.wq=wq
  return original(field,grid,mode='bilinear',padding_mode='border',align_corners=True)
 @staticmethod
 def backward(ctx,g):
  pz,px=ctx.saved_tensors;n,c=g.shape[:2]
  time_sum=g.reshape(n,c,ctx.t,ctx.hq,ctx.wq).sum(dim=2)
  field_grad=torch.matmul(torch.matmul(pz.transpose(-2,-1)[:,None],time_sum),px[:,None])
  return field_grad,None,None,None,None,None,None,None

@contextmanager
def scoped_cartesian_adjoint(records,generic_helper,target_shape=(201,201)):
 original=F.grid_sample;cache={}
 def wrapped(input,grid,mode='bilinear',padding_mode='zeros',align_corners=None):
  supported=input.ndim==4 and grid.ndim==4 and mode=='bilinear' and padding_mode=='border' and align_corners is True
  dispatch='original_mode_fallback';verified=False;repeat=None;cache_hit=False
  if supported and (not torch.is_grad_enabled() or not input.requires_grad and not grid.requires_grad):
   dispatch='native_forward_no_backward_required';output=original(input,grid,mode=mode,padding_mode=padding_mode,align_corners=align_corners)
  elif supported:
   if not bool(torch.isfinite(grid).all()):raise ValueError('registered finite-coordinate sampling only')
   descriptor=verify_cartesian(input,grid,target_shape)
   if descriptor is not None:
    repeat,x,z=descriptor;verified=True;hq,wq=target_shape
    key=(str(input.device),input.dtype,tuple(input.shape),hq,wq);prior=cache.get(key)
    # Cache reuse still verifies actual axis values each call and never aliases grids.
    if prior is not None and torch.equal(prior[0],x) and torch.equal(prior[1],z):pz,px=prior[2:];cache_hit=True
    else:
     pz=axis_matrix(z,input.shape[-2]);px=axis_matrix(x,input.shape[-1]);cache[key]=(x.clone(),z.clone(),pz,px)
    dispatch='verified_cartesian_fast_field_adjoint';output=_NativeForwardFastFieldAdjoint.apply(input,grid,original,pz,px,repeat,hq,wq)
   else:
    dispatch='verified_generic_deterministic_fallback';output=generic_helper._OriginalForwardDeterministicBackward.apply(input,grid,original)
  else:output=original(input,grid,mode=mode,padding_mode=padding_mode,align_corners=align_corners)
  records.append(dict(call=len(records)+1,input_shape=list(input.shape),grid_shape=list(grid.shape),dtype=str(input.dtype),device=str(input.device),field_requires_grad=input.requires_grad,grid_requires_grad=grid.requires_grad,grad_enabled=torch.is_grad_enabled(),mode=mode,padding_mode=padding_mode,align_corners=align_corners,dispatch=dispatch,all_values_Cartesian_time_repeat_verified=verified,time_repeat=repeat,target_shape=list(target_shape),actual_axis_cache_hit=cache_hit,forward='original_F_grid_sample'))
  return output
 F.grid_sample=wrapped
 try:yield
 finally:F.grid_sample=original
