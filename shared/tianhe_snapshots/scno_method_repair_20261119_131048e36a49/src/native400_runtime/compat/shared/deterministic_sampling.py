"""Original bilinear/border/alignTrue forward, deterministic differentiable adjoint."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

class _BorderCoordinate(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,maximum):
  ctx.save_for_backward(x);ctx.maximum=maximum
  return x.clamp(0,maximum)
 @staticmethod
 def backward(ctx,g):
  (x,)=ctx.saved_tensors
  # ATen border coordinates have zero derivative AT and outside both edges.
  return g*((x>0)&(x<ctx.maximum)).to(g.dtype),None

def bilinear_border_values(field,grid):
 if field.ndim!=4 or grid.ndim!=4 or grid.shape[-1]!=2 or grid.shape[0]!=field.shape[0]:raise ValueError('NCHW and NHW2 same batch')
 if field.dtype not in (torch.float32,torch.float64) or field.dtype!=grid.dtype or field.device!=grid.device:raise ValueError('registered matching F32/F64 field/grid')
 n,c,h,w=field.shape;oh,ow=grid.shape[1:3]
 if min(n,c,h,w,oh,ow)<=0:raise ValueError('positive dimensions')
 x=_BorderCoordinate.apply(((grid[...,0]+1)/2)*(w-1),w-1)
 y=_BorderCoordinate.apply(((grid[...,1]+1)/2)*(h-1),h-1)
 x0=x.floor().long();y0=y.floor().long();x1=(x0+1).clamp_max(w-1);y1=(y0+1).clamp_max(h-1)
 u=(x-x0.to(x.dtype)).reshape(n,1,-1);v=(y-y0.to(y.dtype)).reshape(n,1,-1)
 flat=field.reshape(n,c,h*w)
 def corner(ix,iy):return flat.gather(2,(iy*w+ix).reshape(n,1,-1).expand(-1,c,-1))
 a=corner(x0,y0);b=corner(x1,y0);d=corner(x0,y1);e=corner(x1,y1)
 return ((1-u)*(1-v)*a+u*(1-v)*b+(1-u)*v*d+u*v*e).reshape(n,c,oh,ow)

class _OriginalForwardDeterministicBackward(torch.autograd.Function):
 @staticmethod
 def forward(ctx,field,grid,original):
  ctx.save_for_backward(field,grid)
  return original(field,grid,mode='bilinear',padding_mode='border',align_corners=True)
 @staticmethod
 def backward(ctx,g):
  field,grid=ctx.saved_tensors;need_field,need_grid=ctx.needs_input_grad[:2]
  higher=torch.is_grad_enabled()
  requested=tuple(t for t,needed in ((field,need_field),(grid,need_grid)) if needed)
  with torch.enable_grad():
   values=bilinear_border_values(field,grid)
   grads=torch.autograd.grad(values,requested,g,create_graph=higher) if requested else ()
  cursor=iter(grads)
  return next(cursor) if need_field else None,next(cursor) if need_grid else None,None

@contextmanager
def scoped_deterministic_sampling(records):
 original=F.grid_sample
 def wrapped(input,grid,mode='bilinear',padding_mode='zeros',align_corners=None):
  if input.ndim==4 and grid.ndim==4 and mode=='bilinear' and padding_mode=='border' and align_corners is True:
   if not bool(torch.isfinite(grid).all()):raise ValueError('registered finite-coordinate sampling only')
   records.append(dict(call=len(records)+1,input_shape=list(input.shape),grid_shape=list(grid.shape),dtype=str(input.dtype),device=str(input.device),field_requires_grad=input.requires_grad,grid_requires_grad=grid.requires_grad,grad_enabled=torch.is_grad_enabled(),mode=mode,padding_mode=padding_mode,align_corners=True,forward='original_F_grid_sample',backward='four_corner_gather_differentiable_adjoint'))
   return _OriginalForwardDeterministicBackward.apply(input,grid,original)
  return original(input,grid,mode=mode,padding_mode=padding_mode,align_corners=align_corners)
 F.grid_sample=wrapped
 try:yield
 finally:F.grid_sample=original
