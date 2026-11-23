"""Native adaptive-average forward, exact-window deterministic separable adjoint."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

def output_dimensions(input,output_size):
 if type(output_size) is int:result=(output_size,output_size)
 elif isinstance(output_size,(tuple,list)) and len(output_size)==2:result=tuple(input.shape[-2+i] if v is None else v for i,v in enumerate(output_size))
 else:raise ValueError('registered int or two-axis int/None output size')
 if any(type(v) is not int or v<=0 for v in result):raise ValueError('positive output extents')
 return result

def window_matrix(input_size,output_size,like):
 i=torch.arange(output_size,device=like.device,dtype=torch.int64)
 start=(i*input_size)//output_size
 end=((i+1)*input_size+output_size-1)//output_size
 columns=torch.arange(input_size,device=like.device,dtype=torch.int64)
 return ((columns[None,:]>=start[:,None])&(columns[None,:]<end[:,None])).to(like.dtype)/(end-start).to(like.dtype)[:,None]

class _NativeForwardDeterministicBackward(torch.autograd.Function):
 @staticmethod
 def forward(ctx,input,output_size,original):
  oh,ow=output_dimensions(input,output_size)
  ph=window_matrix(input.shape[-2],oh,input);pw=window_matrix(input.shape[-1],ow,input)
  ctx.save_for_backward(ph,pw)
  return original(input,output_size)
 @staticmethod
 def backward(ctx,g):
  ph,pw=ctx.saved_tensors
  # Exact transpose of each adaptive window, including nondivisible overlaps.
  return torch.matmul(torch.matmul(ph.transpose(0,1),g),pw),None,None

@contextmanager
def scoped_deterministic_pooling(records):
 original=F.adaptive_avg_pool2d
 def wrapped(input,output_size):
  supported=input.ndim==4 and input.dtype in (torch.float32,torch.float64)
  if supported:
   try:oh,ow=output_dimensions(input,output_size)
   except ValueError:supported=False
  if supported:
   records.append(dict(call=len(records)+1,input_shape=list(input.shape),output_size=output_size,actual_output_shape=[input.shape[0],input.shape[1],oh,ow],dtype=str(input.dtype),device=str(input.device),input_requires_grad=input.requires_grad,grad_enabled=torch.is_grad_enabled(),forward='original_F_adaptive_avg_pool2d',backward='P_H_transpose_G_P_W',exact_window_definition='floor(i*input/output),ceil((i+1)*input/output)'))
   return _NativeForwardDeterministicBackward.apply(input,output_size,original)
  return original(input,output_size)
 F.adaptive_avg_pool2d=wrapped
 try:yield
 finally:F.adaptive_avg_pool2d=original
