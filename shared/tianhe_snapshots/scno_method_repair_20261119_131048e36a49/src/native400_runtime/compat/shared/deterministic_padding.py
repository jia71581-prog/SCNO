"""Equivalent 4D nonnegative replicate pad2d, scoped functional interception."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

def replicate_pad2d(input,padding):
 if input.ndim!=4:raise ValueError('registered NCHW only')
 if len(padding)!=4 or any(type(p) is not int or p<0 for p in padding):raise ValueError('four nonnegative integer padding widths')
 left,right,top,bottom=padding;h,w=input.shape[-2:]
 if h<=0 or w<=0:raise ValueError('positive spatial extents')
 ih=torch.arange(-top,h+bottom,device=input.device,dtype=torch.int64).clamp(0,h-1)
 iw=torch.arange(-left,w+right,device=input.device,dtype=torch.int64).clamp(0,w-1)
 return input.index_select(-2,ih).index_select(-1,iw)

@contextmanager
def scoped_replicate_pad2d(records):
 original=F.pad
 def wrapped(input,pad,mode='constant',value=None):
  if mode=='replicate' and input.ndim==4 and len(pad)==4 and all(type(p) is int and p>=0 for p in pad):
   output=replicate_pad2d(input,tuple(pad))
   records.append(dict(call=len(records)+1,input_shape=list(input.shape),padding=list(pad),output_shape=list(output.shape),dtype=str(input.dtype),device=str(input.device),input_requires_grad=input.requires_grad,grad_enabled=torch.is_grad_enabled()))
   return output
  return original(input,pad,mode=mode,value=value)
 F.pad=wrapped
 try:yield
 finally:F.pad=original
