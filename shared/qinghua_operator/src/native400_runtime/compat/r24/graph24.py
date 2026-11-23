"""Identity-deduplicated encoder VJP and deterministic bicubic adjoint."""
from dataclasses import is_dataclass,fields,replace
import torch
from torch.nn import functional as F
_CACHE={}
class GraphCut:
    def __init__(self,tree):
        self.original=[];self.proxies=[];self._seen={}
        self.tree=self._convert(tree)
    def _convert(self,x):
        if torch.is_tensor(x):
            if not x.requires_grad:return x
            k=id(x)
            if k not in self._seen:
                proxy=x.detach().requires_grad_(True);self._seen[k]=proxy
                self.original.append(x);self.proxies.append(proxy)
            return self._seen[k]
        if is_dataclass(x) and not isinstance(x,type):
            return replace(x,**{f.name:self._convert(getattr(x,f.name)) for f in fields(x) if f.init})
        if isinstance(x,dict):return {k:self._convert(v) for k,v in x.items()}
        if isinstance(x,list):return [self._convert(v) for v in x]
        if isinstance(x,tuple):
            v=[self._convert(a) for a in x]
            return type(x)(*v) if hasattr(x,'_fields') else tuple(v)
        return x
    def backward(self):
        if self.original:
            torch.autograd.backward(self.original,[p.grad if p.grad is not None else torch.zeros_like(p) for p in self.proxies])
    def release(self):
        self.original.clear();self.proxies.clear();self._seen.clear();self.tree=None

def _matrix(ni,no,device,dtype):
    key=(ni,no,str(device),dtype)
    if key not in _CACHE:
        # Each channel is a source basis vector. Use the official forward kernel
        # on CPU to obtain separable interpolation weights, then transfer.
        eye=torch.eye(ni,dtype=torch.float64).reshape(1,ni,1,ni)
        w=F.interpolate(eye,size=(1,no),mode='bicubic',align_corners=True)[0,:,0].t().contiguous()
        _CACHE[key]=w.to(device=device,dtype=dtype)
    return _CACHE[key]
class Bicubic201(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.hw=x.shape[-2:]
        return F.interpolate(x,size=(201,201),mode='bicubic',align_corners=True)
    @staticmethod
    def backward(ctx,g):
        h,w=ctx.hw;z=_matrix(h,201,g.device,g.dtype);x=_matrix(w,201,g.device,g.dtype)
        return torch.matmul(z.t(),torch.matmul(g,x))
def bicubic201(x):return Bicubic201.apply(x)
