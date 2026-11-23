"""Known-input, time-conditioned residual correction; no future truth input."""
from __future__ import annotations
import math
from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F
WIDTHS=(16,24,40,64,96)
_RESIZE_CACHE={}

def resize_bilinear(x,size):
    """Align-corners bilinear via fixed matrices, without atomic scatter backward."""
    hi,wi=x.shape[-2:];ho,wo=size;key=(hi,wi,ho,wo,str(x.device),x.dtype)
    if key not in _RESIZE_CACHE:
        def matrix(nin,nout):
            loc=torch.linspace(0,nin-1,nout,device=x.device,dtype=x.dtype)
            lower=loc.floor().long();upper=(lower+1).clamp_max(nin-1);weight=loc-lower.to(loc.dtype)
            m=torch.zeros(nout,nin,device=x.device,dtype=x.dtype);row=torch.arange(nout,device=x.device)
            m[row,lower]=1-weight;m[row,upper]+=weight
            return m
        _RESIZE_CACHE[key]=(matrix(hi,ho),matrix(wi,wo))
    hz,wx=_RESIZE_CACHE[key]
    return torch.matmul(hz,torch.matmul(x,wx.t()))

class SpatialBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__();self.conv1=nn.Conv2d(cin,cout,3,stride=stride,padding=1)
        self.norm1=nn.GroupNorm(4,cout);self.conv2=nn.Conv2d(cout,cout,3,padding=1);self.norm2=nn.GroupNorm(4,cout)
    def forward(self,x):return F.silu(self.norm2(self.conv2(F.silu(self.norm1(self.conv1(x))))))

class ConditionedBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__();self.block=SpatialBlock(cin,cout,stride);self.film=nn.Linear(64,2*cout)
        nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
    def forward(self,x,embedding):
        x=self.block(x);gamma,beta=self.film(embedding).chunk(2,dim=-1)
        return x*(1+gamma[:,:,None,None])+beta[:,:,None,None]

class ResidualRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.static=nn.ModuleList([SpatialBlock(14,WIDTHS[0])])
        self.static.extend(SpatialBlock(a,b,2) for a,b in zip(WIDTHS[:-1],WIDTHS[1:]))
        self.dynamic=nn.ModuleList([ConditionedBlock(4,WIDTHS[0])])
        self.dynamic.extend(ConditionedBlock(a,b,2) for a,b in zip(WIDTHS[:-1],WIDTHS[1:]))
        self.up=nn.ModuleList([ConditionedBlock(WIDTHS[i+1]+WIDTHS[i],WIDTHS[i]) for i in range(len(WIDTHS)-2,-1,-1)])
        self.embedding=nn.Sequential(nn.Linear(19,64),nn.SiLU(),nn.Linear(64,64))
        self.token_norm=nn.LayerNorm(WIDTHS[-1]);self.qkv=nn.Linear(WIDTHS[-1],3*WIDTHS[-1]);self.attention_out=nn.Linear(WIDTHS[-1],WIDTHS[-1])
        self.output=nn.Conv2d(WIDTHS[0],1,3,padding=1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('time_hz',torch.tensor([1.,2.,4.,8.,16.,32.]),persistent=True)
    def prepare(self,known):
        if known.ndim!=4 or known.shape[0]!=1 or known.shape[1]!=14:raise ValueError('known [1,14,z,x]')
        result=[]
        for block in self.static:known=block(known);result.append(known)
        return result
    def forward(self,features,base,times,source):
        if base.ndim!=4 or base.shape[1]!=1 or len(features)!=len(WIDTHS):raise ValueError('base [queries,1,z,x]')
        n=base.shape[0]
        if times.numel()!=n or source.shape!=(1,5):raise ValueError('physical times and source')
        scale=base.square().mean(dim=(-2,-1),keepdim=True).add(1e-6).sqrt();u=base/scale
        dx=F.pad(u[:,:,:,1:]-u[:,:,:,:-1],(0,1,0,0));dz=F.pad(u[:,:,1:,:]-u[:,:,:-1,:],(0,0,0,1))
        x=torch.cat((u,dx,dz,u.abs()),1)
        phase=2*math.pi*times.reshape(n,1)*self.time_hz.reshape(1,-1)
        src=source/source.new_tensor([[2000.,2000.,15.,1.,1.]])
        emb=self.embedding(torch.cat((times.reshape(n,1),phase.sin(),phase.cos(),src.expand(n,-1),scale.reshape(n,1).log()),1))
        skips=[]
        for block,condition in zip(self.dynamic,features):
            x=block(x,emb)+condition.expand(n,-1,-1,-1);skips.append(x)
        nz,nx=x.shape[-2:];tokens=x.flatten(2).transpose(1,2);normalized=self.token_norm(tokens)
        qkv=self.qkv(normalized).reshape(n,nz*nx,3,4,WIDTHS[-1]//4).permute(2,0,3,1,4);q,k,v=qkv.unbind(0)
        attention=((q@k.transpose(-2,-1))/math.sqrt(WIDTHS[-1]//4)).softmax(-1)
        global_value=(attention@v).transpose(1,2).reshape(n,nz*nx,WIDTHS[-1]);tokens=tokens+self.attention_out(global_value)
        x=tokens.transpose(1,2).reshape(n,WIDTHS[-1],nz,nx)
        for block,skip in zip(self.up,skips[-2::-1]):
            x=resize_bilinear(x,skip.shape[-2:]);x=block(torch.cat((x,skip),1),emb)
        correction=self.output(x)*scale
        return torch.cat((torch.zeros_like(correction[:,:,:1,:]),correction[:,:,1:,:]),-2)

def build_known(data,normalizer):
    velocity=data['velocity']
    if velocity.shape!=(1,1,201,201) or data['ic'].shape!=(1,8,201,201):raise ValueError('native201 known inputs')
    ic=normalizer.encode_pressure(data['ic'],data['source'][:,4]);z,x=torch.meshgrid(data['z'],data['x'],indexing='ij');source=data['source'][0]
    coords=torch.stack((2*x/2000-1,2*z/2000-1,(x-source[0])/2000,(z-source[1])/2000),0)[None]
    result=torch.cat((velocity/3000-1,ic,data['source_map'],coords),1)
    if result.shape!=(1,14,201,201) or not torch.isfinite(result).all():raise FloatingPointError('known features')
    return result

def loss_sums(error,h_m=10.,length_m=20.,h1_weight=.02):
    """Value error and physically scaled forward-difference derivative error.
    Both use the same true pressure energy denominator in the caller. The H1
    auxiliary is bounded above by .64 * L2 under these default scales.
    """
    if h_m<=0 or length_m<=0 or h1_weight<0:raise ValueError('physical scales')
    e=error.double();l2=e.square().sum((1,2,3))
    gx=(e[:,:,:,1:]-e[:,:,:,:-1])/h_m;gz=(e[:,:,1:,:]-e[:,:,:-1,:])/h_m
    return l2,h1_weight*length_m**2*(gx.square().sum((1,2,3))+gz.square().sum((1,2,3)))
