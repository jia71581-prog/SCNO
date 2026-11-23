"""Signed effective-source propagation for joint Marmousi pressure and B5."""

import torch
from torch import nn
from torch.nn import functional as F


COARSE_TIME = 256
NATIVE_TIME = 1024
DECIMATION = 4
COARSE_GRID = 51
N_BINS = 74
REFERENCE_SPEED = 2200.


def smooth_velocity(velocity, sigma_m):
    """Gaussian smooth in velocity units before forming squared slowness."""
    sigma=sigma_m/10.
    coords=torch.arange(-round(3*sigma),round(3*sigma)+1,device=velocity.device,dtype=velocity.dtype)
    kernel=(-.5*(coords/sigma).square()).exp()
    kernel=kernel/kernel.sum()
    radius=len(kernel)//2
    v=velocity.reshape(1,1,201,201)
    v=F.conv2d(F.pad(v,(radius,radius,0,0),mode="replicate"),kernel[None,None,None,:])
    v=F.conv2d(F.pad(v,(0,0,radius,radius),mode="replicate"),kernel[None,None,:,None])
    return v[0,0]


def spatial_coarse(value):
    return F.avg_pool2d(value.reshape(-1,1,201,201),5,4,2,count_include_pad=False)[:,0]


def normalized_slowness_difference(velocity, smooth):
    return REFERENCE_SPEED**2*(velocity.reciprocal().square()-smooth.reciprocal().square())


def effective_source(velocity, contrast, background_product, source):
    """Return complex normalized source and medium conditions on the coarse grid."""
    if background_product.ndim!=3 or background_product.shape[-2:]!=(COARSE_GRID,COARSE_GRID):
        raise ValueError("Use the decimated pressure-times-slowness product on the shared coarse grid")
    dm=spatial_coarse(contrast)[0]
    cut=min(16,len(background_product))
    window=torch.ones(len(background_product),device=background_product.device,dtype=background_product.dtype)
    window[-cut:]=torch.linspace(1.,0.,cut,device=window.device,dtype=window.dtype)
    p=background_product*window[:,None,None]
    spectrum=torch.fft.rfft(p,n=COARSE_TIME,dim=0,norm="forward")[1:N_BINS+1]
    freq=torch.fft.rfftfreq(COARSE_TIME,.005*DECIMATION,device=velocity.device)[1:N_BINS+1]
    q=spectrum*(freq/source[2]).square()[:,None,None]
    x=torch.linspace(0.,1.,COARSE_GRID,device=velocity.device,dtype=velocity.dtype)[None,:].expand_as(dm)
    z=x.T
    v=spatial_coarse((velocity/REFERENCE_SPEED).log())[0]
    condition=torch.stack((v,dm,2.*x-1.,2.*z-1.,x-source[0]/2000.))
    return torch.view_as_real(q).permute(0,3,1,2).contiguous(),condition,freq/source[2]


class GreenStage(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.local=nn.Conv2d(width,width,3,padding=1,bias=False)
        self.coarse=nn.Sequential(nn.Conv2d(width,width,3,padding=2,dilation=2,bias=False),
                                  nn.SiLU(),nn.Conv2d(width,width,3,padding=3,dilation=3,bias=False))
        self.gate=nn.Conv2d(6,width,1)
        self.norm=nn.GroupNorm(4,width,affine=False)

    def forward(self,x,m):
        pooled=F.avg_pool2d(x,4,stride=3,padding=1,count_include_pad=False)
        propagated=F.interpolate(self.coarse(pooled),size=x.shape[-2:],mode="bilinear",align_corners=True)
        field=self.norm(self.local(x)+propagated)
        return x+.45*F.silu(field)*(1.+.4*self.gate(m).tanh())


class SourcePropagator(nn.Module):
    def __init__(self,width=24):
        super().__init__()
        self.lift=nn.Conv2d(2,width,1,bias=False)
        self.layers=nn.ModuleList([GreenStage(width) for _ in range(3)])
        self.out=nn.Conv2d(width,2,1,bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self,q,medium,freq):
        cond=torch.cat((medium[None].expand(len(q),-1,-1,-1),
                        freq[:,None,None,None].expand(-1,1,*q.shape[-2:])),dim=1)
        # Keep the response homogeneous in source scale, including a zero source.
        source_rms=q.square().mean((1,2,3),keepdim=True).clamp_min(1.e-12).sqrt()
        field=self.lift(q/source_rms)
        for layer in self.layers:
            field=layer(field,cond)
        response=self.out(field)*source_rms
        complex_response=torch.view_as_complex(response.permute(0,2,3,1).contiguous())
        coeff=torch.cat((complex_response.new_zeros((1,*q.shape[-2:])),complex_response))
        coarse=torch.fft.irfft(coeff,n=NATIVE_TIME,dim=0,norm="forward")[:801]
        times=torch.arange(801,device=coarse.device,dtype=coarse.dtype)
        taper=((times-2.)/5.).clamp(0.,1.)*((800.-times)/24.).clamp(0.,1.)
        return coarse*taper[:,None,None]*(torch.arange(COARSE_GRID,device=q.device)>0)[None,:,None]


def full_pressure(coarse, slots):
    return F.interpolate(coarse[slots,None],size=(201,201),mode="bilinear",align_corners=True)[:,0]


def b5_pressure(coarse, observation):
    rows=observation.support_depth_rows
    slab=F.interpolate(coarse[:,None,:2],size=(5,201),mode="bilinear",align_corners=True)[:,0,:rows]
    return slab.flatten(1)@observation.matrix.to(slab)
