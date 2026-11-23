"""Travel-time aligned neural pressure for the smoothed-medium branch."""

import torch
from torch import nn
from torch.nn import functional as F

from scno_forward_source_propagator import COARSE_TIME, N_BINS, spatial_coarse


def ray_geometry(velocity, source):
    axis=torch.linspace(0.,2000.,51,device=velocity.device,dtype=velocity.dtype)
    z,x=torch.meshgrid(axis,axis,indexing="ij")
    sx,sz=source[:2]
    start_z=torch.stack((sz,-sz))
    dx=x-sx
    dz=z[None]-start_z[:,None,None]
    ratio=(torch.arange(24,device=x.device,dtype=x.dtype)+.5)/24.
    nodes_x=(sx+ratio[None,:,None,None]*dx[None,None]).expand(2,-1,-1,-1)
    nodes_z=(start_z[:,None,None,None]+ratio[None,:,None,None]*dz[:,None]).abs()
    positions=torch.stack((nodes_x,nodes_z),dim=-1).reshape(1,48*51,51,2)/1000.-1.
    slowness=F.grid_sample(velocity.reciprocal()[None,None],positions,mode="bilinear",
                           align_corners=True,padding_mode="border").reshape(2,24,51,51).mean(1)
    distance=(dx.square()[None]+dz.square()+16.).sqrt()
    travel=slowness*distance+source[3]
    sm=spatial_coarse((velocity/2200.).log())[0]
    smooth_gradient=F.pad(sm.diff(dim=0),(0,0,0,1))
    features=torch.stack((3.*sm,10.*smooth_gradient,dx/2000.,z/2000.,x/2000.,
                          travel[0],travel[1],distance[0]/2000.))
    return features,travel


def learn_time_basis(fields, arrivals, rank):
    """Fit a temporal subspace on ordinary TRAIN backgrounds after phase alignment."""
    device=fields[0].device
    omega=2.*torch.pi*torch.fft.rfftfreq(COARSE_TIME,.02,device=device)
    covariance=torch.zeros(COARSE_TIME,COARSE_TIME,dtype=torch.float64,device=device)
    for field,delay in zip(fields,arrivals,strict=True):
        p=torch.fft.rfft(field,n=COARSE_TIME,dim=0)
        aligned=torch.fft.irfft(p*torch.exp(1j*omega[:,None,None]*delay[None]),n=COARSE_TIME,dim=0)
        aligned=aligned.flatten(1).double()
        aligned=aligned / aligned.norm().clamp_min(1.e-8)
        covariance.addmm_(aligned,aligned.mT)
    values,vectors=torch.linalg.eigh(covariance)
    base=vectors[:,-rank:].mT.float().contiguous()
    return base, float(values[-rank:].sum()/values.sum())


class BackgroundNeuralOperator(nn.Module):
    """Predict spatial amplitudes and small phase corrections for smooth backgrounds."""

    def __init__(self, basis, width=40):
        super().__init__()
        self.rank=len(basis)
        self.register_buffer("basis",torch.fft.rfft(basis,dim=1)[:,:N_BINS+1])
        self.register_buffer("omega",2.*torch.pi*torch.fft.rfftfreq(COARSE_TIME,.02)[:N_BINS+1])
        self.stem=nn.Sequential(nn.Conv2d(8,width,5,padding=2),nn.GroupNorm(4,width),nn.SiLU())
        self.layers=nn.ModuleList(nn.Sequential(
            nn.Conv2d(width,width,3,padding=d,dilation=d),nn.GroupNorm(4,width),nn.SiLU(),
            nn.Conv2d(width,width,1)) for d in (1,2,4))
        self.amplitude=nn.Conv2d(width,2*self.rank,1)
        self.phase=nn.Conv2d(width,2,1)
        for layer in (self.amplitude,self.phase):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, feature, travel):
        value=self.stem(feature[None])
        for layer in self.layers:
            value=value+.35*layer(value)
        amplitude=self.amplitude(value)[0].reshape(2,self.rank,51,51)
        delay=travel+.085*self.phase(value)[0].tanh()
        spec=torch.einsum("prxy,rf->pfxy",amplitude.to(self.basis),self.basis)
        translated=spec*torch.exp(-1j*self.omega[None,:,None,None]*delay[:,None])
        p=torch.fft.irfft(translated.sum(0),n=COARSE_TIME,dim=0)[:201]
        # Stored background frames start at zero, before the injected wave.
        return torch.cat((torch.zeros_like(p[:1]),p[1:]),dim=0)
