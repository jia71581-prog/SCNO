"""Compose a learned background and a learned signed-source response at native resolution."""

import torch
from torch import nn
from torch.nn import functional as F

from scno_forward_source_propagator import (
    COARSE_TIME, NATIVE_TIME, N_BINS, b5_pressure, effective_source, full_pressure,
    normalized_slowness_difference, smooth_velocity, spatial_coarse,
)
from scno_learned_background import ray_geometry


class NativeBackgroundLift(nn.Module):
    """Coarse pressure rows are cell averages, with samples at 10 m near the surface."""

    def __init__(self):
        super().__init__()
        centers=torch.arange(51,dtype=torch.float32)*40.
        centers[0],centers[-1]=10.,1990.
        axis=torch.cat((torch.zeros(1),centers,torch.tensor([2000.])))
        fine=torch.arange(201,dtype=torch.float32)*10.
        hi=torch.searchsorted(axis,fine,right=True).clamp(1,len(axis)-1)
        lo=hi-1
        self.register_buffer("lower",lo)
        self.register_buffer("upper",hi)
        self.register_buffer("blend",(fine-axis[lo])/(axis[hi]-axis[lo]))

    def forward(self,pressure,depth=201):
        x=F.interpolate(pressure[:,None],size=(51,201),mode="bilinear",align_corners=True)[:,0]
        x=torch.cat((torch.zeros_like(x[:,:1]),x,x[:,-1:]),1)
        return torch.lerp(x[:,self.lower[:depth]],x[:,self.upper[:depth]],self.blend[:depth,None].to(x))

    def native_time(self,pressure):
        # Decimated samples span the same padded physical time interval.
        coeff=torch.fft.rfft(pressure,n=COARSE_TIME,dim=0,norm="forward")[:N_BINS+1]
        decoded=torch.fft.irfft(coeff,n=NATIVE_TIME,dim=0,norm="forward")[:801]
        return torch.cat((torch.zeros_like(decoded[:3]),decoded[3:]),0)


class DirectBackgroundScattering(nn.Module):
    """Forward pressure = neural background + neural propagation of its signed source."""

    def __init__(self,background,scatter,observation):
        super().__init__()
        self.background=background
        self.scatter=scatter
        self.geometry=NativeBackgroundLift()
        self.observation=observation

    def prepare(self,velocity,source,smoothed=None):
        smooth=smooth_velocity(velocity,150.) if smoothed is None else smoothed
        coarse=self.background(*ray_geometry(smooth,source))
        dm=normalized_slowness_difference(velocity,smooth)
        product=torch.cat([spatial_coarse(self.geometry(c)*dm) for c in coarse.split(15)])
        return self.geometry.native_time(coarse),effective_source(velocity,dm,product,source)

    def background_b5(self,wave):
        slab=self.geometry(wave,self.observation.support_depth_rows)
        return slab.flatten(1)@self.observation.matrix.to(slab)

    def propagate(self,prepared):
        background,q=prepared
        return background,self.scatter(*q)

    def pressure(self,fields,slots):
        background,corrected=fields
        return self.geometry(background[slots])+full_pressure(corrected,slots)

    def b5(self,fields):
        background,corrected=fields
        return self.background_b5(background)+b5_pressure(corrected,self.observation)
