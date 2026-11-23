"""Declared seven-state LWC8-4 source, CPML, velocity and B5 contract.

Helpers are differentiable torch operations; this module does not propagate,
create observations, choose a CUDA device, or certify numerical accuracy.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F

NATIVE_N=201
PHYSICAL_N=401
DX=5.
PAD=40
DT=.000125
OUTPUT_DT=.00125
DURATION=4.
UPDATES=32000
RECEIVER_SAMPLES=3201
SOURCE_SCALE=1.554e-8  # minus the historical Deepwave injection coefficient
PROFILE_ORDER=('dx0','dz0','bx0','bz0','dx1','dz1','bx1','bz1')
FORCE_ORDER=('primitive_zero_at_0','ricker','ricker_dt','ricker_dtt')
STATE_ORDER=('p','ux','vz','psi_x','psi_z','phi_x','phi_z')
STAGGERED_WEIGHTS=(1225/1024,-245/3072,49/5120,-5/7168)


def _grid(n,dx,pad):
    if isinstance(n,bool) or int(n)!=n or n<2:raise ValueError('n must be integer >=2')
    if isinstance(pad,bool) or int(pad)!=pad or pad<1:raise ValueError('pad must be positive integer')
    if not math.isfinite(float(dx)) or dx<=0:raise ValueError('dx must be positive finite')
    return int(n),float(dx),int(pad)


def _sources(sources):
    s=torch.as_tensor(sources)
    if not s.is_floating_point():s=s.to(torch.float64)
    else:s=s.to(dtype=torch.float64)
    if s.ndim!=2 or s.shape[1]!=5 or not bool(torch.isfinite(s).all()):
        raise ValueError('sources must be finite[B,5]: x,z,frequency,t0,amplitude')
    if not bool((s[:,2]>0).all()):raise ValueError('Source frequencies must be positive')
    return s


def make_profiles(n=PHYSICAL_N,dx=DX,pad=PAD,*,vmax=4450.,pml_freq=15.,power=3,reflection=1e-8,device=None,dtype=torch.float64):
    """Fixed physical rates [8,n+pad,n+2pad]; suffix1 is positive half-cell.

The top has no PML. Reference vmax, frequency and reflection target are fixed
protocol constants, not estimated from the current differentiable velocity.
"""
    n,dx,pad=_grid(n,dx,pad)
    if not dtype.is_floating_point:raise ValueError('Real floating dtype required')
    if not all(math.isfinite(float(v)) and v>0 for v in (vmax,pml_freq,power)):
        raise ValueError('Positive finite vmax, pml_freq and power required')
    if not 0<reflection<1:raise ValueError('reflection must be between0 and1')
    h,w=n+pad,n+2*pad;extent=(n-1)*dx;width=pad*dx
    dmax=(power+1)*vmax*math.log(1/reflection)/(2*width)
    z=torch.arange(h,device=device,dtype=dtype)*dx
    x=(torch.arange(w,device=device,dtype=dtype)-pad)*dx
    def axis_profiles(axis,half,vertical):
        position=axis+(dx/2 if half else 0.)
        distance=(position-extent).clamp_min(0.) if vertical else torch.maximum(-position,position-extent).clamp_min(0.)
        ratio=(distance/width).clamp(0.,1.)
        damping=dmax*ratio.pow(power)
        alpha=torch.where(distance>0,math.pi*pml_freq*(1-ratio),torch.zeros_like(ratio))
        return damping,damping+alpha
    dx0,bx0=axis_profiles(x,False,False);dz0,bz0=axis_profiles(z,False,True)
    dx1,bx1=axis_profiles(x,True,False);dz1,bz1=axis_profiles(z,True,True)
    return torch.stack((dx0[None].expand(h,w),dz0[:,None].expand(h,w),
        bx0[None].expand(h,w),bz0[:,None].expand(h,w),
        dx1[None].expand(h,w),dz1[:,None].expand(h,w),
        bx1[None].expand(h,w),bz1[:,None].expand(h,w))).contiguous()


def make_source_grid(sources,*,n=PHYSICAL_N,dx=DX,pad=PAD,source_scale=SOURCE_SCALE):
    """Return [B,H,W] discrete q injection, amplitude applied exactly once.

There is intentionally no dx^-2 factor. The seven-state backend must multiply
this injection by v^2 in pressure RHS so that the interior equation is
v^-2 p_tt - Laplacian(p) = injection * Ricker(t).
"""
    n,dx,pad=_grid(n,dx,pad);s=_sources(sources)
    if not math.isfinite(float(source_scale)):raise ValueError('Finite source_scale required')
    extent=(n-1)*dx
    if bool(((s[:,:2]<0)|(s[:,:2]>extent)).any()):raise ValueError('Source outside physical domain')
    h,w=n+pad,n+2*pad
    xf=s[:,0]/dx+pad;zf=s[:,1]/dx
    ix=torch.floor(xf).long();iz=torch.floor(zf).long();ax=xf-ix;az=zf-iz
    indices=[];weights=[]
    for oz,wz in ((0,1-az),(1,az)):
        for ox,wx in ((0,1-ax),(1,ax)):
            indices.append((iz+oz)*w+ix+ox)
            weights.append(wz*wx*s[:,4]*source_scale)
    result=s.new_zeros((len(s),h*w)).scatter_add(1,torch.stack(indices,1),torch.stack(weights,1))
    return result.reshape(len(s),h,w)


def ricker_derivatives(sources,t):
    """Return [...,B,4] analytic force derivatives with amplitude excluded.

A scalar t returns[B,4]; t[T,1] broadcasts all shots and returns[T,B,4].
Columns are time-integrated Ricker with primitive(0)=0, Ricker, Ricker',
and Ricker''. Frequency/t0 remain differentiable. Backend uses forces[4,B].
"""
    s=_sources(sources);time=torch.as_tensor(t,device=s.device,dtype=s.dtype)
    if not bool(torch.isfinite(time).all()):raise ValueError('Finite time required')
    k=math.pi*s[:,2];delay=time-s[:,3];a=k*delay;a2=a.square();envelope=torch.exp(-a2)
    primitive=delay*envelope+s[:,3]*torch.exp(-(k*s[:,3]).square())
    ricker=(1-2*a2)*envelope
    first=k*a*(-6+4*a2)*envelope
    second=k.square()*(-6+24*a2-8*a2.square())*envelope
    return torch.stack((primitive,ricker,first,second),dim=-1)


def extend_velocity(v_native,*,n=PHYSICAL_N,pad=PAD):
    """Bilinear align_corners native->fine, then replicate three PML sides.

Input[Hn,Wn] returns[H,W]; input[B,Hn,Wn] returns[B,H,W]. Input gradients
flow through interpolation and every repeated PML value, including v^2 later.
"""
    n,_,pad=_grid(n,1.,pad);v=torch.as_tensor(v_native)
    if not v.is_floating_point():v=v.to(torch.float64)
    else:v=v.to(dtype=torch.float64)
    if v.ndim not in (2,3) or v.shape[-2]!=v.shape[-1] or v.shape[-1]<2:
        raise ValueError('Expected square native velocity[Hn,Wn] or[B,Hn,Wn]')
    if not bool(torch.isfinite(v).all()) or not bool((v>0).all()):raise ValueError('Finite positive velocity required')
    single=v.ndim==2
    if single:v=v[None]
    fine=F.interpolate(v[:,None],size=(n,n),mode='bilinear',align_corners=True)
    extended=F.pad(fine,(pad,pad,0,pad),mode='replicate')[:,0]
    return extended[0] if single else extended


def receiver_indices_weights(*,n=PHYSICAL_N,pad=PAD,stride=2,device=None,dtype=torch.float64):
    """Flattened B5 gather indices[R,25] and weights[R,25].

Physical z rows2..6; receiver x=0,stride,...; horizontal reflect is applied
at physical x0/n-1 before adding PML padding. No CPML cell enters B5.
"""
    n,_,pad=_grid(n,1.,pad)
    if n<7 or isinstance(stride,bool) or int(stride)!=stride or stride<1:
        raise ValueError('B5 requires n>=7 and positive integer stride')
    if not dtype.is_floating_point:raise ValueError('Real floating receiver dtype required')
    centers=torch.arange(0,n,int(stride),device=device,dtype=torch.long)
    horizontal=centers[:,None]+torch.arange(-2,3,device=device,dtype=torch.long)[None]
    horizontal=torch.where(horizontal<0,-horizontal,horizontal)
    horizontal=torch.where(horizontal>=n,2*(n-1)-horizontal,horizontal)
    rows=torch.arange(2,7,device=device,dtype=torch.long)
    indices=(rows[None,:,None]*(n+2*pad)+horizontal[:,None,:]+pad).reshape(len(centers),25)
    one=torch.tensor([1.,4.,6.,4.,1.],device=device,dtype=dtype)/16
    weights=(one[:,None]*one[None,:]).reshape(1,25).expand(len(centers),25).contiguous()
    return indices.contiguous(),weights


def receiver_forward(p,indices=None,weights=None,*,n=PHYSICAL_N,pad=PAD,stride=2):
    """B5 pressure[...,H,W] ->[...,R], differentiable linear restriction."""
    p=torch.as_tensor(p);n,_,pad=_grid(n,1.,pad)
    if p.ndim<2 or p.shape[-2:]!=(n+pad,n+2*pad):raise ValueError('Receiver pressure grid shape mismatch')
    if indices is None and weights is None:
        indices,weights=receiver_indices_weights(n=n,pad=pad,stride=stride,device=p.device,dtype=p.dtype)
    elif indices is None or weights is None:raise ValueError('Pass both indices and weights')
    indices=indices.to(device=p.device,dtype=torch.long);weights=weights.to(device=p.device,dtype=p.dtype)
    flat=p.reshape(*p.shape[:-2],-1)
    return (flat[...,indices]*weights).sum(-1)


def receiver_transpose(y,indices=None,weights=None,*,n=PHYSICAL_N,pad=PAD,stride=2):
    """Exact transpose of B5: [...,R]->[...,H,W], including reflected repeats."""
    y=torch.as_tensor(y);n,_,pad=_grid(n,1.,pad);h,w=n+pad,n+2*pad
    if y.ndim<1:raise ValueError('Receiver cotangent needs last receiver axis')
    if indices is None and weights is None:
        indices,weights=receiver_indices_weights(n=n,pad=pad,stride=stride,device=y.device,dtype=y.dtype)
    elif indices is None or weights is None:raise ValueError('Pass both indices and weights')
    indices=indices.to(device=y.device,dtype=torch.long);weights=weights.to(device=y.device,dtype=y.dtype)
    if y.shape[-1]!=indices.shape[0]:raise ValueError('Receiver cotangent size mismatch')
    lead=y.shape[:-1];flat=y.reshape(-1,y.shape[-1]);values=(flat[:,:,None]*weights).reshape(len(flat),-1)
    target=y.new_zeros((len(flat),h*w)).scatter_add(1,indices.reshape(1,-1).expand(len(flat),-1),values)
    return target.reshape(*lead,h,w)


def contract():
    return dict(schema='CHAPTER6_LWC84_PROTOCOL_V1',spatial_order=8,time_order=4,
        temporal_integrator='fourth-order Taylor of all seven physical states including CPML and source derivatives',
        state_order=list(STATE_ORDER),profile_order=list(PROFILE_ORDER),force_order=list(FORCE_ORDER),
        physical_nodes=[PHYSICAL_N,PHYSICAL_N],native_velocity_nodes=[NATIVE_N,NATIVE_N],
        spacing_m=DX,domain_m=[2000.,2000.],pml_width_nodes=[0,PAD,PAD,PAD],
        pml_power=3,pml_reflection_target=1e-8,pml_kappa=1.,pml_frequency_hz=15.,fixed_profile_velocity_mps=4450.,
        top_boundary='strict nodal p(z=0)=0; appropriate odd/even staggered ghost parity required in backend',
        candidate_dt_s=DT,update_count=UPDATES,record_duration_s=DURATION,
        receiver_dt_s=OUTPUT_DT,receiver_samples=RECEIVER_SAMPLES,receiver_endpoint_inclusive=True,
        receiver='five vertical physical rows2..6 times horizontal reflect binomial5, x stride2',
        source_scale=SOURCE_SCALE,source_scale_applications=1,source_amplitude_location='injection grid only',
        source_grid_has_dx_inverse_square=False,source_pressure_rhs='v^2 * injection * primitive_Ricker(t)',
        source_pde='v^-2 p_tt - Laplacian(p) = injection * Ricker(t)',
        source_forcing_origin='negative historical Deepwave discrete injection coefficient; actual consistency must be validated',
        velocity_mapping='bilinear align_corners=True, then replicate sides/bottom; all dependencies differentiable',
        arithmetic='float64',accuracy_certified=False,
        required_validation='full-system timestep convergence, stability, source/receiver protocol and velocity-gradient directional tests',
        historical_observation_compatibility='NOT identical: old Deepwave top inner row was not forced zero and time order was2; generate and bind new consistent LWC synthetic observations',
        data_generation_policy='local only; no GPU exclusivity')
