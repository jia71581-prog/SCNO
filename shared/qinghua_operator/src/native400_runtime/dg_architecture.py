"""Element-local modal features and SIPG face exchange inside the decoder.

The output head and ridge modes both consume these features. This module is
an architectural inductive bias; it does not assert accuracy or time stability.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from paso.dg_weak import DGOperator, _legendre_basis
from bootstrap24 import ResidualRefiner


class DGFeatureBlock(nn.Module):
    def __init__(self,channels=16,size=201,element_cells=4,order=3):
        super().__init__()
        self.channels=channels;self.size=size;self.ec=element_cells;self.order=order
        self.elements=(size-1)//element_cells;self.nb=(order+1)**2
        self.enabled=True;self.face_enabled=True
        op=DGOperator(size,size,10.,10.,element_cells,order,order+1,'cpu',torch.float64)
        vn,_=_legendre_basis(np.linspace(-1,1,element_cells+1),order+1)
        projection=np.linalg.pinv(vn)
        p=np.zeros((self.elements*(order+1),size));r=np.zeros((size,self.elements*(order+1)))
        count=np.zeros(size)
        for e in range(self.elements):
            a=e*element_cells;b=e*(order+1)
            p[b:b+order+1,a:a+element_cells+1]=projection
            r[a:a+element_cells+1,b:b+order+1]=vn
            count[a:a+element_cells+1]+=1
        r/=count[:,None]
        for name,value in (('projection',p),('reconstruction',r),('stiffness',op.A_stiff),
                           ('velocity_quadrature',op.Qv),('quadrature_phi',op.Phi),('quadrature_weights',op.w2)):
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32).contiguous())
        for orient,mats in (('x',op.fx),('z',op.fz)):
            for part in ('flux','sym','pen'):
                for side in ('mm','mp','pm','pp'):
                    self.register_buffer(f'{orient}_{part}_{side}',mats[part][side].float())
        self.Jvol=op.Jvol
        self.penalty=2*(order+1)**2/op.hx
        # h^2 M^{-1} A is the natural dimensionless cell operator. The previous
        # extra 16*(p+1)^4 division reduced actual hidden-feature changes to
        # 2e-5..4e-5 and was rejected by the paired accuracy/face-ablation gate.
        self.operator_scale=op.hx**2
        for orient,mats in (('x',op.fx),('z',op.fz)):
            for side in ('mm','mp','pm','pp'):
                combined=mats['flux'][side]+mats['sym'][side]+self.penalty*mats['pen'][side]
                self.register_buffer(f'{orient}_combined_{side}',combined.float(),persistent=False)
        self.modal_mix=nn.Parameter(torch.eye(self.nb))
        self.volume_mix=nn.Parameter(torch.eye(channels));self.face_mix=nn.Parameter(torch.eye(channels))
        # Zero gates preserve every inherited prediction at initialization.
        # Gates get gradients immediately; all DG matrices remain in the graph.
        self.volume_gate=nn.Parameter(torch.zeros(channels));self.face_gate=nn.Parameter(torch.zeros(channels))
        self._mass_cache={}

    def project(self,field):
        p=self.projection.to(field.dtype)
        u=torch.matmul(p,torch.matmul(field,p.T))
        n,c=field.shape[:2];e=self.elements;b=self.order+1
        return u.reshape(n,c,e,b,e,b).permute(0,1,2,4,3,5).reshape(n,c,e,e,b*b)

    def reconstruct(self,u):
        n,c=u.shape[:2];e=self.elements;b=self.order+1
        value=u.reshape(n,c,e,e,b,b).permute(0,1,2,4,3,5).reshape(n,c,e*b,e*b)
        r=self.reconstruction.to(u.dtype)
        return torch.matmul(r,torch.matmul(value,r.T))

    @torch.no_grad()
    def medium_mass(self,velocity,key=None):
        # Squaring an invalid negative velocity would otherwise produce a
        # positive mass. Validate the physical input before consulting cache.
        if not torch.isfinite(velocity).all() or not (velocity>0).all():
            raise ValueError('DG velocity must be finite and strictly positive')
        if key is not None and key in self._mass_cache:return self._mass_cache[key]
        v=velocity[0,0].unfold(0,self.ec+1,self.ec).unfold(1,self.ec+1,self.ec)
        v=v.reshape(self.elements,self.elements,-1)
        q=torch.matmul(v,self.velocity_quadrature.T)
        inv=(3000./q).square()*self.quadrature_weights
        mass=self.Jvol*torch.matmul(inv,self.quadrature_phi.square())
        if not torch.isfinite(mass).all() or not (mass>0).all():raise ValueError('positive DG modal mass')
        if key is not None:self._mass_cache[key]=mass
        return mass

    def _face(self,u,orient,part):
        import torch.nn.functional as F
        mats={s:getattr(self,f'{orient}_{part}_{s}').to(u.dtype) for s in ('mm','mp','pm','pp')}
        if orient=='x':
            left,right=u[:,:,:,:-1],u[:,:,:,1:]
            minus=left@mats['mm'].T+right@mats['mp'].T
            plus=left@mats['pm'].T+right@mats['pp'].T
            return F.pad(minus,(0,0,0,1))+F.pad(plus,(0,0,1,0))
        top,bottom=u[:,:,:-1],u[:,:,1:]
        minus=top@mats['mm'].T+bottom@mats['mp'].T
        plus=top@mats['pm'].T+bottom@mats['pp'].T
        return F.pad(minus,(0,0,0,0,0,1))+F.pad(plus,(0,0,0,0,1,0))

    def modal_components(self,u):
        volume=u@self.stiffness.to(u.dtype).T
        # Fixed consistency, symmetry and penalty matrices are fused once.
        face=self._face(u,'x','combined')+self._face(u,'z','combined')
        return volume,face

    def forward(self,field,mass):
        if not self.enabled:return field
        u=self.project(field)@self.modal_mix.to(field.dtype).T
        volume,face=self.modal_components(u)
        volume=volume/mass[None,None]*self.operator_scale
        face=face/mass[None,None]*self.operator_scale
        volume=torch.einsum('dc,ncexm->ndexm',self.volume_mix.to(field.dtype),volume)
        face=torch.einsum('dc,ncexm->ndexm',self.face_mix.to(field.dtype),face)
        update=volume*self.volume_gate.tanh()[None,:,None,None,None]
        if self.face_enabled:update=update+face*self.face_gate.tanh()[None,:,None,None,None]
        return field+self.reconstruct(update)


class DGResidualRefiner(ResidualRefiner):
    def __init__(self):
        super().__init__();self.dg=DGFeatureBlock()

    def prepare(self,known):
        velocity=(known[:,0:1]+1)*3000.
        # The caller's known input contains no future labels. Cache only the
        # fixed medium term; trainable convolution features are always rebuilt.
        key=getattr(self,'medium_cache_key',None)
        if key is None:
            import hashlib
            key=hashlib.sha256(velocity.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return dict(conv=super().prepare(known),mass=self.dg.medium_mass(velocity,key))

    def forward(self,features,base,times,source):
        def transform(_module,args):return (self.dg(args[0],features['mass']),)
        # Prepend so MetaAcoustic's mode capture sees the transformed features.
        hook=self.output.register_forward_pre_hook(transform,prepend=True)
        try:return super().forward(features['conv'],base,times,source)
        finally:hook.remove()


def make_dg_meta(device,*,enabled=True):
    from r25_model import MetaAcoustic
    meta=MetaAcoustic(device);meta.refiner=DGResidualRefiner().to(device)
    meta.refiner.dg.enabled=enabled
    return meta
