"""Shared full-field nonuniform-time/source-free SIPG regularizer.

This is a coarse spatial regularizer, not the fine-grid CPML solver residual.
Time differentiation uses only the400 registered outputs, centers1..398 and
their actual neighboring times. It is a second derivative approximation;
on arbitrary unequal spacings it is not claimed to be second-order accurate.
The endpoints have no PDE center and remain included in full400 DATA loss.

The denominator is the frozen reference's global mean squared five-component
SIPG energy on valid centers/elements/modes, separately for each example. It
is independent of the prediction. Homogeneous zero pressure necessarily has
zero PDE loss; full DATA/receiver supervision is mandatory and a small PDE
loss alone is never a training-success criterion. Reference residual loss
is bounded by25 mathematically, by Cauchy-Schwarz on the five components.
"""
from __future__ import annotations

import hashlib
import math
import torch
from paso.dg_weak import get_dg_operator


class NonuniformSIPG:
    def __init__(self, times, velocity, *, dx=10., dz=10., element_cells=4,
                 order=3, quadrature_order=4, penalty_gamma=2., source_exclude_cells=4,
                 enforce_full400=True, denominator_floor=1e-20):
        if not isinstance(velocity, torch.Tensor) or velocity.ndim != 2:
            raise ValueError('Original physical velocity must be a(H,W)tensor')
        if velocity.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(velocity).all() & (velocity > 0).all()):
            raise ValueError('Positive finite FP32/FP64 physical velocity required')
        times = torch.as_tensor(times).detach().clone().to(device=velocity.device)
        if times.ndim != 1 or times.dtype not in (torch.float32, torch.float64) or len(times) < 3:
            raise ValueError('A floating-point one-dimensional actual time axis is required')
        if enforce_full400 and len(times) != 400:
            raise ValueError('Production PDE must bind the complete400 output time axis')
        if not bool(torch.isfinite(times).all() & (times[1:] > times[:-1]).all()):
            raise ValueError('Time axis must be finite and strictly increasing')
        if not (math.isfinite(penalty_gamma) and penalty_gamma > 0 and math.isfinite(denominator_floor) and denominator_floor > 0):
            raise ValueError('Positive finite penalty and fixed denominator floor required')
        if type(source_exclude_cells) is not int or source_exclude_cells < 1:
            raise ValueError('A nonzero source exclusion is required for this homogeneous PDE')
        self.velocity = velocity.detach().clone()
        self.times = times
        self.time_count = len(times)
        self.center_count = len(times)-2
        self.gamma = float(penalty_gamma)
        self.source_exclude_cells = source_exclude_cells
        self.denominator_floor = float(denominator_floor)
        self.op = get_dg_operator(*velocity.shape, dx, dz, element_cells, order,
                                  quadrature_order, velocity.device, velocity.dtype)
        # Compute spacing from the exact stored floating-point values inFP64,
        # then cast coefficients to the field arithmetic dtype.
        self.spacing = torch.diff(times.to(torch.float64)).to(velocity.dtype)
        self._mask_cache = {}

    def metadata(self):
        axis = self.times.detach().cpu().contiguous().numpy()
        return dict(schema='NONUNIFORM400_SIPG_FIXED_REFERENCE_COMPONENT_ENERGY_V1',
                    time_axis_payload_sha256=hashlib.sha256(axis.tobytes()).hexdigest(),
                    time_dtype=str(axis.dtype), centers=list(range(1,self.time_count-1)),
                    pde_endpoint_policy='no endpoint PDE center; DATA still covers all400',
                    dx=self.op.dx,dz=self.op.dz,element_cells=self.op.ec,order=self.op.order,
                    quadrature_order=self.op.nq,penalty_gamma=self.gamma,
                    source_exclude_cells=self.source_exclude_cells,
                    normalization='per-example reference mean square over five components, all centers, valid elements, modes',
                    denominator_floor=self.denominator_floor,prediction_denominator=False,
                    physical_velocity='unadapted original m/s',cpml_states=False)

    def _centers(self, centers, device=None):
        indices = torch.as_tensor(centers, device=device or self.velocity.device)
        if indices.dtype != torch.long or indices.ndim != 1 or not len(indices):
            raise ValueError('Nonempty int64 center indices required')
        if not bool(((indices >= 1) & (indices < self.time_count-1)).all()) or len(torch.unique(indices)) != len(indices):
            raise ValueError('PDE centers must be unique interior indices')
        return indices

    def center_chunks(self, chunk_frames=4):
        if type(chunk_frames) is not int or chunk_frames < 1:
            raise ValueError('Positive integer chunk_frames required')
        for start in range(1,self.time_count-1,chunk_frames):
            yield torch.arange(start,min(start+chunk_frames,self.time_count-1),
                               dtype=torch.long,device=self.velocity.device)

    def triplets(self, fields, centers):
        if fields.ndim != 4 or fields.shape[1:] != (self.time_count,*self.velocity.shape):
            raise ValueError('Expected complete(B,T,H,W)field at the bound times')
        indices = self._centers(centers,fields.device)
        return fields[:,torch.stack((indices-1,indices,indices+1),dim=-1)]

    def valid_masks(self,sources):
        sources = torch.as_tensor(sources)
        if sources.ndim != 2 or sources.shape[1] not in (2,5) or not bool(torch.isfinite(sources).all()):
            raise ValueError('Finite original source coordinates(B,2)or source5 required')
        masks=[]
        for sx,sz in sources[:,:2].detach().cpu().tolist():
            if not(0 <= sx <= (self.op.X-1)*self.op.dx and 0 <= sz <= (self.op.Z-1)*self.op.dz):
                raise ValueError('Source must lie inside the original physical grid')
            key=(float(sx),float(sz))
            if key not in self._mask_cache:
                mask=self.op.valid_mask(key,0.,0.,self.source_exclude_cells)
                if not bool(mask.any()):raise ValueError('Source exclusion leaves no valid PDE elements')
                self._mask_cache[key]=mask
            masks.append(self._mask_cache[key])
        return torch.stack(masks)

    def components(self,triplets,centers):
        indices=self._centers(centers,triplets.device)
        if triplets.ndim != 5 or triplets.shape[1:] != (len(indices),3,*self.velocity.shape):
            raise ValueError('Triplets must be(B,K,3,H,W), ordered previous/center/next')
        if triplets.device != self.velocity.device or triplets.dtype != self.velocity.dtype:
            raise ValueError('Bound geometry and fields must share device and dtype')
        batch,chunk=triplets.shape[:2]
        flat=triplets.reshape(batch*chunk,3,*self.velocity.shape)
        um,u0,up=[self.op.project(flat[:,j]) for j in range(3)]
        hm=self.spacing[indices-1].repeat(batch)[:,None,None,None]
        hp=self.spacing[indices].repeat(batch)[:,None,None,None]
        utt=2*((up-u0)/hp-(u0-um)/hm)/(hm+hp)
        invc2=self.op._invc2_quad(self.velocity)
        ptt_q=torch.matmul(utt,self.op.Phi.T)
        mass=self.op.Jvol*torch.matmul(ptt_q*invc2*self.op.w2,self.op.Phi)
        stiff=torch.matmul(u0,self.op.A_stiff.T)
        flux=self.op._face_comp(u0,self.op.fx['flux'],'x')+self.op._face_comp(u0,self.op.fz['flux'],'z')
        sym=self.op._face_comp(u0,self.op.fx['sym'],'x')+self.op._face_comp(u0,self.op.fz['sym'],'z')
        pen=(self.gamma*(self.op.order+1)**2/self.op.hx*self.op._face_comp(u0,self.op.fx['pen'],'x')
             +self.gamma*(self.op.order+1)**2/self.op.hz*self.op._face_comp(u0,self.op.fz['pen'],'z'))
        return {name:value.reshape(batch,chunk,self.op.Ez,self.op.Ex,self.op.nb)
                for name,value in dict(mass=mass,stiff=stiff,flux=flux,sym=sym,pen=pen).items()}

    def _sum_valid(self,values,masks):
        return (values*masks[:,None,:,:,None]).sum(dim=(1,2,3,4))

    @torch.no_grad()
    def fixed_scales(self,reference,sources,chunk_frames=4,return_diagnostics=False):
        masks=self.valid_masks(sources)
        if len(masks)!=len(reference):raise ValueError('Source/reference batch mismatch')
        square_sum=reference.new_zeros(len(reference))
        residual_sum=reference.new_zeros(len(reference))
        for centers in self.center_chunks(chunk_frames):
            components=self.components(self.triplets(reference,centers),centers)
            square_sum += sum(self._sum_valid(value.square(),masks) for value in components.values())
            residual_sum += self._sum_valid(sum(components.values()).square(),masks)
        count=masks.sum((1,2))*self.center_count*self.op.nb
        scale=(square_sum/(5*count)).clamp_min(self.denominator_floor).detach()
        if not bool(torch.isfinite(scale).all() & (square_sum > 0).all()):
            raise ValueError('Reference must have positive finite dynamics for normalization')
        if return_diagnostics:
            return dict(scales=scale,reference_pde=residual_sum/(count*scale),valid_elements=masks.sum((1,2)),
                        normalization_depends_on_prediction=False,reference_center_count=self.center_count)
        return scale

    def per_example_chunk(self,triplets,centers,sources,scales):
        masks=self.valid_masks(sources)
        if len(masks)!=len(triplets):raise ValueError('Source/field batch mismatch')
        scales=torch.as_tensor(scales,device=triplets.device,dtype=triplets.dtype)
        if scales.shape!=(len(triplets),) or scales.requires_grad or not bool(torch.isfinite(scales).all() & (scales>0).all()):
            raise ValueError('Frozen positive per-example reference scales required')
        residual=sum(self.components(triplets,centers).values())
        denominator=self.center_count*masks.sum((1,2))*self.op.nb*scales
        return self._sum_valid(residual.square(),masks)/denominator

    def loss_chunk(self,triplets,centers,sources,scales):
        return self.per_example_chunk(triplets,centers,sources,scales).mean()

    def loss_full(self,prediction,sources,scales,chunk_frames=4):
        return sum(self.loss_chunk(self.triplets(prediction,centers),centers,sources,scales)
                   for centers in self.center_chunks(chunk_frames))
