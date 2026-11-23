"""Rectangular forward-only adapter using the existing, unmodified CUDA kernel.

The seven-state staggered8/Taylor4 equations, ghost parity, source derivatives,
three-side CFS-CPML and physical B5 rule are inherited from the existing code.
Only grid/geometry construction and receiver-only streaming are generalized.
"""
from pathlib import Path
import importlib.util
import math
import time
import numpy as np

def receiver_support(shape_zx, spacing, pad, centers_x, strip_z=(10.,15.,20.,25.,30.), offsets=(-10.,-5.,0.,5.,10.)):
    nz, nx = shape_zx
    xmax = (nx-1)*spacing
    one = np.array([1.,4.,6.,4.,1.])/16.
    all_indices, all_weights = [], []
    coordinates = []
    for center in centers_x:
        indices, weights, coords = [], [], []
        for zi, z in enumerate(strip_z):
            if not 0 < z < (nz-1)*spacing:
                raise ValueError('B5 support intersects the zero-pressure node or leaves domain')
            for xi, offset in enumerate(offsets):
                x = center + offset
                x = -x if x < 0 else 2*xmax-x if x > xmax else x
                coords.append([x,z])
                fx, fz = x/spacing, z/spacing
                ix, iz = min(int(math.floor(fx)),nx-2), min(int(math.floor(fz)),nz-2)
                ax, az = fx-ix, fz-iz
                for oz, wz in [(0,1-az),(1,az)]:
                    for ox, wx in [(0,1-ax),(1,ax)]:
                        indices.append((iz+oz)*(nx+2*pad)+ix+ox+pad)
                        weights.append(one[zi]*one[xi]*wz*wx)
        all_indices.append(indices); all_weights.append(weights); coordinates.append(coords)
    return np.array(all_indices,dtype=np.int64), np.array(all_weights,dtype=np.float64), np.array(coordinates,dtype=np.float64)

def profiles(shape_zx, spacing, pad, vmax, frequency=15., power=3, reflection=1e-8):
    nz, nx = shape_zx
    h, w = nz+pad, nx+2*pad
    dmax=(power+1)*vmax*math.log(1/reflection)/(2*pad*spacing)
    xx=(np.arange(w,dtype=np.float64)-pad)*spacing
    zz=np.arange(h,dtype=np.float64)*spacing
    def rates(axis, extent, half, vertical):
        position=axis+(spacing/2 if half else 0.)
        distance=np.maximum(position-extent,0.) if vertical else np.maximum(np.maximum(-position,position-extent),0.)
        ratio=np.clip(distance/(pad*spacing),0.,1.)
        damping=dmax*ratio**power
        alpha=np.where(distance>0,math.pi*frequency*(1-ratio),0.)
        return damping,damping+alpha
    dx0,bx0=rates(xx,(nx-1)*spacing,False,False)
    dz0,bz0=rates(zz,(nz-1)*spacing,False,True)
    dx1,bx1=rates(xx,(nx-1)*spacing,True,False)
    dz1,bz1=rates(zz,(nz-1)*spacing,True,True)
    return np.stack([np.broadcast_to(v,(h,w)) for v in [dx0[None],dz0[:,None],bx0[None],bz0[:,None],dx1[None],dz1[:,None],bx1[None],bz1[:,None]]]).astype(np.float32)

class RectangularLWC:
    def __init__(self, contract, dataset, *, device='cuda:0', dt=None, refine=1, duration=None, use_graph=True, profile_vmax=None, batch_size=1):
        import torch
        self.torch=torch; self.c=contract; self.dataset=Path(dataset)
        self.device=torch.device(device)
        self.batch_size=int(batch_size)
        if not 1<=self.batch_size<=38:raise ValueError('Shot batch must be in1..38 for one model')
        refine=int(contract['solver'].get('spatial_refinement',1))*refine
        self.dx=contract['grid']['spacing_xz_m'][0]/refine
        self.shape=tuple((n-1)*refine+1 for n in contract['grid']['shape_zx'])
        self.pad=round(contract['solver']['physical_absorbing_width_m']/self.dx)
        self.dt=float(dt or contract['solver']['actual_internal_dt_s'])
        self.output_dt=contract['acquisition']['observation_dt_s']
        self.duration=float(duration if duration is not None else contract['acquisition']['record_duration_s'])
        self.stride=round(self.output_dt/self.dt)
        self.steps=round(self.duration/self.dt); self.nt=round(self.duration/self.output_dt)+1
        if not math.isclose(self.stride*self.dt,self.output_dt,abs_tol=1e-12) or self.steps!=(self.nt-1)*self.stride:
            raise ValueError('Simulation and observation time axes must align exactly')
        solver=contract['solver']
        spec=importlib.util.spec_from_file_location('marmousi_frozen_cuda_generator',solver['generator_file'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.generator=module.CUDAGenerator(self.dataset/'build',str(self.device))
        self.device=self.generator.device
        self.sources=np.load(self.dataset/contract['shared_arrays']['source_parameters']['path'])
        centers=np.load(self.dataset/contract['shared_arrays']['receiver_coordinates']['path'])[:,0]
        indices,weights,_=receiver_support(self.shape,self.dx,self.pad,centers)
        self.indices=torch.as_tensor(indices,device=self.device)
        self.weights=torch.as_tensor(weights.astype(np.float32),device=self.device)
        self.nr=len(centers)
        coeff=profiles(self.shape,self.dx,self.pad,profile_vmax or solver['profile_velocity_m_s'],solver['pml_frequency_hz'],solver['pml_power'],solver['pml_reflection_target'])
        self.profiles=torch.as_tensor(coeff,device=self.device)
        h,w=self.shape[0]+self.pad,self.shape[1]+2*self.pad
        self.state=torch.zeros((7,self.batch_size,h,w),device=self.device,dtype=torch.float32)
        self.k=[torch.empty_like(self.state) for _ in range(4)]
        self.v2=torch.empty((self.batch_size,h,w),device=self.device)
        self.injection=torch.zeros_like(self.v2)
        self.forceblock=torch.zeros((self.stride,4,self.batch_size),device=self.device)
        self.output=torch.zeros((self.batch_size,self.nt,self.nr),device=self.device)
        self.graph=None
        if use_graph:
            stream=torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                self.v2.fill_(2000.**2)
                for _ in range(2): self._block()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph): self._block()
            self.state.zero_(); torch.cuda.synchronize(self.device)
        free,total=torch.cuda.mem_get_info(self.device)
        if free < total*.08:
            raise MemoryError('Forward-only worker leaves less than8% CUDA headroom')
        self.memory={'gpu_name':torch.cuda.get_device_name(self.device),'allocated_bytes':torch.cuda.memory_allocated(self.device),'reserved_bytes':torch.cuda.memory_reserved(self.device),'free_bytes':free,'total_bytes':total,'physical_shot_batch':self.batch_size,'model_workers_per_gpu':1,'cuda_graph':bool(self.graph)}

    def _block(self):
        for j in range(self.stride):
            force=self.forceblock[j]
            args=(self.v2,self.profiles,self.injection)
            self.generator._launch(self.generator.forward_function,(self.state,*args,force[0],self.k[0]),tuple(self.v2.shape),self.dx)
            for q in range(1,4):
                self.generator._launch(self.generator.forward_function,(self.k[q-1],*args,force[q],self.k[q]),tuple(self.v2.shape),self.dx)
            self.generator._launch_flat(self.generator.update_function,(self.state,*self.k,self.state),self.state.numel(),self.dt)

    def _sample(self):
        return (self.state[0].reshape(self.batch_size,-1)[:,self.indices]*self.weights).sum(-1)

    def set_velocity(self, velocity):
        t=self.torch
        canonical=t.as_tensor(np.asarray(velocity,dtype=np.float64),device=self.device)
        if tuple(canonical.shape)!=self.shape:
            canonical=t.nn.functional.interpolate(canonical[None,None],size=self.shape,mode='bilinear',align_corners=True)[0,0]
        extended=t.nn.functional.pad(canonical[None,None],(self.pad,self.pad,0,self.pad),mode='replicate')[0]
        self.v2.copy_(extended.square().float().expand(self.batch_size,-1,-1))

    def shot(self, shot_id):
        if self.batch_size!=1:raise ValueError('Use shot_batch for a batched engine')
        value,receipt=self.shot_batch([shot_id])
        return value[0],receipt

    def shot_batch(self, shot_ids):
        t=self.torch
        if len(shot_ids)!=self.batch_size or len(set(shot_ids))!=len(shot_ids):raise ValueError('Batch shot IDs must be unique and match allocated batch size')
        self.state.zero_();self.injection.zero_();self.output.zero_()
        scale=1.554e-8*(5./self.dx)**2
        for b,shot_id in enumerate(shot_ids):
            source=self.sources[shot_id]
            x,z,f,delay,amplitude=source
            ix,iz=int(x/self.dx)+self.pad,int(z/self.dx)
            ax,az=x/self.dx+ self.pad-ix,z/self.dx-iz
            for oz,wz in [(0,1-az),(1,az)]:
                for ox,wx in [(0,1-ax),(1,ax)]:
                    self.injection[b,iz+oz,ix+ox]+=scale*amplitude*wz*wx
        # Use the existing protocol's analytic source derivatives, including primitive(0)=0.
        from src.marmousi_ch6_lwc84_20261024.protocol import ricker_derivatives
        times=t.arange(self.steps,dtype=t.float64)[:,None]*self.dt
        forces=ricker_derivatives(self.sources[shot_ids],times).permute(0,2,1).float().contiguous().to(self.device)
        self.output[:,0].copy_(self._sample())
        t.cuda.reset_peak_memory_stats(self.device)
        t.cuda.synchronize(self.device); started=time.monotonic()
        for frame in range(1,self.nt):
            offset=(frame-1)*self.stride
            self.forceblock.copy_(forces[offset:offset+self.stride])
            if self.graph: self.graph.replay()
            else: self._block()
            self.output[:,frame].copy_(self._sample())
        t.cuda.synchronize(self.device)
        elapsed=time.monotonic()-started
        result=self.output.cpu().numpy().copy()
        peak=t.cuda.max_memory_reserved(self.device)
        free,total=t.cuda.mem_get_info(self.device)
        actual_memory={**self.memory,'peak_reserved_bytes':peak,'peak_allocated_bytes':t.cuda.max_memory_allocated(self.device),'free_bytes_after_trajectory':free,'reserved_fraction':peak/total}
        if peak>total*.92 or free<total*.08:raise MemoryError('Measured full trajectory leaves less than8% GPU headroom')
        if not np.isfinite(result).all() or not np.any(result):
            raise FloatingPointError('Nonfinite or entirely zero shot')
        return result, {'solver_wall_seconds':elapsed,'memory':actual_memory,'physical_shot_batch':self.batch_size,'shot_ids':list(shot_ids),'dt_s':self.dt,'steps':self.steps,'shape_zx':list(self.shape),'source_scale':scale,'output_alignment':'u(0), then u after exactly stride updates; endpoint inclusive'}
