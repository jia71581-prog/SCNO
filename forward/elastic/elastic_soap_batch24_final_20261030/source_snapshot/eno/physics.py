"""Devito elastic SGFD2; traction-free top; three sponge absorbing strips.

Vx is at (z,x+h/2), Vz at (z+h/2,x); normal stress is nodal and shear
stress at (z+h/2,x+h/2). SI units, displacement output collocated at nodes.
"""
import json
from functools import lru_cache
import numpy as np
from scipy.ndimage import zoom

class DevitoElastic:
    def __init__(self,cfg,device='cpu'):
        from devito import Grid,SpaceDimension,Function,TimeFunction,SparseTimeFunction,Eq,Operator
        self.cfg=cfg;self.device=device;n=cfg['grid']['n'];r=cfg['fd']['refinement']
        self.nf=(n-1)*r+1;self.pad=cfg['fd']['pml_cells'];self.top=1
        self.dx=cfg['grid']['dx_m']/r
        self.substeps=cfg['fd'].get('substeps_per_frame',16)
        self.dt=cfg['grid']['duration_s']/cfg['grid']['frames']/self.substeps
        self.steps=(cfg['grid']['frames']-1)*self.substeps
        if cfg['fd'].get('accuracy',2)!=2:raise ValueError('validated Devito scheme is SGFD2')
        if self.dt*cfg['fd']['max_vel_m_s']/self.dx>=.65:raise ValueError('CFL: increase substeps_per_frame')
        nz=self.nf+self.pad+self.top;nx=self.nf+2*self.pad
        z,x=SpaceDimension('z'),SpaceDimension('x')
        grid=Grid(shape=(nz,nx),extent=((nz-1)*self.dx,(nx-1)*self.dx),dimensions=(z,x))
        self.grid=grid;t=grid.stepping_dim
        self.material={k:Function(name=k,grid=grid,space_order=2) for k in ['lam','mu','muxz','bx','bz','damp']}
        self.fields={k:TimeFunction(name=k,grid=grid,time_order=1,space_order=2) for k in ['vx','vz','sxx','szz','sxz','ux','uz']}
        la,mu,mh,bx,bz,d=[self.material[k] for k in ['lam','mu','muxz','bx','bz','damp']]
        vx,vz,xx,zz,xz,ux,uz=[self.fields[k] for k in ['vx','vz','sxx','szz','sxz','ux','uz']]
        h=self.dx;dt=self.dt;g=self.top
        dvx=(vx[t+1,z,x]-vx[t+1,z,x-1])/h
        dvz=(vz[t+1,z,x]-vz[t+1,z-1,x])/h
        equations=[
            Eq(vx[t+1,z,x],d*(vx[t,z,x]+dt*bx*((xx[t,z,x+1]-xx[t,z,x])/h+(xz[t,z,x]-xz[t,z-1,x])/h))),
            Eq(vz[t+1,z,x],d*(vz[t,z,x]+dt*bz*((xz[t,z,x]-xz[t,z,x-1])/h+(zz[t,z+1,x]-zz[t,z,x])/h))),
            Eq(vz[t+1,g-1,x],vz[t+1,g,x]+la[g,x]/(la[g,x]+2*mu[g,x])*(vx[t+1,g,x]-vx[t+1,g,x-1])),
            Eq(xx[t+1,z,x],d*(xx[t,z,x]+dt*((la+2*mu)*dvx+la*dvz))),
            Eq(zz[t+1,z,x],d*(zz[t,z,x]+dt*(la*dvx+(la+2*mu)*dvz))),
            Eq(xz[t+1,z,x],d*(xz[t,z,x]+dt*mh*((vx[t+1,z+1,x]-vx[t+1,z,x])/h+(vz[t+1,z,x+1]-vz[t+1,z,x])/h))),
        ]
        self.src=SparseTimeFunction(name='src',grid=grid,nt=self.steps+1,npoint=1)
        equations+=self.src.inject(field=xx.forward,expr=self.src*dt/h**2)
        equations+=self.src.inject(field=zz.forward,expr=self.src*dt/h**2)
        equations += [Eq(zz[t+1,g,x],0),Eq(zz[t+1,g-1,x],-zz[t+1,g+1,x]),Eq(xz[t+1,g-1,x],-xz[t+1,g,x]),
                      Eq(ux[t+1,z,x],ux[t,z,x]+dt*(vx[t+1,z,x]+vx[t+1,z,x-1])/2),
                      Eq(uz[t+1,z,x],uz[t,z,x]+dt*(vz[t+1,z,x]+vz[t+1,z-1,x])/2)]
        self.recordings={}
        if device=='cuda':
            from devito import ConditionalDimension,Dimension
            from devito.symbolics import IntDiv
            from sympy import Mod,Eq as SymEq
            # Match CPU frame k exactly after k*substeps updates. Keep both
            # rolling wavefields and coarse output frames resident for a shot.
            ts=ConditionalDimension('eno_save_time',parent=grid.time_dim,
                                    condition=SymEq(Mod(grid.time_dim+1,self.substeps),0))
            qz,qx=Dimension('eno_qz'),Dimension('eno_qx')
            ti=IntDiv(ts+1,self.substeps)
            for key,field in [('ux',ux),('uz',uz)]:
                rec=Function(name='eno_record_'+key,shape=(cfg['grid']['frames'],n,n),
                             dimensions=(ts,qz,qx),space_order=0)
                self.recordings[key]=rec
                equations.append(Eq(rec[ti,qz,qx],field[t+1,g+r*qz,self.pad+r*qx]))
            for key,expr in [('normal',zz[t+1,g,self.pad+r*qx]),
                             ('shear',xz[t+1,g-1,self.pad+r*qx]+xz[t+1,g,self.pad+r*qx])]:
                rec=Function(name='eno_record_'+key,shape=(cfg['grid']['frames'],n),
                             dimensions=(ts,qx),space_order=0)
                self.recordings[key]=rec
                equations.append(Eq(rec[ti,qx],expr))
        options=dict(opt='advanced',language='openmp',name='ENOElasticSGFD2')
        if device=='cuda':options.update(platform='nvidiaX',compiler='clang',name='ENOElasticSGFD2GPU')
        self.op=Operator(equations,**options)
        if device=='cuda':
            from .devito_gpu import configure_compiler
            configure_compiler(self.op)
        z0,x0=np.indices((nz,nx))
        depth=np.maximum.reduce([np.maximum(self.pad-x0,0),np.maximum(x0-(self.pad+self.nf-1),0),np.maximum(z0-(g+self.nf-1),0)])/self.pad
        sigma=3*cfg['fd']['max_vel_m_s']*np.log(1000)/(2*self.pad*h)
        self.material['damp'].data[:]=np.exp(-sigma*depth**2*dt)
        self.last_diagnostics={}

    def prepare(self,velocity,source_scale=1.):
        c=self.cfg;r=c['fd']['refinement'];n=c['grid']['n']
        v=np.asarray(velocity,dtype=np.float64)
        if v.shape!=(2,n,n):raise ValueError('velocity shape must be (2,n,n)')
        fine=np.stack([zoom(vv,self.nf/n,order=1) for vv in v])
        vs=fine[1];p=.9409+2.0947*vs-.8206*vs**2+.2683*vs**3-.0251*vs**4
        rho=1000*(1.6612*p-.4721*p**2+.0671*p**3-.0043*p**4+.000106*p**5)
        if not (np.isfinite(fine).all() and np.all(fine[0]>1.17*vs) and np.all(rho>0)):raise ValueError('unphysical elastic model')
        vp,vs=fine*1000
        pad=((self.top,self.pad),(self.pad,self.pad))
        rho=np.pad(rho,pad,mode='edge');mu=np.pad(vs**2,pad,mode='edge')*rho
        la=np.pad(vp**2-2*vs**2,pad,mode='edge')*rho
        rollx=lambda a:np.concatenate([a[:,1:],a[:,-1:]],axis=1)
        rollz=lambda a:np.concatenate([a[1:],a[-1:]],axis=0)
        values=dict(lam=la,mu=mu,muxz=4/(1/mu+1/rollx(mu)+1/rollz(mu)+1/rollz(rollx(mu))),bx=2/(rho+rollx(rho)),bz=2/(rho+rollz(rho)))
        for k,a in values.items():self.material[k].data[:]=a.astype('float32')
        time=np.arange(self.steps+1)*self.dt
        arg=(np.pi*c['source']['frequency_hz']*(time-c['source']['peak_time_s']))**2
        self.src.data[:,0]=((1-2*arg)*np.exp(-arg)*c['source']['moment']*source_scale).astype('float32')

    def run(self,velocity,locations,source_scale=1.,threads=None,direct_kernel=False):
        c=self.cfg;r=c['fd']['refinement'];n=c['grid']['n']
        self.prepare(velocity,source_scale)
        result=np.zeros((len(locations),n,n,c['grid']['frames'],2),dtype='float32')
        zs=slice(self.top,self.top+self.nf,r);xs=slice(self.pad,self.pad+self.nf,r)
        normal_max=0.;shear_max=0.
        for shot,loc in enumerate(locations):
            loc=np.asarray(loc)
            if np.any(loc<1) or np.any(loc>n-2):raise ValueError('source outside interior')
            for f in self.fields.values():f.data_with_halo[:]=0
            self.src.coordinates.data[0]=[(self.top+loc[0]*r)*self.dx,(self.pad+loc[1]*r)*self.dx]
            if self.device=='cuda':
                for f in self.recordings.values():f.data_with_halo[:]=0
                self.op.apply(time_m=0,time_M=self.steps-1)
                for component,key in enumerate(['ux','uz']):
                    result[shot,:,:,:,component]=np.asarray(self.recordings[key].data).transpose(1,2,0)
                normal_max=max(normal_max,float(np.max(np.abs(self.recordings['normal'].data))))
                shear_max=max(shear_max,float(np.max(np.abs(self.recordings['shear'].data))))
                continue
            if direct_kernel:
                # Validate the complete CPU interval once per shot, then reuse the
                # same pointers and compiled stencil across output-frame chunks.
                # Devito apply() does this preparation again on every invocation.
                kernel=self.op.cfunction
                effective_threads=threads or c['fd'].get('threads',4)
                arguments=self.op.arguments(time_m=0,time_M=self.steps-1,nthreads=effective_threads,
                                            nthreads_nonaffine=effective_threads)
                if arguments.comm is not None and arguments.comm.size!=1:
                    raise ValueError('direct kernel path is only validated for local non-MPI execution')
                names=[p.name for p in self.op.parameters]
                values=[arguments[name] for name in names]
                time_min=names.index('time_m');time_max=names.index('time_M')
            for frame in range(1,c['grid']['frames']):
                start=(frame-1)*self.substeps;stop=frame*self.substeps-1
                if direct_kernel:
                    values[time_min]=start;values[time_max]=stop
                    self.op._postprocess_errors(kernel(*values))
                else:
                    effective_threads=threads or c['fd'].get('threads',4)
                    kwargs={} if self.device=='cuda' else dict(nthreads=effective_threads,nthreads_nonaffine=effective_threads)
                    self.op.apply(time_m=start,time_M=stop,**kwargs)
                idx=(stop+1)%2
                result[shot,:,:,frame,0]=np.asarray(self.fields['ux'].data)[idx,zs,xs]
                result[shot,:,:,frame,1]=np.asarray(self.fields['uz'].data)[idx,zs,xs]
                zz=np.asarray(self.fields['szz'].data)[idx];xz=np.asarray(self.fields['sxz'].data)[idx]
                normal_max=max(normal_max,float(np.max(np.abs(zz[self.top,xs]))))
                shear_max=max(shear_max,float(np.max(np.abs(xz[self.top-1,xs]+xz[self.top,xs]))))
            if direct_kernel:self.op._postprocess_arguments(arguments)
        if not np.isfinite(result).all():raise FloatingPointError('non-finite Devito displacement')
        self.last_diagnostics=dict(top_normal_stress_max=normal_max,top_shear_antisymmetry_error=shear_max,dt_s=self.dt,dx_m=self.dx,backend='devito',space_order=2,device=self.device)
        return result

@lru_cache(maxsize=3)
def _cached(cfg_json):
    from devito import configuration
    configuration['log-level']='WARNING'
    return DevitoElastic(json.loads(cfg_json))

def simulate(velocity,locations,cfg,device='cpu',source_scale=1.,threads=None,direct_kernel=False):
    if device not in ['cpu','cuda','cuda_openmp']:raise ValueError('unknown Devito execution device')
    if device!='cpu' and direct_kernel:raise ValueError('direct CPU kernel arguments cannot be used for GPU execution')
    if device=='cuda':
        locations=np.asarray(locations)
        return simulate_batch(np.repeat(np.asarray(velocity)[None],len(locations),axis=0),locations,cfg,source_scale)
    cached=_cached_gpu if device=='cuda_openmp' else _cached
    return cached(json.dumps(cfg,sort_keys=True)).run(velocity,locations,source_scale,threads,direct_kernel)

@lru_cache(maxsize=2)
def _cached_gpu(cfg_json):
    from devito import configuration
    configuration['log-level']='WARNING'
    return DevitoElastic(json.loads(cfg_json),'cuda')

def diagnostics(cfg,device='cpu'):
    if device=='cuda':return _GPU_DIAGNOSTICS.get(json.dumps(cfg,sort_keys=True),{}).copy()
    cached=_cached_gpu if device=='cuda_openmp' else _cached
    return cached(json.dumps(cfg,sort_keys=True)).last_diagnostics.copy()

_GPU_DIAGNOSTICS={}

@lru_cache(maxsize=2)
def _cached_cuda_graph(cfg_json,batch_size):
    from .physics_cuda import DevitoCUDAGraph
    return DevitoCUDAGraph(json.loads(cfg_json),batch_size)

def simulate_batch(velocities,locations,cfg,source_scale=1.):
    key=json.dumps(cfg,sort_keys=True)
    backend=_cached_cuda_graph(key,len(velocities))
    result=backend.run(velocities,locations,source_scale)
    _GPU_DIAGNOSTICS[key]=backend.last_diagnostics.copy()
    return result

def release_gpu_buffers():
    """Release generation buffers before the same process enters UNO training."""
    _cached_cuda_graph.cache_clear()
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
