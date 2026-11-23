"""Ordinary FP32 CUDA precision variant; seven-state generator and analytic VJP.

No numerical finite-difference gradients, no detached velocity source term.
Kernel outputs use PyTorch's current CUDA stream. Python step/VJP helpers are
manual no-grad arithmetic; callers wrap these in a trajectory autograd Function.
This is not a certified high-precision reference until full-record comparison
against the frozen FP64 trajectory passes. No Kahan compensation is used.
"""
from __future__ import annotations
from scno_release_paths import local_asset as _scno_local_asset
import ctypes
import hashlib
from pathlib import Path
import importlib.util
import sys
import torch

FP64_PARENT_FILE_SHA256="dea09e82beafc13011e74474b9c7921f563011ed70d8ef811891c53e52d903fb"

CUDA_SOURCE=r'''
__device__ float get(const float* a,int b,int z,int x,int h,int w,bool half=false){
 float sign=1.f;if(z<0){if(half)z=-z-1;else{z=-z;sign=-1.f;}}
 if(z<0||z>=h||x<0||x>=w)return 0.f;
 return sign*a[((long long)b*h+z)*w+x];
}
__device__ float masked(const float* a,int c,long long i,long long n,int z){
 if(z==0&&(c==0||c==1||c==3||c==5||c==6))return 0.f;
 return a[c*n+i];
}
// Combined cotangents for the four spatial derivative outputs.
__device__ float coeff(const float* a,const float* v2,const float* profile,
 int kind,int b,int z,int x,int h,int w,long long n){
 if(z<0||z>=h||x<0||x>=w)return 0.f;
 long long m=(long long)h*w,j=z*w+x,i=b*m+j;
 if(kind==0)return masked(a,1,i,n,z)-profile[4*m+j]*masked(a,3,i,n,z);
 if(kind==1)return masked(a,2,i,n,z)-profile[5*m+j]*masked(a,4,i,n,z);
 if(kind==2)return v2[i]*masked(a,0,i,n,z)-profile[j]*masked(a,5,i,n,z);
 return v2[i]*masked(a,0,i,n,z)-profile[m+j]*masked(a,6,i,n,z);
}
extern "C" __global__ void generator_apply_kernel(const float* u,const float* v2,const float* profile,const float* inject,
 const float* force,float* out,long long n,int h,int w,float idx){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int x=i%w,z=(i/w)%h,b=i/((long long)h*w);long long m=(long long)h*w,j=z*w+x;
 const float cf[4]={1225.f/1024.f,-245.f/3072.f,49.f/5120.f,-5.f/7168.f};
 float px=0,pz=0,dxu=0,dzu=0;
 for(int k=1;k<=4;k++){
  px+=cf[k-1]*(get(u,b,z,x+k,h,w)-get(u,b,z,x-k+1,h,w));
  pz+=cf[k-1]*(get(u,b,z+k,x,h,w)-get(u,b,z-k+1,x,h,w));
  dxu+=cf[k-1]*(get(u+n,b,z,x+k-1,h,w)-get(u+n,b,z,x-k,h,w));
  dzu+=cf[k-1]*(get(u+2*n,b,z+k-1,x,h,w,true)-get(u+2*n,b,z-k,x,h,w,true));
 }
 px*=idx;pz*=idx;dxu*=idx;dzu*=idx;
 out[i]=v2[i]*(dxu+dzu+u[5*n+i]+u[6*n+i]+inject[i]*force[b]);
 out[n+i]=px+u[3*n+i];out[2*n+i]=pz+u[4*n+i];
 out[3*n+i]=-profile[6*m+j]*u[3*n+i]-profile[4*m+j]*px;
 out[4*n+i]=-profile[7*m+j]*u[4*n+i]-profile[5*m+j]*pz;
 out[5*n+i]=-profile[2*m+j]*u[5*n+i]-profile[j]*dxu;
 out[6*n+i]=-profile[3*m+j]*u[6*n+i]-profile[m+j]*dzu;
 if(z==0){out[i]=0;out[n+i]=0;out[3*n+i]=0;out[5*n+i]=0;out[6*n+i]=0;}
}
extern "C" __global__ void generator_vjp_kernel(const float* u,const float* v2,const float* profile,const float* inject,
 const float* force,const float* a,float* du,float* dv2,long long n,int h,int w,float idx){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int x=i%w,z=(i/w)%h,b=i/((long long)h*w);long long m=(long long)h*w,j=z*w+x;
 const float cf[4]={1225.f/1024.f,-245.f/3072.f,49.f/5120.f,-5.f/7168.f};
 float dp=0,dux=0,duz=0,dxu=0,dzu=0;
 for(int k=1;k<=4;k++){
  float cx=coeff(a,v2,profile,0,b,z,x-k,h,w,n)-coeff(a,v2,profile,0,b,z,x+k-1,h,w,n);
  float cz=coeff(a,v2,profile,1,b,z-k,x,h,w,n)-coeff(a,v2,profile,1,b,z+k-1,x,h,w,n);
  if(z>0&&k-1-z>=0)cz+=coeff(a,v2,profile,1,b,k-1-z,x,h,w,n);
  dp+=cf[k-1]*(cx+cz);
  dux+=cf[k-1]*(coeff(a,v2,profile,2,b,z,x-k+1,h,w,n)-coeff(a,v2,profile,2,b,z,x+k,h,w,n));
  float uz=coeff(a,v2,profile,3,b,z-k+1,x,h,w,n)-coeff(a,v2,profile,3,b,z+k,x,h,w,n);
  if(k-1-z>=0)uz-=coeff(a,v2,profile,3,b,k-1-z,x,h,w,n);
  duz+=cf[k-1]*uz;
  dxu+=cf[k-1]*(get(u+n,b,z,x+k-1,h,w)-get(u+n,b,z,x-k,h,w));
  dzu+=cf[k-1]*(get(u+2*n,b,z+k-1,x,h,w,true)-get(u+2*n,b,z-k,x,h,w,true));
 }
 const float hp=masked(a,0,i,n,z),hu=masked(a,1,i,n,z),hz=masked(a,2,i,n,z);
 const float hpx=masked(a,3,i,n,z),hpz=masked(a,4,i,n,z);
 const float hfx=masked(a,5,i,n,z),hfz=masked(a,6,i,n,z);
 du[i]=dp*idx;du[n+i]=dux*idx;du[2*n+i]=duz*idx;
 du[3*n+i]=hu-profile[6*m+j]*hpx;
 du[4*n+i]=hz-profile[7*m+j]*hpz;
 du[5*n+i]=v2[i]*hp-profile[2*m+j]*hfx;
 du[6*n+i]=v2[i]*hp-profile[3*m+j]*hfz;
 dv2[i]=hp*((dxu+dzu)*idx+u[5*n+i]+u[6*n+i]+inject[i]*force[b]);
}
extern "C" __global__ void taylor_update_kernel(const float* u,const float* k1,
 const float* k2,const float* k3,const float* k4,float* out,long long n,float dt){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 out[i]=u[i]+dt*(k1[i]+dt*(k2[i]/2.f+dt*(k3[i]/6.f+dt*k4[i]/24.f)));
}
extern "C" __global__ void axpy_kernel(const float* a,const float* b,float* out,long long n,float c){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 out[i]=a[i]+c*b[i];
}
'''

def _nvrtc_library():
    paths=[]
    try:
        spec=importlib.util.find_spec('nvidia.cuda_nvrtc')
        if spec and spec.submodule_search_locations:
            for root in spec.submodule_search_locations:paths.extend(Path(root).glob('lib/libnvrtc.so*'))
    except (ModuleNotFoundError,ValueError):pass
    for prefix in (Path(sys.prefix),Path(_scno_local_asset('environments/envs/deepwave_env'))):
        paths.extend(prefix.glob('lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*'))
    paths.extend(Path('/usr/local/cuda/lib64').glob('libnvrtc.so*'))
    for path in paths:
        try:return ctypes.CDLL(str(path.resolve())),str(path.resolve())
        except OSError:continue
    try:return ctypes.CDLL('libnvrtc.so.12'),'libnvrtc.so.12'
    except OSError as error:raise RuntimeError('NVRTC shared library not found') from error


def _compile_ptx(source,architecture):
    lib,library_path=_nvrtc_library()
    ptr=ctypes.c_void_p;integer=ctypes.c_int;charp=ctypes.c_char_p
    lib.nvrtcCreateProgram.argtypes=[ctypes.POINTER(ptr),charp,charp,integer,ctypes.POINTER(charp),ctypes.POINTER(charp)]
    lib.nvrtcCompileProgram.argtypes=[ptr,integer,ctypes.POINTER(charp)]
    lib.nvrtcGetProgramLogSize.argtypes=[ptr,ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetProgramLog.argtypes=[ptr,ptr]
    lib.nvrtcGetCUBINSize.argtypes=[ptr,ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetCUBIN.argtypes=[ptr,ptr]
    lib.nvrtcDestroyProgram.argtypes=[ctypes.POINTER(ptr)]
    lib.nvrtcGetErrorString.argtypes=[integer];lib.nvrtcGetErrorString.restype=charp
    for name in ('nvrtcCreateProgram','nvrtcCompileProgram','nvrtcGetProgramLogSize','nvrtcGetProgramLog','nvrtcGetCUBINSize','nvrtcGetCUBIN','nvrtcDestroyProgram'):
        getattr(lib,name).restype=integer
    def check(code):
        if code:raise RuntimeError('NVRTC: '+lib.nvrtcGetErrorString(code).decode())
    program=ptr();check(lib.nvrtcCreateProgram(ctypes.byref(program),source.encode(),b'lwc84_generator.cu',0,None,None))
    options=[f'--gpu-architecture={architecture}'.encode(),b'--std=c++14',b'--fmad=false']
    try:
        code=lib.nvrtcCompileProgram(program,len(options),(charp*len(options))(*options))
        size=ctypes.c_size_t();check(lib.nvrtcGetProgramLogSize(program,ctypes.byref(size)))
        log=ctypes.create_string_buffer(max(size.value,1));check(lib.nvrtcGetProgramLog(program,log))
        if code:raise RuntimeError('NVRTC compilation failed: '+log.value.decode())
        check(lib.nvrtcGetCUBINSize(program,ctypes.byref(size)))
        output=ctypes.create_string_buffer(size.value);check(lib.nvrtcGetCUBIN(program,output))
        return ctypes.string_at(output,size.value),dict(nvrtc_library=library_path,options=[v.decode() for v in options],compile_log=log.value.decode())
    finally:lib.nvrtcDestroyProgram(ctypes.byref(program))


class CUDAGenerator:
    arithmetic="float32"
    precision_fidelity_certified=False
    def __init__(self,build_dir,device='cuda'):
        import json
        self.device=torch.device(device)
        if self.device.type!='cuda':raise ValueError('CUDA generator requires CUDA')
        if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        major,minor=torch.cuda.get_device_capability(self.device);self.arch=f'sm_{major}{minor}'
        self.source_sha256=hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()
        build_dir=Path(build_dir);build_dir.mkdir(parents=True,exist_ok=True)
        stem=f'lwc84_generator_fp32_{self.source_sha256[:16]}_{self.arch}_nofma'
        self.ptx_path=build_dir/f'{stem}.ptx'
        if not self.ptx_path.exists():
            import fcntl
            with (build_dir/f'{stem}.build.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if not self.ptx_path.exists():
                    source=build_dir/f'{stem}.cu';source.write_text(CUDA_SOURCE)
                    ptx,receipt=_compile_ptx(CUDA_SOURCE,f'sm_{major}{minor}')
                    temp=build_dir/f'{stem}.partial.ptx';temp.write_bytes(ptx);temp.replace(self.ptx_path)
                    receipt.update(arithmetic="float32",compensation="none",fma=False,source_sha256=self.source_sha256,ptx_sha256=hashlib.sha256(ptx).hexdigest(),architecture=self.arch)
                    (build_dir/f'{stem}.compile.json').write_text(json.dumps(receipt,indent=2)+'\n')
        self.driver=ctypes.CDLL('libcuda.so.1')
        ptr=ctypes.c_void_p;integer=ctypes.c_int;uint=ctypes.c_uint
        self.driver.cuInit.argtypes=[uint];self.driver.cuInit.restype=integer
        self.driver.cuModuleLoadData.argtypes=[ctypes.POINTER(ptr),ptr];self.driver.cuModuleLoadData.restype=integer
        self.driver.cuModuleGetFunction.argtypes=[ctypes.POINTER(ptr),ptr,ctypes.c_char_p];self.driver.cuModuleGetFunction.restype=integer
        self.driver.cuLaunchKernel.argtypes=[ptr]+[uint]*7+[ptr,ctypes.POINTER(ptr),ctypes.POINTER(ptr)]
        self.driver.cuLaunchKernel.restype=integer
        self.driver.cuGetErrorString.argtypes=[integer,ctypes.POINTER(ctypes.c_char_p)];self.driver.cuGetErrorString.restype=integer
        self._check(self.driver.cuInit(0))
        with torch.cuda.device(self.device):
            # Establish PyTorch's primary context; all driver calls use it.
            torch.empty(0,device=self.device)
            self._ptx_buffer=ctypes.create_string_buffer(self.ptx_path.read_bytes())
            self.module=ptr();self._check(self.driver.cuModuleLoadData(ctypes.byref(self.module),self._ptx_buffer))
            self.forward_function=ptr();self.vjp_function=ptr();self.update_function=ptr();self.axpy_function=ptr()
            self._check(self.driver.cuModuleGetFunction(ctypes.byref(self.forward_function),self.module,b'generator_apply_kernel'))
            self._check(self.driver.cuModuleGetFunction(ctypes.byref(self.vjp_function),self.module,b'generator_vjp_kernel'))
            self._check(self.driver.cuModuleGetFunction(ctypes.byref(self.update_function),self.module,b'taylor_update_kernel'))
            self._check(self.driver.cuModuleGetFunction(ctypes.byref(self.axpy_function),self.module,b'axpy_kernel'))

    def _check(self,code):
        if code:
            message=ctypes.c_char_p();self.driver.cuGetErrorString(code,ctypes.byref(message))
            raise RuntimeError(f'CUDA driver {code}: '+(message.value.decode() if message.value else 'unknown'))

    def _launch(self,function,tensors,shape,dx):
        b,h,w=shape;n=b*h*w
        values=[ctypes.c_void_p(t.data_ptr()) for t in tensors]+[ctypes.c_longlong(n),ctypes.c_int(h),ctypes.c_int(w),ctypes.c_float(1./float(dx))]
        pointers=(ctypes.c_void_p*len(values))(*[ctypes.cast(ctypes.byref(v),ctypes.c_void_p) for v in values])
        with torch.cuda.device(self.device):
            self._check(self.driver.cuLaunchKernel(function,(n+255)//256,1,1,256,1,1,0,
                ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream),pointers,None))

    def _launch_flat(self,function,tensors,count,scale):
        values=[ctypes.c_void_p(t.data_ptr()) for t in tensors]+[ctypes.c_longlong(count),ctypes.c_float(scale)]
        pointers=(ctypes.c_void_p*len(values))(*[ctypes.cast(ctypes.byref(v),ctypes.c_void_p) for v in values])
        with torch.cuda.device(self.device):
            self._check(self.driver.cuLaunchKernel(function,(count+255)//256,1,1,256,1,1,0,
                ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream),pointers,None))

    @torch.no_grad()
    def _axpy(self,a,b,c,out=None):
        # Internal generated contiguous tensors only. Aliasing out=a is safe:
        # each CUDA thread reads and writes its own scalar exactly once.
        if out is None:out=torch.empty_like(a)
        self._launch_flat(self.axpy_function,(a,b,out),a.numel(),float(c))
        return out

    def _validate(self,state,v2,profiles,injection,force,dx,adj=None):
        if state.ndim!=4 or state.shape[0]!=7:raise ValueError('state[7,B,H,W] required')
        b,h,w=state.shape[1:]
        if v2.shape!=(b,h,w) or injection.shape!=v2.shape or profiles.shape!=(8,h,w) or force.shape!=(b,):
            raise ValueError('Generator input shape mismatch')
        inputs=(state,v2,profiles,injection,force)+(() if adj is None else (adj,))
        if any(t.device!=self.device or t.dtype!=torch.float32 or not t.is_contiguous() for t in inputs):
            raise ValueError('Generator requires contiguous float32 on configured CUDA device')
        if adj is not None and adj.shape!=state.shape:raise ValueError('Adjoint/state shape mismatch')
        if dx<=0:raise ValueError('Positive dx required')
        return b,h,w

    @torch.no_grad()
    def apply_A(self,state,v2,profiles,injection,force,dx=5.):
        shape=self._validate(state,v2,profiles,injection,force,dx)
        out=torch.empty_like(state)
        self._launch(self.forward_function,(state,v2,profiles,injection,force,out),shape,dx)
        return out

    @torch.no_grad()
    def vjp_A(self,state,v2,profiles,injection,force,adj_out,dx=5.):
        shape=self._validate(state,v2,profiles,injection,force,dx,adj_out)
        du=torch.empty_like(state);dv2=torch.empty_like(v2)
        self._launch(self.vjp_function,(state,v2,profiles,injection,force,adj_out,du,dv2),shape,dx)
        return du,dv2

    @torch.no_grad()
    def step(self,state,v2,profiles,injection,forces,dt,dx=5.):
        if forces.shape!=(4,state.shape[1]) or dt<=0:raise ValueError('forces[4,B] and positive dt required')
        k1=self.apply_A(state,v2,profiles,injection,forces[0],dx)
        k2=self.apply_A(k1,v2,profiles,injection,forces[1],dx)
        k3=self.apply_A(k2,v2,profiles,injection,forces[2],dx)
        k4=self.apply_A(k3,v2,profiles,injection,forces[3],dx)
        out=torch.empty_like(state)
        self._launch_flat(self.update_function,(state,k1,k2,k3,k4,out),state.numel(),float(dt))
        return out

    @torch.no_grad()
    def step_vjp(self,state,v2,profiles,injection,forces,adj_out,dt,dx=5.):
        """Recompute k1..k3; reverse the four generator applications exactly."""
        if forces.shape!=(4,state.shape[1]) or dt<=0:raise ValueError('forces[4,B] and positive dt required')
        k1=self.apply_A(state,v2,profiles,injection,forces[0],dx)
        k2=self.apply_A(k1,v2,profiles,injection,forces[1],dx)
        k3=self.apply_A(k2,v2,profiles,injection,forces[2],dx)
        dk3,dv4=self.vjp_A(k3,v2,profiles,injection,forces[3],adj_out*(dt**4/24),dx)
        del k3
        self._axpy(dk3,adj_out,dt**3/6,out=dk3)
        dk2,dv3=self.vjp_A(k2,v2,profiles,injection,forces[2],dk3,dx)
        del k2,dk3
        self._axpy(dk2,adj_out,dt**2/2,out=dk2)
        dk1,dv2=self.vjp_A(k1,v2,profiles,injection,forces[1],dk2,dx)
        del k1,dk2
        self._axpy(dk1,adj_out,dt,out=dk1)
        du,dv1=self.vjp_A(state,v2,profiles,injection,forces[0],dk1,dx)
        del dk1
        self._axpy(du,adj_out,1.,out=du)
        # Preserve original left-associated dv1+dv2+dv3+dv4 sum order.
        self._axpy(dv1,dv2,1.,out=dv1)
        self._axpy(dv1,dv3,1.,out=dv1)
        self._axpy(dv1,dv4,1.,out=dv1)
        return du,dv1
