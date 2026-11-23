"""Independent FP32-state, FP64-gradient-accumulation seven-state LWC trajectory with exact discrete-adjoint rematerialization.

Major checkpoints live on CPU. During backward a major interval is replayed
into CPU minor boundaries, then each short minor interval is replayed on GPU.
Only minor_steps seven-state frames are live on device. Every CPML state and
all four Taylor-stage velocity/source derivatives participate in the adjoint.
"""
from __future__ import annotations
import json,time,math
from pathlib import Path
import torch
from torch.autograd.function import once_differentiable
from .protocol import receiver_forward,receiver_transpose


def _progress(path,phase,step,total,batch):
    if path is None:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(dict(phase=phase,step=step,total_steps=total,batch=batch,time_unix=time.time()))+'\n');tmp.replace(path)


def forward_trajectory(v2,profiles,injection,forces,indices,weights,generator,*,dt,dx,
                       sample_stride,major_steps=512,save_checkpoints=False,progress_path=None):
    steps=forces.shape[0];batch,h,w=v2.shape
    if steps%sample_stride:raise ValueError('Final time must be an exact receiver sample')
    if forces.shape!=(steps,4,batch):raise ValueError('forces must be[steps,4,batch]')
    if v2.dtype!=torch.float32 or profiles.dtype!=torch.float32:raise ValueError('Explicit FP32 reference variant required')
    state=torch.zeros((7,batch,h,w),device=v2.device,dtype=v2.dtype)
    observed=v2.new_empty((batch,indices.shape[0],steps//sample_stride+1))
    checkpoints={}
    with torch.no_grad():
        for n in range(steps):
            if n%major_steps==0:
                if save_checkpoints:checkpoints[n]=state.to('cpu',copy=True)
                _progress(progress_path,'forward',n,steps,batch)
            if n%sample_stride==0:
                observed[:,:,n//sample_stride]=receiver_forward(state[0],indices,weights,n=2*h-w,pad=w-h)
            state=generator.step(state,v2,profiles,injection,forces[n],dt,dx)
        observed[:,:,-1]=receiver_forward(state[0],indices,weights,n=2*h-w,pad=w-h)
        if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(observed).all()):
            raise RuntimeError('Nonfinite full LWC trajectory')
    _progress(progress_path,'forward_complete',steps,steps,batch)
    return observed,checkpoints


class _Trajectory(torch.autograd.Function):
    @staticmethod
    def forward(ctx,v2,profiles,injection,forces,indices,weights,generator,dt,dx,stride,major,minor,progress_path):
        ctx.save_for_backward(v2,profiles,injection,forces,indices,weights)
        ctx.settings=(generator,dt,dx,stride,major,minor,progress_path)
        observed,ctx.checkpoints=forward_trajectory(v2,profiles,injection,forces,indices,weights,generator,
            dt=dt,dx=dx,sample_stride=stride,major_steps=major,save_checkpoints=True,progress_path=progress_path)
        return observed

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        if getattr(ctx,"consumed",False):raise RuntimeError("LWC trajectory supports one backward per forward")
        ctx.consumed=True
        v2,profiles,injection,forces,indices,weights=ctx.saved_tensors
        generator,dt,dx,stride,major,minor,path=ctx.settings
        steps=forces.shape[0];batch,h,w=v2.shape
        if cotangent.shape!=(batch,indices.shape[0],steps//stride+1):raise ValueError('Receiver cotangent mismatch')
        adj=torch.zeros((7,batch,h,w),device=v2.device,dtype=v2.dtype)
        adj[0]=receiver_transpose(cotangent[:,:,-1],indices,weights,n=2*h-w,pad=w-h)
        gv=torch.zeros_like(v2,dtype=torch.float64)
        try:
            with torch.no_grad():
                for begin in reversed(sorted(ctx.checkpoints)):
                    stop=min(begin+major,steps)
                    state=ctx.checkpoints.pop(begin).to(v2.device)
                    boundaries={}
                    for n in range(begin,stop):
                        if (n-begin)%minor==0:boundaries[n]=state.to('cpu',copy=True)
                        state=generator.step(state,v2,profiles,injection,forces[n],dt,dx)
                    del state
                    for first in reversed(sorted(boundaries)):
                        last=min(first+minor,stop)
                        state=boundaries.pop(first).to(v2.device)
                        states=[state]
                        for n in range(first,last-1):
                            state=generator.step(state,v2,profiles,injection,forces[n],dt,dx);states.append(state)
                        del state
                        for n in range(last-1,first-1,-1):
                            adj,increment=generator.step_vjp(states[n-first],v2,profiles,injection,forces[n],adj,dt,dx)
                            gv.add_(increment.double())
                            if n%stride==0:
                                adj[0].add_(receiver_transpose(cotangent[:,:,n//stride],indices,weights,n=2*h-w,pad=w-h))
                        del states,increment
                    _progress(path,'adjoint',begin,steps,batch)
                if not bool(torch.isfinite(gv).all()):raise RuntimeError('Nonfinite LWC velocity-squared adjoint')
            _progress(path,'adjoint_complete',steps,steps,batch)
            return (gv.to(v2.dtype),)+(None,)*12
        finally:
            ctx.checkpoints.clear()


def differentiable_trajectory(v2,profiles,injection,forces,indices,weights,generator,*,
                              dt,dx,sample_stride,major_steps=512,minor_steps=8,progress_path=None):
    if min(sample_stride,major_steps,minor_steps)<1:raise ValueError('Positive checkpoint and sampling lengths required')
    if torch.is_grad_enabled() and v2.requires_grad:
        return _Trajectory.apply(v2,profiles,injection,forces,indices,weights,generator,
            float(dt),float(dx),int(sample_stride),int(major_steps),int(minor_steps),progress_path)
    return forward_trajectory(v2,profiles,injection,forces,indices,weights,generator,dt=dt,dx=dx,
        sample_stride=sample_stride,major_steps=major_steps,progress_path=progress_path)[0]
