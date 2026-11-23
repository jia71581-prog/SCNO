"""Independent read-only native201->101 view; original sources/fields untouched."""
from pathlib import Path
import hashlib,importlib.util
FUTURE_FRAMES=tuple(88+(712*i+49)//99 for i in range(100))
IC_FRAMES=tuple(range(80,88))
WINDOWS={'early':(0,35),'mid':(35,65),'late':(65,100)}
RESTRICTION=Path(__file__).resolve().parents[1]/'sourcegen/restriction.py'
RESTRICTION_SHA='be53a93bced3a8aa7d3ca303e7f367444fb16ca564f2218a4fcec5ac04224792'
def _restriction():
 if hashlib.sha256(RESTRICTION.read_bytes()).hexdigest()!=RESTRICTION_SHA:raise PermissionError('original generator restriction drift')
 spec=importlib.util.spec_from_file_location('fixed100_immutable_B5',RESTRICTION);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.restrict_nodal_2x

def pressure_view(p):
 if tuple(p.shape[-2:])!=(201,201):raise ValueError('only native201 pressure')
 result=_restriction()(p);result=result.clone() if hasattr(result,'clone') else result.copy();result[...,0,:]=0;return result

def velocity_view(c):
 if tuple(c.shape[-2:])!=(201,201):raise ValueError('only native201 velocity')
 return _restriction()(c)

def source_map101(source_FP64):
 import numpy as np
 sx,sz=float(source_FP64[0]),float(source_FP64[1]);dx=20.
 if not 0<=sx<=2000 or not 0<=sz<=2000:raise ValueError('source in original physical domain')
 x=min(int(sx//dx),99);z=min(int(sz//dx),99);fx=(sx-x*dx)/dx;fz=(sz-z*dx)/dx
 result=np.zeros((101,101),np.float32)
 for iz,ix,w in [(z,x,(1-fz)*(1-fx)),(z,x+1,(1-fz)*fx),(z+1,x,fz*(1-fx)),(z+1,x+1,fz*fx)]:result[iz,ix]+=np.float32(w)
 if (result<0).any() or abs(float(result.sum())-1)>2e-7:raise AssertionError('unitmass bilinear map not delta/400')
 return result

def protocol():
 return dict(native_stored_frames=list(FUTURE_FRAMES),IC_pressure_frames=list(IC_FRAMES),x_m=list(range(0,2001,20)),z_m=list(range(0,2001,20)),pressure='native201 B5reflect ->::2 ->top0 forICandGT',velocity='native201 B5reflect ->::2; no pressuretop reset',source_map='fresh physical-FP64 bilinear101 unitmass; not R(oldmap)',restriction_source=str(RESTRICTION),restriction_sha256=RESTRICTION_SHA,windows={k:dict(ordinals=[a,b],native_frame_range=[FUTURE_FRAMES[a],FUTURE_FRAMES[b-1]]) for k,(a,b) in WINDOWS.items()},time_spacing_native_steps=[7,8],no_uniform_dt_FFT_assumption=True,PDE_loss_enabled=False,not_original713_or_actual101numerical_solver=True)
