"""Units: medium [Vp, Vs] in km/s; arrays [z,x]; output [shot,z,x,t,ux/uz]."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch


def atomic_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def atomic_torch(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)


def quarantine_orphan(path,receipt):
    """Preserve a payload committed just before interruption, then regenerate it."""
    path=Path(path)
    if path.exists() and not Path(receipt).exists():
        archive=path.parent/'orphaned';archive.mkdir(exist_ok=True)
        path.rename(archive/f'{path.stem}_{time.time_ns()}{path.suffix}')


def random_medium(seed,n=64):
    rng=np.random.default_rng(seed)
    # Embed in a larger periodic field, then crop: opposite edges need not match.
    m=4*n
    k=2*np.pi*np.fft.fftfreq(m)
    k2=k[:,None]**2+k[None,:]**2
    def field(spectrum):
        a=np.fft.ifft2(np.fft.fft2(rng.standard_normal((m,m)))*spectrum).real
        a=a[n:2*n,n:2*n];return (a-a.mean())/a.std()
    vs=3.+.3*field((1.+8.**2*k2)**(-.75))  # sqrt(2-D von Karman PSD), H=.5
    ratio=1.732*(1.+.02*field(np.exp(-.25*32.**2*k2)))
    result=np.stack([vs*ratio,vs]).astype('float32')
    if result[1].min() <= .8 or result[1].max() >= 4.5:
        return random_medium(seed+100000000,n)  # deterministic extreme-tail rejection
    return result


def density_from_vs(vs):
    # Brocher (2005) equations 9 and 1; velocities km/s, output kg/m^3.
    p=.9409+2.0947*vs-.8206*vs**2+.2683*vs**3-.0251*vs**4
    return 1000*(1.6612*p-.4721*p**2+.0671*p**3-.0043*p**4+.000106*p**5)


def ring_sources(n=64,count=14,radius=.4):
    a=np.arange(count)*2*np.pi/count
    c=(n-1)/2
    return np.rint(np.stack([c+radius*(n-1)*np.sin(a),c+radius*(n-1)*np.cos(a)],1)).astype('int64')


def source_input(locations,cfg,device='cpu',dtype=torch.float32):
    n=cfg['grid']['n'];nt=cfg['grid']['frames']
    loc=torch.as_tensor(locations,device=device,dtype=dtype)
    z=torch.arange(n,device=device,dtype=dtype)[None,:,None]
    x=torch.arange(n,device=device,dtype=dtype)[None,None,:]
    spatial=torch.exp(-((z-loc[:,0,None,None])**2+(x-loc[:,1,None,None])**2)/cfg['source']['width_cells']**2)
    t=torch.arange(nt,device=device,dtype=dtype)*cfg['grid']['duration_s']/nt
    f=cfg['source']['frequency_hz'];a=(np.pi*f*(t-cfg['source']['peak_time_s']))**2
    w=(1-2*a)*torch.exp(-a)
    return spatial[...,None]*w.abs()


def operator_input(velocity,source):
    if velocity.ndim==3:velocity=velocity[None].expand(source.shape[0],-1,-1,-1)
    v=velocity.permute(0,2,3,1)[:,:,:,None,:].expand(-1,-1,-1,source.shape[-1],-1)
    return torch.cat([v,source[...,None]],-1)


def normalize_input(x,stats):
    return (x-x.new_tensor(stats['input_mean']))/x.new_tensor(stats['input_std'])


def encode_output(x,stats):
    return (x-x.new_tensor(stats['output_mean']))/x.new_tensor(stats['output_std'])


def decode_output(x,stats):
    return x*x.new_tensor(stats['output_std'])+x.new_tensor(stats['output_mean'])


def select_receivers(wave,mode):
    if mode=='full':return wave
    if mode=='surface':return wave[:,0]
    raise ValueError('receiver mode must be full or surface')


def tikhonov(v,reference,dx_m):
    delta=v-reference
    r0=delta.square().mean()
    # First-order penalty on physical km coordinate (velocities in km/s).
    r1=((torch.diff(delta,dim=1)/(dx_m/1000)).square().mean()+
        (torch.diff(delta,dim=2)/(dx_m/1000)).square().mean())/2
    return r0,r1


@torch.no_grad()
def project_velocity(v,vp_bounds,vs_bounds):
    v[1].clamp_(*vs_bounds);v[0].clamp_(*vp_bounds)
    v[0].copy_(torch.maximum(v[0],v[1]*1.17))


def model_metrics(v,truth):
    v=np.asarray(v,dtype='float64');truth=np.asarray(truth,dtype='float64')
    return {name:dict(relative_l2=float(np.linalg.norm(v[i]-truth[i])/np.linalg.norm(truth[i])),
                      rmse_m_s=float(np.sqrt(np.mean((v[i]-truth[i])**2))*1000))
            for i,name in enumerate(['vp','vs'])}


def relative_training_loss(pred,target):
    p=pred.flatten(1);t=target.flatten(1);d=p-t
    l1=d.abs().sum(1)/t.abs().sum(1).clamp_min(1e-20)
    l2=torch.linalg.vector_norm(d,dim=1)/torch.linalg.vector_norm(t,dim=1).clamp_min(1e-20)
    return (.9*l1+.1*l2).mean()
