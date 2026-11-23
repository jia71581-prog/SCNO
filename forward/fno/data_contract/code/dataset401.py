"""Exact 0:2:801 extraction and guarded raw/normalized training access."""
from pathlib import Path
import hashlib
import json
import os
import h5py
import numpy as np

INDICES=np.arange(0,801,2,dtype=np.int64)
TIMES=np.arange(401,dtype=np.float64)*.0025

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def array_sha(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def summarize(a):
    assert a.shape==(401,201,201) and a.dtype==np.float32
    assert np.isfinite(a).all() and not a[0].any() and not a[:,0].any() and np.any(a[-1])
    frame_sum=np.sum(a,dtype=np.float64,axis=(1,2))
    energy=np.array([np.square(x.astype(np.float64)).sum() for x in a])
    return dict(count=int(a.size),sum=float(frame_sum.sum()),sum_squares=float(energy.sum()),
        min=float(a.min()),max=float(a.max()),max_abs=float(np.abs(a).max()),
        per_frame_sum=frame_sum.tolist(),per_frame_sum_squares=energy.tolist(),
        finite=True,t0_exact_zero=True,top_exact_zero=True,last_frame_nonzero=True)

def build_vds(path,rows,*,pressure_scale_pa,spatial_shape=(201,201)):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    n=len(rows); assert n>0 and pressure_scale_pa>0
    wave=h5py.VirtualLayout(shape=(n,401,*spatial_shape),dtype=np.float32)
    velocity=h5py.VirtualLayout(shape=(n,*spatial_shape),dtype=np.float32)
    names=[]
    for i,row in enumerate(rows):
        src=Path(row['source_file']).absolute()
        if not src.is_file():raise FileNotFoundError(src)
        nf=row['source_frames'];assert nf in (801,401)
        with h5py.File(src,'r') as f:
            assert f['wavefield'].shape==(nf,*spatial_shape) and f['wavefield'].dtype==np.float32
            assert f['velocity201'].shape==spatial_shape
        rel=os.path.relpath(src,path.parent);names.append(rel)
        v=h5py.VirtualSource(rel,'wavefield',shape=(nf,*spatial_shape))
        wave[i]=v[::2] if nf==801 else v[:]
        velocity[i]=h5py.VirtualSource(rel,'velocity201',shape=spatial_shape)
    tmp=path.with_name(path.name+'.tmp')
    with h5py.File(tmp,'w',libver='latest') as f:
        f.create_virtual_dataset('wavefield',wave,fillvalue=np.nan)
        f.create_virtual_dataset('velocity',velocity,fillvalue=np.nan)
        f['time_s']=TIMES;f['time_s_fp32']=TIMES.astype(np.float32);f['native_indices']=INDICES
        f['source_parameters']=np.asarray([x['source_model'] for x in rows],np.float32)
        f['source_solver']=np.asarray([x['source_solver'] for x in rows],np.float64)
        text=h5py.string_dtype('utf-8')
        for key,values in [('sample_id',[x['sample_id'] for x in rows]),('family',[x['family'] for x in rows]),('source_files',names)]:
            f.create_dataset(key,data=np.asarray(values,dtype=object),dtype=text)
        f.attrs.update(schema='ACOUSTIC_EXACT_UNIFORM401_V1',pressure_units='Pa',pressure_scale_pa=pressure_scale_pa,
            pressure_normalization='p / (source_amplitude * pressure_scale_pa)',
            temporal_interpolation=False,dt_s=.0025,first_time_s=0.,last_time_s=1.,
            spacing_m=10.,physical_sources_per_record=1,missing_source_fill='NaN; guarded loader raises')
        f.flush()
    with tmp.open('rb') as h:os.fsync(h.fileno())
    os.replace(tmp,path)

def load_record(path,index):
    """Read one record; model inputs are velocity/source and labels have a fixed scale.

    No per-frame normalization. Fail before reading missing VDS sources.
    The zero initial frame is valid, so never divide a loss by its own zero energy.
    """
    path=Path(path)
    with h5py.File(path,'r') as f:
        if not 0<=index<len(f['sample_id']):raise IndexError(index)
        name=f['source_files'].asstr()[index]
        source_path=(path.parent/name).resolve()
        if not source_path.is_file():raise FileNotFoundError(source_path)
        pressure=f['wavefield'][index];velocity=f['velocity'][index];source=f['source_parameters'][index]
        if not np.isfinite(pressure).all() or not np.isfinite(velocity).all():
            raise ValueError('Nonfinite data or unresolved virtual source')
        scale=np.float32(f.attrs['pressure_scale_pa'])*source[4]
        if not scale>0:raise ValueError('Positive source amplitude and pressure scale required')
        return dict(pressure_pa=pressure,pressure_normalized=pressure/scale,velocity_mps=velocity,
            source_parameters=source,time_s=f['time_s'][...],time_s_fp32=f['time_s_fp32'][...],
            sample_id=f['sample_id'].asstr()[index],family=f['family'].asstr()[index],pressure_scale_pa=float(scale))
