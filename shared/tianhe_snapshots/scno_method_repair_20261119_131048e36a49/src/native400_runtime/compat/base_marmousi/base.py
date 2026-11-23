import hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch

def module(n):
 if n.startswith("scno_cycle."):return "scno_"+n.split(".")[1]
 return n.split(".")[0]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def js(p,d):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)

def emit(d):print(json.dumps(d,allow_nan=False),flush=True)

def seed():
 random.seed(101);np.random.seed(101);torch.manual_seed(101)
 if torch.cuda.is_initialized():torch.cuda.manual_seed_all(101)

def thash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()

def norm(t):return float(t.detach().abs().double().square().sum().sqrt())

def input_data(records,spec):
 r=records[spec['local_index']];ic=records.read_wavefield(spec['local_index'],r.time_s[spec['ic_frames']]);assert bool(ic.exact.all());return {'spec':spec,'velocity':r.velocity_mps[None],'source':r.source_parameters[None],'source_map':r.source_map[None],'ic':ic.values[None],'anchor':r.time_s[spec['ic_frames'][:1]],'x':r.x_m,'z':r.z_m,'time_axis':r.time_s}

def prepared_state(model,c,norm,device):
 # P2: phase maps ride along only when the model asks for them (phase_mode != off); legacy models ignore the cache.
 extra={}
 if getattr(model,'phase_mode','off')!='off':
  if 'phase_T' not in c:raise ValueError('phase_mode model needs data.phase_cache_dir so records carry phase maps')
  extra['phase_maps']=(c['phase_T'].to(device),c['phase_r'].to(device))
 # V18 formal operator is family-free. Family remains available only to sampling/metrics;
 # it is deliberately never bound into PreparedV3State.
 # Born scattering tables (scatter_binding.py, 20260926): only when the pool
 # supplied them, so a run without data.scatter_cache_dir is untouched.
 if c.get('scatter_T_scatterer') is not None:
  ts=c['scatter_T_scatterer'];tq=c['scatter_T_query']
  # The cache stores one record; the model wants a leading record axis.
  ts=ts[None] if ts.ndim==2 else ts;tq=tq[None] if tq.ndim==4 else tq
  extra['scatter_tables']=(ts.to(device),tq.to(device),(float(c['x'][1]-c['x'][0]),float(c['z'][1]-c['z'][0])))
 prep=model.prepare_sources(model.encode_medium(c['velocity'].to(device),norm),c['source'].to(device),c['source_map'].to(device),norm,record_to_medium=torch.zeros(1,dtype=torch.long,device=device),ic_snapshots_physical=c['ic'].to(device),anchor_time_s=c['anchor'].to(device),**extra)
 # v17 spectral carrier (20260928): a carrier-augmented dense decoder needs its
 # per-record analytic carrier state bound here, the single chokepoint both the
 # training forward (forward_live) and the dev9 evaluation share. Models without the
 # carrier decoder skip this branch bitwise.
 if hasattr(getattr(model,'dense_decoder',None),'set_carrier'):
  from carrier_wiring import bind_carrier;bind_carrier(model,c,norm,device)
 return prep

def forward_live(model,c,times,norm,device,time_block=1):
 # time_block only re-partitions the training-time checkpoint segments inside
 # dense_normalized; the effective block is min(time_block,len(times)). The
 # default 1 keeps every pre-existing caller (inference, no_cycle checks) bitwise identical.
 prep=prepared_state(model,c,norm,device);return model.dense_normalized(prep,times[None].to(device),x_m=c['x'].to(device),z_m=c['z'].to(device),time_block=int(time_block))

def forward_live_with_raw(model,c,times,norm,device,time_block=1):
 # Same chokepoint as forward_live but returns (final, raw): raw is the field
 # before the free-surface factor, used by the weight_bc physics term.
 prep=prepared_state(model,c,norm,device);return model.dense_normalized_with_raw(prep,times[None].to(device),x_m=c['x'].to(device),z_m=c['z'].to(device),time_block=int(time_block))

def all_param_snapshot(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters()}

def grad_snapshot(model):
 rows={}
 for n,p in model.named_parameters():
  g=p.grad;kind='none' if g is None else 'zero' if not bool(g.count_nonzero()) else 'nonzero';rows[n]={'kind':kind,'finite':True if g is None else bool(torch.isfinite(g).all()),'norm':0. if g is None else norm(g)};assert rows[n]['finite'],n
 return rows

def parameter_changes(model,before):
 rows=[];groups={}
 for n,p in model.named_parameters():
  after=p.detach().cpu();old=before[n];delta=after-old;change=norm(delta);oldnorm=norm(old);cnt=int(torch.count_nonzero(after!=old));group=module(n);rows.append({'name':n,'module':group,'dtype':str(p.dtype),'numel':p.numel(),'before_norm':oldnorm,'absolute_delta_norm':change,'relative_delta_norm':change/oldnorm if oldnorm else None,'zero_initial':oldnorm==0,'changed_elements':cnt,'changed_fraction':cnt/p.numel(),'before_sha256':thash(old),'after_sha256':thash(after)})
  g=groups.setdefault(group,{'delta_l2_squared':0.,'changed_elements':0,'numel':0});g['delta_l2_squared']+=change**2;g['changed_elements']+=cnt;g['numel']+=p.numel()
 for g in groups.values():g['delta_l2']=g.pop('delta_l2_squared')**.5;g['changed_fraction']=g['changed_elements']/g['numel']
 return rows,groups
