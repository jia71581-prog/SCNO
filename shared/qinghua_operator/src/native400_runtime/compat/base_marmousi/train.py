#!/usr/bin/env python3
"""Config-driven all-parameter training. Never starts merely on import/dry-run."""
from __future__ import annotations
import argparse,json,math,os,signal,sys,time,traceback,shutil,scno_posix_lock as fcntl
from pathlib import Path
import yaml,numpy as np,torch
from model import *
from data import Pool
from checkpoint import atomic_json,atomic_checkpoint,capture_rng,restore_rng,check_resume,prune_checkpoints
from base import forward_live
from grouped_ufno_mionet_v3 import eval_metrics
STOP={'signal':None}

def request_stop(signum,_frame):
 STOP['signal']=signum

def resolved_config(path):
 path=Path(path).resolve();c=yaml.safe_load(path.read_text())
 for key in ['model_spec','initial_checkpoint']:
  if c.get(key):c[key]=str((path.parent/c[key]).resolve()) if not Path(c[key]).is_absolute() else c[key]
 for key in ['source_h5','normalization_json','exclusions_json','dev_panel_json','fork_identity_json']:
  if c['data'].get(key):c['data'][key]=str((path.parent/c['data'][key]).resolve()) if not Path(c['data'][key]).is_absolute() else c['data'][key]
 if c['lift_mode'] not in ['additive','homogeneous','no_cycle']:raise ValueError('invalid lift_mode')
 if c['lift_mode']=='no_cycle':
  if c.get('cycle') is not None:raise ValueError('no_cycle control config must not contain a cycle section')
 else:
  if c.get('cycle') is None:raise ValueError('cycle lift modes require a cycle section')
  if c['cycle']['width']!=16 or c['cycle']['levels']!=3:raise ValueError('this released architecture binds width16/levels3')
 if c['data']['energy_chunk_frames'] not in [8,16]:raise ValueError('energy chunks must be8or16')
 if c['data']['frames_per_update'] not in (16,32,51):raise ValueError('frames_per_update must be 16 (legacy), 32 (scatter arm) or 51 (P1 arm); other values are rejected')
 # Late-window anti-collapse terms (20260929 port from v18_accuracy_20260929/agent_manual_accuracy,
 # hybrid_loss.py byte-identical to the audited implementation). The three late keys are
 # popped BEFORE the historical strict equality check, so every pre-existing config is
 # validated bitwise identically; absent or enabled:false blocks are dropped by
 # normalize_late_window_config and add NO key -> semantic digest and loss value unchanged.
 # frame_floor_rel feeds late_amp's norm floor sqrt(frame_floor_rel*E/n_future) (same
 # convention as the audited hybrid arms, which bound it at 0.03) and is only legal
 # alongside an enabled late_amp block.
 _loss=dict(c['loss']);_late_raw={k:_loss.pop(k) for k in ('late_amp','late_ncc') if k in _loss};_ffr=_loss.pop('frame_floor_rel',None)
 if _loss!={'kind':'stratified_fullfuture_record_norm','epsilon':0.0,'strata':c['data']['frames_per_update']}:raise ValueError('unsupported loss; this release binds vector_norm without floor/epsilon and loss.strata must equal data.frames_per_update')
 from hybrid_loss import normalize_late_window_config
 _late=normalize_late_window_config(_late_raw)
 if 'late_amp' in _late:
  if _ffr is None:raise ValueError('loss.frame_floor_rel is required when late_amp is enabled (norm floor sqrt(frame_floor_rel*E/n_future))')
  _ffr=float(_ffr)
  if not 0.0<_ffr<1.0:raise ValueError('loss.frame_floor_rel must lie in (0,1)')
  _loss['frame_floor_rel']=_ffr
 elif _ffr is not None:raise ValueError('loss.frame_floor_rel is only legal alongside an enabled late_amp block')
 _loss.update(_late);c['loss']=_loss
 # 32 frames (scatter arm, 20260926): the only change is the legal value set above; the
 # stratum count is data.frames_per_update and the per-stratum frame RNG is seeded from
 # (seed|draw|sample_id|frames), so 16 and 51 keep their exact draws and strata (verified
 # bitwise against train.py.bak_frames32_20260926).
 for key in ['checkpoint_every','log_every','checkpoint_seconds']:
  if c['runtime'][key]<=0:raise ValueError('runtime intervals must be positive')
 # --- rolling checkpoint retention (20260926) ---
 # Unset (the historical default) means keep everything, bitwise identical to the
 # pre-20260926 behaviour: existing arms stay reproducible. A positive integer N
 # keeps only the N newest 'checkpoint_*' artifacts; best_* and the step-0 lineage
 # checkpoint are never touched. See checkpoint.plan_checkpoint_pruning.
 c['checkpoint_keep_latest']=c['runtime'].get('checkpoint_keep_latest')
 if c['checkpoint_keep_latest'] is not None:
  c['checkpoint_keep_latest']=int(c['checkpoint_keep_latest'])
  if c['checkpoint_keep_latest']<1:raise ValueError('runtime.checkpoint_keep_latest must be a positive integer when set (omit it to keep every checkpoint)')
 # --- rolling BEST-checkpoint retention (20260926) --------------------------
 # Unset keeps every dev9 best checkpoint (historical behaviour). N>0 keeps the
 # N newest 'best_*' artifacts and prunes the rest, so a per-epoch eval cadence
 # cannot fill the disk. The step-0 lineage checkpoint is never a candidate.
 c['checkpoint_keep_best']=c['runtime'].get('checkpoint_keep_best')
 if c['checkpoint_keep_best'] is not None:
  c['checkpoint_keep_best']=int(c['checkpoint_keep_best'])
  if c['checkpoint_keep_best']<1:raise ValueError('runtime.checkpoint_keep_best must be a positive integer when set (omit it to keep every best checkpoint)')
 if c['lift_mode']!='no_cycle':
  schedule=c['cycle']['depth_schedule'];expected=1
  for i,x in enumerate(schedule):
   if x['start']!=expected or x['depth']<1:raise ValueError('depth schedule must be ordered and gap/overlap free')
   if x.get('end') is None:
    if i!=len(schedule)-1:raise ValueError('open-ended depth entry must be final')
   else:
    if x['end']<x['start']:raise ValueError('invalid depth interval')
    expected=x['end']+1
  if schedule[-1].get('end') is not None:raise ValueError('final depth interval must cover extensions and next_update')
 # --- tbfull additions (defaults reproduce the historical behaviour exactly) ---
 c['train_time_block']=int(c.get('train_time_block',1))
 if not 1<=c['train_time_block']<=c['data']['frames_per_update']:raise ValueError('train_time_block must lie in [1,frames_per_update]; the effective block caps at the supervised frames per update')
 c['sampler_draw_offset']=int(c.get('sampler_draw_offset',0))
 if c['sampler_draw_offset']<0:raise ValueError('sampler_draw_offset must be >=0')
 c['init_optimizer_transplant']=bool(c.get('init_optimizer_transplant',False))
 # v17 filtered transplant (20260928): default False keeps every pre-existing config on
 # the strict initialize_model path bitwise; True routes init through
 # model.init_filtered_transplant (name-intersection load with a drop allowlist).
 c['init_partial']=bool(c.get('init_partial',False))
 ls=c.get('lr_schedule')
 if ls is not None:
  if ls.get('kind')!='cosine':raise ValueError('lr_schedule.kind must be cosine')
  if not 0.0<=float(ls['final_scale'])<=1.0:raise ValueError('lr_schedule.final_scale must lie in [0,1]')
  if int(ls['total_draws'])<1:raise ValueError('lr_schedule.total_draws must be >=1')
  c['lr_schedule']={'kind':'cosine','final_scale':float(ls['final_scale']),'total_draws':int(ls['total_draws'])}
  # Optional linear warmup multiplied onto the cosine (lr grid arms, 20260928). The key
  # enters the schedule dict (and hence the semantic digest) ONLY when explicitly given,
  # so every pre-existing config stays bit-identical.
  if 'warmup_draws' in ls:
   if int(ls['warmup_draws'])<0:raise ValueError('lr_schedule.warmup_draws must be >=0')
   c['lr_schedule']['warmup_draws']=int(ls['warmup_draws'])
 else:c['lr_schedule']=None
 # --- ST-RAS additions (20260925; defaults reproduce the historical behaviour exactly) ---
 from ras import normalize_sampler_config
 c['sampler']=normalize_sampler_config(c.get('sampler'))
 sparse_pixels=int(c['data'].get('spatial_samples_per_frame',0))
 if sparse_pixels:
  if sparse_pixels<1 or sparse_pixels>201*201:raise ValueError('invalid data.spatial_samples_per_frame')
  if c['sampler']['kind']!='st_ras' or not c['sampler']['space'] or not c['sampler']['importance_correct']:
   raise ValueError('sparse spatial supervision requires ST-RAS space=true and importance_correct=true')
 c['data']['spatial_samples_per_frame']=sparse_pixels
 # --- physics loss (20260927; absent/disabled reproduces the historical objective exactly) ---
 from physics_loss import normalize_physics_config
 c['physics']=normalize_physics_config(c.get('physics'))
 # Dynamic LR stability controller (disabled by default; enabled configs are semantic)
 from stability_controller import normalize_stability_controller
 c['stability_controller']=normalize_stability_controller((c.get('runtime') or {}).get('stability_controller'))
 pf=dict(c['data'].get('prefetch') or {})
 kind=pf.get('kind','thread')
 if kind not in ('thread','mp'):raise ValueError("data.prefetch.kind must be 'thread' or 'mp'")
 c['data']['prefetch']={'kind':kind,'num_workers':int(pf.get('num_workers',8)),'depth':int(pf.get('depth',4 if kind=='mp' else 2))}
 if c['data']['prefetch']['num_workers']<1 or c['data']['prefetch']['depth']<1:raise ValueError('prefetch num_workers/depth must be >=1')
 return c

def semantic_config(c,pool):
 source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'snapshot').rglob('*.py'))}
 source_hashes.update({n:sha(ROOT/n) for n in ['scno.py','model.py','data.py','base.py','train.py','train_ddp.py','checkpoint.py','ras.py','spatial_importance.py']})
 # ST-RAS (20260925): kind=legacy keeps the historical sampler string and adds NO new keys, but the
 # code_hashes above change with any edit, so a legacy digest is only comparable within one tree state.
 legacy_sampler=f"{'nine' if len(pool.cells)==9 else str(len(pool.cells))+'(families='+','.join(pool.families)+')'}-cell-roundrobin;sortedids;cell-cycle-seedpermutation;uniform-exact{c['data']['frames_per_update']}strata-v1"
 s=c['sampler']
 # Curriculum (20260928): the curriculum config already rides in the sampler dict; when
 # present, additionally bind the implementation hash (same precedent as physics below --
 # curriculum-free configs add NO key, so their digests are unchanged).
 ras={} if s['kind']=='legacy' else {'sampler_ras':dict(s),**({'curriculum_code_sha256':sha(ROOT/'curriculum.py')} if 'curriculum' in s else {})}
 # physics loss (20260927): disabled adds NO key, so every pre-existing digest is unchanged;
 # enabled binds the config AND the implementation hash (physics_loss.py is not in the
 # legacy code_hashes list, deliberately -- adding it there would shift every legacy digest).
 phys={} if not c['physics']['enabled'] else {'physics':{**c['physics'],'code_sha256':sha(ROOT/'physics_loss.py')}}
 stab={} if not c.get('stability_controller',{}).get('enabled') else {'stability_controller':{**c['stability_controller'],'code_sha256':sha(ROOT/'stability_controller.py')}}
 # v17 spectral carrier (20260928): a spec without an enabled carrier section adds NO
 # key, so every pre-existing digest is unchanged; an enabled carrier binds the two
 # implementation hashes (carrier.py is under active repair by its owner -- any fix is
 # a new experiment identity, which is exactly what the digest must express).
 mspec=json.loads(Path(c['model_spec']).read_text())
 carrier_sem={} if not (mspec.get('carrier') or {}).get('enabled') else {'carrier_code_sha256':{'carrier.py':sha(ROOT/'carrier.py'),'carrier_wiring.py':sha(ROOT/'carrier_wiring.py')}}
 sampler_string=legacy_sampler if s['kind']=='legacy' else legacy_sampler.replace('cell-cycle-seedpermutation','st-ras-residual-adaptive').replace(f"uniform-exact{c['data']['frames_per_update']}strata-v1",f"st-ras-{c['data']['frames_per_update']}strata-multinomial-v1")
 return {'schema':'homogeneous_scno_longtrain_v1','lift_mode':c['lift_mode'],'model_spec':mspec,'cycle':c.get('cycle'),'optimizer':c['optimizer'],'loss':c['loss'],'seed':c['seed'],'sampler':sampler_string,**ras,**phys,**stab,**carrier_sem,'spatial_samples_per_frame':c['data']['spatial_samples_per_frame'],'sampler_draw_offset':c['sampler_draw_offset'],'train_time_block':c['train_time_block'],'lr_schedule':c['lr_schedule'],**({'phase_cache':{'dir':str(Path(c['data']['phase_cache_dir']).resolve()),'index_sha256':sha(Path(c['data']['phase_cache_dir'])/'INDEX.json')}} if c['data'].get('phase_cache_dir') else {}),'manifest_digest':pool.manifest.digest,'pool_digest':pool.digest,'normalizer_sha256':sha(c['data']['normalization_json']),'energy_identity':pool.energy_identity,'dev_panel_sha256':sha(c['data']['dev_panel_json']),'code_hashes':source_hashes,'fork_identity_sha256':c['data'].get('fork_identity_sha256'),'precision':'float32TF32off,scnoP-scopedcudnndeterministic','initial_checkpoint_excluded':'init lineage saved separately;resume uses its own weights','runtime_fields_excluded':['max_updates','max_seconds','device','run_dir','log_every','eval_every','checkpoint_every','checkpoint_seconds','checkpoint_keep_latest','data.prefetch']}

def initialize_weights(m,init_path,c):
 # Single dispatch point for both trainers (and both their dry runs): init_partial=true
 # selects the v17 filtered transplant, anything else keeps the strict historical path.
 if c['init_partial']:return init_filtered_transplant(m,init_path,c['lift_mode'])
 return initialize_model(m,init_path,c['lift_mode'])

class EvaluationInterrupted(Exception):pass

class Prefetcher:
 """Single worker thread running Pool.batch_cpu one draw ahead of the GPU.
 Deterministic: draws are planned by index (per-draw seeds, no global RNG),
 delivered strictly in order; the thread only overlaps H5 read + encode with
 the previous step's GPU work. Any worker exception re-raises on the trainer
 thread at the matching draw.
 ST-RAS (20260925): start_iter() accepts an iterable of work items (ints or
 pool.decide() dicts). Decision dicts are produced lazily when the worker pulls
 the iterator, so an st_ras decision sees the tables at pull time (lag <= depth
 in-flight draws; each decision logs ras_decision_step for auditing).
 start(first,last) keeps the historical int path bitwise unchanged."""
 def __init__(self,pool,depth=2):
  import queue as _q,threading as _t
  self.pool=pool;self.q=_q.Queue(maxsize=depth);self.stop=_t.Event();self.expect=None;self.thread=None;self._q=_q;self._t=_t
 def start(self,first_draw,last_draw):
  self.start_iter(range(first_draw,last_draw+1),first_draw)
 def start_iter(self,items,first_draw):
  self.expect=first_draw
  def work():
   for it in items:
    if self.stop.is_set():return
    d=it if isinstance(it,int) else it['draw_index']
    try:item=(d,self.pool.batch_cpu(it),None)
    except BaseException as e:item=(d,None,e)
    while not self.stop.is_set():
     try:self.q.put(item,timeout=1.0);break
     except self._q.Full:continue
    if item[2] is not None:return
  self.thread=self._t.Thread(target=work,daemon=True);self.thread.start()
 def next(self,draw_index,device):
  assert draw_index==self.expect,'prefetch order drift'
  d,payload,err=self.q.get()
  assert d==draw_index,'prefetch draw mismatch'
  if err is not None:raise err
  self.expect=draw_index+1
  c,times,target,weights,E,draw=payload
  return c,times,target.to(device,non_blocking=False),weights.to(device),E.to(device),draw
 def shutdown(self):
  self.stop.set()
  if self.thread is not None:
   try:
    while True:self.q.get_nowait()
   except self._q.Empty:pass
   self.thread.join(timeout=10)

def make_prefetcher(pool,c,stride_hint=1):
 """data.prefetch.kind: 'thread' (this file) or 'mp' (prefetch_mp.MPPrefetch, multi-process pool).
 Both consume ints or decision dicts and yield the batch_cpu tuple in submission order."""
 pf=c['data']['prefetch']
 if pf['kind']=='thread':return Prefetcher(pool,depth=pf['depth'])
 from prefetch_mp import MPPrefetch
 return MPPrefetch(pool,num_workers=pf['num_workers'],depth=pf['depth'])

def build_sampler(c,pool,device='cpu'):
 if c['sampler']['kind']=='legacy':return None
 from ras import STRASTables
 # n_time is only needed by intra_stratum='residual' (the frame-level table Rf, indexed by
 # absolute frame index on the shared saved axis). Passing 0 otherwise leaves Rf absent, so
 # uniform arms keep their checkpoint contents and table shapes unchanged.
 n_time=pool.n_time if c['sampler'].get('intra_stratum')=='residual' else 0
 return STRASTables(c['sampler'],len(pool.roster),int(c['data']['frames_per_update']),device=device,n_time=n_time)

def transplant_optimizer(opt,mapping,init_path,lineage,c):
 """Adam-state transplant from the init (fork) checkpoint, matched by parameter name+shape; names the fork lacks (new modules) start fresh."""
 ck=torch.load(init_path,map_location='cpu',weights_only=False)
 def by_name(pmap,groups):return {n:i for g in groups for n,i in zip([m['name'] for m in pmap if m['group']==g['group_name']],g['params'],strict=True)}
 src=by_name(ck['parameter_map'],ck['optimizer_state']['param_groups']);cur=opt.state_dict();dst=by_name(mapping,cur['param_groups']);src_shape={m['name']:m['shape'] for m in ck['parameter_map']}
 carried=[m['name'] for m in mapping if m['name'] in src and src_shape[m['name']]==m['shape'] and src[m['name']] in ck['optimizer_state']['state']];fresh=[m['name'] for m in mapping if m['name'] not in set(carried)]
 if not carried:
  lineage['optimizer_transplant']={'attempted':True,'loaded':False,'reason':'no fork parameter matches by name and shape;fresh optimizer states'};del ck;return False
 opt.load_state_dict({'state':{dst[n]:ck['optimizer_state']['state'][src[n]] for n in carried},'param_groups':cur['param_groups']})
 # Transplant carries Adam moments only; hyperparameters always come from THIS
 # config (the fork may have been saved mid-LR-schedule, e.g. armB at scale 0.1).
 lr=c['optimizer']['learning_rates']
 for g in opt.param_groups:g['lr']=lr[g['group_name']];g['weight_decay']=c['optimizer']['weight_decay'];g['betas']=tuple(c['optimizer']['betas']);g['eps']=c['optimizer']['eps']
 lineage['optimizer_restored']=True
 lineage['optimizer_transplant']={'attempted':True,'loaded':True,'source_checkpoint':str(init_path),'source_updates':ck.get('updates_completed'),'state_entries':len(carried),'exact_parameter_map_match':ck.get('parameter_map')==mapping,'fresh_state_count':len(fresh),'fresh_state':fresh}
 del ck;return True

def evaluate(model,pool,run_dir,completed,device,dist_ctx=None):
 """Dev-panel evaluation.  dist_ctx=None (single process, original code path) or a
 (rank, world, torch.distributed) triple: records are sharded rank r -> pool.dev[r::world],
 every rank runs its shard on its own GPU, rows are gathered and re-ordered to the
 original panel order on rank 0, which alone writes EVALUATION.json and returns the
 aggregate (other ranks return None).  Per-record math is unchanged (same code, same
 kernels, deterministic flags), so a sharded panel is record-for-record identical to
 the serial one; only wall time changes (9 records / world instead of 9)."""
 rank,world,dist_mod=(0,1,None) if dist_ctx is None else dist_ctx
 is_main=rank==0
 if torch.cuda.is_initialized():torch.cuda.empty_cache()
 model.eval();rows=[];parent=run_dir/'evaluations'/f'update_{completed:08d}'
 if is_main:parent.mkdir(parents=True,exist_ok=True)
 # V18-A deployed contract (TASK_EVAL 20260928): a config that declares
 # evaluation.deployed_contract=v18a_base_only routes this dev evaluation -- the
 # checkpoint-selection signal -- through deployed_forward (no refiner, no family
 # binding, no offline-cache inputs), so selection and formal evaluation run the
 # SAME operator. Configs without the key keep the historical path bitwise.
 _dep=(pool.config.get('evaluation') or {}).get('deployed_contract');_dfw=None
 if _dep is not None:
  if _dep!='v18a_base_only':raise ValueError(f'unknown evaluation.deployed_contract {_dep!r}')
  import deployed_forward as _dfw_mod;_dfw=_dfw_mod;_dfw.assert_contract(model,pool.config)
 import hashlib
 destination=None;cached=None
 if is_main:
  hh=hashlib.sha256()
  for n,v in sorted(model.state_dict().items()):hh.update(n.encode());hh.update(v.detach().cpu().numpy().tobytes())
  evaluation_identity=digest({'model_state_sha256':hh.hexdigest(),'depth':int(model.scno_depth) if hasattr(model,'scno_cycle') else None,'lift_mode':model.scno_cycle.lift_mode if hasattr(model,'scno_cycle') else 'no_cycle','manifest':pool.manifest.digest,'dev_panel_sha256':sha(pool.config['data']['dev_panel_json']),'normalizer_sha256':sha(pool.config['data']['normalization_json']),'policy_code_sha256':sha(__file__),**({} if _dfw is None else _dfw.identity_extras(pool.config))})
  for prior in sorted(parent.glob('attempt_*/EVALUATION.json')):
   old=json.loads(prior.read_text())
   if old.get('evaluation_identity')==evaluation_identity:cached=old['aggregate'];break
  if cached is None:
   attempt=1
   while (parent/f'attempt_{attempt:03d}').exists():attempt+=1
   destination=parent/f'attempt_{attempt:03d}';destination.mkdir()
 if dist_mod is not None:
  sync=[cached is not None,str(destination) if destination is not None else None]
  dist_mod.broadcast_object_list(sync,src=0);skip,dest_s=sync
  if skip:return cached if is_main else None
  destination=Path(dest_s)
 elif cached is not None:return cached
 shard=list(pool.dev)[rank::world]
 for spec in shard:
  if STOP['signal'] is not None:raise EvaluationInterrupted()
  c=pool.inputs(spec);indices=np.arange(c['future_start'],len(c['time_axis']));pred=[]
  with torch.no_grad():
   # Evaluation may reuse preparation within this record; no training activations cached.
   from base import prepared_state
   # Contract mode: family/caches never bound (deployed_prepare); legacy: bitwise historical path.
   state=_dfw.deployed_prepare(model,c,pool.normalizer,device) if _dfw is not None else prepared_state(model,c,pool.normalizer,device)
   for k in range(0,len(indices),16):
    ts=c['time_axis'][indices[k:k+16]][None].to(device);p=model.dense_normalized(state,ts,x_m=c['x'].to(device),z_m=c['z'].to(device),time_block=1);pred.append(pool.normalizer.decode_pressure(p,c['source'][:,4].to(device))[0].cpu().numpy())
  p=np.concatenate(pred);np.save(destination/f"{c['sample_id']}_prediction.npy",p)
  truth=pool.records.read_wavefield(c['local_index'],c['time_axis']);assert bool(truth.exact.all());y=truth.values.numpy();full=y.copy();full[indices]=p;metrics=eval_metrics.evaluate(full,y,np.asarray(pool.manifest.time_s),np.asarray(pool.manifest.x_m),np.asarray(pool.manifest.z_m),c['future_start']-8);rows.append({'sample_id':c['sample_id'],'family':c['family'],'bin':spec['bin'],'metrics':metrics})
 if dist_mod is not None:
  gathered=[None]*world
  dist_mod.all_gather_object(gathered,rows)
  if not is_main:return None
  order={s['sample_id']:i for i,s in enumerate(pool.dev)}
  rows=sorted((r for part in gathered for r in part),key=lambda r:order[r['sample_id']])
 result={'evaluation_identity':evaluation_identity,'updates_completed':completed,'scope':'registered train-development monitoring only,no validation/test',**({} if _dep is None else {'deployed_contract':_dep}),'records':rows,'aggregate':eval_metrics.aggregate(rows)};atomic_json(destination/'EVALUATION.json',result);return result['aggregate']

def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run-dir',required=True);parser.add_argument('--device',default='cuda:0');parser.add_argument('--dry-run',action='store_true');parser.add_argument('--prepare-data',action='store_true');parser.add_argument('--max-updates',type=int);parser.add_argument('--max-seconds',type=float);g=parser.add_mutually_exclusive_group();g.add_argument('--init-checkpoint');g.add_argument('--resume');args=parser.parse_args(argv)
 if args.dry_run and args.prepare_data:parser.error('dry-run cannot compute label energies')
 c=resolved_config(args.config);run_dir=Path(args.run_dir).resolve();pool=Pool(c);sem=semantic_config(c,pool);sem_digest=digest(sem);maximum=int(args.max_updates if args.max_updates is not None else c['runtime']['max_updates']);seconds=args.max_seconds if args.max_seconds is not None else c['runtime'].get('max_seconds');assert maximum>0
 def resolve_checkpoint(path):
  path=Path(path).resolve()
  if path.suffix=='.json':path=Path(json.loads(path.read_text())['checkpoint']).resolve()
  return str(path)
 init_path=args.init_checkpoint or c.get('initial_checkpoint');saved=None
 if init_path:init_path=resolve_checkpoint(init_path)
 if args.resume:args.resume=resolve_checkpoint(args.resume)
 if args.resume:
  saved=torch.load(args.resume,map_location='cpu',weights_only=False);check_resume(saved,c['lift_mode'],sem_digest,pool.digest)
  assert saved['manifest_digest']==pool.manifest.digest and saved['normalizer_sha256']==sha(c['data']['normalization_json']) and saved['next_depth']==(None if c['lift_mode']=='no_cycle' else depth_for(c,saved['next_update']))
  if maximum<saved['updates_completed']:raise ValueError('new budget is below completed updates')
 elif not init_path:raise ValueError('fresh run requires --init-checkpoint or initial_checkpoint in config')
 if args.dry_run:
  model=build(c,'cpu');lineage=None
  if saved:model.load_state_dict(saved['model_state'],strict=True)
  else:lineage=initialize_weights(model,init_path,c);lineage['sampler_start_draw']=c['sampler_draw_offset']
  opt,mapping=optimizer(model,c)
  if saved:
   assert mapping==saved['parameter_map'];opt.load_state_dict(saved['optimizer_state'])
  elif c['init_optimizer_transplant']:transplant_optimizer(opt,mapping,init_path,lineage,c)
  summary={'status':'DRY_RUN_PASS','GPU_initialized':torch.cuda.is_initialized(),'optimizer_steps_executed':0,'wavefield_labels_read':False,'mode':c['lift_mode'],'pool':pool.metadata_summary(),'parameters':sum(p.numel() for p in model.parameters()),'parameter_tensors':len(mapping),'all_trainable':all(p.requires_grad for p in model.parameters()),'optimizer_unique_complete':True,'semantic_config_digest':sem_digest,'resume_completed':saved['updates_completed'] if saved else None,'max_updates':maximum,'frames_per_update':c['data']['frames_per_update'],'train_time_block':c['train_time_block'],'sampler_draw_offset':c['sampler_draw_offset'],'sampler':c['sampler'],'physics':c['physics'],'prefetch':c['data']['prefetch'],'optimizer_transplant':(lineage or {}).get('optimizer_transplant'),'lineage':lineage};print(json.dumps(summary,indent=2));pool.close();return 0
 if args.prepare_data:
  run_dir.mkdir(parents=True,exist_ok=True);pool.attach_energy_cache(run_dir/'energy_cache.json')
  for i,row in enumerate(pool.roster):
   pool.energy(pool.inputs(row))
   if (i+1)%50==0:print(json.dumps({'event':'energy_cache','records':i+1,'total':len(pool.roster)}),flush=True)
  atomic_json(run_dir/'DATA_PREPARATION.json',{'status':'completed','records':len(pool.energy_values),'identity':pool.energy_identity,'optimizer_steps':0});pool.close();return 0
 device=torch.device(args.device)
 if device.type!='cuda':raise ValueError('actual large-model training requires explicit CUDA device;dry-run is CPU')
 if shutil.disk_usage(run_dir.parent if run_dir.parent.exists() else ROOT).free<c['runtime']['min_free_disk_gib']*1024**3:raise RuntimeError('insufficient free disk')
 if run_dir.exists() and (run_dir/'run_identity.json').exists() and not args.resume:raise FileExistsError('run-dir already contains training identity;use --resume or a new run-dir')
 run_dir.mkdir(parents=True,exist_ok=True)
 run_lock=(run_dir/'run.lock').open('a')
 try:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('this run-dir already has an active trainer')
 torch.set_num_threads(c['runtime']['cpu_threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.cuda.set_device(device)
 free,_=torch.cuda.mem_get_info(device)
 if free<c['runtime']['min_free_vram_gib']*1024**3:raise RuntimeError('insufficient free VRAM;no automatic competing launch')
 torch.cuda.set_per_process_memory_fraction(c['runtime']['allocator_limit_gib']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
 m=build(c,device);opt,mapping=optimizer(m,c);pool.attach_energy_cache(run_dir/'energy_cache.json');completed=0;elapsed_before=0.;lineage=None;off=c['sampler_draw_offset'];tb=c['train_time_block'];lsch=c['lr_schedule'];base_lr={g['group_name']:c['optimizer']['learning_rates'][g['group_name']] for g in opt.param_groups}
 if c['physics']['enabled']:raise ValueError('physics loss is wired into the DDP trainer (train_ddp.py) only; this single-GPU loop does not compute it and silently ignoring the section would misreport the objective')
 if c['loss'].get('late_amp') or c['loss'].get('late_ncc'):raise ValueError('late_amp/late_ncc are wired into the DDP trainer (train_ddp.py) only; this single-GPU loop optimises the record loss and would silently drop them')
 sampler=build_sampler(c,pool);pool.attach_sampler(sampler)
 if saved:
  assert mapping==saved['parameter_map'];m.load_state_dict(saved['model_state'],strict=True);opt.load_state_dict(saved['optimizer_state']);completed=int(saved['updates_completed']);elapsed_before=float(saved['elapsed_runtime_seconds']);lineage=saved['lineage'];assert saved['sampler']['draw_index']==completed+off
  if sampler is not None:
   if 'sampler_tables' not in saved:raise ValueError('st_ras resume requires sampler_tables in the checkpoint')
   sampler.load_state_dict(saved['sampler_tables']);sampler.steps=completed
  for key,value in saved['energy_cache_values'].items():
   if key in pool.energy_values and pool.energy_values[key]!=value:raise ValueError('energy cache value drift')
   pool.energy_values[key]=value
  atomic_json(run_dir/'energy_cache.json',{'identity':pool.energy_identity,'values':pool.energy_values})
 else:
  lineage=initialize_weights(m,init_path,c);lineage['sampler_start_draw']=off
  if c['init_optimizer_transplant']:transplant_optimizer(opt,mapping,init_path,lineage,c)
 # Boundary ramp (20260929): sync the blend to the completed count after init/resume
 # (resume restores the buffer too, but the count is the single source of truth).
 m.set_free_surface_blend(completed)
 if c['lift_mode']!='no_cycle':m.scno_depth=depth_for(c,completed+1)
 identity={'schema':'homogeneous_scno_training_identity_v1','lift_mode':c['lift_mode'],'semantic_config_digest':sem_digest,'semantic_config':sem,'lineage':lineage,'pool':pool.metadata_summary(),'all_parameter_tensors':len(mapping),'all_parameter_numel':sum(p.numel() for p in m.parameters()),'device':str(device),'torch':torch.__version__,'cuda':torch.version.cuda,'validation_test_future_accessed':False,'checkpoint_policy':{'keep_latest':c['checkpoint_keep_latest'],'rolling_prefix':'checkpoint','protected_prefixes':['checkpoint_00000000','best_'],'checkpoint_every':c['runtime']['checkpoint_every'],'checkpoint_seconds':c['runtime']['checkpoint_seconds'],'note':'keep_latest=None means every rolling checkpoint is retained (pre-20260926 behaviour); best_* and the step-0 lineage checkpoint are never pruned'}}
 atomic_json(run_dir/'effective_config.json',{'research':sem,'runtime':c['runtime']|{'max_updates':maximum,'max_seconds':seconds,'device':str(device)}})
 identity_path=run_dir/'run_identity.json'
 if not identity_path.exists():atomic_json(identity_path,identity)
 else:
  old=json.loads(identity_path.read_text());assert old['semantic_config_digest']==sem_digest
 atomic_json(run_dir/'POOL.json',{'identity':pool.metadata_summary(),'roster':pool.roster,'dev':pool.dev});start=time.monotonic();last_save=start;last_good=str(Path(args.resume).resolve()) if args.resume else None
 def save(prefix='checkpoint'):
  nonlocal last_good,last_save
  payload={'model_state':m.state_dict(),'optimizer_state':opt.state_dict(),'rng_state':capture_rng(),'lift_mode':c['lift_mode'],'semantic_config_digest':sem_digest,'semantic_config':sem,'pool_digest':pool.digest,'manifest_digest':pool.manifest.digest,'normalizer_sha256':sha(c['data']['normalization_json']),'parameter_map':mapping,'updates_completed':completed,'next_update':completed+1,'next_depth':None if c['lift_mode']=='no_cycle' else depth_for(c,completed+1),'depth':getattr(m,'scno_depth',None),'cycle_depth_schedule':None if c['lift_mode']=='no_cycle' else c['cycle']['depth_schedule'],'sampler':{'algorithm':sem['sampler'],'seed':c['seed'],'pool_digest':pool.digest,'draw_offset':off,'draw_index':completed+off},**({'sampler_tables':sampler.state_dict()} if sampler is not None else {}),'energy_cache_values':pool.energy_values,'elapsed_runtime_seconds':elapsed_before+time.monotonic()-start,'lineage':lineage}
  path=run_dir/'checkpoints'/f'{prefix}_{completed:08d}.pt';path=atomic_checkpoint(path,payload);last_good=str(path);last_save=time.monotonic()
  # Retention runs only AFTER the new file is on disk; a no-op when the config omits the key.
  if c['checkpoint_keep_latest']:prune_checkpoints(path.parent,c['checkpoint_keep_latest'])
  return path
 monitor_path=run_dir/'best_monitor.json'
 def track_best(agg,completed):
  # dev9 train-development monitor only; never a generalization claim. Lower future_record_mean is better.
  state={'metric':'aggregate.all_records.future_record_mean','direction':'lower_is_better','scope':'registered train-dev panel(dev9);in-pool families;no validation/test','best_value':None,'best_checkpoint':None,'history':[]}
  if monitor_path.exists():state=json.loads(monitor_path.read_text())
  value=agg['all_records']['future_record_mean'];fam=agg.get('families',{}).get('marmousi',{})
  epoch=completed//c['runtime']['eval_every'] if c['runtime']['eval_every'] else None
  improved=value is not None and (state['best_value'] is None or value<state['best_value'])
  row={'updates_completed':completed,'epoch':epoch,'value':value,'marmousi_future_record_mean':fam.get('future_record_mean'),'marmousi_future_late_record_mean':fam.get('future_late_record_mean'),'improved':improved}
  if improved:
   path=save(prefix=f'best_e{epoch:04d}' if epoch is not None else 'best');state['best_value']=value;state['best_checkpoint']=str(path);row['checkpoint']=str(path)
  state['history'].append(row);atomic_json(monitor_path,state)
  if improved:print(json.dumps({'event':'best_checkpoint','updates_completed':completed,'value':value,'checkpoint':state['best_checkpoint']}),flush=True)
 signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
 # Restore RNG LAST, after model/optimizer/configuration validation and construction.
 if saved:restore_rng(saved['rng_state']);del saved
 if completed==0:save()
 status='completed';last_eval=-1
 prefetch=make_prefetcher(pool,c)
 if isinstance(prefetch,Prefetcher):
  # thread prefetcher: keep the historical int path for legacy (bitwise), decisions for st_ras
  if sampler is None:prefetch.start(completed+off,maximum+off-1)
  else:prefetch.start_iter(pool.decisions(completed+off,maximum+off-1),completed+off)
 else:prefetch.start(range(completed+off,maximum+off) if sampler is None else pool.decisions(completed+off,maximum+off-1))
 try:
  while completed<maximum:
   if STOP['signal'] is not None:status='interrupted';break
   if seconds is not None and elapsed_before+time.monotonic()-start>=seconds:status='budget_stopped';break
   if shutil.disk_usage(run_dir).free<c['runtime']['min_free_disk_gib']*1024**3:raise RuntimeError('free disk below checkpoint safety threshold')
   data,times,target,weights,E,draw=prefetch.next(completed+off,device)
   if STOP['signal'] is not None:status='interrupted';break
   if c['lift_mode']!='no_cycle':m.scno_depth=depth_for(c,completed+1)
   m.set_free_surface_blend(completed)
   m.train();opt.zero_grad(set_to_none=True);t0=time.monotonic();prediction=forward_live(m,data,times,pool.normalizer,device,time_block=tb)
   W=None
   if sampler is not None:
    W=sampler.spatial_weights(draw['roster_index'],draw['frame_strata'],target.shape[-2:])
    if W is not None:W=W.to(device)
   residual=(prediction-target)*weights.sqrt()[None,:,None,None]
   if W is not None:residual=residual*W.sqrt()[None]
   loss=torch.linalg.vector_norm(residual.reshape(-1))/E.sqrt()
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss before optimizer;last-good checkpoint retained')
   loss.backward()
   if lsch is not None:
    progress=min((completed+1)/lsch['total_draws'],1.0);lr_scale=lsch['final_scale']+(1.0-lsch['final_scale'])*0.5*(1.0+math.cos(math.pi*progress))
    wu=lsch.get('warmup_draws',0)
    if wu:lr_scale*=min(1.0,(completed+1)/wu)
    for g in opt.param_groups:g['lr']=base_lr[g['group_name']]*lr_scale
   # Per-group clipping, same semantics as train_ddp.py: trunk and local_unet.* clipped independently at the same threshold
   # (a single global clip would let the branch's larger norm scale the trunk's step). Without local_unet this is the legacy global clip.
   # dense_decoder.carrier_* (20260928) joins the branch group for the arrival_head reason: its output layer sums over every dense point of every frame.
   # dense_decoder.reflection_branch.* (20260929, UL31): zero-init interface head inside the dense decoder, clipped with the branch group for the same reason.
   branch_prefixes=('local_unet.','arrival_head.','scattering_head.','dense_decoder.carrier_lift.','dense_decoder.carrier_head.','dense_decoder.reflection_branch.');trunk_params=[p for n,p in m.named_parameters() if not n.startswith(branch_prefixes)];branch_params=[p for n,p in m.named_parameters() if n.startswith(branch_prefixes)]
   gn_trunk=torch.nn.utils.clip_grad_norm_(trunk_params,c['optimizer']['gradient_clip']);gn_branch=torch.nn.utils.clip_grad_norm_(branch_params,c['optimizer']['gradient_clip']) if branch_params else torch.zeros((),device=device);gn=torch.maximum(gn_trunk,gn_branch)
   if not torch.isfinite(gn):raise FloatingPointError('nonfinite gradient before optimizer;last-good checkpoint retained')
   opt.step();torch.cuda.synchronize();completed+=1
   ras_row={}
   if sampler is not None:
    # Table update from the already-computed error field (raw squared error, no loss weights).
    from ras import weight_stats
    with torch.no_grad():
     err2=(prediction.detach()-target).square()[0];t2=target.detach().double().square().sum(dim=(2,3))[0]
     upd=sampler.update(draw['roster_index'],draw['frame_strata'],err2,t2,float(E),len(data['time_axis'])-data['future_start'],draw['stratum_sizes'],draw['ras_alloc_q'],frames=draw['frames'])
    sampler.steps=completed
    ras_row={'ras_loss_unbiased':upd['loss_unbiased'],'ras_rho':upd['rho'],**({'ras_entropy_space_visited':upd['entropy_space']} if 'entropy_space' in upd else {}),**({f'sampler_{k}':v for k,v in sampler.table_entropies().items()}),**({} if W is None else weight_stats(W))}
   row={'event':'train_step','updates_completed':completed,'loss':float(loss.detach()),'gradient_norm':float(gn),'depth':getattr(m,'scno_depth',None),'free_surface_blend':(float(m.free_surface_blend) if getattr(m,'free_surface_mode',None)=='ramp_exact_row' else None),'seconds':time.monotonic()-t0,'time_block':tb,'max_mem_gib':round(torch.cuda.max_memory_allocated(device)/2**30,3),**({} if lsch is None else {'lr_scale':lr_scale}),**draw,**ras_row}
   with (run_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if completed%c['runtime']['log_every']==0 or completed==1:print(json.dumps(row),flush=True)
   del prediction,residual,loss,target
   due=completed%c['runtime']['checkpoint_every']==0 or time.monotonic()-last_save>=c['runtime']['checkpoint_seconds']
   if due or STOP['signal'] is not None:save()
   if STOP['signal'] is not None:status='interrupted';break
   if c['runtime']['eval_every'] and completed%c['runtime']['eval_every']==0:
    agg=evaluate(m,pool,run_dir,completed,device);print(json.dumps({'event':'train_dev_eval','updates_completed':completed,'metrics':agg}),flush=True);last_eval=completed;track_best(agg,completed)
  save()
  if status in ('completed','budget_stopped') and c['runtime']['eval_at_end'] and last_eval!=completed:track_best(evaluate(m,pool,run_dir,completed,device),completed)
  atomic_json(run_dir/'terminal.json',{'status':status,'updates_completed':completed,'requested_max_updates':maximum,'last_good_checkpoint':last_good,'signal':STOP['signal'],'elapsed_runtime_seconds':elapsed_before+time.monotonic()-start});return 0 if status!='interrupted' else 128+int(STOP['signal'])
 except EvaluationInterrupted:
  save();atomic_json(run_dir/'terminal.json',{'status':'interrupted','updates_completed':completed,'last_good_checkpoint':last_good,'signal':STOP['signal']});return 128+int(STOP['signal'] or signal.SIGTERM)
 except BaseException as e:
  # Never serialize possibly half-updated weights after an optimizer/OOM failure.
  atomic_json(run_dir/'terminal.json',{'status':'failed','updates_completed_counter':completed,'last_good_checkpoint':last_good,'error':repr(e),'traceback':traceback.format_exc()});raise
 finally:
  prefetch.shutdown();pool.close();fcntl.flock(run_lock,fcntl.LOCK_UN);run_lock.close()

if __name__=='__main__':raise SystemExit(main())
