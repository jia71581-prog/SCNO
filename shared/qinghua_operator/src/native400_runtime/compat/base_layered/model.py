"""Independent model factory; state schema alone never identifies lift algebra."""
import sys,json,hashlib,random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'snapshot'))
from scno import LearnedCycle
from grouped_ufno_mionet_v3.config import V3Config
from scripts.train_grouped_v3 import build_model

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_initialized():torch.cuda.manual_seed_all(seed)
def v3_config(config):
 # v17 spectral carrier (20260928): the spec's top-level "carrier" section is a
 # dev-tree extension consumed by carrier_wiring.py, not a V3Config field; popping it
 # here keeps the strict schema for every field V3Config does own. Specs without the
 # key are bitwise unchanged.
 spec=json.loads(Path(config['model_spec']).read_text());spec.pop('carrier',None);spec['data'].update(source_h5=config['data']['source_h5'],normalization_json=config['data']['normalization_json']);return V3Config.from_mapping(spec)
def carrier_spec(config):
 # The enabled "carrier" section of the model spec, or None. Single accessor so
 # build/semantic_config/tests agree on what counts as enabled.
 cc=json.loads(Path(config['model_spec']).read_text()).get('carrier')
 return cc if isinstance(cc,dict) and cc.get('enabled') else None
def build(config,device):
 seed_all(config['seed']);cfg=v3_config(config)
 # Activation-recompute policy, read from the spec instead of hardcoded.  The
 # defaults preserve the previous behaviour exactly (both on) for every caller
 # that does not set them: eval_panel/identity_check/probe_* all pass a config
 # whose spec carries the two keys with their old values, and V3Config's own
 # defaults are dense_checkpoint=True / dense_outer_checkpoint=False.
 #   dense_checkpoint       -- inner per-spectral-block recompute
 #   dense_outer_checkpoint -- outer per-time-step recompute (bounds peak by one
 #                             time step; needed to fit 32-frame windows)
 m=build_model(cfg,dense_checkpoint=bool(cfg.data.dense_checkpoint),
               dense_outer_checkpoint=bool(cfg.data.dense_outer_checkpoint))
 torch.manual_seed(config['seed'])
 if config['lift_mode']!='no_cycle':m.add_module('scno_cycle',LearnedCycle(config['cycle']['width'],config['cycle']['levels'],lift_mode=config['lift_mode']));m.scno_depth=1
 # v17 spectral carrier (20260928): swap the dense decoder for the carrier-augmented
 # subclass AFTER the cycle attachment, so the legacy RNG consumption order (and hence
 # every carrier-less build) stays bitwise. The swap copies the old decoder's tensors,
 # so the base initialisation is preserved; only the four zero-init carrier keys are new.
 cc=carrier_spec(config)
 if cc is not None:
  from carrier_wiring import attach_carrier;attach_carrier(m,cc,cfg)
 m.requires_grad_(True);return m.to(device)
def _load_inherited(m,state,fresh):
 # strict=False is required because a branch added after this checkpoint has no
 # keys in the parent state. Guard it: every key the parent lacks must be one we
 # already classified as fresh, so nothing else silently skips loading.
 missing,unexpected=m.load_state_dict(
  {n:v for n,v in state.items() if n in set(m.state_dict())},strict=False)
 if sorted(missing)!=sorted(fresh):raise ValueError(f'unexpected missing keys {sorted(missing)} vs declared fresh {sorted(fresh)}')
 if unexpected:raise ValueError(f'checkpoint carries keys the model does not use: {sorted(unexpected)}')
 return missing
def initialize_model(m,path,mode):
 ck=torch.load(path,map_location='cpu',weights_only=False);state=ck.get('model_state')
 if state is None:raise ValueError('init checkpoint must contain model_state')
 has_cycle=any(n.startswith('scno_cycle.') for n in state)
 # A branch added after this checkpoint was written (local_unet) has no keys in
 # the parent state. Its output layer is zero-initialised, so leaving it at init
 # keeps the forked model bitwise equal to the parent on the first forward.
 own=set(m.state_dict())
 fresh=sorted(own-set(state))
 if mode=='no_cycle':
  if hasattr(m,'scno_cycle'):raise ValueError('no_cycle model must not carry a cycle module')
  dropped=sorted(n for n in state if n.startswith('scno_cycle.'))
  _load_inherited(m,{n:v for n,v in state.items() if not n.startswith('scno_cycle.')},fresh)
  return {'checkpoint':str(Path(path).resolve()),'sha256':sha(path),'source_lift_mode':ck.get('lift_mode'),'target_lift_mode':mode,'mode_transfer':False,'cycle_parameters_dropped':dropped,'cycle_parameters_dropped_count':len(dropped),'optimizer_restored':False,'sampler_reset_to':0,'source_updates':ck.get('updates_completed',ck.get('global_step')),'fresh_parameters':fresh,'fresh_parameter_count':len(fresh)}
 if has_cycle:
  if ck.get('lift_mode') not in ['additive','homogeneous']:raise ValueError('cycle checkpoint lacks explicit lift_mode; refuse ambiguous transfer')
  _load_inherited(m,state,fresh)
 else:
  cycle=m.scno_cycle;delattr(m,'scno_cycle');_load_inherited(m,state,sorted(set(m.state_dict())-set(state)));m.add_module('scno_cycle',cycle)
 return {'checkpoint':str(Path(path).resolve()),'sha256':sha(path),'source_lift_mode':ck.get('lift_mode'),'target_lift_mode':mode,'mode_transfer':has_cycle and ck['lift_mode']!=mode,'optimizer_restored':False,'sampler_reset_to':0,'source_updates':ck.get('updates_completed',ck.get('global_step')),'fresh_parameters':fresh,'fresh_parameter_count':len(fresh)}
# v17 filtered transplant (20260928): keys the checkpoint carries but the model does
# not (deleted heads) would make _load_inherited raise. This path loads the NAME
# INTERSECTION instead -- but only when every dropped key belongs to a module that was
# deliberately removed, so a typo'd spec can never silently discard trunk weights.
# v18 (20260928): 'refiner.' joins the prunable set. The refiner is an optional trainer
# attachment (attach_refiner runs BEFORE initialize_weights in both trainers), so when
# the config carries an enabled refiner section the keys are in the model and load
# normally; only a deliberately refiner-less config drops them. It was never part of
# the trunk, so the typo-protection argument is unchanged.
INIT_PARTIAL_DROP_PREFIXES=('arrival_head.','scattering_head.','travel_branch.phase_projection.','refiner.')
def init_filtered_transplant(m,path,mode):
 ck=torch.load(path,map_location='cpu',weights_only=False);state=ck.get('model_state')
 if state is None:raise ValueError('init checkpoint must contain model_state')
 if mode=='no_cycle':raise ValueError('init_partial supports cycle lift modes only; no_cycle keeps its dedicated path in initialize_model')
 if not hasattr(m,'scno_cycle'):raise ValueError('cycle lift mode requires the model to carry scno_cycle')
 has_cycle=any(n.startswith('scno_cycle.') for n in state)
 if has_cycle and ck.get('lift_mode') not in ['additive','homogeneous']:raise ValueError('cycle checkpoint lacks explicit lift_mode; refuse ambiguous transfer')
 own=m.state_dict()
 dropped=sorted(n for n in state if n not in own)
 illegal=[n for n in dropped if not n.startswith(INIT_PARTIAL_DROP_PREFIXES)]
 if illegal:raise ValueError(f'init_partial refuses to drop keys outside the declared prunable modules {INIT_PARTIAL_DROP_PREFIXES}: {illegal}')
 mismatch=[n for n,v in state.items() if n in own and tuple(v.shape)!=tuple(own[n].shape)]
 if mismatch:raise ValueError(f'shape mismatch on shared keys: {mismatch}')
 fresh=sorted(set(own)-set(state))
 missing,unexpected=m.load_state_dict({n:v for n,v in state.items() if n in own},strict=False)
 if sorted(missing)!=fresh or unexpected:raise ValueError(f'filtered transplant drift: missing {sorted(missing)} vs fresh {fresh}; unexpected {sorted(unexpected)}')
 return {'checkpoint':str(Path(path).resolve()),'sha256':sha(path),'init_mode':'init_filtered_transplant','source_lift_mode':ck.get('lift_mode'),'target_lift_mode':mode,'mode_transfer':has_cycle and ck.get('lift_mode')!=mode,'dropped_keys':dropped,'dropped_key_count':len(dropped),'optimizer_restored':False,'sampler_reset_to':0,'source_updates':ck.get('updates_completed',ck.get('global_step')),'fresh_parameters':fresh,'fresh_parameter_count':len(fresh)}
LOCAL_MULTI_PARAMETER_MARKERS = (
 'local_gate2', 'local_gate4', 'local_dw2.', 'local_dw4.',
)
def is_local_multi_parameter_name(name):
 return name.startswith('dense_decoder.blocks.') and any(m in name for m in LOCAL_MULTI_PARAMETER_MARKERS)

def optimizer(m,config):
 lr=config['optimizer']['learning_rates']
 # `scatter` is an OPTIONAL extra group (20260926) for the Born scattering head only.
 # That head's term is BILINEAR in its own two output maps, so its output-layer gradient
 # is several orders of magnitude larger than an additive head's: measured |grad| ~5e10,
 # against the 1e-3 rate every other new module uses, i.e. a ~5e7 step. That is exactly
 # how the first fix attempt blew up (head term drove the loss to 9.9e6 within 10 steps,
 # after which the optimiser switched the head back off). Being multiplicative rather
 # than additive is what makes the inherited rate wrong here. When `scatter` is absent
 # from the config the head falls back to `scno` and behaviour is bitwise unchanged, so
 # every existing arm and checkpoint is unaffected.
 kinds=['parent','correction_scale'] if config['lift_mode']=='no_cycle' else ['parent','correction_scale','scno']
 local_enabled=any(is_local_multi_parameter_name(n) for n,_ in m.named_parameters())
 if local_enabled:
  if 'local' not in lr:raise ValueError('local_multi_dilation parameters require optimizer.learning_rates.local')
  kinds=kinds+['local']
 elif 'local' in lr:
  raise ValueError('optimizer.learning_rates.local is only legal when local_multi_dilation parameters exist')
 upath_enabled=any(n.startswith('dense_decoder.u_path.') for n,_ in m.named_parameters())
 if upath_enabled:
  if 'upath' not in lr:raise ValueError('u_path parameters require optimizer.learning_rates.upath')
  kinds=kinds+['upath']
 elif 'upath' in lr:
  raise ValueError('optimizer.learning_rates.upath is only legal when u_path is enabled')
 physicsfd_enabled=any(n.startswith('dense_decoder.fixed_wave_physics.') for n,_ in m.named_parameters())
 if physicsfd_enabled:
  if 'physicsfd' not in lr:raise ValueError('fixed_wave_physics parameters require optimizer.learning_rates.physicsfd')
  kinds=kinds+['physicsfd']
 elif 'physicsfd' in lr:
  raise ValueError('optimizer.learning_rates.physicsfd is only legal when fixed_wave_physics is enabled')
 if 'scatter' in lr and config['lift_mode']!='no_cycle':kinds=kinds+['scatter']
 # v17 spectral carrier (20260928): OPTIONAL dedicated group for the two zero-init
 # carrier convolutions, same pattern as `scatter`. Absent from the config they fall
 # back to `scno` (the fast rate every other new module uses); both terms are LINEAR
 # in their own weights with external carrier fields as the other factor, so unlike
 # the bilinear scattering head the additive-head rate class is appropriate.
 if 'carrier' in lr and config['lift_mode']!='no_cycle':kinds=kinds+['carrier']
 # Iterative residual refiner (2026-09-27): an OPTIONAL pair of extra groups.  The
 # scalar gate `s_raw` gets its OWN group with its own lr, because V2 measured that
 # sharing the conv lr drove s monotonically to 0.019 and switched the refiner off.
 rf=config.get('refiner') or {}
 if rf.get('enabled'):
  kinds=kinds+['refiner','refiner_s']
 if sorted(lr)!=sorted(kinds):raise ValueError(f'learning_rates keys {sorted(lr)} must exactly match the parameter groups {sorted(kinds)} of lift_mode {config["lift_mode"]!r}')
 groups={k:[] for k in kinds};mapping=[]
 for n,p in m.named_parameters():
  if n=='refiner.s_raw' and 'refiner_s' in groups:
   kind='refiner_s'
  elif n.startswith('refiner.') and 'refiner' in groups:
   kind='refiner'
  elif is_local_multi_parameter_name(n):
   kind='local'
  elif n.startswith('dense_decoder.u_path.'):
   kind='upath'
  elif n.startswith('dense_decoder.fixed_wave_physics.'):
   kind='physicsfd'
  elif n.startswith('scattering_head.') and 'scatter' in groups:
   kind='scatter'
  elif n.startswith(('dense_decoder.carrier_lift.','dense_decoder.carrier_head.')):
   # v17 carrier convolutions: their own group when the config declares one,
   # otherwise the fast new-module rate (see the kinds comment above).
   kind='carrier' if 'carrier' in groups else 'scno'
  elif n.startswith('local_unet.') or n.startswith('travel_branch.phase_projection.') or n.startswith('arrival_head.') or n.startswith('coarse_channel.') or n.startswith('scattering_head.') or n.startswith('medium_encoder.u_enhance.') or n.startswith('dense_decoder.reflection_branch.'):
   # Newly initialised modules (local spatial path; P2 phase projection; multi-arrival
   # additive head; coarse-solver input channel; v18.5 medium U-enhancement; UL31
   # reflection branch) join the
   # 'scno' group, which carries the fast rate reserved for new modules; the parent
   # rate (1e-5) would leave them effectively frozen.
   kind='scno'
  else:
   kind='scno' if n.startswith('scno_cycle.') else 'correction_scale' if n=='dense_decoder.correction_scale' else 'parent'
  if kind not in groups:raise ValueError('no_cycle optimizer refuses cycle parameters')
  groups[kind].append(p);mapping.append({'name':n,'group':kind,'shape':list(p.shape),'dtype':str(p.dtype),'numel':p.numel(),'requires_grad':p.requires_grad})
 opt=torch.optim.AdamW([{'params':v,'lr':lr[k],'group_name':k} for k,v in groups.items()],weight_decay=config['optimizer']['weight_decay'],betas=tuple(config['optimizer']['betas']),eps=config['optimizer']['eps']);ids=[id(p) for g in opt.param_groups for p in g['params']];assert len(ids)==len(set(ids))==len(list(m.parameters())) and set(ids)=={id(p) for p in m.parameters()} and all(p.requires_grad for p in m.parameters());return opt,mapping

def depth_for(config,next_update):
 for entry in config['cycle']['depth_schedule']:
  if next_update>=entry['start'] and (entry.get('end') is None or next_update<=entry['end']):return int(entry['depth'])
 raise ValueError('depth schedule does not cover next update')
