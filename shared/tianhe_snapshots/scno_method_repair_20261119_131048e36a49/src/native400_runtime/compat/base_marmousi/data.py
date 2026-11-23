"""Metadata-bound lazy HDF5 training data; never caches model activations."""
import json,hashlib,os,scno_posix_lock as fcntl
from pathlib import Path
import numpy as np
import h5py
import torch
from model import digest,sha,v3_config
import scatter_binding
from checkpoint import atomic_json
from grouped_ufno_mionet_v3.data.index import build_manifest
from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset
from scripts.train_grouped_v3_ic8_fullfield import load_normalizer
class Pool:
 def __init__(self,config):
  self.config=config;self.manifest=build_manifest(config['data']['source_h5']);self.normalizer=self._normalizer(config);self.records=V3WavefieldDataset(config['data']['source_h5'],self.manifest,split='train');meta=[r for r in self.manifest.records if r.split=='train'];self.byid={r.sample_id:(i,r) for i,r in enumerate(meta)};assert len(self.byid)==len(meta),'duplicate train sample_id'
  excluded=set(json.loads(Path(config['data']['exclusions_json']).read_text())['excluded_group_ids']);bins=config['data']['frequency_bins']
  # Frequency scope (20260927): the historical full-band pool binds [8,15|22,30.01]. The
  # 25 Hz-capped pool (user directive: no content above 25 Hz anywhere, train or eval)
  # binds [8,15|22,25.0] -- strict upper edge 25.0, because the dataset holds records at
  # f0=25.007 which '<= 25 Hz' must exclude. Any other binning stays rejected.
  # Single-frequency pool (20260928, pino1): a fixed-f0 dataset (all records f0=25.0 Hz)
  # carries ONE bin [8,25.01) -- upper edge 25.01 because 25.0 must be INCLUDED here,
  # unlike the mixed-band 25.0-capped pool where 25.007 had to be excluded. Any 3-bin
  # config keeps the historical assertion bitwise.
  assert (len(bins)==3 and bins[0][0]==8 and bins[-1][1] in (25.0,30.01) and all(bins[j][1]==bins[j+1][0] for j in range(2))) or (len(bins)==1 and bins[0][0]==8 and bins[0][1] in (25.01,30.01));assert config['data']['frames_per_update'] in (16,32,51) and config['data']['ic_frames']==8;train_indices=sorted(r.source_index for r in meta)
  with h5py.File(config['data']['source_h5'],'r') as f:
   freq=dict(zip(train_indices,map(float,f['source_f0_hz'][train_indices])));source_metadata={k:dict(zip(train_indices,map(float,f[k][train_indices]))) for k in ['source_x_m','source_z_m','source_f0_hz','source_t0_s','source_amplitude']}
  # data.families (20260925, routed arm): the training pool is built only for the listed
  # families (cells = families x 3 bins, round-robin over the listed cells). Default = all
  # three, which keeps every pre-existing pool, roster and digest bitwise unchanged. The
  # global cell numbering (uniform 0-2, layered 3-5, marmousi 6-8) and the per-cell
  # permutation seeds are kept, so a listed cell visits its records in the same order as
  # in the full pool; only the round-robin stride changes (9 -> len(families)*3).
  all_families=['uniform','layered','marmousi'];families=list(config['data'].get('families',all_families));assert families and len(set(families))==len(families) and all(f in all_families for f in families),families;families.sort(key=all_families.index);self.families=families
  self.cells=[];roster=[]
  for fam in families:
   fi=all_families.index(fam)
   for bi,(lo,hi) in enumerate(bins):
    rows=[r for r in meta if r.medium_type==fam and r.group_id not in excluded and lo<=freq[r.source_index]<hi];rows.sort(key=lambda r:r.sample_id);assert rows
    cell=[]
    for r in rows:
     item={'sample_id':r.sample_id,'local_index':self.byid[r.sample_id][0],'source_index':r.source_index,'group_id':r.group_id,'sample_sha256':r.sample_sha256,'family':fam,'frequency_bin':bi,'cell':3*fi+bi,'f0_hz':freq[r.source_index]};item['source_parameters_fp32']=[float(np.float32(source_metadata[k][r.source_index])) for k in ['source_x_m','source_z_m','source_f0_hz','source_t0_s','source_amplitude']];item['future_start']=int(np.searchsorted(np.asarray(self.manifest.time_s,dtype=np.float32).astype(np.float64),item['source_parameters_fp32'][3]))+8;roster.append(item);cell.append(item)
    self.cells.append(cell)
  assert len(roster)==len({x['sample_id'] for x in roster});self.roster=roster;self.by_sample={r['sample_id']:r for r in roster};self.digest=digest(roster);self.panel=json.loads(Path(config['data']['dev_panel_json']).read_text());self.dev=self.panel['dev'];assert all(x['split']=='train' for x in self.dev)
  # ST-RAS (20260925): roster position is the table index; cells map to roster positions. The
  # sampler object is attached by the trainer (attach_sampler); None or kind=legacy keeps the
  # historical stateless plan. n_time is the shared saved time axis length (all records).
  self.roster_index={r['sample_id']:i for i,r in enumerate(roster)};self.cell_roster_indices=[[self.roster_index[x['sample_id']] for x in cell] for cell in self.cells];self.n_time=len(self.manifest.time_s);self.sampler=None
  for row in self.dev:
   _,actual=self.byid[row['sample_id']];assert row['group_id']==actual.group_id and row['sample_sha256']==actual.sample_sha256 and row['source_index']==actual.source_index;actual_onset=int(np.searchsorted(np.asarray(self.manifest.time_s,dtype=np.float32).astype(np.float64),float(np.float32(source_metadata['source_t0_s'][actual.source_index]))));assert actual_onset==row['onset']
  assert not {r['group_id'] for r in roster}&{r['group_id'] for r in self.dev};assert not {r['sample_sha256'] for r in roster}&{r['sample_sha256'] for r in self.dev}
  expected=config['data'].get('expected_pool_records')
  if expected is not None:assert len(roster)==expected,(len(roster),expected)
  self.energy_identity={'algorithm':'fullfuture-normalized-target-squared-norm-v1','manifest_digest':self.manifest.digest,'pool_digest':self.digest,'normalizer_sha256':sha(config['data']['normalization_json']),'window':'onset=searchsorted(FP32 saved axis.double,FP32 source_t0);future=onset+8..last','dtype':'labels normalizedFP32,sumsfloat64','energy_chunk_frames':config['data']['energy_chunk_frames'],'time_axis_sha256':hashlib.sha256(np.asarray(self.manifest.time_s,dtype=np.float32).tobytes()).hexdigest(),'source_metadata_bound_in_roster':True};self.energy_values={};self.energy_cache=None
 def _normalizer(self,config):
  forkpath=config['data'].get('fork_identity_json')
  if not forkpath:return load_normalizer(v3_config(config),self.manifest.digest)
  expected=config['data'].get('fork_identity_sha256')
  if not expected or sha(forkpath)!=expected:raise ValueError('augmentation requires explicitly bound audited fork identity')
  fork=json.loads(Path(forkpath).read_text());assert fork['status']=='QC_PASS' and fork['composite_manifest_digest']==self.manifest.digest and fork['composite_vds_sha256']==sha(config['data']['source_h5']);assert fork['base_normalizer_sha256']==sha(config['data']['normalization_json']) and fork['normalizer_refitted'] is False
  from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
  payload=json.loads(Path(config['data']['normalization_json']).read_text());norm=PhysicalNormalizer.from_dict(payload,expected_manifest=fork['base_manifest_digest']);assert norm.metadata.record_count==fork['base_fit_record_count']==2240
  for path,expected_sha in fork['new_shard_hashes'].items():assert sha(path)==expected_sha,path
  return norm
 def attach_energy_cache(self,path):
  self.energy_cache=Path(path);self.energy_cache.parent.mkdir(parents=True,exist_ok=True)
  if self.energy_cache.exists():
   obj=json.loads(self.energy_cache.read_text());assert obj['identity']==self.energy_identity;self.energy_values={k:float(v) for k,v in obj['values'].items()};assert all(k in self.by_sample and np.isfinite(v) and v>0 for k,v in self.energy_values.items())
 def metadata_summary(self):
  return {'records':len(self.roster),'groups':len({r['group_id'] for r in self.roster}),'cells':[len(c) for c in self.cells],'unique_sample_sha256':len({r['sample_sha256'] for r in self.roster}),'manifest_digest':self.manifest.digest,'pool_digest':self.digest,'dev_records':len(self.dev),'first_complete_coverage_draws':len(self.cells)*max(map(len,self.cells)),'source_index_is_not_local_index':True,'frequency_scope':'cell-balanced sampling; original frequency gaps retained until a separately bound dataset is supplied',**({'families':self.families} if self.families!=['uniform','layered','marmousi'] else {})}
 def sample(self,draw_index):
  n=len(self.cells);slot=draw_index%n;visit=draw_index//n;values=self.cells[slot];cell=values[0]['cell'];cycle,position=divmod(visit,len(values));seed=int.from_bytes(hashlib.sha256(f"{self.config['seed']}|{cell}|{cycle}".encode()).digest()[:8],'little');permutation=np.random.default_rng(seed).permutation(len(values));return values[int(permutation[position])]
 def phase_maps(self,sample_id):
  # P2 multi-arrival phase maps, precomputed per record (phase_maps.py) and keyed by sample_sha256.
  cache=self.config['data'].get('phase_cache_dir')
  if not cache:return None
  _,rec=self.byid[sample_id];path=Path(cache)/f'{rec.sample_sha256}.npz'
  if not path.exists():raise FileNotFoundError(f'phase map cache missing for {sample_id}: {path}')
  with np.load(path,allow_pickle=False) as f:T=torch.from_numpy(np.ascontiguousarray(f['T'])).float()[None];r=torch.from_numpy(np.ascontiguousarray(f['r'])).float()[None]
  return T,r
 def inputs(self,row):
  if row['sample_id'] not in self.byid:raise ValueError('not a train record')
  local=self.byid[row['sample_id']][0];rec=self.records[local];axis=rec.time_s.double();onset=int(torch.searchsorted(axis,float(rec.source_parameters[3])));ic_indices=torch.arange(onset,onset+8);ic=self.records.read_wavefield(local,rec.time_s[ic_indices]);assert bool(ic.exact.all());out={'sample_id':row['sample_id'],'local_index':local,'velocity':rec.velocity_mps[None],'source':rec.source_parameters[None],'source_map':rec.source_map[None],'ic':ic.values[None],'anchor':rec.time_s[onset:onset+1],'x':rec.x_m,'z':rec.z_m,'time_axis':rec.time_s,'future_start':onset+8,'family':rec.medium_type,'bin':row.get('frequency_bin',row.get('bin_index')),'sample_sha256':rec.sample_sha256}
  maps=self.phase_maps(row['sample_id'])
  if maps is not None:out['phase_T'],out['phase_r']=maps
  st=self.scatter_tables(row['sample_id'])
  if st is not None:out['scatter_T_scatterer'],out['scatter_T_query']=st
  return out
 def _gate_draw_fields(self,family):
  # Per-draw family-gate audit trail (family_gate.py, 20260926).  Recorded for EVERY
  # draw so a post-hoc reader can verify from metrics.jsonl alone that uniform ran the
  # gate-closed path and layered/marmousi ran it open, without re-deriving the label.
  spec=json.loads(Path(self.config['model_spec']).read_text())['model']
  ungated=tuple(spec.get('scatter_ungated_families',('layered','marmousi')))
  phase_gated=tuple(spec.get('phase_gated_families',()))
  f=str(family)
  return {'family':f,
          'scatter_open':bool(f in set(ungated)),
          'phase_open':bool(f not in set(phase_gated)),
          'gated_family':f"{f}|scatter={'open' if f in set(ungated) else 'closed'}|phase={'open' if f not in set(phase_gated) else 'closed'}"}
 def scatter_tables(self,sample_id):
  # Born scattering cache (scatter_precompute.py, 20260926), keyed by sample_sha256.
  # Returns (T(s->y) [g,g], T(y->x) [g,g,z,x]) or None when no cache is configured.
  cache=self.config['data'].get('scatter_cache_dir')
  if not cache:return None
  grid=int(self.config['data'].get('scatter_grid') or json.loads(Path(self.config['model_spec']).read_text())['model'].get('scatter_grid',51))
  _,rec=self.byid[sample_id]
  return scatter_binding.load_tables(cache,rec.sample_sha256,grid=grid,query_shape=(len(self.manifest.z_m),len(self.manifest.x_m)))
 def energy(self,c):
  sid=c['sample_id']
  if sid not in self.by_sample:raise ValueError('energy statistics allowed only for registered training pool')
  if sid in self.energy_values:return self.energy_values[sid]
  n=len(c['time_axis']);total=0.;chunk=int(self.config['data']['energy_chunk_frames'])
  for k in range(c['future_start'],n,chunk):
   times=c['time_axis'][k:min(k+chunk,n)];truth=self.records.read_wavefield(c['local_index'],times);assert bool(truth.exact.all());y=self.normalizer.encode_pressure(truth.values[None],c['source'][:,4]);total+=float(y.double().square().sum())
  assert np.isfinite(total) and total>0
  if self.energy_cache is None:raise RuntimeError('energy cache path not attached')
  # Merge under a lock; pre-existing values are immutable across runs.
  with Path(str(self.energy_cache)+'.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   obj={'identity':self.energy_identity,'values':{}}
   if self.energy_cache.exists():obj=json.loads(self.energy_cache.read_text());assert obj['identity']==self.energy_identity
   if sid in obj['values']:assert abs(obj['values'][sid]-total)<=1e-12*max(total,1.)
   obj['values'][sid]=total;atomic_json(self.energy_cache,obj);fcntl.flock(lock,fcntl.LOCK_UN)
  self.energy_values[sid]=total;return total
 def plan(self,draw_index):
  # Draw plan depends only on config seed, the frozen pool roster, and draw_index; no model state enters.
  # frames_per_update strata (16 legacy / 51 P1 arm); identical seed derivation, so the 16-frame path is bitwise unchanged.
  row=self.sample(draw_index);c=self.inputs(row);future=np.arange(c['future_start'],len(c['time_axis']));n_strata=int(self.config['data']['frames_per_update']);strata=np.array_split(future,n_strata);assert all(len(s)>0 for s in strata);seed=int.from_bytes(hashlib.sha256(f"{self.config['seed']}|{draw_index}|{row['sample_id']}|frames".encode()).digest()[:8],'little');rng=np.random.default_rng(seed);frames=[int(s[rng.integers(len(s))]) for s in strata];weights=[len(s) for s in strata];return row,c,frames,weights,seed
 def batch_cpu(self,draw_index):
  # CPU-side half of batch(): plan, H5 read, encode. Safe to run in a prefetch
  # thread; all sampling is per-draw seeded (no global RNG), energy cache is
  # pre-seeded read-only in production runs.
  # ST-RAS (20260925): a decision dict from decide() is accepted in place of the
  # int draw index; the int path below is the historical one and is untouched.
  if isinstance(draw_index,dict):return self.batch_cpu_decided(draw_index)
  row,c,frames,weights,seed=self.plan(draw_index);times=c['time_axis'][frames];truth=self.records.read_wavefield(c['local_index'],times);assert bool(truth.exact.all());target=self.normalizer.encode_pressure(truth.values[None],c['source'][:,4]);E=self.energy(c)
  return c,times,target,torch.tensor(weights,dtype=torch.float32),torch.tensor(E,dtype=torch.float64),{'draw_index':draw_index,'sample_id':row['sample_id'],'cell':row['cell'],'frames':frames,'stratum_sizes':weights,'inclusion_probabilities':[1/n for n in weights],'frame_seed':seed,**self._gate_draw_fields(c['family'])}
 # ---------------- ST-RAS decision path (20260925) ----------------
 def attach_sampler(self,sampler):
  # sampler: ras.STRASTables (kind st_ras) or None (legacy). Tables index = roster position.
  if sampler is not None:assert sampler.n_records==len(self.roster) and sampler.n_strata==int(self.config['data']['frames_per_update'])
  self.sampler=sampler
 def _strata(self,row):
  future=np.arange(row['future_start'],self.n_time);n_strata=int(self.config['data']['frames_per_update']);strata=np.array_split(future,n_strata);assert all(len(s)>0 for s in strata);return strata
 def decide(self,draw_index):
  """Sampling decision for one draw, made with the CURRENT table state (main process).

  legacy (sampler None or kind legacy): identical record + frames + seed to plan(draw_index).
  st_ras: cell round-robin kept; record ~ p_i inside the cell (record level), frames ~
  multinomial stratum allocation (time level), uniform inside a stratum. The dict carries
  everything the data loader needs, so a worker (thread or process) never consults tables."""
  s=self.sampler;n=len(self.cells);slot=draw_index%n
  if s is None or s.cfg['kind']=='legacy':
   row=self.sample(draw_index);strata=self._strata(row);seed=int.from_bytes(hashlib.sha256(f"{self.config['seed']}|{draw_index}|{row['sample_id']}|frames".encode()).digest()[:8],'little');rng=np.random.default_rng(seed);frames=[int(x[rng.integers(len(x))]) for x in strata];weights=[len(x) for x in strata]
   return {'kind':'legacy','draw_index':draw_index,'sample_id':row['sample_id'],'cell':row['cell'],'frames':frames,'stratum_sizes':weights,'frame_strata':list(range(len(strata))),'weights':[float(w) for w in weights],'inclusion_probabilities':[1/n for n in weights],'frame_seed':seed,**self._gate_draw_fields(self.records[self.byid[row['sample_id']][0]].medium_type)}
  from ras import draw_seed,decision_stats
  # Rehearsal shares (20260927): with sampler.family_shares set, the cell is drawn per
  # draw_index with probability share[family]/3 for each of that family's three frequency
  # cells (seeded, reproducible), replacing the deterministic draw%n rotation. All lower
  # levels (record/time/space) are unchanged. Recorded per draw for the audit trail.
  fshare=s.cfg.get('family_shares');cur=s.cfg.get('curriculum');cur_diag={}
  if cur is not None:
   # Curriculum (20260928, agent_curriculum): p_cell = family_share x frequency_weight,
   # a pure function of (config, draw_index - sampler_draw_offset, RAS record table);
   # family balanced (default) or difficulty-aware with a mandatory per-family floor,
   # staged frequency weights with a low-band replay floor (curriculum.py). Reuses the
   # SAME seeded 'cell' stream as the plain family_shares path; record/time/space RAS,
   # the loss definition and the evaluation weights are untouched. Diagnostics (stage,
   # effective shares, uniform-risk vs sampled-risk) ride into the decision dict.
   from curriculum import cell_probabilities
   fam_of_cell=[c_[0]['family'] for c_ in self.cells];bin_of_cell=[c_[0]['frequency_bin'] for c_ in self.cells]
   rel=max(int(draw_index)-int(self.config.get('sampler_draw_offset',0)),0)
   p_cell,cur_diag=cell_probabilities(s.cfg,fam_of_cell,bin_of_cell,rel,s.r,self.cell_roster_indices)
   rng_c=np.random.default_rng(draw_seed(self.config['seed'],draw_index,'cell'))
   slot=int(rng_c.choice(n,p=p_cell))
  elif fshare is not None:
   fam_of_cell=[c_[0]['family'] for c_ in self.cells]
   missing=set(fshare)-set(fam_of_cell)
   if missing:raise ValueError(f'family_shares names families absent from the pool: {sorted(missing)}')
   nb=len(self.config['data']['frequency_bins'])  # cells per family; /3.0 historically (3-bin pools unchanged)
   p_cell=np.array([fshare.get(f,0.0)/nb for f in fam_of_cell],dtype=np.float64)
   assert abs(p_cell.sum()-1.0)<1e-9,p_cell.sum()
   rng_c=np.random.default_rng(draw_seed(self.config['seed'],draw_index,'cell'))
   slot=int(rng_c.choice(n,p=p_cell))
  values=self.cells[slot];ridx=self.cell_roster_indices[slot]
  if cur is not None:cur_diag={**cur_diag,'ras_cell_slot':int(slot),'ras_family_share':float(cur_diag['ras_family_shares_effective'][values[0]['family']])}
  if s.cfg['record']:
   p=s.record_probabilities(ridx);rng=np.random.default_rng(draw_seed(self.config['seed'],draw_index,'record'));k=int(rng.choice(len(values),p=p));row=values[k];p_i=float(p[k])
  else:
   row=self.sample(draw_index);k=[x['sample_id'] for x in values].index(row['sample_id']);p=np.full(len(values),1.0/len(values));p_i=float(p[k])
  i=self.roster_index[row['sample_id']];strata=self._strata(row);sizes=[len(x) for x in strata];n_frames=len(strata)
  # Causal time curriculum (20260928, agent_causal_min): staged horizon on the TIME level.
  # update = floor((draw_index - sampler_draw_offset)/draws_per_update); each stratum keeps
  # only its frames with record time <= horizon_s (record's OWN time axis); strata with no
  # eligible frame get q=0 (existing Rt floor_mix, then mask + renormalize below), and
  # intra_stratum='residual' runs the existing Rf floor_mix over eligible frames only.
  # pick==strata when disabled, so every downstream line is bitwise the historical one.
  ct=s.cfg.get('causal_time');causal_diag={};pick=strata
  if ct is not None:
   from ras import causal_horizon_s
   rel=max(int(draw_index)-int(self.config.get('sampler_draw_offset',0)),0);cu=rel//int(ct['draws_per_update']);hz=causal_horizon_s(ct,cu)
   t_axis=np.asarray(self.records[self.byid[row['sample_id']][0]].time_s,dtype=np.float64);assert t_axis.shape==(self.n_time,)
   elig=[st_[t_axis[st_]<=hz] for st_ in strata];open_mask=np.array([e.size>0 for e in elig],dtype=bool)
   if open_mask.any():pick=elig
   else:open_mask=np.ones(n_frames,dtype=bool)  # horizon before the record's future window: fall back to the unmasked historical draw
   causal_diag={'causal_horizon_s':float(hz),'causal_update':int(cu),'causal_open_strata':int(open_mask.sum()),'causal_fallback':bool(pick is strata)}
  if s.cfg['time']:
   q=s.time_probabilities(i)
   if ct is not None and not causal_diag['causal_fallback']:
    q=np.where(open_mask,q,0.0);q=q/q.sum()
   rng=np.random.default_rng(draw_seed(self.config['seed'],draw_index,row['sample_id'],'frames_ras'));n_s=rng.multinomial(n_frames,q);frames=[];frame_strata=[];seed=None
   # Intra-stratum choice: 'uniform' is the historical i.i.d. pick (bitwise unchanged).
   # 'residual' draws the frame WITHIN the stratum from the frame-level residual table
   # Rf (floor_mix, same eps floor as the other levels), so supervision concentrates on
   # the high-residual frames inside a stratum, not just on the high-residual strata.
   # Rt[i,s] is one scalar per stratum and cannot express this; at 51 strata each holds
   # 6-8 frames, so the choice is real.  p_frame is recorded per drawn frame because the
   # inclusion probability -- and hence the HT weight below -- now depends on it.
   p_frame=[]
   if s.cfg['intra_stratum']=='residual':
    for si,cnt in enumerate(n_s):
     if not int(cnt):continue
     pf=s.frame_probabilities(i,pick[si])
     for _ in range(int(cnt)):
      k=int(rng.choice(len(pick[si]),p=pf));frames.append(int(pick[si][k]));frame_strata.append(si);p_frame.append(float(pf[k]))
   else:
    for si,cnt in enumerate(n_s):
     for _ in range(int(cnt)):
      L=len(pick[si]);frames.append(int(pick[si][rng.integers(L)]));frame_strata.append(si);p_frame.append(1.0/L)
  else:
   q=np.full(n_frames,1.0/n_frames);n_s=np.ones(n_frames,dtype=np.int64);seed=int.from_bytes(hashlib.sha256(f"{self.config['seed']}|{draw_index}|{row['sample_id']}|frames".encode()).digest()[:8],'little');rng=np.random.default_rng(seed);frames=[int(x[rng.integers(len(x))]) for x in strata];frame_strata=list(range(n_frames));p_frame=[1.0/len(strata[si]) for si in frame_strata]
  # Per-frame inclusion probability: pi_k = (stratum allocation) x (choice inside it).
  # With intra_stratum='uniform' the second factor is 1/|S_s| and this reduces to the
  # historical n_frames*q_s/|S_s| exactly; with 'residual' it is p_frame[k], so the
  # correction below is what keeps the estimator honest when frames inside a stratum
  # are drawn with unequal probability.
  pi=[n_frames*float(q[frame_strata[k]])*p_frame[k] for k in range(len(frames))]
  if s.cfg['importance_correct']:weights=[1.0/(pi[k]*len(values)*p_i) for k in range(len(frames))]
  elif s.cfg['intra_stratum']=='residual':
   # Biased mode still divides out the INTRA-stratum distortion: without it a frame the
   # table favours would be counted both more often and with the same weight, i.e. its
   # residual would be squared into the objective. The deliberate bias of this mode is
   # the stratum allocation (|S_s| weighting), which is left untouched.
   weights=[float(sizes[frame_strata[k]])*(1.0/(len(pick[frame_strata[k]])*p_frame[k])) for k in range(len(frames))]
  else:weights=[float(sizes[si]) for si in frame_strata]
  if ct is not None:
   # Early/mid/late sample shares over the record's own time axis (thirds of [0, t_max]).
   tf=t_axis[np.asarray(frames,dtype=np.int64)];T=float(t_axis[-1])
   causal_diag.update({'causal_share_early':float(np.mean(tf<=T/3)),'causal_share_mid':float(np.mean((tf>T/3)&(tf<=2*T/3))),'causal_share_late':float(np.mean(tf>2*T/3))})
  st=decision_stats(p,q,n_s,n_frames)
  return {'kind':'st_ras','draw_index':draw_index,'sample_id':row['sample_id'],'cell':row['cell'],'frames':frames,'stratum_sizes':sizes,'frame_strata':frame_strata,'weights':weights,'inclusion_probabilities':pi,'frame_seed':seed,'roster_index':i,'ras_record_p':p_i,'ras_record_r':float(s.r[i]),'ras_cell_size':len(values),'ras_alloc_n':[int(x) for x in n_s],'ras_alloc_q':[float(x) for x in q],'ras_decision_step':int(s.steps),'ras_intra_stratum':s.cfg['intra_stratum'],'ras_p_frame':[float(x) for x in p_frame],**({'ras_family_share':float(fshare[values[0]['family']]),'ras_cell_slot':int(slot)} if fshare is not None else {}),**cur_diag,**causal_diag,**{'ras_'+k:v for k,v in st.items()},**self._gate_draw_fields(self.records[self.byid[row['sample_id']][0]].medium_type)}
 def decisions(self,first_draw,last_draw,stride=1):
  # Lazy generator for prefetchers: each decision is made when the consumer pulls it.
  d=first_draw
  while d<=last_draw:
   yield self.decide(d);d+=stride
 def batch_cpu_decided(self,dec):
  row=self.by_sample[dec['sample_id']];c=self.inputs(row);assert c['future_start']==row['future_start'] and len(c['time_axis'])==self.n_time,'roster onset drift';frames=dec['frames'];times=c['time_axis'][frames];truth=self.records.read_wavefield(c['local_index'],times);assert bool(truth.exact.all());target=self.normalizer.encode_pressure(truth.values[None],c['source'][:,4]);E=self.energy(c)
  draw={k:v for k,v in dec.items() if k not in ('kind','weights')} if dec['kind']=='st_ras' else {k:dec[k] for k in ('draw_index','sample_id','cell','frames','stratum_sizes','inclusion_probabilities','frame_seed')}
  return c,times,target,torch.tensor(dec['weights'],dtype=torch.float32),torch.tensor(E,dtype=torch.float64),draw
 def batch(self,draw_index,device):
  c,times,target,weights,E,draw=self.batch_cpu(draw_index)
  return c,times,target.to(device),weights.to(device),E.to(device),draw
 def close(self):
  h=getattr(self.records,'_h5',None)
  if h is not None:h.close()
