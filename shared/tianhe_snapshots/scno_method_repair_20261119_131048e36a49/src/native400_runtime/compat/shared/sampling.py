"""Committed TRAIN-only last-R hard masks, EMA severity and seeded HT extras."""
import hashlib,json
import numpy as np
THRESHOLD=.05
BETA=.9
UNIFORM_FLOOR=.25
SCORE_CAP=20.
DRAWS=32
LAMBDA=1.
ENERGY_FLOOR_RATIO=1e-3
class HardFrameTable:
 def __init__(self,record_ids,seed=2026100611,state=None):
  self.ids=tuple(record_ids);self.seed=int(seed);self.N=len(self.ids)
  if not self.N or len(set(self.ids))!=self.N:raise ValueError('unique fixed TRAIN roster')
  self.last=np.zeros((self.N,100),np.float64);self.ema=self.last.copy();self.defined=np.zeros((self.N,100),bool);self.count=np.zeros(self.N,np.int64);self.measurement_step=np.full(self.N,-1,np.int64);self.completed=0
  self.roster_sha=hashlib.sha256(json.dumps(self.ids,separators=(',',':')).encode()).hexdigest()
  if state is not None:self.load_state(state)
 def plan(self,index,global_step):
  i=int(index)
  if i!=index or not 0<=i<self.N:raise IndexError("exact registeredrecord index")
  if int(global_step)!=self.completed:raise PermissionError('harddraw only consumed committed nextstep; notprefetch')
  h=self.defined[i]&(self.last[i]>THRESHOLD);score=np.where(h,np.maximum(self.ema[i]-THRESHOLD,0),0);score=np.minimum(score,SCORE_CAP)
  if h.any() and score.sum()==0:score=h.astype(np.float64)
  q=np.full(100,1/100,np.float64) if not h.any() else UNIFORM_FLOOR/100+(1-UNIFORM_FLOOR)*score/score.sum()
  if not np.isfinite(q).all() or abs(float(q.sum())-1)>1e-12 or (q<UNIFORM_FLOOR/100-1e-15).any():raise ArithmeticError('finite exactfloor q')
  seed=int.from_bytes(hashlib.sha256(f'{self.seed}|{global_step}|{self.ids[i]}|{self.count[i]}'.encode()).digest()[:8],'little');draw=[] if not h.any() else np.random.Generator(np.random.PCG64(seed)).choice(100,size=DRAWS,replace=True,p=q).tolist()
  return dict(record_index=i,global_step=int(global_step),record_visit=int(self.count[i]),mask=h.tolist(),q=q.tolist(),indices=draw,seed=seed,nohard_skip_draw=True,draw_counter_rule='count increments each successfulrecordvisit includingnohard; no globalRNG consumed',mask_from_last_actualR_not_EMA=True)
 def commit(self,global_step,updates,allrank_optimizer_success):
  if int(global_step)!=self.completed or allrank_optimizer_success is not True:raise PermissionError('one successful synchronized allrankstep only')
  if not updates:raise ValueError('nonempty actual active record observations')
  pending=[];indices=[]
  for obs in updates:
   i=int(obs['record_index'])
   if i!=obs['record_index'] or not 0<=i<self.N:raise IndexError('exact registered update index')
   indices.append(i);r=np.asarray(obs['raw_R_finite_or_zero'],np.float64);d=np.asarray(obs['defined'])
   if r.shape!=(100,) or d.shape!=(100,) or d.dtype!=np.bool_ or not np.isfinite(r).all() or (r<0).any() or np.any(r[~d]!=0):raise ValueError('finite nonnegative100R/boolean defined/undefined0')
   ema=r.copy() if not self.count[i] else BETA*self.ema[i]+(1-BETA)*r;ema[~d]=0
   if not np.isfinite(ema).all() or (ema<0).any():raise FloatingPointError('finite futureEMA')
   pending.append((i,r.copy(),d.copy(),ema))
  if len(indices)!=len(set(indices)):raise ValueError('allrank unique record observations')
  # Full batch validates before any row mutates committed scientific state.
  for i,r,d,ema in sorted(pending,key=lambda x:x[0]):
   self.ema[i]=ema;self.last[i]=r;self.defined[i]=d;self.count[i]+=1;self.measurement_step[i]=int(global_step)
  self.completed+=1
 def assert_cursor(self,cursor,active_indices=None,observations=None):
  if self.completed!=cursor.completed or not np.array_equal(self.count,cursor.visits):raise AssertionError('table/cursor committed step and counts exact')
  if active_indices is not None and sorted(active_indices)!=sorted(int(v['record_index']) for v in observations):raise AssertionError('observed records equal sourceplan active ids')
 def state_dict(self):return dict(schema='FIXED100_COMMITTED_HARD_V1',roster_sha256=self.roster_sha,record_ids=list(self.ids),seed=self.seed,completed_successful_global_steps=self.completed,last_actual_R=self.last.tolist(),EMA_severity=self.ema.tolist(),raw_defined=self.defined.tolist(),record_visit_drawcounter=self.count.tolist(),measurement_preoptimizer_step=self.measurement_step.tolist(),recipe=dict(threshold=THRESHOLD,beta=BETA,uniform_floor=UNIFORM_FLOOR,score_cap=SCORE_CAP,m=DRAWS,lambda_hard=LAMBDA,energy_floor_ratio=ENERGY_FLOOR_RATIO))
 def load_state(self,s):
  expected=self.state_dict()
  for k in ['schema','roster_sha256','record_ids','seed','recipe']:
   if s.get(k)!=expected[k]:raise PermissionError('strict recipe/roster resume')
  last=np.asarray(s['last_actual_R'],np.float64);ema=np.asarray(s['EMA_severity'],np.float64);defined=np.asarray(s['raw_defined']);count=np.asarray(s['record_visit_drawcounter']);measure=np.asarray(s['measurement_preoptimizer_step']);completed=s['completed_successful_global_steps']
  if type(completed) is not int or completed<0 or last.shape!=(self.N,100) or ema.shape!=last.shape or defined.shape!=last.shape or defined.dtype!=np.bool_ or count.shape!=(self.N,) or measure.shape!=(self.N,) or count.dtype.kind not in 'iu' or measure.dtype.kind not in 'iu':raise ValueError('typed exact committedtable shapes')
  if not np.isfinite(last).all() or not np.isfinite(ema).all() or (last<0).any() or (ema<0).any() or (count<0).any() or np.any(last[~defined]!=0) or np.any(ema[~defined]!=0):raise ValueError('finite nonnegative values/undefined lastEMA0')
  unseen=count==0
  if np.any(measure[unseen]!=-1) or defined[unseen].any() or np.any(last[unseen]!=0) or np.any(ema[unseen]!=0) or np.any(measure[~unseen]<0) or np.any(measure[~unseen]>=completed):raise ValueError('count/measurement/observation timeline exact')
  self.last=last.copy();self.ema=ema.copy();self.defined=defined.copy();self.count=count.astype(np.int64,copy=True);self.measurement_step=measure.astype(np.int64,copy=True);self.completed=completed
