"""Full record coverage and unbiased lagged stratified time sampling."""
from __future__ import annotations
import hashlib
import numpy as np
BINS=((0,25),(25,57),(57,92))
DRAWS_PER_BIN=12
UNIFORM_MIX=.5
class RecordCursor:
    def __init__(self,ids,seed=2026100822,state=None):
        self.ids=tuple(ids);self.seed=int(seed);self.epoch=0;self.position=0;self.steps=0;self.exposures=0
        if len(set(self.ids))!=len(self.ids) or not self.ids:raise ValueError('unique roster')
        if state is not None:
            if state['ids']!=list(ids) or state['seed']!=seed:raise ValueError('cursor identity')
            for key in ('epoch','position','steps','exposures'):setattr(self,key,int(state[key]))
        self._validate()
    def _validate(self):
        if not (0<=self.position<len(self.ids)) or self.epoch<0 or self.steps<0:raise ValueError('cursor range')
        if self.exposures!=self.epoch*len(self.ids)+self.position:raise ValueError('real exposures')
    def plan(self):
        perm=np.random.default_rng(self.seed+self.epoch).permutation(len(self.ids))
        values=perm[self.position:min(self.position+4,len(perm))].tolist()
        return dict(step=self.steps,epoch=self.epoch,position=self.position,indices=values+[None]*(4-len(values)),active=len(values))
    def commit(self,plan):
        if plan!=self.plan():raise ValueError('current complete update only')
        self.position+=plan['active'];self.exposures+=plan['active'];self.steps+=1
        if self.position==len(self.ids):self.epoch+=1;self.position=0
        self._validate()
    def state_dict(self):return dict(ids=list(self.ids),seed=self.seed,epoch=self.epoch,position=self.position,steps=self.steps,exposures=self.exposures)
class TimeSampler:
    def __init__(self,n,state=None):
        self.score=np.zeros((n,92),np.float64);self.count=np.zeros((n,92),np.int64)
        if state is not None:self.score=np.asarray(state['score'],np.float64);self.count=np.asarray(state['count'],np.int64)
        if self.score.shape!=(n,92) or self.count.shape!=(n,92) or not np.isfinite(self.score).all() or (self.score<0).any() or (self.count<0).any():raise ValueError('time sampler state')
    def plan(self,index,energy,step,sid):
        energy=np.asarray(energy,np.float64)
        if energy.shape!=(92,) or not np.isfinite(energy).all() or energy.sum()<=0:raise ValueError('full92 energy')
        seed=int.from_bytes(hashlib.sha256(f'R22|{step}|{sid}'.encode()).digest()[:8],'little')
        rng=np.random.default_rng(seed);coeff=np.zeros(92,np.float64);prob=np.zeros(92,np.float64)
        for lo,hi in BINS:
            seen=self.count[index,lo:hi]>0;e=energy[lo:hi]/energy.sum()
            s=np.where(seen,self.score[index,lo:hi],e)
            q=np.full(hi-lo,1/(hi-lo)) if s.sum()<=0 else UNIFORM_MIX/(hi-lo)+(1-UNIFORM_MIX)*s/s.sum()
            draws=rng.choice(hi-lo,size=DRAWS_PER_BIN,replace=True,p=q)
            multiplicity=np.bincount(draws,minlength=hi-lo)
            coeff[lo:hi]=multiplicity/(DRAWS_PER_BIN*q*energy.sum());prob[lo:hi]=q
        selected=np.flatnonzero(coeff)
        return dict(indices=selected.tolist(),coefficients=coeff[selected].tolist(),conditional_q=prob.tolist(),seed=seed,draws=3*DRAWS_PER_BIN,denominator=float(energy.sum()),full92_target=True)
    def commit(self,obs):
        seen=set()
        for row in obs:
            i=row['index'];idx=np.asarray(row['time_indices'],np.int64);vals=np.asarray(row['relative_sse'],np.float64)
            if i in seen or idx.shape!=vals.shape or np.unique(idx).size!=idx.size or not np.isfinite(vals).all() or (vals<0).any():raise ValueError('actual observations')
            seen.add(i);old=self.score[i,idx];measured=self.count[i,idx]>0
            self.score[i,idx]=np.where(measured,.9*old+.1*vals,vals);self.count[i,idx]+=1
    def state_dict(self):return dict(score=self.score.tolist(),count=self.count.tolist())
