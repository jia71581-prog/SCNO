"""Lagged record importance sampling and bounded immutable CPU record caching."""
import copy
import math
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import torch

_reader_paths=None
_reader_contract=None

def _init_reader(paths,contract):
    global _reader_paths,_reader_contract
    _reader_paths=paths;_reader_contract=contract
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    torch.set_num_threads(1)

def _read_index(index):
    from .dataset import read_record
    v,loc,u,_=read_record(_reader_paths[index],_reader_contract)
    return v,loc,u


def per_record_loss(pred,target):
    p=pred.flatten(1);t=target.flatten(1);d=p-t
    return .9*d.abs().sum(1)/t.abs().sum(1).clamp_min(1e-20)+.1*torch.linalg.vector_norm(d,dim=1)/torch.linalg.vector_norm(t,dim=1).clamp_min(1e-20)


class RecordSampler:
    """EMA energy, positive uniform support, actual draw probabilities retained."""
    def __init__(self,count,seed=0,uniform_fraction=.2,ema=.9,power=.5):
        if count<1 or not 0<uniform_fraction<=1 or not 0<=ema<1 or power<=0:raise ValueError('Invalid sampler')
        self.count=int(count);self.uniform_fraction=float(uniform_fraction);self.ema=float(ema);self.power=float(power)
        self.scores=torch.ones(count,dtype=torch.float64);self.visits=torch.zeros(count,dtype=torch.int64)
        self.generator=torch.Generator().manual_seed(seed);self.pending=[];self.commits=0;self.draws=0

    def probabilities(self):
        energy=self.scores.clamp_min(1e-12).pow(self.power)
        return (1-self.uniform_fraction)*energy/energy.sum()+self.uniform_fraction/self.count

    def ensure(self,count):
        if count>len(self.pending):
            q=self.probabilities()
            ids=torch.multinomial(q,count-len(self.pending),replacement=True,generator=self.generator)
            self.pending.extend((int(i),float(q[i])) for i in ids)

    def peek(self,count):
        self.ensure(count);return list(self.pending[:count])

    def commit(self,plan,losses):
        values=torch.as_tensor(losses,dtype=torch.float64).detach().cpu().reshape(-1)
        if len(plan)!=len(values) or self.pending[:len(plan)]!=list(plan) or not bool(torch.isfinite(values).all()) or bool((values<0).any()):
            raise ValueError('Invalid or nonmatching sampler commit')
        ids=torch.tensor([i for i,p in plan],dtype=torch.long)
        for i in ids.unique():
            mask=ids==i;self.scores[i]=self.ema*self.scores[i]+(1-self.ema)*values[mask].square().mean()
            self.visits[i]+=int(mask.sum())
        del self.pending[:len(plan)];self.commits+=1;self.draws+=len(plan)

    def state_dict(self):
        return dict(schema=1,count=self.count,uniform_fraction=self.uniform_fraction,ema=self.ema,power=self.power,
            scores=self.scores.clone(),visits=self.visits.clone(),rng=self.generator.get_state().clone(),
            pending=copy.deepcopy(self.pending),commits=self.commits,draws=self.draws)

    def load_state_dict(self,s):
        if s['schema']!=1 or any(s[k]!=getattr(self,k) for k in ['count','uniform_fraction','ema','power']):raise ValueError('Sampler contract mismatch')
        scores=s['scores'].double().cpu();visits=s['visits'].long().cpu()
        if scores.shape!=(self.count,) or visits.shape!=scores.shape or not bool(torch.isfinite(scores).all()) or bool((scores<0).any()) or bool((visits<0).any()):raise ValueError('Invalid sampler state')
        pending=[(int(i),float(p)) for i,p in s['pending']]
        if any(not 0<=i<self.count or not math.isfinite(p) or not 0<p<=1 for i,p in pending):raise ValueError('Invalid pending draw')
        self.scores=scores.clone();self.visits=visits.clone();self.generator.set_state(s['rng'].cpu())
        self.pending=pending;self.commits=int(s['commits']);self.draws=int(s['draws'])
        if self.draws!=int(visits.sum()):raise ValueError('Sampler draw count mismatch')


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('Cannot determine available host memory')


class RecordCache:
    """Cache raw CPU tensors only; callers must not mutate returned records."""
    def __init__(self,dataset,max_bytes=80*1024**3,reserve_bytes=24*1024**3,workers=2,max_pending=64,process_paths=None,contract=None):
        self.dataset=dataset;self.max_bytes=int(max_bytes);self.reserve=int(reserve_bytes);self.max_pending=int(max_pending)
        self.process_reader=process_paths is not None
        self.pool=(ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_init_reader,
                                      initargs=([str(p) for p in process_paths],contract)) if self.process_reader else ThreadPoolExecutor(max_workers=workers))
        self.pending={};self.cache=OrderedDict();self.bytes=0
        self.hits=0;self.misses=0;self.wait_s=0.;self.evictions=0
        self.allocator_trim_calls=0;self._last_allocator_trim=0.;self._malloc_trim=None
        try:
            import ctypes
            self._malloc_trim=getattr(ctypes.CDLL(None),'malloc_trim',None)
        except OSError:pass

    @staticmethod
    def size(row):return sum(x.numel()*x.element_size() for x in row)

    def trim(self,extra=0):
        deficit=max(0,self.reserve-available_memory()) if self.reserve else 0
        target=max(0,min(self.max_bytes-extra,self.bytes-deficit))
        while self.cache and self.bytes>target:
            _,row=self.cache.popitem(last=False);self.bytes-=self.size(row);self.evictions+=1
            del row
        # Freed NumPy/CPU SOAP allocations may remain in glibc arenas. Return
        # unused pages before memory pressure repeatedly discards useful records.
        now=time.monotonic()
        if deficit and self._malloc_trim is not None and now-self._last_allocator_trim>=5:
            self._malloc_trim(0);self._last_allocator_trim=now;self.allocator_trim_calls+=1

    def prefetch(self,indices):
        self.trim()
        for i in indices:
            i=int(i)
            if i not in self.cache and i not in self.pending and len(self.pending)<self.max_pending:
                self.pending[i]=self.pool.submit(_read_index if self.process_reader else self.dataset.__getitem__,i)

    def get(self,i):
        i=int(i)
        if i in self.cache:
            self.hits+=1;self.cache.move_to_end(i);return self.cache[i]
        self.misses+=1;t=time.perf_counter();future=self.pending.pop(i,None)
        row=future.result() if future else self.dataset[i]
        row=tuple(torch.as_tensor(x) for x in row)
        self.wait_s+=time.perf_counter()-t;size=self.size(row);self.trim(size)
        if size<=self.max_bytes and (not self.reserve or available_memory()>=self.reserve):
            self.cache[i]=row;self.bytes+=size
        return row

    def drain(self):
        for i in list(self.pending):self.get(i)

    def stats(self):
        return dict(records=len(self.cache),bytes=self.bytes,hits=self.hits,misses=self.misses,
                    wait_s=self.wait_s,evictions=self.evictions,prefetch_pending=len(self.pending),
                    allocator_trim_calls=self.allocator_trim_calls)

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True);self.pending.clear();self.cache.clear();self.bytes=0


@torch.no_grad()
def prime_soap(optimizer,adam_state):
    """Same historical-momentum covariance initialization as learned_soap.py.

    SOAP moments and step start at zero; this is not an Adam-state conversion.
    """
    ids=[i for group in adam_state['param_groups'] for i in group['params']]
    if len(ids)!=len(optimizer.originals):raise ValueError('Adam/SOAP parameter roster differs')
    group=optimizer.param_groups[0]
    for ident,(_,p),proxy in zip(ids,optimizer.originals,optimizer.proxies):
        raw=adam_state['state'].get(ident,{}).get('exp_avg')
        if raw is None:raw=torch.zeros_like(p)
        if raw.shape!=p.shape:raise ValueError('Adam momentum shape differs')
        raw=raw.to(device=p.device,dtype=p.dtype)
        g=torch.view_as_real(raw.resolve_conj()) if raw.is_complex() else raw
        state=optimizer.state[proxy]
        state.update(step=0,exp_avg=torch.zeros_like(g),exp_avg_sq=torch.zeros_like(g))
        beta=group['shampoo_beta'] if group['shampoo_beta']>=0 else group['betas'][1]
        optimizer.init_preconditioner(g,state,precondition_frequency=group['precondition_frequency'],
            precondition_1d=group['precondition_1d'],shampoo_beta=beta,max_precond_dim=group['max_precond_dim'],merge_dims=False)
        optimizer.update_preconditioner(g,state,max_precond_dim=group['max_precond_dim'],merge_dims=False,precondition_1d=group['precondition_1d'])
        if hasattr(optimizer,'offload_state'):optimizer.offload_state(proxy)
