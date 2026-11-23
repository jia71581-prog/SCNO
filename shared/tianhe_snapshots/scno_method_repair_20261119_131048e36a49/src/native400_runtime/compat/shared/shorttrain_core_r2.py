"""True TRAIN100 loss/Adam/committed hardtable; frozen original inference source reused."""
from pathlib import Path
from contextlib import ExitStack
import copy,hashlib,json,random,sys,time
import numpy as np
from model_bridge import module,released_parent,restrict_sample,forward100
from data_views import FUTURE_FRAMES,pressure_view
from sampling import HardFrameTable
from objective import loss

def tree_digest(value):
 import torch
 h=hashlib.sha256()
 def visit(v):
  if torch.is_tensor(v):h.update(str(v.dtype).encode());h.update(str(list(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str(v.dtype).encode());h.update(str(list(v.shape)).encode());h.update(np.ascontiguousarray(v).tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=lambda x:(str(type(x)),str(x))):h.update(repr(k).encode());visit(v[k])
  elif isinstance(v,(tuple,list)):
   for x in v:visit(x)
  else:h.update(repr(v).encode())
 visit(value);return h.hexdigest()
def rng_state():
 import torch
 return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def restore_rng(v):
 import torch
 random.setstate(v['python']);np.random.set_state(v['numpy']);torch.set_rng_state(v['torch']);torch.cuda.set_rng_state(v['cuda'])
def scopes(helpers,records):
 st=ExitStack();st.enter_context(helpers['padding'].scoped_replicate_pad2d(records['padding']));st.enter_context(helpers['pooling'].scoped_deterministic_pooling(records['pooling']));st.enter_context(helpers['cartesian'].scoped_cartesian_adjoint(records['sampling'],helpers['sampling'],(101,101)));return st

def fresh_train(family,m):
 model,norm,parent,provider=released_parent(family,'cuda',gpu_authorized=True);model.requires_grad_(True);model.train()
 # Original family groups/LRs, fresh Adam, one globalclip (not branch clipping).
 factory=module('fixed100_optimizer_'+family,Path(m['parent']['code'])/'model.py');parser=module('fixed100_optimizer_config_'+family,Path(m['parent']['code'])/'train.py');cfg=parser.resolved_config(m['parent']['config']);cfg=copy.deepcopy(cfg);cfg['optimizer'].update(weight_decay=1e-6,betas=[.9,.999],eps=1e-8);opt,mapping=factory.optimizer(model,cfg)
 if opt.state:raise AssertionError('fresh optimizer stateempty')
 return model,norm,parent,provider,opt,mapping

def target100(m,norm,sample,sha,stat):
 import torch
 v=m['GT'];stat(v['stat'])
 if sha(v['path'])!=v['sha256']:raise PermissionError('oldregisteredTRAIN targetbank')
 native=np.load(v['path'],mmap_mode='r',allow_pickle=False)
 if native.shape!=(713,201,201) or native.dtype!=np.float32:raise ValueError('oldPa713 target source')
 # Only the fixed100 native stored frame bodies; no other future labels or fine input.
 selected=torch.tensor(np.asarray(native[np.asarray(FUTURE_FRAMES)-88]).copy(),device='cuda');view=pressure_view(selected);return norm.encode_pressure(view[None],sample['source'][:,4])

def train_step(model,norm,sample,provider,opt,table,target,helpers,step,index=0,time_block=100):
 import torch
 model.train();opt.zero_grad(set_to_none=True);plan=table.plan(index,step);records={k:[] for k in ('padding','sampling','pooling')};torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
 with scopes(helpers,records):
  prediction=forward100(model,norm,sample,provider,time_block=time_block);total,census,extra,obs=loss(prediction,target,plan);total.backward()
 if not bool(torch.isfinite(total)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise FloatingPointError('finite true100 loss/grad')
 gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
 if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('finite successful Adam state')
 obs['record_index']=index;table.commit(step,[obs],True);torch.cuda.synchronize();sec=time.monotonic()-tick
 r=dict(step=step+1,loss=float(total.detach()),census=float(census.detach()),hard_extra=float(extra.detach()),global_gradient_before_clip=float(gn),q=plan['q'],indices=plan['indices'],mask=plan['mask'],seed=plan['seed'],cold_or_nohard_skipped=len(plan['indices'])==0,observation=obs,table_digest=tree_digest(table.state_dict()),parameters_digest=tree_digest(model.state_dict()),optimizer_digest=tree_digest(opt.state_dict()),seconds_compute=sec,allocator_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved(),actual_time_block=time_block,full100_one_modelAPI=True,fresh_context=True,success=True)
 del prediction,total,census,extra;return r

def snapshot(model,opt,table,parent,next_step):
 # No inference before a further training step on this instance.
 return dict(schema='FIXED100_TRUE_TRAIN_TECHNICAL_CHECKPOINT_V1',scope='TECHNICAL_ONLY_NOT_LONG_INITIALIZATION',family=parent['family'],base_sha256=parent['checkpoint_sha256'],updates_completed=next_step,model_state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()},optimizer_state=opt.state_dict(),hard_table=table.state_dict(),rng=rng_state(),next_plan=table.plan(0,next_step),time_block=100)

def restore(family,m,path,record_ids):
 import torch
 saved=torch.load(path,map_location='cpu',weights_only=False)
 if saved['schema']!='FIXED100_TRUE_TRAIN_TECHNICAL_CHECKPOINT_V1' or saved['scope']!='TECHNICAL_ONLY_NOT_LONG_INITIALIZATION' or saved['base_sha256']!=m['parent']['sha256'] or saved['family']!=family or type(saved['updates_completed']) is not int or saved['updates_completed']<0 or saved['time_block']!=100:raise PermissionError('technical exact family/base/update/TB lineage')
 if set(saved['rng'])!={'python','numpy','torch','cuda'}:raise PermissionError('all four RNG domains required')
 model,norm,parent,provider,opt,mapping=fresh_train(family,m);model.load_state_dict(saved['model_state'],strict=True);opt.load_state_dict(saved['optimizer_state']);table=HardFrameTable(record_ids,state=saved['hard_table']);restore_rng(saved['rng'])
 if table.completed!=saved['updates_completed'] or table.plan(0,saved['updates_completed'])!=saved['next_plan']:raise AssertionError('restored committed update/table/next hard draw exact')
 if tree_digest(rng_state())!=tree_digest(saved['rng']):raise AssertionError('complete Python NumPy torch CUDA restored RNG bytes')
 return model,norm,parent,provider,opt,table,saved
