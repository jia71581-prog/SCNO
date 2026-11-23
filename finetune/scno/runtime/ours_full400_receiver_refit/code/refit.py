#!/usr/bin/env python3
"""One full400 supervised rank-1 3x3 pressure-readout ridge fit; two fixed states."""
from scno_release_paths import local_asset as _scno_local_asset
from pathlib import Path
import os,sys,json,time,math,traceback,hashlib,argparse
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np, torch
from torch.nn import functional as F
ROOT=Path(_scno_local_asset('project'));RUNTIME=Path(_scno_local_asset('runtime'));WORK=RUNTIME/'ours_full400_receiver_refit'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(RUNTIME/'primary_evidence_revision/early_nonlinear_probe'))
from probe import load_marmousi,MarmSession
from model_adapter import FrozenBaseCache,_local_protocol_path
from native400_runtime.cf_expanded_lora import ConvLoRA,expand
from native400_runtime import r24_cf_preserved_operator as preserved,r24_cf_marmousi_dg_operator as marm_runtime
from native400_runtime.r25_cf_full400_operator import build_r25_cf_full400_operator

REF=ROOT/'runs/paper_marmousi201_single_source_fullfit_20261020/case_0/reference_pa.npy'
POINTER=ROOT/'CURRENT_PAPER_LORA_CONTINUE1000_20261027.json'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def load(arm):
 m,x=load_marmousi('cpu')
 if arm=='pretrained_init':
  ckpt=ROOT/'paper_assets_main_fresh_20261013_marmousi_20261019/models/marmousi_epoch65.pt';m=m.to('cuda').eval().requires_grad_(False);return m,x,ckpt,None
 savedir=Path(json.loads(POINTER.read_text())['cases']['marmousi']['results']);ckpt=savedir/'best.pt';state=torch.load(ckpt,map_location='cpu',weights_only=False,mmap=True);cfg=state['config'];preserved.COMPAT=Path(cfg['compat']);marm_runtime.COMPAT=Path(cfg['compat']);mcfg=json.loads(Path(cfg['model_config']).read_text())['native_config'];mcfg.update(parent_checkpoint=cfg['initializer'],dense_time_block=1,refiner_time_block=1,checkpoint_queries=False)
 del m
 with _local_protocol_path():m=build_r25_cf_full400_operator(mcfg,device='cpu',kind='marmousi')
 expand(m,[],rank=cfg['lora_rank'],seed=cfg['seed']);m.load_state_dict(state['model'],strict=True);del state;m=m.to('cuda').eval().requires_grad_(False);return m,x,ckpt,savedir/'BEST_pa.npy'

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['pretrained_init','adapted_init'],required=True);p.add_argument('--precheck',action='store_true');args=p.parse_args();OUT=WORK/args.arm;OUT.mkdir(parents=True,exist_ok=True)
 def status(stage,**kw):write(OUT/'STATUS.json',dict(stage=stage,pid=os.getpid(),**kw));print(stage,kw,flush=True)
 torch.set_num_threads(2);torch.manual_seed(20261106);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 free,total=torch.cuda.mem_get_info()
 if free<3*1024**3:raise RuntimeError('Need 3 GiB free')
 start=time.perf_counter();status('loading');model,x,ckpt,oldpred=load(args.arm);device='cuda';x['velocity']=x['velocity'].to(device);x['source']=x['source'].to(device);times=x['full_times'].to(device);scale=float(x['source'][4])*x['pressure_scale_pa'];truth=np.load(REF,mmap_mode='r',allow_pickle=False);assert truth.shape==(400,201,201)
 cache=FrozenBaseCache(model,x['velocity'],x['source'],cache_device='cpu',block=1);session=MarmSession(model,x,cache);head=model.refiner.output
 if isinstance(head,torch.nn.Conv2d):
  gen=torch.Generator(device='cpu').manual_seed(20261106);head=ConvLoRA(head,rank=1,generator=gen).to(device);head.eval().requires_grad_(False);model.refiner.output=head
 assert isinstance(head,ConvLoRA) and head.rank==1 and head.flat==144 and head.active_columns.numel()==144
 base_w=head.base.weight.detach().clone();w0=head.weight.detach().clone();b0=head.bias.detach().clone();a0=head.A.detach().clone();bb0=head.B.detach().clone();p0=head.base.bias.detach().clone();capt=[];last={};counts={'dg':0,'head':0}
 def dg(_,a,o):counts['dg']+=1;last['dg']=o
 def capture(_,a):counts['head']+=1;capt.append(a[0])
 hdg=model.refiner.dg.register_forward_hook(dg);hh=head.register_forward_pre_hook(capture)
 def query(t):
  capt.clear();last.clear();u=session.query(t.reshape(1),grad=False)[0];assert len(capt)==1 and torch.equal(capt[0],last['dg']) and capt[0].shape==(1,16,201,201);return u,capt[0]
 def set_delta(d):
  d=torch.as_tensor(d,device=device,dtype=head.A.dtype)
  with torch.no_grad():head.A.copy_((w0-base_w).reshape(1,144)+d[:144].reshape(1,144));head.B.fill_(1.);head.base.bias.copy_(b0+d[144:145])
 def restore():
  with torch.no_grad():head.A.copy_(a0);head.B.copy_(bb0);head.base.bias.copy_(p0)
 def design(hidden,t):
  base=cache.get(t.reshape(1));gain=base.square().mean(dim=(-2,-1),keepdim=True).add(1e-6).sqrt().double().reshape(());v=F.unfold(hidden,kernel_size=3,padding=1)[0].double();v=torch.cat((v,torch.ones((1,v.shape[1]),device=device,dtype=torch.float64)),0)*gain;v[:,:201]=0.;return v
 try:
  model.eval();expected=torch.tensor(np.random.default_rng(20261106).normal(0,.004,size=145),device=device,dtype=torch.float64);checks=[]
  with torch.no_grad(),session.training_scope():
   for idx in (16,83):
    t=times[idx];before,h=query(t);v=design(h,t);analytic=(expected@v).reshape(201,201);set_delta(expected);after,_=query(t);actual=(after.double()-before.double());rel=float((actual-analytic).norm()/analytic.norm().clamp_min(1e-30));restore();recovered,_=query(t);checks.append({'frame_index':idx,'time_s':float(t),'relative_l2_delta':rel,'nonzero_delta_norm':float(analytic.norm()),'restore_bitwise_equal':bool(torch.equal(recovered,before))})
  assert all(c['relative_l2_delta']<2e-4 and c['nonzero_delta_norm']>0 and c['restore_bitwise_equal'] for c in checks)
  pre={'status':'PASS','arm':args.arm,'DG_enabled':bool(model.refiner.dg.enabled),'postDG_hook_verified':True,'checks':checks,'source_checkpoint':str(ckpt),'source_checkpoint_sha256':sha(ckpt),'updates':0,'future_truth_used_for_selection':False,'elapsed_s':time.perf_counter()-start};write(OUT/'PREFLIGHT.json',pre);status('preflight_pass',checks=checks)
  if args.precheck:return
  protocol={'arm':args.arm,'source_checkpoint':str(ckpt),'source_checkpoint_sha256':sha(ckpt),'truth':str(REF),'truth_sha256':sha(REF),'fit':'one 145-dimensional readout-only full400 supervised ridge fit, no receiver or PDE loss','design':'postDG unfold3x3 x 16 channels + bias, multiplied by base RMS sqrt(mean(base^2)+1e-6), top row zero','ridge':'1e-6 * trace(Gram)/145','DG':'retained','time_selection':'all 400 reference frames; receiver and first88 not used for selecting delta','original_prior_prediction':str(oldpred) if oldpred else None};write(OUT/'PROTOCOL.json',protocol)
  gram=torch.zeros((145,145),device=device,dtype=torch.float64);rhs=torch.zeros(145,device=device,dtype=torch.float64);initial_sse=0.;energy=0.;old_sse=old_energy=0.;oldarray=np.load(oldpred,mmap_mode='r',allow_pickle=False) if oldpred else None;train_start=time.perf_counter();status('accumulating',done=0,total=400)
  with torch.no_grad(),session.training_scope():
   for i,t in enumerate(times):
    pred,h=query(t);v=design(h,t);y=torch.as_tensor(np.array(truth[i],copy=True),device=device,dtype=torch.float64)/scale;r=(y-pred.double()).reshape(-1);gram.add_(v@v.T);rhs.add_(v@r);initial_sse+=float((r@r).item());energy+=float((y*y).sum());
    if oldarray is not None:
     old=torch.as_tensor(np.array(oldarray[i],copy=True),device=device,dtype=torch.float64);d=pred.double()*scale-old;old_sse+=float(d.square().sum());old_energy+=float(old.square().sum())
    del pred,h,v,y,r
    if (i+1)%25==0:status('accumulating',done=i+1,total=400)
  torch.cuda.synchronize();accumulate_s=time.perf_counter()-train_start;baseline_error=100*math.sqrt(initial_sse/energy);binding=math.sqrt(old_sse/old_energy) if oldarray is not None else None
  if binding is not None and binding>=2e-4:raise RuntimeError(f'Adapted init differs from original full400 output relL2={binding}')
  d=torch.sqrt(torch.diag(gram).clamp_min(1e-300));ridge=float(1e-6*torch.trace(gram)/145.);reg=gram+ridge*torch.eye(145,device=device,dtype=torch.float64);D=torch.sqrt(torch.diag(reg));balance=reg/D[:,None]/D[None,:];delta=(torch.linalg.solve(balance,rhs/D))/D
  assert bool(torch.isfinite(delta).all());condition=float(torch.linalg.cond(balance));predicted_sse=initial_sse-2*float(delta@rhs)+float(delta@(gram@delta));assert predicted_sse<=initial_sse*(1+1e-9)
  set_delta(delta);torch.save({'A':head.A.detach().cpu(),'B':head.B.detach().cpu(),'bias':head.base.bias.detach().cpu(),'delta_fp64':delta.cpu(),'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),'readout_only':True},OUT/'HEAD_FIT.pt')
  status('scoring',done=0,total=400);output=np.lib.format.open_memmap(OUT/'prediction400_pa.npy',mode='w+',dtype=np.float32,shape=(400,201,201));rx=[];final_sse=final_energy=0.;score_start=time.perf_counter()
  with torch.no_grad(),session.training_scope():
   for i,t in enumerate(times):
    pred,_=query(t);pa=(pred*scale).cpu().numpy();output[i]=pa;ref=np.asarray(truth[i],dtype=np.float64);d=pa.astype(np.float64)-ref;final_sse+=float((d*d).sum());final_energy+=float((ref*ref).sum());rx.append(float(pa[2,50]));
    if (i+1)%50==0:status('scoring',done=i+1,total=400)
  output.flush();del output;final_error=100*math.sqrt(final_sse/final_energy);assert final_error<=baseline_error+1e-5,(baseline_error,final_error)
  status('early_prediction',done=0,total=88);earlytimes=np.arange(88,dtype=np.float64)*.00125;rxearly=[]
  with torch.no_grad(),session.training_scope():
   for i,t in enumerate(earlytimes):
    pred,_=query(torch.tensor(t,device=device,dtype=torch.float32));rxearly.append(float(pred[2,50]*scale));
    if (i+1)%22==0:status('early_prediction',done=i+1,total=88)
  reffull=np.load(RUNTIME/'fno_fulltime_supervised/reference/training_reference_pa.npy',mmap_mode='r',allow_pickle=False);rtimes=np.load(RUNTIME/'fno_fulltime_supervised/reference/time_s.npy');frames=np.rint(times.detach().cpu().numpy()/.00125).astype(int);tall=np.r_[earlytimes,times.cpu().numpy()];rpred=np.r_[rxearly,rx];rtruth=np.r_[np.asarray(reffull[:88,2,50]),np.asarray(truth[:,2,50])];
  with (OUT/'receiver_trace.csv').open('w') as stream:stream.write('time_s,ours_Pa,reference_Pa\n');np.savetxt(stream,np.column_stack((tall,rpred,rtruth)),delimiter=',',fmt='%.18e')
  receiver_err=100*float(np.linalg.norm(rpred-rtruth)/np.linalg.norm(rtruth));out={'status':'COMPLETE','arm':args.arm,'scope':protocol['fit'],'source_checkpoint':str(ckpt),'source_checkpoint_sha256':sha(ckpt),'initial_full400_relative_l2_percent':baseline_error,'full400_relative_l2_percent':final_error,'initial_vs_existing_prediction_relative_l2':binding,'receiver_488_relative_l2_percent':receiver_err,'full400_label_count':400,'early88_predict_only':True,'all_original_400_supervised':True,'gram_balance_condition':condition,'ridge':ridge,'predicted_normalized_sse':predicted_sse,'preflight':checks,'timing_s':{'accumulate_400':accumulate_s,'score_and_early':time.perf_counter()-score_start,'total':time.perf_counter()-start},'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'prediction_sha256':sha(OUT/'prediction400_pa.npy'),'receiver_trace_sha256':sha(OUT/'receiver_trace.csv'),'not_architecture_or_unseen_time_comparison':True};write(OUT/'RESULT.json',out);status('complete',initial=baseline_error,final=final_error,receiver488=receiver_err)
 finally:hh.remove();hdg.remove()
if __name__=='__main__':
 try:main()
 except Exception:
  print(traceback.format_exc(),flush=True);raise
