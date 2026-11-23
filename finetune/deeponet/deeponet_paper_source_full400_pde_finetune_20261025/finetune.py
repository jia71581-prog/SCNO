"""Independent all400 single-paper-source FIT; pretrained TRAIN500 remains paused."""
from scno_release_paths import local_asset as _scno_local_asset
from pathlib import Path
import argparse,copy,scno_posix_lock as fcntl,gc,hashlib,json,math,os,shutil,signal,subprocess,sys,time,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=Path(_scno_local_asset('project'));RUN=Path(__file__).resolve().parent
BASE=ROOT/'runs/deeponet_joint_full400_10ep_b12_20261025'
sys.path[:0]=[str(BASE),str(ROOT/'src'),str(ROOT/'scripts')]
import train as inherited
from model import RepairDeepONet,ARCH
from baselines.marmousi_pde_20261020.physics import NonuniformSIPG
previous=inherited.previous;old=inherited.old
PARENT=BASE/'latest.pt';PARENT_SHA='0ccd463431d200f466db70b0878e8cf26bf0ee295d477e0d7dad2a7455cfb168'
INPUTS=ROOT/'runs/paper_marmousi_matched_baselines_20261020/inputs.npz'
INPUTS_SHA='16a40a4a192a74d96b6dd8f0fbad464d29380952fb3f68a8988b541306ef9637'
REFERENCE=ROOT/'runs/paper_marmousi201_single_source_fullfit_20261020/case_0/reference_pa.npy'
REFERENCE_SHA='7ddda0be80017d13e1a94a3ca787c0658ce83d1ba33cd558821ee39fcae7c669'
SCHEMA='DEEPONET_PAPER_SOURCE_ALL400_PDE_FIT_V1'
CONFIG=dict(schema=SCHEMA,max_steps=2000,architecture=ARCH,chunk_candidates=[4096,1024,256,128],initial_chunk_points=4096,learning_rates={'branch_body':1e-4,'branch_head':6e-6,'trunk_body':5e-5,'trunk_head':3e-6},weight_decay=1e-4,grad_clip=1.,evaluation_every=100,physics_chunk_frames=2,source_id='paper_source',scope='Independent all400 supervised FIT, not heldout forecasting or completion of original TRAIN500 pretraining',data_objective='.5 whole-trajectory relativeMSE +.5 mean per-frame relativeMSE',receiver_weight=.01,pde_weight=.001,best_selection='minimum complete FIT objective among initial/every100/final evaluations; never VAL')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def append(path,value):
 with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def configure():inherited.configure()
def identity():return dict(pid=os.getpid(),start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]))
def initial_progress(chunk=4096):return dict(updates=0,chunk_points=chunk,training_seconds=0.,evaluation_seconds=0.,failed_attempt_seconds=0.,failed_attempts=0,last_evaluation_step=-1,last_CPU_verified_update=0,committed_fit_record_exposures=0)
def validate_progress(p,cfg):
 assert 0<=p['updates']<=cfg['max_steps'] and p['committed_fit_record_exposures']==p['updates']
 assert p['chunk_points'] in cfg['chunk_candidates'] and p['last_evaluation_step']<=p['updates']
 assert all(math.isfinite(p[k]) and p[k]>=0 for k in ('training_seconds','evaluation_seconds','failed_attempt_seconds'))
def next_chunk(chunk,candidates):
 position=candidates.index(chunk)
 return candidates[position+1] if position+1<len(candidates) else None

def mixed_supervision(truth,stage):
 sup=previous.base.supervision(truth,stage)
 frame=truth.square().mean((-2,-1),dtype=torch.float32)
 assert bool(torch.isfinite(frame).all() and (frame>0).all()),'Every complete reference frame must have positive finite energy; no epsilon'
 sup['frame_weights']=(.5+.5*sup['energy'][:,None]/frame).detach()
 sup['frame_energy']=frame.detach()
 return sup

def load_data(saved,device='cpu'):
 """Paper time axis is read verbatim from inputs.npz; never use TRAIN times."""
 for path,digest in saved['data_bindings'].items():assert sha(path)==digest,('Data binding changed',path)
 stage=saved['stage_config'];data=old.load_data(stage,device)
 with np.load(stage['inputs_path'],allow_pickle=False) as z:
  velocity=z['velocity_mps'].copy();source=z['source_model_fp32'].copy();times=z['time_s'].copy()
 assert np.array_equal(source,np.asarray([868.0423,688.824,15.,.1,1.],np.float32))
 assert np.array_equal(velocity,np.load(ROOT/'data/marmousi_pglopt_train500_raw400_20261020/velocity201.npy',allow_pickle=False))
 data['truth']=data['truth'][None];data['inputs']=torch.cat((data['inputs'],data['truth'][:,:8]),1)
 data['sources']=torch.from_numpy(source[None]).to(device);data['ids']=['paper_source']
 data['velocity']=torch.from_numpy(velocity).to(device)
 physics=NonuniformSIPG(data['times'],data['velocity'],**stage['physics'])
 if 'paper_times' in saved:assert torch.equal(data['times'].cpu(),saved['paper_times'])
 if 'physics_metadata' in saved:assert physics.metadata()==saved['physics_metadata']
 scales=saved.get('physics_scales');scales=None if scales is None else scales.to(device)
 return data,physics,scales,mixed_supervision(data['truth'],stage)

def make_optimizer(net,cfg):return inherited.make_optimizer(net,'joint',cfg)
def restore_objects(saved,device='cpu',restore_rng=True):
 assert saved['schema']==SCHEMA;validate_progress(saved['progress'],saved['config'])
 net=RepairDeepONet(saved['config']['architecture']).to(device);net.load_state_dict(saved['model'],strict=True)
 opt=make_optimizer(net,saved['config']);opt.load_state_dict(saved['optimizer'])
 assert old.equal_tree(net.state_dict(),saved['model']) and old.equal_tree(opt.state_dict(),saved['optimizer'])
 if restore_rng:old.set_rng(saved['rng'],device)
 return net,opt

def atomic_checkpoint(saved,path=None):
 path=Path(path or RUN/'latest.pt');tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('wb') as f:torch.save(saved,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 if path==RUN/'latest.pt':write(RUN/'CHECKPOINT.json',dict(status='PASS',checkpoint_sha256=sha(path),progress=saved['progress']))

def snapshot(saved,net,opt,p,device):
 return dict(saved,model=old.cpu_tree(net.state_dict()),optimizer=old.cpu_tree(opt.state_dict()),progress=copy.deepcopy(p),rng=old.get_rng(device))

def freeze_sources():
 sys.path.insert(0,str(RUN));import evaluation
 files={str(Path(m.__file__).resolve()) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT.resolve())) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file()}
 files.update(str((RUN/n).resolve()) for n in ('finetune.py','evaluation.py','cpu_tests.py','PROTOCOL.md','CONFIG.json'))
 write(RUN/'SOURCE_MANIFEST.json',dict(files={p:sha(p) for p in sorted(files)}));return sha(RUN/'SOURCE_MANIFEST.json')
def check_sources(saved):
 assert sha(RUN/'SOURCE_MANIFEST.json')==saved['source_manifest_sha256']
 for p,d in json.loads((RUN/'SOURCE_MANIFEST.json').read_text())['files'].items():assert sha(p)==d,('Frozen source changed',p)

def prepare():
 assert not torch.cuda.is_initialized() and not (RUN/'latest.pt').exists()
 assert sha(PARENT)==PARENT_SHA and sha(INPUTS)==INPUTS_SHA and sha(REFERENCE)==REFERENCE_SHA
 parent=torch.load(PARENT,map_location='cpu',weights_only=False)
 assert (parent['progress']['epochs'],parent['progress']['updates'],parent['progress']['exposures'])==(2,85,1000)
 stage=copy.deepcopy(parent['stage_config']);stage.update(inputs_path=str(INPUTS),reference_path=str(REFERENCE),pressure_rms_pa=parent['pressure_rms_pa'])
 stage['training'].update(receiver_weight=.01,pde_weight=.001,physics_chunk_frames=2)
 plot=ROOT/'outputs/deeponet_vs_paper_marmousi_20261025/snapshot_arrays.npz'
 saved=dict(schema=SCHEMA,config=copy.deepcopy(CONFIG),stage_config=stage,normalization=copy.deepcopy(parent['normalization']),pressure_rms_pa=parent['pressure_rms_pa'],data_bindings={str(INPUTS):INPUTS_SHA,str(REFERENCE):REFERENCE_SHA,str(plot):sha(plot)},parent_checkpoint_path=str(PARENT),parent_checkpoint_sha256=PARENT_SHA,parent_pretraining_progress=copy.deepcopy(parent['progress']),progress=initial_progress(),rng=copy.deepcopy(parent['rng']),restore8=None,best_fit_objective=None,best_fit_step=None,last_evaluation=None)
 assert saved['pressure_rms_pa']==1.252277344221455e-8
 data,physics,_,sup=load_data(saved,'cpu');diag=physics.fixed_scales(data['truth'],data['sources'],chunk_frames=2,return_diagnostics=True)
 saved.update(paper_times=data['times'].cpu().clone(),paper_source=data['sources'].cpu().clone(),physics_scales=diag['scales'].cpu().clone(),physics_metadata=physics.metadata(),reference_physics_diagnostics=old.cpu_tree(diag),reference_frame_energy=sup['frame_energy'].cpu().clone())
 net=RepairDeepONet(CONFIG['architecture']);net.load_state_dict(parent['model'],strict=True);opt=make_optimizer(net,CONFIG)
 assert sum(p.numel() for p in net.parameters())==14596489 and len(list(net.parameters()))==39 and not opt.state
 saved.update(model=old.cpu_tree(net.state_dict()),optimizer=old.cpu_tree(opt.state_dict()))
 assert old.equal_tree(saved['model'],parent['model']) and old.equal_tree(saved['rng'],parent['rng'])
 write(RUN/'CONFIG.json',CONFIG);saved['source_manifest_sha256']=freeze_sources();atomic_checkpoint(saved)
 write(RUN/'PREPARE.json',dict(status='PASS',parent_checkpoint_sha256=PARENT_SHA,source_manifest_sha256=saved['source_manifest_sha256'],parameters=14596489,optimizer='fresh four-group AdamW, no parent moments inherited',paper_time_axis_sha256=physics.metadata()['time_axis_payload_sha256'],parent_train_time_axis_sha256=parent['physics_metadata']['time_axis_payload_sha256'],physics_scales=saved['physics_scales'].tolist(),all400_frame_energies_positive=True,source_fp32=saved['paper_source'].tolist(),pretraining_not_completed=True,scope=CONFIG['scope']))
 write(RUN/'INITIAL_CPU_RESTORE.json',cpu_verify());return saved

@torch.no_grad()
def witness(net,data,chunk):
 slots=torch.tensor([0,133,266,399],device=data['times'].device);times=data['times'][slots]
 branch=net.encode_branch(data['inputs']);shape=(4,*data['shape'][1:]);result=torch.empty((1,math.prod(shape)),device=data['times'].device)
 for start in range(0,result.shape[1],chunk):
  stop=min(start+chunk,result.shape[1]);coords,*_=old.point_coordinates(start,stop,shape,times,data['norm']);result[:,start:stop]=net.decode_trunk(branch,coords)
 return result.cpu()
def cpu_verify():
 assert not torch.cuda.is_initialized();saved=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=False);net,opt=restore_objects(saved,'cpu',True)
 gen=torch.Generator().manual_seed(27);x=torch.randn(1,16,17,17,generator=gen)*.01;q=torch.rand(37,3,generator=gen)
 with torch.no_grad():a=net(x,q)
 del net,opt;gc.collect();net,opt=restore_objects(saved,'cpu',True)
 with torch.no_grad():b=net(x,q)
 assert torch.equal(a,b)
 actual_rng=old.get_rng('cpu')
 assert all(old.equal_tree(actual_rng[k],saved['rng'][k]) for k in ('python','numpy','torch'))
 return dict(status='PASS',CPU_RNG_exact=True,CUDA_RNG_archived=saved['rng']['cuda'] is not None,updates=saved['progress']['updates'],checkpoint_sha256=sha(RUN/'latest.pt'),complete_model_optimizer_RNG_cursor_scales_present=True,CPU_prediction_bitwise=True,CUDA_initialized=False)
def isolated_cpu(step):
 path=RUN/f'CPU_RESTORE_STEP{step:04d}.json';p=subprocess.run([sys.executable,'-B',str(RUN/'finetune.py'),'cpu-check','--output',str(path)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
 assert p.returncode==0,p.stderr[-3000:];return json.loads(path.read_text())

def gradient_summary(net):
 roles=inherited.parameter_roles(net);groups={k:[] for k in CONFIG['learning_rates']}
 for n,p in net.named_parameters():
  assert p.requires_grad and p.grad is not None and bool(torch.isfinite(p.grad).all()),('Nonfinite/disconnected gradient',n)
  groups[roles[n]].append(p.grad.norm())
 return {k:float(torch.linalg.vector_norm(torch.stack(v))) for k,v in groups.items()}
def parameter_change(before,after,roles):
 sums={k:[0.,0.] for k in CONFIG['learning_rates']}
 for n,role in roles.items():
  a=before[n].double();b=after[n].double();assert bool(torch.isfinite(b).all())
  sums[role][0]+=float((b-a).square().sum());sums[role][1]+=float(a.square().sum())
 return {k:dict(absolute_l2=math.sqrt(v[0]),relative_l2=math.sqrt(v[0]/v[1]) if v[1]>0 else None) for k,v in sums.items()}
def is_oom(exc):return isinstance(exc,torch.cuda.OutOfMemoryError) or ('CUDA out of memory' in str(exc))

def capacity():
 saved=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=False);check_sources(saved);attempts=[]
 for chunk in saved['config']['chunk_candidates']:
  tick=time.monotonic();net=opt=data=physics=scales=sup=None
  try:
   net,opt=restore_objects(saved,'cuda');data,physics,scales,sup=load_data(saved,'cuda')
   for g in opt.param_groups:
    for p in g['params']:
     if p not in opt.state:opt.state[p]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
   del g,p
   torch.cuda.reset_peak_memory_stats();loss=previous.backward_complete(net,'deeponet',data,sup,physics,scales,saved['stage_config'],chunk)
   grad=gradient_summary(net);torch.cuda.synchronize();attempts.append(dict(chunk_points=chunk,status='PASS',seconds=time.monotonic()-tick,losses=loss,gradient_norms=grad,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
   result=dict(status='PASS',source_manifest_sha256=saved['source_manifest_sha256'],checkpoint_sha256=sha(RUN/'latest.pt'),selected_chunk_points=chunk,optimizer_updates=0,frames=400,spatial_nodes=40401,PDE_centers=398,attempts=attempts);write(RUN/'CAPACITY.json',result);print(json.dumps(result),flush=True);return
  except Exception as exc:
   if not is_oom(exc):raise
   message=str(exc);exc.__traceback__=None
   attempts.append(dict(chunk_points=chunk,status='CUDA_OOM',seconds=time.monotonic()-tick,error=message));write(RUN/'CAPACITY_ATTEMPTS.json',attempts)
  finally:
   del net,opt,data,physics,scales,sup;gc.collect();torch.cuda.empty_cache()
 raise RuntimeError('All permitted query chunks OOM; no optimizer update committed')

def run(resume=False):
 lock=(RUN/'fit.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 saved=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=False);check_sources(saved)
 review=json.loads((RUN/'RUN_REVIEW.json').read_text());cap=json.loads((RUN/'CAPACITY.json').read_text())
 assert review['status']=='PASS' and review['source_manifest_sha256']==saved['source_manifest_sha256']
 assert cap['status']=='PASS' and cap['optimizer_updates']==0 and cap['source_manifest_sha256']==saved['source_manifest_sha256']
 if not resume:assert saved['progress']['updates']==0
 net,opt=restore_objects(saved,'cuda');data,physics,scales,sup=load_data(saved,'cuda');p=copy.deepcopy(saved['progress'])
 p['chunk_points']=min(p['chunk_points'],cap['selected_chunk_points'])
 stop=[]
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.append(True))
 write(RUN/'WORKER.json',dict(**identity(),command=sys.argv,source_manifest_sha256=saved['source_manifest_sha256']))
 sys.path.insert(0,str(RUN));import evaluation
 def save_now():
  nonlocal saved
  saved=snapshot(saved,net,opt,p,'cuda');atomic_checkpoint(saved)
 def report(state,**extra):
  value=dict(status=state,**identity(),progress=p,**extra);write(RUN/'STATUS.json',value);print(json.dumps(value),flush=True)
 report('RUNNING_SINGLE_SOURCE_FIT')
 attempt_id=sum(1 for line in (RUN/'attempts.jsonl').open()) if (RUN/'attempts.jsonl').exists() else 0
 while True:
  if stop or (RUN/'STOP').exists():save_now();report('STOPPED_AT_COMPLETE_UPDATE');return
  check_sources(saved);u=p['updates'];evaluate_due=(u==0 or u%100==0 or u==saved['config']['max_steps']) and p['last_evaluation_step']!=u
  needs_restore=u==8 and ((saved.get('restore8') or {}).get('runner_pid')!=os.getpid())
  if u==saved['config']['max_steps'] and not evaluate_due:break
  kind='evaluate' if evaluate_due else 'restore8' if needs_restore else 'train';tick=time.monotonic();attempt_id+=1;stepped=False
  append(RUN/'attempts.jsonl',dict(attempt_id=attempt_id,event='START',kind=kind,updates_before=u,chunk_points=p['chunk_points'],**identity()))
  try:
   if kind=='evaluate':
    result=evaluation.evaluate(net,dict(saved,progress=p),data,physics,scales,sup,u,RUN/'evaluations')
    objective=float(result['objective']['loss']);assert math.isfinite(objective)
    p['last_evaluation_step']=u;p['evaluation_seconds']+=time.monotonic()-tick;saved['last_evaluation']=result
    improved=saved.get('best_fit_objective') is None or objective<saved['best_fit_objective']
    if improved:saved['best_fit_objective']=objective;saved['best_fit_step']=u
    save_now()
    if improved:
     shutil.copy2(RUN/'latest.pt',RUN/'best_fit.pt.tmp');os.replace(RUN/'best_fit.pt.tmp',RUN/'best_fit.pt');write(RUN/'BEST_FIT.json',dict(step=u,complete_fit_objective=objective,checkpoint_sha256=sha(RUN/'best_fit.pt'),selection=saved['config']['best_selection']))
   elif kind=='restore8':
    before=witness(net,data,p['chunk_points']);save_now();del net,opt;gc.collect();torch.cuda.empty_cache()
    saved=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=False);net,opt=restore_objects(saved,'cuda');after=witness(net,data,p['chunk_points'])
    assert torch.equal(before,after) and old.equal_tree(old.get_rng('cuda'),saved['rng'])
    saved['restore8']=dict(status='RESTORED_AWAITING_UPDATE9',runner_pid=os.getpid(),restored_update=8,next_update=9,model_Adam_RNG_complete=True,fixed_four_full_spatial_frames_bitwise=True,slots=[0,133,266,399],committed_update9=False)
    write(RUN/'CUDA_RESTORE8_9.json',saved['restore8']);save_now();del before,after
   else:
    opt.zero_grad(set_to_none=True);loss=previous.backward_complete(net,'deeponet',data,sup,physics,scales,saved['stage_config'],p['chunk_points'])
    grads=gradient_summary(net);norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),saved['config']['grad_clip'],error_if_nonfinite=True));opt.step();stepped=True;torch.cuda.synchronize()
    after=old.cpu_tree(net.state_dict());changes=parameter_change(saved['model'],after,inherited.parameter_roles(net))
    p['updates']+=1;p['committed_fit_record_exposures']+=1;p['training_seconds']+=time.monotonic()-tick
    if p['updates']==9:
     assert saved['restore8']['status']=='RESTORED_AWAITING_UPDATE9' and saved['restore8']['runner_pid']==os.getpid()
     saved['restore8'].update(status='PASS',committed_update9=True);write(RUN/'CUDA_RESTORE8_9.json',saved['restore8'])
    row=dict(update=p['updates'],losses=loss,gradient_norms_before_clip=grads,total_gradient_norm_before_clip=norm,relative_parameter_updates=changes,actual_chunk_points=p['chunk_points'],seconds=time.monotonic()-tick,learning_rates={g['label']:g['lr'] for g in opt.param_groups})
    saved['last_update_metrics']=row;save_now();append(RUN/'metrics.jsonl',row);report('RUNNING_SINGLE_SOURCE_FIT',latest=row)
    if p['updates']%100==0:
     proof=isolated_cpu(p['updates']);assert proof['updates']==p['updates'];p['last_CPU_verified_update']=p['updates'];save_now()
   append(RUN/'attempts.jsonl',dict(attempt_id=attempt_id,event='END',kind=kind,outcome='COMMITTED' if kind=='train' else 'COMPLETE',updates_after=p['updates'],seconds=time.monotonic()-tick,optimizer_step_executed=stepped))
  except Exception as exc:
   elapsed=time.monotonic()-tick;message=str(exc);oom=is_oom(exc)
   append(RUN/'attempts.jsonl',dict(attempt_id=attempt_id,event='END',kind=kind,outcome='CUDA_OOM' if oom else 'FAILED',seconds=elapsed,optimizer_step_executed=stepped,error=message))
   if not oom:
    report('FAILED',operation=kind,error=message);raise
   smaller=next_chunk(p['chunk_points'],saved['config']['chunk_candidates'])
   if smaller is None:report('OOM_BLOCKED_AT_MINIMUM_CHUNK',error=message);raise
   # Drop the failed autograd traceback before recreating CUDA objects.
   exc.__traceback__=None
   before=after=result=None
   del net,opt;gc.collect();torch.cuda.empty_cache();saved=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=False);net,opt=restore_objects(saved,'cuda');p=copy.deepcopy(saved['progress'])
   p['chunk_points']=smaller;p['failed_attempts']+=1;p['failed_attempt_seconds']+=elapsed
   if kind=='train':p['training_seconds']+=elapsed
   else:p['evaluation_seconds']+=elapsed
   if p['updates']==8:saved['restore8']=None
   save_now();report('RETRY_SMALLER_CHUNK',failed_operation=kind,failed_seconds=elapsed)
 report('FIT_EFFECTIVE' if saved['last_evaluation']['single_source_fit_effective'] else 'FIT_NOT_EFFECTIVE',generalization_proven=False,pretraining_still_paused=True,best_fit_step=saved['best_fit_step'])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','capacity','cpu-check','train']);parser.add_argument('--resume',action='store_true');parser.add_argument('--output');args=parser.parse_args();configure()
 if args.mode=='prepare':prepare()
 elif args.mode=='capacity':capacity()
 elif args.mode=='cpu-check':write(args.output or RUN/'CPU_RESTORE.json',cpu_verify())
 else:run(args.resume)
if __name__=='__main__':
 try:main()
 except Exception:
  failure=dict(**identity(),traceback=traceback.format_exc())
  write(RUN/'LAST_ERROR.json',failure)
  receipt=json.loads((RUN/'CHECKPOINT.json').read_text()) if (RUN/'CHECKPOINT.json').exists() else {}
  write(RUN/'STATUS.json',dict(status='FAILED',**identity(),progress=receipt.get('progress'),error_file=str(RUN/'LAST_ERROR.json')))
  raise
