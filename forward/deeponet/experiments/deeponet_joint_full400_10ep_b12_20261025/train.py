"""Exactly ten complete TRAIN500 epochs; complete400 DATA/REC/SIPG gradients."""
from scno_release_paths import local_asset as _scno_local_asset
from pathlib import Path
import inspect, argparse, copy, scno_posix_lock as fcntl, gc, hashlib, importlib, json, math, os, random, signal, subprocess, sys, time, traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
ROOT=Path(_scno_local_asset('project'));RUN=Path(__file__).resolve().parent
sys.path[:0]=[str(RUN),str(ROOT/'src'),str(ROOT/'scripts')]
import train_marmousi_baseline_pde_continue_20261020 as previous
from marmousi_baseline50_data_20261020 import Marmousi500Dataset,epoch_order
from model import RepairDeepONet,ARCH
old=previous.old
PARENT=ROOT/'runs/deeponet_marmousi_repair_20261024/latest.pt'
PARENT_SHA='2a7c3d0cc02e886438216feccb577be0cda30408465634cc4893d144768fd6e0'
SCHEMA='DEEPONET_ACTUAL_BRANCH_JOINT_FULL400_CONTINUATION_V1'
ORIGIN_RUN=ROOT/'runs/deeponet_pod512_full400_10ep_b12_20261025'
ORIGIN_SHA='a58e633c48e5707880ce98b3baf3769266d7954b796ce2901db47fbe3ebf74fe'
ORIGIN=dict(epochs=0,offset=296,updates=26,exposures=296)
CONFIG=dict(schema=SCHEMA,architecture=ARCH,epochs=10,records_per_epoch=500,batch_size=12,updates=421,segment_updates=395,exposures=5000,origin=ORIGIN,warmup_exposures=200,frames=400,chunk_points=32768,seed=20261025,weight_decay=1e-4,grad_clip=1.,phase_policy='actual branch and trunk jointly for every remaining update',learning_rates={'branch_body':1e-4,'branch_head':6e-6,'trunk_body':5e-5,'trunk_head':3e-6},coefficient_loss=False,QR=False,criteria={'train_forecast_mean_max':.85,'val_forecast_mean_max':.85,'val_relative_reduction_min':.15,'val_norm_ratio_min':.2,'val_norm_ratio_max':2.,'val_cosine_mean_min':.5,'val_each_forecast_max':1.1},budget='retain parent26updates/296exposures; 395 newjointupdates; cumulative421updates/5000exposures; tenepochs unchanged')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def configure():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)

def identity():
 return dict(pid=os.getpid(),start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]))

def phase_for_epoch(epoch):return 'joint'
def initial_progress(seed=CONFIG['seed']):
 return dict(epochs=0,offset=296,updates=26,exposures=296,order=list(map(int,epoch_order(500,seed,0))),phase='joint',phase_updates=0,phase_start_updates=26,segment_updates=0,training_seconds=0.,segment_training_seconds=0.)
def next_ids(p):return p['order'][p['offset']:p['offset']+CONFIG['batch_size']] if p['epochs']<10 else []
def advance(p,ids,seed):
 if ids!=next_ids(p) or not ids:raise ValueError('Cursor does not match exact next batch')
 p['updates']+=1;p['segment_updates']+=1;p['phase_updates']+=1;p['exposures']+=len(ids);p['offset']+=len(ids)
 if p['offset']==500:
  p['epochs']+=1;p['offset']=0;p['order']=list(map(int,epoch_order(500,seed,p['epochs']))) if p['epochs']<10 else []
 validate_progress(p,seed)
def expected_segment_state(updates):
 epoch,offset,exposures=0,296,296
 for _ in range(updates):
  take=min(CONFIG['batch_size'],500-offset);offset+=take;exposures+=take
  if offset==500:epoch+=1;offset=0
 return epoch,offset,exposures
def validate_progress(p,seed):
 assert 0<=p['segment_updates']<=CONFIG['segment_updates']
 assert (p['epochs'],p['offset'],p['exposures'])==expected_segment_state(p['segment_updates'])
 assert p['updates']==ORIGIN['updates']+p['segment_updates']
 expected=list(map(int,epoch_order(500,seed,p['epochs']))) if p['epochs']<10 else []
 assert p['order']==expected and p['phase']=='joint'
 assert p['phase_updates']==p['updates']-p['phase_start_updates']
 assert p['phase_start_updates']==ORIGIN['updates'] and p['phase_updates']==p['segment_updates']
 assert ORIGIN['exposures']<=p['exposures']<=5000

def warmup_factor(progress,batch_records):
 start=ORIGIN['exposures']
 return min(1.,(progress['exposures']-start+batch_records)/CONFIG['warmup_exposures'])

def parameter_roles(model):
 branch_head={id(p) for p in model.branch[-1].parameters()}
 trunk_head={id(p) for p in model.trunk.output.parameters()}
 result={}
 for name,p in model.named_parameters():
  if id(p) in branch_head:role='branch_head'
  elif id(p) in trunk_head:role='trunk_head'
  elif name.startswith('trunk.') or name=='output_bias':role='trunk_body'
  else:role='branch_body'
  result[name]=role
 return result

def make_optimizer(model,phase,cfg):
 assert phase=='joint','This run only trains branch and trunk jointly'
 for p in model.parameters():p.requires_grad_(True)
 roles=parameter_roles(model);rates=cfg['learning_rates'];groups=[]
 if set(rates)=={'trunk_body','trunk_head','branch_body','branch_head'}:
  labels=('trunk_body','trunk_head','branch_body','branch_head')
  for label in labels:
   pars=[p for n,p in model.named_parameters() if roles[n]==label]
   groups.append(dict(params=pars,lr=rates[label],base_lr=rates[label],label=label))
 else:
  assert set(rates)=={'joint_trunk','joint_branch'}
  for label in ('trunk','branch'):
   pars=[p for n,p in model.named_parameters() if roles[n].startswith(label+'_')]
   rate=rates['joint_'+label];groups.append(dict(params=pars,lr=rate,base_lr=rate,label=label))
 active=[p for g in groups for p in g['params']]
 assert len(active)==len(list(model.parameters()))==len({id(p) for p in active})
 return torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])

def restore_objects(saved,device='cpu'):
 validate_progress(saved['progress'],saved['config']['seed'])
 assert saved['schema']==SCHEMA and saved['config']==CONFIG
 model=RepairDeepONet(saved['config']['architecture']).to(device);model.load_state_dict(saved['model'],strict=True)
 opt=make_optimizer(model,saved['progress']['phase'],saved['config']);opt.load_state_dict(saved['optimizer'])
 assert old.equal_tree(model.state_dict(),saved['model']) and old.equal_tree(opt.state_dict(),saved['optimizer'])
 return model,opt

def bind_dataset(saved):
 ds=Marmousi500Dataset(saved['stage_config']);expected=copy.deepcopy(saved['data_state']);actual=ds.state_dict()
 expected['dataset_path']=actual['dataset_path'];assert old.equal_tree(expected,actual),'TRAIN data identity drift'
 return ds

def bind_validation(saved):
 root=RUN/'data/validation';complete=json.loads((root/'COMPLETE.json').read_text());meta=json.loads((root/'metadata.json').read_text());verification=json.loads((RUN/'data/VALIDATION_VERIFICATION.json').read_text())
 assert complete['status']=='COMPLETE' and verification['status']=='PASS'
 rows=np.load(root/'rows.npy');sources=np.load(root/'sources_fp64.npy');sources32=np.load(root/'sources_fp32.npy')
 assert rows.tolist()==np.linspace(0,199,16,dtype=int).tolist() and np.array_equal(sources.astype(np.float32),sources32)
 assert meta['validation_rows']==rows.tolist() and len(meta['records'])==16
 for i,row in enumerate(rows):
  item=meta['records'][i];assert item['roster_row']==int(row) and item['file']==f'{row:06d}.npy'
  assert np.array_equal(np.asarray(item['source_registered_fp64']),sources[i])
 verified={r['row']:r['sha256'] for r in verification['records']}
 for name,item in complete['artifacts'].items():
  assert sha(root/name)==item['sha256']
  if name.endswith('.npy') and name[:6].isdigit():assert verified[int(name[:6])]==item['sha256']
 for name,item in meta['input_artifacts'].items():
  assert complete['input_artifacts'][name]['sha256']==item['sha256']==sha(root/name)
 for item in meta['original_metadata'].values():assert sha(item['path'])==item['sha256']
 assert complete['metadata_sha256']==verification['metadata_sha256']==sha(root/'metadata.json')
 assert verification['complete_sha256']==sha(root/'COMPLETE.json')
 baseline=json.loads((RUN/'baseline_failed.json').read_text())
 assert baseline['status']=='COMPLETE' and baseline['parent_sha256']==PARENT_SHA and baseline['row_order']==rows.tolist()
 assert baseline['pressure_scale_pa']==saved['pressure_rms_pa'] and baseline['normalization']==saved['normalization']
 for path,digest in baseline['input_hashes'].items():assert sha(path)==digest
 paths=[root/n for n in ('COMPLETE.json','metadata.json','rows.npy','sources_fp64.npy','sources_fp32.npy')]+[RUN/'data/VALIDATION_VERIFICATION.json',RUN/'baseline_failed.json']
 return {str(path):sha(path) for path in paths}

def check_validation(saved):
 for path,digest in saved['validation_binding'].items():assert sha(path)==digest,('VAL pairing or baseline changed',path)

def freeze_sources():
 # Exercise imports used at runtime before freezing the local dependency closure.
 import eval as evaluation
 files={str(Path(m.__file__).resolve()) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT.resolve())) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file()}
 files.update(str((RUN/n).resolve()) for n in ('model.py','metrics.py','train.py','eval.py','tests.py','tests_training.py','prepare.py','test_prepare.py','CONFIG.json','DESIGN.md','PLAN.md','STABILITY_ADMISSION.json'))
 write(RUN/'SOURCE_MANIFEST.json',dict(files={p:sha(p) for p in sorted(files)}))
 return sha(RUN/'SOURCE_MANIFEST.json')

def check_sources(expected):
 assert sha(RUN/'SOURCE_MANIFEST.json')==expected
 for p,h in json.loads((RUN/'SOURCE_MANIFEST.json').read_text())['files'].items():assert sha(p)==h,('Frozen source changed',p)

def save(saved,model,opt,p,device):
 saved=dict(saved,model=old.cpu_tree(model.state_dict()),optimizer=old.cpu_tree(opt.state_dict()),rng=old.get_rng(device),progress=copy.deepcopy(p))
 tmp=RUN/'latest.pt.tmp'
 with tmp.open('wb') as f:torch.save(saved,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,RUN/'latest.pt')
 write(RUN/'CHECKPOINT.json',dict(status='PASS',sha256=sha(RUN/'latest.pt'),progress={k:v for k,v in p.items() if k!='order'}))
 return saved

def validate_stability(saved=None):
 admission=json.loads((RUN/'STABILITY_ADMISSION.json').read_text())
 assert admission['status']=='PASS' and admission['parent_checkpoint_sha256']==ORIGIN_SHA
 assert admission['selected_learning_rates']==CONFIG['learning_rates']
 assert admission['actual_branch_joint_training'] is True and admission['diagnostic_updates_discarded'] is True
 for path,digest in admission['evidence_files'].items():assert sha(path)==digest,('Stability evidence changed',path)
 if saved is not None:assert saved['stability_admission_sha256']==sha(RUN/'STABILITY_ADMISSION.json')
 return admission

def initialize():
 assert not (RUN/'latest.pt').exists(),'Do not replace a joint continuation'
 validate_stability()
 assert sha(RUN/'parent/latest.pt')==ORIGIN_SHA==sha(ORIGIN_RUN/'latest.pt')
 parent=torch.load(RUN/'parent/latest.pt',map_location='cpu',weights_only=False)
 assert {key:parent['progress'][key] for key in ORIGIN}==ORIGIN
 assert json.loads((ORIGIN_RUN/'STATUS.json').read_text())['status']=='STOPPED_AT_COMPLETE_BATCH'
 saved=copy.deepcopy(parent);saved['schema']=SCHEMA;saved['config']=copy.deepcopy(CONFIG)
 parent_progress=copy.deepcopy(parent['progress'])
 saved['progress'].update(phase='joint',phase_updates=0,phase_start_updates=ORIGIN['updates'],segment_updates=0,segment_training_seconds=0.)
 saved['stability_admission_sha256']=sha(RUN/'STABILITY_ADMISSION.json')
 saved['parent_restore8']=copy.deepcopy(parent.get('restore8'));saved['restore8']=None
 saved['phase_transition']=dict(continuation_parent_run=str(ORIGIN_RUN),continuation_parent_sha256=ORIGIN_SHA,archived_parent_checkpoint=str(RUN/'parent/latest.pt'),failed_baseline_sha256=PARENT_SHA,parent_progress=parent_progress,parent_config=parent['config'],remaining_exposures=4704,segment_updates=395,cumulative_updates=421,optimizer='fresh four-parameter-group AdamW for two jointly trained networks; teacher-code trunk moments are not continued',warmup='new200 record joint-phase warmup within remaining4704 exposures',old_POD_codes='retained only as historical state; never used by training or inference',restore_test='newsegment8 to9; cumulative global34 to35')
 net=RepairDeepONet(CONFIG['architecture']);net.load_state_dict(parent['model'],strict=True);opt=make_optimizer(net,'joint',CONFIG);assert not opt.state
 saved['optimizer']=old.cpu_tree(opt.state_dict())
 ds=bind_dataset(saved);physics=previous.make_physics(ds,saved['stage_config'],'cpu');check_validation(saved)
 assert physics.metadata()==saved['physics_metadata']
 write(RUN/'CONFIG.json',CONFIG);saved['source_manifest_sha256']=freeze_sources()
 for key in ('model','rng','codes','normalization','pressure_rms_pa','physics_scales','data_state','stage_config','validation_binding','qr'):
  assert old.equal_tree(saved[key],parent[key]),key
 for key in ('epochs','offset','updates','exposures','order','training_seconds'):
  assert old.equal_tree(saved['progress'][key],parent_progress[key]),key
 assert saved['rng']['cuda'] is not None
 tmp=RUN/'latest.pt.tmp'
 with tmp.open('wb') as f:torch.save(saved,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,RUN/'latest.pt');check=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=False);assert old.equal_tree(check,saved)
 write(RUN/'CHECKPOINT.json',dict(status='PASS',sha256=sha(RUN/'latest.pt'),progress={k:v for k,v in saved['progress'].items() if k!='order'}))
 write(RUN/'INITIALIZATION.json',dict(status='PASS',parameters=sum(p.numel() for p in net.parameters()),parent_checkpoint_sha256=ORIGIN_SHA,source_manifest_sha256=saved['source_manifest_sha256'],inherited_model_RNG_codes_normalizers_cursor_exact=True,optimizer_continuation=False,optimizer_fresh_groups=[g['label'] for g in opt.param_groups],optimizer_state_entries=0,new_optimizer_updates=0,phase_transition=saved['phase_transition']))
 return saved

def one_backward(model,data,physics,scales,stage,cfg,codes=None,phase='joint',epoch=None):
 assert phase=='joint'
 sup=previous.base.supervision(data['truth'],stage)
 losses=previous.backward_complete(model,'deeponet',data,sup,physics,scales,stage,cfg['chunk_points'])
 losses['coefficient_weight']=0.;losses['coefficient_loss']=0.
 # Every named parameter is active from the first joint update. Report groups separately.
 all_pars=list(model.parameters());group_norms={}
 for label,is_trunk in (('trunk',True),('branch',False)):
  pars=[p for n,p in model.named_parameters() if (n.startswith('trunk.') or n=='output_bias')==is_trunk]
  assert all(p.requires_grad and p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in pars),'Nonfinite/disconnected '+label
  group_norms[label]=float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in pars])))
  assert group_norms[label]>0,'Zero-gradient '+label
 losses['gradient_norms_before_clip']=group_norms
 losses['gradient_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(all_pars,cfg['grad_clip'],error_if_nonfinite=True))
 return losses

def batch(ds,saved,ids,device):
 data=previous.to_device(ds.load_batch(ids),saved['stage_config'],device)
 scales=torch.tensor([saved['physics_scales'][str(i)] for i in ids],device=device,dtype=torch.float32)
 return data,scales,None

def capacity():
 saved=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=False);assert saved['progress']['segment_updates']==0
 check_sources(saved['source_manifest_sha256']);ds=bind_dataset(saved);net=RepairDeepONet().cuda();net.load_state_dict(saved['model'])
 opt=make_optimizer(net,'joint',CONFIG)
 # Allocate the real Adam moment footprint without committing a probe update.
 for group in opt.param_groups:
  for p in group['params']:opt.state[p]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
 physics=previous.make_physics(ds,saved['stage_config'],'cuda');data,scales,codes=batch(ds,saved,list(range(CONFIG['batch_size'])),'cuda')
 torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 losses=one_backward(net,data,physics,scales,saved['stage_config'],CONFIG,codes,'joint',6);torch.cuda.synchronize()
 proof=dict(status='PASS',**identity(),phase='worst active joint; real allocated Adam moments',batch_size=CONFIG['batch_size'],frames=400,query_points_per_record=400*201*201,pde_centers_per_record=398,chunk_points=CONFIG['chunk_points'],optimizer_updates=0,seconds=time.monotonic()-start,allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),losses=losses)
 write(RUN/'CAPACITY.json',proof);print(json.dumps(proof),flush=True)

def capacity_kernel_fingerprints():
 paths=[RUN/'model.py',Path(previous.__file__),Path(previous.base.__file__),Path(old.__file__),ROOT/'scripts/marmousi_baseline50_data_20261020.py',ROOT/'src/baselines/marmousi_pde_20261020/physics.py',ROOT/'src/baselines/marmousi_matched_20261020/models.py']
 return dict(one_backward_sha256=hashlib.sha256(inspect.getsource(one_backward).encode()).hexdigest(),files={str(path.resolve()):sha(path) for path in paths})

def validate_capacity(saved):
 cap=json.loads((RUN/'CAPACITY.json').read_text());binding=json.loads((RUN/'CAPACITY_COMPATIBILITY.json').read_text())
 assert cap['status']=='PASS' and cap['optimizer_updates']==0
 assert (cap['chunk_points'],cap['batch_size'],cap['frames'])==(CONFIG['chunk_points'],CONFIG['batch_size'],CONFIG['frames'])
 assert binding['status']=='PASS' and binding['capacity_sha256']==sha(RUN/'CAPACITY.json')
 assert binding['compatible_final_manifest']==saved['source_manifest_sha256']
 assert binding['kernel_fingerprints']==capacity_kernel_fingerprints()
 assert cap['parent_checkpoint_sha256']==binding['actual_capacity_parent_checkpoint_sha256']
 assert binding['continuation_parent_checkpoint_sha256']==ORIGIN_SHA
 assert binding['same_tensor_geometry'] is True and binding['capacity_auxiliary_loss_removed'] is True

def cpu_check():
 assert not torch.cuda.is_initialized();saved=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=False);net,opt=restore_objects(saved)
 gen=torch.Generator().manual_seed(27);inputs=torch.randn(1,16,17,17,generator=gen)*.01;coords=torch.rand(37,3,generator=gen)
 with torch.no_grad():before=net(inputs,coords)
 del net,opt;gc.collect();net,opt=restore_objects(saved)
 with torch.no_grad():after=net(inputs,coords)
 assert torch.equal(before,after)
 return dict(status='PASS',checkpoint_sha256=sha(RUN/'latest.pt'),updates=saved['progress']['updates'],complete_model_optimizer_cursor_codes_normalizers=True,CPU_prediction_bitwise=True,CUDA_initialized=False)

def isolated_cpu(name):
 proc=subprocess.run([sys.executable,'-B',str(RUN/'train.py'),'cpu-check','--output',str(RUN/name)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
 assert proc.returncode==0,proc.stderr[-3000:]
 return json.loads((RUN/name).read_text())

def run(resume=False):
 lock=(RUN/'training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 saved=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=False);check_sources(saved['source_manifest_sha256'])
 review=json.loads((RUN/'RUN_REVIEW.json').read_text());assert review['status']=='PASS' and review['source_manifest_sha256']==saved['source_manifest_sha256']
 validate_capacity(saved);validate_stability(saved)
 if not resume:assert saved['progress']['segment_updates']==0
 ds=bind_dataset(saved);net,opt=restore_objects(saved,'cuda');old.set_rng(saved['rng'],'cuda');p=copy.deepcopy(saved['progress']);physics=previous.make_physics(ds,saved['stage_config'],'cuda')
 stop=[]
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.append(True))
 write(RUN/'WORKER.json',dict(**identity(),command=sys.argv,source_manifest_sha256=saved['source_manifest_sha256']))
 def status(state,**extra):
  value=dict(status=state,**identity(),progress={k:v for k,v in p.items() if k!='order'},**extra);write(RUN/'STATUS.json',value);print(json.dumps(value),flush=True)
 status('TRAINING',training_prediction_mode='actual input branch and trunk jointly; no POD coefficient targets or QR')
 while p['epochs']<10 and not stop:
  check_sources(saved['source_manifest_sha256']);phase=phase_for_epoch(p['epochs'])
  assert phase==p['phase']=='joint'
  if p['segment_updates']==8 and not (saved.get('restore8') or {}).get('restored_objects_awaiting_or_committed_update9'):
   ids=next_ids(p);data,_,_=batch(ds,saved,ids,'cuda')
   with torch.no_grad():before=previous.predict_normalized(net,'deeponet',data,CONFIG['chunk_points']).cpu()
   saved=save(saved,net,opt,p,'cuda');del net,opt;gc.collect();torch.cuda.empty_cache();net,opt=restore_objects(saved,'cuda');old.set_rng(saved['rng'],'cuda')
   with torch.no_grad():after=previous.predict_normalized(net,'deeponet',data,CONFIG['chunk_points']).cpu()
   assert torch.equal(before,after) and old.equal_tree(old.get_rng('cuda'),saved['rng'])
   saved['restore8']=dict(global_restored_update=ORIGIN['updates']+8,segment_restored_update=8,global_next_update=ORIGIN['updates']+9,segment_next_update=9,status='RESTORED_AWAITING_UPDATE9',full400_prediction_bitwise=True,complete_model_Adam_RNG=True,restored_objects_awaiting_or_committed_update9=True,committed_update9=False)
   write(RUN/'CUDA_RESTORE8_9.json',saved['restore8']);del before,after,data
   saved=save(saved,net,opt,p,'cuda');isolated_cpu('CPU_RESTORE_STEP008.json')
  ids=next_ids(p);epoch=p['epochs'];tick=time.monotonic();data,scales,codes=batch(ds,saved,ids,'cuda');opt.zero_grad(set_to_none=True)
  for group in opt.param_groups:group['lr']=group['base_lr']*warmup_factor(p,len(ids))
  losses=one_backward(net,data,physics,scales,saved['stage_config'],CONFIG,codes,phase,epoch);opt.step();torch.cuda.synchronize()
  elapsed=time.monotonic()-tick;advance(p,ids,CONFIG['seed']);p['training_seconds']+=elapsed;p['segment_training_seconds']+=elapsed
  if p['segment_updates']==9:
   assert saved['restore8']['status']=='RESTORED_AWAITING_UPDATE9';saved['restore8'].update(status='PASS',committed_update9=True);write(RUN/'CUDA_RESTORE8_9.json',saved['restore8'])
  saved=save(saved,net,opt,p,'cuda')
  record=dict(update=p['updates'],segment_update=p['segment_updates'],epoch=epoch+1,phase=phase,ids=ids,seconds=time.monotonic()-tick,losses=losses,learning_rates=[g['lr'] for g in opt.param_groups])
  with (RUN/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
  status('TRAINING',latest=record);del data,scales,codes
  if p['offset']==0:
   import eval as evaluation
   evaluation.evaluate_model(net,saved,'train',[0,166,333,499],RUN/f'TRAIN_PANEL_EPOCH{p["epochs"]:02d}.json')
 if stop:status('STOPPED_AT_COMPLETE_BATCH');return
 assert (p['epochs'],p['updates'],p['exposures'])==(10,CONFIG['updates'],5000)
 isolated_cpu('FINAL_CPU_RESTORE.json');status('TRAINING_COMPLETE_FINAL_EVALUATION')
 del net,opt,physics,ds;gc.collect();torch.cuda.empty_cache()
 import eval as evaluation
 evaluation.final_evaluation();status('COMPLETE',scientific_result=json.loads((RUN/'RESULT.json').read_text())['status'])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['initialize','capacity','cpu-check','train']);parser.add_argument('--resume',action='store_true');parser.add_argument('--output');args=parser.parse_args();configure()
 if args.mode=='initialize':initialize();write(RUN/'INITIAL_CPU_RESTORE.json',cpu_check())
 elif args.mode=='capacity':capacity()
 elif args.mode=='cpu-check':write(args.output or RUN/'CPU_RESTORE.json',cpu_check())
 else:run(args.resume)
if __name__=='__main__':
 try:main()
 except Exception:
  write(RUN/'LAST_ERROR.json',dict(**identity(),traceback=traceback.format_exc()));raise
