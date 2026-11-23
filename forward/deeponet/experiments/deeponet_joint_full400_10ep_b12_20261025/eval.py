"""Final all-TRAIN500 and fixed-VAL16 physical-Pa macro scoring; actual branch only."""
from pathlib import Path
import argparse,gc,json,sys,time
import numpy as np
import torch
RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN))
import train as run
from metrics import field_metrics,aggregate,framewise_field_metrics,aggregate_framewise

def validation_batch(saved,row):
 root=RUN/'data/validation';metadata=json.loads((root/'metadata.json').read_text());rows=np.load(root/'rows.npy');loc=list(map(int,rows)).index(row)
 verified=json.loads((RUN/'data/VALIDATION_VERIFICATION.json').read_text());assert verified['status']=='PASS'
 evidence={r['row']:r for r in verified['records']}[row];path=root/f'{row:06d}.npy';assert run.sha(path)==evidence['sha256']
 truth=np.load(path,allow_pickle=False);assert truth.shape==(400,201,201) and truth.dtype==np.float32
 source=np.load(root/'sources_fp32.npy')[loc];norm=saved['normalization']
 velocity=np.load(run.ROOT/'data/marmousi_pglopt_train500_raw400_20261020/velocity201.npy')
 times=np.load(run.ROOT/'data/marmousi_pglopt_train500_raw400_20261020/times_fp32.npy')
 inputs=np.empty((1,8,201,201),np.float32);inputs[0,0]=(velocity-norm['velocity_mean'])/norm['velocity_std'];inputs[0,1:6]=((source-np.array(norm['source_mean'],np.float32))/np.array(norm['source_std'],np.float32))[:,None,None]
 inputs[0,6]=np.arange(201,dtype=np.float32)[None,:]*np.float32(10/norm['coord_extent_m'][0]);inputs[0,7]=np.arange(201,dtype=np.float32)[:,None]*np.float32(10/norm['coord_extent_m'][1])
 return dict(inputs=torch.from_numpy(inputs),pressure_pa=torch.from_numpy(truth[None]),times=torch.from_numpy(times),scale=saved['pressure_rms_pa'],ids=[row],source_model_fp32=source[None])

@torch.no_grad()
def evaluate_model(model,saved,split,rows,output):
 device=next(model.parameters()).device;was_training=model.training;model.eval()
 ds=run.bind_dataset(saved) if split=='train' else None
 if split!='train':run.check_validation(saved)
 receivers=saved['stage_config']['training']['receiver_indices'];primary={};audit={};framewise={};framewise32={};started=time.monotonic()
 # Shared spacetime trunk is evaluated once per configured record batch, including tails.
 for begin in range(0,len(rows),saved['config']['batch_size']):
  selected=list(rows[begin:begin+saved['config']['batch_size']])
  if ds is not None:cpu=ds.load_batch(selected)
  else:
   parts=[validation_batch(saved,row) for row in selected]
   cpu=dict(inputs=torch.cat([p['inputs'] for p in parts]),pressure_pa=torch.cat([p['pressure_pa'] for p in parts]),times=parts[0]['times'],scale=parts[0]['scale'],ids=selected,source_model_fp32=np.concatenate([p['source_model_fp32'] for p in parts]));del parts
  data=run.previous.to_device(cpu,saved['stage_config'],device)
  pred=(run.previous.predict_normalized(model,'deeponet',data,saved['config']['chunk_points'])*data['scale']).cpu()
  truth=cpu['pressure_pa'];scores=field_metrics(pred,truth,receivers);scores32=field_metrics(pred,truth,receivers,torch.float32)
  temporal=framewise_field_metrics(pred,truth,receivers);temporal32=framewise_field_metrics(pred,truth,receivers,torch.float32)
  for name,items in scores.items():primary.setdefault(name,[]).extend(dict(row=int(row),**item) for row,item in zip(selected,items))
  for name,items in scores32.items():audit.setdefault(name,[]).extend(dict(row=int(row),**item) for row,item in zip(selected,items))
  for name,items in temporal.items():framewise.setdefault(name,[]).extend(dict(row=int(row),**item) for row,item in zip(selected,items))
  for name,items in temporal32.items():framewise32.setdefault(name,[]).extend(dict(row=int(row),**item) for row,item in zip(selected,items))
  progress=dict(status='RUNNING',split=split,completed=len(primary['full400']),total=len(rows),last_row=int(selected[-1]),elapsed_seconds=time.monotonic()-started)
  run.write(Path(output).with_suffix('.progress.json'),progress);print(json.dumps(dict(evaluation=progress)),flush=True)
  del cpu,data,pred,truth
 result=dict(status='COMPLETE',split=split,rows=list(map(int,rows)),prediction='actual branch(input original8+known8); no teacher codes',batch_size=saved['config']['batch_size'],pressure_units='Pa',primary_arithmetic='FP64 subtraction/squares/dot/reductions after FP32 model and physical Pa multiplication',audit_arithmetic='FP32 subtraction/squares/dot/reductions and FP32 chunk accumulator; final scalar ratios/square roots and macro means use Python float (FP64)',windows=primary,summary={k:aggregate(v) for k,v in primary.items()},fp32_audit=dict(windows=audit,summary={k:aggregate(v) for k,v in audit.items()}),framewise_relative_l2=dict(definition='sqrt(sum_spatial((predictionPa-referencePa)^2)/sum_spatial(referencePa^2)) perframe; keep reference energy strictly>0; no epsilon or amplitude cutoff',windows=framewise,summary={k:aggregate_framewise(v) for k,v in framewise.items()},fp32_audit=dict(windows=framewise32,summary={k:aggregate_framewise(v) for k,v in framewise32.items()}),baseline_comparison='UNAVAILABLE: frozen failedbaseline has only whole-trajectory metrics; framewise reported absolute only'),normalization_refitted=False,elapsed_seconds=time.monotonic()-started)
 run.write(output,result);model.train(was_training);return result

def assess_criteria(train,val,baseline,cfg):
 ts=train['summary']['forecast392'];vs=val['summary']['forecast392'];bs=baseline['summary']['forecast392'];criteria=cfg['criteria']
 required=[(train,500),(val,16),(baseline,16)]
 complete=all(r['status']=='COMPLETE' and len(r['rows'])==n and all(v['records']==n and v['defined_relative_l2']==n and v['defined_norm_ratio']==n and v['all_finite'] for v in r['summary'].values()) for r,n in required)
 forecast_defined=all(s['macro_relative_l2'] is not None for s in (ts,vs,bs))
 reduction=1-vs['macro_relative_l2']/bs['macro_relative_l2'] if forecast_defined and bs['macro_relative_l2']>0 else None
 vals=val['windows']['forecast392'];checks=dict(complete_finite_records=complete,train_forecast_mean=ts['macro_relative_l2'] is not None and ts['macro_relative_l2']<=criteria['train_forecast_mean_max'],val_forecast_mean=vs['macro_relative_l2'] is not None and vs['macro_relative_l2']<=criteria['val_forecast_mean_max'],val_relative_improvement=reduction is not None and reduction>=criteria['val_relative_reduction_min'],val_each_amplitude=all(r['norm_ratio'] is not None and criteria['val_norm_ratio_min']<=r['norm_ratio']<=criteria['val_norm_ratio_max'] for r in vals),val_cosine=vs['defined_cosine']==16 and vs['macro_cosine'] is not None and vs['macro_cosine']>=criteria['val_cosine_mean_min'],val_each_forecast=all(r['relative_l2'] is not None and r['relative_l2']<=criteria['val_each_forecast_max'] for r in vals))
 return dict(status='PASS' if all(checks.values()) else 'FAILED_EFFECTIVENESS',checks=checks,criteria=criteria,val_relative_reduction_vs_frozen_failed=reduction,scope='unseen source positions on the same Marmousi crop; not unseen-media generalization')

def final_evaluation():
 saved=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=False);run.check_sources(saved['source_manifest_sha256'])
 assert saved['progress']['updates']==saved['config']['updates'] and saved['progress']['exposures']==5000
 net,opt=run.restore_objects(saved,'cuda');del opt
 train=evaluate_model(net,saved,'train',list(range(500)),RUN/'FINAL_TRAIN500.json')
 rows=list(map(int,np.load(RUN/'data/validation/rows.npy')))
 val=evaluate_model(net,saved,'validation',rows,RUN/'FINAL_VAL16.json')
 run.check_validation(saved)
 raw_baseline=json.loads((RUN/'baseline_failed.json').read_text())
 assert raw_baseline['parent_sha256']==run.PARENT_SHA and raw_baseline['row_order']==rows
 baseline=dict(status=raw_baseline['status'],rows=raw_baseline['row_order'],summary=raw_baseline['aggregates'])
 result=assess_criteria(train,val,baseline,saved['config']);result.update(training_progress=saved['progress'],checkpoint_sha256=run.sha(RUN/'latest.pt'),training_completed=True,framewise_relative_l2_summary=dict(train=train['framewise_relative_l2']['summary'],validation=val['framewise_relative_l2']['summary'],baseline_comparison_available=False,existing_whole_trajectory_thresholds_unchanged=True))
 run.write(RUN/'RESULT.json',result);return result

if __name__=='__main__':run.configure();final_evaluation()
