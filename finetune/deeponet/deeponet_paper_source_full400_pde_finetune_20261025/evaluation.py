"""Full-grid FIT metrics; later392 labels are supervised in this experiment."""
from scno_release_paths import local_asset as _scno_local_asset
from pathlib import Path
import json,os,sys,time
import numpy as np
import torch
ROOT=Path(_scno_local_asset('project'))
BASE=ROOT/'runs/deeponet_joint_full400_10ep_b12_20261025'
sys.path.insert(0,str(BASE))
import train as inherited
from metrics import field_metrics,aggregate,framewise_field_metrics,aggregate_framewise
sys.path.insert(0,str(ROOT/'runs/deeponet_wave_propagation_repair_20261025/propagation_guard'))
from guard import assess_propagation
SOURCE_ID='paper_source'

def write(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def paper_framewise(pred,truth,receivers):
 result={};rec=torch.as_tensor(receivers,dtype=torch.long)
 for label,p,y in [('field',pred[:,8:],truth[:,8:]),('receivers',pred[:,8:,rec[:,0],rec[:,1]],truth[:,8:,rec[:,0],rec[:,1]])]:
  p=p[0].double().reshape(392,-1);y=y[0].double().reshape(392,-1)
  rms=y.square().mean(1).sqrt();mask=(rms>0)&(rms>=.01*rms.max())
  values=((p-y).square().sum(1)[mask]/y.square().sum(1)[mask]).sqrt()
  result[label]=dict(relative_l2_mean=float(values.mean()),retained_frames=int(mask.sum()),excluded_frames=int((~mask).sum()),candidate_frames=392,threshold_pa=float(.01*rms.max()),rule='reference RMS>=1% of reference peak over candidate392, reference RMS>0',retained_frame_indices=(torch.arange(8,400)[mask]).tolist())
 assert result['field']['retained_frames']==392 and result['receivers']['retained_frames']==293
 return result

def plot_snapshots(pred,truth,times,step,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 slots=[40,160,280,399]
 source=ROOT/'outputs/deeponet_vs_paper_marmousi_20261025/snapshot_arrays.npz'
 with np.load(source,allow_pickle=False) as z:
  ref=z['reference_pa'].copy();ours=z['ours_8snapshot_tuned_pa'].copy();paper_times=z['time_s'].copy()
 assert np.array_equal(ref,truth[0,slots].numpy()) and np.array_equal(paper_times,times[slots])
 current=pred[0,slots].numpy();arrays=[ref,ours,current];limit=max(float(np.abs(x).max()) for x in arrays)*1e9
 labels=['Reference','AURORA + 8-frame tuning\n(paper result)',f'DeepONet full400 + PDE\nFIT step {step}']
 plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42})
 fig,axes=plt.subplots(3,4,figsize=(11.8,8.4),sharex=True,sharey=True,layout='constrained')
 for i,(a,label) in enumerate(zip(arrays,labels)):
  for j,slot in enumerate(slots):
   ax=axes[i,j];im=ax.imshow(a[j]*1e9,cmap='RdBu_r',vmin=-limit,vmax=limit,extent=(-.005,2.005,2.005,-.005),interpolation='nearest',resample=False)
   ax.set(xlim=(0,2),ylim=(2,0),xticks=[0,1,2],yticks=[0,1,2])
   if i==0:ax.text(.5,1.04,f't={float(times[slot]):.5f} s',transform=ax.transAxes,ha='center')
   if j==0:ax.set_ylabel(label+'\nz (km)')
   if i==2:ax.set_xlabel('x (km)')
   if i:
    err=np.linalg.norm(a[j].astype(np.float64)-ref[j].astype(np.float64))/np.linalg.norm(ref[j].astype(np.float64))
    ax.text(.03,.96,f'L2={100*err:.2f}%',transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
 fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,pad=.015,label='Pressure (nPa)')
 for ext in ['png','pdf']:fig.savefig(out/f'wavefields_step{step:04d}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
 np.savez(out/f'snapshots_step{step:04d}.npz',reference_pa=ref,prediction_pa=current,time_s=times[slots],slots=np.array(slots))
 return dict(pressure_scale_npa=[-limit,limit],same_scale_all_panels=True,paper_result_unchanged=True,comparison='DeepONet FIT uses all400 labels; paper AURORA tuning uses8 observed frames. Different supervision budgets, not a matched protocol.')

@torch.no_grad()
def evaluate(model,saved,data,physics,scales,sup,step,outdir):
 start=time.monotonic();out=Path(outdir);out.mkdir(parents=True,exist_ok=True);was_training=model.training;model.eval()
 prediction=inherited.previous.predict_normalized(model,'deeponet',data,int(saved['progress'].get('chunk_points',1024)))
 objective=inherited.previous.output_objective(prediction,data,sup,physics,scales,saved['stage_config'],False)
 pred=(prediction*data['scale']).cpu();del prediction
 # Read original Pa labels, not divide/multiply roundtrips of normalized labels.
 path=ROOT/'runs/paper_marmousi201_single_source_fullfit_20261020/case_0/reference_pa.npy'
 truth=torch.from_numpy(np.load(path,allow_pickle=False)[None])
 assert pred.shape==truth.shape==(1,400,201,201) and torch.isfinite(pred).all()
 receivers=saved['stage_config']['training']['receiver_indices'];times=data['times'].cpu().numpy()
 windows={k:[dict(row=SOURCE_ID,**r) for r in v] for k,v in field_metrics(pred,truth,receivers).items()}
 temporal={k:[dict(row=SOURCE_ID,**r) for r in v] for k,v in framewise_field_metrics(pred,truth,receivers).items()}
 result=dict(status='COMPLETE',step=step,rows=[SOURCE_ID],objective=objective,windows=windows,summary={k:aggregate(v) for k,v in windows.items()},framewise_relative_l2=dict(windows=temporal,summary={k:aggregate_framewise(v) for k,v in temporal.items()}),paper_compatible_framewise=paper_framewise(pred,truth,receivers),scope='Single-source all400 supervised FIT. forecast392 is a legacy metric key only: these later labels are part of training. No heldout/generalization claim.',pressure_units='Pa',arithmetic='FP32 physical predictions; FP64 scoring of original Pa reference',PDE_scope='All398 nonuniform-time centers; coarse source-free SIPG outside registered source exclusion. Not a fine-grid CPML residual.',time_supervision='All400 original times/all201x201 spatial nodes every optimizer update')
 result['propagation_checks']=assess_propagation(result,[SOURCE_ID])
 result['single_source_fit_effective']=result['propagation_checks']['status']=='PASS_COMPONENT_CHECKS'
 result['generalization_proven']=False
 result['plot']=plot_snapshots(pred,truth,times,step,out)
 full=out/'prediction_latest_pa.npy';tmp=out/'prediction_latest_pa.tmp'
 with tmp.open('wb') as f:np.save(f,pred[0].numpy(),allow_pickle=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,full)
 if step==0:np.save(out/'prediction_initial_pa.npy',pred[0].numpy(),allow_pickle=False)
 if step==saved['config']['max_steps']:np.save(out/'prediction_final_pa.npy',pred[0].numpy(),allow_pickle=False)
 result['prediction_path_if_final']=str(out/'prediction_final_pa.npy') if step==saved['config']['max_steps'] else None
 result['elapsed_seconds']=time.monotonic()-start;result['metrics_path']=str(out/f'EVAL_STEP{step:04d}.json')
 write(result['metrics_path'],result);write(out/'LATEST_EVALUATION.json',result)
 model.train(was_training)
 print(json.dumps(dict(evaluation_step=step,objective=objective,full400=result['summary']['full400']['macro_relative_l2'],late130=result['summary']['late130']['macro_relative_l2'],fit_effective=result['single_source_fit_effective'],seconds=result['elapsed_seconds'])),flush=True)
 return result
