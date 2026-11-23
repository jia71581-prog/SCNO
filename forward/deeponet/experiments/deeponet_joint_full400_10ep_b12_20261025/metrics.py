"""Per-record physical-Pa relative L2; no epsilon or pooled substitution."""
import math
import torch

def relative_l2_records(prediction,truth,dtype=torch.float64):
 if prediction.shape!=truth.shape or prediction.ndim<2:raise ValueError('Expected matching batch arrays')
 result=[]
 # Chunk individual records so full400 FP64 scoring does not multiply host memory.
 for pred,ref in zip(prediction,truth):
  p=pred.detach().reshape(-1);t=ref.detach().reshape(-1)
  totals=torch.zeros(4,dtype=dtype,device=p.device)
  for a,b in zip(p.split(1<<20),t.split(1<<20)):
   a=a.to(dtype);b=b.to(dtype)
   totals+=torch.stack(((a-b).square().sum(dtype=dtype),b.square().sum(dtype=dtype),a.square().sum(dtype=dtype),(a*b).sum(dtype=dtype)))
  error,energy,predenergy,dot=map(float,totals)
  finite=all(math.isfinite(x) for x in (error,energy,predenergy,dot))
  result.append(dict(relative_l2=math.sqrt(error/energy) if finite and energy>0 else None,norm_ratio=math.sqrt(predenergy/energy) if finite and energy>0 else None,cosine=dot/math.sqrt(energy*predenergy) if finite and energy>0 and predenergy>0 else None,reference_energy=energy if math.isfinite(energy) else None,prediction_energy=predenergy if math.isfinite(predenergy) else None,squared_error=error if math.isfinite(error) else None,finite=finite))
 return result

def aggregate(records):
 def mean(key):
  values=[r[key] for r in records if r[key] is not None]
  return sum(values)/len(values) if values else None
 return dict(records=len(records),defined_relative_l2=sum(r['relative_l2'] is not None for r in records),defined_cosine=sum(r['cosine'] is not None for r in records),defined_norm_ratio=sum(r['norm_ratio'] is not None for r in records),macro_relative_l2=mean('relative_l2'),macro_norm_ratio=mean('norm_ratio'),macro_cosine=mean('cosine'),all_finite=all(r['finite'] for r in records))

def field_metrics(prediction,truth,receivers,dtype=torch.float64):
 if prediction.shape[1]!=400:raise ValueError('Must score all400 original times')
 rec=torch.as_tensor(receivers,device=prediction.device,dtype=torch.long)
 return {'full400':relative_l2_records(prediction,truth,dtype),'forecast392':relative_l2_records(prediction[:,8:],truth[:,8:],dtype),'late130':relative_l2_records(prediction[:,270:],truth[:,270:],dtype),'receivers400':relative_l2_records(prediction[:,:,rec[:,0],rec[:,1]],truth[:,:,rec[:,0],rec[:,1]],dtype),'receivers392':relative_l2_records(prediction[:,8:,rec[:,0],rec[:,1]],truth[:,8:,rec[:,0],rec[:,1]],dtype)}

def _summarize_frame_arrays(values,positive,zero,finite):
 valid=sum(positive);defined=sum(v is not None for v in values)
 complete=defined==valid and all(finite)
 return dict(mean_relative_l2=sum(v for v in values if v is not None)/valid if valid and complete else None,frames=len(values),valid_reference_frames=valid,defined_frames=defined,zero_reference_frames=sum(zero),nonfinite_frames=sum(not f for f in finite),all_finite=all(finite),per_frame_relative_l2=values,per_frame_reference_positive=positive,per_frame_reference_zero=zero,per_frame_finite=finite)

def framewise_relative_l2_records(prediction,truth,dtype=torch.float64):
 """Mean of per-frame spatial relative L2, retaining only reference energy >0."""
 if prediction.shape!=truth.shape or prediction.ndim<3:raise ValueError('Expected matching B,T,spatial arrays')
 result=[]
 for pred,ref in zip(prediction,truth):
  p=pred.detach().reshape(pred.shape[0],-1);t=ref.detach().reshape(ref.shape[0],-1)
  values=[];positive=[];zero=[];finite=[]
  for a,b in zip(p.split(16),t.split(16)):
   a=a.to(dtype);b=b.to(dtype);energy=b.square().sum(1,dtype=dtype);error=(a-b).square().sum(1,dtype=dtype)
   ok=torch.isfinite(a).all(1)&torch.isfinite(b).all(1)&torch.isfinite(energy)&torch.isfinite(error)
   for e,d,f in zip(energy.tolist(),error.tolist(),ok.tolist()):
    supported=math.isfinite(e) and e>0
    values.append(math.sqrt(d/e) if supported and f else None);positive.append(supported);zero.append(e==0);finite.append(bool(f))
  result.append(_summarize_frame_arrays(values,positive,zero,finite))
 return result

def _frame_slice(records,start,stop):
 return [_summarize_frame_arrays(r['per_frame_relative_l2'][start:stop],r['per_frame_reference_positive'][start:stop],r['per_frame_reference_zero'][start:stop],r['per_frame_finite'][start:stop]) for r in records]

def framewise_field_metrics(prediction,truth,receivers,dtype=torch.float64):
 if prediction.shape[1]!=400:raise ValueError('Must score all400 original frames')
 rec=torch.as_tensor(receivers,device=prediction.device,dtype=torch.long)
 field=framewise_relative_l2_records(prediction,truth,dtype)
 receiver=framewise_relative_l2_records(prediction[:,:,rec[:,0],rec[:,1]],truth[:,:,rec[:,0],rec[:,1]],dtype)
 return dict(full400=field,forecast392=_frame_slice(field,8,400),late130=_frame_slice(field,270,400),receivers400=receiver,receivers392=_frame_slice(receiver,8,400))

def aggregate_framewise(records):
 values=[r['mean_relative_l2'] for r in records if r['mean_relative_l2'] is not None]
 return dict(records=len(records),defined_records=len(values),macro_mean_framewise_relative_l2=sum(values)/len(values) if values else None,total_frames=sum(r['frames'] for r in records),valid_reference_frames=sum(r['valid_reference_frames'] for r in records),zero_reference_frames=sum(r['zero_reference_frames'] for r in records),nonfinite_frames=sum(r['nonfinite_frames'] for r in records),all_finite=all(r['all_finite'] for r in records),weighting='equal valid frames within each record, then equal record macro mean; undefined records counted explicitly')
