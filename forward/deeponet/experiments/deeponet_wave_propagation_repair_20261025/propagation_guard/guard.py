"""Additional propagation guard for the next repair; frozen training is unchanged.

This is necessary evidence only. A formal success also requires the original
TRAIN500/VAL16 roster, baseline-improvement, budget and provenance checks.
Thresholds reuse existing .85/1.1/.5/[.2,2] criteria across late/receiver windows.
Receiver *framewise* errors are reported, but are not thresholded here because
strictly positive pre-arrival energies can make that ratio ill-conditioned.
"""
import math

POLICY=dict(mean_relative_l2_max=.85,each_relative_l2_max=1.1,
            mean_cosine_min=.5,norm_ratio_min=.2,norm_ratio_max=2.,
            mean_framewise_relative_l2_max=.85)

def finite_number(v):return isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v)

def assess_propagation(report,expected_rows):
 expected_rows=list(expected_rows);checks={};details={}
 checks['complete_exact_roster']=bool(expected_rows) and len(set(expected_rows))==len(expected_rows) and report.get('status')=='COMPLETE' and report.get('rows')==expected_rows
 for name in ('full400','forecast392','late130','receivers400','receivers392'):
  rows=report.get('windows',{}).get(name,[])
  complete=bool(expected_rows) and len(rows)==len(expected_rows) and [r.get('row') for r in rows]==expected_rows and all(r.get('finite') is True and all(finite_number(r.get(k)) for k in ('relative_l2','norm_ratio','cosine')) for r in rows)
  checks[name+'_complete']=complete
  if not complete:continue
  l2=sum(r['relative_l2'] for r in rows)/len(rows);cos=sum(r['cosine'] for r in rows)/len(rows)
  checks[name+'_mean_error']=0<=l2<=POLICY['mean_relative_l2_max']
  checks[name+'_each_error']=all(0<=r['relative_l2']<=POLICY['each_relative_l2_max'] for r in rows)
  checks[name+'_cosine']=POLICY['mean_cosine_min']<=cos<=1.+1e-6
  checks[name+'_each_amplitude']=all(POLICY['norm_ratio_min']<=r['norm_ratio']<=POLICY['norm_ratio_max'] for r in rows)
  details[name]=dict(mean_relative_l2=l2,mean_cosine=cos)
 framewise=report.get('framewise_relative_l2',report.get('framewise',{})).get('windows',{})
 for name,frames in [('full400',400),('forecast392',392),('late130',130)]:
  rows=framewise.get(name,[])
  complete=bool(expected_rows) and len(rows)==len(expected_rows) and all(r.get('row')==expected_rows[i] and r.get('all_finite') is True and r.get('frames')==frames and 0<r.get('valid_reference_frames',0)<=frames and r.get('defined_frames')==r.get('valid_reference_frames') and finite_number(r.get('mean_relative_l2')) and r['mean_relative_l2']>=0 for i,r in enumerate(rows))
  checks[name+'_framewise_complete']=complete
  if complete:
   mean=sum(r['mean_relative_l2'] for r in rows)/len(rows)
   checks[name+'_framewise_error']=0<=mean<=POLICY['mean_framewise_relative_l2_max']
   details.setdefault(name,{})['mean_framewise_relative_l2']=mean
 return dict(status='PASS_COMPONENT_CHECKS' if all(checks.values()) else 'FAILED_PROPAGATION',checks=checks,failed_checks=[k for k,v in checks.items() if not v],details=details,policy=POLICY,formal_training_success=False,meaning='Necessary component checks only; formal success additionally requires complete TRAIN500/VAL16 evaluation and the existing baseline/provenance/budget gates.')
