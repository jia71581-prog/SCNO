import os,json,random,re
from pathlib import Path
import numpy as np
import torch

# --- rolling checkpoint retention (20260926) ---------------------------------
# Explicit, prefix-scoped list of artifacts that are NEVER rolling candidates:
# the init/lineage origin (step 0, including any _saveNNN collision copies) and
# every dev9 best checkpoint. Everything else matching '<prefix>_NNNNNNNN' is
# rolled. Listing the protected names keeps this independent of keep_latest.
PROTECTED_CHECKPOINT_PREFIXES=('checkpoint_00000000','best_')
# '<prefix>_<8-digit step>' with an optional atomic_checkpoint '_saveNNN' suffix.
_CHECKPOINT_NAME_RE=re.compile(r'^(?P<prefix>.*)_(?P<step>\d{8})(?:_save(?P<version>\d+))?$')

def checkpoint_name(prefix,completed,version=0):
 return f'{prefix}_{int(completed):08d}'+(f'_save{int(version):03d}' if version else '')+'.pt'

def plan_checkpoint_pruning(directory,keep_latest,prefix='checkpoint'):
 """Pure planner: return the rolling checkpoints to delete, newest kept.
 directory: run_dir/'checkpoints' (path or str)
 keep_latest: None/0 => disabled (return []). N>0 => keep the N artifacts with
              the largest (step, save-version) among the candidates returned.
 Protected and never returned: best_* (all), checkpoint_00000000* (init lineage),
 anything that does not match '<prefix>_<8-digit step>' and every '*.tmp'.
 _saveNNN collision copies ARE candidates: they are ranked by their own status
 number, and the newest saveNNN of a step outranks the plain file of that step.
 """
 if not keep_latest:return []
 directory=Path(directory)
 if not directory.is_dir():return []
 candidates=[]
 for entry in directory.iterdir():
  if not entry.is_file() or entry.suffix!='.pt':continue
  if entry.name.endswith('.tmp'):continue
  stem=entry.stem
  if any(stem.startswith(p) for p in PROTECTED_CHECKPOINT_PREFIXES):continue
  m=_CHECKPOINT_NAME_RE.match(stem)
  if not m or m.group('prefix')!=prefix:continue
  candidates.append((int(m.group('step')),int(m.group('version') or 0),entry))
 candidates.sort(key=lambda r:(r[0],r[1]))
 excess=len(candidates)-int(keep_latest)
 return [entry for _,_,entry in (candidates[:excess] if excess>0 else [])]

def prune_checkpoints(directory,keep_latest,prefix='checkpoint'):
 """Delete the planner's rolling excess. Returns the sorted deleted paths.
 Retention only ever runs after a new checkpoint is safely on disk, so an
 interrupted run can never strand the newest state."""
 removed=[]
 for p in plan_checkpoint_pruning(directory,keep_latest,prefix):
  try:p.unlink()
  except FileNotFoundError:continue
  removed.append(p);print(json.dumps({'event':'checkpoint_pruned','path':str(p)}),flush=True)
 removed.sort()
 return removed

def capture_rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}
def restore_rng(r):
 random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
 if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])
def atomic_json(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def atomic_checkpoint(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);base=path;version=0
 while path.exists() or path.with_suffix(path.suffix+'.tmp').exists():
  version+=1;path=base.with_name(base.stem+f'_save{version:03d}'+base.suffix)
 tmp=path.with_suffix(path.suffix+'.tmp');torch.save(data,tmp);os.replace(tmp,path)
 # Plain JSON pointer, replaced only after the checkpoint is complete.
 atomic_json(path.parent.parent/'latest.json',{'checkpoint':str(path.resolve()),'updates_completed':data['updates_completed']})
 return path
def check_resume(saved,mode,semantic_digest,pool_digest):
 if saved.get('lift_mode')!=mode:raise ValueError('resume lift_mode mismatch; model-only fork requires --init-checkpoint')
 if saved.get('semantic_config_digest')!=semantic_digest:raise ValueError('resume research configuration changed; use model-only --init-checkpoint')
 if saved.get('pool_digest')!=pool_digest:raise ValueError('resume data pool mismatch; use model-only fork')
 if saved.get('next_update')!=saved.get('updates_completed',-1)+1:raise ValueError('invalid resume cursor')


# best_* names carry an epoch tag before the step: 'best_e0005_00001770.pt'
# (optionally with an atomic '_saveNNN' suffix). The rolling 'checkpoint_*' form
# is 'checkpoint_00001770.pt'. Both are handled here so the planner works for
# either prefix while still ignoring unrelated files.
_BEST_NAME_RE=re.compile(r'^best(?:_e\d+)?_(?P<step>\d{8})(?:_save(?P<version>\d+))?$')

def plan_best_pruning(directory, keep_best, prefix='best'):
 """Rolling retention for dev9 best checkpoints (20260926).

 Separate from the rolling plan so that the init/lineage artifact
 ('checkpoint_00000000*') stays protected while best_* can be bounded.
 keep_best: None/0 => disabled (return []). N>0 => keep the N best_* artifacts
 with the largest (step, save-version); the rest are returned for deletion.
 Rationale: with eval_every = 3540 (10 epochs) a 100-epoch run writes ~10 best
 checkpoints (4.7 GiB), but a per-epoch cadence would write ~100 (47 GiB) and
 trip runtime.min_free_disk_gib.
 """
 if not keep_best:return []
 directory=Path(directory)
 if not directory.is_dir():return []
 candidates=[]
 for entry in directory.iterdir():
  if not entry.is_file() or entry.suffix!='.pt':continue
  if entry.name.endswith('.tmp'):continue
  stem=entry.stem
  m=_BEST_NAME_RE.match(stem)
  if not m:continue
  candidates.append((int(m.group('step')),int(m.group('version') or 0),entry))
 candidates.sort(key=lambda r:(r[0],r[1]))
 excess=len(candidates)-int(keep_best)
 return [entry for _,_,entry in (candidates[:excess] if excess>0 else [])]


def prune_best_checkpoints(directory, keep_best, prefix='best'):
 """Delete the best-checkpoint excess. Never touches best_e* naming if unset."""
 removed=[]
 for p in plan_best_pruning(directory, keep_best, prefix):
  try:
   p.unlink();removed.append(str(p))
  except FileNotFoundError:
   pass
 return sorted(removed)
