"""Original strict PASO factory/normalizer loading, no import-time model action."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import torch
PROJECT=Path(__file__).resolve().parents[1]
PARENT_MANIFEST=Path(__file__).resolve().parent/'PARENT_MANIFEST.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value
def load_parent(family,device='cpu',gpu_authorized=False):
    device=torch.device(device)
    if device.type=='cuda' and not gpu_authorized:raise PermissionError('GPU requires new exact ROOT scope')
    if family not in ('layered','marmousi'):raise ValueError('P0 fixed L/M families')
    metadata=json.loads(PARENT_MANIFEST.read_text());spec=metadata['parents'][family];code=Path(spec['code'])
    if sha(spec['path'])!=spec['sha256'] or sha(metadata['normalizer']['path'])!=metadata['normalizer']['sha256']:raise PermissionError('original base/norm SHA drift')
    sys.path.insert(0,str(code));sys.path.insert(0,str(code/'snapshot'))
    factory=module('al_p0_factory_'+family,code/'model.py');train=module('al_p0_config_parser_'+family,code/'train.py');config=train.resolved_config(spec['config'])
    from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
    normalizer=PhysicalNormalizer.from_dict(json.loads(Path(metadata['normalizer']['path']).read_text()),expected_manifest=metadata['semantic_manifest_digest'])
    saved=torch.load(spec['path'],map_location='cpu',weights_only=False)
    if saved['updates_completed']!=50000 or saved['depth']!=2 or saved['lift_mode']!='homogeneous' or saved['normalizer_sha256']!=metadata['normalizer']['sha256']:raise ValueError('exact released checkpoint header')
    model=factory.build(config,str(device));model.load_state_dict(saved['model_state'],strict=True);model.scno_depth=2;model.eval();model.requires_grad_(False)
    identity=dict(family=family,checkpoint_path=spec['path'],checkpoint_sha256=spec['sha256'],step=50000,normalizer_sha256=metadata['normalizer']['sha256'],original_train_digest=metadata['semantic_manifest_digest'],factory_sha256=sha(code/'model.py'),operator_sha256=sha(code/'snapshot/grouped_ufno_mionet_v3/model/operator.py'),model_spec_sha256=sha(config['model_spec']),config_sha256=sha(spec['config']),git_commit=None,git_reason='not_a_git_worktree',original_IC8_required=True)
    return model,normalizer,identity

def prepare(model,normalizer,sample,velocity=None):
    c=sample['velocity'] if velocity is None else velocity
    if c.ndim!=4 or c.shape[1]!=1 or sample['ic'].shape[1]!=8:raise ValueError('physical c B1ZX/original IC8 rawPa')
    # Never detach c: freezeW0 still leaves input-velocity AD live.
    medium=model.encode_medium(c,normalizer)
    state=model.prepare_sources(medium,sample['source'],sample['source_map'],normalizer,record_to_medium=torch.zeros(1,dtype=torch.long,device=c.device),ic_snapshots_physical=sample['ic'],anchor_time_s=sample['anchor'])
    grid=model.prepare_dense_grid(state,x_m=sample['x'],z_m=sample['z'])
    return state,grid

def dense_forward(model,normalizer,sample,times,velocity=None,physical=False,time_block=1):
    state,grid=prepare(model,normalizer,sample,velocity)
    result=model.dense_normalized(state,times,dense_grid=grid,time_block=time_block,apply_correction=True)
    return normalizer.decode_pressure(result,sample['source'][:,4]) if physical else result
