"""Bounded TRAIN-only coefficient supervision and exact boundary checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .basis import SpaceBasis, TimeBasis
from .input_pack import pack_input
from .marmousi_data import atomic_json, sha256
from .mass_loss import MassLoss
from .model import TraceSpectralWave
from .project_teacher import ProjectionRecord


def load_row(cache, entry, signature):
    path = cache/entry['file']
    item = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    provenance = item['provenance']
    if provenance['split'] != 'train' or provenance['sample_id'] != entry['sample_id']:
        raise ValueError('Training cache row is not the registered TRAIN sample')
    projection = ProjectionRecord.from_state_dict(item['projection'])
    if projection.basis_signature != signature:
        raise ValueError('Projection basis hash differs from the qualified basis')
    return item['velocity'], item['source'], projection


def fixed_plan(rng, manifest, logical_batch, single=False):
    records = manifest['records']
    if single:
        return [([0],[1.])]
    by_family = {name: [i for i,row in enumerate(records) if row['family']==name]
                 for name in sorted({row['family'] for row in records})}
    if set(by_family) == {'uniform','layered'}:
        n = logical_batch//2
        target = max(map(len,by_family.values()))
        result = []
        shuffled = {name:rng.permutation(group).tolist() for name,group in by_family.items()}
        for offset in range(0,target,n):
            k = min(n,target-offset)
            ids = [group[j%len(group)] for group in shuffled.values() for j in range(offset,offset+k)]
            result.append((ids,[1.]*len(ids)))
        return result
    ids = rng.permutation(len(records)).tolist()
    return [(ids[start:start+logical_batch],[1.]*len(ids[start:start+logical_batch]))
            for start in range(0,len(ids),logical_batch)]


def model_and_loss(manifest, device):
    space = SpaceBasis(length=manifest['extent_m'])
    temporal = TimeBasis(bubbles=manifest['bubbles'])
    model = TraceSpectralWave(space_x=space,space_z=space,time_basis=temporal).to(device)
    objective = MassLoss(space.matrix,space.matrix[:,1:],temporal.matrix).to(device)
    if objective.basis_signature != manifest['basis_signature']:
        raise ValueError('Current basis differs from the frozen manifest')
    return model, objective


def input_kwargs(manifest):
    extent = manifest['extent_m']
    return dict(extent_xz=(extent,extent),duration_s=1.,**manifest['normalization'])


def code_digest():
    digest = hashlib.sha256()
    model_sources = ('assembly.py','basis.py','decoder.py','encoder.py','entity_heads.py','input_pack.py',
                     'mass_loss.py','model.py','project_teacher.py','train.py')
    for name in model_sources:
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_state(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state,temporary)
    temporary.replace(path)


def run(args):
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text())
    if config['weak_enabled'] or config['logical_batch'] != 16 or config['micro_batch'] != 2:
        raise ValueError('Expected supervised v1 batch and loss contract')
    manifest_path = args.cache/'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['split'] != 'train' or any(r['split'] != 'train' for r in manifest['records']):
        raise ValueError('Training accepts TRAIN coefficient cache only')
    if not args.single and len(manifest['records']) != 32:
        raise ValueError('The multi-example short run requires all 32 qualified TRAIN examples')
    for row in manifest['records']:
        if sha256(args.cache/row['file']) != row['sha256']:
            raise ValueError('Corrupt TRAIN coefficient cache')
    digests = dict(manifest=sha256(manifest_path), config=sha256(args.config), code=code_digest(),
                   basis=manifest['basis_signature'], single=args.single, device=args.device)
    if not args.single and args.max_steps >= 256:
        if args.smoke is None or args.smoke_prediction is None:
            raise ValueError('The single-sample overfit gate must run before the 256-step panel')
        smoke = json.loads(args.smoke.read_text())
        prediction = json.loads(args.smoke_prediction.read_text())
        if (not smoke['improved'] or any(smoke['digests'][key]!=digests[key] for key in ('basis','config','code','device'))
                or not smoke['digests']['single'] or prediction['sample_id'] != manifest['records'][0]['sample_id']
                or prediction['split'] != 'train' or prediction['step'] != smoke['steps']
                or prediction['checkpoint_sha256'] != sha256(args.smoke.parent/'last.pt')
                or max(prediction['relative_l2_full'],prediction['relative_l2_windows'][-1]) > .1):
            raise ValueError('The single-sample overfit gate is missing or failed')
    torch.manual_seed(config['model_seed'])
    random.seed(config['model_seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(config['sampler_seed'])
    device = args.device
    use_amp = device=='cuda' and config['autocast']=='bf16'
    model, objective = model_and_loss(manifest,device)
    optim = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda step: min((step+1)/config['warmup_steps'],1.))
    last = args.out/'last.pt'
    plan = fixed_plan(rng,manifest,config['logical_batch'],args.single)
    step, cursor, history = 0, 0, []
    if args.resume:
        saved = torch.load(last,map_location='cpu',weights_only=True)
        if saved['digests'] != digests:
            raise ValueError('Incompatible checkpoint data, code, configuration or basis hash')
        model.load_state_dict(saved['model'])
        optim.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        step, cursor, plan, history = (saved[key] for key in ('step','cursor','plan','history'))
        torch.set_rng_state(saved['torch_rng'])
        if device=='cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        rng.bit_generator.state = saved['numpy_rng']
    else:
        args.out.mkdir(exist_ok=False)
    if args.max_steps > 256:
        if step < 256 or not args.resume:
            raise ValueError('The extension requires a completed 256-step checkpoint')
        if np.mean([r['loss'] for r in history[224:256]]) >= np.mean([r['loss'] for r in history[:32]]):
            raise ValueError('The 256-step loss screen failed')
    prep = input_kwargs(manifest)
    while step < args.max_steps:
        if cursor == len(plan):
            plan, cursor = fixed_plan(rng,manifest,config['logical_batch'],args.single), 0
        ids, weights = plan[cursor]
        started = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        value, score = 0., []
        for offset in range(0,len(ids),config['micro_batch']):
            indexes = ids[offset:offset+config['micro_batch']]
            speeds,sources,records = zip(*(load_row(args.cache,manifest['records'][i],manifest['basis_signature']) for i in indexes))
            inputs = pack_input(torch.stack(speeds).to(device),torch.stack(sources).to(device),**prep)
            projection = ProjectionRecord.stack(records).to(device)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=use_amp):
                prediction = model.predict_coefficients(inputs)
            stats = objective(prediction.float(),projection,weights[offset:offset+len(indexes)])
            micro = stats.loss*len(indexes)/len(ids)
            micro.backward()
            value += float(micro.detach())
            score.extend(stats.score.tolist())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip_norm'],error_if_nonfinite=True)
        optim.step()
        scheduler.step()
        step, cursor = step+1, cursor+1
        history.append(dict(step=step,loss=value,gradient_norm=float(norm),score=score,seconds=time.perf_counter()-started))
        if step % config['checkpoint_every'] == 0 or step == args.max_steps:
            save_state(last,dict(digests=digests,model=model.state_dict(),optimizer=optim.state_dict(),
                                 scheduler=scheduler.state_dict(),step=step,cursor=cursor,plan=plan,history=history,
                                 torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],
                                 python_rng=random.getstate(),numpy_rng=rng.bit_generator.state,
                                 normalization=manifest['normalization'],extent_m=manifest['extent_m'],
                                 pressure_scale_pa=manifest['pressure_scale_pa'],bubbles=manifest['bubbles']))
            k = min(len(history)//2,8)
            improved = bool(k and np.mean([r['loss'] for r in history[-k:]]) < .8*np.mean([r['loss'] for r in history[:k]]))
            atomic_json(args.out/'TRAINING.json',dict(digests=digests,steps=step,improved=improved,history=history))
            print(json.dumps(history[-1]),flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--cache',type=Path,required=True)
    cli.add_argument('--config',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--max-steps',type=int,choices=(32,256,1000),required=True)
    cli.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    cli.add_argument('--single',action='store_true')
    cli.add_argument('--smoke',type=Path)
    cli.add_argument('--smoke-prediction',type=Path)
    cli.add_argument('--resume',action='store_true')
    args = cli.parse_args()
    run(args)


if __name__ == '__main__':
    main()
