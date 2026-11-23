"""Matched Marmousi-only fine-tuning pilot with two pressure objectives."""

import argparse
import copy
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from trace_spectral_v1.basis import SpaceBasis, TimeBasis
from trace_spectral_v1.full_train import PressureObjective
from trace_spectral_v1.input_pack import pack_input
from trace_spectral_v1.marmousi_data import atomic_json, load_bound, sha256
from trace_spectral_v1.mass_loss import WINDOWS
from trace_spectral_v1.model import TraceSpectralWave
from trace_spectral_v1.prefetch_train import load_arrays
from trace_spectral_v1.train import code_digest, input_kwargs, save_state


def train_step(model, optimizer, objective, arrays, ids, normalization, late_weight):
    velocity, source, adjoint, stats = (torch.from_numpy(array[ids]) for array in arrays)
    if late_weight != 1.:
        stats[:, 2, 1] *= late_weight
    optimizer.zero_grad(set_to_none=True)
    coefficients = model.predict_coefficients(pack_input(velocity, source, **normalization))
    loss, rss = objective(coefficients, adjoint, stats)
    if (rss < 0).any():
        loss, rss = objective(coefficients, adjoint, stats, precision=torch.float64)
        if (rss < -1e-6*stats[...,0]).any():
            raise ValueError('Invalid negative window residual')
    loss = loss.mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


@torch.inference_mode()
def evaluate(model, index, normalization, pressure_scale):
    model.eval()
    records = []
    with h5py.File(index, 'r') as data:
        for i, sample_id in enumerate(data['sample_id'].asstr()[:]):
            velocity = torch.from_numpy(data['velocity401'][i:i+1])
            source = data['source_solver'][i:i+1]
            target = torch.from_numpy(data['wavefield'][i]).double()
            predicted = model.predict_pressure(pack_input(velocity,source,**normalization),
                                               pressure_scale=pressure_scale)[0]
            error = predicted.double()-target
            norm = torch.linalg.vector_norm
            records.append(dict(sample_id=sample_id, rel_l2=[
                float(norm(error[lo:hi])/norm(target[lo:hi]))
                for lo,hi in ((0,len(target)),*WINDOWS)
            ]))
    values = np.asarray([row['rel_l2'] for row in records])
    assert len(values) == 12 and np.isfinite(values).all()
    return {'mean': values.mean(axis=0).tolist(), 'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path, help='directory for both fine-tuned checkpoints')
    parser.add_argument('checkpoint', type=Path, help='parent pressure-model checkpoint')
    parser.add_argument('manifest', type=Path, help='frozen train and validation manifest')
    parser.add_argument('cache', type=Path, help='training pressure cache')
    args = parser.parse_args()
    out, checkpoint, manifest_path, cache = args.out, args.checkpoint, args.manifest, args.cache
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    manifest = load_bound(manifest_path, state['digests']['manifest'])
    normalization = input_kwargs(state)
    pressure_scale = state['pressure_scale_pa']
    space = SpaceBasis(length=state['extent_m'])
    temporal = TimeBasis(bubbles=state['bubbles'])
    objective = PressureObjective(space.matrix, space.matrix[:,1:], temporal.matrix)
    import trace_spectral_v1.full_train as train_module
    expected = dict(manifest=state['digests']['manifest'], code=sha256(Path(train_module.__file__)),
                    base_code=code_digest(), basis=objective.basis_signature)
    if expected != state['digests']:
        raise ValueError('Checkpoint code, basis or manifest mismatch')
    rows = [row for row in manifest['records'] if row['split']=='train']
    arrays = load_arrays(cache, expected['manifest'], expected['basis'], rows, expected['code'])
    train_ids = np.asarray([i for i,row in enumerate(rows) if row['sample_id'].startswith('marmousi1_')])
    assert len(train_ids) == 109
    contract = manifest['split_indexes']['development']
    index = Path(contract['file'])
    assert sha256(index) == contract['sha256']
    assert not {rows[i]['medium_id'] for i in train_ids} & {
        row['medium_id'] for row in manifest['records'] if row['split']=='development'
    }
    out.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(20261114).permutation(train_ids)
    batches = np.array_split(order, 37)
    models, optimizers = {}, {}
    for arm in ('original_loss','late_x3'):
        model = TraceSpectralWave(space_x=space,space_z=space,time_basis=temporal)
        model.load_state_dict(state['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-6, fused=False)
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        for group in optimizer.param_groups:
            group.update(lr=3e-5, fused=False, foreach=False, capturable=False)
        models[arm], optimizers[arm] = model, optimizer
    result = {'columns':['full','early','middle','late'], 'parent_step': state['step'],
              'parent_sha256': sha256(checkpoint), 'manifest_sha256': expected['manifest'],
              'training_samples': [rows[i]['sample_id'] for i in order], 'learning_rate': 3e-5,
              'batch_sizes':[len(ids) for ids in batches], 'arms': {}}
    result['baseline'] = evaluate(models['original_loss'], index, normalization, pressure_scale)
    print('baseline', result['baseline']['mean'], flush=True)
    for model in models.values():
        model.train()
    start = time.perf_counter()
    weights = {'original_loss':1., 'late_x3':3.}
    for i, ids in enumerate(batches, 1):
        values = {arm:train_step(model,optimizers[arm],objective,arrays,ids,normalization,weights[arm])
                  for arm,model in models.items()}
        if i == 1 or i % 5 == 0 or i == len(batches):
            print(json.dumps(dict(update=i, elapsed=time.perf_counter()-start, losses=values)),flush=True)
    for arm, model in models.items():
        metrics = evaluate(model, index, normalization, pressure_scale)
        result['arms'][arm] = metrics
        # This is a separate fine-tuning experiment with its own sampling and objective.
        model_state = {key:state[key] for key in ('extent_m','bubbles','pressure_scale_pa','normalization','digests')}
        model_state.update(model=model.state_dict(), optimizer=optimizers[arm].state_dict(),
                           step=state['step']+len(batches), parent_step=state['step'],
                           pilot_updates=len(batches), arm=arm)
        save_state(out/f'{arm}.pt',model_state)
        print(arm, metrics['mean'], flush=True)
    atomic_json(out/'FINETUNE_DEV.json', result)


if __name__ == '__main__':
    main()
