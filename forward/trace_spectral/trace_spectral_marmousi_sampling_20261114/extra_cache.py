"""Bind extra pressure statistics to their verified labels and frozen decoder."""

import argparse
from pathlib import Path
import json

import numpy as np
import torch

from trace_spectral_v1.marmousi_data import atomic_json, sha256
from trace_spectral_v1.prefetch_train import load_arrays


class CombinedArray:
    def __init__(self, original, extra):
        if original.shape[1:] != extra.shape[1:] or original.dtype != extra.dtype:
            raise ValueError('Extra cache format differs from the original cache')
        self.original = original
        self.extra = extra

    def __getitem__(self, ids):
        ids = np.asarray(ids)
        is_original = ids < len(self.original)
        result = np.empty((len(ids), *self.original.shape[1:]), dtype=self.original.dtype)
        result[is_original] = self.original[ids[is_original]]
        result[~is_original] = self.extra[ids[~is_original] - len(self.original)]
        return result


def read_addendum(receipt_path, parent_hash):
    receipt = json.loads(receipt_path.read_text())
    rows = receipt['records']
    if (receipt['status'] != 'COMPLETE' or receipt['parent_sha256'] != parent_hash
            or len(rows) != 24 or len({r['sample_id'] for r in rows}) != 24
            or any(r['split'] != 'train' for r in rows)):
        raise ValueError('Invalid additional training set')
    return rows


def build(parent_path, receipt_path, target):
    from trace_spectral_v1.basis import SpaceBasis, TimeBasis
    from trace_spectral_v1.full_train import PressureObjective

    if (target/'COMPLETE.json').exists():
        raise FileExistsError(target/'COMPLETE.json')
    target.mkdir(exist_ok=False)
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
    base = checkpoint['digests']['manifest']
    rows = read_addendum(receipt_path, base)
    space = SpaceBasis(length=checkpoint['extent_m'])
    temporal = TimeBasis(bubbles=checkpoint['bubbles'])
    objective = PressureObjective(space.matrix, space.matrix[:,1:], temporal.matrix)
    assert objective.basis_signature == checkpoint['digests']['basis']
    names = ('velocity', 'source', 'adjoint', 'stats')
    arrays = None
    import h5py
    for i,row in enumerate(rows):
        path = Path(row['file'])
        if sha256(path) != row['sha256']:
            raise ValueError(f'Label hash mismatch {path}')
        with h5py.File(path,'r') as file:
            assert file.attrs['split'] == 'train' and file.attrs['sample_id'] == row['sample_id']
            assert file.attrs['units'] == 'Pa' and file.attrs['physical_sources'] == 1
            pressure = file['wavefield'][:]
            velocity = file['velocity401'][:]
            source = file['source_solver'][:]
        if (pressure.shape!=(801,201,201) or velocity.shape!=(401,401)
                or not np.array_equal(source,row['source_solver'])):
            raise ValueError('Bad pressure or input shape')
        adjoint, stats = objective.statistics(pressure, checkpoint['pressure_scale_pa'])
        values = (velocity, source, adjoint.numpy(), stats.numpy())
        if arrays is None:
            arrays = [np.lib.format.open_memmap(target/f'{name}.npy', mode='w+',
                      shape=(len(rows), *value.shape),dtype=value.dtype)
                      for name,value in zip(names,values,strict=True)]
        for array,value in zip(arrays,values,strict=True):
            array[i] = value
        if (i+1)%6 == 0:
            print('cached',i+1,flush=True)
    for array in arrays:
        array.flush()
    atomic_json(target/'COMPLETE.json', dict(status='COMPLETE', parent_manifest=base,
                basis=objective.basis_signature, addendum_sha256=sha256(receipt_path),
                source=str(receipt_path), ids=[r['sample_id'] for r in rows],
                files={name:sha256(target/f'{name}.npy') for name in names}))
    print('cache complete',len(rows),flush=True)


def join(cache, manifest_rows, digests, extension):
    original = load_arrays(cache, digests['manifest'], digests['basis'], manifest_rows, digests['code'])
    if extension is None:
        return original, manifest_rows, None
    contract = json.loads((extension/'COMPLETE.json').read_text())
    source = Path(contract['source'])
    if (contract['status'] != 'COMPLETE' or contract['parent_manifest'] != digests['manifest']
            or contract['basis'] != digests['basis'] or sha256(source) != contract['addendum_sha256']):
        raise ValueError('Extra cache identity mismatch')
    rows = read_addendum(source, digests['manifest'])
    if [r['sample_id'] for r in rows] != contract['ids']:
        raise ValueError('Extra cache ordering mismatch')
    arrays = []
    for array, name in zip(original, ('velocity','source','adjoint','stats'),strict=True):
        path = extension/f'{name}.npy'
        if sha256(path) != contract['files'][name]:
            raise ValueError(f'Extra cache digest mismatch: {name}')
        extra = np.load(path, mmap_mode='r', allow_pickle=False)
        if len(extra)!=len(rows):
            raise ValueError('Unexpected extra cache length')
        arrays.append(CombinedArray(array,extra))
    return arrays, [*manifest_rows, *rows], contract['addendum_sha256']


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('parent',type=Path)
    cli.add_argument('receipt',type=Path)
    cli.add_argument('target',type=Path)
    args=cli.parse_args()
    build(args.parent,args.receipt,args.target)
