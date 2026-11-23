from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np

SEED = 20261119
COUNTS = {'train': [128, 128, 64, 64, 64, 64], 'validation': [16, 16, 8, 8, 8, 8], 'test': [16, 16, 8, 8, 8, 8]}
FAMILIES = 'ABCDEF'
EXACT_INITIAL = 'train_A_exact_initial'
EXACT_TARGET = 'test_E_exact_marmousi'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def sha_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def sync_directory(path):
    if os.name == 'posix':
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(canonical(value) + b'\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def save_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)

def package_hash():
    root = Path(__file__).parent
    return digest({p.name: sha_file(p) for p in sorted(root.glob('*.py'))})

def load_contract(dataset):
    d = Path(dataset)
    c = read_json(d / 'acquisition_contract.json')
    actual = digest({k: v for k, v in c.items() if k != 'contract_hash'})
    if actual != c['contract_hash']:
        raise ValueError('Contract hash mismatch')
    for name, record in c['shared_arrays'].items():
        if sha_file(d / record['path']) != record['file_sha256']:
            raise ValueError('Changed contract array: ' + name)
    for key in ['truth', 'initial']:
        if sha_file(c['templates'][key+'_file']) != c['templates'][key+'_file_sha256']:
            raise ValueError('Changed immutable template: ' + key)
    for key in ['generator', 'protocol', 'legacy_reference']:
        if sha_file(c['solver'][key+'_file']) != c['solver'][key+'_file_sha256']:
            raise ValueError('Changed inherited numerical implementation: ' + key)
    if 'cpu_adapter_file' in c['solver'] and sha_file(c['solver']['cpu_adapter_file'])!=c['solver']['cpu_adapter_file_sha256']:
        raise ValueError('Changed CPU numerical implementation')
    if sha_file(c['source_fwi_config']) != c['source_fwi_config_sha256']:
        raise ValueError('Changed resolved full-domain FWI configuration')
    return c

def load_manifest(dataset):
    d = Path(dataset)
    c = load_contract(d)
    recipe = read_json(d / 'dataset_recipe.json')
    if digest({k: v for k, v in recipe.items() if k != 'dataset_recipe_hash'}) != recipe['dataset_recipe_hash']:
        raise ValueError('Recipe hash mismatch')
    rows = [json.loads(line) for line in (d / 'generation_manifest.jsonl').read_text().splitlines() if line]
    receipt = read_json(d / 'manifest_receipt.json')
    if digest(rows) != receipt['manifest_hash'] or recipe['contract_hash'] != c['contract_hash']:
        raise ValueError('Manifest/contract binding mismatch')
    return c, recipe, rows, receipt

def bindings(dataset):
    c, r, rows, m = load_manifest(dataset)
    return {'contract_hash': c['contract_hash'], 'dataset_recipe_hash': r['dataset_recipe_hash'], 'manifest_hash': m['manifest_hash']}

def stage_rows(rows, stage):
    if stage == 'full640':
        return sorted(rows, key=lambda r: r['priority'])
    return sorted([r for r in rows if stage in r['stages']], key=lambda r: r['priority'])

def rng_for(split, family, parent, component=0, retry=0):
    entropy = [SEED, {'train': 1, 'validation': 2, 'test': 3}[split], FAMILIES.index(family) + 1, int(parent), int(component), int(retry)]
    seq = np.random.SeedSequence(entropy)
    return np.random.default_rng(seq), {'entropy': entropy, 'spawn_key': list(seq.spawn_key)}
