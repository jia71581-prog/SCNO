"""Generate separate, source-verified 8 Hz Marmousi TRAIN pilot labels."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_bound(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'Unexpected provenance hash for {path}')
    return json.loads(path.read_text())


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare(config, limit):
    parent = Path(config['parent'])
    rows = load_bound(parent/'RECORDS.json', config['parent_records_sha256'])
    splits = load_bound(parent/'SPLITS.json', config['parent_splits_sha256'])
    train_ids = set(splits['train'])
    if train_ids & (set(splits['test']) | set(splits['validation'])):
        raise ValueError('Source roster has overlapping shot IDs')
    selected = {}
    for row in rows:
        if row['record_id'] not in train_ids:
            continue
        if row['split'] != 'train' or row['family'] != 'marmousi_crop':
            raise ValueError('Pilot requires registered Marmousi TRAIN crops')
        selected.setdefault(row['medium_id'], row)
    held_out_test = {row['medium_id'] for row in rows if row['record_id'] in splits['test']}
    if held_out_test & selected.keys():
        raise ValueError('A registered TEST medium appears in the TRAIN crops')
    if not 1 <= limit <= min(config['max_crops'], len(selected)):
        raise ValueError('Requested size exceeds the registered TRAIN pilot')
    return parent, list(selected.values())[:limit]


def atomic_record(path, wave, fine, source, row):
    temporary = path.with_suffix('.partial')
    with h5py.File(temporary, 'x') as file:
        file.attrs.update(split='train', family='marmousi_crop', units='Pa', axis_order='TZX',
                          medium_id=row['medium_id'], parent_record_id=row['record_id'])
        file.create_dataset('wavefield', data=wave, chunks=(1, 201, 201), compression='lzf', shuffle=True)
        file['velocity401'] = fine
        file['velocity201'] = fine[::2, ::2]
        file['source_solver'] = source
        file['time_s'] = np.arange(len(wave), dtype=np.float64)*0.00125
    with h5py.File(temporary, 'r') as file:
        for first in range(0, len(wave), 32):
            if not np.array_equal(file['wavefield'][first:first+32], wave[first:first+32]):
                raise ValueError('Label changed on HDF5 readback')
        if not np.array_equal(file['source_solver'][:], source):
            raise ValueError('Source changed on HDF5 readback')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    parent, rows = prepare(config, args.limit)
    physics = load_bound(Path(config['teacher_contract']), config['teacher_contract_sha256'])['physics_config']
    f0 = config['source']['peak_frequency_hz']
    delay = config['source']['delay_s']
    shape = tuple(config['output']['shape_tzx'])
    if f0 != 8 or delay != 1.5/f0 or shape != (801, 201, 201):
        raise ValueError('Unexpected 8 Hz pilot contract')
    if args.dry_run:
        print(json.dumps({'status': 'VERIFIED_INPUT_PLAN', 'device': args.device, 'count': len(rows),
                          'f0_hz': f0, 't0_s': delay, 'output_shape': shape,
                          'medium_ids': [row['medium_id'] for row in rows]}))
        return
    if args.device == 'cuda':
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                             '--format=csv,noheader,nounits'], text=True)
        if processes.strip():
            raise RuntimeError('GPU has an active compute process')
    if shutil.disk_usage(args.out.parent).free < len(rows)*140_000_000 + 1_073_741_824:
        raise RuntimeError('Insufficient free disk for requested pilot and reserve')
    args.out.mkdir(exist_ok=True)
    import torch
    sys.path.insert(0, str(Path(config['teacher_project'])/'src'))
    from label_quality_fd.data_generation.grid import AcousticGrid, BoundaryConfig
    from label_quality_fd.data_generation.raw_dataset import NodalLWC84Solver
    from label_quality_fd.data_generation.ricker import ricker_triplet

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    b = physics['boundaries']
    boundary_keys = ('top', 'left', 'right', 'bottom', 'npml', 'cpml_target_reflection',
                     'cpml_polynomial_order', 'cpml_outside_physical_domain')
    times = np.arange(shape[0], dtype=np.float64)*config['output']['dt_s']
    teacher = NodalLWC84Solver(
        grid=AcousticGrid(**{key: value for key, value in physics['grid'].items() if key != 'dimension'}),
        boundaries=BoundaryConfig(**{key: b[key] for key in boundary_keys}), dt_s=physics['time']['dt_used_s'],
        output_times_s=times, c_ref_mps=physics['solver']['c_ref_mps'], device=args.device,
        dtype=torch.float32, kappa_max=b['kappa_max'], minimum_frequency_hz=b['minimum_frequency_hz'],
        output_restriction_factor=2, cuda_graphs=args.device == 'cuda',
        block_step_count=10 if args.device == 'cuda' else None)
    frequency = np.fft.rfftfreq(8192, config['output']['dt_s'])
    wavelet = ricker_triplet(times, f0_hz=f0, t0_s=delay)[0]
    spectral_peak = float(frequency[np.abs(np.fft.rfft(wavelet, n=8192)).argmax()])
    if abs(spectral_peak - f0) >= .1:
        raise ValueError(f'Unexpected Ricker FFT peak {spectral_peak}')
    receipts = []
    config_hash = sha256(args.config)
    for row in rows:
        source = np.array([*row['source_parameters'][:2], f0, delay, config['source']['amplitude']], dtype=np.float64)
        parent_model = parent/row['model_file']
        if sha256(parent_model) != row['model_sha256']:
            raise ValueError(f'Marmousi velocity hash mismatch: {row["medium_id"]}')
        fine = np.load(parent_model, allow_pickle=False)
        if fine.shape != (401, 401) or not np.isfinite(fine).all() or fine.min() <= 0:
            raise ValueError('Expected the registered positive fine-grid velocity')
        path = args.out/(row['medium_id'] + '.h5')
        receipt = path.with_suffix('.json')
        if path.exists() or receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous['config_sha256'] != config_hash or previous['sha256'] != sha256(path):
                raise ValueError(f'Existing receipt does not match: {path}')
            receipts.append(previous)
            continue
        start = time.perf_counter()
        kwargs = dict(zip(('source_x_m','source_z_m','source_f0_hz','source_t0_s','source_amplitude'), source))
        with torch.no_grad():
            if args.device == 'cuda':
                wave = teacher.simulate_device(fine[None], **kwargs).wavefield_device[0].cpu().numpy()
            else:
                wave = teacher.simulate(fine[None], **kwargs).wavefield[0]
        if (wave.shape != shape or wave.dtype != np.float32 or not np.isfinite(wave).all()
                or np.any(wave[0]) or np.any(wave[:,0]) or not np.any(wave[-1])):
            raise ValueError('Marmousi 8 Hz label failed shape, finite, initial or boundary checks')
        atomic_record(path, wave, fine, source, row)
        rec = {'status': 'PASS', 'split': 'train', 'medium_id': row['medium_id'],
               'parent_record_id': row['record_id'], 'parent_model_sha256': row['model_sha256'],
               'source_solver': source.tolist(), 'wavelet_fft_peak_hz': spectral_peak,
               'shape': list(shape), 'max_abs_pa': float(max(wave.max(), -wave.min())),
               'full_field_l2_pa': float(np.sqrt(np.einsum('ijk,ijk->', wave, wave, dtype=np.float64))),
               'config_sha256': config_hash, 'seconds': time.perf_counter()-start,
               'device': args.device, 'file': str(path), 'sha256': sha256(path)}
        atomic_json(receipt, rec)
        receipts.append(rec)
        print(json.dumps(rec), flush=True)
    atomic_json(args.out/'INDEX.json', {'status': 'COMPLETE', 'scope': 'TRAIN pilot', 'records': receipts})


if __name__ == '__main__':
    main()
