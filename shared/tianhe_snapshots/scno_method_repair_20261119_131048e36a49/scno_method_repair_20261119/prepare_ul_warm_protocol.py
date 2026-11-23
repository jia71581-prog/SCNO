"""Prepare six true central-source validation fields and the UL warm-start split.

The inherited checkpoint and its original corpus must be restored first.
Then call load_ul_warm_protocol(parent_corpus, protocol_dir) for new rows.
No normalizer is fitted or replaced. TEST100 retain prior-exposure status.
"""
from scno_release_paths import local_asset as _scno_local_asset

import argparse
from collections import Counter
from copy import copy
from dataclasses import replace
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data_v2 import Corpus, ROOTS, _file_sha, _json_sha, _read_json, _write_json, open_corpus


ROOT = Path(_scno_local_asset('package'))
SOURCE = [1000., 1000., 15., .1, 1.]
SCHEMA = 'SCNO_UL_WARM200_FIXED_NORMALIZER_V1'
ORIGINAL_VAL_IDS = tuple('nyquist15_fresh800_20261004_r1:fresh_validation:' + value for value in (
    'uniform:0016', 'uniform:0050', 'uniform:0083',
    'layered:0017', 'layered:0056', 'layered:0084'))


class ULWarmCorpus(Corpus):
    """An ordinary Corpus view, with exposure-aware TEST access disabled by default."""

    def read(self, record_id, slots):
        if str(record_id) in self.test_ids and not self.allow_test_read:
            raise ValueError('warm-start TEST100 have prior CP exposure and are not loaded during training')
        return super().read(record_id, slots)


def _source_release(corpus):
    return {'kind': corpus.kind, 'root': str(corpus.root),
            'complete_sha256': _file_sha(corpus.root / 'DATASET_COMPLETE.json'),
            'records_sha256': _file_sha(corpus.root / 'RECORDS.json'),
            'normalization_file': str(corpus.root / 'normalization_ul.json'),
            'normalization_file_sha256': _file_sha(corpus.root / 'normalization_ul.json'),
            'effective_normalization_sha256': _json_sha(corpus.metadata)}


def load_ul_warm_protocol(parent_corpus, protocol_dir, *, allow_test_read=False):
    """Preserve the parent's normalization, rebind 900/6/100 roles, and return Corpus.

    Restore CP against its original config/corpus before calling this helper.
    Only source metadata and new small manifests are read here. Native TRAIN
    pressures remain in place; validation uses actual newly generated NPYs.
    """
    directory = Path(protocol_dir).resolve(strict=True)
    seal = _read_json(directory / 'PROTOCOL_COMPLETE.json')
    if seal.get('schema') != SCHEMA or seal.get('status') != 'COMPLETE':
        raise ValueError('the warm200 data protocol is not complete')
    for filename, digest in seal['artifacts'].items():
        if _file_sha(directory / filename) != digest:
            raise ValueError('the sealed warm200 protocol changed: ' + filename)
    spec = _read_json(directory / 'WARM_PROTOCOL.json')
    return _load_view(parent_corpus, directory, spec, allow_test_read)


def _load_view(parent, directory, spec, allow_test_read=False):
    if parent.kind != 'centered_uniform_layered' or _source_release(parent) != spec['source_release']:
        raise ValueError('warm CP and source release/normalizer differ; restore original CP binding first')
    train, test = tuple(spec['train_ids']), tuple(spec['test_ids'])
    if (len(train) != 900 or len(test) != 100 or len(set(train + test)) != 1000
            or set(train + test) != set(parent.train_ids)):
        raise ValueError('open the original complete TRAIN1000 corpus before wrapping 900/100')
    wrapper = object.__new__(ULWarmCorpus)
    wrapper.__dict__.update(parent.__dict__)
    wrapper._bound = {}
    for role, ids in (('train', train), ('test', test)):
        for key in ids:
            information = {**parent.records[key], 'split': role,
                           'warm_checkpoint_prior_training_exposure': True}
            wrapper._bound[key] = replace(parent._bound[key], information=information)
    wrapper._artifacts = dict(parent._artifacts)
    wrapper._add_holdout(directory / spec['validation_manifest'], 'validation')
    wrapper.records = {key: value.information for key, value in wrapper._bound.items()}
    validation = tuple(key for key, row in wrapper.records.items() if row['split'] == 'validation')
    if len(validation) != 6:
        raise ValueError('warm200 requires exactly six external central-source validation cases')
    all_training_media = {parent.records[key]['velocity_sha256'] for key in parent.train_ids}
    if any(wrapper.records[key]['velocity_sha256'] in all_training_media for key in validation):
        raise ValueError('external validation velocity already appears in the full TRAIN1000 release')
    if any(not np.array_equal(wrapper._bound[key].source[:2], [1000., 1000.])
           for key in train + test + validation):
        raise ValueError('warm200 training and external validation must use the same central source')
    wrapper.train_ids, wrapper.validation_ids, wrapper.test_ids = train, validation, test
    wrapper.allow_test_read = bool(allow_test_read)
    # metadata is intentionally unchanged; its numerical values are inherited
    # from the CP's original published normalizer, never from TRAIN900 refitting.
    wrapper.metadata = dict(parent.metadata)
    files = {key: {'pressure': str(rec.pressure_path), 'stamp': list(rec.pressure_stamp),
                   'velocity': str(rec.velocity_path) if rec.velocity_path else None,
                   'velocity_stamp': list(rec.velocity_stamp) if rec.velocity_stamp else None}
             for key, rec in wrapper._bound.items()}
    wrapper.binding = {'schema': SCHEMA, 'source_release': spec['source_release'],
                       'selected_ids': {'train': list(train), 'validation': list(validation), 'test': list(test)},
                       'source_file_metadata_sha256': _json_sha(files),
                       'actual_times_sha256': parent.binding['actual_times_sha256'],
                       'normalization_sha256': _json_sha(wrapper.metadata),
                       'warm_protocol_sha256': _file_sha(directory / 'WARM_PROTOCOL.json'),
                       'validation_manifest_sha256': _file_sha(directory / spec['validation_manifest']),
                       'prior_exposure': spec['prior_exposure']}
    wrapper.warm_protocol = spec
    return wrapper


def prepare(args):
    import build_centered_validation as teacher
    started = perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'PROTOCOL_COMPLETE.json').is_file():
        existing = load_ul_warm_protocol(open_corpus({'kind': 'centered_uniform_layered',
                                                    'root': str(args.source_root)}), args.out)
        print(json.dumps({'status': 'ALREADY_COMPLETE', 'protocol': str(args.out),
                          'train': len(existing.train_ids), 'validation': len(existing.validation_ids)}), flush=True)
        return
    base = open_corpus({'kind': 'centered_uniform_layered', 'root': str(args.source_root)})
    partition = _read_json(args.partition)
    if set(partition['train_ids'] + partition['test_ids']) != set(base.train_ids):
        raise ValueError('the fixed TEST100 partition does not match the source TRAIN1000 roster')
    rows, fields, times, physics, binding = teacher.review_inputs(args.repo, args.archive, args.source_root)
    selected = [(i, row) for i, row in enumerate(rows) if row['sample_id'] in ORIGINAL_VAL_IDS]
    if len(selected) != 6:
        raise ValueError('the frozen six validation media are missing from the verified roster')
    train_hashes = {base.records[key]['velocity_sha256'] for key in base.train_ids}
    if any(row['velocity_sha256'] in train_hashes for _, row in selected):
        raise ValueError('the requested validation velocities occur in TRAIN1000')
    new_rows, new_fields = [], []
    for i, row in selected:
        new_rows.append({**row, 'sample_id': row['sample_id'] + ':source1000',
                         'original_sample_id': row['sample_id'], 'source': SOURCE,
                         'source_model_fp32': np.asarray(SOURCE, np.float32).astype(float).tolist()})
        new_fields.append(fields[i])
    binding = {**binding, 'source_parameters': SOURCE, 'record_count': 6,
               'families': {'uniform': 3, 'layered': 3},
               'selected_original_ids': [row['original_sample_id'] for row in new_rows],
               'generation': 'new physical pressure solves at z1000; old shallow labels not relabelled',
               'training_velocity_overlap_full1000': 0,
               'normalization': 'reuse original CP published TRAIN normalizer unchanged'}
    validation_dir = args.validation_storage or args.out / 'validation_center1000'
    validation_dir = validation_dir.resolve()
    _write_json(args.out / 'GENERATION_PLAN.json', {'pid': os.getpid(), 'source': SOURCE,
                'records': [row['sample_id'] for row in new_rows], 'source_release': _source_release(base),
                'output_bytes_estimate': 6 * teacher.FIELD_BYTES, 'device': args.device,
                'validation_storage': str(validation_dir),
                'validation_storage_is_volatile': str(validation_dir).startswith('/dev/shm/')})
    print(json.dumps({'stage': 'GENERATING_SIX_CENTERED_VALIDATION', 'pid': os.getpid(),
                      'records': [row['sample_id'] for row in new_rows], 'device': args.device}), flush=True)
    teacher.generate(validation_dir, new_rows, np.stack(new_fields), times,
                     physics, binding, args.device, args.max_batch)
    spec = {'schema': SCHEMA, 'source_release': _source_release(base),
            'train_ids': partition['train_ids'], 'test_ids': partition['test_ids'],
            'fixed_partition_file': str(args.partition), 'fixed_partition_sha256': _file_sha(args.partition),
            'ignored_partition_refitted_normalizer': True,
            'validation_manifest': str(validation_dir / 'VALIDATION_MANIFEST.json'),
            'validation_storage_is_volatile': str(validation_dir).startswith('/dev/shm/'),
            'validation_storage_policy': 'native generated labels; preserve files throughout training/resume; archive when persistent disk is available',
            'prior_exposure': {'initial_checkpoint': str(args.initial_checkpoint),
                              'initial_checkpoint_sha256': _file_sha(args.initial_checkpoint),
                              'checkpoint_trained_on_original_train1000': True,
                              'test100_prior_checkpoint_training_exposure': True,
                              'test100_is_independent_test': False,
                              'test100_use': 'prior-exposed diagnostic only; not loaded during warm training',
                              'external_validation_velocity_overlap_full1000': 0},
            'new_training_epochs': 200, 'normalizer_policy': 'original CP published normalizer, fixed'}
    _write_json(args.out / 'WARM_PROTOCOL.json', spec)
    wrapper = _load_view(base, args.out, spec)
    audit = wrapper.audit()
    _write_json(args.out / 'CORPUS_AUDIT.json', audit)
    _write_json(args.out / 'LOADER_CONFIG.json', {
        'restore_checkpoint_first_against_original_corpus': True,
        'loader_module': 'prepare_ul_warm_protocol', 'loader_function': 'load_ul_warm_protocol',
        'protocol_dir': str(args.out), 'train': 900, 'validation': 6, 'test': 100,
        'normalizer': spec['source_release'], 'test_labels_load': False})
    _write_json(args.out / 'PROTOCOL_COMPLETE.json', {'schema': SCHEMA, 'status': 'COMPLETE',
        'elapsed_s': perf_counter() - started,
        'artifacts': {name: _file_sha(args.out / name) for name in (
            'WARM_PROTOCOL.json', 'CORPUS_AUDIT.json', 'LOADER_CONFIG.json',
            str(validation_dir / 'VALIDATION_MANIFEST.json'))}})
    print(json.dumps({'status': 'COMPLETE', 'pid': os.getpid(), 'protocol': str(args.out),
                      'train': 900, 'validation': 6, 'test': 100,
                      'elapsed_s': perf_counter() - started}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path(ROOTS['centered_uniform_layered']))
    p.add_argument('--partition', type=Path, default=ROOT/'results/pretraining/scno_fno_time_channels_20261119/protocols/ul900_test100_center1000_20261119/PARTITION.json')
    p.add_argument('--repo', type=Path, default=Path(_scno_local_asset('project')))
    p.add_argument('--archive', type=Path, default=Path.home()/'acoustic_centered_20261110/dataset_uniform500_layered434_raw801_centered')
    p.add_argument('--initial-checkpoint', type=Path, default=ROOT/'results/pretraining/scno_fno_time_channels_20261119/uniform_layered/long_200e/checkpoint.pt')
    p.add_argument('--out', type=Path, default=ROOT/'results/pretraining/scno_ul_warm200_20261119/data_protocol')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--validation-storage', type=Path)
    p.add_argument('--max-batch', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(2)
    prepare(args)


if __name__ == '__main__':
    main()
