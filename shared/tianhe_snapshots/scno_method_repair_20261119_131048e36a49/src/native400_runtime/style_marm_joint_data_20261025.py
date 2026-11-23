"""Completed mixed TRAIN corpus; exact native401 -> nodal201 labels only.

No historical checkpoints, TEST/VAL labels, accelerator, or old normalizer are
opened. The publisher owns full source-file audits; this reader additionally
binds the published files and rejects missing VDS sources before reading data.
"""
from pathlib import Path
import hashlib
import json
import math

import h5py
import numpy as np
import torch

from .feature_enhanced_objective_20261022 import SampledLabels

SCHEMA = 'STYLE_MARM_JOINT201_V1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def roster_digest(records):
    raw = json.dumps(records, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def restrict401(value):
    value = np.asarray(value)
    if value.shape[-2:] != (401, 401):
        raise ValueError('exact native401 spatial axes required')
    return np.ascontiguousarray(value[..., ::2, ::2])


class JointTrainingData:
    def __init__(self, root):
        self.root = Path(root).resolve()
        marker = self.root / 'DATASET_COMPLETE.json'
        if not marker.is_file():
            raise RuntimeError('WAITING_DATASET_COMPLETE: ' + str(marker))
        complete = json.loads(marker.read_text())
        self.complete = complete
        expected = dict(schema=SCHEMA, status='COMPLETE', training_grid=[201, 201],
                        teacher_grid=[401, 401], time_frames=801, pressure_interpolation=False)
        if any(complete.get(k) != v for k, v in expected.items()):
            raise ValueError('completed native401/nodal201 dataset contract differs')
        names = ('RECORDS.json', 'normalization_stats_train.json', 'dataset_train.h5')
        artifacts = complete.get('artifacts', {})
        self.file_hashes = {}
        for name in names:
            expected_hash = artifacts.get(name)
            if isinstance(expected_hash, dict):
                expected_hash = expected_hash.get('sha256')
            if not expected_hash or sha(self.root / name) != expected_hash:
                raise ValueError('completion artifact SHA missing/mismatched: ' + name)
            self.file_hashes[name] = expected_hash
        records = json.loads((self.root / 'RECORDS.json').read_text())
        self.records = tuple(r for r in records if r['split'] == 'train')
        self.validation_records = tuple(r for r in records if r['split'] == 'validation')
        if len(self.records) != complete['train_records'] or not self.records:
            raise ValueError('TRAIN census differs from completion marker')
        if len({r['record_id'] for r in records}) != len(records):
            raise ValueError('duplicate global record IDs')
        if any(r['family'] not in ('style_a', 'marmousi_crop') for r in self.records):
            raise ValueError('paper TEST medium is not an admitted training family')
        if any(r['split'] != 'train' for r in records if r['family'] == 'style_a'):
            raise ValueError('all selected Style-A records must be TRAIN')
        self.normalization = json.loads((self.root / 'normalization_stats_train.json').read_text())
        for key in ('pressure_rms', 'receiver_rms', 'velocity_std_mps'):
            if not math.isfinite(self.normalization[key]) or self.normalization[key] <= 0:
                raise ValueError('positive finite TRAIN statistic required: ' + key)
        if not math.isfinite(self.normalization['velocity_mean_mps']):
            raise ValueError('finite TRAIN velocity mean required')
        self.normalization = dict(self.normalization, record_count=len(self.records),
                                  train_manifest_sha256=roster_digest(self.records))
        self.path = self.root / 'dataset_train.h5'
        self._identities = {}
        for name in (*names, 'DATASET_COMPLETE.json'):
            self._track(self.root/name)
        with h5py.File(self.path, 'r') as h:
            n = len(self.records)
            for key, shape in [('pressure201', (n, 801, 201, 201)),
                               ('velocity201', (n, 201, 201)),
                               ('source_parameters', (n, 5)), ('time_field_s', (801,)),
                               ('record_ids', (n,))]:
                if h[key].shape != shape:
                    raise ValueError('published array shape differs: ' + key)
                if h[key].is_virtual:
                    for mapping in h[key].virtual_sources():
                        filename = mapping.file_name
                        if isinstance(filename, bytes):
                            filename = filename.decode()
                        source_path = (self.path.parent / filename).resolve()
                        if not source_path.is_file():
                            raise ValueError('missing VDS source: ' + str(source_path))
                        self._track(source_path)
            if h['pressure201'].dtype != np.float32:
                raise ValueError('physical pressure must remain FP32')
            if list(h['record_ids'][:]) != [r['record_id'] for r in self.records]:
                raise ValueError('H5 TRAIN order differs from the immutable roster')
            times = np.asarray(h['time_field_s'][:], np.float32)
            expected_times = (np.arange(801, dtype=np.float64) * .005).astype(np.float32)
            if not np.array_equal(times, expected_times):
                raise ValueError('physical 0..4s, 801-frame time axis differs')
            self.times = torch.from_numpy(times)
        self.binding = dict(schema=SCHEMA, root=str(self.root), train_count=len(self.records),
                            file_hashes=self.file_hashes, complete_sha256=sha(marker),
                            train_manifest_sha256=self.normalization['train_manifest_sha256'],
                            label_operation='native401[:,::2,::2]; no pressure interpolation',
                            labels_normalized=False, val_test_labels_read=False)

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _identity(path):
        s = path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def _track(self, path):
        path = Path(path)
        identity = self._identity(path)
        if path in self._identities and self._identities[path] != identity:
            raise ValueError('immutable dataset file changed: ' + str(path))
        self._identities[path] = identity

    def _check_unchanged(self):
        for path in self._identities:
            if not path.is_file() or self._identity(path) != self._identities[path]:
                raise ValueError('immutable dataset file changed or disappeared: ' + str(path))

    def batch(self, ids, frames):
        return self._batch(ids, frames, self.records, self.path)

    def validation_batch(self, ids, frames):
        """Explicit monitoring-only source holdout; never used for normalization."""
        path = self.root / 'dataset_validation.h5'
        expected = self.complete.get('artifacts', {}).get(path.name)
        if isinstance(expected, dict):
            expected = expected.get('sha256')
        if not expected or sha(path) != expected:
            raise ValueError('validation VDS must be bound to the completion receipt')
        if len(self.validation_records) != self.complete['validation_records']:
            raise ValueError('validation roster census differs')
        self._track(path)
        return self._batch(ids, frames, self.validation_records, path)

    def _batch(self, ids, frames, records, path):
        self._check_unchanged()
        ids, frames = list(ids), sorted(set(int(f) for f in frames))
        if not ids or any(type(i) is not int or not 0 <= i < len(records) for i in ids):
            raise ValueError('only positional IDs in the selected roster may be read')
        if not frames or frames[0] < 0 or frames[-1] > 800:
            raise ValueError('physical frame indices must be in 0..800')
        c, sources, labels = [], [], []
        with h5py.File(path, 'r') as h:
            if h['pressure201'].shape != (len(records), 801, 201, 201):
                raise ValueError('selected split pressure shape differs')
            if h['pressure201'].is_virtual:
                for mapping in h['pressure201'].virtual_sources():
                    filename = mapping.file_name.decode() if isinstance(mapping.file_name, bytes) else mapping.file_name
                    source_path = (path.parent/filename).resolve()
                    if not source_path.is_file():
                        raise ValueError('missing VDS source: ' + str(source_path))
                    self._track(source_path)
            for i in ids:
                if int(h['record_ids'][i]) != records[i]['record_id']:
                    raise ValueError('H5 positional ID does not match the selected split')
                c.append(np.asarray(h['velocity201'][i], np.float32))
                source = np.asarray(h['source_parameters'][i], np.float32)
                if not np.array_equal(source, np.asarray(records[i]['source_parameters'], np.float32)):
                    raise ValueError('per-record source differs from manifest')
                sources.append(source)
                labels.append(np.asarray(h['pressure201'][i, frames], np.float32))
        c, sources, labels = map(np.stack, (c, sources, labels))
        if not all(np.isfinite(a).all() for a in (c, sources, labels)) or not (c > 0).all():
            raise ValueError('nonfinite labels/source or invalid velocity')
        return dict(ids=ids, c=torch.from_numpy(c[:, None]), source=torch.from_numpy(sources),
                    labels=SampledLabels(frames, torch.from_numpy(labels)))
