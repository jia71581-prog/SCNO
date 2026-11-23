"""Verified TRAIN500 batches, complete-field losses and bounded CPU prefetch.

Importing or using this module never starts CUDA, training, or a controller.
Production batches contain raw pressure in Pa; the worker copies it to the
device before the inherited FP32 division by ``scale``. Only immutable raw
inputs are cached. Neither caches nor pending futures belong in checkpoints.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / 'data/marmousi_pglopt_train500_raw400_20261020'
EXPECTED_COUNT = 500
PRESSURE_SHAPE = (400, 201, 201)
DATA_SCHEMA = 'PAPER_MARMOUSI201_PGLOPT_RAW400_DATA_V1'
NORMALIZER_SHA = '1df298243715f12b11fcc68132a9df84081e7c7aa6c17960417b64be93073c52'
PRESSURE_SCALE_PA = 1.252277344221455e-8


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _identity(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _array_sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def _normalization(raw):
    return dict(velocity_mean=raw['velocity_center_mps'], velocity_std=raw['velocity_scale_mps'],
                source_mean=[0, 0, 0, 0, 0], source_std=raw['source_scales'],
                coord_extent_m=[2000, 2000], time_scale_s=1.0)


class Marmousi500Dataset:
    """Read-only verified roster and lazy, bounded cache of original Pa arrays.

    Every small manifest/receipt is hash checked at construction. Pressure
    contents are SHA checked, finite checked and surface checked on first read.
    Subsequent uses check the original filesystem identity and timestamps;
    cached bytes remain read-only and are never re-normalized in place.
    """

    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        options = cfg.get('data', cfg)
        self.root = Path(options.get('dataset_path', options.get('dataset', DEFAULT_DATASET))).resolve()
        meta_path = self.root / 'DATASET.json'
        self.metadata = meta = _read(meta_path)
        self.dataset_sha256 = sha256(meta_path)
        if 'dataset_sha256' in options:
            require(options['dataset_sha256'] == self.dataset_sha256, 'Configured dataset SHA mismatch')
        complete_path = self.root / 'DATASET_COMPLETE.json'
        complete = _read(complete_path)
        require(meta['schema'] == DATA_SCHEMA and meta['count'] == EXPECTED_COUNT,
                'Exactly the complete TRAIN500 dataset is required')
        require(tuple(meta['pressure_shape']) == PRESSURE_SHAPE and meta['pressure_dtype'] == 'float32'
                and meta['pressure_units'] == 'Pa', 'Original full400 raw-Pa shape/dtype/units required')
        require(meta['pressure_scale_pa'] == PRESSURE_SCALE_PA
                and meta['normalizer_sha256'] == NORMALIZER_SHA, 'Inherited pressure normalization required')
        require(complete['status'] == 'COMPLETE' and complete['completed_records'] == EXPECTED_COUNT
                and complete['expected_records'] == EXPECTED_COUNT
                and complete['dataset_sha256'] == self.dataset_sha256, 'Dataset completion binding mismatch')
        require([r['row'] for r in complete['records']] == list(range(EXPECTED_COUNT)),
                'Complete manifest must cover every TRAIN row exactly once')
        self._bound_identities = {str(meta_path): _identity(meta_path), str(complete_path): _identity(complete_path)}
        for name, entry in meta['files'].items():
            path = self.root / name
            require(path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'],
                    'Dataset input SHA mismatch: ' + name)
            self._bound_identities[str(path)] = _identity(path)
        normalizer = _read(self.root / 'normalization_ul.json')
        self.norm = _normalization(normalizer)
        require(normalizer['pressure_scale_pa'] == PRESSURE_SCALE_PA, 'Frozen pressure scale changed')
        if 'normalization' in cfg:
            require(cfg['normalization'] == self.norm, 'Feature normalization must be inherited exactly')
        if 'pressure_rms_pa' in cfg:
            require(cfg['pressure_rms_pa'] == PRESSURE_SCALE_PA, 'Configuration pressure scale mismatch')
        self.scale = PRESSURE_SCALE_PA
        roster = _read(self.root / 'TRAIN_ROSTER.json')
        self.records = roster['records']
        require(roster['count'] == len(self.records) == EXPECTED_COUNT, 'TRAIN roster length mismatch')
        self.velocity = np.load(self.root / 'velocity201.npy', allow_pickle=False)
        self.sources = np.load(self.root / 'sources_fp32.npy', allow_pickle=False)
        source64 = np.load(self.root / 'sources_fp64.npy', allow_pickle=False)
        self.times = np.load(self.root / 'times_fp32.npy', allow_pickle=False)
        self.frames = np.load(self.root / 'full_frames.npy', allow_pickle=False)
        native = np.load(self.root / 'native_times_fp32.npy', allow_pickle=False)
        require(self.velocity.shape == PRESSURE_SHAPE[1:] and self.velocity.dtype == np.float32
                and np.isfinite(self.velocity).all() and (self.velocity > 0).all(), 'Invalid sharp velocity')
        require(_array_sha(self.velocity) == meta['velocity201_payload_sha256'], 'Velocity payload mismatch')
        require(self.sources.shape == (EXPECTED_COUNT, 5) and self.sources.dtype == np.float32
                and source64.shape == self.sources.shape and source64.dtype == np.float64
                and np.isfinite(self.sources).all() and np.array_equal(source64.astype(np.float32), self.sources),
                'Source FP64 registration / FP32 model codec mismatch')
        require(self.times.shape == (PRESSURE_SHAPE[0],) and self.times.dtype == np.float32
                and self.frames.shape == self.times.shape and self.frames.dtype == np.int64
                and np.isfinite(self.times).all() and (np.diff(self.times) > 0).all(), 'Invalid complete400 time axis')
        exact_native = np.asarray(np.arange(801, dtype=np.float64) * .00125, dtype=np.float32)
        require(np.array_equal(native, exact_native)
                and np.array_equal(self.frames, np.asarray(meta['full_frames'], dtype=np.int64))
                and np.array_equal(self.times, native[self.frames]), 'Corrected native times must match bitwise')
        require(np.array_equal(self.sources[:, 2:], np.broadcast_to(np.array([15., .1, 1.], np.float32),
                                                                   (EXPECTED_COUNT, 3))),
                'Registered 15 Hz / t0 .1 / amplitude1 sources required')
        require(len({tuple(row[:2]) for row in self.sources}) == EXPECTED_COUNT, 'Source positions must be distinct')
        parent = meta.get('imported_parent')
        if parent is not None:
            parent_path = self.root / parent['manifest']
            require(sha256(parent_path) == parent['dataset_sha256'], 'Imported parent manifest changed')
            old = _read(parent_path)
            for key in ('schema', 'pressure_shape', 'pressure_dtype', 'pressure_units', 'pressure_scale_pa',
                        'full_frames', 'normalizer_sha256', 'protocol_sha256', 'velocity201_payload_sha256', 'grid'):
                require(old[key] == meta[key], 'Imported physical binding differs: ' + key)
            self._bound_identities[str(parent_path)] = _identity(parent_path)
        self._pressure_entries = []
        self._receipt_hashes = []
        self._pressure_identities = []
        self._receipt_identities = []
        for index, (row, committed) in enumerate(zip(self.records, complete['records'])):
            require(row['row'] == index and row['sample_id'] == f'train_marmousi_n15_{index:04d}'
                    and row['split'] == 'train' and row['medium_type'] == 'marmousi', 'TRAIN row identity mismatch')
            require((row['crop_x0_m'], row['crop_z0_m'], row['mirror_x']) == (4500., 0., False),
                    'Only the original Marmousi crop is allowed')
            require(np.array_equal(np.asarray(row['source_model_fp32'], np.float32), self.sources[index])
                    and np.array_equal(np.asarray(row['source'], np.float64), source64[index]), 'Roster source mismatch')
            directory = self.root / 'shards' / f'{index:06d}'
            receipt_path = directory / 'RECORD.json'
            receipt_hash = sha256(receipt_path)
            require(receipt_hash == committed['receipt_sha256'], 'Committed receipt changed')
            receipt = _read(receipt_path)
            require(receipt['schema'] == DATA_SCHEMA and receipt['status'] == 'COMPLETE'
                    and receipt['row'] == index and receipt['sample_id'] == row['sample_id'] == committed['sample_id'],
                    'Receipt source identity mismatch')
            if receipt['dataset_sha256'] != self.dataset_sha256:
                require(parent is not None and receipt['dataset_sha256'] == parent['dataset_sha256']
                        and parent['receipts'].get(str(index)) == receipt_hash, 'Unbound parent import')
            require(np.array_equal(np.asarray(receipt['source_model_fp32'], np.float32), self.sources[index])
                    and np.array_equal(np.asarray(receipt['source_registered_fp64'], np.float64), source64[index]),
                    'Receipt source codec mismatch')
            require(receipt['pressure_units'] == 'Pa' and receipt['pressure_processing'] == 'nodal_stride2_NO_FILTER'
                    and receipt['quality']['finite'] and receipt['quality']['free_surface_exact_zero'],
                    'Raw pressure QC receipt required')
            pressure_path = directory / 'pressure_pa.npy'
            entry = receipt['files']['pressure_pa.npy']
            require(entry['sha256'] == committed['pressure_sha256']
                    and pressure_path.stat().st_size == entry['bytes'], 'Pressure binding / size mismatch')
            self._pressure_entries.append(dict(path=str(pressure_path), **entry))
            self._receipt_hashes.append(receipt_hash)
            self._pressure_identities.append(_identity(pressure_path))
            self._receipt_identities.append(_identity(receipt_path))
        self.cache_records = options.get('cache_records', EXPECTED_COUNT)
        require(type(self.cache_records) is int and 0 <= self.cache_records <= EXPECTED_COUNT, 'Bounded cache required')
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._row_locks = [threading.Lock() for _ in self.records]
        self._verified = set()
        self._hash_reads = 0
        self._cache_hits = 0
        height, width = PRESSURE_SHAPE[1:]
        x = np.arange(width, dtype=np.float32) * np.float32(10. / self.norm['coord_extent_m'][0])
        z = np.arange(height, dtype=np.float32) * np.float32(10. / self.norm['coord_extent_m'][1])
        zz, xx = np.meshgrid(z, x, indexing='ij')
        self._static_maps = np.stack(((self.velocity - np.float32(self.norm['velocity_mean'])) /
                                     np.float32(self.norm['velocity_std']), xx, zz))
        self._source_normalized = (self.sources - np.asarray(self.norm['source_mean'], np.float32)) / np.asarray(
            self.norm['source_std'], np.float32)
        for array in (self.velocity, self.sources, self.times, self.frames, self._static_maps, self._source_normalized):
            array.flags.writeable = False

    def __len__(self):
        return len(self.records)

    def _guard(self, index):
        require(_identity(self._pressure_entries[index]['path']) == self._pressure_identities[index],
                'Pressure file changed after binding')
        receipt = self.root / 'shards' / f'{index:06d}' / 'RECORD.json'
        require(_identity(receipt) == self._receipt_identities[index], 'Receipt changed after binding')

    def verify_metadata_unchanged(self):
        for path, expected in self._bound_identities.items():
            require(_identity(path) == expected, 'Dataset metadata changed after binding: ' + path)

    def _raw(self, index):
        require(type(index) is int and 0 <= index < len(self), 'TRAIN index out of bounds')
        with self._row_locks[index]:
            self._guard(index)
            with self._lock:
                if index in self._cache:
                    self._cache_hits += 1
                    self._cache.move_to_end(index)
                    return self._cache[index]
            entry = self._pressure_entries[index]
            if index not in self._verified:
                require(sha256(entry['path']) == entry['sha256'], 'Pressure SHA mismatch')
            array = np.load(entry['path'], allow_pickle=False)
            require(array.shape == PRESSURE_SHAPE and array.dtype == np.float32
                    and array.flags.c_contiguous, 'Full400 raw pressure contract required')
            if index not in self._verified:
                require(np.isfinite(array).all() and not np.any(array[:, 0]), 'Pressure finite/free-surface failure')
            self._guard(index)
            array.flags.writeable = False
            with self._lock:
                if index not in self._verified:
                    self._verified.add(index)
                    self._hash_reads += 1
                if self.cache_records:
                    self._cache[index] = array
                    while len(self._cache) > self.cache_records:
                        self._cache.popitem(last=False)
            return array

    def load_batch(self, indices, device='cpu'):
        require(str(device) == 'cpu', 'This loader is CPU-only; the worker owns CUDA transfer')
        indices = list(indices)
        require(all(isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)) for index in indices),
                'TRAIN indices must be integers, not coerced floats or booleans')
        ids = [int(index) for index in indices]
        require(ids and len(set(ids)) == len(ids), 'A batch must contain nonempty distinct TRAIN rows')
        self.verify_metadata_unchanged()
        tick = time.monotonic()
        pressure = torch.empty((len(ids), *PRESSURE_SHAPE), dtype=torch.float32)
        inputs = torch.empty((len(ids), 8, *PRESSURE_SHAPE[1:]), dtype=torch.float32)
        for local, index in enumerate(ids):
            array = self._raw(index)
            # np.copyto avoids exporting writable aliases to read-only cache.
            np.copyto(pressure[local].numpy(), array)
            maps = inputs[local].numpy()
            maps[0], maps[6], maps[7] = self._static_maps
            maps[1:6] = self._source_normalized[index, :, None, None]
        options = self.cfg.get('data', self.cfg)
        truth = pressure / self.scale if options.get('prepare_normalized_truth', False) else None
        return dict(inputs=inputs, pressure_pa=pressure, truth=truth,
                    times=torch.from_numpy(self.times.copy()), time_s=self.times.copy(),
                    scale=self.scale, ids=ids, shape=PRESSURE_SHAPE, norm=copy.deepcopy(self.norm),
                    source_model_fp32=self.sources[ids].copy(), prepare_seconds=time.monotonic()-tick)

    def cache_stats(self):
        with self._lock:
            return dict(cached_records=len(self._cache), verified_pressure_records=len(self._verified),
                        pressure_hash_reads=self._hash_reads, cache_hits=self._cache_hits,
                        cached_payload_bytes=sum(a.nbytes for a in self._cache.values()))

    def state_dict(self):
        """Stable data bindings only: no cache, future, global RNG or live cursor."""
        return dict(schema='BASELINE50_TRAIN500_BINDING_V1', dataset_path=str(self.root),
                    dataset_sha256=self.dataset_sha256,
                    completion_sha256=sha256(self.root / 'DATASET_COMPLETE.json'),
                    roster_sha256=self.metadata['files']['TRAIN_ROSTER.json']['sha256'],
                    times_sha256=self.metadata['files']['times_fp32.npy']['sha256'],
                    sources_sha256=self.metadata['files']['sources_fp32.npy']['sha256'],
                    pressure_sha256=[e['sha256'] for e in self._pressure_entries],
                    receipt_sha256=list(self._receipt_hashes), count=len(self),
                    shape=list(PRESSURE_SHAPE), normalization=copy.deepcopy(self.norm),
                    pressure_rms_pa=self.scale, full_native_frames=self.frames.tolist())


Dataset = Marmousi500Dataset


def load_roster(cfg):
    return Marmousi500Dataset(cfg)


def load_batch(records, indices, cfg=None):
    if cfg is not None:
        require(cfg == records.cfg, 'Batch configuration changed after roster binding')
    return records.load_batch(indices)


def epoch_order(count, seed, epoch):
    require(type(count) is int and count > 0 and type(seed) is int and seed >= 0
            and type(epoch) is int and epoch >= 0, 'Positive roster and nonnegative integer seed/epoch required')
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch])))
    return generator.permutation(count).tolist()


def epoch_batches(order, batch_size, offset=0):
    require(type(batch_size) is int and batch_size > 0 and type(offset) is int and 0 <= offset <= len(order),
            'Valid batch size and resume offset required')
    require(sorted(order) == list(range(len(order))), 'Epoch order must be a complete permutation')
    return [list(order[start:start+batch_size]) for start in range(offset, len(order), batch_size)]


class Prefetch:
    """Ordered iterator with at most two future CPU batches; exceptions propagate."""

    def __init__(self, dataset, batches, *, depth=2, workers=2):
        require(type(depth) is int and 1 <= depth <= 2 and type(workers) is int and 1 <= workers <= 4,
                'At most two future batches and one to four CPU workers required')
        self.dataset, self.batches, self.depth = dataset, iter(batches), depth
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='baseline50-input')
        self.pending = deque()
        self.closed = False
        self._fill()

    def _fill(self):
        while not self.closed and len(self.pending) < self.depth:
            try:
                ids = list(next(self.batches))
            except StopIteration:
                break
            self.pending.append((ids, self.pool.submit(self.dataset.load_batch, ids)))

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed or not self.pending:
            self.close()
            raise StopIteration
        ids, future = self.pending.popleft()
        ready, tick = future.done(), time.monotonic()
        try:
            batch = future.result()
            require(batch['ids'] == ids, 'Prefetched source order changed')
            batch['prefetch'] = dict(ready_before_take=ready, wait_seconds=time.monotonic()-tick,
                                     prepare_seconds=batch['prepare_seconds'])
            self._fill()
            return batch
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pending.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def fixed_supervision(truth, boundaries, receiver_indices):
    """Freeze each example's old four-window and receiver denominators."""
    require(truth.ndim == 4 and truth.dtype == torch.float32 and truth.shape[1] == 400,
            'FP32 (B,400,H,W) full-field supervision required')
    lengths = [b-a for a, b in zip(boundaries[:-1], boundaries[1:])]
    require(list(boundaries) == [0, 100, 200, 300, 400] and lengths == [100] * 4,
            'The original four complete equal windows are required')
    receiver = torch.as_tensor(receiver_indices, device=truth.device, dtype=torch.long)
    require(receiver.ndim == 2 and receiver.shape[1] == 2 and receiver.shape[0] > 0,
            'Nonempty receiver [z,x] pairs required')
    pairs = receiver.detach().cpu().tolist()
    require(len({tuple(pair) for pair in pairs}) == len(pairs)
            and all(0 <= z < truth.shape[2] and 0 <= x < truth.shape[3] for z, x in pairs),
            'Receiver positions must be unique and in bounds')
    with torch.no_grad():
        frame_energy = truth.square().mean(dim=(-2, -1), dtype=torch.float32)
        energy = frame_energy.mean(dim=1, dtype=torch.float32)
        shares = torch.stack([frame_energy[:, a:b].sum(dim=1, dtype=torch.float32)
                              for a, b in zip(boundaries[:-1], boundaries[1:])], dim=1) / frame_energy.sum(dim=1)[:, None]
        weights = .9 + .025 / shares.clamp_min(.001)
        receiver_energy = truth[:, :, receiver[:, 0], receiver[:, 1]].square().sum(dim=(1, 2), dtype=torch.float32)
        require(bool((torch.isfinite(energy) & (energy > 0) & torch.isfinite(receiver_energy)
                      & (receiver_energy > 0)).all()), 'Each example needs positive finite field/receiver energy')
    return dict(frame_weights=weights.repeat_interleave(100, dim=1).detach(), energy=energy.detach(),
                receiver=receiver, receiver_energy=receiver_energy.detach(),
                window_shares=shares.detach(), window_weights=weights.detach())


def weighted_loss(prediction, truth, supervision, receiver_weight):
    """Complete400 loss: normalize per example, then average examples equally."""
    require(prediction.shape == truth.shape and truth.ndim == 4 and truth.shape[1] == 400
            and prediction.dtype == truth.dtype == torch.float32, 'Complete equal FP32 prediction/target required')
    require(math.isfinite(receiver_weight) and receiver_weight >= 0, 'Nonnegative finite receiver weight required')
    error = (prediction - truth).square()
    field_each = (error * supervision['frame_weights'][:, :, None, None]).mean(dim=(1, 2, 3), dtype=torch.float32) / supervision['energy']
    rec = supervision['receiver']
    receiver_each = error[:, :, rec[:, 0], rec[:, 1]].sum(dim=(1, 2), dtype=torch.float32) / supervision['receiver_energy']
    field, receiver = field_each.mean(dtype=torch.float32), receiver_each.mean(dtype=torch.float32)
    return field + receiver_weight * receiver, field, receiver
