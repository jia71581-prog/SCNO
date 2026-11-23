"""A direct physical-source reader; keep the original tensor and normalization contracts."""
from pathlib import Path
import hashlib
import sys
import threading

import h5py
import numpy as np

BASE_CODE = Path(__file__).resolve().parents[2] / 'code'
sys.path.insert(0, str(BASE_CODE))
import trainer as base


class DirectCorpus(base.Corpus):
    def __init__(self, case, cache_records):
        super().__init__(case, cache_records)
        self.cache_lock = threading.Lock()

    def record(self, row):
        source_path = (base.DATA / self.source_files[row]).resolve()
        stat = source_path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        with self.cache_lock:
            cached = self.cache.get(row)
            if cached is not None:
                assert identity == self.source_identities[row], 'A cached source changed'
                return cached
        with h5py.File(source_path, 'r') as f:
            pa = f['wavefield'][...]
            velocity = f['velocity201'][...]
        if pa.shape[0] == 801:
            pa = pa[::2].copy()
        assert pa.shape == (401, 201, 201) and pa.dtype == np.float32
        assert np.isfinite(pa).all() and not pa[0].any() and not pa[:, 0].any()
        assert hashlib.sha256(memoryview(pa).cast('B')).hexdigest() == self.expected_payload_hashes[row]
        energy = float(np.square(pa, dtype=np.float64).sum())
        y = base.encode_pressure(pa, base.PROFILE)
        del pa
        source = self.sources[row]
        c = (velocity - np.float32(3728.63427734375)) / np.float32(2141.111083984375)
        known = y[base.KNOWN]
        if self.case == 'ul':
            sx, sz = self.source_solver[row, :2] / 10.
            ix, iz = int(np.floor(sx)), int(np.floor(sz))
            ax, az = sx - ix, sz - iz
            smap = np.zeros((201, 201), np.float32)
            for dz, wz in [(0, 1 - az), (1, az)]:
                for dx, wx in [(0, 1 - ax), (1, ax)]:
                    if wz * wx:
                        smap[iz + dz, ix + dx] += wz * wx
            assert abs(smap.sum(dtype=np.float64) - 1) < 1e-7
            maps = np.stack((c, *known, smap, 2 * self.x / 2000 - 1, 2 * self.z / 2000 - 1,
                             (self.x - source[0]) / 2000, (self.z - source[1]) / 2000))
        else:
            scaled = source / np.array([2000., 2000., 50., 1.2, 1.], np.float32)
            constants = np.broadcast_to(scaled[:, None, None], (5, 201, 201))
            maps = np.stack((c, *constants, 2 * self.x / 2000 - 1, 2 * self.z / 2000 - 1, *known))
        maps = np.ascontiguousarray(maps, dtype=np.float32)
        y.flags.writeable = False
        maps.flags.writeable = False
        result = maps, y
        with self.cache_lock:
            self.source_identities[row] = identity
            self.energy[row] = energy
            if len(self.cache) < self.cache_records:
                self.cache[row] = result
        return result
