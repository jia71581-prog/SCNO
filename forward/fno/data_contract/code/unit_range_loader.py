"""Load physical Pa and one TRAIN-fitted affine [-1, 1] pressure encoding."""
import json
from pathlib import Path

import numpy as np

from dataset401 import load_record as _load_physical_record


def encode_pressure(pressure_pa, profile):
    pressure_pa = np.asarray(pressure_pa)
    if profile['method'] == 'TRAIN_global_affine_minmax':
        center = float(profile['pressure_center_pa'])
        scale = float(profile['pressure_half_range_pa'])
        if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
            raise ValueError('Finite TRAIN center and positive half-range required')
        return ((pressure_pa.astype(np.float64) - center) / scale).astype(np.float32)
    if profile['method'] == 'TRAIN_global_max_abs_zero_preserving':
        return pressure_pa / np.float32(profile['pressure_scale_pa'])
    raise ValueError('Unsupported normalization method')


def decode_pressure(normalized, profile):
    normalized = np.asarray(normalized)
    if profile['method'] == 'TRAIN_global_affine_minmax':
        return (normalized.astype(np.float64) * float(profile['pressure_half_range_pa'])
                + float(profile['pressure_center_pa'])).astype(np.float32)
    if profile['method'] == 'TRAIN_global_max_abs_zero_preserving':
        return normalized * np.float32(profile['pressure_scale_pa'])
    raise ValueError('Unsupported normalization method')


def load_record(dataset_path, index, normalization_path=None):
    """Raw Pa stays unchanged; heldout records reuse TRAIN extrema without clipping.

    Physical zero has a nonzero encoded value. Decode to Pa before applying
    pressure-release/initial constraints, physical residuals or physical metrics.
    """
    dataset_path = Path(dataset_path)
    if normalization_path is None:
        normalization_path = dataset_path.parent / 'normalization_unit_range.json'
    profile = json.loads(Path(normalization_path).read_text())
    record = _load_physical_record(dataset_path, index)
    record['pressure_normalized'] = encode_pressure(record['pressure_pa'], profile)
    record['pressure_scale_pa'] = float(profile['pressure_scale_pa'])
    record['normalization_method'] = profile['method']
    record['normalization_profile'] = str(Path(normalization_path).resolve())
    return record
