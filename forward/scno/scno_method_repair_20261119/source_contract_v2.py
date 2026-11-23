"""Read-only acquisition checks before a newly bound training run can start."""

import hashlib
import json
import math
import numpy as np


def assert_source_contract(corpus, specification):
    """Validate physical source metadata; never relocate sources or relabel fields.

    This opt-in assertion is for new FWI training configurations. Existing
    checkpoints retain their original data/normalization binding. Metadata
    validation does not read observations or full pressure labels.
    """
    spec = dict(specification)
    supported = {'z_m', 'frequency_hz', 'delay_s', 'amplitude', 'x_range_m',
                 'source_x_m', 'frames', 'start_s', 'end_s', 'dt_s',
                 'coordinate_tolerance_m', 'parameter_tolerance', 'purpose'}
    if set(spec) - supported:
        raise ValueError(f"unknown source-contract fields: {sorted(set(spec) - supported)}")
    coordinate_tol = float(spec.get('coordinate_tolerance_m', 1e-3))
    parameter_tol = float(spec.get('parameter_tolerance', 1e-6))
    if not all(math.isfinite(x) and x >= 0 for x in (coordinate_tol, parameter_tol)):
        raise ValueError('acquisition tolerances must be finite and nonnegative')
    checks = [('z_m', 1, coordinate_tol), ('frequency_hz', 2, parameter_tol),
              ('delay_s', 3, parameter_tol), ('amplitude', 4, parameter_tol)]
    sources = []
    for role in ('train', 'validation', 'test'):
        for key in getattr(corpus, role + '_ids'):
            source = np.asarray(corpus._bound[key].source, dtype=np.float64)
            if source.shape != (5,) or not np.isfinite(source).all():
                raise ValueError(f'invalid source metadata in {role} record {key}')
            for name, index, tolerance in checks:
                if name in spec and abs(source[index] - float(spec[name])) > tolerance:
                    raise ValueError(f'{role} record {key}: source {name}={source[index]:.9g}; '
                                     f'required {spec[name]}. Use matching physical labels.')
            if 'x_range_m' in spec:
                lo, hi = map(float, spec['x_range_m'])
                if not lo - coordinate_tol <= source[0] <= hi + coordinate_tol:
                    raise ValueError(f'{role} record {key}: source x is outside the required acquisition range')
            if 'source_x_m' in spec:
                expected_x = np.asarray(spec['source_x_m'], dtype=np.float64)
                if expected_x.ndim != 1 or not len(expected_x) or not np.isfinite(expected_x).all():
                    raise ValueError('source_x_m must list finite physical coordinates')
                if np.min(np.abs(expected_x - source[0])) > coordinate_tol:
                    raise ValueError(f'{role} record {key}: source x is absent from the required acquisition array')
            sources.append({'role': role, 'id': str(key), 'source': source.tolist()})
    axis = np.asarray(corpus.times, dtype=np.float64)
    if not len(sources) or not len(axis):
        raise ValueError('acquisition checks require selected records and a native time axis')
    if 'frames' in spec and len(axis) != int(spec['frames']):
        raise ValueError(f'native pressure has {len(axis)} frames; required {spec["frames"]}')
    for key, actual in [('start_s', axis[0]), ('end_s', axis[-1])]:
        if key in spec and abs(actual - float(spec[key])) > 1e-8:
            raise ValueError(f'native pressure {key} differs from the required time contract')
    if 'dt_s' in spec and not np.allclose(np.diff(axis), float(spec['dt_s']), rtol=0, atol=1e-8):
        raise ValueError('native pressure time spacing differs from the required contract')
    content = json.dumps(sources, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    receipt = {'schema': 'SCNO_SOURCE_ACQUISITION_ASSERTION_V1', 'specification': spec,
               'selected_records': len(sources), 'source_metadata_sha256': hashlib.sha256(content).hexdigest(),
               'frames': len(axis), 'pressure_labels_read_by_assertion': False}
    corpus.binding = {**corpus.binding, 'source_acquisition_assertion': receipt}
    return corpus
