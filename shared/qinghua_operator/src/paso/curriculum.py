"""Family/frequency sampling curriculum for the V18 arms (20260928, agent_curriculum).

Scope and contract
------------------
This module reweights ONLY the seeded cell draw of Pool.decide (which of the
families x frequency-bins cells supplies the record). Everything below the cell
level -- record/time/space RAS, the loss definition (stratified_fullfuture_record_norm,
the per-record full-future relative L2), and the dev9/panel evaluation -- is untouched.
Family therefore keeps exactly its two sanctioned roles: sampling shares and metrics
grouping; it never gates the model forward (v18 specs: scattering_head=false,
phase_mode=off, no phase_gated_families, no refiner).

Determinism
-----------
Every function here is a pure function of (config, rel_draw, table state); no global
RNG, no wall clock, no file I/O. rel_draw = draw_index - sampler_draw_offset, so the
schedule is resume-safe: a restart from a checkpoint replays the same stage at the
same draw. The difficulty-aware family mode reads the RAS record table r (EMA of the
per-record unbiased relative L2), which is itself deterministic given the run's step
stream and is persisted in checkpoints -- the same reproducibility class as the
existing record-level RAS probabilities. There is NO auto-tuned loss weighting here:
the physics weights stay static config values (see physics_loss.PHYSICS_REJECTED).

Anti-suppression floor (user directive 20260928)
------------------------------------------------
The historical replay recipe (marmousi 0.8 / uniform 0.1 / layered 0.1) is exactly
the long-lived suppression pattern the V18 curriculum retires. Under a curriculum:
  * the default family shares are balanced 1/3,1/3,1/3;
  * BOTH family modes (fixed and difficulty) carry a mandatory per-family floor
    (floor > 0 is validated; difficulty shares are floor + (1-3*floor)*mix, so every
    family receives >= floor of the draw budget at every step);
  * every frequency stage keeps weights[0] >= low_replay_floor (continual low-band
    replay, 8-15 Hz) and the high band (22-25 Hz) enters monotonically.
"""
from __future__ import annotations

import numpy as np

KNOWN_FAMILIES = ('uniform', 'layered', 'marmousi')
N_BINS = 3          # 8-15 / 15-22 / 22-25 Hz (data.frequency_bins, validated in data.py)
BALANCED_SHARE = 1.0 / 3.0

FAMILY_MODES = ('fixed', 'difficulty')
FREQUENCY_MODES = ('fixed', 'staged')

FAMILY_DEFAULTS = {'mode': 'fixed', 'floor': 0.15, 'alpha': 1.0}


def balanced_shares():
    return {f: BALANCED_SHARE for f in KNOWN_FAMILIES}


def _is_balanced(shares):
    return shares is not None and set(shares) == set(KNOWN_FAMILIES) and \
        all(abs(v - BALANCED_SHARE) <= 1e-12 for v in shares.values())


def normalize_curriculum_config(raw, family_shares=None):
    """Validate sampler.curriculum. Returns the normalized dict; raises on any violation.

    family_shares: the already-validated numeric sampler.family_shares (or None); the
    curriculum constraints (per-family floor, difficulty-mode base) are checked against it.
    """
    if not isinstance(raw, dict):
        raise ValueError('sampler.curriculum must be a mapping')
    unknown = set(raw) - {'family', 'frequency'}
    if unknown:
        raise ValueError(f'unknown sampler.curriculum keys {sorted(unknown)}')

    fam_raw = dict(raw.get('family') or {})
    unknown = set(fam_raw) - set(FAMILY_DEFAULTS)
    if unknown:
        raise ValueError(f'unknown sampler.curriculum.family keys {sorted(unknown)}')
    fam = {**FAMILY_DEFAULTS, **fam_raw}
    fam['mode'] = str(fam['mode'])
    if fam['mode'] not in FAMILY_MODES:
        raise ValueError(f'curriculum.family.mode must be one of {FAMILY_MODES}, got {fam["mode"]!r}')
    fam['floor'] = float(fam['floor'])
    fam['alpha'] = float(fam['alpha'])
    # The floor is MANDATORY in every mode: 0 (or negative) would re-legalise the
    # suppression pattern the directive forbids, so it is rejected outright.
    if not 0.0 < fam['floor'] <= BALANCED_SHARE + 1e-12:
        raise ValueError('curriculum.family.floor must lie in (0, 1/3]: every family must keep '
                         'a positive share and 3*floor cannot exceed the whole budget')
    if not 0.0 < fam['alpha'] <= 1.0:
        raise ValueError('curriculum.family.alpha must lie in (0, 1]')
    if fam['mode'] == 'difficulty':
        if family_shares is not None and not _is_balanced(family_shares):
            raise ValueError('curriculum.family.mode=difficulty derives the shares from the RAS '
                             'record table (floor + (1-3*floor)*mix); sampler.family_shares must '
                             "be omitted or 'balanced' so a static share cannot silently override it")
    else:
        base = family_shares if family_shares is not None else balanced_shares()
        missing = set(KNOWN_FAMILIES) - set(base)
        if missing:
            raise ValueError(f'curriculum requires a share for every family; missing {sorted(missing)}')
        low = {f: v for f, v in base.items() if v < fam['floor'] - 1e-12}
        if low:
            raise ValueError(f'curriculum.family.floor={fam["floor"]} violated by fixed shares '
                             f'{low}; long-lived suppression of a family is prohibited -- raise '
                             'the low shares or lower the floor (floor must stay > 0)')

    freq_raw = dict(raw.get('frequency') or {})
    unknown = set(freq_raw) - {'mode', 'low_replay_floor', 'stages'}
    if unknown:
        raise ValueError(f'unknown sampler.curriculum.frequency keys {sorted(unknown)}')
    mode = str(freq_raw.get('mode', 'fixed'))
    if mode not in FREQUENCY_MODES:
        raise ValueError(f'curriculum.frequency.mode must be one of {FREQUENCY_MODES}, got {mode!r}')
    if mode == 'fixed':
        if set(freq_raw) - {'mode'}:
            raise ValueError('curriculum.frequency.mode=fixed accepts no other keys '
                             '(uniform 1/3 per bin, the historical split)')
        freq = {'mode': 'fixed'}
    else:
        stages_raw = freq_raw.get('stages')
        if not isinstance(stages_raw, list) or not stages_raw:
            raise ValueError('curriculum.frequency.mode=staged requires a nonempty stages list')
        floor = float(freq_raw.get('low_replay_floor', 0.2))
        if not 0.0 < floor <= 1.0:
            raise ValueError('curriculum.frequency.low_replay_floor must lie in (0, 1]: continual '
                             'low-band replay is mandatory in a staged curriculum')
        stages = []
        prev_until = None
        prev_high = None
        for k, st in enumerate(stages_raw):
            if not isinstance(st, dict) or set(st) != {'until_draw', 'weights'}:
                raise ValueError(f'stage {k} must be exactly {{until_draw, weights}}')
            w = st['weights']
            if not isinstance(w, (list, tuple)) or len(w) != N_BINS:
                raise ValueError(f'stage {k}: weights must list {N_BINS} bins (8-15/15-22/22-25 Hz)')
            w = [float(v) for v in w]
            if any(v <= 0 for v in w):
                raise ValueError(f'stage {k}: every bin weight must be > 0 (no bin is ever dropped)')
            if abs(sum(w) - 1.0) > 1e-9:
                raise ValueError(f'stage {k}: weights must sum to 1, got {sum(w)}')
            if w[0] < floor - 1e-12:
                raise ValueError(f'stage {k}: low-band weight {w[0]} violates low_replay_floor '
                                 f'{floor} -- the 8-15 Hz band must keep replaying in every stage')
            if prev_high is not None and w[-1] < prev_high - 1e-12:
                raise ValueError(f'stage {k}: high-band weight {w[-1]} decreases from {prev_high}; '
                                 'the 22-25 Hz band is introduced monotonically')
            prev_high = w[-1]
            until = st['until_draw']
            if k == len(stages_raw) - 1:
                if until is not None:
                    raise ValueError('the final stage must be open-ended (until_draw: null)')
            else:
                if until is None:
                    raise ValueError('only the final stage may have until_draw: null')
                until = int(until)
                if until <= 0 or (prev_until is not None and until <= prev_until):
                    raise ValueError('stage until_draw values must be positive and strictly increasing')
                prev_until = until
            stages.append({'until_draw': until, 'weights': w})
        freq = {'mode': 'staged', 'low_replay_floor': floor, 'stages': stages}
    return {'family': fam, 'frequency': freq}


def frequency_weights(freq_cfg, rel_draw):
    """(weights[N_BINS], stage_index) at rel_draw = draw_index - sampler_draw_offset.

    Piecewise constant over the stages; 'fixed' returns the uniform split with stage -1.
    """
    if freq_cfg['mode'] == 'fixed':
        return [1.0 / N_BINS] * N_BINS, -1
    rel = max(int(rel_draw), 0)
    stages = freq_cfg['stages']
    for k, st in enumerate(stages):
        if st['until_draw'] is None or rel < st['until_draw']:
            return list(st['weights']), k
    raise AssertionError('unreachable: final stage is open-ended by validation')


def family_risk(r_table, fam_of_cell, cell_roster_indices):
    """Per-family mean of the RAS record table r (EMA of the unbiased per-record loss)."""
    r = np.asarray(r_table, dtype=np.float64)
    out = {}
    for fam in dict.fromkeys(fam_of_cell):
        idx = np.concatenate([np.asarray(cell_roster_indices[j], dtype=np.int64)
                              for j, f in enumerate(fam_of_cell) if f == fam])
        out[str(fam)] = float(r[idx].mean())
    return out

def difficulty_shares(risk, floor, alpha):
    """shares = floor + (1 - n*floor) * risk^alpha / sum(risk^alpha).

    Every family receives >= floor exactly (the anti-suppression guarantee), the rest of
    the budget goes to the harder families. Pure function; ties give balanced shares.
    """
    fams = sorted(risk)
    v = np.asarray([risk[f] for f in fams], dtype=np.float64)
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError('family risk values must be finite and nonnegative')
    w = np.power(v, alpha)
    total = float(w.sum())
    mix = w / total if total > 0 else np.full(v.size, 1.0 / v.size)
    p = floor + (1.0 - v.size * floor) * mix
    return {f: float(x) for f, x in zip(fams, p)}


def cell_probabilities(sampler_cfg, fam_of_cell, bin_of_cell, rel_draw, r_table,
                       cell_roster_indices):
    """(p_cell over the pool's cells, per-draw diagnostics dict).

    p_cell[j] = share[family_j] * freq_weight[bin_j]; requires the full 3-family x
    3-bin pool (the curriculum is about generalising across all families -- a reduced
    pool must not silently renormalise).
    Diagnostics are plain-JSON scalars/lists and ride into metrics.jsonl via the
    decision dict: the biased sampling objective stays measured against the
    uniform-risk view without touching the formal loss or evaluation weights.
    """
    cur = sampler_cfg['curriculum']
    fams = list(dict.fromkeys(fam_of_cell))
    if set(fams) != set(KNOWN_FAMILIES) or len(fam_of_cell) != len(KNOWN_FAMILIES) * N_BINS:
        raise ValueError(f'curriculum needs the full {len(KNOWN_FAMILIES)}x{N_BINS} cell pool, '
                         f'got families {fams} over {len(fam_of_cell)} cells')
    risk = family_risk(r_table, fam_of_cell, cell_roster_indices)
    fam_cfg = cur['family']
    if fam_cfg['mode'] == 'difficulty':
        shares = difficulty_shares(risk, fam_cfg['floor'], fam_cfg['alpha'])
    else:
        shares = dict(sampler_cfg['family_shares'] or balanced_shares())
    fw, stage = frequency_weights(cur['frequency'], rel_draw)
    p = np.asarray([shares[f] * fw[b] for f, b in zip(fam_of_cell, bin_of_cell)],
                   dtype=np.float64)
    assert abs(float(p.sum()) - 1.0) <= 1e-9, p.sum()
    p = p / p.sum()
    cell_mean = np.asarray([np.asarray(r_table, dtype=np.float64)[np.asarray(ix, dtype=np.int64)].mean()
                            for ix in cell_roster_indices], dtype=np.float64)
    diag = {
        'ras_curriculum_stage': int(stage),
        'ras_freq_weights': [float(x) for x in fw],
        'ras_family_mode': fam_cfg['mode'],
        'ras_family_shares_effective': {f: float(shares[f]) for f in KNOWN_FAMILIES},
        'ras_family_risk': risk,
        # Uniform-risk views of the SAME residual table the biased sampler steers by:
        # *_record weights every record equally, *_cell reproduces the historical
        # 9-cell round-robin weighting; sampled_risk is the risk under the current
        # biased p_cell. All diagnostics only -- the formal relative-L2 metric and the
        # dev9 evaluation weights are defined elsewhere and unchanged.
        'ras_uniform_risk_record': float(np.asarray(r_table, dtype=np.float64).mean()),
        'ras_uniform_risk_cell': float(cell_mean.mean()),
        'ras_sampled_risk': float((p * cell_mean).sum()),
    }
    return p, diag
