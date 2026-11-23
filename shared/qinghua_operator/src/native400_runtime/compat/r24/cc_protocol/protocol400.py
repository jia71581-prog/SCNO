"""R24 protocol400: FULL = IC8 + FUTURE392 = 400 native frames, 201x201 h10, dt=0.00125 s.

Same physical window as the R21/92 protocol, 4x denser in time:
  - IC (known):      frames (88..138), times 0.11..0.1725 s, unchanged.
  - FUTURE (unknown): 392 exact native stored indices in [146, 800],
                      first unknown 146 -> 0.1825 s, endpoint 800 -> 1.0 s (NOT 4 s).
  - The original 92 forecast frames are a strict subset of FUTURE392
    (positions given by OLD92_POS_IN_FUTURE400).
  - No interpolation anywhere: every label frame is a stored native index.

Construction: between each pair of consecutive old-92 anchors (91 gaps of width
7 or 8), insert N_INSERT=300 internal native frames. Per-gap counts come from a
Bresenham largest-remainder walk proportional to gap width (every gap gets 3 or
4), and each gap's internal points sit at round(g*k/(n+1)), so the final 392
spacing is 1 or 2 native steps everywhere (mean 654/391 ~= 1.67).

Stratified sampling design (per-record, for any future trainer; no trainer here):
  - Denominator: ALWAYS the full-392 true-label energy sum of the record,
    never the energy of the selected subset.
  - 3 strata with EQUAL draw counts (DRAWS_PER_BIN each); boundaries scale
    R22's (0,25,57,92) by 392/92 -> (0,107,243,392).
  - Conditional q per stratum: positive uniform mixture
    q = UNIFORM_MIX/width + (1-UNIFORM_MIX)*s/s.sum(), so q >= 0.5/width > 0.
  - Importance coefficient mu_t = multiplicity_t / (DRAWS_PER_BIN * q_t * E_full392),
    hence E[mu_t] = 1/E_full392 and sum_t mu_t * sse_t is an unbiased estimate
    of the full-392 relative SSE of the record.

Source-field note: this protocol carries no source parameters. Downstream code
must take f0 from SOURCE_FIELDS index 2 ("source_f0_hz"), never index 0
("source_x_m"), per R21 reader92.SOURCE_FIELDS.
"""
from fractions import Fraction
import hashlib
import json

import numpy as np

DT_NATIVE_S = 0.00125
GRID_N = 201
DX_M = 10.0

IC_FRAMES = (88, 95, 102, 110, 117, 124, 131, 138)

# Verbatim copy of R21 protocol92.FORECAST_FRAMES (test re-verifies against the R21 file).
OLD_FORECAST_92 = (
    146, 153, 160, 167, 174, 181, 189, 196, 203, 210, 217, 225, 232, 239, 246,
    253, 261, 268, 275, 282, 289, 297, 304, 311, 318, 325, 333, 340, 347, 354,
    361, 368, 376, 383, 390, 397, 404, 412, 419, 426, 433, 440, 448, 455, 462,
    469, 476, 484, 491, 498, 505, 512, 520, 527, 534, 541, 548, 555, 563, 570,
    577, 584, 591, 599, 606, 613, 620, 627, 635, 642, 649, 656, 663, 671, 678,
    685, 692, 699, 707, 714, 721, 728, 735, 742, 750, 757, 764, 771, 778, 786,
    793, 800)

N_FUTURE = 392
N_INSERT = N_FUTURE - len(OLD_FORECAST_92)  # 300 internal points over 91 gaps


def _round_half_up(x: Fraction) -> int:
    if x < 0:
        raise ValueError("nonnegative only")
    return int(x + Fraction(1, 2))


def _build_future392():
    anchors = OLD_FORECAST_92
    gaps = tuple(anchors[i + 1] - anchors[i] for i in range(len(anchors) - 1))
    span = sum(gaps)  # 654 native steps
    frames = []
    cum = Fraction(0)
    alloc = 0
    for i, g in enumerate(gaps):
        frames.append(anchors[i])
        cum += Fraction(g * N_INSERT, span)
        n = _round_half_up(cum) - alloc  # Bresenham: per-gap insert count, 3 or 4
        alloc += n
        if not 1 <= n <= g - 1:
            raise AssertionError("insert count must fit strictly inside the gap")
        for k in range(1, n + 1):
            frames.append(anchors[i] + _round_half_up(Fraction(g * k, n + 1)))
    frames.append(anchors[-1])
    if alloc != N_INSERT or len(frames) != N_FUTURE:
        raise AssertionError("future392 construction")
    return tuple(frames)


FORECAST_FRAMES_392 = _build_future392()
FULL_FRAMES_400 = IC_FRAMES + FORECAST_FRAMES_392
OLD92_POS_IN_FUTURE400 = tuple(FORECAST_FRAMES_392.index(f) for f in OLD_FORECAST_92)
IC_OFFSET_IN_FULL400 = len(IC_FRAMES)  # pos in FULL400 = IC_OFFSET_IN_FULL400 + pos in FUTURE392

FIRST_UNKNOWN_TIME_S = OLD_FORECAST_92[0] * DT_NATIVE_S   # 0.1825 s
FORECAST_END_TIME_S = OLD_FORECAST_92[-1] * DT_NATIVE_S   # 1.0 s


def frame_time_s(frame):
    return frame * DT_NATIVE_S


# ---- stratified time sampling over the 392 future frames --------------------
BINS_392 = ((0, 107), (107, 243), (243, 392))  # R22 (0,25,57,92) scaled by 392/92
DRAWS_PER_BIN = 48                             # equal count per stratum; 4x R22's 12
UNIFORM_MIX = 0.5                              # q >= UNIFORM_MIX/width > 0 everywhere


def stratified_plan(energy392, score=None, count=None, *, step, sid):
    """Unbiased lagged stratified plan over the 392 future frames.

    energy392: full-392 TRUE-label per-frame energy of this record (the
    denominator is its sum; the selected subset never redefines it).
    Returns indices, mu coefficients, conditional q, seed, denominator.
    """
    energy = np.asarray(energy392, np.float64)
    if (energy.shape != (N_FUTURE,) or not np.isfinite(energy).all()
            or (energy < 0).any() or energy.sum() <= 0):
        raise ValueError("full392 true-label energy required")
    denom = float(energy.sum())
    score = np.zeros(N_FUTURE) if score is None else np.asarray(score, np.float64)
    count = np.zeros(N_FUTURE, np.int64) if count is None else np.asarray(count, np.int64)
    if (score.shape != (N_FUTURE,) or count.shape != (N_FUTURE,)
            or not np.isfinite(score).all() or (score < 0).any() or (count < 0).any()):
        raise ValueError("sampler state")
    seed = int.from_bytes(hashlib.sha256(f"R24|{step}|{sid}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    coeff = np.zeros(N_FUTURE, np.float64)
    prob = np.zeros(N_FUTURE, np.float64)
    for lo, hi in BINS_392:
        width = hi - lo
        seen = count[lo:hi] > 0
        e = energy[lo:hi] / denom
        s = np.where(seen, score[lo:hi], e)
        q = (np.full(width, 1.0 / width) if s.sum() <= 0
             else UNIFORM_MIX / width + (1.0 - UNIFORM_MIX) * s / s.sum())
        draws = rng.choice(width, size=DRAWS_PER_BIN, replace=True, p=q)
        multiplicity = np.bincount(draws, minlength=width)
        coeff[lo:hi] = multiplicity / (DRAWS_PER_BIN * q * denom)  # E[mu_t] = 1/denom
        prob[lo:hi] = q
    selected = np.flatnonzero(coeff)
    return dict(indices=selected.tolist(), coefficients=coeff[selected].tolist(),
                conditional_q=prob.tolist(), seed=seed, draws=3 * DRAWS_PER_BIN,
                denominator=denom, full392_denominator=True)


# ---- protocol identity -------------------------------------------------------
def protocol_hash():
    identity = dict(
        protocol="R24_time400",
        dt_native_s="0.00125",
        grid=[GRID_N, GRID_N], dx_m=DX_M,
        IC_frames=list(IC_FRAMES),
        forecast_frames_392=list(FORECAST_FRAMES_392),
        old_forecast_92=list(OLD_FORECAST_92),
        old92_pos_in_future=list(OLD92_POS_IN_FUTURE400),
        first_unknown_time_s="0.1825", forecast_end_time_s="1.0",
        exact_native=True, no_interpolation=True,
        sampling=dict(bins=[list(b) for b in BINS_392],
                      draws_per_bin=DRAWS_PER_BIN, uniform_mix=UNIFORM_MIX,
                      denominator="full392_true_energy_per_record"),
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


PROTOCOL400_SHA256 = protocol_hash()
