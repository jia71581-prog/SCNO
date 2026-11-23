"""All-801 supervision policy over 0-1 s with adaptive frame weights."""
import numpy as np

DT = 0.00125
N = 801


def adaptive_frame_weights(ema, endpoint=100):
    ema = np.asarray(ema, dtype=np.float64)
    if ema.ndim != 1 or ema.size < 2 * endpoint + 1:
        raise ValueError('Expected a 1-D EMA with at least two endpoint blocks')
    if not np.isfinite(ema).all() or (ema < 0).any():
        raise ValueError('Expected finite nonnegative frame MSE values')
    prior = np.ones(ema.size, dtype=np.float64)
    prior[:endpoint] = 2.0
    prior[-endpoint:] = 2.0
    q90 = max(float(np.quantile(ema, 0.9)), 1e-30)
    return prior * (1.0 + 3.0 * np.clip(ema / q90, 0.0, 1.0))


def pooled_percent(mse, reference_energy):
    mse = np.asarray(mse, dtype=np.float64)
    reference_energy = np.asarray(reference_energy, dtype=np.float64)
    if mse.shape != reference_energy.shape:
        raise ValueError('mse and reference_energy must have matching shapes')
    if not np.isfinite(mse).all() or not np.isfinite(reference_energy).all():
        raise ValueError('Expected finite mse and reference_energy values')
    return 100.0 * float(np.sqrt(mse.sum() / max(reference_energy.sum(), 1e-300)))


def window_errors_801(mse, reference_energy):
    mse = np.asarray(mse, dtype=np.float64)
    reference_energy = np.asarray(reference_energy, dtype=np.float64)
    n = mse.size
    q = n // 3
    windows = [('early', slice(0, q)), ('middle', slice(q, 2 * q)), ('late', slice(2 * q, n))]
    return {name: pooled_percent(mse[s], reference_energy[s]) for name, s in windows}


def window_errors_400compat(mse, reference_energy, idx400):
    mse = np.asarray(mse, dtype=np.float64)[idx400]
    reference_energy = np.asarray(reference_energy, dtype=np.float64)[idx400]
    windows = [('early', slice(0, 100)), ('middle', slice(100, 300)), ('late', slice(300, 400))]
    return {name: pooled_percent(mse[s], reference_energy[s]) for name, s in windows}
