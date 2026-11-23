"""CPU-authoritative physical time axis for the original paper201 operator.

The numerical teacher formed its 801 physical instants in NumPy float64, then
stored their float32 values. Multiplication in float32 (on CPU or CUDA) yields
different values at some instants. Construct this small authority on CPU and
copy its already-rounded bits to the requested device; never rebuild on CUDA.

Each call owns independent storage. The operator registers this axis as a
nonpersistent buffer, so repairing its construction adds no checkpoint state
and changes no learned weights or registered native frame indices.
"""
from __future__ import annotations

import numpy as np
import torch


SCHEMA = "PAPER201_TEACHER_FP64_TO_FP32_CPU_TIME_AUTHORITY_V1"
NATIVE_FRAME_COUNT = 801
NATIVE_DT_SECONDS = .00125


def native_time_axis(device="cpu"):
    """Return independent FP32 ``[801]`` teacher times on ``device``.

    The arithmetic order exactly matches the original teacher codec:
    ``np.asarray(np.arange(801, dtype=np.float64) * .00125, np.float32)``.
    Device transfer happens only after the final FP32 rounding on CPU.
    """
    values = np.asarray(np.arange(NATIVE_FRAME_COUNT, dtype=np.float64) * NATIVE_DT_SECONDS,
                        dtype=np.float32)
    return torch.from_numpy(values).to(device=device)
