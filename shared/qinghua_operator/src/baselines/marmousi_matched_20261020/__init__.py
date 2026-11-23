"""Independent, parameter-matched Marmousi baselines (20 October 2026)."""

from .models import (
    TARGET_TRAINABLE_REAL_PARAMETERS,
    DeepONet,
    FNO2d,
    SpectralConv2d,
    build_model,
    parameter_report,
)

__all__ = [
    "TARGET_TRAINABLE_REAL_PARAMETERS",
    "DeepONet",
    "FNO2d",
    "SpectralConv2d",
    "build_model",
    "parameter_report",
]
