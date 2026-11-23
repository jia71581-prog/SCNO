"""Helpers used by the public training entry points."""
from contextlib import ExitStack, contextmanager

@contextmanager
def forbid_rollouts():
    """Fail on the established Marmousi numerical forward and adjoint entry points."""
    from unittest.mock import patch
    from coarse_propagation import CoarsePropagation
    from lwc_reference import LWCReference
    from src.marmousi_ch6_lwc84_20261024 import cuda_generator_fp32, trajectory_fp32
    from tools.marmousi_target_dataset_refined.lwc_rectangular import RectangularLWC

    def reject(*_, **__):
        raise RuntimeError("Physical propagation and proposals are disabled in the native neural pressure pilot")

    blocked = ((LWCReference, "__init__"), (LWCReference, "predict"), (CoarsePropagation, "__init__"),
               (CoarsePropagation, "predict_snapshots"), (RectangularLWC, "__init__"), (RectangularLWC, "shot_batch"),
               (cuda_generator_fp32.CUDAGenerator, "__init__"), (trajectory_fp32, "differentiable_trajectory"))
    with ExitStack() as guard:
        for owner, name in blocked:
            guard.enter_context(patch.object(owner, name, reject))
        yield
