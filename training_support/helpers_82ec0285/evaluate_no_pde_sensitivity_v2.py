"""Helpers used by the public training entry points."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from train_sensitivity import array_sha, require

FORBIDDEN = {
    "lwc_reference.py": {"__init__", "predict", "predict_replay", "_replay"},
    "lwc_rectangular.py": {"__init__", "set_velocity", "shot_batch", "_block"},
    "coarse_propagation.py": {"__init__", "predict", "predict_snapshots", "shot_batch_with_snapshots"},
    "cuda_generator_fp32.py": {"__init__", "_launch", "_launch_flat", "apply_A", "step", "vjp_A"},
    "trajectory_fp32.py": {"differentiable_trajectory", "forward", "backward"},
    "score_fullfield_e12.py": {"replay"},
    "native_pressure_readout.py": {"load_pressure_head", "compose_pressure", "make_pressure_context"},
}

GUARDED_FUNCTIONS = set().union(*FORBIDDEN.values(), {"make_proposal", "load_kernel_corrected_readout", "load_kernel_residual"})

class ForbiddenEvaluation(RuntimeError):
    """Access to a true target asset or numerical propagation path."""

@contextmanager
def native_only():
    """Guard native audit calls, including future accidental imports or fallback paths."""
    before = sys.getprofile()

    def guard(frame, event, _):
        if event != "call":
            return
        name = frame.f_code.co_name
        if name not in GUARDED_FUNCTIONS:
            return
        source = frame.f_code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
        if name in FORBIDDEN.get(source, ()) or (source == "shared_framework.py" and name == "make_proposal") \
                or (source == "shared_framework.py" and name == "__init__"
                    and frame.f_locals.get("self").__class__.__name__ == "SharedFrameworkReadout") \
                or (source == "kernel_corrected_readout.py" and name == "load_kernel_corrected_readout") \
                or (source == "kernel_receiver.py" and name in ("forward", "load_kernel_residual")):
            raise ForbiddenEvaluation(f"No-PDE audit attempted forbidden {source}:{name}")

    sys.setprofile(guard)
    try:
        yield
    finally:
        sys.setprofile(before)

def forbid_exact_label_files(dataset):
    """Permit the registered template's upstream byte-hash check and no target samples."""
    from tools.marmousi_target_dataset_refined import common

    root = dataset.resolve()
    header = common.read_json(root / "acquisition_contract.json")
    contract_sha = header["contract_hash"]
    require(common.digest({key: value for key, value in header.items() if key != "contract_hash"}) == contract_sha,
            "Cannot authorize a template integrity read from an invalid contract header")
    template = header["templates"]
    declared_path = Path(template["truth_file"])
    expected_name = "existing_marmousi_true201"
    require(declared_path.is_absolute() and expected_name in declared_path.name.lower(),
            "Only the original immutable Marmousi velocity template may be hashed")
    truth = declared_path.resolve(strict=True)
    contract_code, hash_code = common.load_contract.__code__, common.sha_file.__code__
    readonly_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    receipt = {"template_path": str(truth), "registered_file_sha256": template["truth_file_sha256"],
               "byte_hash_reads": 0, "decoded_as_model_or_tensor": False, "exact_target_waveforms_allowed": False,
               "authorized_call": "tools.marmousi_target_dataset_refined.common.load_contract -> sha_file"}

    def guard(event, parameters):
        if event != "open" or not isinstance(parameters[0], (str, bytes)):
            return
        name, mode, flags = parameters
        path = os.fsdecode(name)
        lower = path.replace("\\", "/").lower()
        if "test_e_exact_marmousi" in lower:
            raise ForbiddenEvaluation(f"Exact target sample forbidden: {path}")
        if expected_name not in lower and os.path.basename(lower) != truth.name.lower():
            return

        # The original verifier streams this NPY solely into hashlib. Permit its
        # exact callable, its immediate contractual caller and its bound path.
        frame = sys._getframe(1)
        while frame is not None and frame.f_code is not hash_code:
            frame = frame.f_back
        caller = frame.f_back if frame is not None else None
        if (caller is not None and caller.f_code is contract_code and caller.f_locals.get("key") == "truth"
                and caller.f_locals["d"].resolve() == root and caller.f_locals["c"]["contract_hash"] == contract_sha
                and Path(path).resolve() == truth and mode in ("r", "rb", None) and not flags & readonly_flags):
            receipt["byte_hash_reads"] += 1
            return
        raise ForbiddenEvaluation(f"Exact target template open outside registered read-only integrity check: {path}")

    sys.addaudithook(guard)
    return receipt
