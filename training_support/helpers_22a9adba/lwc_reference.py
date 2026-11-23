"""Helpers used by the public training entry points."""
from __future__ import annotations
import json
import math
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from src.marmousi_ch6_lwc84_20261024 import cuda_generator_fp32, protocol
from tools.marmousi_target_dataset_refined import common, lwc_rectangular

CONTRACT_HASH = "a8902f32c1ffe9a568f2d9370c78ccc761ffe948202658b9e0b2735c3ab5b047"

def _verify_file(path: Path, expected: str) -> None:
    if common.sha_file(path) != expected:
        raise ValueError(f"Frozen LWC input hash mismatch for {path}")

class LWCReference:
    """Predict physical-pressure B5 gathers as [shot, 801 times, 201 receivers].

    Accepts one native [201, 201] velocity in m/s per call. Caller controls
    shot batching and consumes each differentiable call with one backward.
    """

    def __init__(self, dataset, device="cuda:0", build_dir=None, major_steps=512, minor_steps=16,
                 progress_path=None):
        self.dataset = Path(dataset).resolve()
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The forward reference requires a CUDA device")
        if not (isinstance(minor_steps, Integral) and isinstance(major_steps, Integral)
                and 1 <= minor_steps <= major_steps):
            raise ValueError("Require positive checkpoint lengths with minor_steps <= major_steps")
        self.major_steps, self.minor_steps = major_steps, minor_steps

        contract = json.loads((self.dataset / "acquisition_contract.json").read_text(encoding="utf-8"))
        if (contract["contract_hash"] != CONTRACT_HASH
                or common.digest({k: v for k, v in contract.items() if k != "contract_hash"}) != CONTRACT_HASH):
            raise ValueError("Expected the released 38-shot, 801-frame sparse Marmousi contract")
        grid, solver, acquisition = (contract[key] for key in ("grid", "solver", "acquisition"))
        for module, expected in (
            (cuda_generator_fp32, solver["generator_file_sha256"]),
            (protocol, solver["protocol_file_sha256"]),
            (lwc_rectangular, solver["adapter_code_sha256"]),
        ):
            _verify_file(Path(module.__file__), expected)

        self.native_shape = tuple(grid["shape_zx"])
        refinement = int(solver["spatial_refinement"])
        self.dx = grid["spacing_xz_m"][0] / refinement
        self.n = (self.native_shape[0] - 1) * refinement + 1
        self.pad = round(solver["physical_absorbing_width_m"] / self.dx)
        self.dt = solver["actual_internal_dt_s"]
        self.steps = solver["actual_simulation_step_count"]
        self.stride = round(acquisition["observation_dt_s"] / self.dt)
        self.nt = acquisition["record_samples"]
        self.nr = acquisition["receivers"]
        self.velocity_bounds = tuple(grid["velocity_bounds_m_s"])
        if (self.native_shape != (201, 201) or (self.n, self.pad, self.dx) != (401, 40, 5.0)
                or self.steps != self.stride * (self.nt - 1)
                or not math.isclose(self.dt * self.stride, acquisition["observation_dt_s"], abs_tol=1e-12)):
            raise ValueError("Unexpected solver geometry or time sampling")

        arrays, paths = {}, {}
        for name in ("source_parameters", "receiver_coordinates", "observation_times"):
            spec = contract["shared_arrays"][name]
            path = paths[name] = self.dataset / spec["path"]
            _verify_file(path, spec["file_sha256"])
            arrays[name] = np.load(path, allow_pickle=False)
        self.sources = arrays["source_parameters"]
        if self.sources.shape != (38, 5) or arrays["receiver_coordinates"].shape != (self.nr, 2):
            raise ValueError("Unexpected frozen source or receiver shape")
        np.testing.assert_allclose(arrays["observation_times"], np.arange(self.nt) * self.stride * self.dt,
                                   atol=1e-12, rtol=0)

        self._scratch = TemporaryDirectory(prefix="marmousi-lwc-build-") if build_dir is None else None
        self.cache = Path(self._scratch.name if self._scratch else build_dir).resolve()
        self.progress_path = None if progress_path is None else Path(progress_path).resolve()
        if any(path is not None and path.is_relative_to(self.dataset) for path in (self.cache, self.progress_path)):
            raise ValueError("LWC artifacts must be outside the released dataset")
        self.generator = cuda_generator_fp32.CUDAGenerator(self.cache / "build", str(self.device))
        self.device = self.generator.device
        # RectangularLWC uses dataset/build for its compilation cache. Absolute
        # shared-array paths preserve the released inputs while relocating the cache.
        self.forward_contract = {
            **contract,
            "shared_arrays": {
                name: {**contract["shared_arrays"][name], "path": str(path)}
                for name, path in paths.items()
            },
        }
        self._forward_engine = None

        indices, weights, _ = lwc_rectangular.receiver_support(
            (self.n, self.n), self.dx, self.pad, arrays["receiver_coordinates"][:, 0]
        )
        self.indices = torch.as_tensor(indices, device=self.device)
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        coefficients = lwc_rectangular.profiles(
            (self.n, self.n), self.dx, self.pad, solver["profile_velocity_m_s"],
            solver["pml_frequency_hz"], solver["pml_power"], solver["pml_reflection_target"]
        )
        self.profiles = torch.as_tensor(coefficients, device=self.device)

    def memory_estimate(self, shots, grad=False):
        """Conservative array-only estimate, excluding CUDA context and system overhead."""
        state = 7 * shots * (self.n + self.pad) * (self.n + 2 * self.pad) * 4
        major_checkpoints = math.ceil(self.steps / self.major_steps)
        minor_boundaries = math.ceil(self.major_steps / self.minor_steps)
        return {
            "host_bytes": (major_checkpoints + minor_boundaries) * state if grad else 0,
            "gpu_bytes": (self.minor_steps + 15) * state if grad else 10 * state,
        }

    def predict(self, velocity, shot_ids, grad=False):
        """Use the fast production forward or the exact differentiable replay."""
        v, ids = self._validate_input(velocity, shot_ids, grad)
        if grad:
            raise ValueError("The training reference returns fixed forward labels")
        if self._forward_engine is None or self._forward_engine.batch_size != len(ids):
            self._forward_engine = None
            self._forward_engine = lwc_rectangular.RectangularLWC(
                self.forward_contract, self.cache, device=str(self.device), batch_size=len(ids)
            )
        self._forward_engine.set_velocity(v.detach().cpu().numpy())
        records, _ = self._forward_engine.shot_batch(ids)
        if records.shape != (len(ids), self.nt, self.nr):
            raise RuntimeError("Production reference returned an unexpected receiver layout")
        return torch.as_tensor(records, device=self.device)


    def _validate_input(self, velocity, shot_ids, grad):
        ids = list(shot_ids)
        if not ids or any(isinstance(i, bool) or not isinstance(i, Integral) for i in ids):
            raise ValueError("Expected nonempty integer source IDs")
        if len(set(ids)) != len(ids) or not all(0 <= i < len(self.sources) for i in ids):
            raise ValueError("Expected unique source IDs in the frozen roster")
        v = torch.as_tensor(velocity, device=self.device, dtype=torch.float64)
        lo, hi = self.velocity_bounds
        if v.shape != self.native_shape or not bool(torch.isfinite(v).all() & (v >= lo).all() & (v <= hi).all()):
            raise ValueError(f"Expected finite native {self.native_shape} velocity within [{lo}, {hi}] m/s")
        if grad and not v.requires_grad:
            raise ValueError("Pass a differentiable velocity tensor when grad=True")

        budget = self.memory_estimate(len(ids), grad)
        free, _ = torch.cuda.mem_get_info(self.device)
        reusable = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        if budget["gpu_bytes"] > 0.92 * (free + reusable):
            raise MemoryError(f"Reference shot batch {len(ids)} exceeds estimated GPU capacity; decrease the batch")
        return v, ids
