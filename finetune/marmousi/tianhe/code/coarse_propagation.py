"""Frozen-geometry, velocity-only 10 m LWC8-4 coarse propagation."""

from numbers import Integral

import numpy as np
import torch

from lwc_reference import LWCReference
from tools.marmousi_target_dataset_refined import lwc_rectangular


class _SnapshotEngine(lwc_rectangular.RectangularLWC):
    """Use the production CUDA graph and retain selected in-domain pressure frames."""

    _snapshot_slots = None

    def _sample(self):
        if self._snapshot_slots is not None:
            slot = self._snapshot_slots.get(self._frame)
            if slot is not None:
                nz, nx = self.shape
                pressure = self.state[0, :, :nz, self.pad:self.pad + nx]
                self._native_pressure[:, slot].copy_(pressure)
            self._frame += 1
        return super()._sample()

    def shot_batch_with_snapshots(self, shot_ids, frame_indices):
        self._native_pressure = self.state.new_empty((self.batch_size, len(frame_indices), *self.shape))
        self._snapshot_slots = {sample: slot for slot, sample in enumerate(frame_indices)}
        self._frame = 0
        try:
            b5, _ = self.shot_batch(shot_ids)
            pressure = self._native_pressure
            if self._frame != self.nt or not bool(torch.isfinite(pressure).all()):
                raise RuntimeError("Incomplete or nonfinite coarse pressure snapshots")
        finally:
            self._snapshot_slots = None
            del self._native_pressure
        return pressure, b5


class CoarsePropagation(LWCReference):
    """Independent 10 m propagation using the frozen Marmousi acquisition.

    Inputs are native [z,x] velocity in m/s and frozen source IDs. Outputs
    are raw numerical pressure, with no training or observation values read.
    The 5 m physical teacher remains separate from this 10 m proposal.
    """

    def __init__(self, dataset, device="cuda:0", build_dir=None, major_steps=512, minor_steps=16):
        super().__init__(dataset, device=device, build_dir=build_dir, major_steps=major_steps,
                         minor_steps=minor_steps)
        fine = self.forward_contract["solver"]
        self.n = self.native_shape[0]
        self.dx = self.forward_contract["grid"]["spacing_xz_m"][0]
        self.dt = 2 * fine["actual_internal_dt_s"]
        self.stride = self.stride // 2
        self.steps = (self.nt - 1) * self.stride
        self.pad = round(fine["physical_absorbing_width_m"] / self.dx)
        if (self.n, self.dx, self.dt, self.stride, self.steps, self.pad) != (201, 10., .000625, 8, 6400, 20):
            raise ValueError("Unexpected coarse Marmousi step, grid or boundary geometry")
        self.forward_contract["solver"] = {
            **fine, "spatial_refinement": 1, "actual_internal_dt_s": self.dt,
            "actual_simulation_step_count": self.steps,
        }
        receiver_path = self.forward_contract["shared_arrays"]["receiver_coordinates"]["path"]
        centers_x = np.load(receiver_path, allow_pickle=False)[:, 0]
        index, weight, _ = lwc_rectangular.receiver_support(self.native_shape, self.dx, self.pad, centers_x)
        self.indices = torch.as_tensor(index, device=self.device)
        self.weights = torch.as_tensor(weight, device=self.device, dtype=torch.float32)
        coefficients = lwc_rectangular.profiles(
            self.native_shape, self.dx, self.pad, fine["profile_velocity_m_s"],
            fine["pml_frequency_hz"], fine["pml_power"], fine["pml_reflection_target"],
        )
        self.profiles = torch.as_tensor(coefficients, device=self.device)

    def predict_snapshots(self, velocity, shot_ids, frame_indices, *, include_b5=False):
        """Return selected coarse pressure [B,K,z,x] and optional B5 [B,801,201].

        Snapshots are forward-only on the native grid and exclude the CPML
        exterior. The t=0 frame is allowed. For velocity gradients of B5,
        call predict(velocity, shot_ids, grad=True).
        """
        frames = list(frame_indices)
        if (not frames or any(isinstance(i, bool) or not isinstance(i, Integral) for i in frames)
                or frames != sorted(set(frames)) or not 0 <= frames[0] <= frames[-1] < self.nt):
            raise ValueError("Frame indices must be unique increasing integers in the 801-frame acquisition")
        v, ids = self._validate_input(velocity, shot_ids, grad=False)
        if not isinstance(self._forward_engine, _SnapshotEngine) or self._forward_engine.batch_size != len(ids):
            self._forward_engine = None
            self._forward_engine = _SnapshotEngine(self.forward_contract, self.cache,
                                                    device=str(self.device), batch_size=len(ids))
        samples_bytes = len(ids) * len(frames) * self.n**2 * self._forward_engine.state.element_size()
        free, total = torch.cuda.mem_get_info(self.device)
        if free - samples_bytes < total * .08:
            raise MemoryError("Reduce the coarse shot batch or number of retained pressure frames")
        self._forward_engine.set_velocity(v.detach().cpu().numpy())
        pressure, b5 = self._forward_engine.shot_batch_with_snapshots(ids, frames)
        if include_b5:
            return pressure, torch.as_tensor(b5, device=self.device)
        return pressure
