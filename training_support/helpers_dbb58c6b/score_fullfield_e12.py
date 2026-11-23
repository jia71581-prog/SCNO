"""Helpers used by the public training entry points."""
from contextlib import contextmanager
import torch
from lwc_reference import LWCReference
from tools.marmousi_target_dataset_refined import lwc_rectangular

SHAPE = (201, 201)

def require(ok, description):
    if not ok:
        raise ValueError(description)

@contextmanager
def run_lock(directory):
    import fcntl

    with (directory / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

class NativeFinePressure(lwc_rectangular.RectangularLWC):
    """Collect p on collocated 10 m nodes inside the original 5 m production run."""

    frame_slots = None

    def _sample(self):
        if self.frame_slots is not None:
            slot = self.frame_slots.get(self.frame)
            if slot is not None:
                nz, nx = self.shape
                self.pressure[:, slot].copy_(self.state[0, :, :nz:2, self.pad:self.pad + nx:2])
            self.frame += 1
        return super()._sample()

    def replay(self, velocity, shot_ids, frames):
        require(self.shape == (401, 401) and self.dx == 5. and self.dt == .0003125
                and self.stride == 16 and self.pad == 40 and self.nt == 801,
                "Fine production solver geometry differs from the registered 5 m LWC")
        self.set_velocity(velocity)
        self.pressure = self.state.new_empty((self.batch_size, len(frames), *SHAPE))
        self.frame_slots = {frame: slot for slot, frame in enumerate(frames)}
        self.frame = 0
        try:
            b5, receipt = self.shot_batch(shot_ids)
            pressure = self.pressure
            require(self.frame == self.nt and bool(torch.isfinite(pressure).all()),
                    "Incomplete or nonfinite fine pressure snapshots")
        finally:
            self.frame_slots = None
            del self.pressure
        return pressure, b5, receipt
