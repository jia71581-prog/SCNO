"""Direct pressure coefficients on the native Marmousi mesh."""

import torch
from torch import nn
from torch.nn import functional as F

from marmousi_native_time_operator import BINS, NativeTimeOperator


NATIVE = 201
SCHEMA = "marmousi_native201_joint_v1"


class NativeGridOperator(NativeTimeOperator):
    """Use a multiscale encoder and a learned 10 m pressure decoder."""

    def __init__(self, sources, pressure_prior, receiver_prior, observation, lift):
        super().__init__(sources, pressure_prior, receiver_prior, observation, lift)
        width = 16
        self.native_velocity = nn.Sequential(nn.Conv2d(1, width, 3, padding=1, padding_mode="replicate"),
                                             nn.GroupNorm(4, width))
        self.native_context = nn.Conv2d(32, width, 1)
        self.native_readout = nn.Sequential(nn.SiLU(), nn.Conv2d(width, 2 * BINS, 1))
        nn.init.zeros_(self.native_readout[-1].weight)
        nn.init.zeros_(self.native_readout[-1].bias)

    @classmethod
    def restore(cls, checkpoint, device="cpu"):
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if saved["binding"]["schema"] != SCHEMA:
            raise ValueError("Use a native 201×201 pressure checkpoint")
        state = saved["model"]
        state["pressure_prior"] = torch.cat(saved["pressure_prior_shards"])
        model = cls(*(state[key] for key in ("sources", "pressure_prior", "receiver_prior", "observation", "lift")))
        model.load_state_dict(state, strict=True)
        return model.to(device).eval().requires_grad_(False)

    @classmethod
    def from_coarse(cls, seed):
        old = NativeTimeOperator.restore(seed)
        state = old.state_dict()
        model = cls(*(state[key] for key in ("sources", "pressure_prior", "receiver_prior", "observation", "lift")))
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys or not all(key.startswith("native_") for key in result.missing_keys):
            raise ValueError("Only the native 10 m decode parameters may be new")
        return model

    def forward(self, velocity, shots):
        hidden = self.encode(velocity, shots)
        resize = dict(size=(NATIVE, NATIVE), mode="bilinear", align_corners=True)
        prior = self.pressure_prior[shots].movedim(-1, 2).flatten(1, 2).float()
        base = F.interpolate(prior + self.pressure_readout(hidden).float(), **resize)
        local = self.native_velocity((velocity[:, None] / 2200.).log() / .3)
        context = F.interpolate(self.native_context(hidden), **resize)
        fine = base + self.native_readout(local + context).float()
        pressure = torch.view_as_complex(fine.unflatten(1, (BINS, 2)).movedim(2, -1).contiguous())
        return pressure, self.receiver(hidden, shots)

    def pressure_at(self, wave, receiver, slots):
        """Apply the physical boundary and B5 consistency directly at 10 m."""
        value = wave[:, slots] * self.active_frames[slots][None, :, None, None]
        surface = torch.zeros_like(value[:, :, :1])
        depth = self.observation.shape[0] // NATIVE
        upper = torch.cat((surface, value[:, :, 1:depth]), dim=2)
        observed = upper.flatten(2) @ self.observation
        update = ((receiver[:, slots] - observed) @ self.lift).unflatten(-1, (depth - 1, NATIVE))
        return torch.cat((surface, upper[:, :, 1:] + update, value[:, :, depth:]), dim=2)
