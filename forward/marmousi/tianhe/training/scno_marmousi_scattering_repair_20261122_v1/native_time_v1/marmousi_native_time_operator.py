"""Marmousi neural forward operator on the original recording clock."""

import torch
from torch import nn
from torch.nn import functional as F


SIDE = 101
FRAMES = 801
BINS = 81


class NativeTimeOperator(nn.Module):
    """Predict complex Fourier coefficients from velocity and registered source IDs."""

    def __init__(self, sources, pressure_prior, receiver_prior, observation, lift):
        super().__init__()
        self.register_buffer("sources", sources.float())
        self.register_buffer("pressure_prior", pressure_prior)
        self.register_buffer("receiver_prior", receiver_prior)
        self.register_buffer("observation", observation)
        self.register_buffer("lift", lift)
        self.register_buffer("axis", torch.linspace(0., 1., SIDE))
        self.register_buffer("active_frames", torch.arange(FRAMES) >= 3)
        width = 32
        self.velocity_lift = nn.Conv2d(1, 4, 3, stride=2, padding=1, padding_mode="replicate")
        self.local = nn.Sequential(nn.Conv2d(10, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU())
        self.regional = nn.Sequential(nn.Conv2d(width, width, 5, padding=2), nn.GroupNorm(4, width), nn.SiLU())
        self.global_context = nn.Sequential(nn.Conv2d(width, width, 5, padding=2), nn.GroupNorm(4, width), nn.SiLU())
        self.body = nn.Sequential(nn.Conv2d(3 * width, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU())
        self.pressure_readout = nn.Conv2d(width, 2 * BINS, 1)
        self.receiver_readout = nn.Sequential(nn.Conv1d(width, width, 5, padding=2), nn.SiLU(),
                                              nn.Conv1d(width, 2 * BINS, 1))
        for layer in (self.pressure_readout, self.receiver_readout[-1]):
            nn.init.normal_(layer.weight, std=.001)
            nn.init.zeros_(layer.bias)

    @classmethod
    def restore(cls, checkpoint, device="cpu"):
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if saved["binding"]["schema"] != "marmousi_native_time_joint_v1":
            raise ValueError("Use a native-time joint Marmousi checkpoint")
        state = saved["model"]
        state["pressure_prior"] = torch.cat(saved["pressure_prior_shards"])
        model = cls(*(state[key] for key in ("sources", "pressure_prior", "receiver_prior", "observation", "lift")))
        model.load_state_dict(state, strict=True)
        return model.to(device).eval().requires_grad_(False)

    def encode(self, velocity, shots):
        source = self.sources[shots]
        v = (velocity[:, None] / 2200.).log() / .3
        speed = F.interpolate(v, (SIDE, SIDE), mode="bilinear", align_corners=True)
        smooth = F.avg_pool2d(F.pad(speed, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
        z, x = torch.meshgrid(self.axis, self.axis, indexing="ij")
        x = x[None] - source[:, 0, None, None] / 2000.
        z = z[None] - source[:, 1, None, None] / 2000.
        coords = torch.stack((x, z, x.abs(), z.abs()), dim=1)
        local = self.local(torch.cat((speed, speed - smooth, self.velocity_lift(v), coords), dim=1))
        regional = self.regional(F.adaptive_avg_pool2d(local, (26, 26)))
        global_context = self.global_context(F.adaptive_avg_pool2d(regional, (9, 9)))
        resize = dict(size=(SIDE, SIDE), mode="bilinear", align_corners=True)
        return self.body(torch.cat((local, F.interpolate(regional, **resize),
                                    F.interpolate(global_context, **resize)), dim=1))

    def receiver(self, hidden, shots):
        b0 = self.receiver_prior[shots].float()
        surface = F.interpolate(hidden[:, :, :4].mean(2), size=201, mode="linear", align_corners=True)
        bc = self.receiver_readout(surface).float().unflatten(1, (BINS, 2)).movedim(2, -1)
        receiver = torch.fft.irfft(torch.view_as_complex((b0 + bc).contiguous()), n=FRAMES, dim=1, norm="ortho")
        return receiver * self.active_frames[None, :, None]

    def receiver_gathers(self, velocity, shots):
        """Run the differentiable B5 path without decoding the pressure field."""
        return self.receiver(self.encode(velocity, shots), shots)

    def forward(self, velocity, shots):
        """Return pressure spectrum and the 801-frame B5 gather for each input pair."""
        hidden = self.encode(velocity, shots)
        p0 = self.pressure_prior[shots].float()
        pc = self.pressure_readout(hidden).float().unflatten(1, (BINS, 2)).movedim(2, -1)
        return torch.view_as_complex((p0 + pc).contiguous()), self.receiver(hidden, shots)

    def wavefield(self, spectrum):
        return torch.fft.irfft(spectrum, n=FRAMES, dim=1, norm="ortho")

    def pressure_at(self, wave, receiver, slots):
        """Decode requested native frames with the fixed B5 observation enforced."""
        selected = wave[:, slots]
        shape = selected.shape[:2]
        raw = selected.flatten(0, 1)[:, None]
        linear = F.interpolate(raw, (201, 201), mode="bilinear", align_corners=True)[:, 0]
        cubic = F.interpolate(raw, (201, 201), mode="bicubic", align_corners=True)[:, 0]
        p = (.5 * (linear + cubic)).unflatten(0, shape)
        surface = torch.zeros_like(p[:, :, :1])
        pressure = torch.cat((surface, p[:, :, 1:]), dim=2)
        pressure = pressure * self.active_frames[slots][None, :, None, None]
        depth = self.observation.shape[0] // 201
        observed = pressure[:, :, :depth].flatten(2) @ self.observation
        update = ((receiver[:, slots] - observed) @ self.lift).unflatten(-1, (depth - 1, 201))
        return torch.cat((surface, pressure[:, :, 1:depth] + update, pressure[:, :, depth:]), dim=2)
