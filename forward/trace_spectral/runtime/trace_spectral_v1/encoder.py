"""A single multiscale spatial pass with global mixing of hidden features."""

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                         nn.GroupNorm(8, cout), nn.SiLU())


class SpectralMix(nn.Module):
    """Low Fourier modes with two real, mode-independent channel factors."""

    def __init__(self, channels, rank=32, modes=12, padding=3):
        super().__init__()
        self.modes, self.padding = modes, padding
        self.reduce = nn.Conv2d(channels, rank, 1, bias=False)
        self.expand = nn.Conv2d(rank, channels, 1, bias=False)
        weight = torch.randn(2, modes, modes, rank, rank, 2) / rank**.5
        self.weight = nn.Parameter(weight)

    def mix_modes(self, spectrum, height):
        """Preserve Hermitian symmetry on the stored real-axis boundary."""
        mz = min(self.modes, height // 2)
        mx = min(self.modes, spectrum.shape[-1] - 1)
        low = torch.stack((spectrum[:, :, :mz, :mx], spectrum[:, :, -mz:, :mx]), dim=1)
        weight = torch.view_as_complex(self.weight[:, :mz, :mx].contiguous())
        modes = torch.einsum('bsizx,szxio->bsozx', low, weight)
        positive, negative = modes.unbind(dim=1)
        # rfft2 stores x >= 0. For x=0, the z spectrum must itself have
        # conjugate pairs. The unmatched lowest negative mode is zero unless
        # it is the Nyquist mode, in which case it must be real.
        dc = torch.complex(positive[..., :1, 0].real, torch.zeros_like(positive[..., :1, 0].real))
        pair = 0.5 * (positive[..., 1:, 0] + negative[..., 1:, 0].flip(-1).conj())
        lone = negative[..., :1, 0].real
        lone = torch.complex(lone if mz * 2 == height else torch.zeros_like(lone), torch.zeros_like(lone))
        positive = torch.cat((torch.cat((dc, pair), dim=-1).unsqueeze(-1), positive[..., 1:]), dim=-1)
        negative = torch.cat((torch.cat((lone, pair.flip(-1).conj()), dim=-1).unsqueeze(-1),
                              negative[..., 1:]), dim=-1)
        middle = positive.new_zeros((*positive.shape[:-2], height - 2 * mz, mx))
        return F.pad(torch.cat((positive, middle, negative), dim=-2), (0, spectrum.shape[-1] - mx))

    def forward(self, features):
        reduced = self.reduce(features)
        dtype = torch.float64 if reduced.dtype == torch.float64 else torch.float32
        p = self.padding
        with torch.autocast(device_type=reduced.device.type, enabled=False):
            padded = F.pad(reduced.to(dtype=dtype), (p, p, p, p))
            spectrum = torch.fft.rfft2(padded)
            mixed = self.mix_modes(spectrum, padded.shape[-2])
            result = torch.fft.irfft2(mixed, s=padded.shape[-2:])
            cropped = result[..., p:-p, p:-p] if p else result
        return self.expand(cropped.to(dtype=reduced.dtype))


class GlobalBlock(nn.Module):
    def __init__(self, channels=96):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.spectral = SpectralMix(channels)
        self.local = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                   nn.Conv2d(channels, channels, 1))
        self.mlp = nn.Sequential(nn.Conv2d(channels, 2 * channels, 1), nn.SiLU(),
                                 nn.Conv2d(2 * channels, channels, 1))

    def forward(self, x):
        h = F.silu(self.norm(x))
        return x + self.spectral(h) + self.local(h) + self.mlp(h)


class MultiScaleEncoder(nn.Module):
    """Native detail and a pooled global context, shared by every output time."""

    native_channels = 32
    context_channels = 96

    def __init__(self, in_channels):
        super().__init__()
        self.stem = conv_block(in_channels, 32)
        self.down1 = conv_block(32, 48, stride=2)
        self.down2 = conv_block(48, 64, stride=2)
        self.down3 = conv_block(64, 96, stride=2)
        self.global_blocks = nn.Sequential(*(GlobalBlock() for _ in range(4)))
        self.up2 = conv_block(96 + 64, 64)
        self.up1 = conv_block(64 + 48, 48)
        self.up0 = conv_block(48 + 32, 32)

    def forward(self, inputs):
        f0 = self.stem(inputs)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        coarse = self.global_blocks(self.down3(f2))
        context = coarse.mean(dim=(-2, -1))
        d2 = self.up2(torch.cat((F.interpolate(coarse, size=f2.shape[-2:], mode='nearest'), f2), dim=1))
        d1 = self.up1(torch.cat((F.interpolate(d2, size=f1.shape[-2:], mode='nearest'), f1), dim=1))
        detail = self.up0(torch.cat((F.interpolate(d1, size=f0.shape[-2:], mode='nearest'), f0), dim=1))
        return detail, context
