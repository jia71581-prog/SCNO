"""Lightweight U-shaped spatial operator for V18 U-FNO arms.

The Fourier branch supplies global/low-wavenumber propagation. This module is a
single low-resolution U-shaped spatial path on the dense hidden field, intended
to recover local multiscale/high-wavenumber/interface structure without turning
the full 401-frame inference into a heavy per-frame U-Net.

It is part of the main neural operator: its 1-channel correction is added inside
the dense decoder correction, before correction_scale and before the final hard
free-surface factor. It is not a post-processing refiner.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LowResUPath(nn.Module):
    """201->101->51 U-shaped spatial path with skip connections.

    Full-resolution work is deliberately narrow. All 3x3 feature convolutions
    use replicate padding, which is only feature extrapolation; it does not
    impose a physical boundary condition. The final network output still owns
    the top free-surface condition, while left/right/bottom CPML remain learned
    from data in the physical domain.
    """
    def __init__(
        self,
        *,
        in_width: int,
        channels: tuple[int, int, int] = (16, 32, 48),
        output_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("u_path channels must have exactly three levels")
        c0, c1, c2 = (int(c) for c in channels)
        if in_width <= 0 or min(c0, c1, c2) <= 0:
            raise ValueError("u_path channel widths must be positive")
        if not 0.0 < float(output_scale) <= 1.0:
            raise ValueError("u_path output_scale must lie in (0,1]")
        self.channels = (c0, c1, c2)
        self.output_scale = float(output_scale)

        self.reduce = nn.Conv2d(in_width, c0, 1)
        self.enc0 = nn.Conv2d(
            c0, c0, 3, padding=1, groups=c0, padding_mode="replicate"
        )
        self.down1 = nn.Conv2d(
            c0, c1, 3, stride=2, padding=1, padding_mode="replicate"
        )
        self.enc1 = nn.Conv2d(
            c1, c1, 3, padding=1, groups=c1, padding_mode="replicate"
        )
        self.down2 = nn.Conv2d(
            c1, c2, 3, stride=2, padding=1, padding_mode="replicate"
        )
        self.mid = nn.Conv2d(c2, c2, 3, padding=1, padding_mode="replicate")
        self.up2_proj = nn.Conv2d(c2, c1, 1)
        self.up2_fuse = nn.Conv2d(2 * c1, c1, 1)
        self.up2_dw = nn.Conv2d(
            c1, c1, 3, padding=1, groups=c1, padding_mode="replicate"
        )
        self.up1_proj = nn.Conv2d(c1, c0, 1)
        self.up1_fuse = nn.Conv2d(2 * c0, c0, 1)
        self.up1_dw = nn.Conv2d(
            c0, c0, 3, padding=1, groups=c0, padding_mode="replicate"
        )
        self.head = nn.Conv2d(c0, 1, 1)
        # Exact step-0 equivalence with the parent operator. The head has a
        # nonzero first-step gradient; earlier U-path layers start receiving
        # gradient after the head opens.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4:
            raise ValueError("u_path hidden must be [batch,channel,z,x]")
        s0 = F.gelu(self.enc0(self.reduce(hidden)))
        h = F.gelu(self.down1(s0))
        s1 = F.gelu(self.enc1(h))
        h = F.gelu(self.down2(s1))
        h = F.gelu(self.mid(h))
        h = self.up2_proj(
            F.interpolate(h, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        )
        h = F.gelu(self.up2_dw(self.up2_fuse(torch.cat((h, s1), dim=1))))
        h = self.up1_proj(
            F.interpolate(h, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        )
        h = F.gelu(self.up1_dw(self.up1_fuse(torch.cat((h, s0), dim=1))))
        return self.output_scale * self.head(h)


__all__ = ["LowResUPath"]
