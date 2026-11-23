"""Fuse UNO's linear spectral/local branches before one inverse FFT.

Pointwise channel mixing commutes with FFT. Preserve the upstream local
branch's input-grid mask, irfftn crop/pad convention and backward scaling,
including its nonstandard resizing, so all learned tensors stay unchanged.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .uno import OperatorBlock_3D, SpectralConv3d_UNO


class FusedSpectralLocalBlock3d(OperatorBlock_3D):
    def __init__(self, original):
        nn.Module.__init__(self)
        if not isinstance(original.conv, SpectralConv3d_UNO):
            raise ValueError('FFT fusion requires the dense spectral module')
        self.conv = original.conv
        self.w = original.w
        self.normalize = original.normalize
        self.non_lin = original.non_lin
        if self.normalize:
            self.normalize_layer = original.normalize_layer

    def forward(self, x, res1=None, res2=None, res3=None):
        dims = (self.conv.d1, self.conv.d2, self.conv.d3) if res1 is None else (res1, res2, res3)
        original_dims = x.shape[-3:]
        ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm='forward')
        n1, n2, n3 = original_dims
        o1, o2, o3 = dims
        kept = (min(n1, o1), min(n2, o2), min(n3//2+1, o3//2+1))
        a = torch.arange(kept[0], device=x.device)
        b = torch.arange(kept[1], device=x.device)
        c = torch.arange(kept[2], device=x.device)
        mask1 = (a < o1//2) | (a >= max(0, n1-o1//2))
        mask2 = (b < o2//2) | (b >= max(0, n2-o2//2))
        mask3 = c < o3//2
        mask = mask1[:, None, None] & mask2[None, :, None] & mask3[None, None, :]
        # The mask is channel-independent, so apply it after channel mixing.
        # This avoids copying all input-channel Fourier coefficients.
        local_input = ft[:, :, :kept[0], :kept[1], :kept[2]]
        channel_weight = self.w.conv.weight[:, :, 0, 0, 0].to(ft.dtype)
        local_output = torch.einsum('bixyt,oi->boxyt', local_input, channel_weight)
        local_output.mul_(mask)
        output = ft.new_zeros(x.shape[0], self.conv.out_channels, o1, o2, o3//2+1)
        # Original local path uses unnormalized forward FFT and normalized IFFT.
        ratio = math.prod(original_dims) / math.prod(dims)
        output[:, :, :kept[0], :kept[1], :kept[2]] = local_output * ratio
        del local_output, local_input
        if self.w.conv.bias is not None:
            output[:, :, 0, 0, 0] = output[:, :, 0, 0, 0] + self.w.conv.bias * ratio
        m1, m2, m3 = self.conv.modes1, self.conv.modes2, self.conv.modes3
        quadrants = [(slice(0, m1), slice(0, m2)), (slice(-m1, None), slice(0, m2)),
                     (slice(0, m1), slice(-m2, None)), (slice(-m1, None), slice(-m2, None))]
        for i, (s1, s2) in enumerate(quadrants, start=1):
            contribution = self.conv.compl_mul3d(ft[:, :, s1, s2, :m3], getattr(self.conv, f'weights{i}'))
            output[:, :, s1, s2, :m3] = output[:, :, s1, s2, :m3] + contribution
        del ft, contribution
        value = torch.fft.irfftn(output, s=dims, norm='forward')
        # These linear outputs are no longer used by this forward. Autograd
        # retains any tensors it needs; checkpointed execution can release the
        # large Python-owned buffers before normalization and GELU.
        del output
        if self.normalize:
            value = self.normalize_layer(value)
        if self.non_lin:
            value = F.gelu(value)
        return value


def fuse_operator_blocks(model):
    for name in [f'conv{i}' for i in range(9)]:
        setattr(model, name, FusedSpectralLocalBlock3d(getattr(model, name)))
    return model
