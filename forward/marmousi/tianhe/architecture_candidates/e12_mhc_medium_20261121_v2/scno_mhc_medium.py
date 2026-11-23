"""Checkpoint-preserving two-stream routing inside the SCNO medium pyramid."""

from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F


class TwoStreamMedium(nn.Module):
    """Keep the original mean path and carry a signed contrast across 51→26."""

    def __init__(self, parent, mix=1.):
        super().__init__()
        if len(parent.blocks) != 4 or len(parent.down) != 3 or not 0. < mix <= 1.:
            raise ValueError("Expected the four-stage SCNO encoder and a positive contrast gain")
        self.parent = parent
        self.mix = float(mix)
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.read = nn.Parameter(parent.lift.weight.new_zeros((2, parent.width)))
        source = import_module(type(parent).__module__)
        self.position_features = source._position_features
        self.encoding = source.MediumEncoding

    def routed_block(self, index, mean, contrast):
        correction = self.read[index].tanh()[None, :, None, None] * contrast
        query = mean + correction
        predicted = self.parent.blocks[index + 2](query)
        # A common residual update plus an asymmetric write into the contrast.
        return predicted - correction, contrast + .2 * (predicted - query)

    def forward(self, velocity):
        parent = self.parent
        if velocity.ndim != 4 or velocity.shape[1] != 1:
            raise ValueError("Physical medium wrapper expects [B,1,Z,X]")
        velocity = velocity if velocity.is_floating_point() else velocity.float()
        position, _ = self.position_features(*velocity.shape[-2:], bands=parent.position_bands,
                                             dtype=velocity.dtype, device=velocity.device)
        level = parent.lift(torch.cat((velocity, position[None].expand(len(velocity), -1, -1, -1)), dim=1))
        pyramid = []
        for i in (0, 1):
            level = parent.blocks[i](level)
            pyramid.append(level)
            level = parent.down[i](level)
        smooth = F.avg_pool2d(level, 3, stride=1, padding=1, count_include_pad=False)
        level, contrast = self.routed_block(0, level, .1 * (level - smooth))
        pyramid.append(level)
        # The conventional Conv+GELU acts on the mean only. An unbiased
        # linear map transports the signed difference before the next read.
        stencil = parent.down[2][0]
        contrast = F.conv2d(contrast, stencil.weight, stride=stencil.stride, padding=stencil.padding,
                            dilation=stencil.dilation, groups=stencil.groups)
        level, _ = self.routed_block(1, parent.down[2](level), self.mix * contrast)
        pyramid.append(level)
        if parent.u_enhance is not None:
            pyramid[0] = pyramid[0] + parent.u_enhance(pyramid)
        pooled = F.adaptive_avg_pool2d(level, (parent.token_grid, parent.token_grid))
        tokens = pooled.flatten(2).transpose(1, 2)
        features, grid = self.position_features(parent.token_grid, parent.token_grid,
                                               bands=parent.position_bands, dtype=tokens.dtype, device=tokens.device)
        tokens = parent.token_norm(tokens + parent.token_position_projection(features.flatten(1).transpose(0, 1))[None])
        positions = grid.reshape(1, -1, 2).expand(len(velocity), -1, -1)
        return self.encoding(pyramid=tuple(pyramid), tokens=tokens, token_positions=positions,
                             rank=parent.rank_projection(tokens.mean(dim=1)))


def attach_mhc(model, mix):
    """Touch the fresh in-memory instance after restoring the frozen checkpoint."""
    parent = model.core.anchor.model.parent
    if isinstance(parent.medium_encoder, TwoStreamMedium):
        raise ValueError("The medium was already wrapped")
    adapter = TwoStreamMedium(parent.medium_encoder, mix=mix)
    parent.medium_encoder = adapter
    return adapter
