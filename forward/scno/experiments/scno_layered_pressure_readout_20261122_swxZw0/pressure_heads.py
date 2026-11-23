"""Matched multiscale pressure residual heads on the native SCNO grid."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualHead(nn.Module):
    """The same local feature MLP and temporal factorization in both arms."""

    def __init__(self, decoder, slots, seed):
        super().__init__()
        self.layers = nn.ModuleList()
        self.temporal = nn.ParameterList()
        self.slots = tuple(slots)
        self.rank = decoder.detail_latent
        for scale, (template, temporal, count) in enumerate(
            zip(decoder.detail_heads, decoder.detail_time, self.slots, strict=True)
        ):
            width = template[-1].in_features
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + scale)
                layers = nn.Sequential(
                    nn.Linear(template[0].in_features, width), nn.SiLU(),
                    nn.Linear(width, width), nn.SiLU(),
                )
                for hidden in (layers[0], layers[2]):
                    nn.init.xavier_uniform_(hidden.weight)
                    nn.init.zeros_(hidden.bias)
                layers.append(nn.Linear(width, self.rank * count))
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            self.layers.append(layers)
            self.temporal.append(nn.Parameter(temporal.weight.detach().cpu().clone()))

    def hidden(self, features):
        return tuple(layers[:-1](block) for layers, block in zip(self.layers, features, strict=True))

    def fold(self, rows):
        result = []
        for layers, temporal, slots in zip(self.layers, self.temporal, self.slots, strict=True):
            factor = temporal.index_select(0, rows)
            output = layers[-1]
            weight = factor @ output.weight.reshape(self.rank, -1)
            bias = factor @ output.bias.reshape(self.rank, slots)
            # Use one GEMM over all requested rows and slots for each scale.
            result.append((weight.reshape(-1, output.in_features), bias.flatten(), slots))
        return tuple(result)

    @staticmethod
    def packets(hidden, folded, count):
        return tuple(
            F.linear(block, weight, bias).reshape(*block.shape[:-1], count, slots).movedim(3, 1)
            for block, (weight, bias, slots) in zip(hidden, folded, strict=True)
        )


def local_inputs(decoder, features):
    _, packets = decoder.detail_transform.analyze(decoder.detail_projection(features[:, :, 1:]))
    inputs = []
    for packet, level in zip(packets, decoder.detail_transform.levels, strict=True):
        block = packet.permute(0, 2, 3, 1, 4).flatten(-2)
        mask = level.valid_detail.to(block).expand(len(block), -1, -1, -1)
        inputs.append(torch.cat((block, mask), dim=-1))
    return tuple(inputs)


def orthogonal_pressure(transform, packets):
    batch, count = packets[0].shape[:2]
    last = transform.levels[-1]
    coarse = packets[0].new_zeros(batch * count, 1, last.gz * 4, last.gx * 4)
    details = [packet.flatten(0, 1).unsqueeze(1) for packet in packets]
    return transform.synthesize(coarse, details).reshape(batch, count, *transform.shape)


def _patch_axis(length, blocks, stride, knots):
    # On this grid the last x block carries one extra physical node.
    leaves = [8] * (length // 8)
    leaves[-1] += length % 8
    lengths = [sum(leaves[first:first + stride]) for first in range(0, len(leaves), stride)]
    if len(lengths) != blocks:
        raise ValueError("Direct residual patches must cover the same native blocks as SCNO")
    offset = torch.arange(blocks, dtype=torch.float32) * knots
    coordinates = torch.cat([
        torch.linspace(0, knots - 1, nodes) + start
        for nodes, start in zip(lengths, offset, strict=True)
    ])
    return 2 * coordinates / (blocks * knots - 1) - 1


class DirectPressurePatches(nn.Module):
    """Output ordinary nodal pressure patches, with added coarse-grid context."""

    # The fine head predicts every native node. The other two heads predict
    # nodal values on 5 x 8 lattices inside the same larger support blocks.
    # 72 + 40 + 40 = 56 + 48 + 48, so total trainable capacity is identical.
    slots = (72, 40, 40)

    def __init__(self, transform):
        super().__init__()
        fine = transform.levels[0]
        self.shape = tuple(transform.shape)
        if self.shape != (200, 201) or (fine.gz_modes, fine.gx_modes) != (8, 9):
            raise ValueError("Expected the native 200 x 201 active grid with 8 x 9 padded fine patches")
        self.register_buffer("fine_nodes", fine.unpad.detach().cpu().clone())
        for scale, level in enumerate(transform.levels[1:], start=1):
            z = _patch_axis(self.shape[0], level.gz, 2**scale, 5)
            x = _patch_axis(self.shape[1], level.gx, 2**scale, 8)
            grid_z, grid_x = torch.meshgrid(z, x, indexing="ij")
            self.register_buffer(f"grid_{scale}", torch.stack((grid_x, grid_z), dim=-1)[None])

    def forward(self, packets):
        fine, *coarse = packets
        batch, count, gz, gx = fine.shape[:4]
        rectangles = fine.reshape(batch, count, gz, gx, 8, 9).transpose(-3, -2).flatten(-4)
        pressure = rectangles.index_select(-1, self.fine_nodes).reshape(batch, count, *self.shape)
        for scale, values in enumerate(coarse, start=1):
            gz, gx = values.shape[2:4]
            tiled = values.reshape(batch, count, gz, gx, 5, 8).transpose(-3, -2)
            tiled = tiled.reshape(batch, count, gz * 5, gx * 8)
            grid = getattr(self, f"grid_{scale}").expand(batch, -1, -1, -1)
            pressure = pressure + F.grid_sample(tiled, grid, mode="bilinear", align_corners=True)
        return pressure


def build_pair(decoder, seed, device):
    orthogonal_slots = [level.valid_detail.shape[-1] for level in decoder.detail_transform.levels]
    arms = {
        "orthogonal": ResidualHead(decoder, orthogonal_slots, seed),
        "direct_pressure": ResidualHead(decoder, DirectPressurePatches.slots, seed),
    }
    a, b = arms.values()
    if sum(p.numel() for p in a.parameters()) != sum(p.numel() for p in b.parameters()):
        raise ValueError("The paired residual heads have different trainable budgets")
    for la, lb, ta, tb in zip(a.layers, b.layers, a.temporal, b.temporal, strict=True):
        for pa, pb in zip(la[:-1].parameters(), lb[:-1].parameters(), strict=True):
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)
        torch.testing.assert_close(ta, tb, rtol=0, atol=0)
    return {name: head.to(device) for name, head in arms.items()}
