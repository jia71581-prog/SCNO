"""Fixed local polynomial analysis and learned full-resolution detail synthesis."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class _Block:
    nodes: np.ndarray
    scaling: np.ndarray


def _legendre(nodes):
    u = 2.0 * (nodes - nodes[0]) / (nodes[-1] - nodes[0]) - 1.0
    return np.polynomial.legendre.legvander(u, 3)


def _orthogonal(x):
    q, r = np.linalg.qr(x, mode="complete")
    diag = np.diag(r[:4])
    if np.min(np.abs(diag)) < 1e-10:
        raise ValueError("degenerate polynomial block")
    q[:, :4] *= np.where(diag < 0, -1.0, 1.0)
    return q


def _axis(n, merges):
    sizes = [8] * (n // 8)
    tail = n % 8
    if not sizes:
        raise ValueError("expected at least 8 native nodes on each axis")
    if tail >= 4:
        sizes.append(tail)
    elif tail:
        sizes[-1] += tail
    leaves, level, start = [], [], 0
    for size in sizes:
        nodes = np.arange(start, start + size, dtype=np.float64)
        q = _orthogonal(_legendre(nodes))
        leaves.append(_Block(nodes, q[:, :4]))
        level.append((q, nodes.size))
        start += size
    result = [level]
    blocks = leaves
    for _ in range(merges):
        level, parents = [], []
        for j in range(0, len(blocks), 2):
            children = blocks[j:j + 2]
            if len(children) == 1:
                level.append((np.eye(4), 4))
                parents.append(children[0])
                continue
            a, b = children
            nodes = np.concatenate((a.nodes, b.nodes))
            values = _legendre(nodes)
            na = len(a.nodes)
            q = _orthogonal(np.concatenate((a.scaling.T @ values[:na],
                                             b.scaling.T @ values[na:])))
            parent = np.concatenate((a.scaling @ q[:4, :4], b.scaling @ q[4:, :4]))
            level.append((q, 8))
            parents.append(_Block(nodes, parent))
        result.append(level)
        blocks = parents
    return result


def _pack(level):
    lengths = np.asarray([n for _, n in level])
    g, w = len(level), int(lengths.max())
    basis = np.zeros((g, w, w), dtype=np.float64)
    select = np.zeros((g, w), dtype=np.int64)
    unpad = []
    start = 0
    for k, (q, n) in enumerate(level):
        basis[k, :n, :n] = q
        select[k, :n] = np.arange(start, start + n)
        unpad.extend(k * w + np.arange(n))
        start += n
    return basis, select, np.asarray(unpad), lengths


class _Level(nn.Module):
    def __init__(self, z, x):
        super().__init__()
        qz, iz, rz, nz = _pack(z)
        qx, ix, rx, nx = _pack(x)
        self.nz, self.nx = int(nz.sum()), int(nx.sum())
        self.gz, self.gz_modes = iz.shape
        self.gx, self.gx_modes = ix.shape
        self.modes = self.gz_modes * self.gx_modes
        is_scaling = ((np.arange(self.gz_modes)[:, None] < 4)
                      & (np.arange(self.gx_modes)[None, :] < 4))
        # Fixed storage order: SS, DS, SD, DD (axis order z, x).
        elements = np.arange(self.modes).reshape(self.gz_modes, self.gx_modes)
        order = np.concatenate((elements[:4, :4].ravel(), elements[4:, :4].ravel(),
                                elements[:4, 4:].ravel(), elements[4:, 4:].ravel()))
        valid = ((np.arange(self.gz_modes)[None, None, :, None] < nz[:, None, None, None])
                 & (np.arange(self.gx_modes)[None, None, None, :] < nx[None, :, None, None]))
        self.subband_sizes = tuple(int((valid & band).sum()) for band in
                                   (~is_scaling & (elements % self.gx_modes < 4),
                                    ~is_scaling & (elements // self.gx_modes < 4),
                                    (elements // self.gx_modes >= 4) &
                                    (elements % self.gx_modes >= 4)))
        tables = {
            "qz": qz[:, None].astype("float32"), "qx": qx[None].astype("float32"),
            "gather": (iz[:, None, :, None] * self.nx + ix[None, :, None, :]).ravel(),
            "unpad": (rz[:, None] * (self.gx * self.gx_modes) + rx[None, :]).ravel(),
            "order": order, "inverse_order": np.argsort(order),
            "valid_detail": valid.reshape(self.gz, self.gx, self.modes)[..., order[16:]],
        }
        for key, value in tables.items():
            # QR complements can differ across numerical libraries; persist
            # the bases so a restored head sees exactly the same coordinates.
            self.register_buffer(key, torch.from_numpy(value), persistent=key in {"qz", "qx"})

    def analyze(self, field):
        batch, channels = field.shape[:2]
        blocks = field.flatten(-2).index_select(-1, self.gather)
        blocks = blocks.reshape(batch, channels, self.gz, self.gx, self.gz_modes, self.gx_modes)
        modes = self.qz.transpose(-2, -1) @ (blocks @ self.qx)
        modes = modes.flatten(-2).index_select(-1, self.order)
        coarse = modes[..., :16].reshape(batch, channels, self.gz, self.gx, 4, 4)
        coarse = coarse.transpose(-3, -2).reshape(batch, channels, self.gz * 4, self.gx * 4)
        return coarse, modes

    def synthesize(self, coarse, detail):
        batch, channels = coarse.shape[:2]
        ss = coarse.reshape(batch, channels, self.gz, 4, self.gx, 4)
        ss = ss.transpose(-3, -2).flatten(-2)
        modes = torch.cat((ss, detail.masked_fill(~self.valid_detail, 0)), dim=-1)
        modes = modes.index_select(-1, self.inverse_order)
        modes = modes.reshape(batch, channels, self.gz, self.gx, self.gz_modes, self.gx_modes)
        blocks = self.qz @ (modes @ self.qx.transpose(-2, -1))
        field = blocks.transpose(-3, -2).flatten(-4)
        return field.index_select(-1, self.unpad).reshape(batch, channels, self.nz, self.nx)


class PolynomialPyramid(nn.Module):
    """Separable transform; only SS recurses. Stored padding has no physical DOF."""

    def __init__(self, shape=(200, 201), merges=2):
        super().__init__()
        self.shape = tuple(shape)
        self.levels = nn.ModuleList(_Level(z, x) for z, x in zip(
            _axis(shape[0], merges), _axis(shape[1], merges)))

    def analyze(self, field):
        if field.ndim != 4 or field.shape[-2:] != self.shape:
            raise ValueError("expected [T,C,Z,X] on the fixed interior grid")
        with torch.autocast(device_type=field.device.type, enabled=False):
            field, packets = field.float(), []
            for level in self.levels:
                field, modes = level.analyze(field)
                packets.append(modes)
        return field, packets

    def synthesize(self, coarse, details):
        if len(details) != len(self.levels):
            raise ValueError("expected one detail packet per scale")
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            field = coarse.float()
            for level, detail in zip(reversed(self.levels), reversed(details)):
                field = level.synthesize(field, detail.float())
        return field


class SpectralDetailReadout(nn.Module):
    """Local band predictors with same-block scaling and source/time conditioning."""

    def __init__(self, fused_channels=2, hidden_channels=16, time_features=132, width=32):
        super().__init__()
        self.transform = PolynomialPyramid()
        self.fused_projection = nn.Conv2d(fused_channels, 3, 1)
        self.refined_projection = nn.Conv2d(hidden_channels, 4, 1)
        self.time_projection = nn.Linear(time_features, 8)
        self.source_projection = nn.Linear(5, 8)
        self.heads = nn.ModuleList()
        for level in self.transform.levels:
            detail_slots = level.modes - 16
            head = nn.Sequential(nn.Linear(level.modes * 8 + 16 + detail_slots, width),
                                 nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, detail_slots))
            nn.init.normal_(head[-1].weight, std=1e-3)
            nn.init.zeros_(head[-1].bias)
            self.heads.append(head)

    def forward(self, fused, refined, velocity, source, temporal_features):
        count = refined.shape[0]
        v = (velocity[:, :, 1:] / 3000 - 1).expand(count, -1, -1, -1)
        fine_fusion = F.interpolate(self.fused_projection(fused), size=(201, 201),
                                    mode="bilinear", align_corners=True)
        field = torch.cat((fine_fusion[:, :, 1:], self.refined_projection(refined[:, :, 1:]), v), dim=1)
        coarse, packets = self.transform.analyze(field)
        src = source / source.new_tensor([2000., 2000., 15., 1., 1.])
        cond = torch.cat((self.source_projection(src).expand(count, -1),
                          self.time_projection(temporal_features)), dim=-1)
        details = []
        for level, predictor, modes in zip(self.transform.levels, self.heads, packets):
            modes = modes.permute(0, 2, 3, 1, 4).flatten(-2)
            condition = cond[:, None, None, :].expand(-1, level.gz, level.gx, -1)
            mask = level.valid_detail[None].expand(count, -1, -1, -1)
            incoming = torch.cat((modes, condition, mask.to(modes)), dim=-1)
            detail = predictor(incoming).masked_fill(~mask, 0)
            details.append(detail[:, None])
        interior = self.transform.synthesize(torch.zeros_like(coarse[:, :1]), details)
        return F.pad(interior, (0, 0, 1, 0))
