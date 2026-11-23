"""Marmousi direct-query operator with shared temporal Gabor features."""

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


_dense_time_grid = ContextVar("scno_dense_time_grid", default=None)


@contextmanager
def share_gabor_on_dense_grid(point_count):
    """Scope the repeated-time layout already constructed by the dense query."""
    token = _dense_time_grid.set(point_count)
    try:
        yield
    finally:
        _dense_time_grid.reset(token)

class SharedTemporalGabor(nn.Module):
    """Source-frequency quadratures at fixed, spatially shared time centers."""

    def __init__(self):
        super().__init__()
        self.register_buffer("centers", torch.tensor(
            [-0.10, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.42, 0.56, 0.72, 0.90]
        ))
        self.register_buffer("widths", torch.tensor([0.045, 0.12]))
        self.register_buffer("frequency_ratios", torch.tensor([0.75, 1.0, 1.25]))
        self.out_features = 2 * len(self.centers) * len(self.widths) * len(self.frequency_ratios)

    def forward(self, time, source_t0, source_f0, horizon):
        offset = (time - source_t0)[..., None] - horizon * self.centers
        standardized = offset[..., None, :] / (horizon * self.widths[:, None])
        envelope = (0.5 * F.log_softmax(-standardized.square(), dim=-1)).exp()
        phase = (2 * math.pi) * source_f0[..., None, None] * offset[..., None]
        phase = phase * self.frequency_ratios
        quadrature = torch.stack((phase.sin(), phase.cos()), dim=-1)
        bank = envelope[..., None, None] * quadrature[..., None, :, :, :]
        return bank.flatten(-4) / math.sqrt(len(self.widths) * len(self.frequency_ratios))


@dataclass(frozen=True)
class Geometry:
    local_speed_mps: torch.Tensor


@dataclass(frozen=True)
class GlobalCoordinates:
    geometry_inputs: torch.Tensor


class GlobalGaborCoordinateEncoder(nn.Module):
    def __init__(self, inherited):
        super().__init__()
        width = inherited.raw_projection.out_features
        self.fourier_bands = inherited.fourier_bands
        self.gabor = SharedTemporalGabor()
        self.register_buffer("fourier_frequencies",
                             math.pi * (2.0 ** torch.arange(self.fourier_bands)))
        n_input = 9 + 6 * self.fourier_bands + self.gabor.out_features
        self.raw_projection = nn.Linear(n_input, width)
        self.periodic_in = type(inherited.periodic_in)(
            n_input, width, first=True, omega0=inherited.periodic_in.omega0
        )
        self.periodic_hidden = inherited.periodic_hidden
        self.periodic_out = inherited.periodic_out
        self.norm = inherited.norm
        self.anchor_raw_projection = inherited.anchor_raw_projection

    def forward(self, coords, source, geometry, *, domain_x_m, domain_z_m,
                domain_t_s, anchor_time_s=None):
        x, z, time = coords.unbind(dim=-1)
        sx, sz, f0, t0, _ = source[:, None].unbind(dim=-1)
        dx, dz = (x - sx) / domain_x_m, (z - sz) / domain_z_m
        relative = (time - t0) / domain_t_s
        radius = torch.linalg.vector_norm(torch.stack((dx, dz), dim=-1), dim=-1)
        f0_scaled = (f0 / 50.0).expand_as(time)
        local_speed = geometry.local_speed_mps / 3000.0
        spatial = torch.stack((dx, dz, radius, relative, f0_scaled, local_speed), dim=-1)
        normalized = torch.stack((x / domain_x_m, z / domain_z_m, time / domain_t_s), dim=-1)
        angles = normalized[..., None] * self.fourier_frequencies
        periodic = torch.cat((angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)
        geometry_inputs = torch.cat((normalized, spatial), dim=-1)
        points = _dense_time_grid.get()
        if points is None:
            bank = self.gabor(time, t0, f0, domain_t_s)
            values = torch.cat((geometry_inputs, periodic, bank), dim=-1)
        else:
            batch, query_points = time.shape
            if query_points % points:
                raise ValueError("dense Gabor reuse requires a complete spatial grid per time")
            frames = query_points // points
            times = time.reshape(batch, frames, points)[..., 0]
            bank = self.gabor(times, t0, f0, domain_t_s)
            values = torch.cat((geometry_inputs.reshape(batch, frames, points, -1),
                                periodic.reshape(batch, frames, points, -1),
                                bank.unsqueeze(-2).expand(-1, -1, points, -1)), dim=-1).flatten(1, 2)
        hidden = self.periodic_hidden(self.periodic_in(values)) + self.raw_projection(values)
        if anchor_time_s is not None:
            anchor = anchor_time_s[:, None]
            anchor_inputs = torch.stack(((time - anchor) / domain_t_s,
                                         (anchor - t0).expand_as(time) / domain_t_s), dim=-1)
            hidden = hidden + self.anchor_raw_projection(anchor_inputs)
        return self.periodic_out(self.norm(hidden)), GlobalCoordinates(geometry_inputs)


class GeometryBranch(nn.Module):
    def __init__(self, inherited_network):
        super().__init__()
        self.network = inherited_network
        first = self.network[0]
        self.network[0] = nn.Linear(first.in_features, first.out_features)

    def forward(self, geometry, bundle, *, domain_t_s, domain_diagonal_m,
                carrier_features=None, phase_features=None):
        return self.network(bundle.geometry_inputs)


def _geometry(prepared, normalized_xy):
    record_velocity = prepared.medium.velocity_mps[prepared.record_to_medium]
    local = F.grid_sample(record_velocity, (2 * normalized_xy - 1)[:, None],
                          mode="bilinear", padding_mode="border", align_corners=True)
    return Geometry(local[:, 0, 0])


def _repeat_geometry(geometry, count):
    speed = geometry.local_speed_mps
    return Geometry(speed[:, None].expand(-1, count, -1).reshape(speed.shape[0], -1))


def _prepare_geometry_grid(self, prepared, *, x_m, z_m):
    from grouped_ufno_mionet_v3.model.dense import DenseGridCache
    from grouped_ufno_mionet_v3.model.medium import sample_medium_pyramid

    x = torch.as_tensor(x_m, dtype=torch.float32, device=self.device)
    z = torch.as_tensor(z_m, dtype=torch.float32, device=self.device)
    full_size = prepared.medium.encoding.pyramid[0].shape[-2:]
    if x.ndim != 1 or z.ndim != 1 or (len(z), len(x)) != full_size:
        raise ValueError("Dense axes must match the encoded full-resolution grid")
    zz, xx = torch.meshgrid(z, x, indexing="ij")
    xy = torch.stack((xx, zz), dim=-1).reshape(1, -1, 2)
    xy = xy.expand(prepared.source_parameters.shape[0], -1, -1)
    normalized = xy / xy.new_tensor((self.domain_x_m, self.domain_z_m))
    geometry = _geometry(prepared, normalized)
    static = {}
    if not torch.is_grad_enabled():
        local = sample_medium_pyramid(prepared.medium.encoding, normalized, prepared.record_to_medium)
        static = {
            "fusion_velocity_rank_local": self.fusion.velocity_local_to_rank(local),
            "fusion_local_hidden": self.fusion.local_to_hidden(local),
            "decoder_spatial": self.dense_decoder.spatial_term(
                prepared.medium.encoding, prepared.source_encoding, prepared.record_to_medium
            ),
        }
    # The original decoder's cache field name is retained for its call protocol.
    return DenseGridCache(x_m=x, z_m=z, coords_xy_m=xy, coords_xy_normalized=normalized,
                          travel=geometry, height=len(z), width=len(x), **static)


def _query_geometry_chunk(self, prepared, coords_xyz_m):
    coords = torch.as_tensor(coords_xyz_m, dtype=torch.float32, device=self.device)
    normalized = coords[..., :2] / coords.new_tensor((self.domain_x_m, self.domain_z_m))
    geometry = _geometry(prepared, normalized)
    trunk, bundle = self.coordinate_encoder(
        coords, prepared.source_parameters, geometry,
        domain_x_m=self.domain_x_m, domain_z_m=self.domain_z_m, domain_t_s=self.domain_t_s,
        anchor_time_s=prepared.anchor_time_s,
    )
    rank = self.travel_branch(geometry, bundle, domain_t_s=self.domain_t_s,
                              domain_diagonal_m=math.hypot(self.domain_x_m, self.domain_z_m))
    fused = self.fusion(prepared.medium.encoding, prepared.source_encoding, normalized,
                        prepared.record_to_medium, rank, trunk)
    return fused.normalized_pressure * self._free_surface_factor(coords[..., 1])


def _install_global_queries(base):
    if base.phase_mode != "off" or base.arrival_head is not None or base.scattering_head is not None:
        raise ValueError("Use a base configuration with the optional arrival and scattering modules off")
    device = base.device
    base.coordinate_encoder = GlobalGaborCoordinateEncoder(base.coordinate_encoder).to(device)
    # Legacy attribute name is shared with the dense decoder and optimizer groups.
    base.travel_branch = GeometryBranch(base.travel_branch.network).to(device)
    base.carrier_mode = "off"
    base.prepare_dense_grid = MethodType(_prepare_geometry_grid, base)
    base._repeat_travel = _repeat_geometry
    base._query_chunk = MethodType(_query_geometry_chunk, base)



