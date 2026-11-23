"""Explicit four-input DeepONet/MIONet fusion with a V2 local residual."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .features import PhaseFeatureBundle
from .medium import MediumEncoding, sample_medium_pyramid
from .source import SourceEncoding
from .travel_time import RayTravelTime


@dataclass(frozen=True)
class FusionOutput:
    normalized_pressure: torch.Tensor
    coarse: torch.Tensor
    local_residual: torch.Tensor
    # The four MIONet rank tensors (velocity, source, travel, trunk) that formed
    # the product; exposed (references only, no extra compute) so the optional
    # multi-arrival additive head (arrival_head.py) can condition on them.
    ranks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None


class TravelTimeBranch(nn.Module):
    """Independent propagation branch Btau of the MIONet product.

    carrier_mode="single_surface" adds the L2 v2 mirror carrier: a bias-free
    zero-initialised Linear(4,width) whose output is added to the first
    layer's pre-GELU activations.  With the weight at its zero init the added
    term is exactly zero, so the branch computes the same function as
    carrier_mode="off" (design section 3's A4-faithful start; the bitwise
    property is asserted by test, not assumed here).  The original
    Linear(9,width) keeps its name, shape and position so parent weights and
    Adam state migrate one-to-one.
    """

    CARRIER_MODES = ("off", "single_surface")
    CARRIER_FEATURES = 4

    def __init__(self, *, width: int, rank: int, carrier_mode: str = "off",
                 phase_features: int = 0) -> None:
        super().__init__()
        if carrier_mode not in self.CARRIER_MODES:
            raise ValueError(f"carrier_mode must be one of {self.CARRIER_MODES}")
        self.carrier_mode = carrier_mode
        # P2 multi-arrival phase channels: zero-initialised bias-free projection
        # added to the same pre-GELU activations as the carrier.  Zero weight ->
        # exactly the parent function (bitwise identity asserted by test).
        self.phase_features = int(phase_features)
        if self.phase_features > 0:
            self.phase_projection = nn.Linear(self.phase_features, width, bias=False)
            nn.init.zeros_(self.phase_projection.weight)
        self.network = nn.Sequential(
            nn.Linear(9, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, rank),
        )
        if carrier_mode == "single_surface":
            self.carrier_projection = nn.Linear(self.CARRIER_FEATURES, width, bias=False)
            nn.init.zeros_(self.carrier_projection.weight)

    def forward(
        self,
        travel: RayTravelTime,
        bundle: PhaseFeatureBundle,
        *,
        domain_t_s: float,
        domain_diagonal_m: float,
        carrier_features: torch.Tensor | None = None,
        phase_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = torch.stack(
            (
                travel.seconds / domain_t_s,
                bundle.tau_s / domain_t_s,
                travel.distance_m / domain_diagonal_m,
                travel.path_velocity_mps / 6000.0,
                travel.endpoint_velocity_mps / 6000.0,
                travel.mean_slowness_s_per_m * 6000.0,
                torch.sin(bundle.phase_rad),
                torch.cos(bundle.phase_rad),
                bundle.causal_feature,
            ),
            dim=-1,
        )
        if self.phase_features > 0:
            if phase_features is None:
                raise ValueError("phase_mode is enabled but no phase_features were supplied")
            if phase_features.shape[-1] != self.phase_features:
                raise ValueError(
                    f"phase_features last dim must be {self.phase_features}, got {phase_features.shape[-1]}")
        elif phase_features is not None:
            raise ValueError("phase_features passed to a branch without phase channels")
        if self.carrier_mode == "off":
            if carrier_features is not None:
                raise ValueError("carrier_features passed to a carrier_mode='off' branch")
            if self.phase_features == 0:
                return self.network(features)
            hidden = self.network[0](features) + self.phase_projection(phase_features)
            return self.network[1:](hidden)
        if carrier_features is None:
            raise ValueError("carrier_mode='single_surface' requires carrier_features")
        if carrier_features.shape[-1] != self.CARRIER_FEATURES:
            raise ValueError(
                f"carrier_features last dim must be {self.CARRIER_FEATURES}, "
                f"got {carrier_features.shape[-1]}")
        hidden = self.network[0](features) + self.carrier_projection(carrier_features)
        if self.phase_features > 0:
            hidden = hidden + self.phase_projection(phase_features)
        return self.network[1:](hidden)


class PhaseAlignedMIONetFusion(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        rank: int,
        pyramid_levels: int,
        heads: int,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by attention heads")
        local_width = width * pyramid_levels
        self.rank = int(rank)
        self.velocity_local_to_rank = nn.Linear(local_width, rank)
        self.local_to_hidden = nn.Linear(local_width, width)
        self.trunk_to_hidden = nn.Linear(rank, width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.local_output = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.coarse_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.local_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        # Default-off research extension: no new legacy state_dict keys.
        self.ic_to_query = None

    def enable_query_local_ic(self) -> nn.Linear:
        """Enable one zero-initialized additive IC projection before DDP setup."""
        if self.ic_to_query is None:
            reference = self.trunk_to_hidden.weight
            projection = nn.Linear(self.attention.embed_dim, self.attention.embed_dim,
                                   device=reference.device, dtype=reference.dtype)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.ic_to_query = projection
        return self.ic_to_query

    def forward(
        self,
        medium: MediumEncoding,
        source: SourceEncoding,
        coords_xy_normalized: torch.Tensor,
        record_to_medium: torch.Tensor,
        travel_rank: torch.Tensor,
        trunk_rank: torch.Tensor,
        cached_velocity_rank_local: torch.Tensor | None = None,
        cached_local_hidden: torch.Tensor | None = None,
    ) -> FusionOutput:
        mapping = torch.as_tensor(
            record_to_medium,
            dtype=torch.long,
            device=coords_xy_normalized.device,
        )
        # P0 record-side cache (v18.5, 20260929): the two local projections and the
        # pyramid sampling depend only on (record, point), not on time, yet the dense
        # per-frame path recomputed them for every frame.  When the caller supplies
        # the cached projections (already tiled to this call's point axis) the three
        # time-independent ops are skipped; every op below this block is IDENTICAL,
        # so the outputs match the uncached path (bitwise asserted by probe_p0_v185).
        # Eval-only: the caller gates on torch.is_grad_enabled() being False.
        if (cached_velocity_rank_local is None) != (cached_local_hidden is None):
            raise ValueError("cached fusion projections must be supplied together")
        if cached_velocity_rank_local is not None:
            if torch.is_grad_enabled():
                raise ValueError("cached fusion projections are an eval-only (no-grad) fast path")
            velocity_rank = medium.rank[mapping, None, :] + cached_velocity_rank_local
            local_hidden = cached_local_hidden
        else:
            local = sample_medium_pyramid(medium, coords_xy_normalized, mapping)
            velocity_rank = (
                medium.rank[mapping, None, :] + self.velocity_local_to_rank(local)
            )
            local_hidden = self.local_to_hidden(local)
        source_rank = source.rank[:, None, :]
        if travel_rank.shape != trunk_rank.shape or travel_rank.shape != velocity_rank.shape:
            raise ValueError("MIONet branch rank shapes disagree")
        product = velocity_rank * source_rank * travel_rank * trunk_rank
        coarse = self.coarse_scale * product.sum(dim=-1) / math.sqrt(self.rank)

        query = (
            self.trunk_to_hidden(trunk_rank)
            + source.hidden[:, None, :]
            + local_hidden
        )
        if self.ic_to_query is not None:
            if source.ic_field is None or source.ic_field.ndim != 4 or source.ic_field.shape[:2] != (coords_xy_normalized.shape[0], self.attention.embed_dim):
                raise ValueError("query-local IC requires per-record encoded IC field")
            grid = coords_xy_normalized.mul(2.0).sub(1.0)[:, None, :, :]
            local_ic = F.grid_sample(source.ic_field, grid, mode="bilinear",
                                     padding_mode="border", align_corners=True).squeeze(2).transpose(1, 2)
            query = query + self.ic_to_query(local_ic)
        tokens = medium.tokens[mapping]
        attended, _ = self.attention(query, tokens, tokens, need_weights=False)
        local_residual = self.local_scale * self.local_output(attended + query).squeeze(-1)
        return FusionOutput(
            normalized_pressure=coarse + local_residual,
            coarse=coarse,
            local_residual=local_residual,
            ranks=(velocity_rank, source_rank, travel_rank, trunk_rank),
        )

    def required_gradient_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        named = tuple(self.named_parameters())
        return {
            "query_local_residual": tuple(
                p
                for n, p in named
                if n.startswith("local_to_hidden")
                or n.startswith("trunk_to_hidden")
                or n.startswith("attention")
                or n.startswith("local_output")
                or n == "local_scale"
            ),
            "mionet_product": tuple(
                p for n, p in named if n.startswith("velocity_local_to_rank") or n == "coarse_scale"
            ),
        }


__all__ = ["FusionOutput", "PhaseAlignedMIONetFusion", "TravelTimeBranch"]
