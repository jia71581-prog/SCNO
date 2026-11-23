"""V2-informed FiLM plus learned-complex dense wavefield correction."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .medium import MediumEncoding
from .source import SourceEncoding
from .spectral import ComplexSpectralResidualBlock
from .travel_time import RayTravelTime


@dataclass(frozen=True)
class DenseGridCache:
    x_m: torch.Tensor
    z_m: torch.Tensor
    coords_xy_m: torch.Tensor
    coords_xy_normalized: torch.Tensor
    travel: RayTravelTime
    height: int
    width: int
    # L2 carrier: per-grid single-surface mirror travel time [record, points];
    # time-independent, so it is integrated once per grid and reused for every
    # time block (QUADRATURE_CONVERGENCE caching plan).  None when carrier off.
    mirror_tau_s: torch.Tensor | None = None
    # P2 multi-arrival phase maps flattened to the grid: [record, arrivals, points]
    # travel times (inf = absent) and [record, arrivals] reflection proxies.
    phase_T: torch.Tensor | None = None
    phase_r: torch.Tensor | None = None
    # P0 record-side caches (v18.5, 20260929): time-independent per-point terms the
    # per-frame path used to recompute for every frame.  Built ONLY under
    # torch.no_grad in prepare_dense_grid and consumed ONLY when grad is disabled,
    # so the training path (gradients, checkpointing) is bitwise untouched.
    #   fusion_velocity_rank_local  [record, points, rank]  = fusion.velocity_local_to_rank(local)
    #   fusion_local_hidden         [record, points, width] = fusion.local_to_hidden(local)
    #   decoder_spatial             [record, width, H, W]   = dense spatial term (fuse+src+map)
    fusion_velocity_rank_local: torch.Tensor | None = None
    fusion_local_hidden: torch.Tensor | None = None
    decoder_spatial: torch.Tensor | None = None


class ReflectionBranch(nn.Module):
    """UL31 interface-aware residual correction on the final dense hidden field.

    Inputs: hidden_f [B*T, width, H, W] and the matching velocity [B*T, 1, H, W]
    (m/s).  Interface features are differentiable central differences of log(c)
    under replicate padding (dx, dz, |grad|) plus log(c/3000) as a normalized
    velocity channel; the 4-channel stack is projected and multiplied against a
    projection of the hidden field, then passed through depthwise 3x3 convs at
    dilation 1 and 2 and a pointwise mix.  The 1-channel head is exact-zero
    initialised (weight AND bias), so the enabled model's initial function is
    bitwise the baseline.
    """

    def __init__(self, *, width: int, channels: int = 32) -> None:
        super().__init__()
        if width <= 0 or channels <= 0:
            raise ValueError("reflection branch width and channels must be positive")
        self.iface_proj = nn.Conv2d(4, channels, kernel_size=1)
        self.hidden_proj = nn.Conv2d(width, channels, kernel_size=1)
        self.dw_dil1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels)
        self.dw_dil2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.head = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _interface_features(velocity: torch.Tensor) -> torch.Tensor:
        logc = torch.log(torch.clamp(velocity, min=1.0))
        padded = F.pad(logc, (1, 1, 1, 1), mode="replicate")
        dx = 0.5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2])
        dz = 0.5 * (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1])
        mag = torch.sqrt(dx * dx + dz * dz + 1.0e-12)
        return torch.cat((dx, dz, mag, logc - math.log(3000.0)), dim=1)

    def forward(self, hidden_f: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        if hidden_f.shape[0] != velocity.shape[0] or velocity.shape[1] != 1:
            raise ValueError("reflection branch expects matching [B*T,*,H,W] inputs")
        iface = self.iface_proj(self._interface_features(velocity))
        mixed = F.gelu(iface * self.hidden_proj(hidden_f))
        local = F.gelu(self.dw_dil1(mixed) + self.dw_dil2(mixed))
        return self.head(F.gelu(self.pointwise(local)))


class TimeConditionedComplexDenseDecoder(nn.Module):
    """Correct shared MIONet frames with retained V2 FiLM and complex FNO blocks."""

    def __init__(
        self,
        *,
        width: int,
        spectral_rank: int,
        pyramid_levels: int,
        dense_modes: tuple[int, ...],
        time_block: int,
        domain_t_s: float,
        use_checkpoint: bool = True,
        equivariant: bool = False,
        radial_cutoff: int = 0,
        # V18-A speed branch (20260928): multi-dilation local path inside each
        # dense spectral block.  Default off adds no parameters and no keys.
        local_multi_dilation: bool = False,
        u_path_enabled: bool = False,
        u_path_channels: tuple[int, int, int] = (16, 32, 48),
        u_path_output_scale: float = 0.01,
        fixed_wave_physics: bool = False,
        # UL31 reflection branch (20260929): default off adds no parameters/keys.
        reflection_branch: bool = False,
        reflection_channels: int = 32,
    ) -> None:
        super().__init__()
        if not dense_modes or time_block <= 0 or domain_t_s <= 0:
            raise ValueError("dense modes, time block, and time scale must be positive")
        if equivariant and radial_cutoff <= 0:
            raise ValueError("equivariant dense blocks require a positive radial cutoff")
        self.use_checkpoint = bool(use_checkpoint)
        self.time_block = int(time_block)
        self.domain_t_s = float(domain_t_s)
        self.multiscale_fuse = nn.Conv2d(width * pyramid_levels, width, kernel_size=1)
        self.source_projection = nn.Linear(width, width)
        self.map_projection = nn.Conv2d(width, width, kernel_size=1)
        self.coarse_lift = nn.Conv2d(1, width, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(5, width),
            nn.GELU(),
            nn.Linear(width, 2 * width),
        )
        self.blocks = nn.ModuleList(
            [
                ComplexSpectralResidualBlock(
                    width=width,
                    spectral_rank=spectral_rank,
                    modes_y=radial_cutoff if equivariant else mode,
                    modes_x=radial_cutoff if equivariant else mode,
                    equivariant=equivariant,
                    local_multi_dilation=local_multi_dilation,
                )
                for mode in dense_modes
            ]
        )
        self.output = nn.Conv2d(width, 1, kernel_size=1)
        self.correction_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # V18 U-FNO arm: one narrow U-shaped spatial operator on the final dense
        # hidden field. Construction is RNG-neutral for exact paired ablations;
        # its zero head makes the enabled step-0 function identical to baseline.
        self.u_path = None
        if bool(u_path_enabled):
            from .u_path import LowResUPath
            rng_state = torch.get_rng_state()
            self.u_path = LowResUPath(
                in_width=width,
                channels=tuple(int(v) for v in u_path_channels),
                output_scale=float(u_path_output_scale),
            )
            torch.set_rng_state(rng_state)

        # V18 physics arm: fixed eighth-order features of the pressure-like
        # coarse field, projected with an exact zero-initialised 1x1 layer.
        self.fixed_wave_physics = None
        if bool(fixed_wave_physics):
            from .fixed_wave_physics import FixedWavePhysicsBranch
            self.fixed_wave_physics = FixedWavePhysicsBranch(width=width)

        # UL31 reflection branch: RNG-neutral construction (u_path pattern) so
        # enabling it leaves every other module's initialisation bitwise unchanged.
        self.reflection_branch = None
        if bool(reflection_branch):
            rng_state = torch.get_rng_state()
            self.reflection_branch = ReflectionBranch(
                width=width, channels=int(reflection_channels))
            torch.set_rng_state(rng_state)

    def spatial_term(
        self,
        medium: MediumEncoding,
        source: SourceEncoding,
        record_to_medium: torch.Tensor,
    ) -> torch.Tensor:
        """Time-independent spatial conditioning term [record, width, H, W].

        Pure refactor of the first stage of forward() (identical ops in identical
        order, so the training path is bitwise unchanged); also the P0 cache
        builder (prepare_dense_grid) and the cached forward share one definition.
        """
        mapping = torch.as_tensor(
            record_to_medium, dtype=torch.long, device=medium.pyramid[0].device)
        full_size = medium.pyramid[0].shape[-2:]
        levels = [medium.pyramid[0][mapping]]
        levels.extend(
            F.interpolate(
                level[mapping],
                size=full_size,
                mode="bilinear",
                align_corners=True,
            )
            for level in medium.pyramid[1:]
        )
        return (
            self.multiscale_fuse(torch.cat(levels, dim=1))
            + self.source_projection(source.hidden)[:, :, None, None]
            + self.map_projection(source.map_field)
        )

    def forward(
        self,
        medium: MediumEncoding,
        source: SourceEncoding,
        source_parameters: torch.Tensor,
        record_to_medium: torch.Tensor,
        time_s: torch.Tensor,
        coarse: torch.Tensor,
        cached_spatial: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mapping = torch.as_tensor(record_to_medium, dtype=torch.long, device=coarse.device)
        if coarse.ndim != 4 or time_s.shape != coarse.shape[:2]:
            raise ValueError("coarse frames must be [record,time,z,x] with matching times")
        records, count, height, width = coarse.shape
        if source_parameters.shape != (records, 5) or mapping.shape != (records,):
            raise ValueError("source or medium mapping shape does not match dense frames")
        full_size = medium.pyramid[0].shape[-2:]
        if full_size != (height, width):
            raise ValueError("dense grid must match the encoded full-resolution medium grid")
        if cached_spatial is not None:
            # P0 (20260929): record-side cache of the SAME spatial_term computation;
            # consumed only on no-grad paths (see _dense_block_impl guard).
            if torch.is_grad_enabled():
                raise ValueError("cached_spatial is an eval-only (no-grad) fast path")
            if tuple(cached_spatial.shape) != (records, self.multiscale_fuse.weight.shape[0], height, width):
                raise ValueError("cached_spatial shape does not match the dense frames")
            spatial = cached_spatial
        else:
            spatial = self.spatial_term(medium, source, record_to_medium)
        relative_time = time_s - source_parameters[:, None, 3]
        frequency = source_parameters[:, None, 2].expand_as(time_s)
        phase = 2.0 * math.pi * frequency * relative_time
        time_features = torch.stack(
            (
                time_s / self.domain_t_s,
                relative_time / self.domain_t_s,
                frequency / 50.0,
                torch.sin(phase),
                torch.cos(phase),
            ),
            dim=-1,
        )
        scale, bias = self.time_mlp(time_features).chunk(2, dim=-1)
        conditioned = (
            spatial[:, None]
            * (1.0 + scale[:, :, :, None, None])
            + bias[:, :, :, None, None]
        )
        flat_coarse = coarse.reshape(records * count, 1, height, width)
        hidden = conditioned.reshape(records * count, -1, height, width) + self.coarse_lift(flat_coarse)
        if self.fixed_wave_physics is not None:
            hidden = hidden + self.fixed_wave_physics(flat_coarse)
        # Activation checkpointing on the dense spectral blocks bounds peak
        # memory of the wide (40M) decoder.  Each block is a stateless function
        # of its input, so recomputation on the backward pass is exact (no
        # numerical change).
        #
        # `use_reentrant=False` is required under DDP.  The reentrant
        # implementation runs the block again in backward without telling DDP's
        # reducer, so the same parameters are marked ready twice and DDP aborts;
        # that failure was previously read as "checkpointing is incompatible
        # with DDP" and checkpointing was switched off for DDP runs.  The
        # non-reentrant implementation hooks the recomputation into autograd
        # properly and is DDP-safe.
        from torch.utils.checkpoint import checkpoint as _checkpoint

        for block in self.blocks:
            if self.use_checkpoint:
                hidden = _checkpoint(block.forward, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        hidden_f = hidden.float()
        correction = self.output(hidden_f)
        if self.u_path is not None:
            correction = correction + self.u_path(hidden_f)
        if self.reflection_branch is not None:
            if velocity is None:
                raise ValueError("reflection_branch requires the per-record velocity field")
            if velocity.shape != (records, 1, height, width):
                raise ValueError("velocity must be [record, 1, z, x] matching the dense frames")
            velocity_repeated = (
                velocity[:, None]
                .expand(-1, count, -1, -1, -1)
                .reshape(records * count, 1, height, width)
            )
            correction = correction + self.reflection_branch(hidden_f, velocity_repeated)
        correction = correction.reshape(records, count, height, width)
        return coarse + self.correction_scale * correction

    def required_gradient_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        named = tuple(self.named_parameters())
        return {
            "dense_spectral": tuple(
                p for n, p in named
                if n.endswith(("spectral.weight", "spectral.weight_top", "spectral.weight_bottom"))
            ),
            "dense_film": tuple(p for n, p in named if n.startswith("time_mlp")),
            "dense_local": tuple(
                p
                for n, p in named
                if n.startswith("multiscale_fuse")
                or n.startswith("source_projection")
                or n.startswith("map_projection")
                or n.startswith("coarse_lift")
                or n.startswith("output")
                or n == "correction_scale"
            ),
            "dense_upath": tuple(p for n, p in named if n.startswith("u_path.")),
            "dense_reflection": tuple(
                p for n, p in named if n.startswith("reflection_branch.")
            ),
            "dense_fixed_physics": tuple(
                p for n, p in named if n.startswith("fixed_wave_physics.")
            ),
        }


__all__ = ["DenseGridCache", "ReflectionBranch", "TimeConditionedComplexDenseDecoder"]
