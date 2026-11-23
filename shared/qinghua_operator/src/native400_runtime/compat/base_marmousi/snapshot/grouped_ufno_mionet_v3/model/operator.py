"""Phase-aligned complex-FNO multi-input DeepONet point-query operator."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn

from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer

from .dense import DenseGridCache, TimeConditionedComplexDenseDecoder
from .coarse_channel import gather_coarse
from .family_gate import family_gate as family_gate_tensor, gate_view, resolve_ungated
from .features import PhaseAlignedCoordinateEncoder
from .fusion import PhaseAlignedMIONetFusion, TravelTimeBranch
from .mirror_travel_time import mirror_phase_features, single_surface_mirror_travel_time
from .medium import ComplexMediumEncoder, MediumEncoding, sample_medium_pyramid
from .phase_features import multi_arrival_phase_features, phase_feature_count
from .source import SourceEncoderV3, SourceEncoding
from .travel_time import RayTravelTime, dense_query_coordinates, straight_ray_travel_time


@dataclass(frozen=True)
class EncodedMediumState:
    velocity_mps: torch.Tensor
    encoding: MediumEncoding


@dataclass(frozen=True)
class PreparedV3State:
    medium: EncodedMediumState
    source_parameters: torch.Tensor
    source_normalized: torch.Tensor
    source_map: torch.Tensor
    source_encoding: SourceEncoding
    record_to_medium: torch.Tensor
    normalizer: PhysicalNormalizer
    anchor_time_s: torch.Tensor | None = None
    # P2 multi-arrival phase maps: [record, arrivals, z, x] travel times (inf =
    # absent) and [record, arrivals] reflection proxies.  None when phase_mode="off".
    phase_T: torch.Tensor | None = None
    phase_r: torch.Tensor | None = None
    # Coarse-solver channel (coarse_channel.py, 20260926): the encoded coarse field
    # ([T,H,W] or [record,T,H,W], NORMALIZED pressure, the same units as the fusion
    # output) plus its frame spacing in seconds.  Both None unless the caller binds a
    # coarse cache; a model built without the channel ignores them.
    coarse_field: torch.Tensor | None = None
    coarse_dt_s: float | None = None
    # Family gate (family_gate.py, 20260926): [record] float32, 1.0 = branch open,
    # exactly 0.0 = branch closed.  None (the default) means "no gate", so every
    # pre-existing caller keeps the ungated path bitwise.  The labels themselves
    # are kept only for the audit trail written into the run metrics.
    family_gate: torch.Tensor | None = None
    family_labels: tuple[str, ...] | None = None
    # Born scattering tables (scatter_binding.py, 20260926): the scatterer leg
    # T(s->y) at the scatterer lattice, the query leg T(y->x), and the scatterer
    # lattice geometry.  All None unless the caller binds a cache; a model built
    # without the scattering head ignores them.
    scatter_T_scatterer: torch.Tensor | None = None       # [record, g, g]
    scatter_T_query: torch.Tensor | None = None           # [record, g, g, z, x]
    scatter_geometry: tuple[float, float] | None = None    # (dx_m, dz_m) of the query grid


def free_surface_factor(
    z_m: torch.Tensor,
    *,
    transition_m: float = 20.0,
    mode: str = "legacy_tanh20",
    blend: torch.Tensor | float | None = None,
) -> torch.Tensor:
    if mode == "exact_row":
        # UL31: hard-zero ONLY the z=0 surface; every z>0 sample passes at 1.0
        # exactly (z=10 m -> 1, z=20 m -> 1), matching the generator's row0=0.
        return (z_m > 0).to(z_m.dtype)
    if mode == "ramp_exact_row":
        # Boundary hotfix (20260929): a function-continuous migration
        # legacy -> exact_row.  An instant switch on the same UL30@1250
        # checkpoint jumped uniform full relL2 0.01844 -> 0.16405 (z=20
        # receiver 0.02589 -> 0.72822), so the factor is blended:
        #   ramp = legacy + alpha * (exact - legacy),  alpha clamped to [0,1].
        # The endpoints are BITWISE the legacy / exact_row factors (torch.where
        # selects the unblended tensor there, so no float round-off leaks in).
        if blend is None:
            raise ValueError("ramp_exact_row requires the current blend value")
        if transition_m <= 0:
            raise ValueError("free-surface transition must be positive")
        alpha = torch.clamp(
            torch.as_tensor(blend, dtype=z_m.dtype, device=z_m.device), 0.0, 1.0
        )
        legacy = torch.tanh(torch.clamp_min(z_m, 0.0) / transition_m).square()
        exact = (z_m > 0).to(z_m.dtype)
        ramp = legacy + alpha * (exact - legacy)
        ramp = torch.where(alpha >= 1.0, exact, ramp)
        return torch.where(alpha <= 0.0, legacy, ramp)
    if mode != "legacy_tanh20":
        raise ValueError(
            "free_surface_mode must be 'legacy_tanh20', 'exact_row' or 'ramp_exact_row'"
        )
    if transition_m <= 0:
        raise ValueError("free-surface transition must be positive")
    return torch.tanh(torch.clamp_min(z_m, 0.0) / transition_m).square()


class PhaseAlignedComplexFNOMIONet(nn.Module):
    def __init__(
        self,
        *,
        width: int = 64,
        rank: int = 128,
        spectral_rank: int = 40,
        modes: tuple[int, ...] = (20, 16, 12, 8),
        dense_modes: tuple[int, ...] = (20, 16),
        dense_time_block: int = 4,
        dense_equivariant: bool = False,
        dense_radial_cutoff: int = 0,
        # V18-A speed branch (20260928): multi-dilation local path inside the
        # dense decoder's spectral blocks only.  Default off is bitwise inert.
        local_multi_dilation: bool = False,
        # v18.5 branch-heavy (20260929): U-shaped medium-pyramid refinement with a
        # zero-init head on the full-resolution level.  Default off is bitwise inert.
        medium_u_enhance: bool = False,
        u_path_enabled: bool = False,
        u_path_channels: tuple[int, int, int] = (16, 32, 48),
        u_path_output_scale: float = 0.01,
        fixed_wave_physics: bool = False,
        heads: int = 4,
        token_grid: int = 8,
        position_bands: int = 4,
        fourier_bands: int = 6,
        gabor_scales_s: Sequence[float] = (0.01, 0.025, 0.05, 0.1),
        ray_samples: int = 12,
        ic_frames: int = 0,
        domain_x_m: float = 2000.0,
        domain_z_m: float = 2000.0,
        domain_t_s: float = 1.0,
        dense_checkpoint: bool = True,
        dense_outer_checkpoint: bool = False,
        carrier_mode: str = "off",
        local_unet_channels: tuple[int, ...] = (),
        local_unet_rank: int = 24,
        phase_mode: str = "off",
        phase_interfaces: int = 3,
        phase_harmonics: int = 2,
        arrival_head: bool = False,
        arrival_head_hidden: int = 128,
        arrival_head_harmonics: int = 2,
        arrival_head_arrivals: int = 8,
        # Coarse-solver input channel (coarse_channel.py, 20260926); default off.
        coarse_channel: bool = False,
        coarse_channel_hidden: int = 0,
        coarse_channel_gain: float = 1.0,
        # Born single-scattering head (scattering_head.py, 20260926); default off.
        scattering_head: bool = False,
        scatter_grid: int = 51,
        scatter_harmonics: int = 1,
        scatter_channels: int = 16,
        scatter_hidden: int = 32,
        scatter_dilations: tuple[int, ...] = (1, 1),
        # Family gate (family_gate.py, 20260926); empty tuple = every family open.
        phase_gated_families: tuple[str, ...] = (),
        scatter_ungated_families: tuple[str, ...] = ("layered", "marmousi"),
        # UL31 (20260929): reflection branch inside the dense decoder + free-surface
        # output mode.  Defaults are bitwise inert (no parameters, legacy factor).
        reflection_branch: bool = False,
        reflection_channels: int = 32,
        free_surface_mode: str = "legacy_tanh20",
        free_surface_blend_updates: int = 500,
    ) -> None:
        super().__init__()
        if phase_mode not in ("off", "multi_arrival", "multi_arrival_direct"):
            raise ValueError("phase_mode must be 'off', 'multi_arrival' or 'multi_arrival_direct'")
        self.phase_mode = str(phase_mode)
        self.phase_interfaces = int(phase_interfaces)
        self.phase_harmonics = int(phase_harmonics)
        # K primary reflections + K source-side free-surface multiples come from
        # the per-record cache; "multi_arrival_direct" (P2 v2) appends the direct
        # arrival (existing straight-ray travel time, r=+1) and the free-surface
        # mirror arrival (existing single-surface mirror tau, r=-1) with the same
        # encoding, so uniform records no longer see an all-zero channel.
        self.phase_cache_arrivals = 2 * self.phase_interfaces if self.phase_mode != "off" else 0
        self.phase_direct = self.phase_mode == "multi_arrival_direct"
        self.phase_arrivals = self.phase_cache_arrivals + (2 if self.phase_direct else 0)
        self.phase_feature_count = (
            phase_feature_count(self.phase_arrivals, self.phase_harmonics) if self.phase_mode != "off" else 0
        )
        # Multi-arrival additive head (arrival_head.py, 20260925): enumerated
        # Born layer added to the fusion output.  Default off adds no parameters
        # and no computation.  Requires the full arrival set (cache reflections
        # + multiples + direct + mirror), i.e. phase_mode="multi_arrival_direct".
        # Evaluation-only travel-time jitter hook (mechanism ablation): a
        # [record, arrivals] tensor of shifts in seconds, or None; plain
        # attributes, never part of the state_dict.
        self.arrival_head = None
        self.arrival_jitter_s = None
        self.arrival_jitter_scope = "all"
        if arrival_head:
            from .arrival_head import MultiArrivalAdditiveHead

            if not self.phase_direct:
                raise ValueError("arrival_head requires phase_mode='multi_arrival_direct'")
            if int(arrival_head_arrivals) != self.phase_arrivals:
                raise ValueError(
                    f"arrival_head_arrivals={arrival_head_arrivals} but the phase configuration "
                    f"enumerates {self.phase_arrivals} arrivals")
            # Cache order [p1..pK, m1..mK] then the appended direct and mirror arrivals.
            arrival_types = (
                [MultiArrivalAdditiveHead.TYPE_REFLECTION] * self.phase_interfaces
                + [MultiArrivalAdditiveHead.TYPE_MULTIPLE] * self.phase_interfaces
                + [MultiArrivalAdditiveHead.TYPE_DIRECT, MultiArrivalAdditiveHead.TYPE_MIRROR]
            )
            self.arrival_head = MultiArrivalAdditiveHead(
                rank=rank,
                arrival_types=arrival_types,
                hidden=int(arrival_head_hidden),
                harmonics=int(arrival_head_harmonics),
                domain_t_s=float(domain_t_s),
            )
        # Coarse-solver input channel (coarse_channel.py, 20260926): zero-initialised
        # additive term over [own normalized prediction, normalized coarse field],
        # added at the same point as the arrival head.  Default off adds no parameters
        # and no computation.  The coarse field itself rides on PreparedV3State (a
        # [T,H,W] or [R,T,H,W] normalized tensor) and is bound by the caller, because
        # it is a per-record data product, not a model constant.
        self.coarse_channel = None
        if coarse_channel:
            from .coarse_channel import CoarseFieldChannel

            self.coarse_channel = CoarseFieldChannel(
                hidden=int(coarse_channel_hidden),
                gain=float(coarse_channel_gain),
            )
        self.domain_x_m = float(domain_x_m)
        self.domain_z_m = float(domain_z_m)
        self.domain_t_s = float(domain_t_s)
        # Born single-scattering head (scattering_head.py, 20260926).  Default off
        # adds no parameters, no computation and no state_dict keys.  The head is a
        # pure function of (velocity, source, query) with its travel tables supplied
        # from outside, so nothing here reads a cache: the tables ride on
        # PreparedV3State and a model built without the head ignores them entirely.
        self.scatter_grid = int(scatter_grid)
        self.scatter_harmonics = int(scatter_harmonics)
        self.scattering_head = None
        if scattering_head:
            from .scattering_head import ScatteringIntegralHead

            self.scattering_head = ScatteringIntegralHead(
                channels=int(scatter_channels),
                hidden=int(scatter_hidden),
                domain_t_s=self.domain_t_s,
                reflectivity_dilations=tuple(scatter_dilations),
            )
        # Family gate (family_gate.py, 20260926): the ungated-family list is a model
        # constant (it decides nothing per-batch, only how the caller's [record] gate
        # tensor is interpreted), so it is stored here and the tensor itself rides on
        # PreparedV3State.  Empty phase_gated_families = the phase channel is ungated
        # for every family, which is the behaviour every pre-existing arm had.
        self.phase_gated_families = tuple(phase_gated_families)
        self.scatter_ungated_families = resolve_ungated(scatter_ungated_families)
        self.ray_samples = int(ray_samples)
        self.ic_frames = int(ic_frames)
        self.carrier_mode = str(carrier_mode)
        if free_surface_mode not in ("legacy_tanh20", "exact_row", "ramp_exact_row"):
            raise ValueError(
                "free_surface_mode must be 'legacy_tanh20', 'exact_row' or 'ramp_exact_row'"
            )
        self.free_surface_mode = str(free_surface_mode)
        # Boundary ramp (20260929): the blend rides in a PERSISTENT buffer so the
        # final checkpoint carries its trained-with factor into inference; the
        # trainer advances it via set_free_surface_blend(completed).  Registered
        # only under ramp_exact_row, so legacy/exact state_dicts are unchanged.
        # Buffers consume no RNG, so parameter initialisation stays bitwise.
        self.free_surface_blend_updates = int(free_surface_blend_updates)
        if self.free_surface_mode == "ramp_exact_row":
            if self.free_surface_blend_updates <= 0:
                raise ValueError("free_surface_blend_updates must be positive")
            self.register_buffer(
                "free_surface_blend", torch.tensor(0.0), persistent=True
            )
        self.dense_outer_checkpoint = bool(dense_outer_checkpoint)
        self.medium_encoder = ComplexMediumEncoder(
            width=width,
            rank=rank,
            spectral_rank=spectral_rank,
            modes=tuple(modes),
            token_grid=token_grid,
            position_bands=position_bands,
            u_enhance=bool(medium_u_enhance),
        )
        self.source_encoder = SourceEncoderV3(width=width, rank=rank, ic_frames=self.ic_frames)
        self.coordinate_encoder = PhaseAlignedCoordinateEncoder(
            width=width,
            rank=rank,
            fourier_bands=fourier_bands,
            gabor_scales_s=gabor_scales_s,
        )
        self.travel_branch = TravelTimeBranch(
            width=width, rank=rank, carrier_mode=self.carrier_mode,
            phase_features=self.phase_feature_count,
        )
        self.fusion = PhaseAlignedMIONetFusion(
            width=width,
            rank=rank,
            pyramid_levels=len(modes),
            heads=heads,
        )
        self.dense_decoder = TimeConditionedComplexDenseDecoder(
            width=width,
            spectral_rank=spectral_rank,
            pyramid_levels=len(modes),
            dense_modes=tuple(dense_modes),
            time_block=dense_time_block,
            domain_t_s=self.domain_t_s,
            use_checkpoint=dense_checkpoint,
            equivariant=bool(dense_equivariant),
            radial_cutoff=int(dense_radial_cutoff),
            local_multi_dilation=bool(local_multi_dilation),
            u_path_enabled=bool(u_path_enabled),
            u_path_channels=tuple(int(v) for v in u_path_channels),
            u_path_output_scale=float(u_path_output_scale),
            fixed_wave_physics=bool(fixed_wave_physics),
            reflection_branch=bool(reflection_branch),
            reflection_channels=int(reflection_channels),
        )
        # Optional local spatial path.  Empty channels tuple (the default) keeps
        # every pre-existing state_dict key untouched, so legacy checkpoints and
        # their Adam state migrate unchanged.
        self.local_unet = None
        if local_unet_channels:
            from .local_unet import LocalSpatialUNet

            self.local_unet = LocalSpatialUNet(
                channels=tuple(int(c) for c in local_unet_channels),
                spectral_rank=int(local_unet_rank),
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _free_surface_factor(self, z_m: torch.Tensor) -> torch.Tensor:
        # Single chokepoint for the output factor: the point query and the dense
        # renderer both call this, so they always share the SAME current blend.
        return free_surface_factor(
            z_m,
            mode=self.free_surface_mode,
            blend=getattr(self, "free_surface_blend", None),
        )

    def set_free_surface_blend(self, completed_updates: int) -> None:
        """Advance the ramp: alpha = min(completed/free_surface_blend_updates, 1).

        A no-op under legacy_tanh20/exact_row.  The trainer calls this before
        EVERY optimizer-step forward and once after init/resume, so the blend is
        a pure function of ``updates_completed`` (deterministic on resume); the
        persistent buffer carries the final value into inference checkpoints.
        """
        if self.free_surface_mode != "ramp_exact_row":
            return
        alpha = min(
            max(float(completed_updates), 0.0) / float(self.free_surface_blend_updates),
            1.0,
        )
        with torch.no_grad():
            self.free_surface_blend.fill_(alpha)

    def encode_medium(
        self,
        velocity_mps: torch.Tensor,
        normalizer: PhysicalNormalizer,
    ) -> EncodedMediumState:
        velocity = torch.as_tensor(velocity_mps, dtype=torch.float32, device=self.device)
        if velocity.ndim != 4 or velocity.shape[1] != 1:
            raise ValueError("velocity_mps must be [medium,1,z,x]")
        encoding = self.medium_encoder(normalizer.encode_velocity(velocity))
        return EncodedMediumState(velocity_mps=velocity, encoding=encoding)

    def _mapping(
        self,
        medium_count: int,
        record_count: int,
        record_to_medium: torch.Tensor | None,
    ) -> torch.Tensor:
        if record_to_medium is None:
            if medium_count == 1:
                return torch.zeros(record_count, dtype=torch.long, device=self.device)
            if medium_count == record_count:
                return torch.arange(record_count, dtype=torch.long, device=self.device)
            raise ValueError("record_to_medium is required for grouped source records")
        mapping = torch.as_tensor(record_to_medium, dtype=torch.long, device=self.device)
        if mapping.shape != (record_count,) or torch.any(mapping < 0) or torch.any(mapping >= medium_count):
            raise ValueError("record_to_medium contains invalid indices")
        return mapping

    def prepare_sources(
        self,
        medium: EncodedMediumState,
        source_parameters: torch.Tensor,
        source_map: torch.Tensor,
        normalizer: PhysicalNormalizer,
        *,
        record_to_medium: torch.Tensor | None = None,
        ic_snapshots_physical: torch.Tensor | None = None,
        anchor_time_s: torch.Tensor | None = None,
        phase_maps: tuple[torch.Tensor, torch.Tensor] | None = None,
        coarse_field: torch.Tensor | None = None,
        coarse_dt_s: float | None = None,
        family_gate: torch.Tensor | None = None,
        family_labels: "tuple[str, ...] | None" = None,
        scatter_tables: "tuple[torch.Tensor, torch.Tensor, tuple[float, float]] | None" = None,
    ) -> PreparedV3State:
        source = torch.as_tensor(source_parameters, dtype=torch.float32, device=self.device)
        source_map_tensor = torch.as_tensor(source_map, dtype=torch.float32, device=self.device)
        if source.ndim != 2 or source.shape[-1] != 5:
            raise ValueError("source_parameters must be [record,5]")
        # Anchor-time contract (L1): the anchor is the physical time of the
        # first IC frame.  Legacy callers that never pass it fall back to the
        # source onset t0 (within one stored frame of the exact onset time);
        # anchored training/inference must pass the exact frame time.
        if anchor_time_s is None:
            anchor = source[:, 3].clone()
        else:
            anchor = torch.as_tensor(
                anchor_time_s, dtype=torch.float32, device=self.device
            )
            if anchor.shape != (source.shape[0],):
                raise ValueError("anchor_time_s must be [record]")
            if not torch.isfinite(anchor).all() or torch.any(anchor < 0) or torch.any(anchor > self.domain_t_s):
                raise ValueError("anchor_time_s must lie inside the physical time domain")
        mapping = self._mapping(
            medium.velocity_mps.shape[0],
            source.shape[0],
            record_to_medium,
        )
        source_normalized = normalizer.encode_source(source)
        ic_normalized: torch.Tensor | None = None
        if self.ic_frames:
            if ic_snapshots_physical is None:
                raise ValueError("ic_snapshots_physical is required when ic_frames is enabled")
            ic_physical = torch.as_tensor(
                ic_snapshots_physical, dtype=torch.float32, device=self.device
            )
            ic_normalized = normalizer.encode_pressure(ic_physical, source[:, 4])
        elif ic_snapshots_physical is not None:
            raise ValueError("ic_snapshots_physical supplied but the model has ic_frames=0")
        source_encoding = self.source_encoder(
            source_normalized,
            source_map_tensor,
            medium.encoding,
            mapping,
            ic_snapshots=ic_normalized,
            anchor_normalized=(anchor / self.domain_t_s)[:, None],
        )
        phase_T = phase_r = None
        if self.phase_mode != "off":
            if phase_maps is None:
                raise ValueError("phase_mode is enabled: prepare_sources requires phase_maps=(T, r)")
            phase_T = torch.as_tensor(phase_maps[0], dtype=torch.float32, device=self.device)
            phase_r = torch.as_tensor(phase_maps[1], dtype=torch.float32, device=self.device)
            full_size = medium.encoding.pyramid[0].shape[-2:]
            if phase_T.shape != (source.shape[0], self.phase_cache_arrivals, *full_size):
                raise ValueError(
                    f"phase_maps T must be [record,{self.phase_cache_arrivals},z,x]={(source.shape[0], self.phase_cache_arrivals, *full_size)}, got {tuple(phase_T.shape)}")
            if phase_r.shape != (source.shape[0], self.phase_cache_arrivals):
                raise ValueError("phase_maps r must be [record, arrivals]")
            if torch.isnan(phase_T).any() or torch.any(phase_T[torch.isfinite(phase_T)] < 0) or not torch.isfinite(phase_r).all():
                raise ValueError("phase_maps must be nonnegative (inf = absent) with finite r")
        scatter_T_scatterer = scatter_T_query = scatter_geometry = None
        if self.scattering_head is not None:
            if scatter_tables is None:
                raise ValueError(
                    "scattering_head is enabled: prepare_sources requires scatter_tables="
                    "(T_source_scatterer, T_scatterer_query, (dx_m, dz_m))")
            scatter_T_scatterer, scatter_T_query, scatter_geometry = scatter_tables
            scatter_T_scatterer = torch.as_tensor(
                scatter_T_scatterer, dtype=torch.float32, device=self.device)
            scatter_T_query = torch.as_tensor(
                scatter_T_query, dtype=torch.float32, device=self.device)
            expected_source = (source.shape[0], self.scatter_grid, self.scatter_grid)
            if scatter_T_scatterer.shape != expected_source:
                raise ValueError(
                    f"scatter T(s->y) must be [record,{self.scatter_grid},{self.scatter_grid}]="
                    f"{expected_source}, got {tuple(scatter_T_scatterer.shape)}")
            if scatter_T_query.ndim != 5 or scatter_T_query.shape[:3] != expected_source:
                raise ValueError(
                    "scatter T(y->x) must be [record,g,g,z,x] with the same record/g block as "
                    f"T(s->y); got {tuple(scatter_T_query.shape)}")
            if len(scatter_geometry) != 2 or min(scatter_geometry) <= 0:
                raise ValueError("scatter geometry must be a positive (dx_m, dz_m) pair")
            scatter_geometry = (float(scatter_geometry[0]), float(scatter_geometry[1]))
            if torch.isnan(scatter_T_scatterer).any() or torch.isnan(scatter_T_query).any():
                raise ValueError("scatter tables must use inf (not nan) for absent cells")
        gate_tensor = None
        gate_labels = None
        if family_gate is not None:
            gate_tensor = torch.as_tensor(family_gate, dtype=torch.float32, device=self.device)
            if gate_tensor.shape != (source.shape[0],):
                raise ValueError(
                    f"family_gate must be [record]={source.shape[0]}, got {tuple(gate_tensor.shape)}")
            if not torch.isfinite(gate_tensor).all() or torch.any(gate_tensor < 0):
                raise ValueError("family_gate must be finite and nonnegative")
        if family_labels is not None:
            gate_labels = tuple(str(v) for v in family_labels)
            if len(gate_labels) != source.shape[0]:
                raise ValueError(
                    f"family_labels must have {source.shape[0]} entries, got {len(gate_labels)}")
        # Family gate, phase side (family_gate.py, 20260926).  The phase channel is
        # closed for a record whose family is listed in model.phase_gated_families;
        # an empty list means "no phase gate", which is exactly the pre-20260926
        # behaviour and keeps every older caller bitwise.  The gate is derived from
        # the same `family_labels` the scatter gate uses, so the two branches can
        # never disagree about a record's family.
        phase_gate = None
        if self.phase_gated_families and gate_labels is not None:
            open_families = tuple(
                f for f in dict.fromkeys(gate_labels) if f not in set(self.phase_gated_families))
            phase_gate = family_gate_tensor(
                gate_labels, ungated=open_families, device=self.device)
        if family_gate is not None and gate_labels is not None and self.phase_gated_families:
            # The caller-supplied [record] gate and the label-derived phase gate must
            # agree on every record; a silent disagreement would mean the audit trail
            # (`family_labels`) misdescribes the forward.
            supplied = torch.as_tensor(family_gate, dtype=torch.float32, device=self.device)
            if not torch.equal((supplied != 0), (phase_gate != 0)):
                raise ValueError(
                    "family_gate disagrees with phase_gated_families for at least one record; "
                    f"labels={gate_labels}, phase_gated_families={self.phase_gated_families}")
        return PreparedV3State(
            medium=medium,
            source_parameters=source,
            source_normalized=source_normalized,
            source_map=source_map_tensor,
            source_encoding=source_encoding,
            record_to_medium=mapping,
            normalizer=normalizer,
            anchor_time_s=anchor,
            phase_T=phase_T,
            phase_r=phase_r,
            coarse_field=None if coarse_field is None else torch.as_tensor(
                coarse_field, dtype=torch.float32, device=self.device),
            coarse_dt_s=None if coarse_dt_s is None else float(coarse_dt_s),
            family_gate=gate_tensor,
            family_labels=gate_labels,
            scatter_T_scatterer=scatter_T_scatterer,
            scatter_T_query=scatter_T_query,
            scatter_geometry=scatter_geometry,
        )

    def _query_chunk(
        self,
        prepared: PreparedV3State,
        coords_xyz_m: torch.Tensor,
    ) -> torch.Tensor:
        coords = torch.as_tensor(coords_xyz_m, dtype=torch.float32, device=self.device)
        travel = straight_ray_travel_time(
            prepared.medium.velocity_mps,
            prepared.source_parameters[:, :2],
            coords[..., :2],
            x_extent_m=(0.0, self.domain_x_m),
            z_extent_m=(0.0, self.domain_z_m),
            record_to_medium=prepared.record_to_medium,
            samples=self.ray_samples,
        )
        trunk_rank, bundle = self.coordinate_encoder(
            coords,
            prepared.source_parameters,
            travel,
            domain_x_m=self.domain_x_m,
            domain_z_m=self.domain_z_m,
            domain_t_s=self.domain_t_s,
            anchor_time_s=prepared.anchor_time_s,
        )
        phase = None
        sampled = phase_r = None
        if self.phase_mode != "off":
            grid = torch.stack(
                (2.0 * coords[..., 0] / self.domain_x_m - 1.0, 2.0 * coords[..., 1] / self.domain_z_m - 1.0), dim=-1
            )[:, None, :, :]
            sampled = torch.nn.functional.grid_sample(
                prepared.phase_T, grid, mode="nearest", padding_mode="border", align_corners=True
            ).squeeze(2)                                                     # [record, arrivals, query]
            phase_r = prepared.phase_r
            if self.phase_direct:
                mirror = single_surface_mirror_travel_time(
                    prepared.medium.velocity_mps, prepared.source_parameters[:, :2], coords[..., :2],
                    x_extent_m=(0.0, self.domain_x_m), z_extent_m=(0.0, self.domain_z_m),
                    record_to_medium=prepared.record_to_medium,
                ).to(sampled.dtype)
                sampled, phase_r = self._append_direct_arrivals(sampled, phase_r, travel.seconds, mirror)
            sampled = self._jitter_arrivals(sampled, "all")
            phase = multi_arrival_phase_features(
                sampled, phase_r, coords[..., 2], prepared.source_parameters[:, 3],
                prepared.source_parameters[:, 2], harmonics=self.phase_harmonics, domain_t_s=self.domain_t_s,
            )
            # Family gate, phase side: for a CLOSED record every one of the
            # phase_feature_count channels becomes exactly 0.0, so the travel branch
            # sees the same all-zero phase block it sees for a model built with
            # phase_mode="off".  torch.where returns the literal zero tensor, so this
            # is bitwise 0, not "small": no carrier structure can survive the gate.
            if phase_gate is not None:
                phase = gate_view(phase, phase_gate)
        travel_rank = self.travel_branch(
            travel,
            bundle,
            domain_t_s=self.domain_t_s,
            domain_diagonal_m=math.sqrt(self.domain_x_m**2 + self.domain_z_m**2),
            carrier_features=self._carrier_features(
                prepared, coords[..., :2], coords[..., 2]),
            phase_features=phase,
        )
        coords_normalized = torch.stack(
            (coords[..., 0] / self.domain_x_m, coords[..., 1] / self.domain_z_m),
            dim=-1,
        )
        fused = self.fusion(
            prepared.medium.encoding,
            prepared.source_encoding,
            coords_normalized,
            prepared.record_to_medium,
            travel_rank,
            trunk_rank,
        )
        pressure = fused.normalized_pressure
        if self.arrival_head is not None:
            pressure = pressure + self._arrival_head_term(prepared, fused.ranks, sampled, phase_r, coords[..., 2])
        if self._scatter_active(prepared):
            # Born single-scattering term, added at the same point as the arrival head
            # so it enters the same coarse frame.  The family gate is applied to the
            # TERM (not to the input tables), so a closed record contributes exactly
            # 0.0 while the head's parameters still exist in the state_dict.  A batch
            # with every record closed skips the sum entirely (exactly 0 either way).
            term = self._scatter_term(prepared, coords, coords[..., 2])
            pressure = pressure + gate_view(term, self._scatter_gate(prepared))
        return pressure * self._free_surface_factor(coords[..., 1])

    def query_normalized(
        self,
        prepared: PreparedV3State,
        coords_xyz_m: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        coords = torch.as_tensor(coords_xyz_m, dtype=torch.float32, device=self.device)
        if coords.ndim != 3 or coords.shape[0] != prepared.source_parameters.shape[0] or coords.shape[-1] != 3:
            raise ValueError("coords_xyz_m must be [record,query,3]")
        step = coords.shape[1] if chunk_size is None else int(chunk_size)
        if step <= 0:
            raise ValueError("chunk_size must be positive")
        return torch.cat(
            [self._query_chunk(prepared, coords[:, start : start + step]) for start in range(0, coords.shape[1], step)],
            dim=1,
        )

    def query_pressure(
        self,
        prepared: PreparedV3State,
        coords_xyz_m: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        normalized = self.query_normalized(prepared, coords_xyz_m, chunk_size=chunk_size)
        return prepared.normalizer.decode_pressure(
            normalized.float(),
            prepared.source_parameters[:, 4],
        )

    def prepare_dense_grid(
        self,
        prepared: PreparedV3State,
        *,
        x_m: torch.Tensor,
        z_m: torch.Tensor,
        travel_time_s: torch.Tensor | None = None,
    ) -> DenseGridCache:
        x = torch.as_tensor(x_m, dtype=torch.float32, device=self.device)
        z = torch.as_tensor(z_m, dtype=torch.float32, device=self.device)
        full_size = prepared.medium.encoding.pyramid[0].shape[-2:]
        if x.ndim != 1 or z.ndim != 1 or (len(z), len(x)) != full_size:
            raise ValueError("dense x/z axes must match the encoded full-resolution grid")
        coords = dense_query_coordinates(x, z, records=prepared.source_parameters.shape[0])
        travel = straight_ray_travel_time(
            prepared.medium.velocity_mps,
            prepared.source_parameters[:, :2],
            coords,
            x_extent_m=(0.0, self.domain_x_m),
            z_extent_m=(0.0, self.domain_z_m),
            record_to_medium=prepared.record_to_medium,
            samples=self.ray_samples,
        )
        if travel_time_s is not None:
            cached = torch.as_tensor(
                travel_time_s, dtype=torch.float32, device=self.device
            )
            expected = (prepared.source_parameters.shape[0], len(z), len(x))
            if cached.shape != expected:
                raise ValueError(
                    f"cached dense travel time must have shape {expected}, got {tuple(cached.shape)}"
                )
            if not torch.isfinite(cached).all() or torch.any(cached < 0.0):
                raise ValueError("cached dense travel time must be finite and nonnegative")
            seconds = cached.flatten(1)
            distance = travel.distance_m
            positive_distance = distance > 1.0e-8
            positive_time = seconds > 1.0e-8
            mean_slowness = torch.where(
                positive_distance,
                seconds / distance.clamp_min(1.0e-8),
                travel.mean_slowness_s_per_m,
            )
            path_velocity = torch.where(
                positive_time,
                distance / seconds.clamp_min(1.0e-8),
                travel.endpoint_velocity_mps,
            )
            travel = RayTravelTime(
                seconds=seconds,
                distance_m=distance,
                path_velocity_mps=path_velocity,
                endpoint_velocity_mps=travel.endpoint_velocity_mps,
                mean_slowness_s_per_m=mean_slowness,
            )
        normalized = torch.stack(
            (coords[..., 0] / self.domain_x_m, coords[..., 1] / self.domain_z_m), dim=-1
        )
        mirror_tau = None
        if self.carrier_mode != "off" or self.phase_direct:
            mirror_tau = single_surface_mirror_travel_time(
                prepared.medium.velocity_mps,
                prepared.source_parameters[:, :2],
                coords,
                x_extent_m=(0.0, self.domain_x_m),
                z_extent_m=(0.0, self.domain_z_m),
                record_to_medium=prepared.record_to_medium,
            ).float()
        phase_T = phase_r = None
        if self.phase_mode != "off":
            if prepared.phase_T is None:
                raise ValueError("phase_mode is enabled but the prepared state carries no phase maps")
            if tuple(prepared.phase_T.shape[-2:]) != (len(z), len(x)):
                raise ValueError("phase maps must be defined on the dense grid")
            phase_T = prepared.phase_T.flatten(2)                            # [record, arrivals, points]
            phase_r = prepared.phase_r
        # P0 record-side caches (v18.5, 20260929): the pyramid sampling, the two
        # fusion local projections and the decoder spatial term depend only on
        # (record, point) yet were recomputed for every frame.  Build them once per
        # grid, ONLY on no-grad paths (evaluation/deployment); with grad enabled
        # (training) nothing is built or consumed, keeping training bitwise intact.
        fusion_vrl = fusion_lh = decoder_spatial = None
        if not torch.is_grad_enabled():
            local = sample_medium_pyramid(
                prepared.medium.encoding, normalized, prepared.record_to_medium)
            fusion_vrl = self.fusion.velocity_local_to_rank(local)
            fusion_lh = self.fusion.local_to_hidden(local)
            decoder_spatial = self.dense_decoder.spatial_term(
                prepared.medium.encoding, prepared.source_encoding, prepared.record_to_medium)
        return DenseGridCache(
            x_m=x,
            z_m=z,
            coords_xy_m=coords,
            coords_xy_normalized=normalized,
            travel=travel,
            height=len(z),
            width=len(x),
            mirror_tau_s=mirror_tau,
            phase_T=phase_T,
            phase_r=phase_r,
            fusion_velocity_rank_local=fusion_vrl,
            fusion_local_hidden=fusion_lh,
            decoder_spatial=decoder_spatial,
        )

    @staticmethod
    def _append_direct_arrivals(
        phase_T: torch.Tensor, phase_r: torch.Tensor, direct_s: torch.Tensor, mirror_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[record, A, points] + direct/mirror [record, points] -> [record, A+2, points]; r = +1 / -1."""
        T = torch.cat((phase_T, direct_s[:, None, :].to(phase_T.dtype), mirror_s[:, None, :].to(phase_T.dtype)), dim=1)
        ones = torch.ones(phase_r.shape[0], 1, dtype=phase_r.dtype, device=phase_r.device)
        r = torch.cat((phase_r, ones, -ones), dim=1)
        return T, r

    def _jitter_arrivals(self, travel_s: torch.Tensor, scope: str) -> torch.Tensor:
        """Evaluation-only mechanism ablation: shift every finite arrival time by
        arrival_jitter_s[record, arrival] seconds (clamped at 0).  scope "all"
        perturbs the enumerated arrival set feeding both the P2 phase channel and
        the head; "head" perturbs only the head's copy.  The parent's direct
        travel-time branch is never touched.  No-op (returns the input tensor
        object) unless arrival_jitter_s is set, so the default path is unchanged."""
        jitter = self.arrival_jitter_s
        if jitter is None or self.arrival_jitter_scope != scope:
            return travel_s
        shift = torch.as_tensor(jitter, dtype=travel_s.dtype, device=travel_s.device)
        if shift.shape != travel_s.shape[:2]:
            raise ValueError(f"arrival_jitter_s must be [record, arrivals]={tuple(travel_s.shape[:2])}, got {tuple(shift.shape)}")
        finite = torch.isfinite(travel_s)
        return torch.where(finite, torch.clamp_min(travel_s + shift[:, :, None], 0.0), travel_s)

    def _scatter_gate(self, prepared: PreparedV3State) -> torch.Tensor | None:
        """[record] gate for the scattering head, or None when the head is inert.

        The ungated set is a MODEL constant (spec), the labels are per-batch data.
        Only families listed in `scatter_ungated_families` see the head at all; an
        unlisted or unknown family is closed by construction in family_gate.py.

        Returns None (the "no gate, no term" path) when EVERY record in the batch is
        closed.  That is what makes a uniform-only draw cheap: the Born sum is skipped
        outright instead of being computed and then multiplied by zero.  It changes
        nothing numerically -- a fully gated batch contributes exactly 0 either way --
        but it removes a ~1.1 s/step cost from 3 of the 9 pool cells.
        """
        if self.scattering_head is None or prepared.family_labels is None:
            return None
        gate = family_gate_tensor(
            prepared.family_labels,
            ungated=self.scatter_ungated_families,
            device=self.device,
        )
        if not bool(gate.any()):
            return None
        return gate

    def _scatter_active(self, prepared: PreparedV3State) -> bool:
        """True when at least one record in the batch has the head open."""
        if self.scattering_head is None or prepared.scatter_T_scatterer is None:
            return False
        if prepared.family_labels is None:
            return True
        return any(str(f) in set(self.scatter_ungated_families)
                   for f in prepared.family_labels)

    def _scatter_term(
        self,
        prepared: PreparedV3State,
        coords_xyz_m: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        """Born single-scattering term [record, points] in NORMALIZED pressure units.

        The head works in physical travel times (seconds) but must return the same
        units as the fusion output it is added to.  The fusion output is the
        NORMALIZED pressure; the scattering head's own output is a raw amplitude
        sum over the lattice (physical, arbitrary scale), so the term is mapped
        through the record's own pressure normalisation:

            u_scat_normalized = u_scat_physical / (pressure_scale * source_amplitude)

        That is the same division `normalizer.encode_pressure` applies to the
        labels, so the head competes with the parent on the same scale rather than
        on an arbitrary one.
        """
        if prepared.scatter_T_scatterer is None or prepared.scatter_T_query is None:
            raise ValueError(
                "scattering_head is enabled but the prepared state carries no scatter tables")
        head = self.scattering_head
        # Scatterer lattice: the cached tables are on the model's own g x g grid.
        # The reflectivity/amplitude CNN runs on the normalised velocity resampled
        # to that same g x g grid, so R(y) and A(y) are defined exactly where the
        # travel times are.
        velocity = prepared.medium.velocity_mps                        # [record,1,z,x]
        scatterer_velocity = torch.nn.functional.interpolate(
            velocity, size=(self.scatter_grid, self.scatter_grid),
            mode="bilinear", align_corners=True)
        scatterer_amplitude = head.scatterer_fields(scatterer_velocity)  # [record,2,g,g]
        records, queries = time_s.shape
        # Query leg: the cache holds T(y -> x) on the full query grid [record,g,g,z,x];
        # this path evaluates at arbitrary query points, so the column is gathered by
        # bilinear interpolation of the cached field at the normalised query position.
        full = prepared.scatter_T_query.shape[-2:]
        grid = torch.stack((
            2.0 * coords_xyz_m[..., 0] / self.domain_x_m - 1.0,
            2.0 * coords_xyz_m[..., 1] / self.domain_z_m - 1.0,
        ), dim=-1)                                                     # [record,queries,2]
        flat = prepared.scatter_T_query.reshape(records, self.scatter_grid * self.scatter_grid, *full)
        sampled = []
        for r in range(records):
            row = torch.nn.functional.grid_sample(
                flat[r][None], grid[r][None, :, None, :],
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            sampled.append(row.reshape(self.scatter_grid, self.scatter_grid, -1))
        query_leg = torch.stack(sampled, dim=0)                        # [record,g,g,queries]
        query_leg = query_leg.permute(0, 3, 1, 2)                      # [record,queries,g,g]
        dx_m, dz_m = prepared.scatter_geometry if prepared.scatter_geometry else (
            self.domain_x_m / max(full[1] - 1, 1), self.domain_z_m / max(full[0] - 1, 1))
        term = head(
            scatterer_amplitude,
            prepared.scatter_T_scatterer,
            query_leg,
            time_s,
            prepared.source_parameters[:, 3],
            prepared.source_parameters[:, 2],
            dx_m, dz_m,
        )
        scale = (prepared.normalizer.metadata.pressure_scale_pa
                 * prepared.source_parameters[:, 4])[:, None]
        return term / scale

    def _arrival_head_term(
        self,
        prepared: PreparedV3State,
        ranks: tuple[torch.Tensor, ...] | None,
        travel_s: torch.Tensor | None,
        reflection: torch.Tensor | None,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        """Additive multi-arrival term [record, points] from the enumerated arrival set."""
        if ranks is None or travel_s is None or reflection is None:
            raise ValueError("arrival_head needs the fusion ranks and the enumerated arrival set")
        return self.arrival_head(
            ranks,
            self._jitter_arrivals(travel_s, "head"),
            reflection,
            time_s,
            prepared.source_parameters[:, 3],
            prepared.source_parameters[:, 2],
        )

    @staticmethod
    def _repeat_travel(travel: RayTravelTime, count: int) -> RayTravelTime:
        def repeat(value: torch.Tensor) -> torch.Tensor:
            return value[:, None, :].expand(-1, count, -1).reshape(value.shape[0], -1)

        return RayTravelTime(
            seconds=repeat(travel.seconds),
            distance_m=repeat(travel.distance_m),
            path_velocity_mps=repeat(travel.path_velocity_mps),
            endpoint_velocity_mps=repeat(travel.endpoint_velocity_mps),
            mean_slowness_s_per_m=repeat(travel.mean_slowness_s_per_m),
        )

    def _carrier_features(
        self,
        prepared: PreparedV3State,
        coords_xy_m: torch.Tensor,
        time_s: torch.Tensor,
        *,
        mirror_tau_s: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """L2 v2 carrier features, or None when carrier_mode='off'.

        mirror_tau_s lets the dense path reuse a per-grid tau (the mirror path
        is time-independent) instead of re-integrating per time block; the
        query path computes it in place.  256 samples/segment is the frozen
        constant from QUADRATURE_CONVERGENCE_20260914.json.
        """
        if self.carrier_mode == "off":
            return None
        if mirror_tau_s is None:
            mirror_tau_s = single_surface_mirror_travel_time(
                prepared.medium.velocity_mps,
                prepared.source_parameters[:, :2],
                coords_xy_m,
                x_extent_m=(0.0, self.domain_x_m),
                z_extent_m=(0.0, self.domain_z_m),
                record_to_medium=prepared.record_to_medium,
            )
        return mirror_phase_features(
            mirror_tau_s.to(time_s.dtype),
            time_s,
            prepared.source_parameters[:, None, 3],
            prepared.source_parameters[:, None, 2],
            domain_t_s=self.domain_t_s,
        )

    def _dense_block(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        dense_grid: DenseGridCache,
        *,
        apply_correction: bool,
        return_raw: bool = False,
    ) -> torch.Tensor:
        # The per-time-step call is the memory-scaling unit: the loop in
        # `iter_dense_normalized` invokes this once per time step, so a 51-frame
        # supervision window holds 51 of these activation sets live at once
        # (measured ~0.7 GiB per frame with the inner-block checkpointing in
        # place).  Checkpointing here instead of only inside the decoder's
        # blocks bounds peak memory by ONE time-step's activations, at the cost
        # of recomputing the block during backward.
        #
        # `apply_correction` must be threaded through the checkpoint call: a
        # captured (closure) value would not be part of the recomputation
        # signature and the correction branch would drop out of the graph on
        # the backward pass.
        if self.dense_outer_checkpoint and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint as _checkpoint

            return _checkpoint(
                self._dense_block_impl,
                prepared,
                time_s,
                dense_grid,
                apply_correction,
                int(getattr(self, "scno_depth", 1)),
                return_raw,
                use_reentrant=False,
            )
        return self._dense_block_impl(
            prepared, time_s, dense_grid, apply_correction=apply_correction,
            scno_depth=int(getattr(self, "scno_depth", 1)),
            return_raw=return_raw,
        )

    def _dense_block_impl(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        dense_grid: DenseGridCache,
        apply_correction: bool,
        scno_depth: int = 1,
        return_raw: bool = False,
    ) -> torch.Tensor:
        records, count = time_s.shape
        points = dense_grid.height * dense_grid.width
        xy = dense_grid.coords_xy_m[:, None].expand(-1, count, -1, -1)
        time_values = time_s[:, :, None].expand(-1, -1, points)
        coords = torch.cat((xy, time_values[..., None]), dim=-1).reshape(records, count * points, 3)
        travel = self._repeat_travel(dense_grid.travel, count)
        trunk_rank, bundle = self.coordinate_encoder(
            coords,
            prepared.source_parameters,
            travel,
            domain_x_m=self.domain_x_m,
            domain_z_m=self.domain_z_m,
            domain_t_s=self.domain_t_s,
            anchor_time_s=prepared.anchor_time_s,
        )
        carrier = None
        if self.carrier_mode != "off":
            if dense_grid.mirror_tau_s is None:
                raise ValueError(
                    "carrier_mode is enabled but the dense grid cache carries no mirror tau; "
                    "rebuild it with prepare_dense_grid on this model")
            mirror_tau = (
                dense_grid.mirror_tau_s[:, None, :]
                .expand(-1, count, -1)
                .reshape(records, count * points)
            )
            carrier = self._carrier_features(
                prepared, coords[..., :2], coords[..., 2], mirror_tau_s=mirror_tau)
        phase = None
        grid_T = grid_r = phase_T = None
        if self.phase_mode != "off":
            if dense_grid.phase_T is None:
                raise ValueError("phase_mode is enabled but the dense grid cache carries no phase maps")
            grid_T, grid_r = dense_grid.phase_T, dense_grid.phase_r
            if self.phase_direct:
                if dense_grid.mirror_tau_s is None:
                    raise ValueError("phase_mode 'multi_arrival_direct' needs the dense grid mirror tau")
                grid_T, grid_r = self._append_direct_arrivals(
                    grid_T, grid_r, dense_grid.travel.seconds, dense_grid.mirror_tau_s)
            grid_T = self._jitter_arrivals(grid_T, "all")
            phase_T = (
                grid_T[:, :, None, :]
                .expand(-1, -1, count, -1)
                .reshape(records, self.phase_arrivals, count * points)
            )
            phase = multi_arrival_phase_features(
                phase_T, grid_r, coords[..., 2], prepared.source_parameters[:, 3],
                prepared.source_parameters[:, 2], harmonics=self.phase_harmonics, domain_t_s=self.domain_t_s,
            )
        travel_rank = self.travel_branch(
            travel,
            bundle,
            domain_t_s=self.domain_t_s,
            domain_diagonal_m=math.sqrt(self.domain_x_m**2 + self.domain_z_m**2),
            carrier_features=carrier,
            phase_features=phase,
        )
        normalized_xy = dense_grid.coords_xy_normalized[:, None].expand(-1, count, -1, -1)
        # P0 (v18.5, 20260929): consume the record-side caches on no-grad paths only.
        # The cached [record, points, C] tensors are tiled across this call's frame
        # axis; every downstream op is identical to the uncached path.
        use_p0_cache = (
            not torch.is_grad_enabled()
            and dense_grid.fusion_velocity_rank_local is not None
            and dense_grid.fusion_local_hidden is not None
        )
        cached_vrl = cached_lh = None
        if use_p0_cache:
            cached_vrl = (
                dense_grid.fusion_velocity_rank_local[:, None]
                .expand(-1, count, -1, -1)
                .reshape(records, count * points, -1)
            )
            cached_lh = (
                dense_grid.fusion_local_hidden[:, None]
                .expand(-1, count, -1, -1)
                .reshape(records, count * points, -1)
            )
        fused = self.fusion(
            prepared.medium.encoding,
            prepared.source_encoding,
            normalized_xy.reshape(records, count * points, 2),
            prepared.record_to_medium,
            travel_rank,
            trunk_rank,
            cached_velocity_rank_local=cached_vrl,
            cached_local_hidden=cached_lh,
        )
        pressure = fused.normalized_pressure
        if self.arrival_head is not None:
            # Same live graph as the fusion output: the head's term enters the
            # coarse frame the dense decoder / cycle / local path all see.
            pressure = pressure + self._arrival_head_term(prepared, fused.ranks, phase_T, grid_r, coords[..., 2])
        if self.coarse_channel is not None:
            # Coarse-channel term, added at the same point as the arrival head so it
            # enters the same coarse frame.  `pressure` here is [record, points] in
            # normalized pressure units; gather_coarse returns the coarse field at the
            # identical (record, frame, point) set and in the same units (the cache
            # stores physical Pa and the caller encodes it once).  Zero-initialised, so
            # this is a no-op at the fork.
            if prepared.coarse_field is None or prepared.coarse_dt_s is None:
                raise ValueError(
                    "coarse_channel is enabled but the prepared state carries no coarse field")
            # `pressure` is [record, count * points] (the flattened (frame, point) axis
            # _dense_block_impl builds); gather_coarse returns [record, count, points] in
            # the identical (frame, z-major point) order, so flattening gives point-for-point
            # correspondence.
            coarse_values = gather_coarse(
                prepared.coarse_field, time_s, dt_s=prepared.coarse_dt_s, points=points)
            pressure = pressure + self.coarse_channel(
                pressure, coarse_values.reshape(records, -1))
        if self._scatter_active(prepared):
            # Born single-scattering term on the SAME flattened (frame, point) axis the
            # dense block builds, added before the reshape so it enters the coarse frame
            # the dense decoder / cycle / local path all see.  `time_s` here is
            # [record, count] (one time per frame, shared by every point of that frame),
            # so it is expanded to the point axis to match the flattened layout.
            flat_times = time_s[:, :, None].expand(-1, -1, points).reshape(records, -1)
            term = self._scatter_term(prepared, coords, flat_times)
            pressure = pressure + gate_view(term, self._scatter_gate(prepared))
        coarse = pressure.reshape(
            records, count, dense_grid.height, dense_grid.width
        )
        if apply_correction:
            raw = self.dense_decoder(
                prepared.medium.encoding,
                prepared.source_encoding,
                prepared.source_parameters,
                prepared.record_to_medium,
                time_s,
                coarse,
                cached_spatial=(
                    dense_grid.decoder_spatial
                    if (not torch.is_grad_enabled()) and dense_grid.decoder_spatial is not None
                    else None
                ),
                velocity=(
                    prepared.medium.velocity_mps[prepared.record_to_medium]
                    if self.dense_decoder.reflection_branch is not None
                    else None
                ),
            )
        else:
            raw = coarse
        # Same live graph, inside outer checkpoint. Explicit depth is frozen
        # in checkpoint arguments through backward recomputation.
        if hasattr(self, "scno_cycle"):
            if records != 1:
                raise ValueError("pilot binds one record per update")
            velocity = prepared.medium.velocity_mps[prepared.record_to_medium]
            delta = self.scno_cycle(
                raw[0, :, None],
                prepared.medium.encoding.pyramid[0][prepared.record_to_medium],
                prepared.source_encoding.map_field,
                velocity, prepared.normalizer.encode_velocity(velocity),
                time_s[0], prepared.source_parameters[0, 2],
                prepared.source_parameters[0, 3], dense_grid.x_m[1] - dense_grid.x_m[0],
                depth=scno_depth,
            )
            if bool(getattr(self, "verify_zero_cycle", False)):
                if bool(torch.count_nonzero(delta)):
                    raise RuntimeError("initial cycle output is not zero")
                if not torch.equal(raw + delta[:, 0][None], raw):
                    raise RuntimeError("initial cycle changes parent raw")
            raw = raw + delta[:, 0][None]
        if self.local_unet is not None:
            # Local spatial path: the coarse field has no spatial convolution
            # anywhere upstream (see local_unet.py), so this supplies the missing
            # multi-scale local coupling.  Time is folded into the batch axis and
            # broadcast as a channel, so the same spatial weights serve every
            # frame.  The branch is zero-initialised, so this is the identity
            # until training opens it.
            records_b, count_b = time_s.shape
            height_b, width_b = raw.shape[-2:]
            # Normalised velocity (same encoding the medium encoder and the
            # scno_cycle receive). Raw m/s values (~2.6e3..4.4e3) drove the
            # branch output to ~1.2e4x the field amplitude (v2 arm, step 2
            # loss 11.06); the field and time channels are already O(1).
            velocity_b = prepared.normalizer.encode_velocity(
                prepared.medium.velocity_mps[prepared.record_to_medium]
            )
            field_in = raw.reshape(records_b * count_b, 1, height_b, width_b)
            vel_in = (
                velocity_b[:, None]
                .expand(-1, count_b, -1, -1, -1)
                .reshape(records_b * count_b, 1, height_b, width_b)
            )
            time_in = (
                time_s[:, :, None, None]
                .expand(-1, -1, height_b, width_b)
                .reshape(records_b * count_b, 1, height_b, width_b)
            )
            correction = self.local_unet(field_in, vel_in, time_in)
            raw = raw + correction.reshape(records_b, count_b, height_b, width_b)
        surface = self._free_surface_factor(dense_grid.z_m)[None, None, :, None]
        if return_raw:
            return raw * surface, raw
        return raw * surface

    def _expand_dense_times(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        times = torch.as_tensor(time_s, dtype=torch.float32, device=self.device)
        records = prepared.source_parameters.shape[0]
        if times.ndim == 1:
            times = times[None].expand(records, -1)
        if times.ndim != 2 or times.shape[0] != records or times.shape[1] == 0:
            raise ValueError("dense time_s must be [time] or [record,time]")
        if not torch.isfinite(times).all() or torch.any(times < 0) or torch.any(times > self.domain_t_s):
            raise ValueError("dense times lie outside the physical time domain")
        return times

    def iter_dense_normalized(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        *,
        x_m: torch.Tensor | None = None,
        z_m: torch.Tensor | None = None,
        dense_grid: DenseGridCache | None = None,
        time_block: int | None = None,
        apply_correction: bool = True,
    ):
        times = self._expand_dense_times(prepared, time_s)
        if dense_grid is None:
            if x_m is None or z_m is None:
                raise ValueError("x_m and z_m are required when dense_grid is not supplied")
            dense_grid = self._memoized_dense_grid(prepared, x_m, z_m)
        block = self.dense_decoder.time_block if time_block is None else int(time_block)
        if block <= 0:
            raise ValueError("time_block must be positive")
        for start in range(0, times.shape[1], block):
            yield start, self._dense_block(
                prepared,
                times[:, start : start + block],
                dense_grid,
                apply_correction=apply_correction,
            )

    def _memoized_dense_grid(self, prepared: PreparedV3State, x_m, z_m) -> DenseGridCache:
        """R5 lazy grid reuse (v18.5 pre-audit, 20260929): evaluation callers issue
        one dense_normalized call per frame chunk with the SAME prepared state,
        rebuilding the grid (straight-ray travel, mirror tau, P0 caches) every
        chunk (~0.09 s/frame).  Memoize the grid on the prepared state, keyed by
        exact x/z axis equality.  NO-GRAD ONLY: with grad enabled this delegates
        to prepare_dense_grid unconditionally, so training (one call per draw,
        graph-connected grid) is bitwise and semantically untouched."""
        if torch.is_grad_enabled():
            return self.prepare_dense_grid(prepared, x_m=x_m, z_m=z_m)
        x = torch.as_tensor(x_m, dtype=torch.float32, device=self.device)
        z = torch.as_tensor(z_m, dtype=torch.float32, device=self.device)
        memo = getattr(prepared, "_dense_grid_memo", None)
        if memo is not None and torch.equal(memo.x_m, x) and torch.equal(memo.z_m, z):
            return memo
        grid = self.prepare_dense_grid(prepared, x_m=x, z_m=z)
        try:
            object.__setattr__(prepared, "_dense_grid_memo", grid)
        except (AttributeError, TypeError):
            pass  # exotic prepared containers just skip memoization
        return grid

    def dense_normalized(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        *,
        x_m: torch.Tensor | None = None,
        z_m: torch.Tensor | None = None,
        dense_grid: DenseGridCache | None = None,
        time_block: int | None = None,
        apply_correction: bool = True,
    ) -> torch.Tensor:
        return torch.cat(
            [
                value
                for _, value in self.iter_dense_normalized(
                    prepared,
                    time_s,
                    x_m=x_m,
                    z_m=z_m,
                    dense_grid=dense_grid,
                    time_block=time_block,
                    apply_correction=apply_correction,
                )
            ],
            dim=1,
        )

    def dense_normalized_with_raw(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        *,
        x_m: torch.Tensor | None = None,
        z_m: torch.Tensor | None = None,
        dense_grid: DenseGridCache | None = None,
        time_block: int | None = None,
        apply_correction: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        times = self._expand_dense_times(prepared, time_s)
        if dense_grid is None:
            if x_m is None or z_m is None:
                raise ValueError("x_m and z_m are required when dense_grid is not supplied")
            dense_grid = self._memoized_dense_grid(prepared, x_m, z_m)
        block = self.dense_decoder.time_block if time_block is None else int(time_block)
        if block <= 0:
            raise ValueError("time_block must be positive")
        finals: list[torch.Tensor] = []
        raws: list[torch.Tensor] = []
        for start in range(0, times.shape[1], block):
            final, raw = self._dense_block(
                prepared,
                times[:, start : start + block],
                dense_grid,
                apply_correction=apply_correction,
                return_raw=True,
            )
            finals.append(final)
            raws.append(raw)
        return torch.cat(finals, dim=1), torch.cat(raws, dim=1)

    def predict_wavefield(
        self,
        prepared: PreparedV3State,
        time_s: torch.Tensor,
        **dense_kwargs,
    ) -> torch.Tensor:
        normalized = self.dense_normalized(prepared, time_s, **dense_kwargs)
        return prepared.normalizer.decode_pressure(
            normalized.float(),
            prepared.source_parameters[:, 4],
        )

    def required_gradient_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        groups: dict[str, tuple[nn.Parameter, ...]] = {}
        groups.update(self.medium_encoder.required_gradient_groups())
        groups.update(self.source_encoder.required_gradient_groups())
        groups["travel_branch"] = tuple(self.travel_branch.parameters())
        groups["periodic_trunk"] = tuple(self.coordinate_encoder.parameters())
        groups.update(self.fusion.required_gradient_groups())
        groups.update(self.dense_decoder.required_gradient_groups())
        return groups


__all__ = [
    "EncodedMediumState",
    "DenseGridCache",
    "PhaseAlignedComplexFNOMIONet",
    "PreparedV3State",
    "free_surface_factor",
]
