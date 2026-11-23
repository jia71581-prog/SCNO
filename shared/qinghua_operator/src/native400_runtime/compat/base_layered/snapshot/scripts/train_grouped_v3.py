"""Helpers used by the public training entry points."""
from __future__ import annotations
from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.model.operator import PhaseAlignedComplexFNOMIONet

def build_model(config: V3Config, *, dense_checkpoint: bool = True,
                dense_outer_checkpoint: bool = False) -> PhaseAlignedComplexFNOMIONet:
    model = config.model
    token_grid = round(model.token_count**0.5)
    if token_grid * token_grid != model.token_count:
        raise ValueError("model.token_count must be a perfect square")
    return PhaseAlignedComplexFNOMIONet(
        width=model.width,
        rank=model.mionet_rank,
        spectral_rank=model.spectral_rank,
        modes=model.modes,
        dense_modes=model.dense_modes,
        dense_time_block=model.dense_time_block,
        dense_equivariant=model.dense_equivariant,
        dense_radial_cutoff=model.dense_radial_cutoff,
        local_multi_dilation=bool(getattr(model, "local_multi_dilation", False)),
        medium_u_enhance=bool(getattr(model, "medium_u_enhance", False)),
        u_path_enabled=bool(getattr(model, "u_path_enabled", False)),
        u_path_channels=tuple(getattr(model, "u_path_channels", (16, 32, 48))),
        u_path_output_scale=float(getattr(model, "u_path_output_scale", 0.01)),
        fixed_wave_physics=bool(getattr(model, "fixed_wave_physics", False)),
        heads=model.heads,
        token_grid=token_grid,
        position_bands=model.position_bands,
        fourier_bands=model.fourier_bands,
        gabor_scales_s=model.gabor_scales_s,
        ray_samples=model.ray_samples,
        ic_frames=model.ic_frames,
        domain_x_m=model.domain_x_m,
        domain_z_m=model.domain_z_m,
        domain_t_s=model.domain_t_s,
        dense_checkpoint=dense_checkpoint,
        dense_outer_checkpoint=dense_outer_checkpoint,
        carrier_mode=model.carrier_mode,
        local_unet_channels=tuple(model.local_unet_channels),
        local_unet_rank=int(model.local_unet_rank),
        phase_mode=str(model.phase_mode),
        phase_interfaces=int(model.phase_interfaces),
        phase_harmonics=int(model.phase_harmonics),
        arrival_head=bool(model.arrival_head),
        arrival_head_hidden=int(model.arrival_head_hidden),
        arrival_head_harmonics=int(model.arrival_head_harmonics),
        arrival_head_arrivals=int(model.arrival_head_arrivals),
        coarse_channel=bool(model.coarse_channel),
        coarse_channel_hidden=int(model.coarse_channel_hidden),
        coarse_channel_gain=float(model.coarse_channel_gain),
        scattering_head=bool(getattr(model, "scattering_head", False)),
        scatter_grid=int(getattr(model, "scatter_grid", 51)),
        scatter_harmonics=int(getattr(model, "scatter_harmonics", 1)),
        scatter_channels=int(getattr(model, "scatter_channels", 16)),
        scatter_hidden=int(getattr(model, "scatter_hidden", 32)),
        scatter_dilations=tuple(getattr(model, "scatter_dilations", (1, 1))),
        phase_gated_families=tuple(getattr(model, "phase_gated_families", ())),
        scatter_ungated_families=tuple(getattr(model, "scatter_ungated_families", ())),
        reflection_branch=bool(getattr(model, "reflection_branch", False)),
        reflection_channels=int(getattr(model, "reflection_channels", 32)),
        free_surface_mode=str(getattr(model, "free_surface_mode", "legacy_tanh20")),
        free_surface_blend_updates=int(getattr(model, "free_surface_blend_updates", 500)),
    )
