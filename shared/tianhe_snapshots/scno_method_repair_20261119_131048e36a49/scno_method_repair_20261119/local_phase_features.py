"""Input-only local travel-phase features for the nonlinear SCNO readouts.

The added features explicitly pair sin/cos of physical time with sin/cos of
medium/source travel delay. They enter before the first of the existing two
32-wide SiLU layers. They use no pressure observations or trajectory targets.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from time_conditioned_detail import (
    PreparedDetailFeatures, TimeConditionedDetailHead, TimeConditionedCoarsestBaseHead)


def pack_spatial_phase_features(features, travel_times, source):
    """Append direct/reflected delays and f0; all fields are prediction inputs."""
    if (features.ndim != 4 or travel_times.shape != (features.shape[0], 2, *features.shape[-2:])
            or source.shape != (features.shape[0], 5)):
        raise ValueError("use shared features, two physical travel-delay grids, and source [B,5]")
    frequency = source[:, 2, None, None, None].expand(-1, 1, *features.shape[-2:])
    return torch.cat((features, travel_times, frequency), dim=1)


def pack_temporal_phase_features(global_temporal, physical_times, source):
    """Append physical t, f0 and t0 to the existing source-Gabor features."""
    if global_temporal.ndim != 3 or source.shape != (global_temporal.shape[0], 5):
        raise ValueError("supply global time features [B,T,Q] and source [B,5]")
    times = torch.as_tensor(physical_times, device=global_temporal.device, dtype=global_temporal.dtype)
    if times.ndim == 1:
        times = times[None].expand(global_temporal.shape[0], -1)
    if times.shape != global_temporal.shape[:2]:
        raise ValueError("physical times must match the selected time-feature rows")
    frequency = source[:, None, 2].expand_as(times)
    start = source[:, None, 3].expand_as(times)
    return torch.cat((global_temporal, times[..., None], frequency[..., None], start[..., None]), dim=-1)


def orthogonal_phase_packets(transform, travel_times, source_frequency, frequency_ratios):
    """Return fixed-transform cosine/sine delay modes [B,R,2,GZ,GX,M]."""
    height, width = transform.shape
    if travel_times.shape[-2:] == (height + 1, width):
        travel_times = travel_times[:, :, 1:]
    elif travel_times.shape[-2:] != (height, width):
        raise ValueError("travel-delay geometry must match the active or full pressure grid")
    batch, paths = travel_times.shape[:2]
    if paths != 2 or source_frequency.shape != (batch,):
        raise ValueError("use direct/reflected delay fields and one physical f0 per record")
    ratios = torch.as_tensor(frequency_ratios, device=travel_times.device, dtype=travel_times.dtype)
    phase = ((2 * math.pi) * source_frequency[:, None, None, None, None]
             * ratios[None, :, None, None, None] * travel_times[:, None])
    fields = torch.cat((phase.cos().flatten(1, 2), phase.sin().flatten(1, 2)), dim=1)
    _, packets = transform.analyze(fields)
    result = []
    for packet, level in zip(packets, transform.levels):
        pair = packet.reshape(batch, 2, len(ratios), paths, level.gz, level.gx, level.modes)
        result.append((pair[:, 0], pair[:, 1]))
    return tuple(result)


class _LocalPhaseMixin:
    def _initialize_local_phase(self, global_time_features, frequency_ratios, balance_spatial_input):
        ratios = torch.as_tensor(frequency_ratios, dtype=torch.float32)
        if ratios.ndim != 1 or not len(ratios) or not bool(torch.isfinite(ratios).all()) or bool((ratios <= 0).any()):
            raise ValueError("use a nonempty fixed bank of positive source-frequency ratios")
        self.global_time_features = int(global_time_features)
        self.time_features = self.global_time_features + 3
        self.register_buffer("local_frequency_ratios", ratios)
        self.balance_spatial_input = bool(balance_spatial_input)
        self.register_buffer("spatial_input_balance", torch.tensor(self.balance_spatial_input))
        self.phase_levels = ((len(self.transform.levels) - 1,) if isinstance(self, TimeConditionedCoarsestBaseHead)
                             else tuple(range(len(self.transform.levels))))
        self.phase_projections = nn.ModuleList()
        ones = self.transform.levels[0].qz.new_ones((1, 1, *self.transform.shape))
        with torch.no_grad():
            _, constant_packets = self.transform.analyze(ones)
        for scale, level_index in enumerate(self.phase_levels):
            level = self.transform.levels[level_index]
            projections = nn.ModuleList(nn.Linear(2 * level.modes, self.hidden_width, bias=False)
                                        for _ in range(len(ratios)))
            for layer in projections:
                nn.init.xavier_uniform_(layer.weight, gain=1.)
            self.phase_projections.append(projections)
            # The transform of a constant has block energy equal to the number
            # of represented physical nodes. Normalize input feature amplitudes;
            # this does not alter the final orthogonal pressure synthesis.
            scale_field = constant_packets[level_index].square().sum(-1).sqrt().clamp_min(1.)
            self.register_buffer(f"phase_input_scale_{scale}", scale_field, persistent=False)

    def _after_state_load(self, module, incompatible_keys):
        super()._after_state_load(module, incompatible_keys)
        self.balance_spatial_input = bool(self.spatial_input_balance.detach().cpu())

    def local_phase_semantics(self):
        """Persist this identity with the wrapper's architecture/checkpoint settings."""
        return {"schema": "scno_input_only_orthogonal_local_phase_v1",
                "balance_spatial_input": self.balance_spatial_input,
                "spatial_conditioning": "first_spatial_preactivation/sqrt_physical_block_nodes"
                    if self.balance_spatial_input else "none",
                "frequency_ratios": self.local_frequency_ratios.detach().cpu().tolist(),
                "hidden_width": self.hidden_width, "hidden_silu_layers": 2,
                "spatial_input_tail": ["direct_delay_s", "reflected_delay_s", "source_f0_hz"],
                "temporal_input_tail": ["physical_query_s", "source_f0_hz", "source_t0_s"],
                "reference_pressure_input": False}

    def prepare_spatial(self, features):
        if features.ndim != 4 or features.shape[1] != self.in_channels + 3:
            raise ValueError("append two physical travel delays and f0 to the original feature channels")
        shared = super().prepare_spatial(features[:, :self.in_channels])
        travel_times = features[:, self.in_channels:self.in_channels + 2]
        frequency = features[:, self.in_channels + 2, 0, 0]
        phase_packets = orthogonal_phase_packets(
            self.transform, travel_times, frequency, self.local_frequency_ratios)
        prepared = []
        for scale, level_index in enumerate(self.phase_levels):
            cosine, sine = phase_packets[level_index]
            # [B,R,paths,GZ,GX,modes] -> [B,R,GZ,GX,paths*modes].
            cosine = cosine.permute(0, 1, 3, 4, 2, 5).flatten(-2)
            sine = sine.permute(0, 1, 3, 4, 2, 5).flatten(-2)
            normalizer = getattr(self, f"phase_input_scale_{scale}")[:, :, :, :, None]
            cosine, sine = cosine / normalizer, sine / normalizer
            c = torch.stack([layer(cosine[:, k]) for k, layer in enumerate(self.phase_projections[scale])], dim=-2)
            s = torch.stack([layer(sine[:, k]) for k, layer in enumerate(self.phase_projections[scale])], dim=-2)
            paired = torch.stack((c, s), dim=-3).flatten(-3)
            ordinary = shared.spatial[scale]
            if self.balance_spatial_input:
                ordinary = ordinary / getattr(self, f"phase_input_scale_{scale}").squeeze(1)[..., None]
            prepared.append(torch.cat((ordinary, paired), dim=-1))
        return PreparedDetailFeatures(tuple(prepared))

    def _hidden_scale(self, scale, spatial, temporal):
        width = self.hidden_width
        ordinary = self.temporal_first[scale](temporal[..., :self.global_time_features])
        time, frequency, start = temporal[..., -3], temporal[..., -2], temporal[..., -1]
        phase = (2 * math.pi) * ((time - start) * frequency)[..., None] * self.local_frequency_ratios
        geometry = spatial[..., width:].reshape(*spatial.shape[:-1], 2, len(self.local_frequency_ratios), width)
        cosine, sine = geometry[..., 0, :, :], geometry[..., 1, :, :]
        paired = (torch.einsum("btr,bzxrh->btzxh", phase.sin(), cosine)
                  - torch.einsum("btr,bzxrh->btzxh", phase.cos(), sine)) / math.sqrt(len(self.local_frequency_ratios))
        first = F.silu(spatial[..., :width][:, None] + ordinary[:, :, None, None] + paired)
        return F.silu(self.second_hidden[scale](first))


class LocalPhaseDetailHead(_LocalPhaseMixin, TimeConditionedDetailHead):
    """Candidate detail head; preserves two hidden SiLUs and the original readout."""

    def __init__(self, in_channels, time_features, transform, *, frequency_ratios=(.75, 1., 1.25),
                 balance_spatial_input=True, **kwargs):
        super().__init__(in_channels, time_features, transform, **kwargs)
        self._initialize_local_phase(time_features, frequency_ratios, balance_spatial_input)


class LocalPhaseCoarsestBaseHead(_LocalPhaseMixin, TimeConditionedCoarsestBaseHead):
    """Optional complementary base head using the same input-only phase pairing."""

    def __init__(self, in_channels, time_features, transform, *, frequency_ratios=(.75, 1., 1.25),
                 balance_spatial_input=True, **kwargs):
        super().__init__(in_channels, time_features, transform, **kwargs)
        self._initialize_local_phase(time_features, frequency_ratios, balance_spatial_input)
