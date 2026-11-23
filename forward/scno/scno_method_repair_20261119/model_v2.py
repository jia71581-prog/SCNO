"""Stage-aware training adapter for the existing SCNO pressure architecture."""

from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fast_scno import DetailHead, FineTuneRegistry, _detail_features
from joint_curriculum import JointSpectralSCNO
from lora_v2 import CompensatedRank4, place_saved_adapters
from spectral_marmousi import _capture


DETAIL_PREFIX = "refiner.spectral_detail."
INACTIVE_NAME = "base.source_encoder.ic_encoder.0.weight"


class SCNOTrainingV2(JointSpectralSCNO):
    def __init__(self, model_config, metadata, times, device):
        grid_size = model_config.get("decoder_grid_size", 101)
        if type(grid_size) is not int or grid_size not in (101, 201):
            raise ValueError("the SCNO decoder grid must be 101 or 201")
        super().__init__(model_config, metadata, np.asarray(times, dtype=np.float32), device=device)
        self.decoder_grid_size = grid_size
        self.initialization_receipt["decoder_grid_size"] = grid_size
        self.phase = "C"
        self.set_phase("C")

    @property
    def detail_transform(self):
        return self.refiner.spectral_detail.transform

    @property
    def detail_output_layers(self):
        return tuple(predictor[-1] for predictor in self.refiner.spectral_detail.heads)

    @property
    def no_decay_names(self):
        # The inactive IC slices of the spatial refiner are gradient-masked.
        return {"refiner.static.0.conv1.weight"}

    def named_roles(self):
        roles = {"BASE": [], "DETAIL": []}
        frozen = {id(parameter) for layer in self.modules() if isinstance(layer, CompensatedRank4)
                  for parameter in layer.parameters()}
        for name, value in self.named_parameters():
            if name != INACTIVE_NAME and id(value) not in frozen:
                key = "DETAIL" if name.startswith(DETAIL_PREFIX) else "BASE"
                roles[key].append((name, value))
        return roles

    def load_state_dict(self, state_dict, *args, **kwargs):
        place_saved_adapters(self, state_dict)
        return super().load_state_dict(state_dict, *args, **kwargs)

    def set_phase(self, phase):
        if phase not in {"A", "B", "C"}:
            raise ValueError("SCNO training phase must be A, B or C")
        self.phase = phase
        self.spectral_detail_enabled = phase != "A"
        for layer in self.modules():
            if isinstance(layer, CompensatedRank4):
                layer.requires_grad_(False)
        roles = self.named_roles()
        for _, parameter in roles["BASE"]:
            parameter.requires_grad_(phase != "B")
        for _, parameter in roles["DETAIL"]:
            parameter.requires_grad_(phase != "A")
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "phase", None) == "B":
            self.base.eval()
            self.refiner.eval()
            self.refiner.spectral_detail.train()
        elif mode and getattr(self, "phase", None) == "A":
            self.refiner.spectral_detail.eval()
        return self

    def _prepare_cf(self, velocity, source):
        if self.decoder_grid_size == 101:
            return super()._prepare_cf(velocity, source)
        zero_ic = velocity.new_zeros((1, 8, 201, 201))
        source_map = self._source_map(source, 201, 10.)
        medium = self.base.encode_medium(velocity.contiguous(), self.normalizer)
        prepared = self.base.prepare_sources(
            medium, source, source_map, self.normalizer,
            record_to_medium=torch.zeros(1, dtype=torch.long, device=self.device),
            ic_snapshots_physical=zero_ic, anchor_time_s=self.native_times_s[88:89],
        )
        grid = self.base.prepare_dense_grid(prepared, x_m=self.x_m, z_m=self.z_m)
        z, x = torch.meshgrid(self.z_m, self.x_m, indexing="ij")
        coords = torch.stack((2*x/2000-1, 2*z/2000-1,
                              (x-source[0, 0])/2000, (z-source[0, 1])/2000))[None]
        static = torch.cat((velocity/3000-1, zero_ic, source_map, coords), dim=1)
        return prepared, grid, (self.refiner.prepare(static), velocity)

    def _decoder_base_pressure(self, prepared, grid, source, times, *, lift=None):
        """Normalized backbone pressure on the output lattice as [T,1,201,201]."""
        side = self.decoder_grid_size
        if (grid.height, grid.width) != (side, side):
            raise ValueError("prepared grid and decoder resolution differ")
        pressure = self.base.dense_normalized(prepared, times[None], dense_grid=grid,
                                              time_block=self.dense_time_block, apply_correction=True)
        if pressure.shape != (1, times.numel(), side, side):
            raise RuntimeError("dense backbone pressure must match its physical query grid")
        if side == 201:
            return pressure[0, :, None]
        physical = self.normalizer.decode_pressure(pressure, source[:, 4])
        lift_pressure = self._bicubic201 if lift is None else lift
        field = lift_pressure(physical[0, :, None])
        return self.normalizer.encode_pressure(field[:, 0][None], source[:, 4])[0, :, None]

    def _query_base(self, prepared, grid, features, source, times):
        pressure = self._decoder_base_pressure(prepared, grid, source, times)
        value = pressure + self.refiner(features, pressure, times, source)
        return torch.cat((torch.zeros_like(value[:, :, :1]), value[:, :, 1:]), dim=-2)[:, 0]

    def forecast_at_times(self, c, f, times_s, *, normalized=True):
        if self.decoder_grid_size == 101:
            return super().forecast_at_times(c, f, times_s, normalized=normalized)
        return self(c, f, times_s, normalized=normalized)

    def _parts(self, prepared, grid, context, source, times, detail_enabled, detail_synthesis=True):
        features, velocity = context
        active = {"fusion": self.base.fusion, "readout": self.refiner.output, "frames": [], "refined": []}
        token = _capture.set(active)
        try:
            with torch.no_grad() if self.phase == "B" else nullcontext():
                base = self._query_base(prepared, grid, features, source, times)
        finally:
            _capture.reset(token)
        if len(active["refined"]) != 1 or not active["frames"]:
            raise RuntimeError("expected the inherited fusion and pressure readout activations")
        raw = torch.stack([torch.cat(part, dim=1) for part in zip(*active["frames"])], dim=-1)
        if raw.shape != (1, times.numel() * grid.height * grid.width, 2):
            raise RuntimeError("inherited fusion readout differs from the requested physical time blocks")
        fused = raw.reshape(-1, grid.height, grid.width, 2).permute(0, 3, 1, 2)
        gabor = self.base.coordinate_encoder.gabor(times, source[0, 3], source[0, 2], self.duration_s)
        branch = self.refiner.spectral_detail
        heads = tuple(DetailHead(head) for head in branch.heads)
        registry = FineTuneRegistry(self.refiner.output, heads)
        first = _detail_features(branch, registry, fused, active["refined"][0], velocity, source, gabor)
        hidden = tuple(head.module[3](head.fc2(value)) for head, value in zip(heads, first))
        masks = tuple(level.valid_detail for level in self.detail_transform.levels)
        coefficients = tuple(head.out(value).masked_fill(~mask, 0)[:, None]
                             for head, value, mask in zip(heads, hidden, masks))
        if not detail_synthesis:
            prediction = None
        elif not detail_enabled:
            coefficients = tuple(torch.zeros_like(packet) for packet in coefficients)
            prediction = base
        else:
            last = self.detail_transform.levels[-1]
            coarse_zero = base.new_zeros((len(times), 1, last.gz * 4, last.gx * 4))
            detail = F.pad(self.detail_transform.synthesize(coarse_zero, coefficients), (0, 0, 1, 0))
            prediction = base + detail[:, 0]
        return prediction, base, coefficients, hidden

    def forward(self, velocity, source, times=None, *, detail_enabled=None, return_parts=False,
                normalized=True, detail_synthesis=True):
        if not detail_synthesis and (self.phase != "B" or not return_parts or detail_enabled is False or not normalized):
            raise ValueError("coefficient-only queries require the fixed-base DETAIL training phase")
        source = torch.as_tensor(source, dtype=torch.float32, device=self.device)
        if source.shape == (5,):
            source = source[None]
        no_signal = source.shape == (1, 5) and source[0, 4].item() == 0.
        if no_signal:
            checked_source = source.clone()
            checked_source[0, 4] = 1.
            velocity, _ = self._inputs(velocity, checked_source)
        else:
            velocity, source = self._inputs(velocity, source)
        times = self.full_times_s if times is None else torch.as_tensor(times, dtype=torch.float32, device=self.device)
        times = times.reshape(-1)
        if not len(times) or not (torch.isfinite(times).all() and ((0 <= times) & (times <= self.duration_s)).all()):
            raise ValueError("queries must be finite physical times within the configured domain")
        enabled = self.spectral_detail_enabled if detail_enabled is None else bool(detail_enabled)
        masks = tuple(level.valid_detail for level in self.detail_transform.levels)
        if no_signal:
            # Zero amplitude has an exactly zero target and output. Attach an
            # active leaf so a homogeneous all-zero microbatch can backpropagate.
            leaf = next((p for p in self.parameters() if p.requires_grad), None)
            pressure = velocity.new_zeros((len(times), 201, 201))
            if leaf is not None and torch.is_grad_enabled():
                pressure = pressure + leaf.reshape(-1)[0] * 0
            if not return_parts:
                return pressure
            return {"prediction": pressure if detail_synthesis else None,
                    "base_pressure": pressure, "valid_detail_masks": masks,
                    "detail_coefficients": tuple(velocity.new_zeros((len(times), 1, *m.shape)) for m in masks),
                    "detail_last_hidden_features": tuple(velocity.new_zeros((len(times), *m.shape[:2], layer.in_features))
                                                         for m, layer in zip(masks, self.detail_output_layers))}

        frozen = self.phase == "B"
        with self.scopes(), (torch.no_grad() if frozen else nullcontext()):
            prepared, grid, context = self._prepare_cf(velocity, source)

        def query(block):
            with self.scopes():
                if not (enabled or return_parts):
                    return self._query_base(prepared, grid, context[0], source, block)
                parts = self._parts(prepared, grid, context, source, block, enabled, detail_synthesis)
                return parts if return_parts else parts[0]

        recompute = self.training and torch.is_grad_enabled() and self.checkpoint_queries
        pieces = [checkpoint(query, block, use_reentrant=False) if recompute else query(block)
                  for block in times.split(self.refiner_time_block)]
        if not return_parts:
            pressure = torch.cat(pieces)
            return pressure if normalized else self.normalizer.decode_pressure(pressure[None], source[:, 4])[0]
        if not normalized:
            raise ValueError("return_parts reports normalized pressure and coefficients")
        return {"prediction": torch.cat([p[0] for p in pieces]) if detail_synthesis else None,
                "base_pressure": torch.cat([p[1] for p in pieces]),
                "detail_coefficients": tuple(torch.cat([p[2][i] for p in pieces]) for i in range(len(masks))),
                "valid_detail_masks": masks,
                "detail_last_hidden_features": tuple(torch.cat([p[3][i] for p in pieces]) for i in range(len(masks)))}


def build_model(config, dataset, device):
    settings = {**config.get("model", {}), "seed": config["seed"]}
    if config["data"]["kind"] == "centered_uniform_layered":
        settings.setdefault("family", "layered")
    return SCNOTrainingV2(settings, dataset.metadata, dataset.times, device)
