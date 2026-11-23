"""SCNO medium and source conditioning with a joint, registered-time pressure readout."""

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from joint_fno_decoder import JointFNOTimeDecoder


class JointFNOContext(NamedTuple):
    features: torch.Tensor
    source: torch.Tensor


def _source_map(source, side, spacing):
    x, z = (source[:, :2] / spacing).unbind(-1)
    ix, iz = x.floor().long().clamp(0, side - 2), z.floor().long().clamp(0, side - 2)
    fx, fz = x - ix, z - iz
    origin = iz * side + ix
    indices = torch.stack((origin, origin + 1, origin + side, origin + side + 1), dim=1)
    mass = torch.stack(((1 - fx) * (1 - fz), fx * (1 - fz), (1 - fx) * fz, fx * fz), dim=1)
    result = source.new_zeros((len(source), side * side)).scatter(1, indices, mass)
    return result.reshape(len(source), 1, side, side)


class JointFNOSCNO(nn.Module):
    """Produce [B,K,201,201] pressure. K indexes a fixed 401- or 801-frame model axis."""

    variant = "scno_joint_fno_time_channels_v1"
    _inherited_modules = {
        "medium_encoder": "base.medium_encoder",
        "source_encoder": "base.source_encoder",
        "multiscale_fuse": "base.dense_decoder.multiscale_fuse",
        "source_projection": "base.dense_decoder.source_projection",
        "map_projection": "base.dense_decoder.map_projection",
        "gabor": "base.coordinate_encoder.gabor",
        "gabor_embedding": "refiner.spectral_detail.time_projection",
        "decoder.detail_transform": "refiner.spectral_detail.transform",
    }

    @classmethod
    def from_native(cls, native_model, *, freeze_encoder=False, detail_width=32, detail_latent=8):
        return cls(native_model, freeze_encoder=freeze_encoder, detail_width=detail_width, detail_latent=detail_latent)

    @classmethod
    def from_checkpoint(cls, native_model, payload, *, freeze_encoder=None):
        """Restore a joint model, using a native architecture template and the joint state alone."""
        if isinstance(payload, (Path, str)):
            payload = torch.load(payload, map_location="cpu", weights_only=True)
        semantics = payload.get("model_semantics", {})
        if not isinstance(semantics, dict) or semantics.get("variant") != cls.variant:
            raise ValueError("expected a recorded joint FNO time-channel model")
        if freeze_encoder is None:
            freeze_encoder = not payload.get("encoder_trainable", True)
        devices = [native_model.x_m.device] if native_model.x_m.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            model = cls.from_native(native_model, freeze_encoder=freeze_encoder,
                                    detail_width=semantics["detail_width"], detail_latent=semantics["detail_latent"])
        if semantics != model.checkpoint_semantics():
            raise ValueError("joint checkpoint and target grid, time axis, normalization or architecture differ")
        state = payload.get("model")
        current = model.state_dict()
        if not isinstance(state, dict) or state.keys() != current.keys():
            raise ValueError("joint checkpoint has missing or unexpected state keys")
        for name, target in current.items():
            saved = state[name]
            if not torch.is_tensor(saved) or saved.shape != target.shape or saved.dtype != target.dtype:
                raise ValueError(f"incompatible joint checkpoint tensor {name}")
        for fixed in ("decoder.native_times", "anchor_time_s"):
            if not torch.equal(state[fixed].detach().cpu(), current[fixed].detach().cpu()):
                raise ValueError(f"joint checkpoint changed the physical meaning of {fixed}")
        model.load_state_dict(state, strict=True)
        model.initialization_receipt = {"variant": cls.variant, "restored_joint_checkpoint": True}
        return model

    def __init__(self, native_model, *, freeze_encoder=False, detail_width=32, detail_latent=8):
        super().__init__()
        if native_model.decoder_grid_size != 201:
            raise ValueError("inherit the medium and source pathways from a native 201-grid SCNO")
        self.family = native_model.family
        if self.family == "marmousi":
            self.profile = "marmousi"
        elif self.family in ("uniform", "homogeneous", "layered", "uniform_layered"):
            self.profile = "uniform_layered"
        else:
            raise ValueError(f"unknown SCNO model family {self.family}")
        times = native_model.full_times_s.detach().clone().float()
        if times.ndim != 1 or len(times) not in (401, 801) or times[0] != 0 or not (times[1:] > times[:-1]).all():
            raise ValueError("the native time axis must increase from zero and contain 401 or 801 frames")
        self.duration_s = float(times[-1])
        self.normalizer = deepcopy(native_model.normalizer)
        self.model_spec_sha256 = native_model.initialization_receipt["model_spec_sha256"]
        self._scope_factory = getattr(getattr(native_model, "_core", None), "scopes", None)
        self._scope_helpers = getattr(native_model, "helpers", None)
        for target, origin in self._inherited_modules.items():
            if target != "decoder.detail_transform":
                setattr(self, target, deepcopy(native_model.get_submodule(origin)))
        transform = deepcopy(native_model.detail_transform)
        self.width = self.multiscale_fuse.out_channels
        self.detail_width, self.detail_latent = int(detail_width), int(detail_latent)
        channels = self.width + 8
        self.decoder = JointFNOTimeDecoder(channels, times, detail_transform=transform, detail_width=detail_width,
                                           detail_latent=detail_latent, profile=self.profile)
        self.gabor_spatial = nn.Conv2d(channels, self.gabor_embedding.out_features, kernel_size=1)
        nn.init.xavier_uniform_(self.gabor_spatial.weight, gain=1.)
        nn.init.zeros_(self.gabor_spatial.bias)
        self.gabor_embedding.requires_grad_(True)
        z, x = torch.meshgrid(native_model.z_m.detach(), native_model.x_m.detach(), indexing="ij")
        self.register_buffer("x_m", x.reshape(1, 1, 201, 201).clone(), persistent=False)
        self.register_buffer("z_m", z.reshape(1, 1, 201, 201).clone(), persistent=False)
        self.register_buffer("source_scales", times.new_tensor(self.normalizer.metadata.source_scales), persistent=False)
        self.register_buffer("anchor_time_s", native_model.native_times_s[88].detach().clone())
        self.to(times.device)
        self.set_encoder_trainable(not freeze_encoder)
        self.train(native_model.training)
        source_names = set(native_model.state_dict())
        inherited = {}
        for name in self.state_dict():
            for target, origin in self._inherited_modules.items():
                if name.startswith(target + "."):
                    old = origin + name[len(target):]
                    if old not in source_names:
                        raise ValueError(f"missing inherited state {old}")
                    inherited[name] = old
                    break
        self.initialization_receipt = {
            "variant": self.variant, "profile": self.profile, "loaded": inherited,
            "new_parameters": sorted(set(dict(self.named_parameters())) - set(inherited)),
            "excluded_native_state": sorted(source_names - set(inherited.values())),
            "model_spec_sha256": self.model_spec_sha256,
        }

    @property
    def full_times_s(self):
        return self.decoder.native_times

    @property
    def detail_transform(self):
        return self.decoder.detail_transform

    @property
    def device(self):
        return self.x_m.device

    def checkpoint_semantics(self):
        stamp = hashlib.sha256(self.full_times_s.detach().cpu().numpy().astype("<f4").tobytes()).hexdigest()
        stats = self.normalizer.metadata
        return {"variant": self.variant, "family": self.family, "profile": self.profile,
                "model_spec_sha256": self.model_spec_sha256, "native_axis_sha256": stamp,
                "pressure_scale_pa": float(stats.pressure_scale_pa), "velocity_center_mps": float(stats.velocity_center_mps),
                "velocity_scale_mps": float(stats.velocity_scale_mps), "source_scales": list(stats.source_scales),
                "normalization_train_manifest_sha256": stats.train_manifest_sha256,
                "detail_width": self.detail_width, "detail_latent": self.detail_latent,
                "spatial_grid": [201, 201], "active_grid": [200, 201], "output_frames": len(self.full_times_s)}

    def checkpoint_payload(self):
        return {"model_semantics": self.checkpoint_semantics(), "encoder_trainable": self.encoder_trainable,
                "model": self.state_dict()}

    def set_encoder_trainable(self, enabled):
        for module in (self.medium_encoder, self.source_encoder, self.multiscale_fuse,
                       self.source_projection, self.map_projection):
            module.requires_grad_(enabled)
        if self.source_encoder.ic_frames:
            self.source_encoder.ic_encoder[0].weight.requires_grad_(False)
        self.encoder_trainable = bool(enabled)
        return self

    def _inputs(self, velocity, source):
        device = self.x_m.device
        velocity = torch.as_tensor(velocity, device=device, dtype=torch.float32)
        source = torch.as_tensor(source, device=device, dtype=torch.float32)
        if velocity.ndim == 2:
            velocity = velocity[None, None]
        elif velocity.ndim == 3:
            velocity = velocity[:, None]
        if source.ndim == 1:
            source = source[None]
        if (source.ndim != 2 or source.shape[1] != 5 or len(source) == 0 or velocity.ndim != 4
                or velocity.shape[1:] != (1, 201, 201) or len(velocity) not in (1, len(source))):
            raise ValueError("supply one shared or one 201-grid velocity per five-parameter source")
        valid = (torch.isfinite(velocity).all() & (velocity > 0).all() & torch.isfinite(source).all()
                 & ((source[:, :2] >= 0) & (source[:, :2] <= 2000)).all() & (source[:, 2] > 0).all()
                 & ((source[:, 3] >= 0) & (source[:, 3] <= self.duration_s)).all() & (source[:, 4] >= 0).all())
        if not valid:
            raise ValueError("velocity and source must be finite, physical and inside the registered space/time domain")
        return velocity, source

    def encode(self, velocity, source):
        """Shared spatial work; training needs this once per record, irrespective of queried frame count."""
        return self._encode_validated(*self._inputs(velocity, source))

    def _encode_validated(self, velocity, source, *, inference_source=None):
        """Encoder for already shape- and physics-validated device inputs, also capturable on CUDA."""
        count = len(source)
        mapping = (torch.arange(count, device=source.device) if len(velocity) == count
                   else torch.zeros(count, dtype=torch.long, device=source.device))
        scope = (self._scope_factory(self._scope_helpers, {k: [] for k in ("padding", "sampling", "pooling")})
                 if self._scope_factory is not None and inference_source is None else nullcontext())
        with scope:
            normalized_velocity = self.normalizer.encode_velocity(velocity)
            medium = self.medium_encoder(normalized_velocity)
            source_map = _source_map(source, 201, 10.)
            anchor = (self.anchor_time_s / self.duration_s).expand(count, 1)
            if inference_source is None:
                zero_ic = (velocity.new_zeros((count, self.source_encoder.ic_frames, 201, 201))
                           if self.source_encoder.ic_frames else None)
                src = self.source_encoder(source / self.source_scales, source_map, medium, mapping,
                                          ic_snapshots=zero_ic, anchor_normalized=anchor)
            else:
                src = inference_source(source / self.source_scales, source_map, medium, mapping, anchor)
            levels = [medium.pyramid[0][mapping]]
            levels.extend(F.interpolate(level[mapping], size=(201, 201), mode="bilinear", align_corners=True)
                          for level in medium.pyramid[1:])
            spatial = (self.multiscale_fuse(torch.cat(levels, dim=1)) + self.source_projection(src.hidden)[:, :, None, None]
                       + self.map_projection(src.map_field))
        dx = (self.x_m - source[:, 0, None, None, None]) / 2000.
        dz = (self.z_m - source[:, 1, None, None, None]) / 2000.
        radius = torch.linalg.vector_norm(torch.cat((dx, dz), dim=1), dim=1, keepdim=True)
        rank = (medium.rank[mapping] * src.rank).sum(-1) / math.sqrt(src.rank.shape[-1])
        features = torch.cat((spatial, (2 * self.x_m / 2000. - 1).expand_as(dx),
                              (2 * self.z_m / 2000. - 1).expand_as(dz), dx, dz, radius,
                              normalized_velocity[mapping], source_map, rank[:, None, None, None].expand_as(dx)), dim=1)
        return JointFNOContext(features, source)

    def _selected_times(self, slots, batch):
        if slots is None:
            return self.full_times_s[None].expand(batch, -1)
        times = self.full_times_s[slots]
        return times.expand(batch, -1) if slots.ndim == 1 else times

    def _project_once(self, features, slots, temporal, factor, detail_enabled, extra=None):
        spatial, coeffs = self.decoder.prepare_spatial(features, detail_enabled=detail_enabled)
        final = self.decoder.proj[-1]
        base_rows = final.weight[:, :, 0, 0]
        weights = (torch.cat((base_rows, *(layer.weight for layer in self.decoder.detail_time)), dim=-1)
                   if coeffs else base_rows)
        bias = final.bias
        if slots is not None:
            weights = weights[slots]
            bias = bias[slots]
        if weights.ndim == 2:
            weights = weights.unsqueeze(0).expand(features.shape[0], -1, -1)
            bias = bias.unsqueeze(0).expand(features.shape[0], -1)
        row_groups = (weights, temporal, bias[..., None])
        gabor_spatial = self.gabor_spatial(features[:, :, 1:])
        constant = torch.ones_like(gabor_spatial[:, :1])
        # Pad spatial basis fields before multiplying, so no full trajectory is
        # allocated for a separate Gabor field, bias addition, or boundary copy.
        if extra is not None:
            extra_rows, extra_basis = extra
            row_groups = tuple(item * self.residual_gain for item in row_groups)
            row_groups += (extra_rows,)
            spatial = torch.cat((spatial, gabor_spatial, constant, extra_basis[:, :, 1:]), dim=1)
        else:
            spatial = torch.cat((spatial, gabor_spatial, constant), dim=1)
        rows = torch.cat(row_groups, dim=-1) * factor[:, None, None]
        basis = F.pad(spatial, (0, 0, 1, 0))
        return torch.bmm(rows, basis.flatten(2)).reshape(features.shape[0], rows.shape[1], 201, 201)

    def decode(self, context, frame_indices=None, *, normalized=True, detail_enabled=True, return_parts=False):
        """Read only selected output rows for training, or all native times for deployment."""
        features, source = context[:2]
        slots = None if frame_indices is None else torch.as_tensor(frame_indices, device=features.device)
        if slots is not None and (slots.dtype not in (torch.int32, torch.int64)
                                  or (slots.numel() and (slots.min() < 0 or slots.max() >= len(self.full_times_s)))):
            raise ValueError("frame indices must be integers within the native time axis")
        if slots is not None and not (slots.ndim == 1 or (slots.ndim == 2 and slots.shape[0] == len(source))):
            raise ValueError("frame indices must have shape [K] or [B,K]")
        times = self._selected_times(slots, len(source))
        bank = self.gabor(times, source[:, 3:4], source[:, 2:3], self.duration_s)
        temporal = self.gabor_embedding(bank)
        factor = ((source[:, 4] != 0).to(features.dtype) if normalized
                  else self.normalizer.metadata.pressure_scale_pa * source[:, 4])
        extra = self.propagation_factors(context, slots) if len(context) > 2 else None
        if not return_parts:
            return self._project_once(features, slots, temporal, factor, detail_enabled, extra)
        parts = self.decoder(features, slots=slots, detail_enabled=detail_enabled, return_parts=True)
        spatial = self.gabor_spatial(features)[:, :, 1:].flatten(2)
        # One separable, source-conditioned contribution to the base pressure for all requested times.
        active = torch.bmm(temporal, spatial).reshape(len(source), times.shape[1], 200, 201)
        gabor_base = F.pad(active, (0, 0, 1, 0))
        scale = factor[:, None, None, None]
        gain = self.residual_gain if extra is not None else 1.
        carrier = torch.bmm(extra[0], extra[1].flatten(2)).reshape_as(gabor_base) if extra is not None else 0.
        return {**parts, "prediction": ((parts["prediction"] + gabor_base) * gain + carrier) * scale,
                "base_pressure": ((parts["base_pressure"] + gabor_base) * gain + carrier) * scale,
                "detail_coefficients": tuple(packet * gain * factor.view(-1, *([1] * (packet.ndim - 1)))
                                             for packet in parts["detail_coefficients"])}

    def forward(self, velocity, source, frame_indices=None, *, normalized=True, detail_enabled=True, return_parts=False):
        return self.decode(self.encode(velocity, source), frame_indices, normalized=normalized,
                           detail_enabled=detail_enabled, return_parts=return_parts)
