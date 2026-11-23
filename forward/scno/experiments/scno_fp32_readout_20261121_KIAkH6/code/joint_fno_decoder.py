"""FNO time-channel pressure readout with scale-specific orthogonal SCNO corrections."""

import torch
from torch import nn
from torch.nn import functional as F


_HIDDEN_WIDTHS = {"uniform_layered": (128,), "marmousi": (89, 67)}


def _frame_indices(slots, batch, device):
    if slots is None:
        return None
    indices = torch.as_tensor(slots, device=device)
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("frame indices must be integers on the registered time axis")
    if indices.ndim == 1 or (indices.ndim == 2 and indices.shape[0] == batch):
        return indices.to(dtype=torch.long)
    raise ValueError("expected shared indices [K] or per-record indices [B,K]")


def _time_projection(features, weight, bias, slots):
    """Apply selected rows of the final bias-optional 1x1 convolution."""
    batch, channels, *shape = features.shape
    if slots is not None:
        selected = slots.reshape(-1)
        weight = weight.index_select(0, selected)
        if bias is not None:
            bias = bias.index_select(0, selected)
        if slots.shape[-1] == 0:
            return features[:, :0]
        if slots.ndim == 2:
            count = slots.shape[1]
            flat = features.reshape(batch, channels, -1)
            projected = torch.bmm(weight.reshape(batch, count, channels), flat)
            if bias is not None:
                projected = projected + bias.reshape(batch, count, 1)
            return projected.reshape(batch, count, *shape)
    if features.ndim == 4:
        return F.conv2d(features, weight[:, :, None, None], bias)
    projected = weight @ features.reshape(batch, channels, -1)
    if bias is not None:
        projected = projected + bias[:, None]
    return projected.reshape(batch, weight.shape[0], *shape)


class JointFNOTimeDecoder(nn.Module):
    """Decode source-conditioned spatial features into native normalized pressure frames.

    The base projection uses the hidden channels of the corresponding archived
    spatial FNO. Each active scale has its own coefficient predictor and time
    matrix. Fixed orthogonal synthesis commutes with those linear time rows.
    """

    def __init__(self, in_channels, native_times, *, detail_transform=None,
                 detail_width=32, detail_latent=8, profile="uniform_layered"):
        super().__init__()
        if profile not in _HIDDEN_WIDTHS:
            raise ValueError("expected the uniform_layered or marmousi FNO readout profile")
        if min(in_channels, detail_width, detail_latent) < 1:
            raise ValueError("feature dimensions must be positive")
        axis = torch.as_tensor(native_times, dtype=torch.float32).detach().clone()
        if axis.ndim != 1 or not axis.numel() or not torch.isfinite(axis).all() or not (axis[1:] > axis[:-1]).all():
            raise ValueError("native query times must form a finite, strictly increasing vector")
        self.register_buffer("native_times", axis)
        self.in_channels = in_channels
        self.profile = profile
        self.detail_latent = detail_latent
        self.detail_transform = detail_transform

        widths = (in_channels, *_HIDDEN_WIDTHS[profile])
        base = []
        for incoming, outgoing in zip(widths, widths[1:]):
            base.extend((nn.Conv2d(incoming, outgoing, 1), nn.GELU()))
        base.append(nn.Conv2d(widths[-1], axis.numel(), 1))
        self.proj = nn.Sequential(*base)

        self.detail_heads = nn.ModuleList()
        self.detail_time = nn.ModuleList()
        if detail_transform is not None:
            if not len(detail_transform.levels):
                raise ValueError("the pressure-detail transform must contain a detail scale")
            projection_channels = 4
            self.detail_projection = nn.Conv2d(in_channels, projection_channels, 1)
            for level in detail_transform.levels:
                output_slots = level.valid_detail.shape[-1]
                head = nn.Sequential(
                    nn.Linear(level.modes * projection_channels + output_slots, detail_width), nn.SiLU(),
                    nn.Linear(detail_width, detail_width), nn.SiLU(),
                    nn.Linear(detail_width, detail_latent * output_slots))
                self.detail_heads.append(head)
                self.detail_time.append(nn.Linear(detail_latent, axis.numel(), bias=False))
            for module in (self.detail_projection, *self.detail_heads.modules(), *self.detail_time):
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(module.weight, gain=1.)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _coefficient_basis(self, features):
        _, packets = self.detail_transform.analyze(self.detail_projection(features))
        coefficients = []
        for predictor, packet, level in zip(self.detail_heads, packets, self.detail_transform.levels):
            batch = packet.shape[0]
            mask = level.valid_detail
            block = packet.permute(0, 2, 3, 1, 4).flatten(-2)
            local = torch.cat((block, mask.to(block).expand(batch, -1, -1, -1)), dim=-1)
            coeff = predictor(local).reshape(batch, level.gz, level.gx, self.detail_latent, mask.shape[-1])
            coefficients.append(coeff.permute(0, 3, 1, 2, 4).masked_fill(~mask, 0))
        return tuple(coefficients)

    def _synthesize_basis(self, coeffs):
        # Reserve disjoint spatial channels for each scale's time matrix.
        channels = len(coeffs) * self.detail_latent
        packets = []
        for i, coeff in enumerate(coeffs):
            left = i * self.detail_latent
            right = channels - left - self.detail_latent
            packets.append(F.pad(coeff, (0, 0, 0, 0, 0, 0, left, right)))
        last = self.detail_transform.levels[-1]
        coarse = coeffs[0].new_zeros((coeffs[0].shape[0], channels, last.gz * 4, last.gx * 4))
        return self.detail_transform.synthesize(coarse, packets)

    def prepare_spatial(self, features, *, detail_enabled=True):
        """Compute the base and all requested multiscale spatial features once."""
        if features.ndim != 4 or features.shape[1] != self.in_channels or features.shape[-2] < 2:
            raise ValueError("expected spatial conditioning features [B,C,Z,X] including the free surface")
        active = features[:, :, 1:]
        if self.detail_transform is not None and active.shape[-2:] != self.detail_transform.shape:
            raise ValueError("feature grid and active pressure transform do not agree")
        base = active
        for i in range(len(self.proj) - 1):
            base = self.proj[i](base)
        coeffs = ()
        combined = base
        if detail_enabled and self.detail_transform is not None:
            coeffs = self._coefficient_basis(active)
            details = self._synthesize_basis(coeffs)
            combined = torch.cat((base, details), dim=1)
        return combined, coeffs

    def decode_prepared(self, prepared, slots=None, *, return_parts=False):
        """Mix the spatial fields into all time channels, or only the training rows requested."""
        combined, coeffs = prepared
        slots = _frame_indices(slots, combined.shape[0], combined.device)
        projection = self.proj[-1]
        base_rows = projection.weight[:, :, 0, 0]
        weight = torch.cat((base_rows, *(linear.weight for linear in self.detail_time)), dim=1) if coeffs else base_rows
        pressure = _time_projection(combined, weight, projection.bias, slots)
        pressure = F.pad(pressure, (0, 0, 1, 0))
        if not return_parts:
            return pressure
        masks = tuple(level.valid_detail for level in self.detail_transform.levels) if self.detail_transform is not None else ()
        if coeffs:
            base = _time_projection(combined[:, :base_rows.shape[1]], base_rows, projection.bias, slots)
            base = F.pad(base, (0, 0, 1, 0))
            details = tuple(_time_projection(value, linear.weight, None, slots)
                            for value, linear in zip(coeffs, self.detail_time))
        else:
            base = pressure
            details = tuple(pressure.new_zeros((*pressure.shape[:2], *mask.shape)) for mask in masks)
        return {"prediction": pressure, "base_pressure": base, "detail_coefficients": details,
                "valid_detail_masks": masks}

    def forward(self, features, slots=None, *, detail_enabled=True, return_parts=False):
        spatial = self.prepare_spatial(features, detail_enabled=detail_enabled)
        return self.decode_prepared(spatial, slots, return_parts=return_parts)
