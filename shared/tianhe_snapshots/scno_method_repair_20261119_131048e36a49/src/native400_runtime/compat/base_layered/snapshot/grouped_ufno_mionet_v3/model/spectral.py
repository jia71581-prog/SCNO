"""True learned-complex Fourier contractions for the V3 medium and dense paths."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LearnedComplexSpectralConv2d(nn.Module):
    """Contract retained Fourier modes with independent learned complex matrices."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        modes_y: int,
        modes_x: int,
        fft_norm: str = "ortho",
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or modes_y <= 0 or modes_x <= 0:
            raise ValueError("channels and retained modes must be positive")
        if fft_norm not in {"backward", "forward", "ortho"}:
            raise ValueError(f"unsupported FFT normalization: {fft_norm}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_y = int(modes_y)
        self.modes_x = int(modes_x)
        self.fft_norm = fft_norm
        shape = (self.in_channels, self.out_channels, self.modes_y, self.modes_x, 2)
        self.weight_top = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.weight_bottom = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        scale = 1.0 / math.sqrt(self.in_channels * self.out_channels)
        nn.init.uniform_(self.weight_top, -scale, scale)
        nn.init.uniform_(self.weight_bottom, -scale, scale)

    def retained_modes(self, height: int, width: int) -> tuple[int, int]:
        if height < 2 or width < 1:
            raise ValueError("spectral input grid is too small")
        modes_y = min(self.modes_y, height // 2)
        modes_x = min(self.modes_x, width // 2 + 1)
        if modes_y < 1 or modes_x < 1:
            raise ValueError("input has no usable retained Fourier modes")
        return modes_y, modes_x

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.in_channels:
            raise ValueError(
                f"spectral input must be [batch,{self.in_channels},y,x], got {tuple(value.shape)}"
            )
        height, width = value.shape[-2:]
        modes_y, modes_x = self.retained_modes(height, width)
        with torch.autocast(device_type=value.device.type, enabled=False):
            value_fft = torch.fft.rfft2(value.float(), norm=self.fft_norm)
            output_fft = torch.zeros(
                value.shape[0],
                self.out_channels,
                height,
                width // 2 + 1,
                dtype=value_fft.dtype,
                device=value.device,
            )
            top = torch.view_as_complex(self.weight_top.contiguous())[
                :, :, :modes_y, :modes_x
            ]
            bottom = torch.view_as_complex(self.weight_bottom.contiguous())[
                :, :, :modes_y, :modes_x
            ]
            output_fft[:, :, :modes_y, :modes_x] = torch.einsum(
                "bixy,ioxy->boxy",
                value_fft[:, :, :modes_y, :modes_x],
                top,
            )
            output_fft[:, :, -modes_y:, :modes_x] = torch.einsum(
                "bixy,ioxy->boxy",
                value_fft[:, :, -modes_y:, :modes_x],
                bottom,
            )
            return torch.fft.irfft2(
                output_fft,
                s=(height, width),
                norm=self.fft_norm,
            )


class EquivariantSpectralConv2d(nn.Module):
    """Share complex weights across equal-|k| modes for an isotropic spectral kernel.

    Uses the full ``fft2`` with a circular wavenumber cutoff rather than the
    half-plane ``rfft2`` with a square mode window: a square window and a
    half-plane layout are not themselves invariant under 90 degree rotation,
    so radial weight sharing alone does not buy equivariance.  Buckets key on
    the exact integer ``ky^2 + kx^2``, which is a C4 and reflection invariant of
    the signed frequency lattice, making the resulting operator equivariant to
    the wave equation's spatial translations, rotations, and reflections up to
    floating point.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        cutoff: int,
        fft_norm: str = "ortho",
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or cutoff <= 0:
            raise ValueError("channels and wavenumber cutoff must be positive")
        if fft_norm not in {"backward", "forward", "ortho"}:
            raise ValueError(f"unsupported FFT normalization: {fft_norm}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.cutoff = int(cutoff)
        self.fft_norm = fft_norm
        self._lattice: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.bucket_count = self._radii(2 * self.cutoff + 1, 2 * self.cutoff + 1)[2]
        shape = (self.in_channels, self.out_channels, self.bucket_count, 2)
        self.weight = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        scale = 1.0 / math.sqrt(self.in_channels * self.out_channels)
        nn.init.uniform_(self.weight, -scale, scale)

    def _radii(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return the retained-mode mask, per-mode bucket index, and bucket count."""
        frequency_y = torch.fft.fftfreq(height) * height
        frequency_x = torch.fft.fftfreq(width) * width
        squared = (
            frequency_y[:, None].round().long() ** 2
            + frequency_x[None, :].round().long() ** 2
        )
        mask = squared <= self.cutoff * self.cutoff
        retained = torch.unique(squared[mask])
        table = torch.full((int(squared.max()) + 1,), 0, dtype=torch.long)
        table[retained] = torch.arange(retained.numel())
        return mask, torch.where(mask, table[squared], torch.zeros_like(squared)), int(retained.numel())

    def _cached_lattice(self, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache the flat positions of retained modes and their bucket ids.

        Only the modes inside the circular cutoff are contracted.  Masking the
        full ``height x width`` lattice instead would make the contraction scale
        with the grid rather than with the retained mode count, which costs an
        order of magnitude at the 201x201 production grid.
        """
        key = (int(height), int(width))
        cached = self._lattice.get(key)
        if cached is None or cached[0].device != device:
            mask, index, count = self._radii(height, width)
            if count != self.bucket_count:
                raise ValueError(
                    f"grid {key} yields {count} radial buckets, expected {self.bucket_count}; "
                    "the cutoff must be resolvable on this grid"
                )
            flat = mask.reshape(-1).nonzero(as_tuple=True)[0]
            cached = (flat.to(device), index.reshape(-1)[flat].to(device))
            self._lattice[key] = cached
        return cached

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.in_channels:
            raise ValueError(
                f"spectral input must be [batch,{self.in_channels},y,x], got {tuple(value.shape)}"
            )
        height, width = value.shape[-2:]
        if min(height, width) < 2 * self.cutoff + 1:
            raise ValueError("input grid is too small to resolve the wavenumber cutoff")
        mask, index = self._cached_lattice(height, width, value.device)
        with torch.autocast(device_type=value.device.type, enabled=False):
            value_fft = torch.fft.fft2(value.float(), norm=self.fft_norm)
            flat = value_fft.reshape(*value_fft.shape[:2], -1)[:, :, mask]
            kernel = torch.view_as_complex(self.weight.contiguous())[:, :, index]
            if getattr(self, "s0_kmajor_bmm", True):
                # S0 (v18.5 pre-audit, 20260929): the SAME per-k contraction
                # out[b,o,k] = sum_i flat[b,i,k] * kernel[i,o,k] issued as one
                # k-major contiguous bmm instead of einsum's strided path (the
                # einsum dispatch cloned ~12k views per call).  Bitwise equality
                # with the einsum path on the real weights is asserted by
                # probe_s0_spectral.py (CPU+GPU, full frames); the instance flag
                # exists only so that probe can A/B the two paths.
                contracted = (
                    torch.bmm(flat.permute(2, 0, 1).contiguous(),
                              kernel.permute(2, 0, 1).contiguous())
                    .permute(1, 2, 0)
                )
            else:
                contracted = torch.einsum("bik,iok->bok", flat, kernel)
            output_fft = torch.zeros(
                value.shape[0],
                self.out_channels,
                height * width,
                dtype=value_fft.dtype,
                device=value.device,
            )
            output_fft[:, :, mask] = contracted
            return torch.fft.ifft2(
                output_fft.reshape(value.shape[0], self.out_channels, height, width),
                norm=self.fft_norm,
            ).real


def _group_count(width: int) -> int:
    for groups in (8, 4, 2, 1):
        if width % groups == 0:
            return groups
    return 1


class ComplexSpectralResidualBlock(nn.Module):
    """Low-rank learned spectral path plus the useful V2 local residual path.

    ``local_multi_dilation`` (V18-A speed branch, 20260928) widens the local
    path's receptive field by summing two extra dilated depthwise 3x3 branches
    *before* the existing expensive 1x1 pointwise, so the pointwise is reused
    rather than replicated (no concat, no widened fusion):

        local_sum = legacy_dw(value) + g2 * dw2(value) + g4 * dw4(value)
        local     = legacy_local_pointwise(local_sum)

    * ``g2``/``g4`` are per-channel gates of shape ``[width]`` initialised to
      exactly zero, so the enabled block computes the baseline function at
      init (the gated terms are exact zeros) while the gates still receive a
      structured first-step gradient: with the delta-stencil init below,
      ``dwK(value) = value``, so ``dL/dgK[c] = <upstream_grad, value>_c`` — the
      channel correlation with the residual, not noise.  This avoids the
      documented local_unet v3 stall (random branch times scalar gate never
      opened); the dilated taps themselves get gradient as soon as a gate
      moves off zero.
    * Both new convolutions are initialised with a deterministic delta stencil
      (centre tap 1, bias 0) inside an RNG save/restore guard, so building the
      enabled block consumes exactly the same RNG stream as the baseline:
      every shared parameter — including those of *later* modules in the same
      seeded build — stays bitwise equal.
    * Off (the default) constructs no new modules and adds no state_dict keys.
    """

    _MULTI_DILATION_SUFFIXES = (
        "local_dw2.weight",
        "local_dw2.bias",
        "local_dw4.weight",
        "local_dw4.bias",
        "local_gate2",
        "local_gate4",
    )

    def __init__(
        self,
        *,
        width: int,
        spectral_rank: int,
        modes_y: int,
        modes_x: int,
        equivariant: bool = False,
        local_multi_dilation: bool = False,
    ) -> None:
        super().__init__()
        if width <= 0 or spectral_rank <= 0:
            raise ValueError("width and spectral rank must be positive")
        self.in_projection = nn.Conv2d(width, spectral_rank, kernel_size=1)
        self.spectral: nn.Module
        if equivariant:
            self.spectral = EquivariantSpectralConv2d(
                spectral_rank,
                spectral_rank,
                cutoff=min(int(modes_y), int(modes_x)),
            )
        else:
            self.spectral = LearnedComplexSpectralConv2d(
                spectral_rank,
                spectral_rank,
                modes_y=modes_y,
                modes_x=modes_x,
            )
        self.out_projection = nn.Conv2d(spectral_rank, width, kernel_size=1)
        self.local_depthwise = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=1,
            groups=width,
        )
        self.local_pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(_group_count(width), width)
        self.local_multi_dilation = bool(local_multi_dilation)
        if self.local_multi_dilation:
            # The Conv2d constructors draw from the global CPU generator; the
            # guard keeps the outer stream untouched so every parameter built
            # after this block under the same seed is bitwise unchanged.
            rng_state = torch.get_rng_state()
            self.local_dw2 = nn.Conv2d(
                width, width, kernel_size=3, padding=2, dilation=2, groups=width
            )
            self.local_dw4 = nn.Conv2d(
                width, width, kernel_size=3, padding=4, dilation=4, groups=width
            )
            torch.set_rng_state(rng_state)
            with torch.no_grad():
                for conv in (self.local_dw2, self.local_dw4):
                    conv.weight.zero_()
                    conv.weight[:, 0, 1, 1] = 1.0
                    conv.bias.zero_()
            self.local_gate2 = nn.Parameter(torch.zeros(width, dtype=torch.float32))
            self.local_gate4 = nn.Parameter(torch.zeros(width, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError("residual block input must be [batch,channel,y,x]")
        spectral = self.out_projection(self.spectral(self.in_projection(value).float()))
        local_sum = self.local_depthwise(value)
        if self.local_multi_dilation:
            local_sum = (
                local_sum
                + self.local_gate2.view(1, -1, 1, 1) * self.local_dw2(value)
                + self.local_gate4.view(1, -1, 1, 1) * self.local_dw4(value)
            )
        local = self.local_pointwise(local_sum)
        return value + F.gelu(self.norm(spectral + local))


def migrate_local_multi_dilation_state(
    module: nn.Module,
    legacy_state: dict[str, torch.Tensor],
    *,
    init_from_legacy_dw: bool = True,
) -> dict[str, torch.Tensor]:
    """Complete a legacy (multi-dilation-off) state_dict for an enabled module.

    Returns a state_dict loadable with ``strict=True`` into ``module``.  Only
    the six multi-dilation keys per block may be absent from ``legacy_state``;
    any other mismatch raises.  Gates come in as exact zeros, so the migrated
    model's forward is bitwise identical to the legacy checkpoint regardless
    of the dw2/dw4 fill.  With ``init_from_legacy_dw`` (deterministic, no RNG)
    the dilated taps start from the *trained* legacy depthwise kernel — a
    dilated copy of a useful stencil — instead of the constructor's delta.
    """
    merged = dict(legacy_state)
    own = module.state_dict()
    unexpected = sorted(set(merged) - set(own))
    if unexpected:
        raise ValueError(f"legacy state has keys the module lacks: {unexpected}")
    suffixes = ComplexSpectralResidualBlock._MULTI_DILATION_SUFFIXES
    for key in own:
        if key in merged:
            continue
        if not key.endswith(suffixes):
            raise ValueError(f"missing key {key} is not a multi-dilation parameter")
        if init_from_legacy_dw and "local_dw" in key:
            # "<prefix.>local_dwK.<field>" -> "<prefix.>local_depthwise.<field>";
            # the prefix is empty when migrating a bare block.
            prefix = key[: key.rindex("local_dw")]
            field = key.rsplit(".", 1)[1]
            legacy_key = f"{prefix}local_depthwise.{field}"
            merged[key] = legacy_state[legacy_key].clone()
        else:
            merged[key] = own[key].clone()
    return merged


__all__ = [
    "ComplexSpectralResidualBlock",
    "EquivariantSpectralConv2d",
    "LearnedComplexSpectralConv2d",
    "migrate_local_multi_dilation_state",
]
