"""Independent Hermitian coordinates for the archived ordinary rFFT kernel.

This is an opt-in, fixed-grid conversion, not an in-place training migration.
At kx=0 (and a retained x Nyquist column), irfft2 observes only the Hermitian
part across ky. Store one complex matrix per non-self-conjugate orbit and one
real matrix per self-conjugate frequency. Other kx columns remain complex.
The conversion preserves the exact-arithmetic function; FP32 operation order
changes, so bitwise equivalence is not promised. Old Adam moments cannot in
general be merged into an equivalent Adam state by this parameter conversion.
"""
from __future__ import annotations

import torch
from torch import nn


SCHEMA = "PAPER201_FIXED_GRID_HERMITIAN_SPECTRAL_V1"


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


class HermitianSpectralConv2d(nn.Module):
    """Minimal real-linear kernel coordinates for a specified H x W grid.

    For a special kx column, the positive-ky representative is
        q = (K_plus + conj(K_minus)) / 2,
    with absent original modes treated as zero. Consequently an unpaired
    negative edge uses q=conj(K_minus)/2, including its formerly missing positive
    partner in forward. A ky Nyquist row is self-conjugate and uses Re(K), with
    no factor of one half. Source parameters are copied, never retained as a
    child module. All installed parameters represent independent directions in
    the space of real-input linear operators, not necessarily for one input.
    """

    def __init__(self, source, height: int, width: int):
        super().__init__()
        _require(type(height) is int and type(width) is int and height >= 2 and width >= 1,
                 "fixed integer grid height>=2, width>=1 required")
        top, bottom = source.weight_top, source.weight_bottom
        _require(isinstance(top, nn.Parameter) and isinstance(bottom, nn.Parameter)
                 and top.ndim == 5 and top.shape == bottom.shape and top.shape[-1] == 2,
                 "ordinary top/bottom real-pair complex parameters required")
        _require(top.dtype in (torch.float32, torch.float64) and top.dtype == bottom.dtype
                 and top.device == bottom.device and top.requires_grad == bottom.requires_grad,
                 "matching floating dtype, device and trainability required")
        _require(tuple(top.shape[:2]) == (source.in_channels, source.out_channels)
                 and tuple(top.shape[2:4]) == (source.modes_y, source.modes_x),
                 "source channel/mode metadata differs from parameter shape")
        _require(source.fft_norm in ("backward", "forward", "ortho"), "unsupported FFT norm")
        _require(torch.isfinite(top).all() and torch.isfinite(bottom).all(),
                 "finite source spectral parameters required")
        self.in_channels, self.out_channels = source.in_channels, source.out_channels
        self.height, self.width, self.fft_norm = height, width, source.fft_norm
        self.source_shape = tuple(top.shape)
        self.modes_y = min(source.modes_y, height // 2)
        self.modes_x = min(source.modes_x, width // 2 + 1)
        _require(min(self.modes_y, self.modes_x) >= 1, "at least one retained mode required")
        self.selected_y = tuple(range(self.modes_y)) + tuple(range(height-self.modes_y, height))
        self.special_x = (0,) + ((width//2,) if width % 2 == 0 and self.modes_x > width//2 else ())
        regular_x = tuple(x for x in range(self.modes_x) if x not in self.special_x)
        self.regular_modes = tuple((y, x) for y in self.selected_y for x in regular_x)
        # A top/bottom lookup also records the original (possibly uncropped)
        # allocation, needed for exact old-gradient pullback including zero tails.
        self._source_y = {y: ("weight_top", y) for y in range(self.modes_y)}
        self._source_y.update({y: ("weight_bottom", y-(height-self.modes_y))
                               for y in range(height-self.modes_y, height)})
        orbits = sorted({min(y, (-y) % height) for y in self.selected_y})
        self.pair_modes = tuple((q, x) for x in self.special_x for q in orbits if q != (-q) % height)
        self.self_modes = tuple((q, x) for x in self.special_x for q in orbits if q == (-q) % height)
        values = {"weight_top": top.detach(), "weight_bottom": bottom.detach()}

        def original(y, x):
            if y not in self._source_y:
                return top.new_zeros(self.in_channels, self.out_channels, 2)
            name, row = self._source_y[y]
            return values[name][:, :, row, x]

        def parameter(name, pieces):
            self.register_parameter(name, nn.Parameter(torch.stack(pieces, dim=2).clone(),
                                                       requires_grad=top.requires_grad) if pieces else None)

        parameter("weight_regular", [original(y, x) for y, x in self.regular_modes])
        pairs = []
        for y, x in self.pair_modes:
            positive, negative = original(y, x), original((-y) % height, x)
            pairs.append(torch.stack(((positive[..., 0]+negative[..., 0])*.5,
                                      (positive[..., 1]-negative[..., 1])*.5), dim=-1))
        parameter("weight_pair", pairs)
        parameter("weight_self", [original(y, x)[..., 0] for y, x in self.self_modes])
        half_width = width//2+1
        for name, modes in (("regular_indices", self.regular_modes), ("pair_indices", self.pair_modes),
                            ("pair_conjugate_indices", tuple(((-y) % height, x) for y, x in self.pair_modes)),
                            ("self_indices", self.self_modes)):
            self.register_buffer(name, torch.tensor([y*half_width+x for y, x in modes],
                                                    dtype=torch.long, device=top.device), persistent=False)

    @classmethod
    def from_original(cls, source, height: int, width: int):
        return cls(source, height, width)

    def get_extra_state(self):
        return dict(schema=SCHEMA, height=self.height, width=self.width,
                    in_channels=self.in_channels, out_channels=self.out_channels,
                    source_shape=list(self.source_shape), retained_modes=[self.modes_y, self.modes_x],
                    fft_norm=self.fft_norm, representative="positive_ky_hermitian_projection")

    def set_extra_state(self, state):
        _require(state == self.get_extra_state(), "Hermitian checkpoint grid/source/schema mismatch")

    def retained_modes(self, height, width):
        _require((height, width) == (self.height, self.width), "fixed Hermitian spectral grid changed")
        return self.modes_y, self.modes_x

    def receipt(self):
        matrix_count = self.in_channels*self.out_channels
        original = 2*matrix_count*self.source_shape[2]*self.source_shape[3]*2
        retained = 2*matrix_count*self.modes_y*self.modes_x*2
        compact = sum(parameter.numel() for parameter in self.parameters())
        return dict(**self.get_extra_state(), original_parameters=original,
                    retained_original_parameters=retained, effective_parameters=compact,
                    removed_unindexed=original-retained, removed_hermitian_redundancy=retained-compact,
                    regular_complex_matrices=len(self.regular_modes),
                    representative_complex_matrices=len(self.pair_modes),
                    self_conjugate_real_matrices=len(self.self_modes),
                    gradient_rule="g_old=A^T g_new for q=A p; not an optimizer-state migration",
                    exact_arithmetic_equivalent=True, fp32_bitwise_equivalence_claim=False)

    def forward(self, value):
        _require(value.ndim == 4 and value.shape[1] == self.in_channels,
                 "spectral input must be [batch,in_channels,z,x]")
        self.retained_modes(*value.shape[-2:])
        reference = self.weight_self  # DC is always retained, even on a2x1 grid.
        _require(value.device == reference.device and value.is_floating_point(),
                 "floating input on the parameter device required")
        with torch.autocast(device_type=value.device.type, enabled=False):
            # Original deployment is FP32. .double() is supported deliberately
            # for independent high-precision algebra/gradient verification.
            transform = torch.fft.rfft2(value.to(reference.dtype), norm=self.fft_norm).flatten(-2)
            output = transform.new_zeros(value.shape[0], self.out_channels,
                                         self.height*(self.width//2+1))
            if self.weight_regular is not None:
                kernel = torch.view_as_complex(self.weight_regular.contiguous())
                output[:, :, self.regular_indices] = torch.einsum(
                    "bik,iok->bok", transform[:, :, self.regular_indices], kernel)
            if self.weight_pair is not None:
                kernel = torch.view_as_complex(self.weight_pair.contiguous())
                positive = torch.einsum("bik,iok->bok", transform[:, :, self.pair_indices], kernel)
                output[:, :, self.pair_indices] = positive
                output[:, :, self.pair_conjugate_indices] = positive.conj()
            self_values = torch.einsum("bik,iok->bok", transform[:, :, self.self_indices].real,
                                       self.weight_self)
            output[:, :, self.self_indices] = self_values.to(output.dtype)
            return torch.fft.irfft2(output.reshape(value.shape[0], self.out_channels,
                                                   self.height, self.width//2+1),
                                   s=(self.height, self.width), norm=self.fft_norm)

    def legacy_gradient_map(self, gradients=None):
        """Pull compact loss gradients back to both full original allocations.

        If gradients is omitted, use this module's populated .grad tensors.
        For paired modes, old plus receives(real,imag)/2 and old minus receives
        (real,-imag)/2. A lone negative edge receives that same half/conjugate;
        self-conjugate real receives the full gradient, with imaginary zero.
        Unused allocated tails remain zero. This is A^T, not A or an average of
        old gradients. Applying it to Adam moments is not an exact Adam resume.
        """
        parameters = dict(self.named_parameters())
        if gradients is None:
            gradients = {name: parameter.grad for name, parameter in parameters.items()}
        _require(set(gradients) == set(parameters), "compact gradient parameter names mismatch")
        for name, parameter in parameters.items():
            gradient = gradients[name]
            _require(isinstance(gradient, torch.Tensor) and gradient.shape == parameter.shape
                     and gradient.dtype == parameter.dtype and gradient.device == parameter.device,
                     "populated matching compact gradients required: "+name)
        result = {name: self.weight_self.new_zeros(self.source_shape)
                  for name in ("weight_top", "weight_bottom")}
        for index, (y, x) in enumerate(self.regular_modes):
            name, row = self._source_y[y]
            result[name][:, :, row, x] = gradients["weight_regular"][:, :, index].detach()
        for index, (y, x) in enumerate(self.pair_modes):
            gradient = gradients["weight_pair"][:, :, index].detach()
            if y in self._source_y:
                name, row = self._source_y[y]
                result[name][:, :, row, x] = gradient*.5
            negative = (-y) % self.height
            if negative in self._source_y:
                name, row = self._source_y[negative]
                result[name][:, :, row, x, 0] = gradient[..., 0]*.5
                result[name][:, :, row, x, 1] = -gradient[..., 1]*.5
        for index, (y, x) in enumerate(self.self_modes):
            name, row = self._source_y[y]
            result[name][:, :, row, x, 0] = gradients["weight_self"][:, :, index].detach()
        return result


__all__ = ["HermitianSpectralConv2d", "SCHEMA"]
