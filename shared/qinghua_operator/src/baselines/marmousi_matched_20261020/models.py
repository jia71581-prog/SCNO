"""FNO and DeepONet with exactly 14,552,439 trainable real scalars each.

The count matches the paper's *single-source full-fit* model, excluding its
25,600 permanently frozen scalars. It does not match the newer reduced model.
All parameter tensors participate in the forward map; no parameter tensor is
added solely to pad the count. Conventional real-FFT Hermitian redundancy
(including an ineffective DC imaginary component) is retained, so this is a
trainable-scalar count, not a claim of independent functional degrees of freedom.

Inputs are normalized by the caller, with channels in this exact order:
``[velocity, source_x, source_z, source_frequency, source_t0, source_amplitude,
x_coordinate, z_coordinate]``. The five source channels are spatially constant.
No pressure observation or target is an input. DeepONet query columns are
``(t, x, z)``; query normalization is also the caller's responsibility.

FNO directly returns all registered time channels on the input spatial nodes.
DeepONet directly queries every requested space-time node. Neither model
interpolates or copies output labels. The DeepONet pooling compresses *input
features*, not a pressure prediction.

Architecture references (primary author repositories):
https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/legacy_spectral_convolution.py
https://github.com/lululxvi/deepxde/blob/master/deepxde/nn/pytorch/deeponet.py
These are self-contained implementations; neither repository is a dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TARGET_TRAINABLE_REAL_PARAMETERS = 14_552_439

FNO_CONFIGURATION: dict[str, Any] = {
    "in_channels": 8,
    "out_channels": 400,
    "width": 56,
    "modes1": 17,
    "modes2": 17,
    "layers": 4,
    "head_hidden": [89, 67],
    "padding": 9,
}

DEEPONET_CONFIGURATION: dict[str, Any] = {
    "in_channels": 8,
    "branch_channels": [32, 64, 64],
    "pool_size": 5,
    "branch_hidden": [2944, 3030],
    "trunk_width": 256,
    "trunk_depth": 4,
    "fourier_bands": 16,
    "rank": 128,
}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def parameter_report(
    model: nn.Module, target_parameters: int | None = None
) -> dict[str, Any]:
    """Count complex parameters as two real scalars, including frozen tensors.

    The FNO stores real/imaginary pairs in real tensors, so normal optimizer
    parameter counting also gives the intended real-scalar count. This helper
    additionally supports models using torch complex parameter tensors.
    """
    records = []
    for name, parameter in model.named_parameters():
        records.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
                "tensor_elements": parameter.numel(),
                "real_scalars": parameter.numel() * (2 if parameter.is_complex() else 1),
            }
        )
    trainable = sum(r["real_scalars"] for r in records if r["requires_grad"])
    total = sum(r["real_scalars"] for r in records)
    return {
        "kind": getattr(model, "kind", model.__class__.__name__),
        "configuration": dict(getattr(model, "configuration", {})),
        "trainable_real_scalars": trainable,
        "total_real_scalars": total,
        "frozen_real_scalars": total - trainable,
        "tensor_elements": sum(r["tensor_elements"] for r in records),
        "parameter_tensors": len(records),
        "target_parameters": target_parameters,
        "difference": None if target_parameters is None else trainable - target_parameters,
        "exact_match": None if target_parameters is None else trainable == target_parameters,
        "parameters": records,
    }


class SpectralConv2d(nn.Module):
    """Dense two-corner rFFT convolution, with independent complex weights.

    Positive and negative frequencies of the first transformed axis have
    independent learned weights. The rFFT reduction is only on the last axis.
    No conjugate tying or overlap between the two retained corners is used.
    Real-pair storage gives ordinary real Adam/AdamW state and explicit counts.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = _positive_integer(in_channels, "in_channels")
        self.out_channels = _positive_integer(out_channels, "out_channels")
        self.modes1 = _positive_integer(modes1, "modes1")
        self.modes2 = _positive_integer(modes2, "modes2")
        shape = (in_channels, out_channels, modes1, modes2, 2)
        self.weight_positive = nn.Parameter(torch.empty(shape))
        self.weight_negative = nn.Parameter(torch.empty(shape))
        # Each complex component has variance 1/(2*in_channels).
        std = 1.0 / math.sqrt(2 * in_channels)
        nn.init.normal_(self.weight_positive, std=std)
        nn.init.normal_(self.weight_negative, std=std)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected (B,{self.in_channels},H,W), got {tuple(x.shape)}")
        height, width = x.shape[-2:]
        if 2 * self.modes1 > height or self.modes2 > width // 2 + 1:
            raise ValueError(
                f"Grid {(height, width)} cannot contain disjoint retained corners "
                f"({self.modes1}, {self.modes2}); reduce modes for small-grid tests"
            )
        spectrum = torch.fft.rfft2(x, norm="backward")
        output = spectrum.new_zeros((x.shape[0], self.out_channels, height, width // 2 + 1))
        positive = torch.view_as_complex(self.weight_positive)
        negative = torch.view_as_complex(self.weight_negative)
        output[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[:, :, : self.modes1, : self.modes2], positive
        )
        output[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[:, :, -self.modes1 :, : self.modes2], negative
        )
        # Passing both dimensions is necessary for odd spatial sizes.
        return torch.fft.irfft2(output, s=(height, width), norm="backward")


class FNO2d(nn.Module):
    """Fourier operator with time represented by 400 direct output channels."""

    kind = "fno"

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 400,
        width: int = 56,
        modes1: int = 17,
        modes2: int = 17,
        layers: int = 4,
        head_hidden: Sequence[int] = (89, 67),
        padding: int = 9,
    ):
        super().__init__()
        for name, value in (("in_channels", in_channels), ("out_channels", out_channels),
                            ("width", width), ("layers", layers)):
            _positive_integer(value, name)
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a nonnegative integer")
        for hidden in head_hidden:
            _positive_integer(hidden, "head_hidden")
        self.configuration = dict(
            in_channels=in_channels, out_channels=out_channels, width=width,
            modes1=modes1, modes2=modes2, layers=layers,
            head_hidden=list(head_hidden), padding=padding,
        )
        self.padding = padding
        self.in_channels = in_channels
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes1, modes2) for _ in range(layers)
        )
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        dimensions = [width, *head_hidden, out_channels]
        head: list[nn.Module] = []
        for index, (n_in, n_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            head.append(nn.Conv2d(n_in, n_out, 1))
            if index + 1 < len(dimensions) - 1:
                head.append(nn.GELU())
        self.head = nn.Sequential(*head)

    def forward(self, input_maps: Tensor) -> Tensor:
        if input_maps.ndim != 4 or input_maps.shape[1] != self.in_channels:
            raise ValueError(f"Expected (B,{self.in_channels},H,W) input maps")
        x = self.lift(input_maps)
        if self.padding:
            x = F.pad(x, (0, self.padding, 0, self.padding))
        for index, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            x = spectral(x) + local(x)
            if index + 1 < len(self.spectral):
                x = F.gelu(x)
        if self.padding:
            x = x[..., : input_maps.shape[-2], : input_maps.shape[-1]]
        return self.head(x)


def _mlp(dimensions: Sequence[int]) -> nn.Sequential:
    modules: list[nn.Module] = []
    for index, (n_in, n_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        modules.append(nn.Linear(n_in, n_out))
        if index + 1 < len(dimensions) - 1:
            modules.append(nn.GELU())
    return nn.Sequential(*modules)


class RectangularAdaptiveAvgPool2d(nn.Module):
    """Adaptive-average bins implemented with deterministic slices and means.

    Standard AdaptiveAvgPool2d has a nondeterministic CUDA backward on the
    installed Torch 2.1.2. These bins use the same floor/ceil boundaries and
    means without its atomic backward kernel. Overlapping bins retain their
    separate contributions through ordinary autograd accumulation. FP32 mean
    reductions may differ from the fused pooling operation by roundoff.
    """

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = _positive_integer(output_size, "output_size")

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or min(x.shape[-2:]) < 1:
            raise ValueError("Expected nonempty (B,C,H,W) input to rectangular pooling")
        height, width = x.shape[-2:]
        size = self.output_size
        rows = []
        for row in range(size):
            z_start = row * height // size
            z_stop = ((row + 1) * height + size - 1) // size
            cells = []
            for col in range(size):
                x_start = col * width // size
                x_stop = ((col + 1) * width + size - 1) // size
                cells.append(x[..., z_start:z_stop, x_start:x_stop].mean(dim=(-2, -1)))
            rows.append(torch.stack(cells, dim=-1))
        return torch.stack(rows, dim=-2)


class FourierTrunk(nn.Module):
    """Fourier coordinates followed by a pre-LayerNorm residual MLP.

    Log-spaced frequencies follow the established raw934 DeepONet input
    encoding. The buffer is fixed, serialized, and does not count as learned
    parameters. At default width/depth this branch has 455,296 parameters;
    it is not reduced to a small coordinate head for the single-input fit.
    """

    def __init__(self, rank: int, width: int, depth: int, fourier_bands: int):
        super().__init__()
        _positive_integer(rank, "rank")
        _positive_integer(width, "trunk_width")
        _positive_integer(depth, "trunk_depth")
        _positive_integer(fourier_bands, "fourier_bands")
        self.register_buffer("bands", torch.exp(torch.linspace(0.0, 4.0, fourier_bands)) * torch.pi)
        self.input = nn.Linear(3 + 6 * fourier_bands, width)
        self.blocks = nn.ModuleList(
            nn.ModuleDict({"norm": nn.LayerNorm(width), "fc1": nn.Linear(width, width),
                           "fc2": nn.Linear(width, width)}) for _ in range(depth - 1)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, rank)

    def forward(self, coords: Tensor) -> Tensor:
        phase = coords.unsqueeze(-1) * self.bands
        features = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        x = self.input(torch.cat((coords, features), dim=-1))
        for block in self.blocks:
            residual = block["fc2"](F.gelu(block["fc1"](block["norm"](x))))
            x = x + residual
        return self.output(self.output_norm(x))


class DeepONet(nn.Module):
    """CNN/MLP branch and Fourier residual trunk joined by an inner product.

    Encode the branch once per input and call ``predict_points`` on disjoint
    coordinate chunks. A complete full-field objective must include every
    chunk and accumulate its branch gradient before one optimizer update.
    The model itself does not subsample points or construct a training loss.
    """

    kind = "deeponet"

    def __init__(
        self,
        in_channels: int = 8,
        branch_channels: Sequence[int] = (32, 64, 64),
        pool_size: int = 5,
        branch_hidden: Sequence[int] = (2944, 3030),
        trunk_width: int = 256,
        trunk_depth: int = 4,
        fourier_bands: int = 16,
        rank: int = 128,
    ):
        super().__init__()
        if in_channels != 8:
            raise ValueError("This input contract requires 8 channels, including source5")
        if not branch_channels:
            raise ValueError("At least one branch convolution is required")
        _positive_integer(pool_size, "pool_size")
        _positive_integer(rank, "rank")
        for width in [*branch_channels, *branch_hidden]:
            _positive_integer(width, "hidden width")
        self.configuration = dict(
            in_channels=in_channels, branch_channels=list(branch_channels),
            pool_size=pool_size, branch_hidden=list(branch_hidden),
            trunk_width=trunk_width, trunk_depth=trunk_depth,
            fourier_bands=fourier_bands, rank=rank,
        )
        self.rank = rank
        self.in_channels = in_channels
        convolutions: list[nn.Module] = []
        previous = in_channels
        for channels in branch_channels:
            convolutions.extend((nn.Conv2d(previous, channels, 3, stride=2, padding=1), nn.GELU()))
            previous = channels
        convolutions.extend((RectangularAdaptiveAvgPool2d(pool_size), nn.Flatten(1)))
        self.branch_encoder = nn.Sequential(*convolutions)
        self.branch = _mlp([previous * pool_size * pool_size + 5, *branch_hidden, rank])
        self.trunk = FourierTrunk(rank, trunk_width, trunk_depth, fourier_bands)
        self.output_bias = nn.Parameter(torch.zeros(()))

    def encode_branch(self, input_maps: Tensor, source: Tensor | None = None) -> Tensor:
        if input_maps.ndim != 4 or input_maps.shape[1] != self.in_channels:
            raise ValueError(f"Expected (B,{self.in_channels},H,W) input maps")
        if source is None:
            source = input_maps[:, 1:6, 0, 0]
        if source.shape != (input_maps.shape[0], 5):
            raise ValueError("source must contain five normalized values per input")
        encoded = self.branch_encoder(input_maps)
        return self.branch(torch.cat((encoded, source), dim=1))

    def predict_points(self, branch: Tensor, coords: Tensor) -> Tensor:
        if branch.ndim != 2 or branch.shape[-1] != self.rank:
            raise ValueError(f"branch must have shape (B,{self.rank})")
        if coords.ndim not in (2, 3) or coords.shape[-1] != 3:
            raise ValueError("coords must have shape (N,3) or (B,N,3), columns (t,x,z)")
        features = self.trunk(coords)
        if coords.ndim == 2:
            return torch.einsum("br,nr->bn", branch, features) + self.output_bias
        if coords.shape[0] != branch.shape[0]:
            raise ValueError("Per-example coordinates must have the branch batch size")
        return torch.einsum("br,bnr->bn", branch, features) + self.output_bias

    def decode_trunk(self, branch: Tensor, coords: Tensor) -> Tensor:
        """Alias used by the full-field training worker."""
        return self.predict_points(branch, coords)

    def forward(self, input_maps: Tensor, coords: Tensor) -> Tensor:
        return self.predict_points(self.encode_branch(input_maps), coords)


def build_model(
    kind: str,
    target_parameters: int | None = TARGET_TRAINABLE_REAL_PARAMETERS,
    *,
    architecture: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> FNO2d | DeepONet:
    """Construct a model and fail closed if the requested count does not match.

    ``architecture`` (or keyword overrides) can restore an explicit saved
    configuration. Set ``target_parameters=None`` only for small CPU fixtures
    or another explicitly counted architecture. This function never silently
    changes a requested target, appends unused parameters, or upgrades models.
    """
    kind = kind.lower()
    constructors = {"fno": (FNO2d, FNO_CONFIGURATION),
                    "deeponet": (DeepONet, DEEPONET_CONFIGURATION)}
    if kind not in constructors:
        raise ValueError(f"Unknown baseline {kind!r}; expected 'fno' or 'deeponet'")
    constructor, defaults = constructors[kind]
    configuration = dict(defaults)
    if architecture is not None:
        configuration.update(architecture)
    configuration.update(overrides)
    model = constructor(**configuration)
    if target_parameters is not None:
        _positive_integer(target_parameters, "target_parameters")
        report = parameter_report(model, target_parameters)
        if not report["exact_match"]:
            raise ValueError(
                f"{kind} has {report['trainable_real_scalars']} trainable real scalars, "
                f"requested {target_parameters}; difference {report['difference']:+d}. "
                "Provide an explicitly reviewed matching architecture."
            )
    return model
