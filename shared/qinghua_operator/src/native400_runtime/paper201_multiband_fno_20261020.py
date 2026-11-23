"""Independent temporal-band FNO corrections for the registered paper operator.

The pretrained full-band backbone is retained. Each new spatial FNO predicts
coefficients of a *specified temporal* Fourier basis, evaluated at actual query
times. There is no FFT of the irregular 400-frame axis, inverse coefficient fit,
pressure input, temporal interpolation, or restriction of the parent spectrum.
The 2-second basis period avoids equating the physical endpoints 0 and 1 s.
These are additive band corrections, not a decomposition of the parent model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .paper201_hermitian_spectral_20261020 import HermitianSpectralConv2d

SCHEMA = "PAPER201_MULTIBAND_FNO_CORRECTIONS_V1"


@dataclass(frozen=True)
class MultiBandConfig:
    cutoffs_hz: tuple = (8., 12., 16., 20., 40.)
    period_s: float = 2.
    width: int = 12
    spatial_modes: int = 6
    depth: int = 2
    internal_sizes: tuple = (51, 51, 101, 101, 101)
    activation_steps: tuple = (0, 100, 200, 300, 400)
    coefficient_l2: float = 1e-6

    def __post_init__(self):
        for name in ('cutoffs_hz', 'internal_sizes', 'activation_steps'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.cutoffs_hz)
        if not n or len(self.internal_sizes) != n or len(self.activation_steps) != n:
            raise ValueError('one grid and activation step per temporal band required')
        if (not math.isfinite(self.period_s) or self.period_s <= 1.
                or not math.isfinite(self.coefficient_l2) or self.coefficient_l2 < 0):
            raise ValueError('period > physical one-second window and nonnegative regularization required')
        if any(type(v) is not int or v < 1 for v in (self.width, self.spatial_modes, self.depth)):
            raise ValueError('positive integer FNO dimensions required')
        if any(type(v) is not int or v < max(3, 2*self.spatial_modes+1) for v in self.internal_sizes):
            raise ValueError('internal spatial grids must resolve retained spatial modes')
        if (self.activation_steps[0] != 0
                or any(type(v) is not int or v < 0 for v in self.activation_steps)
                or any(b <= a for a, b in zip(self.activation_steps, self.activation_steps[1:]))):
            raise ValueError('strictly increasing activation steps starting at zero required')
        previous = 0.
        for cutoff in self.cutoffs_hz:
            if (not math.isfinite(cutoff) or cutoff <= previous
                    or abs(cutoff*self.period_s-round(cutoff*self.period_s)) > 1e-9):
                raise ValueError('increasing cutoffs on the temporal basis grid required')
            previous = cutoff

    def active_bands(self, step):
        if type(step) is not int or step < 0:
            raise ValueError('nonnegative committed curriculum step required')
        return sum(step >= start for start in self.activation_steps)


class _SpatialBlock(nn.Module):
    def __init__(self, width, modes, size):
        super().__init__()
        # Reuse the independently audited minimal Hermitian parameterization.
        shape = (width, width, modes, modes, 2)
        source = SimpleNamespace(in_channels=width, out_channels=width,
            modes_y=modes, modes_x=modes, fft_norm='ortho',
            weight_top=nn.Parameter(torch.randn(shape)/width),
            weight_bottom=nn.Parameter(torch.randn(shape)/width))
        self.spectral = HermitianSpectralConv2d(source, size, size)
        self.local = nn.Conv2d(width, width, 1)

    def forward(self, value):
        return F.gelu(self.spectral(value)+self.local(value))


class TemporalBandFNO(nn.Module):
    """One independent FNO, with temporal support fixed by its coefficient head."""
    def __init__(self, config: MultiBandConfig, index: int):
        super().__init__()
        if not 0 <= index < len(config.cutoffs_hz):
            raise ValueError('band index outside configuration')
        self.config, self.index = config, index
        self.size = config.internal_sizes[index]
        low = 0 if index == 0 else round(config.cutoffs_hz[index-1]*config.period_s)+1
        high = round(config.cutoffs_hz[index]*config.period_s)
        modes = list(range(low, high+1))
        frequencies, sine = [], []
        for mode in modes:
            frequencies.append(mode/config.period_s); sine.append(False)
            if mode:
                frequencies.append(mode/config.period_s); sine.append(True)
        self.register_buffer('frequencies', torch.tensor(frequencies, dtype=torch.float32))
        self.register_buffer('sine_columns', torch.tensor(sine, dtype=torch.bool))
        self.lift = nn.Conv2d(8, config.width, 1)
        self.blocks = nn.ModuleList(_SpatialBlock(config.width, config.spatial_modes, self.size)
                                   for _ in range(config.depth))
        self.output = nn.Conv2d(config.width, len(frequencies), 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        if (features.ndim != 4 or features.shape[1] != 8 or min(features.shape[-2:]) < 3
                or not features.is_floating_point() or not bool(torch.isfinite(features).all())):
            raise ValueError('finite [batch,8,z,x] physical features required')
        native_shape = features.shape[-2:]
        hidden = F.interpolate(features, size=(self.size, self.size), mode='bilinear', align_corners=True)
        hidden = self.lift(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        # Only hidden features are interpolated; coefficients are predicted on
        # native nodes by the final convolution. No final pressure interpolation.
        hidden = F.interpolate(hidden, size=native_shape, mode='bilinear', align_corners=True)
        coefficients = self.output(hidden)
        return torch.cat((torch.zeros_like(coefficients[:, :, :1]), coefficients[:, :, 1:]), dim=-2)

    def basis(self, times):
        times = torch.as_tensor(times, device=self.frequencies.device, dtype=self.frequencies.dtype)
        if (times.ndim != 1 or times.numel() == 0 or not bool(torch.isfinite(times).all())
                or bool((times < 0).any()) or bool((times > 1).any())):
            raise ValueError('nonempty physical times in [0,1] required')
        phase = (2*math.pi)*times[:, None]*self.frequencies[None]
        return torch.where(self.sine_columns[None], phase.sin(), phase.cos())


class BandCorrections(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = MultiBandConfig() if config is None else config
        self.heads = nn.ModuleList(TemporalBandFNO(self.config, i)
                                   for i in range(len(self.config.cutoffs_hz)))
        self.register_load_state_dict_post_hook(_verify_temporal_buffers)

    def get_extra_state(self):
        return dict(schema=SCHEMA, config=asdict(self.config),
                    temporal_support='fixed Fourier synthesis; no inverse fit',
                    high_frequency_remainder='retained full-band backbone')

    def set_extra_state(self, value):
        if value != self.get_extra_state():
            raise ValueError('multiband checkpoint configuration mismatch')

    def coefficients(self, features):
        return [head(features) for head in self.heads]

    def synthesize(self, coefficients, times, step):
        if len(coefficients) != len(self.heads):
            raise ValueError('one coefficient field per temporal band required')
        active = self.config.active_bands(step)
        result = None
        for i, (head, values) in enumerate(zip(self.heads, coefficients)):
            if values.ndim != 4 or values.shape[1] != head.frequencies.numel():
                raise ValueError('coefficient shape does not match temporal basis')
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError('nonfinite band coefficients')
            value = torch.einsum('tk,bkzx->btzx', head.basis(times), values)
            value = value * float(i < active)
            result = value if result is None else result + value
        return result

    def regularization(self, coefficients, step):
        active = self.config.active_bands(step)
        return self.config.coefficient_l2 * sum(
            values.square().mean()*float(i < active) for i, values in enumerate(coefficients))


def _verify_temporal_buffers(module, incompatible_keys):
    """Configuration checks alone cannot detect tampered same-shape buffers."""
    for i, head in enumerate(module.heads):
        low = 0 if i == 0 else round(module.config.cutoffs_hz[i-1]*module.config.period_s)+1
        high = round(module.config.cutoffs_hz[i]*module.config.period_s)
        frequency, sine = [], []
        for mode in range(low, high+1):
            frequency.append(mode/module.config.period_s); sine.append(False)
            if mode:
                frequency.append(mode/module.config.period_s); sine.append(True)
        expected = head.frequencies.new_tensor(frequency)
        flags = head.sine_columns.new_tensor(sine)
        if not torch.equal(head.frequencies, expected) or not torch.equal(head.sine_columns, flags):
            raise ValueError('loaded temporal basis differs from configured physical frequencies')


class MultiBandPaperOperator(nn.Module):
    """Keep every parent branch, adding zero-initialized band-specific FNOs.

    Curriculum controls the new corrections only. The inherited backbone stays
    trainable and retains its full spectrum. PDE and IC constrain the *total*
    prediction. The optional coefficient penalty bounds extrapolation without
    estimating ill-conditioned spectral coefficients from irregular samples.
    """
    def __init__(self, backbone, config=None):
        super().__init__()
        self.backbone = backbone
        self.config = MultiBandConfig() if config is None else config
        self.bands = BandCorrections(self.config).to(device=self.device)
        self.register_buffer('curriculum_step', torch.tensor(0, dtype=torch.int64))
        self.register_load_state_dict_post_hook(_verify_curriculum_buffer)

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    @property
    def normalizer(self):
        return self.backbone.normalizer

    @property
    def full_times_s(self):
        return self.backbone.full_times_s

    @property
    def native_times_s(self):
        return self.backbone.native_times_s

    @property
    def full_native_frames(self):
        return self.backbone.full_native_frames

    def scopes(self):
        return self.backbone.scopes()

    def set_training_step(self, step):
        self.config.active_bands(step)
        self.curriculum_step.fill_(step)

    def input_features(self, c, f):
        c, f = self.backbone._inputs(c, f)
        n, m = c.shape[-2:]
        z, x = torch.meshgrid(torch.linspace(0, 2000, n, device=c.device, dtype=c.dtype),
                             torch.linspace(0, 2000, m, device=c.device, dtype=c.dtype), indexing='ij')
        planes = torch.stack((x/1000-1, z/1000-1, (x-f[0, 0])/2000,
                              (z-f[0, 1])/2000, torch.ones_like(x)*f[0, 2]/15,
                              torch.ones_like(x)*f[0, 3]))[None]
        source = self.backbone._source_map(f, n, 2000/(n-1))
        return c, f, torch.cat((c/3000-1, source, planes), dim=1)

    def coefficient_penalty(self, c, f):
        _, _, features = self.input_features(c, f)
        return self.bands.regularization(self.bands.coefficients(features), int(self.curriculum_step))

    def forward(self, c, f, times_s=None, *, normalized=True):
        c, f, features = self.input_features(c, f)
        times = self.full_times_s if times_s is None else times_s
        times = torch.as_tensor(times, device=self.device, dtype=torch.float32).reshape(-1)
        coefficients = self.bands.coefficients(features)
        correction = self.bands.synthesize(coefficients, times, int(self.curriculum_step))[0]
        value = self.backbone.forecast_at_times(c, f, times, normalized=True)+correction
        return value if normalized else self.normalizer.decode_pressure(value[None], f[:, 4])[0]

    def forecast_at_times(self, c, f, times_s, *, normalized=True):
        return self(c, f, times_s, normalized=normalized)

    @torch.no_grad()
    def forward_cf_full(self, c, f):
        was = self.training
        self.eval()
        try:
            value = self(c, f, normalized=False)
            if value.shape != (400, 201, 201) or value.dtype != torch.float32:
                raise ValueError('registered paper delivery requires FP32 [400,201,201]')
            if not bool(torch.isfinite(value).all()) or bool((value[:, 0] != 0).any()):
                raise ValueError('finite pressure and exact top boundary required')
            return value.permute(1, 2, 0).contiguous()
        finally:
            self.train(was)


class CurriculumAdamW(torch.optim.AdamW):
    """Dormant heads retain exactly their weights, moments and zero step age.

    Unlike lr=0 alone, dropping their gradients *during* AdamW.step also avoids
    advancing their bias-correction clock. Connected zero gradients are restored
    afterwards for the existing fixed-block learned-controller interface.
    """
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('curriculum optimizer uses explicit pre/post objectives, no closure')
        removed = []
        for group in self.param_groups:
            if any(group.get(option, False) for option in ('amsgrad', 'capturable', 'differentiable', 'fused')):
                raise ValueError('curriculum AdamW requires default unfused noncapturable moment layout')
            if not group.get('band_active', True):
                for parameter in group['params']:
                    removed.append((parameter, parameter.grad))
                    parameter.grad = None
        try:
            result = super().step()
            # The conditional block controller observes zero moments for unused
            # heads; these are explicit untrained coordinates with age zero.
            for parameter, _ in removed:
                if not self.state[parameter]:
                    self.state[parameter] = dict(step=torch.tensor(0.),
                        exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
            return result
        finally:
            for parameter, gradient in removed:
                parameter.grad = gradient


def _verify_curriculum_buffer(module, incompatible_keys):
    if module.curriculum_step.shape != () or module.curriculum_step.dtype != torch.int64:
        raise ValueError('scalar integer curriculum step required')
    module.config.active_bands(int(module.curriculum_step))


def set_curriculum(model, optimizer, step):
    model.set_training_step(step)
    active = model.config.active_bands(step)
    for group in optimizer.param_groups:
        index = group.get('temporal_band')
        if index is not None:
            group['band_active'] = index < active
    return active


def backward_task(model, c, f, truth, sampler, weights=None, plan=None):
    """Reuse the complete original field/receiver/SIPG/IC objective, then regularize."""
    from .marmousi_direct400_continue_20261020 import backward_task as original_backward
    metrics, feedback = original_backward(model, c, f, truth, sampler, weights=weights, plan=plan)
    penalty = model.coefficient_penalty(c, f)
    weight = 1. if weights is None else weights.get('weight', 1.)
    (weight*penalty).backward()
    value = float(penalty.detach())
    metrics.update(coefficient_loss=value, loss=metrics['loss']+value,
                   weighted_loss=metrics['weighted_loss']+weight*value,
                   active_temporal_bands=model.config.active_bands(int(model.curriculum_step)))
    return metrics, feedback
