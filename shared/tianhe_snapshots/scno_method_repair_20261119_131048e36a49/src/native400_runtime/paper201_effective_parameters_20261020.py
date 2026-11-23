"""Opt-in removal of proven null coordinates in the archived Marmousi model.

The original checkpoint and default architecture remain available. This adapter
preserves every output/physical branch, the 201 grid and all 400 query times.
It is NOT a full-state migration for a running joint learned-optimizer job:
parameter shapes and the learned optimizer's block-statistic denominators change.
Construct new optimizer/config/checkpoint receipts before starting a new job.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'PAPER201_PROVEN_NULL_PARAMETER_COMPACTION_V1'
REGISTERED_ORIGINAL_CHECKPOINTS = {
    'bc3b955472b59a3d26cdacfbeff5641e07deccfa0b5c7422dee36ad00e9210ff': 'paper_epoch65',
    '60fb5ac2a2072e469d9be7250eb7058b848a5dba05954388a9bb825ea39f9db5': 'paper201_fullfit2000',
}


class RealRadialSpectralConv2d(nn.Module):
    """Real radial multiplier on the full signed FFT lattice.

    A radial multiplier has W(k)=W(-k). For real inputs, taking the real part
    of IFFT therefore cancels Im(W). Store only Re(W). Ordinary signed rFFT
    kernels are intentionally left complex.
    """
    def __init__(self, source):
        super().__init__()
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.cutoff = source.cutoff
        self.fft_norm = source.fft_norm
        self.bucket_count = source.bucket_count
        self.s0_kmajor_bmm = getattr(source, 's0_kmajor_bmm', True)
        self.weight = nn.Parameter(source.weight.detach()[..., 0].clone(),
                                   requires_grad=source.weight.requires_grad)
        self._lattice = {}

    def _cached_lattice(self, height, width, device):
        key = (height, width, str(device))
        if key not in self._lattice:
            ky = (torch.fft.fftfreq(height)*height).round().long()
            kx = (torch.fft.fftfreq(width)*width).round().long()
            squared = ky[:, None].square()+kx[None, :].square()
            mask = squared <= self.cutoff**2
            radii = torch.unique(squared[mask])
            if radii.numel() != self.bucket_count:
                raise ValueError('radial cutoff not resolved by this grid')
            positions = mask.reshape(-1).nonzero(as_tuple=True)[0]
            buckets = torch.searchsorted(radii, squared.reshape(-1)[positions])
            self._lattice[key] = positions.to(device), buckets.to(device)
        return self._lattice[key]

    def forward(self, value):
        if value.ndim != 4 or value.shape[1] != self.in_channels:
            raise ValueError('spectral input must be [batch,in_channels,z,x]')
        height, width = value.shape[-2:]
        if min(height, width) < 2*self.cutoff+1:
            raise ValueError('radial cutoff not resolved by this grid')
        positions, buckets = self._cached_lattice(height, width, value.device)
        with torch.autocast(device_type=value.device.type, enabled=False):
            transform = torch.fft.fft2(value.float(), norm=self.fft_norm)
            retained = transform.flatten(-2)[:, :, positions]
            kernel = self.weight[:, :, buckets].to(transform.dtype)
            if self.s0_kmajor_bmm:
                contracted = torch.bmm(retained.permute(2,0,1).contiguous(),
                                       kernel.permute(2,0,1).contiguous()).permute(1,2,0)
            else:
                contracted = torch.einsum('bik,iok->bok', retained, kernel)
            output = transform.new_zeros(value.shape[0], self.out_channels, height*width)
            output[:, :, positions] = contracted
            return torch.fft.ifft2(output.reshape(-1,self.out_channels,height,width),
                                   norm=self.fft_norm).real


class ZeroInputBiasConv2d(nn.Module):
    """Exact zero-input convolution, keeping its useful trained bias field."""
    def __init__(self, source):
        super().__init__()
        if source.groups != 1 or source.bias is None or source.weight.requires_grad:
            raise ValueError('expected permanently frozen zero-IC convolution')
        self.in_channels, self.out_channels = source.in_channels, source.out_channels
        self.kernel_size, self.stride = source.kernel_size, source.stride
        self.padding, self.dilation = source.padding, source.dilation
        self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=source.bias.requires_grad)

    def forward(self, value):
        if value.ndim != 4 or value.shape[1] != self.in_channels or bool(value.detach().any()):
            raise ValueError('this c/f-only model requires identically zero internal IC inputs')
        shape = [(value.shape[-2+i]+2*self.padding[i]-self.dilation[i]*(self.kernel_size[i]-1)-1)
                 //self.stride[i]+1 for i in range(2)]
        return self.bias[None,:,None,None].expand(value.shape[0], -1, *shape)


class ActiveStaticConv2d(nn.Conv2d):
    """Keep sharp velocity, source map and four coordinates from the static14."""
    keep = (0,9,10,11,12,13)

    @classmethod
    def from_original(cls, source):
        if source.in_channels != 14 or source.groups != 1:
            raise ValueError('expected archived fourteen-channel static input')
        with torch.random.fork_rng(devices=[]):
            result = cls(len(cls.keep), source.out_channels, source.kernel_size,
                         source.stride, source.padding, source.dilation,
                         bias=source.bias is not None, padding_mode=source.padding_mode,
                         device=source.weight.device, dtype=source.weight.dtype)
        result.weight = nn.Parameter(source.weight.detach()[:, cls.keep].clone(),
                                     requires_grad=source.weight.requires_grad)
        if source.bias is not None:
            result.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=source.bias.requires_grad)
        # The old zero-IC hook belongs to the discarded parameter. Do not attach
        # [:,1:9] to the six useful channels. The original zero-decay group remains.
        return result

    def forward(self, value):
        if value.shape[1] != 14 or bool(value[:,1:9].detach().any()):
            raise ValueError('expected sharp velocity / zero8 / source / coordinates')
        return self._conv_forward(value[:,self.keep], self.weight, self.bias)


def effective_trainable_receipt(model):
    """Current parameter contract; the discarded fourteen-channel mask is absent."""
    return dict(schema=SCHEMA,
        active=[dict(name=n,shape=list(p.shape),numel=p.numel())
                for n,p in model.named_parameters() if p.requires_grad],
        frozen=[n for n,p in model.named_parameters() if not p.requires_grad],
        inactive_refiner_slice=None,
        source_IC_first_layer='bias broadcast from strictly zero inputs; no weight parameter',
        refiner_static_original_input_columns=list(ActiveStaticConv2d.keep),
        refiner_static_gradient_mask=None,
        required_optimizer_group='refiner_masked_input with weight_decay=0; six useful channels, no gradient mask',
        optimizer_and_joint_state_migration_required=True,
        active_group_gradient_and_parameter_delta_must_be_measured=True)


def compact_null_parameters(model):
    """Mutate a private, strictly restored Marmousi model; return its receipt."""
    if getattr(model, 'family', None) != 'marmousi' or hasattr(model, '_effective_parameter_receipt'):
        raise ValueError('one uncompacted Marmousi model required')
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('compact a private CPU model before moving it to an accelerator')
    original_names = [n for n,p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    radial = model.base.dense_decoder.blocks[0].spectral
    if type(radial).__name__ != 'EquivariantSpectralConv2d' or radial.weight.shape[-1] != 2:
        raise ValueError('expected archived complex radial spectrum')
    medium = model.base.medium_encoder.blocks[3].spectral
    my,mx = medium.retained_modes(13,13)
    if (my,mx) != (6,7) or tuple(medium.weight_top.shape[2:4]) != (8,8):
        raise ValueError('expected archived 101/51/26/13 medium pyramid')
    first_ic = model.base.source_encoder.ic_encoder[0]
    first_static = model.refiner.static[0].conv1
    if (first_ic.groups != 1 or first_ic.bias is None or first_ic.weight.requires_grad
            or first_static.in_channels != 14 or first_static.groups != 1):
        raise ValueError('expected archived zero-IC source and static input convolutions')
    removed_radial = radial.weight[...,1].numel()
    model.base.dense_decoder.blocks[0].spectral = RealRadialSpectralConv2d(radial)
    removed_medium = 0
    for key in ('weight_top','weight_bottom'):
        parameter = getattr(medium,key)
        cropped = parameter.detach()[:,:,:my,:mx,:].clone()
        removed_medium += parameter.numel()-cropped.numel()
        setattr(medium,key,nn.Parameter(cropped, requires_grad=parameter.requires_grad))
    medium.modes_y,medium.modes_x = my,mx
    removed_frozen = first_ic.weight.numel()
    model.base.source_encoder.ic_encoder[0] = ZeroInputBiasConv2d(first_ic)
    compact = ActiveStaticConv2d.from_original(first_static)
    removed_static = first_static.weight.numel()-compact.weight.numel()
    model.refiner.static[0].conv1 = compact
    if original_names != [n for n,p in model.named_parameters() if p.requires_grad]:
        raise AssertionError('active tensor names/order must remain unchanged')
    receipt = dict(schema=SCHEMA, original_total_parameters=total,
        original_trainable_parameters=active,
        effective_total_parameters=sum(p.numel() for p in model.parameters()),
        effective_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        removed=dict(radial_imaginary=removed_radial, unused_medium_modes=removed_medium,
                     frozen_zeroIC_weight=removed_frozen, masked_static_zeroIC_weight=removed_static),
        active_tensor_names_and_order_unchanged=True, original_checkpoints_unchanged=True,
        all_physics_and_output_branches_preserved=True, full400_grid_preserved=True,
        exact_arithmetic_output_equivalent=True, fp32_bitwise_equivalence_claim=False,
        joint_training_resume_supported=False,
        caution='Not all parameter identifiability redundancies are removed. New optimizer and learned-optimizer feature contracts required.')
    model.__dict__['_effective_parameter_receipt'] = receipt
    model.__dict__['trainable_receipt'] = MethodType(effective_trainable_receipt, model)
    historical = copy.deepcopy(model.initialization_receipt)
    model.initialization_receipt.update(parameter_schema=SCHEMA,
        archival_constructor_receipt=historical,
        source_IC_first_conv_weight_frozen=None,
        source_IC_first_layer='trainable bias broadcast, frozen weight removed',
        refiner_zero_IC_weight_slices=None,
        refiner_static_original_input_columns=list(ActiveStaticConv2d.keep),
        refiner_static_gradient_mask=None,
        effective_parameter_receipt=receipt)
    return receipt


def build_effective_paper_operator(config=None, checkpoint=None, expected_sha256=None, device='cpu'):
    """Default: epoch65. The registered paper201 fit2000 parent is also allowed.

    A compact state is restored by calling this factory, then strict-loading that
    compact state_dict. This factory never loads old optimizer or Gphi states.
    """
    from .r25_cf_full400_operator import build_r25_cf_full400_operator
    asset = Path(__file__).resolve().parents[2] / "paper_assets_main_fresh_20261013_marmousi_20261019"
    entry = next(v for v in json.loads((asset/'MODEL_INDEX.json').read_text())
                 if v['name']=='marmousi_epoch65')
    cfg = copy.deepcopy(config if config is not None else json.loads((asset/entry['config']).read_text()))
    cfg['parent_checkpoint'] = str(asset/'models/initializers/marmousi.pt')
    if checkpoint is None:
        checkpoint, expected_sha256 = asset/entry['local_path'], entry['sha256']
    if not expected_sha256:
        raise ValueError('explicit SHA required for custom original-shape checkpoint')
    if expected_sha256 not in REGISTERED_ORIGINAL_CHECKPOINTS:
        raise ValueError('only registered paper201 checkpoints with the reviewed normalizer/grid/source contract are supported')
    checkpoint = Path(checkpoint)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('checkpoint SHA differs')
    model = build_r25_cf_full400_operator(cfg, device='cpu', kind='marmousi')
    saved = torch.load(str(checkpoint), map_location='cpu', weights_only=False, mmap=True)
    state = saved.get('model_state', saved.get('model'))
    if not isinstance(state, dict):
        raise ValueError('full original-shape model state required')
    model.load_state_dict(state, strict=True)
    compact_null_parameters(model)
    return model.to(device)
