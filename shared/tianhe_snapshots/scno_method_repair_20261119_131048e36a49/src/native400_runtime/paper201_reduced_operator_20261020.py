"""V2: remove the audited structural parameter redundancies, preserving branches.

The V1 null-coordinate adapter remains unchanged and reproducible. This new
entry additionally ties Hermitian rFFT columns, merges duplicate additive
features/biases and reduces affine conditioning features. A new optimizer and
learned-optimizer contract is required; no running job is migrated here.
"""
from __future__ import annotations
import copy
import hashlib
import json
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from .paper201_effective_parameters_20261020 import (
    build_effective_paper_operator, effective_trainable_receipt)
from .r25_cf_full400_operator import R25CFFull400MarmousiDGOperator

SCHEMA = 'PAPER201_STRUCTURAL_REDUNDANCY_REDUCTION_V2'


class ReducedPaperOperator(R25CFFull400MarmousiDGOperator):
    """Post-restore type with a persistent semantic contract for compact state."""
    def __init__(self, *args, **kwargs):
        raise TypeError('use build_reduced_paper_operator to restore and reduce registered original weights')

    def get_extra_state(self):
        from .paper201_conditioning_reduction_20261020 import conditioning_metadata
        normalizer = vars(self.normalizer.metadata)
        return dict(schema=SCHEMA, conditioning=conditioning_metadata(self),
            native_time_fp32_sha256=hashlib.sha256(self.native_times_s.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            registered_native_frame_indices=list(self.full_native_frames),
            normalizer_sha256=hashlib.sha256(json.dumps(normalizer,sort_keys=True).encode()).hexdigest(),
            public_grid=[201,201,400],physical_time_propagator=False,
            fixed_physics_features=['laplacian8','dz8','dx8'],
            source_contract='original x/z domain, f0/t0 validation tolerances, arbitrary positive amplitude',
            old_optimizer_direct_resume_supported=False)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('reduced checkpoint protocol/conditioning/normalizer/time schema differs')
        self.__dict__['_loaded_reduction_contract'] = copy.deepcopy(state)


def reduced_state_post_load(module, incompatible_keys):
    """Check persistent child constants after PyTorch has loaded all descendants."""
    expected = module.__dict__.pop('_loaded_reduction_contract', None)
    if expected is None or module.get_extra_state() != expected:
        raise ValueError('loaded child constants differ from the reduced checkpoint semantic contract')


class DerivativeOnlyPhysics(nn.Module):
    """Keep all three fixed eighth-order derivative channels, with no duplicate value/bias."""
    def __init__(self, source):
        super().__init__()
        self.width = source.width
        self.register_buffer('stencils', source.stencils.detach().clone(), persistent=False)
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Conv2d(3, self.width, 1, bias=False,
                device=source.projection.weight.device, dtype=source.projection.weight.dtype)
        self.projection.weight = nn.Parameter(source.projection.weight.detach()[:,1:].clone(),
            requires_grad=source.projection.weight.requires_grad)

    def features(self, coarse):
        if coarse.ndim != 4 or coarse.shape[1] != 1 or min(coarse.shape[-2:]) < 9:
            raise ValueError('fixed derivative prior expects [batch,1,z,x] on a grid of at least9')
        return F.pad(F.conv2d(coarse, self.stencils.to(coarse.dtype)), (4,4,4,4))

    def forward(self, coarse):
        return self.projection(self.features(coarse))


def reduced_trainable_receipt(model):
    receipt = effective_trainable_receipt(model)
    receipt.update(schema=SCHEMA,
        structural_reduction=copy.deepcopy(model._redundancy_reduction_receipt),
        active_tensor_names_and_shapes_changed=True,
        old_joint_optimizer_state_direct_load_supported=False)
    return receipt


def reduce_structural_redundancy(model):
    """Reduce one private CPU V1 model; the production model is never used."""
    from .paper201_hermitian_spectral_20261020 import HermitianSpectralConv2d
    from .paper201_conditioning_reduction_20261020 import reduce_conditioning
    if (getattr(model, 'family', None) != 'marmousi'
            or not hasattr(model, '_effective_parameter_receipt')
            or hasattr(model, '_redundancy_reduction_receipt')):
        raise ValueError('one V1 private Marmousi model is required')
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('reduce the CPU model before moving it to an accelerator')
    before = sum(p.numel() for p in model.parameters())
    initial_rng = torch.get_rng_state().clone()
    spectral_receipts = []
    locations = [(f'base.medium_encoder.blocks.{i}.spectral', block, size)
                 for i,(block,size) in enumerate(zip(model.base.medium_encoder.blocks, (101,51,26,13)))]
    if model.base.medium_encoder.u_enhance is not None:
        locations.append(('base.medium_encoder.u_enhance.block51.spectral',
                          model.base.medium_encoder.u_enhance.block51, 26))
    for name, parent, size in locations:
        old = parent.spectral
        replacement = HermitianSpectralConv2d.from_original(old, size, size)
        parent.spectral = replacement
        spectral_receipts.append(dict(module=name, removed_parameters=sum(p.numel() for p in old.parameters())-
            sum(p.numel() for p in replacement.parameters())))
    dense = model.base.dense_decoder
    old_physics = dense.fixed_wave_physics
    lift = dense.coarse_lift
    if (old_physics is None or old_physics.projection.weight.shape[1] != 4
            or lift.kernel_size != (3,3) or lift.bias is None
            or old_physics.projection.bias is None):
        raise ValueError('expected archived coarse lift and four-feature fixed physical prior')
    with torch.no_grad():
        lift.weight[:,0,1,1].add_(old_physics.projection.weight[:,0,0,0])
        lift.bias.add_(old_physics.projection.bias)
    dense.fixed_wave_physics = DerivativeOnlyPhysics(old_physics)
    physics_removed = old_physics.width*2
    conditioning = reduce_conditioning(model)
    if not torch.equal(initial_rng, torch.get_rng_state()):
        raise AssertionError('structural reduction changed CPU RNG')
    after = sum(p.numel() for p in model.parameters())
    null_receipt = copy.deepcopy(model._effective_parameter_receipt)
    receipt = dict(schema=SCHEMA, original_parameters=null_receipt['original_total_parameters'],
        original_trainable_parameters=null_receipt['original_trainable_parameters'],
        V1_parameters=before, effective_parameters=after,
        effective_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        active_parameter_tensors=sum(p.requires_grad for p in model.parameters()),
        removed_after_V1=before-after, removed_from_original=null_receipt['original_total_parameters']-after,
        radial_and_zeroIC_reduction=null_receipt, Hermitian_spectral=spectral_receipts,
        duplicate_fixed_value_and_bias_removed=physics_removed, conditioning=conditioning,
        all_physics_and_output_branches_preserved=True, all400_and_grid_preserved=True,
        legal_source_frequency_delay_tolerance_and_amplitude_preserved=True,
        old_AdamW_and_Gphi_direct_resume_supported=False,
        exact_arithmetic_output_equivalent=True, fp32_bitwise_equivalence_claim=False,
        remaining_scope='No claim of global neural-network parameter identifiability; audited structural duplicate features are reduced.')
    model.__dict__['_redundancy_reduction_receipt'] = receipt
    # Both classes are ordinary Python nn.Module subclasses with the same
    # instance layout. Tensor objects are retained; the derived class adds only
    # state-contract methods. PyTorch checks get_extra_state on the class, so an
    # instance-bound method alone would silently omit this persistent contract.
    model.__class__ = ReducedPaperOperator
    model.register_load_state_dict_post_hook(reduced_state_post_load)
    model.__dict__['trainable_receipt'] = MethodType(reduced_trainable_receipt, model)
    model.initialization_receipt.update(parameter_schema=SCHEMA,
        null_coordinate_compaction_receipt=null_receipt, effective_parameter_receipt=receipt,
        structural_reduction_receipt=receipt)
    return receipt


def legacy_parameter_gradients(model, original_shapes):
    """Pull V2 gradients back to original coordinates using g_old=A^T g_new.

    Compare this result with an original-model backward. Duplicate summed
    coordinates each receive the merged gradient; they are never added twice.
    """
    from .paper201_hermitian_spectral_20261020 import HermitianSpectralConv2d
    from .paper201_conditioning_reduction_20261020 import legacy_conditioning_gradients
    if not hasattr(model, '_redundancy_reduction_receipt'):
        raise ValueError('V2 reduced model required')
    result = {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    name = 'base.dense_decoder.blocks.0.spectral.weight'
    result[name] = torch.stack((result[name], torch.zeros_like(result[name])), dim=-1)
    name = 'refiner.static.0.conv1.weight'
    static = result[name].new_zeros(original_shapes[name])
    static[:,(0,9,10,11,12,13)] = result[name]
    result[name] = static
    for name,module in model.named_modules():
        if isinstance(module, HermitianSpectralConv2d):
            for suffix,gradient in module.legacy_gradient_map().items():
                key = name+'.'+suffix
                if key not in original_shapes:
                    raise KeyError('unregistered legacy spectral gradient '+key)
                expanded = gradient.new_zeros(original_shapes[key])
                expanded[:,:,:gradient.shape[2],:gradient.shape[3],:] = gradient
                result[key] = expanded
    name = 'base.dense_decoder.fixed_wave_physics.projection.weight'
    weight = result[name].new_zeros(original_shapes[name])
    weight[:,1:] = result[name]
    lift_gradient = result['base.dense_decoder.coarse_lift.weight']
    weight[:,0,0,0] = lift_gradient[:,0,1,1]
    result[name] = weight
    result['base.dense_decoder.fixed_wave_physics.projection.bias'] = result['base.dense_decoder.coarse_lift.bias'].clone()
    result.update(legacy_conditioning_gradients(model))
    missing = set(original_shapes)-set(result)
    if missing:
        raise AssertionError('missing original gradient coordinates: '+str(sorted(missing)))
    return {name:result[name] for name in original_shapes}


def build_reduced_paper_operator(config=None, checkpoint=None, expected_sha256=None, device='cpu'):
    """Strict registered original weights -> private CPU V1 -> reviewed V2 -> device."""
    model = build_effective_paper_operator(config, checkpoint, expected_sha256, device='cpu')
    reduce_structural_redundancy(model)
    return model.to(device)
