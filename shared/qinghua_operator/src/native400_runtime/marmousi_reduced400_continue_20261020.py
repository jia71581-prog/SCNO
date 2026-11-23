"""V2 joint continuation: project final theta and explicitly rebase changed coordinates.

The full400 objective and one-step conditional controller VJP are reused unchanged.
Only untransformed, equal parameter coordinates retain AdamW history. New/merged
coordinates and meta-Adam restart; this is not an exact old optimizer trajectory.
"""
from __future__ import annotations
import copy
from pathlib import Path

import torch

from . import marmousi_direct400_continue_20261020 as joint
from .paper201_effective_parameters_20261020 import compact_null_parameters
from .paper201_reduced_operator_20261020 import reduce_structural_redundancy
from .r25_cf_full400_operator import build_r25_cf_full400_operator

SCHEMA = 'MARMOUSI_REDUCED400_JOINT_CONTINUATION_V2'
PARENT = joint.ROOT/'runs/marmousi_joint400_learned_pretrain_20261020/latest.pt'
PARENT_SHA = '10a19a4cad1a92b8656ef34d1316a8de5d17a7afacb6025c1bf6068032612d82'
INHERITED_JOINT_UPDATES = 200
EXPECTED_TENSORS, EXPECTED_PARAMETERS, META_PARAMETERS = 393, 12667079, 5481
# Explicit semantic changes are fresh even when particular old values happen to
# make the numeric projection an identity. Shape/equality alone is insufficient.
TRANSFORMED = frozenset((
    'base.dense_decoder.blocks.0.spectral.weight', 'refiner.static.0.conv1.weight',
    'base.source_encoder.parameter_mlp.2.bias',
    'base.coordinate_encoder.raw_projection.weight', 'base.coordinate_encoder.raw_projection.bias',
    'base.coordinate_encoder.periodic_in.linear.weight', 'base.scno_cycle.La.0.weight',
    'base.dense_decoder.multiscale_fuse.bias', 'base.dense_decoder.coarse_lift.weight',
    'base.dense_decoder.coarse_lift.bias', 'base.dense_decoder.fixed_wave_physics.projection.weight'))

# Keep the existing numerical algorithm in one place.
BlockOptimizerConfig = joint.BlockOptimizerConfig
BlockLearnedOptimizer = joint.BlockLearnedOptimizer
make_learned = joint.make_learned
propose_update = joint.propose_update
finish_meta = joint.finish_meta
backward_task = joint.backward_task
episode_inputs = joint.episode_inputs
sha, cpu_tree, equal_tree = joint.sha, joint.cpu_tree, joint.equal_tree
require, immutable_digest, trainable_digest = joint.require, joint.immutable_digest, joint.trainable_digest
fixed_supervision, pressure_metrics, assess, dg_geometry = (
    joint.fixed_supervision, joint.pressure_metrics, joint.assess, joint.dg_geometry)


def _original(cfg):
    native = copy.deepcopy(cfg['native_config'])
    native.update(parent_checkpoint=str(joint.ASSET/'models/initializers/marmousi.pt'),
                  parent_checkpoint_SHA='2638504b1c2806c29752aed5189c1d09fb0e5f4a6c746e91d91d8eb324adb8d6',
                  dense_time_block=400, refiner_time_block=400, checkpoint_queries=True)
    return build_r25_cf_full400_operator(native, device='cpu', kind='marmousi')


def _reduce(model):
    compact_null_parameters(model)
    receipt = reduce_structural_redundancy(model)
    require(sum(p.requires_grad for p in model.parameters()) == EXPECTED_TENSORS
            and sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS,
            'V2 architecture signature changed')
    return receipt


def _groups(model, specifications):
    """Match native group identities; preserve all native optimizer options."""
    native = model.parameter_groups(base_lr=1e-4, refiner_lr=1e-4, weight_decay=1e-4)
    require([g['group_name'] for g in native] == [g['group_name'] for g in specifications],
            'native optimizer group identities/order changed')
    names = {id(p): n for n,p in model.named_parameters()}
    for group, specification in zip(native, specifications):
        expected = specification.get('parameter_names')
        require(expected is None or [names[id(p)] for p in group['params']] == list(expected),
                'optimizer group parameter identities/order changed')
        group.update(copy.deepcopy({k:v for k,v in specification.items() if k not in ('params','parameter_names')}))
        if 'betas' in group:
            group['betas'] = tuple(group['betas'])
    return native


def optimizer_specifications(model, optimizer):
    names = {id(p): n for n,p in model.named_parameters()}
    return [dict({k:copy.deepcopy(v) for k,v in group.items() if k != 'params'},
                 parameter_names=[names[id(p)] for p in group['params']]) for group in optimizer.param_groups]


def create_operator(cfg, device='cpu'):
    """Rebuild compact architecture; caller MUST strict-load its compact checkpoint.

    Constructor values are only a template and are never a continuation state.
    This path intentionally does not reload the historical 2000/2200-step model.
    """
    model = _original(cfg)
    reduction = _reduce(model)
    model.to(device)
    optimizer = torch.optim.AdamW(_groups(model, cfg['optimizer_groups']), foreach=False)
    model.train()
    return model, optimizer, dict(schema=SCHEMA, rebuild_only=True,
        requires_strict_compact_state_load=True, structural_reduction=reduction,
        tensors=EXPECTED_TENSORS, parameters=EXPECTED_PARAMETERS)


def _sealed_parent(payload, cfg):
    path = Path(cfg.get('migration_parent_checkpoint', PARENT)).resolve()
    require(cfg.get('migration_parent_sha256', PARENT_SHA) == PARENT_SHA and sha(path) == PARENT_SHA,
            'final stopped joint200 checkpoint SHA required')
    sealed = torch.load(str(path), map_location='cpu', mmap=True, weights_only=False)
    # Bind the supplied payload, not just a same-named file beside it.
    for key in ('model_state','optimizer_state','learned_optimizer_state','normalizer',
                'native_times_s','registered_times_s','active_parameter_names','updates',
                'total_optimizer_updates','dataset_sha256','dataset_complete_sha256'):
        require(equal_tree(payload[key], sealed[key]), 'supplied parent differs from sealed checkpoint: '+key)
    require(payload['schema'] == 'MARMOUSI_JOINT400_LEARNED_OPTIMIZER_V1'
            and payload['updates'] == INHERITED_JOINT_UPDATES
            and payload['total_optimizer_updates'] == 2200,
            'stopped complete joint200 / Adam2200 parent required')
    for key in ('dataset_sha256','dataset_complete_sha256'):
        require(cfg[key] == payload[key], 'dataset binding changed: '+key)
    return path


def _transfer_controller(model, payload, cfg):
    old = payload['learned_optimizer_state']
    contract = old['_extra_state']
    learned = make_learned(model, cfg)
    require(contract['config'] == learned.get_extra_state()['config'], 'controller configuration changed')
    old_names = tuple(contract['parameter_names'])
    require(old_names == tuple(payload['active_parameter_names']), 'parent controller block identity mismatch')
    indices = {name:i for i,name in enumerate(old_names)}
    state = learned.state_dict()
    mapping = {}
    for key in state:
        if key not in ('_extra_state','embedding.weight'):
            require(key in old and old[key].shape == state[key].shape, 'shared controller shape changed')
            state[key] = old[key].detach().clone()
    for i,name in enumerate(learned.parameter_names):
        if name in indices:
            sources, rule = [name], 'same-name embedding copy'
        else:
            prefix, suffix = name.rsplit('.',1)
            require(suffix in ('weight_regular','weight_pair','weight_self') and
                    prefix.startswith('base.medium_encoder.'), 'unregistered new controller block: '+name)
            sources = [prefix+'.weight_top'] if suffix == 'weight_self' else [prefix+'.weight_top',prefix+'.weight_bottom']
            rule = 'top embedding initialization' if len(sources)==1 else 'mean top/bottom embedding initialization'
        state['embedding.weight'][i] = torch.stack([old['embedding.weight'][indices[n]] for n in sources]).mean(0)
        mapping[name] = dict(sources=sources, rule=rule)
    learned.load_state_dict(state, strict=True)
    require(sum(p.numel() for p in learned.parameters()) == META_PARAMETERS, 'V2 controller parameter count changed')
    return learned, mapping


def migrate_from_joint(payload, cfg, device='cpu'):
    """Strict sealed joint200 theta -> V2, with explicit per-coordinate Adam rebase.

    AIS, RNG and training cursors remain driver-owned. The controller must receive
    the original global update count (200 next), not the new stage-local count.
    """
    path = _sealed_parent(payload, cfg)
    model = _original(cfg)
    model.load_state_dict(payload['model_state'], strict=True)
    require(equal_tree(model.state_dict(), payload['model_state']), 'strict final joint theta restore differs')
    require(vars(model.normalizer.metadata) == payload['normalizer'], 'fixed pressure normalization changed')
    require(torch.equal(model.native_times_s, payload['native_times_s'])
            and torch.equal(model.full_times_s, payload['registered_times_s']), 'teacher physical time binding changed')
    original = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    require(list(original) == payload['active_parameter_names'], 'parent active parameter order changed')
    old_base = torch.optim.AdamW(model.parameter_groups(1e-4,1e-4,weight_decay=1e-4), foreach=False)
    old_base.load_state_dict(payload['optimizer_state'])
    require(equal_tree(old_base.state_dict(), payload['optimizer_state']), 'parent AdamW restore differs')
    old_state = {n:old_base.state[p] for n,p in model.named_parameters() if p.requires_grad}
    require(all(float(s['step']) == 2200 for s in old_state.values()), 'every parent AdamW block must have age2200')
    options = [{k:copy.deepcopy(v) for k,v in g.items() if k != 'params'} for g in old_base.param_groups]
    reduction = _reduce(model)
    base = torch.optim.AdamW(_groups(model, options), foreach=False)
    blocks = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        previous = original.get(name)
        inherited = (name not in TRANSFORMED and previous is not None
                     and previous.shape == parameter.shape and torch.equal(previous, parameter))
        if inherited:
            base.state[parameter] = copy.deepcopy(old_state[name])
        reason = ('unchanged name/shape/value/coordinate' if inherited else
                  'explicit transformed coordinate' if name in TRANSFORMED else
                  'new Hermitian coordinate' if previous is None else 'shape/value changed')
        parent_names = [name] if previous is not None else [name.rsplit('.',1)[0]+'.weight_'+s for s in ('top','bottom')]
        require(all(n in original for n in parent_names), 'unregistered reduced coordinate '+name)
        blocks[name] = dict(parent_shape=None if previous is None else list(previous.shape),
            parent_coordinate_shapes={n:list(original[n].shape) for n in parent_names},
            shape=list(parameter.shape), inherited=inherited, reason=reason, base_age=2200 if inherited else 0)
    # Fresh blocks are absent, rather than fake zero-age entries. Native AdamW
    # initializes them on their first real update. Preserve old group options.
    learned, embedding_map = _transfer_controller(model, payload, cfg)
    if torch.device(device).type != 'cpu':
        base_state = base.state_dict()
        specifications = optimizer_specifications(model, base)
        model.to(device); learned.to(device)
        base = torch.optim.AdamW(_groups(model, specifications), foreach=False)
        base.load_state_dict(base_state)
    meta = torch.optim.Adam(learned.parameters(), lr=cfg['meta_learning_rate'],
                            weight_decay=cfg['meta_weight_decay'], foreach=False)
    receipt = dict(schema=SCHEMA, parent_checkpoint=str(path), parent_sha256=PARENT_SHA,
        inherited_joint_updates=200, parent_optimizer_age=2200, tensors=EXPECTED_TENSORS,
        parameters=EXPECTED_PARAMETERS, meta_parameters=META_PARAMETERS,
        exact_old_optimizer_trajectory=False, model_from_final_joint_checkpoint=True,
        shared_controller_weights_preserved=True, controller_embedding_mapping=embedding_map,
        controller_mapping_is_initialization_heuristic=True, meta_optimizer_fresh=True,
        blocks=blocks, base_age_by_name={n:r['base_age'] for n,r in blocks.items()},
        inherited_blocks=sum(r['inherited'] for r in blocks.values()),
        fresh_blocks=sum(not r['inherited'] for r in blocks.values()),
        optimizer_groups=optimizer_specifications(model, base), structural_reduction=reduction,
        global_controller_step_continues=True, AIS_RNG_cursors='driver preserves original complete state')
    model.train()
    require(sha(path) == PARENT_SHA, 'sealed parent changed during migration')
    return model, base, learned, meta, receipt
