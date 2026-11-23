"""Strict local reconstruction of the feature-enhancement experiment's parent.

Only the original R24 architecture initializer is read. Its canonical V1/V2
coordinate transformations construct the existing architecture, after which the
complete running 4 s checkpoint replaces every state value. Historical training
stages are neither replayed nor edited. The returned payload is the unchanged
parent, including its optimizer/controller/meta/RNG/cursor and remote provenance.
"""
from __future__ import annotations
from scno_release_paths import local_asset as _scno_local_asset

import copy
from contextlib import contextmanager
import _imp
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import torch

from . import paper4s_effectiveband_state_20261021 as state

ROOT = Path(__file__).resolve().parents[2]
PARENT_RUN = ROOT / 'runs/paper4s_effectiveband10_resume_20261021'
PARENT_CHECKPOINT = PARENT_RUN / 'latest.pt'
PARENT_SHA256 = 'f76b9612d1b1d89cb7c87940e91d892dac8c76aa1ee71cfd2ffbefa71d3583ff'
V2_CONFIG_SHA256 = '9d72c8a27679b5316b270e83ed831bb95a1c956f3928d01a58249ee84f039aad'
INITIALIZER = (ROOT / 'paper_assets_main_fresh_20261013_marmousi_20261019'
               / 'models/initializers/marmousi.pt')
INITIALIZER_SHA256 = '2638504b1c2806c29752aed5189c1d09fb0e5f4a6c746e91d91d8eb324adb8d6'
REMOTE_ROOT = _scno_local_asset('project')
PROTOCOL_PINS = {
    'src/native400_runtime/compat/r24/START.json':
        'e980eb5f96b385983d3636946c488c9e76d857d382e12d22dcc119e99a98fe9e',
    'src/native400_runtime/compat/r21/protocol92.py':
        '73de54ed2513859926d89ce88a0c662909d1f7e6202ded8b6e497b955e0d34fe',
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(Path(path).read_text())


def _verify_local_dependencies(payload):
    """Check existing frozen sources and the exact constructor config/weights."""
    manifest = _json(PARENT_RUN / 'SOURCE_MANIFEST.json')
    for relative, expected in manifest['files'].items():
        path = ROOT / relative
        _require(path.is_file() and state.sha(path) == expected,
                 'Frozen parent source missing or changed: ' + relative)
    for relative, expected in PROTOCOL_PINS.items():
        _require(state.sha(ROOT / relative) == expected,
                 'Migrated protocol dependency changed: ' + relative)
    lineage = payload['lineage']
    _require(lineage['parent_checkpoint'].startswith(REMOTE_ROOT + '/runs/'),
             'Unexpected V2 parent provenance path')
    relative_parent = Path(lineage['parent_checkpoint']).relative_to(REMOTE_ROOT)
    config_path = ROOT / relative_parent.parent / 'TRAINING_CONFIG.json'
    _require(lineage['parent_config_sha256'] == V2_CONFIG_SHA256
             and state.sha(config_path) == V2_CONFIG_SHA256,
             'Pinned V2 constructor configuration changed')
    _require(state.same(_json(config_path), lineage['parent_config']),
             'Payload and local V2 constructor configurations differ')
    _require(INITIALIZER.is_file() and state.sha(INITIALIZER) == INITIALIZER_SHA256,
             'Original R24 constructor initializer is missing or changed')
    return config_path


@contextmanager
def _local_protocol_import():
    """Rebind the sole old import path read from frozen R24 START.json.

    Execute both original protocol sources normally, retaining all their frame
    assertions and protocol digests. The standard import lock protects this
    temporary allowlisted resolver; no file or canonical factory is changed.
    """
    relative = 'src/native400_runtime/compat/r21/protocol92.py'
    old_path, local_path = str(Path(REMOTE_ROOT) / relative), ROOT / relative
    _imp.acquire_lock()
    original = importlib.util.spec_from_file_location
    # bootstrap24 also imports the genuine frozen R22 io_state/refiner/sampler/
    # engine modules. Its START.json search directory has the same stale root.
    # Prepend the canonical local directory; never manufacture substitute modules.
    local_r22 = str(ROOT / 'src/native400_runtime/compat/r22')
    remote_r22 = REMOTE_ROOT + '/src/native400_runtime/compat/r22'
    inserted_r22 = local_r22 not in sys.path
    if inserted_r22:
        sys.path.insert(0, local_r22)
    def rebound(name, location=None, *args, **kwargs):
        if location is not None and str(location) == old_path:
            location = local_path
        return original(name, location, *args, **kwargs)
    importlib.util.spec_from_file_location = rebound
    try:
        yield
    finally:
        importlib.util.spec_from_file_location = original
        if inserted_r22:
            sys.path.remove(local_r22)
        sys.path[:] = [path for path in sys.path if path != remote_r22]
        _imp.release_lock()


def local_constructor_config(payload):
    """A private, explicitly rebound config; frozen payload/files stay unchanged.

    Only the native constructor consumes this dictionary. Its historical cache
    is not read by a constructor, and rebinding it does not authorize using old
    labels. The caller must bind the rebuilt TRAIN dataset separately.
    """
    native = copy.deepcopy(payload['lineage']['parent_config']['native_config'])
    for key, value in tuple(native.items()):
        if isinstance(value, str) and value.startswith(REMOTE_ROOT + '/'):
            native[key] = str(ROOT / Path(value).relative_to(REMOTE_ROOT))
    _require(native['parent_checkpoint_SHA'] == INITIALIZER_SHA256,
             'Historical initializer SHA does not match local constructor')
    native.update(parent_checkpoint=str(INITIALIZER),
                  parent_checkpoint_SHA=INITIALIZER_SHA256,
                  dense_time_block=400, refiner_time_block=400,
                  checkpoint_queries=True)
    return native


def _verify_model(model, payload):
    cfg = payload['config']
    _require(state.same(model.state_dict(), payload['model']),
             'Actual reconstructed model differs from exact parent state')
    _require([n for n, p in model.named_parameters() if p.requires_grad]
             == cfg['active_parameter_names'], 'Inherited trainability roster changed')
    _require(sum(p.numel() for p in model.parameters()) == cfg['parameter_count'],
             'Inherited parameter count changed')
    actual_digest = state.immutable_digest(model)
    if actual_digest != payload['immutable_digest']:
        receipt = getattr(model, 'local_numerical_rebinding', None)
        _require(isinstance(receipt, dict)
                 and receipt['schema'] == 'LOCAL_NUMERICAL_REBINDING_V1'
                 and receipt['historical_immutable_digest'] == payload['immutable_digest']
                 and receipt['local_immutable_digest'] == actual_digest,
                 'Inherited frozen features/buffers changed without explicit local numerical rebinding')
    _require(torch.equal(model.full_times_s.cpu(), payload['field_times']),
             'Physical 801-frame time axis changed')
    _require(state.same(vars(model.normalizer.metadata), payload['normalizer']),
             'Inherited physical normalizer changed')


def _buffer_differences(model, saved):
    differences = []
    for name, value in model.named_buffers():
        if name == 'training_exposures' or name not in saved:
            continue
        reference = saved[name]
        if not torch.equal(value, reference):
            differences.append(dict(name=name, shape=list(value.shape),
                different_elements=int((value != reference).sum()),
                max_abs=float((value.to(torch.float64)-reference.to(torch.float64)).abs().max())))
    return differences


def _numerical_rebinding_receipt(model, payload, differences):
    """Record an actual limitation; this is not a historical digest assertion."""
    # This narrow admission identifies the observed DG roundoff; any unrelated
    # constructor drift remains a hard failure even with an explicit opt-in.
    _require(bool(differences) and all(row['name'].startswith('backbone.refiner.dg.')
             and row['max_abs'] < 2e-14 for row in differences),
             'Constructor buffer drift exceeds the diagnosed DG near-zero constants')
    nonpersistent = []
    for name, value in model.named_buffers():
        _require(bool(torch.isfinite(value).all()), 'Nonfinite reconstructed constant: ' + name)
        if name not in payload['model']:
            array = value.detach().cpu().contiguous().numpy()
            nonpersistent.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                                      sha256=hashlib.sha256(array.tobytes()).hexdigest()))
    return dict(schema='LOCAL_NUMERICAL_REBINDING_V1',
        status='REQUIRES_RUNTIME_PREDICTION_VJP_AND_RESTORE_ADMISSION',
        parent_checkpoint_sha256=PARENT_SHA256,
        historical_immutable_digest=payload['immutable_digest'],
        historical_immutable_digest_verified=False,
        local_immutable_digest=state.immutable_digest(model),
        persistent_state_exact=state.same(model.state_dict(), payload['model']),
        persistent_entries=len(payload['model']),
        original_nonpersistent_buffers_saved=False,
        constructor_buffer_differences=differences,
        local_nonpersistent_buffers=nonpersistent,
        observed_cause='Only DG near-zero persistent constructor constants differ before strict loading; cross-version numerical construction is suspected. Historical nonpersistent constants are unavailable, so exclusive cause is unproven.',
        torch_version=str(torch.__version__),
        action='Retain canonical locally constructed nonpersistent constants; strict-load every historical persistent value. Preserve both digests and require explicit runtime admission before training.')


def load_parent(device='cpu', *, allow_local_numerical_rebinding=False):
    """Return ``(model, unchanged_payload)`` from the exact 675-update parent.

    CPU construction preserves the caller's CPU RNG and never initializes CUDA.
    Loading this function alone does not restore global RNGs or start training.
    The migrated environment has diagnosed DG roundoff in derived constants.
    An explicit opt-in returns a receipt on ``model.local_numerical_rebinding``;
    it never labels the mismatched historical immutable digest as verified.
    """
    receipt = _json(PARENT_RUN / 'CHECKPOINT.json')
    _require(receipt['sha256'] == PARENT_SHA256
             and state.sha(PARENT_CHECKPOINT) == PARENT_SHA256,
             'Selected feature-enhancement parent checkpoint SHA mismatch')
    payload = torch.load(PARENT_CHECKPOINT, map_location='cpu', weights_only=False)
    _require(state.same(payload['config'], _json(PARENT_RUN / 'CONFIG.json'))
             and state.same(payload['runtime'], _json(PARENT_RUN / 'RUNTIME.json')),
             'Parent configuration/runtime differs from frozen files')
    state.validate(payload, payload['config'], payload['runtime'])
    _require(payload['progress']['updates'] == 675
             and payload['progress']['epoch'] == 4
             and payload['progress']['offset'] == 300
             and payload['progress']['exposures'] == 2700,
             'Selected parent historical training cursor changed')
    _verify_local_dependencies(payload)

    from .r25_cf_full400_operator import build_r25_cf_full400_operator
    from .marmousi_reduced400_continue_20261020 import _reduce
    from .paper4s_effectiveband_operator_20261021 import Paper4sConfig, Paper4sMultiBandOperator

    # Canonical _original() uses this same R25 constructor and overrides. Its
    # historical absolute initializer is explicitly rebound above, not patched.
    with state.RNG_LOCK, torch.random.fork_rng(devices=[]), _local_protocol_import():
        torch.random.default_generator.manual_seed(payload['config']['seed'])
        backbone = build_r25_cf_full400_operator(local_constructor_config(payload),
                                                device='cpu', kind='marmousi')
        _reduce(backbone)
        model = Paper4sMultiBandOperator(backbone, Paper4sConfig(**payload['config']['model']))
        differences = _buffer_differences(model, payload['model'])
        model.load_state_dict(payload['model'], strict=True)
        model.set_training_exposures(payload['progress']['exposures'])
    if state.immutable_digest(model) != payload['immutable_digest']:
        _require(allow_local_numerical_rebinding,
                 'Historical immutable digest differs; explicit local numerical rebinding is required')
        model.__dict__['local_numerical_rebinding'] = _numerical_rebinding_receipt(model, payload, differences)
    _verify_model(model, payload)
    _require(state.sha(PARENT_CHECKPOINT) == PARENT_SHA256,
             'Parent checkpoint changed during reconstruction')
    model.to(device)
    return model, payload


def restore_training_objects(model, payload):
    """Return exact ``(CurriculumAdamW, controller, metaAdam)`` before extension.

    No learned-optimizer behavior is replaced. The candidate can subsequently
    extend these objects under its own tested parameter-coordinate contract.
    This helper intentionally rejects an already extended parameter roster.
    """
    state.validate(payload, payload['config'], payload['runtime'])
    _verify_model(model, payload)
    cfg = payload['config']
    from .paper201_multiband_fno_20261020 import CurriculumAdamW
    from . import marmousi_direct400_continue_20261020 as joint
    with state.RNG_LOCK, torch.random.fork_rng(devices=[]):
        groups = model.backbone.parameter_groups(base_lr=cfg['learning_rate'],
            refiner_lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        for index, head in enumerate(model.heads):
            groups.append(dict(params=list(head.parameters()), lr=cfg['learning_rate'],
                weight_decay=cfg['weight_decay'], temporal_band=index,
                band_active=index == 0, group_name=f'temporal_band_{index}'))
        optimizer = CurriculumAdamW(groups, foreach=False)
        controller = joint.make_learned(model, dict(block_learned_optimizer=cfg['controller']))
        meta = torch.optim.Adam(controller.parameters(), lr=cfg['meta_learning_rate'], foreach=False)
    state.set_curriculum(model, optimizer, payload['progress']['exposures'])
    for label, obj, decoupled in (('optimizer', optimizer, True), ('meta', meta, False)):
        options = lambda g: {k: v for k, v in g.items() if k != 'params'}
        local_state = historical_optimizer_state(obj, decoupled=decoupled)
        _require(state.same([options(g) for g in local_state['param_groups']],
                            [options(g) for g in payload[label]['param_groups']]),
                 'Registered optimizer options differ: ' + label)
    _require(state.optimizer_roster(model, optimizer) == payload['optimizer_roster']
             and state.optimizer_roster(controller, meta) == payload['meta_roster'],
             'Reconstructed optimizer parameter identities differ')
    optimizer.load_state_dict(payload['optimizer'])
    controller.load_state_dict(payload['controller'], strict=True)
    meta.load_state_dict(payload['meta'])
    for label, obj, decoupled in (('optimizer', optimizer, True), ('controller', controller, None),
                                  ('meta', meta, False)):
        local_state = obj.state_dict() if decoupled is None else historical_optimizer_state(obj, decoupled=decoupled)
        _require(state.same(state.freeze(local_state), payload[label]),
                 'Actual complete training-state restoration differs: ' + label)
    model.__dict__['optimizer_schema_rebinding'] = dict(
        schema='TORCH_ADAM_EXPLICIT_DECOUPLED_WEIGHT_DECAY_V1',
        original_options_and_all_moments_exact=True,
        new_main_group_field=dict(decoupled_weight_decay=True),
        new_meta_group_field=dict(decoupled_weight_decay=False),
        algorithm='Original CurriculumAdamW and torch.optim.Adam; old implicit decay semantics are explicit in Torch 2.7.',
        old_payload_unmodified=True)
    return optimizer, controller, meta


def historical_optimizer_state(optimizer, *, decoupled):
    """Compare the old Adam schema while validating the new semantic flag.

    Torch 2.7 merges Adam/AdamW and materializes this formerly implicit option.
    Do not remove it from live groups: the canonical optimizer needs it at step.
    All old keys/tensors remain in this projection, and the payload stays intact.
    """
    value = optimizer.state_dict()
    groups = []
    for group in value['param_groups']:
        group = dict(group)
        _require(group.pop('decoupled_weight_decay', decoupled) is decoupled,
                 'Adam/AdamW weight decay semantics changed')
        groups.append(group)
    return dict(value, param_groups=groups)


def load_objects(device='cpu', *, allow_local_numerical_rebinding=False):
    """Convenience API: ``model, optimizer, controller, meta, payload``."""
    model, payload = load_parent(device, allow_local_numerical_rebinding=allow_local_numerical_rebinding)
    optimizer, controller, meta = restore_training_objects(model, payload)
    return model, optimizer, controller, meta, payload


__all__ = ['load_parent', 'load_objects', 'restore_training_objects', 'local_constructor_config',
           'historical_optimizer_state', 'PARENT_CHECKPOINT', 'PARENT_SHA256']
