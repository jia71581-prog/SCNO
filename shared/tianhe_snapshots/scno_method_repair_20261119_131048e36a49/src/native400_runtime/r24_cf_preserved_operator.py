"""Warm-started R24 MIONet/U-FNO, with strictly c/f-only conditioning.

The eight pressure channels required by the inherited architecture are created
as zeros internally. Real observations enter only forward_cf_full's final copy.
forecast_at_times/forward are TRAIN query APIs, not sparse deployment APIs.
No dataset, pressure cache, ridge, solver, or fresh initialization is imported.
"""
from __future__ import annotations
from scno_release_paths import local_asset as _scno_local_asset

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .paper201_time_axis_20261020 import native_time_axis

ROOT = Path(__file__).resolve().parents[2]
COMPAT = Path(__file__).resolve().parent / 'compat'
DEFAULT_CHECKPOINT = _scno_local_asset('fno-reference/a_research/experiments_101x101x100/experiments/cc_time400_r24_20261008T134744Z/layered/frozen/run/checkpoint_final_added000256.pt')
DEFAULT_SHA = 'da69252452e52a22dd0edc29bd01d0e95ddc514ab32f590d9131fb6b9d13f733'
SCHEMA = 'R24_PRESERVED_MIONET_UFNO_ZERO_IC_CF_V1'


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _module(name, path):
    path = Path(path).resolve()
    if not path.is_relative_to(COMPAT.resolve()):
        raise PermissionError('all implementation imports must be canonical')
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


class R24CFPreservedOperator(nn.Module):
    """One record per call; c201 sharp + known five source parameters.

    base and refiner inherit the exact R24 state shapes. All useful conditioning
    groups are trainable. The source first IC-convolution weight is explicitly
    frozen because its input is identically zero. The refiner's zero-IC input
    slices have a gradient mask and a zero-weight-decay optimizer group.
    """
    def __init__(self, device='cpu', *, family='layered', checkpoint_path=DEFAULT_CHECKPOINT,
                 expected_checkpoint_sha=DEFAULT_SHA, dense_time_block=2,
                 refiner_time_block=4, checkpoint_queries=True):
        super().__init__()
        if family not in ('layered', 'marmousi'):
            raise ValueError('inherited R24 factory supports layered or marmousi')
        if min(int(dense_time_block), int(refiner_time_block)) < 1:
            raise ValueError('positive internal execution blocks required')
        self.family = family
        self.dense_time_block = int(dense_time_block)
        self.refiner_time_block = int(refiner_time_block)
        self.checkpoint_queries = bool(checkpoint_queries)
        parents = json.loads((COMPAT/'shared/PARENT_MANIFEST.json').read_text())
        spec = parents['parents'][family]
        config = copy.deepcopy(spec['resolved_config'])
        code = COMPAT / ('base_' + family)
        config['model_spec'] = str(code/'configs/model_spec_v185b.json')
        config['data']['normalization_json'] = str(COMPAT/'shared/normalization_ul.json')
        if _sha(config['model_spec']) != spec['model_spec_sha256']:
            raise PermissionError('inherited model architecture specification changed')
        for directory in (code, code/'snapshot', COMPAT/'shared'):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        factory = _module('cf_preserved_r24_factory_' + family, code/'model.py')
        from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
        normalizer_path = COMPAT/'shared/normalization_ul.json'
        if _sha(normalizer_path) != parents['normalizer']['sha256']:
            raise PermissionError('inherited pressure normalizer changed')
        self.normalizer = PhysicalNormalizer.from_dict(json.loads(normalizer_path.read_text()),
            expected_manifest=parents['semantic_manifest_digest'])
        # build creates the old architecture; no fresh initializer or parent
        # optimizer is called. Every model/refiner tensor is then strictly loaded.
        self.base = factory.build(config, 'cpu')
        self.base.scno_depth = 2
        refiner_module = _module('cf_preserved_r24_refiner', COMPAT/'r22/refiner.py')
        self.refiner = refiner_module.ResidualRefiner()
        path = Path(checkpoint_path).resolve()
        if not expected_checkpoint_sha or _sha(path) != expected_checkpoint_sha:
            raise PermissionError('explicit original R24 checkpoint SHA mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        protocol = _module('cf_preserved_r24_protocol', COMPAT/'r24/protocol400.py')
        if (payload.get('schema') != 'R24_NATIVE400_ENDTOEND_FULLSTATE_V1'
                or payload.get('family') != family or payload.get('protocol_SHA') != protocol.PROTOCOL_SHA):
            raise PermissionError('original native400 full-state checkpoint identity required')
        self.base.load_state_dict(payload['base_model_state'], strict=True)
        self.refiner.load_state_dict(payload['refiner_state'], strict=True)
        boundary = _module('cf_preserved_r24_boundary', COMPAT/'r21/boundary_exact.py')
        boundary.install_exact_boundary(self.base)
        self.base.requires_grad_(True)
        self.refiner.requires_grad_(True)
        inactive_name = 'source_encoder.ic_encoder.0.weight'
        dict(self.base.named_parameters())[inactive_name].requires_grad_(False)
        self.refiner.static[0].conv1.weight.register_hook(self._mask_zero_ic_gradient)
        self._restriction = _module('cf_preserved_r24_B5', COMPAT/'sourcegen/restriction.py').restrict_nodal_2x
        self._bicubic201 = _module('cf_preserved_r24_graph', COMPAT/'r24/graph24.py').bicubic201
        self._core = _module('cf_preserved_r24_core', COMPAT/'shared/shorttrain_core_r2.py')
        self.helpers = {key:_module('cf_preserved_r24_'+key, COMPAT/'shared'/filename) for key,filename in {
            'padding':'deterministic_padding.py', 'sampling':'deterministic_sampling.py',
            'pooling':'deterministic_pooling.py', 'cartesian':'cartesian_adjoint.py'}.items()}
        self.register_buffer('x_m', torch.arange(201, dtype=torch.float32)*10, persistent=False)
        self.register_buffer('z_m', torch.arange(201, dtype=torch.float32)*10, persistent=False)
        self.register_buffer('native_times_s', native_time_axis(), persistent=False)
        self.register_buffer('future_native_indices', torch.tensor(protocol.FORECAST_FRAMES, dtype=torch.long), persistent=False)
        self.known_native_frames = tuple(protocol.IC_FRAMES)
        self.future_native_frames = tuple(protocol.FORECAST_FRAMES)
        self.full_native_frames = tuple(protocol.FULL_FRAMES)
        self.protocol_SHA = protocol.PROTOCOL_SHA
        self.initialization_receipt = dict(schema=SCHEMA, family=family, checkpoint_path=str(path),
            checkpoint_SHA=expected_checkpoint_sha, strict_base_model_state=True, strict_refiner_state=True,
            normalizer_SHA=parents['normalizer']['sha256'], protocol_SHA=self.protocol_SHA,
            optimizer_restored=False, fresh_initializer_called=False, original_R24_accuracy_is_not_CF_accuracy=True,
            predictor_input_fields=['velocity_mps','source_x_z_f0_t0_amplitude'], internal_IC='always_zero_Pa8',
            base_velocity='B5(sharp201)->101', refiner_velocity='sharp201 unchanged',
            source_IC_first_conv_weight_frozen=inactive_name, refiner_zero_IC_weight_slices='static.0.conv1.weight[:,1:9,:,:]',
            known8_delivery_only=True, model_branches=['medium','source','coordinate','travel','fusion','dense','scno','refiner'])
        del payload
        self.to(device)

    @property
    def norm(self):
        return self.normalizer

    @property
    def device(self):
        return next(self.base.parameters()).device

    @property
    def _refiner_masked_weight(self):
        return self.refiner.static[0].conv1.weight

    @property
    def future_times_s(self):
        return self.native_times_s[self.future_native_indices]

    @staticmethod
    def _mask_zero_ic_gradient(gradient):
        gradient = gradient.clone()
        gradient[:,1:9,:,:] = 0
        return gradient

    def scopes(self):
        """Trainer may cover forward+backward with this deterministic context."""
        return self._core.scopes(self.helpers, {key:[] for key in ('padding','sampling','pooling')})

    def parameter_groups(self, base_lr, refiner_lr, weight_decay=1e-6):
        groups = []
        for label,prefix in [('medium','medium_encoder.'),('source','source_encoder.'),
                ('coordinate','coordinate_encoder.'),('travel','travel_branch.'),('fusion','fusion.'),
                ('dense','dense_decoder.'),('scno','scno_cycle.'),('local','local_unet.')]:
            params = [p for name,p in self.base.named_parameters() if name.startswith(prefix) and p.requires_grad]
            if params:
                groups.append(dict(params=params, lr=float(base_lr), weight_decay=float(weight_decay), group_name='base_'+label))
        seen = {id(p) for group in groups for p in group['params']}
        extra = [p for p in self.base.parameters() if p.requires_grad and id(p) not in seen]
        if extra:
            groups.append(dict(params=extra,lr=float(base_lr),weight_decay=float(weight_decay),group_name='base_other'))
        groups.append(dict(params=[p for p in self.refiner.parameters() if p is not self._refiner_masked_weight and p.requires_grad],
            lr=float(refiner_lr),weight_decay=float(weight_decay),group_name='refiner'))
        # No weight decay (and initially zero optimizer momentum) on masked IC
        # slices, so inactive slices cannot acquire artificial parameter updates.
        groups.append(dict(params=[self._refiner_masked_weight],lr=float(refiner_lr),weight_decay=0.,group_name='refiner_masked_input'))
        actual = [id(p) for group in groups for p in group['params']]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise AssertionError('each active parameter must occur in exactly one optimizer group')
        return groups

    def trainable_receipt(self):
        return dict(active=[dict(name=name,shape=list(p.shape),numel=p.numel())
                           for name,p in self.named_parameters() if p.requires_grad],
            frozen=[name for name,p in self.named_parameters() if not p.requires_grad],
            inactive_refiner_slice='refiner.static.0.conv1.weight[:,1:9,:,:]',
            required_optimizer_group='refiner_masked_input with weight_decay=0 and fresh zero momentum',
            active_group_gradient_and_parameter_delta_must_be_measured=True)

    def _inputs(self, c, f):
        if isinstance(c, dict) or isinstance(f, dict):
            raise TypeError('only explicit c and f tensors; no pressure/IC input dictionaries')
        c = torch.as_tensor(c, device=self.device, dtype=torch.float32)
        f = torch.as_tensor(f, device=self.device, dtype=torch.float32)
        if c.shape == (201,201): c = c[None,None]
        if f.shape == (5,): f = f[None]
        if c.shape != (1,1,201,201) or f.shape != (1,5):
            raise ValueError('one sharp c201 and five known source parameters required')
        if not bool(torch.isfinite(c).all()) or not bool(torch.all(c > 0)) or not bool(torch.isfinite(f).all()):
            raise ValueError('finite positive velocity and finite source required')
        if (bool(torch.any(f[:,:2]<0)) or bool(torch.any(f[:,:2]>2000)) or float(f[0,4]) <= 0
                or abs(float(f[0,2])-15.)>1e-5 or abs(float(f[0,3])-.1)>1e-6):
            # 2026-11-10: centered-source rebuild uses in-domain x/z, not the
            # historical z=100 horizontal line.
            raise ValueError('frozen 15Hz in-domain x/z scope: t0=.1, positive known A')
        return c, f

    @staticmethod
    def _source_map(f, n, spacing):
        loc = f[0,:2]/spacing
        ix,iz = torch.floor(loc).long().clamp(0,n-2)
        fx,fz = loc[0]-ix,loc[1]-iz
        field = f.new_zeros((1,1,n,n))
        field[0,0,iz,ix] = (1-fx)*(1-fz)
        field[0,0,iz,ix+1] = fx*(1-fz)
        field[0,0,iz+1,ix] = (1-fx)*fz
        field[0,0,iz+1,ix+1] = fx*fz
        return field

    def _prepare_cf(self, c, f):
        # This dict is built locally from c/f; it contains no observed pressure.
        zero101 = c.new_zeros((1,8,101,101))
        zero201 = c.new_zeros((1,8,201,201))
        coarse_c = self._restriction(c).contiguous()
        medium = self.base.encode_medium(coarse_c, self.normalizer)
        source101 = self._source_map(f,101,20.)
        prepared = self.base.prepare_sources(medium,f,source101,self.normalizer,
            record_to_medium=torch.zeros(1,dtype=torch.long,device=self.device),
            ic_snapshots_physical=zero101,anchor_time_s=self.native_times_s[88:89])
        grid = self.base.prepare_dense_grid(prepared,x_m=self.x_m[::2],z_m=self.z_m[::2])
        z,x = torch.meshgrid(self.z_m,self.x_m,indexing='ij')
        coords = torch.stack((2*x/2000-1,2*z/2000-1,(x-f[0,0])/2000,(z-f[0,1])/2000))[None]
        static = torch.cat((c/3000-1,zero201,self._source_map(f,201,10.),coords),1)
        features = self.refiner.prepare(static)
        return prepared,grid,features

    def _query(self, prepared,grid,features,f,times):
        base101 = self.base.dense_normalized(prepared,times[None],dense_grid=grid,
            time_block=self.dense_time_block,apply_correction=True)
        if base101.shape != (1,times.numel(),101,101):
            raise AssertionError('inherited dense MIONet/U-FNO base shape')
        base_pa = self.normalizer.decode_pressure(base101,f[:,4])
        pa201 = self._bicubic201(base_pa[0,:,None])
        base201 = self.normalizer.encode_pressure(pa201[:,0][None],f[:,4])[0,:,None]
        value = base201 + self.refiner(features,base201,times,f)
        return torch.cat((torch.zeros_like(value[:,:,:1,:]),value[:,:,1:,:]),-2)[:,0]

    def forecast_at_times(self, c, f, times_s, *, normalized=True):
        """Graph-connected TRAIN queries [T,201,201], not a deployment return."""
        c,f = self._inputs(c,f)
        times = torch.as_tensor(times_s,device=self.device,dtype=torch.float32).reshape(-1)
        if times.numel()==0 or not bool(torch.isfinite(times).all()) or bool(torch.any(times<0)) or bool(torch.any(times>1)):
            raise ValueError('nonempty actual physical query times in [0,1] required')
        with self.scopes():
            prepared,grid,features = self._prepare_cf(c,f)
        pieces = []
        for start in range(0,times.numel(),self.refiner_time_block):
            query_times = times[start:start+self.refiner_time_block]
            def evaluate(t):
                with self.scopes():
                    return self._query(prepared,grid,features,f,t)
            if self.training and torch.is_grad_enabled() and self.checkpoint_queries:
                part = checkpoint(evaluate,query_times,use_reentrant=False)
            else:
                part = evaluate(query_times)
            pieces.append(part)
        value = torch.cat(pieces,0)
        return value if normalized else self.normalizer.decode_pressure(value[None],f[:,4])[0]

    def forward(self, c, f, times_s=None, *, normalized=True):
        """TRAIN-friendly nn.Module call; complete future392 when times omitted."""
        return self.forecast_at_times(c,f,self.future_times_s if times_s is None else times_s,normalized=normalized)

    @torch.no_grad()
    def forward_cf_full(self, c, f, *, known8_pa):
        """Single deployment API returns contiguous float32 Pa ZXT201x201x400.

        c/f prediction is finished before known8 is inspected or copied. Internal
        query blocks bound memory; callers receive one complete tensor. The
        model has no access to real8 for any future computation or cache key.
        """
        was_training = self.training
        self.eval()
        try:
            future = self.forecast_at_times(c,f,self.future_times_s,normalized=False)
        finally:
            self.train(was_training)
        known = torch.as_tensor(known8_pa)
        if known.dtype != torch.float32:
            raise TypeError('known8 delivery requires original float32 Pa bytes')
        if known.shape == (1,8,201,201): known = known[0]
        if known.shape == (201,201,8): known = known.permute(2,0,1)
        if known.shape != (8,201,201) or not bool(torch.isfinite(known).all()):
            raise ValueError('known8 output-only shape must be [8,201,201] or ZXT8')
        whole = future.new_empty((400,201,201))
        whole[:8].copy_(known.to(device=future.device))
        whole[8:].copy_(future)
        return whole.permute(1,2,0).contiguous()


def build_r24_cf_preserved_operator(device='cpu', **kwargs):
    return R24CFPreservedOperator(device, **kwargs)
