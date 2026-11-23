"""Native201 four-second transfer operator with ten temporal-band FNOs.

All trained V2 tensors remain the same Parameter objects. Actual physical
seconds enter inherited features with their original one-second normalization
and fixed .11-second anchor. The query range is extended, not time-rescaled.
The new Deepwave labels have a nonzero physical top row, so raw base/refiner
outputs and new FNO corrections are composed without the old top projection.
This is transfer to a different teacher; it is not an identical parent function.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .paper201_multiband_fno_20261020 import MultiBandConfig, TemporalBandFNO

SCHEMA = 'PAPER4S_NATIVE201_BATCHED_MULTIBAND_TRANSFER_V1'


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class Paper4sConfig:
    duration_s: float = 4.
    period_s: float = 5.
    cutoffs_hz: tuple = tuple(float(i) for i in range(8, 81, 8))
    width: int = 12
    spatial_modes: int = 6
    depth: int = 2
    internal_sizes: tuple = (51, 51, 51, 101, 101, 101, 101, 101, 101, 101)
    activation_epochs: tuple = tuple(range(0, 50, 5))
    records_per_epoch: int = 600
    coefficient_l2: float = 1e-6

    def __post_init__(self):
        for name in ('cutoffs_hz', 'internal_sizes', 'activation_epochs'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        _require(self.duration_s == 4. and math.isfinite(self.period_s) and
                 self.period_s > self.duration_s, 'four-second domain needs a strictly longer basis period')
        _require(type(self.records_per_epoch) is int and self.records_per_epoch > 0,
                 'positive integer records_per_epoch required')
        self.head_config()  # Reuse validated numerical dimensions and cutoff grid.

    @property
    def activation_exposures(self):
        return tuple(epoch*self.records_per_epoch for epoch in self.activation_epochs)

    def active_bands(self, exposures):
        _require(type(exposures) is int and exposures >= 0, 'nonnegative integer committed exposures required')
        return sum(exposures >= first for first in self.activation_exposures)

    def head_config(self):
        return MultiBandConfig(cutoffs_hz=self.cutoffs_hz, period_s=self.period_s,
            width=self.width, spatial_modes=self.spatial_modes, depth=self.depth,
            internal_sizes=self.internal_sizes, activation_steps=self.activation_exposures,
            coefficient_l2=self.coefficient_l2)


def native_field_times():
    """Teacher's exact FP64 construction, cast once on CPU to model FP32."""
    receiver = np.arange(3201, dtype=np.int64)*10*.000125
    return torch.from_numpy(receiver[::4].copy().astype(np.float32))


class FourSecondBandFNO(TemporalBandFNO):
    """Identical spectral coordinates; raw native output and four-second basis."""
    def forward(self, features):
        hidden = F.interpolate(features, size=(self.size, self.size), mode='bilinear', align_corners=True)
        hidden = self.lift(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = F.interpolate(hidden, size=features.shape[-2:], mode='bilinear', align_corners=True)
        # The native201 coefficient head runs AFTER resizing hidden features.
        return self.output(hidden)

    def basis(self, times):
        phase = (2*math.pi)*times[..., None]*self.frequencies
        return torch.where(self.sine_columns, phase.sin(), phase.cos())


@dataclass
class EncodedBatch:
    owner: int
    exposures: int
    c: torch.Tensor
    f: torch.Tensor
    prepared: object
    grid: object
    static_features: tuple
    mass: torch.Tensor
    coefficients: tuple
    dormant_connection: torch.Tensor


def source_maps(source, size, spacing):
    """B independent bilinear source maps; no summing different records."""
    loc = source[:, :2]/spacing
    lower = torch.floor(loc).long().clamp(0, size-2)
    fraction = loc-lower
    ix, iz = lower.unbind(1)
    fx, fz = fraction.unbind(1)
    indices = torch.stack((iz*size+ix, iz*size+ix+1, (iz+1)*size+ix, (iz+1)*size+ix+1), 1)
    weights = torch.stack(((1-fx)*(1-fz), fx*(1-fz), (1-fx)*fz, fx*fz), 1)
    return source.new_zeros((len(source), size*size)).scatter_add(1, indices, weights).reshape(-1, 1, size, size)


def batched_medium_mass(dg, velocity):
    """Same DG quadrature for each medium, retaining the velocity derivative."""
    cells = velocity[:, 0].unfold(1, dg.ec+1, dg.ec).unfold(2, dg.ec+1, dg.ec)
    cells = cells.reshape(len(velocity), dg.elements, dg.elements, -1)
    quadrature = torch.matmul(cells, dg.velocity_quadrature.T)
    inverse = (3000./quadrature).square()*dg.quadrature_weights
    return dg.Jvol*torch.matmul(inverse, dg.quadrature_phi.square())


def batched_dg(dg, field, mass, batch, count):
    """Actual fused B*T feature exchange with each record's own modal mass."""
    if not dg.enabled:
        return field
    modal = dg.project(field)@dg.modal_mix.to(field.dtype).T
    volume, face = dg.modal_components(modal)
    denominator = mass[:, None].expand(batch, count, *mass.shape[1:]).reshape(batch*count, 1, *mass.shape[1:])
    volume = volume/denominator*dg.operator_scale
    face = face/denominator*dg.operator_scale
    volume = torch.einsum('dc,ncexm->ndexm', dg.volume_mix.to(field.dtype), volume)
    face = torch.einsum('dc,ncexm->ndexm', dg.face_mix.to(field.dtype), face)
    update = volume*dg.volume_gate.tanh()[None, :, None, None, None]
    if dg.face_enabled:
        update = update+face*dg.face_gate.tanh()[None, :, None, None, None]
    return field+dg.reconstruct(update)


def batched_cycle(cycle, raw, medium, source, velocity, normalized_velocity, times,
                  source_parameters, spacing, depth):
    """The inherited learned cycle, fused over B*T with record-local conditions."""
    batch, count, height, width = raw.shape
    n = batch*count
    field = raw.reshape(n, 1, height, width)
    f0, t0 = source_parameters[:, 2:3], source_parameters[:, 3:4]
    relative = times-t0
    phase = (2*math.pi)*f0*relative
    temporal = torch.stack((times, relative, (f0/50).expand_as(times), phase.sin(), phase.cos()), 2)
    temporal = temporal.reshape(n, 5, 1, 1).expand(n, 5, height, width)
    def repeated(value):
        return value[:, None].expand(batch, count, *value.shape[1:]).reshape(n, *value.shape[1:])
    features = torch.cat((field, repeated(cycle.medium_proj(medium)),
                          repeated(cycle.source_proj(source)), temporal), 1)
    if cycle.lift_mode == 'additive':
        initial, forcing = cycle.Lu(features), cycle.Lf(features)
    else:
        condition = features[:, 1:]
        def lift(layer):
            value = F.conv2d(field, layer.weight[:, :1], bias=None)
            gate = F.conv2d(condition, layer.weight[:, 1:], bias=layer.bias)
            return value*(1+.1*gate.tanh())
        initial, forcing = lift(cycle.Lu), lift(cycle.Lf)
    kappa = (2*math.pi)*cycle.alpha[None, :, None, None]*f0[:, :, None, None]*spacing/velocity
    condition = repeated(cycle.La(torch.cat((normalized_velocity, kappa), 1)))
    final = cycle.compose(condition, forcing, initial, depth)
    return cycle.Q(final-initial).reshape(batch, count, height, width)


def raw_base_block(base, prepared, times, grid):
    """Pinned V2 active branches, replacing only its single-record cycle call."""
    batch, count = times.shape
    points = grid.height*grid.width
    xy = grid.coords_xy_m[:, None].expand(-1, count, -1, -1)
    time_values = times[:, :, None].expand(-1, -1, points)
    coordinates = torch.cat((xy, time_values[..., None]), -1).reshape(batch, count*points, 3)
    travel = base._repeat_travel(grid.travel, count)
    trunk, bundle = base.coordinate_encoder(coordinates, prepared.source_parameters, travel,
        domain_x_m=base.domain_x_m, domain_z_m=base.domain_z_m, domain_t_s=base.domain_t_s,
        anchor_time_s=prepared.anchor_time_s)
    carrier = None
    if base.carrier_mode != 'off':
        mirror = grid.mirror_tau_s[:, None, :].expand(-1, count, -1).reshape(batch, count*points)
        carrier = base._carrier_features(prepared, coordinates[..., :2], coordinates[..., 2], mirror_tau_s=mirror)
    travel_rank = base.travel_branch(travel, bundle, domain_t_s=base.domain_t_s,
        domain_diagonal_m=math.sqrt(base.domain_x_m**2+base.domain_z_m**2),
        carrier_features=carrier, phase_features=None)
    normalized_xy = grid.coords_xy_normalized[:, None].expand(-1, count, -1, -1).reshape(batch, count*points, 2)
    # Uncached live fusion is identical algebra and accepts a genuine B axis.
    fused = base.fusion(prepared.medium.encoding, prepared.source_encoding, normalized_xy,
                        prepared.record_to_medium, travel_rank, trunk)
    coarse = fused.normalized_pressure.reshape(batch, count, grid.height, grid.width)
    velocity = prepared.medium.velocity_mps[prepared.record_to_medium]
    raw = base.dense_decoder(prepared.medium.encoding, prepared.source_encoding, prepared.source_parameters,
        prepared.record_to_medium, times, coarse, cached_spatial=None,
        velocity=velocity if base.dense_decoder.reflection_branch is not None else None)
    delta = batched_cycle(base.scno_cycle, raw,
        prepared.medium.encoding.pyramid[0][prepared.record_to_medium], prepared.source_encoding.map_field,
        velocity, prepared.normalizer.encode_velocity(velocity), times, prepared.source_parameters,
        grid.x_m[1]-grid.x_m[0], int(base.scno_depth))
    return raw+delta


def raw_batched_refiner(refiner, features, base, times, source, mass):
    """Original refiner algebra in record-major B*T order, without top zeroing."""
    batch, count = times.shape
    n = batch*count
    base = base.reshape(n, 1, 201, 201)
    scale = base.square().mean(dim=(-2, -1), keepdim=True).add(1e-6).sqrt()
    value = base/scale
    dx = F.pad(value[:, :, :, 1:]-value[:, :, :, :-1], (0, 1, 0, 0))
    dz = F.pad(value[:, :, 1:, :]-value[:, :, :-1, :], (0, 0, 0, 1))
    hidden = torch.cat((value, dx, dz, value.abs()), 1)
    flat_times = times.reshape(n, 1)
    phase = (2*math.pi)*flat_times*refiner.time_hz.reshape(1, -1)
    src = source/source.new_tensor([[2000., 2000., 15., 1., 1.]])
    src = src[:, None].expand(batch, count, 5).reshape(n, 5)
    embedding = refiner.embedding(torch.cat((flat_times, phase.sin(), phase.cos(), src,
                                              scale.reshape(n, 1).log()), 1))
    skips = []
    for block, condition in zip(refiner.dynamic, features):
        condition = condition[:, None].expand(batch, count, *condition.shape[1:]).reshape(n, *condition.shape[1:])
        hidden = block(hidden, embedding)+condition
        skips.append(hidden)
    nz, nx = hidden.shape[-2:]
    channels = hidden.shape[1]
    tokens = hidden.flatten(2).transpose(1, 2)
    normalized = refiner.token_norm(tokens)
    q, k, v = refiner.qkv(normalized).reshape(n, nz*nx, 3, 4, channels//4).permute(2, 0, 3, 1, 4).unbind(0)
    attention = ((q@k.transpose(-2, -1))/math.sqrt(channels//4)).softmax(-1)
    global_value = (attention@v).transpose(1, 2).reshape(n, nz*nx, channels)
    tokens = tokens+refiner.attention_out(global_value)
    hidden = tokens.transpose(1, 2).reshape(n, channels, nz, nx)
    # Use the same deterministic hidden-feature resize as the inherited module.
    resize = refiner.forward.__func__.__globals__['resize_bilinear']
    for block, skip in zip(refiner.up, skips[-2::-1]):
        hidden = resize(hidden, skip.shape[-2:])
        hidden = block(torch.cat((hidden, skip), 1), embedding)
    hidden = batched_dg(refiner.dg, hidden, mass, batch, count)
    return (refiner.output(hidden)*scale).reshape(batch, count, 201, 201)


class Paper4sMultiBandOperator(nn.Module):
    """True batch transfer: static B encoding, B*T decoding, native201 output.

    ``encode`` retains a shared autograd graph. A trainer that backpropagates
    multiple time chunks must retain that graph until its last query or use a
    leaf/VJP bridge. Encodings must never cross optimizer updates.
    """
    def __init__(self, backbone, config=None):
        super().__init__()
        self.backbone = backbone
        self.config = Paper4sConfig() if config is None else config
        _require(isinstance(self.config, Paper4sConfig), 'Paper4sConfig required')
        _require(backbone.base.domain_t_s == 1., 'inherited one-second feature normalization required')
        base = backbone.base
        _require(base.phase_mode == 'off' and all(getattr(base, key) is None for key in
                 ('arrival_head', 'coarse_channel', 'scattering_head', 'local_unet')),
                 'pinned V2 branch contract changed; additional branches require explicit batched implementation')
        self.heads = nn.ModuleList(FourSecondBandFNO(self.config.head_config(), i)
                                  for i in range(len(self.config.cutoffs_hz))).to(self.device)
        self.register_buffer('field_times_s', native_field_times().to(self.device))
        self.register_buffer('training_exposures', torch.tensor(0, dtype=torch.int64, device=self.device))
        self.register_load_state_dict_post_hook(_verify_fixed_contract)

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    @property
    def normalizer(self):
        return self.backbone.normalizer

    @property
    def full_times_s(self):
        return self.field_times_s

    @property
    def native_times_s(self):
        return self.field_times_s

    @property
    def full_native_frames(self):
        return tuple(range(801))

    @property
    def active_band_count(self):
        return self.config.active_bands(int(self.training_exposures))

    def scopes(self):
        return self.backbone.scopes()

    def set_training_exposures(self, exposures):
        active = self.config.active_bands(exposures)
        self.training_exposures.fill_(exposures)
        return active

    def get_extra_state(self):
        return dict(schema=SCHEMA, config=asdict(self.config), input='B sharp201 velocities and B single-source5',
            output='B,T,201,201 directly decoded native nodes; full field801',
            physical_duration_s=4., inherited_feature_time_scale_s=1., inherited_anchor_s=.11,
            inherited_normalizer=copy.deepcopy(vars(self.normalizer.metadata)),
            boundary='raw teacher top; no nodal zero projection', physical_time_rescaling=False,
            receiver='native201 raw row2; not fine401 B5',
            optimizer_policy='new-data phase resets main and meta optimizers; parent complete state retained')

    def set_extra_state(self, state):
        _require(state == self.get_extra_state(), 'four-second model semantic contract differs')

    def _inputs(self, c, f):
        c = torch.as_tensor(c, device=self.device, dtype=torch.float32)
        f = torch.as_tensor(f, device=self.device, dtype=torch.float32)
        if c.shape == (201, 201):
            c = c[None, None]
        if f.shape == (5,):
            f = f[None]
        _require(c.ndim == 4 and c.shape[1:] == (1, 201, 201) and c.shape[0] > 0 and
                 f.shape == (c.shape[0], 5), 'c[B,1,201,201] and f[B,5] required')
        _require(torch.isfinite(c).all() & (c > 0).all() & torch.isfinite(f).all(), 'finite positive medium and source required')
        _require((f[:, :2] >= 0).all() & (f[:, :2] <= 2000).all() & (f[:, 4] > 0).all() &
                 ((f[:, 2]-20.).abs() <= 1e-5).all() & ((f[:, 3]-.075).abs() <= 1e-6).all(),
                 'registered source requires x/z0..2000m,20Hz,t0=.075,positiveA')
        return c, f

    def encode(self, c, f):
        c, f = self._inputs(c, f)
        batch = len(c)
        parent, base = self.backbone, self.backbone.base
        with self.scopes():
            coarse = parent._restriction(c).contiguous()
            medium = base.encode_medium(coarse, self.normalizer)
            # The .11s anchor is a retained model feature, not the new frame88.
            anchor = parent.native_times_s[88:89].expand(batch)
            prepared = base.prepare_sources(medium, f, source_maps(f, 101, 20.), self.normalizer,
                record_to_medium=torch.arange(batch, device=self.device),
                ic_snapshots_physical=c.new_zeros((batch, 8, 101, 101)), anchor_time_s=anchor)
            grid = base.prepare_dense_grid(prepared, x_m=parent.x_m[::2], z_m=parent.z_m[::2])
            z, x = torch.meshgrid(parent.z_m, parent.x_m, indexing='ij')
            x, z = x.expand(batch, -1, -1), z.expand(batch, -1, -1)
            coords = torch.stack((2*x/2000-1, 2*z/2000-1,
                                  (x-f[:, 0, None, None])/2000, (z-f[:, 1, None, None])/2000), 1)
            source = source_maps(f, 201, 10.)
            static = torch.cat((c/3000-1, c.new_zeros((batch, 8, 201, 201)), source, coords), 1)
            features = []
            for block in parent.refiner.static:
                static = block(static)
                features.append(static)
            mass = batched_medium_mass(parent.refiner.dg, c)
            frequency = (f[:, 2]/15)[:, None, None, None].expand(batch, 1, 201, 201)
            delay = f[:, 3, None, None, None].expand(batch, 1, 201, 201)
            band_inputs = torch.cat((c/3000-1, source, coords, frequency, delay), 1)
            coefficients, dormant = [], c.new_zeros(())
            for i, head in enumerate(self.heads):
                if i < self.active_band_count:
                    coefficients.append(head(band_inputs))
                else:
                    coefficients.append(None)
                    # Preserve fixed-block controller gradients at zero cost in
                    # spatial execution; optimizer controls dormant step clocks.
                    dormant = dormant+sum(p.reshape(-1)[0]*0 for p in head.parameters())
        return EncodedBatch(id(self), int(self.training_exposures), c, f, prepared, grid,
                            tuple(features), mass, tuple(coefficients), dormant)

    def query(self, encoded, times, *, normalized=True):
        _require(isinstance(encoded, EncodedBatch) and encoded.owner == id(self), 'encoding belongs to a different model')
        _require(encoded.exposures == int(self.training_exposures), 'encoding curriculum changed')
        batch = len(encoded.c)
        times = torch.as_tensor(times, device=self.device, dtype=torch.float32)
        if times.ndim == 1:
            times = times[None].expand(batch, -1)
        _require(times.ndim == 2 and times.shape[0] == batch and times.shape[1] > 0,
                 'times[T] or times[B,T] required')
        _require(torch.isfinite(times).all() & (times >= 0).all() & (times <= self.config.duration_s).all(),
                 'physical query seconds must lie in [0,4]')
        count = times.shape[1]
        def evaluate(query_times):
            with self.scopes():
                raw = raw_base_block(self.backbone.base, encoded.prepared, query_times, encoded.grid)
                _require(raw.shape == (batch, count, 101, 101), 'inherited batched coarse shape differs')
                physical = self.normalizer.decode_pressure(raw, encoded.f[:, 4])
                intermediate = self.backbone._bicubic201(physical.reshape(batch*count, 1, 101, 101))
                base = self.normalizer.encode_pressure(intermediate.reshape(batch, count, 201, 201), encoded.f[:, 4])
                correction = raw_batched_refiner(self.backbone.refiner, encoded.static_features,
                                                 base, query_times, encoded.f, encoded.mass)
                result = base+correction+encoded.dormant_connection
                for head, coefficients in zip(self.heads, encoded.coefficients):
                    if coefficients is not None:
                        result = result+torch.einsum('btk,bkzx->btzx', head.basis(query_times), coefficients)
                return result
        if self.training and torch.is_grad_enabled() and self.backbone.checkpoint_queries:
            result = checkpoint(evaluate, times, use_reentrant=False)
        else:
            result = evaluate(times)
        return result if normalized else self.normalizer.decode_pressure(result, encoded.f[:, 4])

    def coefficient_regularization(self, encoded):
        _require(encoded.owner == id(self) and encoded.exposures == int(self.training_exposures), 'stale/foreign encoding')
        penalty = encoded.dormant_connection
        for coefficient in encoded.coefficients:
            if coefficient is not None:
                penalty = penalty+coefficient.square().mean()
        return self.config.coefficient_l2*penalty

    def forward(self, c, f, times_s=None, *, normalized=True):
        encoded = self.encode(c, f)
        return self.query(encoded, self.full_times_s if times_s is None else times_s, normalized=normalized)

    def forecast_at_times(self, c, f, times_s, *, normalized=True):
        return self(c, f, times_s, normalized=normalized)


def _verify_fixed_contract(module, incompatible_keys):
    _require(module.training_exposures.shape == () and module.training_exposures.dtype == torch.int64,
             'scalar int64 exposure clock required')
    module.config.active_bands(int(module.training_exposures))
    _require(torch.equal(module.field_times_s.cpu(), native_field_times()), 'exact teacher field axis changed')
    _require(module.backbone.base.domain_t_s == 1., 'inherited time feature scale changed')
    for index, head in enumerate(module.heads):
        first = 0 if index == 0 else round(module.config.cutoffs_hz[index-1]*module.config.period_s)+1
        last = round(module.config.cutoffs_hz[index]*module.config.period_s)
        frequencies, flags = [], []
        for mode in range(first, last+1):
            frequencies.append(mode/module.config.period_s); flags.append(False)
            if mode:
                frequencies.append(mode/module.config.period_s); flags.append(True)
        _require(torch.equal(head.frequencies, head.frequencies.new_tensor(frequencies)) and
                 torch.equal(head.sine_columns, head.sine_columns.new_tensor(flags)), 'physical temporal basis changed')


def build_from_parent(parent_dir=None, config=None, device='cpu'):
    """Load the complete pinned V2 parent, transfer tensors, retain its provenance.

    Old optimizers/controller are validated by load_parent and deliberately not
    returned as new-stage optimizers. The driver explicitly creates fresh main
    and meta states for the new teacher/data/task domain.
    """
    from . import paper201_multiband_state_20261020 as state
    parent_dir = state.PARENT_DIR if parent_dir is None else Path(parent_dir)
    payload, cfg, backbone, old_opt, old_controller, old_meta = state.load_parent(parent_dir, 'cpu')
    ids = {n: id(p) for n, p in backbone.named_parameters()}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026102141)
        model = Paper4sMultiBandOperator(backbone, config)
    _require(all(id(dict(model.backbone.named_parameters())[n]) == value for n, value in ids.items()),
             'transfer replaced a parent parameter')
    lineage = dict(schema=SCHEMA+'_LINEAGE', parent_checkpoint=str((parent_dir/'latest.pt').resolve()),
        parent_sha256=state.PARENT_SHA, parent_config=copy.deepcopy(cfg),
        parent_config_sha256=state.rt.sha(parent_dir/'TRAINING_CONFIG.json'),
        parent_progress={key: copy.deepcopy(payload[key]) for key in ('updates', 'epoch', 'offset', 'task_exposures')},
        parent_main_optimizer_groups=len(old_opt.param_groups), parent_controller_blocks=len(old_controller.parameter_names),
        parent_meta_optimizer_states=len(old_meta.state), parent_full_state_validated=True,
        all_parent_parameter_objects_retained=True,
        transfer='original parameter values; new physical query domain, source and top-boundary semantics',
        optimizer_policy='new main AdamW and controller/meta Adam initialized for this new-data phase')
    return model.to(device), lineage
