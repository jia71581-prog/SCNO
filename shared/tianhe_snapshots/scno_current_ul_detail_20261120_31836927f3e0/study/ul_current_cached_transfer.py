"""Frozen E200 encodings, folded details and U96 features for one instance."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from finetune_ul_transfer import detail_heads
from ul_cached_transfer_readout import CachedTransferReadout, _context_tensors, _version
from ul_current_lora import clone_current, current_layers


@dataclass(frozen=True)
class CurrentInstanceContext:
    velocity: torch.Tensor
    source: torch.Tensor
    backbone: object
    geometry: object
    hidden_u96: torch.Tensor | None

    @property
    def eligible(self):
        return self.hidden_u96 is not None


@torch.no_grad()
def encode_instance(model, velocity, source):
    v, s = model.backbone._inputs(velocity, source)
    context = model.backbone.encode(v, s)
    geometry = None
    hidden = None
    if bool(model.profiles(v)[-1].any()):
        geometry = model.geometry(v, s)
        inherited = context.features[:, :, 1:].flatten(2).transpose(1, 2)
        hidden = model.readout.spatial[:4](torch.cat((inherited, geometry.features), dim=-1))
    return CurrentInstanceContext(v, s, context, geometry, hidden)


def render_arrival(model, instance, flat_coefficients, rows):
    pulses, late = flat_coefficients.split((30, 81), dim=-1)
    coefficients = pulses.reshape(*pulses.shape[:-1], 6, 5), late
    geometry = instance.geometry
    pieces = []
    times = model.full_times_s[rows][None, None]
    for part in times.split(8, dim=-1):
        rendered = model.readout.render(coefficients, part, geometry.times, geometry.mask, geometry.frequency)
        pieces.append(rendered[0].transpose(0, 1).reshape(part.shape[-1], 200, 201))
    return F.pad(torch.cat(pieces), (0, 0, 1, 0))


@torch.no_grad()
def original_pressure(model, instance, rows):
    backbone = model.backbone.decode(instance.backbone, rows, normalized=True)[0]
    if instance.eligible:
        head = render_arrival(model, instance, model.readout.spatial[4](instance.hidden_u96), rows)
        return backbone + head, head
    return backbone, None


class CachedCurrentTransferReadout:
    def __init__(self, master, instance, direction, original_head, *, adapt_arrival=True):
        self.model = master
        self.instance = instance
        self.direction = direction
        self.original_head = original_head
        self.all_rows = torch.arange(401, device=master.device)
        self.backbone_cache = CachedTransferReadout(master.backbone, instance.backbone, self.all_rows)
        self.arrival = current_layers(master)[-1]
        self.adapt_arrival = adapt_arrival and instance.eligible
        if not adapt_arrival and any(parameter.requires_grad or not torch.equal(parameter, anchor)
                for parameter, anchor in zip(self.arrival.lora_parameters(), (self.arrival.U0, self.arrival.V0))):
            raise ValueError("The frozen U96 readout must retain its original affine")
        with torch.no_grad():
            self.anchor = ((instance.hidden_u96 @ self.arrival.V0) @ self.arrival.U0.T
                           if self.adapt_arrival else None)
        self.fixed_stamp = self._stamp()

    def _stamp(self):
        fixed = [(n, _version(p)) for n, p in self.model.readout.named_parameters() if not p.requires_grad]
        fixed += [(n, _version(b)) for n, b in self.model.readout.named_buffers()]
        fixed += [(n, _version(t)) for n, t in _context_tensors(self.instance, "instance")
                  if not n.startswith("instance.backbone")]
        fixed += [("direction", _version(t)) for t in self.direction.tensors()]
        return tuple(fixed)

    def fork(self):
        """Clone the parameter topology while reusing the exact frozen cache values."""
        self.check_current()
        model = clone_current(self.model)
        other = object.__new__(type(self))
        other.__dict__.update(vars(self))
        other.model = model
        other.arrival = current_layers(model)[-1]
        # Only the adapter parameters and their ownership change in this exact clone.
        old = self.backbone_cache
        new = object.__new__(CachedTransferReadout)
        new.__dict__.update(vars(old))
        new.model = model.backbone
        new.heads = detail_heads(model.backbone)
        new.factor_ids = {id(p) for head in new.heads for p in head[2].lora_parameters()}
        new.fixed_stamp = new._stamp()
        other.backbone_cache = new
        other.fixed_stamp = other._stamp()
        return other

    def check_current(self):
        if self.fixed_stamp != self._stamp():
            raise RuntimeError("An input, frozen U96 tensor or fixed direction changed; rebuild the cache")
        self.backbone_cache.check_current()

    def decode(self, rows=None, *, normalized=True):
        if self.fixed_stamp != self._stamp():
            raise RuntimeError("An input, frozen U96 tensor or fixed direction changed; rebuild the cache")
        if rows is None:
            rows = self.all_rows
        pressure = self.backbone_cache.decode(self.instance.backbone, rows, normalized=True)[0]
        if self.instance.eligible:
            pressure = pressure + self.original_head.index_select(0, rows)
        if self.adapt_arrival:
            delta = self.arrival.delta_output(self.instance.hidden_u96, self.anchor)
            pressure = pressure + render_arrival(self.model, self.instance, delta, rows)
        pressure = pressure + self.model.propagation_gain * self.direction.rows(rows)
        if not normalized:
            pressure = self.model.normalizer.decode_pressure(pressure[None], self.instance.source[:, 4])[0]
        return pressure

    def direct(self, rows=None, *, normalized=True, fresh_encoding=False):
        rows = self.all_rows if rows is None else rows
        if fresh_encoding:
            pressure = self.model(self.instance.velocity, self.instance.source, rows, normalized=True)[0]
        else:
            pressure = self.model.backbone.decode(self.instance.backbone, rows, normalized=True)[0]
            if self.instance.eligible:
                coeff = self.model.readout.spatial[4](self.instance.hidden_u96)
                pressure = pressure + render_arrival(self.model, self.instance, coeff, rows)
        pressure = pressure + self.model.propagation_gain * self.direction.rows(rows)
        return (pressure if normalized
                else self.model.normalizer.decode_pressure(pressure[None], self.instance.source[:, 4])[0])
