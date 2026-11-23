"""Exact frozen-part cache for native-401 rank-four SCNO instance transfer.

Only three second-hidden U,V pairs may change. No predictions or labels are
inserted into this cache: BASE and first-SiLU features come from the frozen
model/input, and final affine/time contractions come from frozen weights.
Fixed extra_pressure belongs to NativePhysicalObjective and stays outside
this readout, so it is added exactly once under the existing loss protocol.
"""

from __future__ import annotations

from time import perf_counter

import torch

from finetune_ul_transfer import detail_heads, native_axis
from lora_v2 import CompensatedRank4


def _version(tensor):
    try:
        return (id(tensor), tensor.data_ptr(), tensor._version, tuple(tensor.shape),
                str(tensor.dtype), str(tensor.device), tensor.requires_grad)
    except RuntimeError as error:
        raise ValueError("the transfer cache requires versioned tensors; construct outside inference_mode") from error


def _context_tensors(value, prefix="context"):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _context_tensors(child, f"{prefix}.{index}")
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _context_tensors(value[key], f"{prefix}.{key}")
    elif hasattr(value, "__dict__"):
        # Production TravelContext is a NamedTuple; SimpleNamespace is useful
        # for controlled CPU checks. Contexts must not hold a model or cycles.
        for key, child in sorted(vars(value).items()):
            yield from _context_tensors(child, f"{prefix}.{key}")


class CachedTransferReadout:
    """Drop-in ``decode`` view of one frozen context and a registered row union.

    Integration without modifying NativePhysicalObjective::

        cached = CachedTransferReadout.from_objective(objective)
        objective.model = cached
        result = fit(original_model, objective, parent_sha, ...)

    Keep ``original_model`` for optimizer/state operations. After adapter
    restoration, model/device changes or any fixed-weight edit, rebuild this
    cache. No trainable activation or residual-target cache is retained.
    """

    def __init__(self, model, context, query_indices):
        started = perf_counter()
        native_axis(model.full_times_s)
        self.model = model
        self.context = context
        self.heads = detail_heads(model)
        if len(context.source) != 1:
            raise ValueError("cached physical transfer is bound to one explicit target instance")
        if any(not isinstance(head[2], CompensatedRank4) for head in self.heads):
            raise ValueError("insert compensated rank-four factors before binding the transfer cache")
        # Newly inserted nn.Modules default to training=True even when their
        # containing frozen head was already in eval mode. This reviewed
        # affine-only adapter has no training-dependent computation.
        for head in self.heads:
            head[2].eval()
        self.factor_ids = {id(p) for head in self.heads for p in head[2].lora_parameters()}
        self._validate_roles()
        indices = torch.as_tensor(query_indices, device=model.device)
        if (indices.dtype not in (torch.int32, torch.int64) or indices.ndim != 1 or not indices.numel()
                or bool((indices < 0).any() or (indices > 400).any())):
            raise ValueError("cache indices must be a nonempty vector of native integer rows in [0,400]")
        self.indices = indices.long().unique(sorted=True)
        self.lookup = torch.full((401,), -1, dtype=torch.long, device=model.device)
        self.lookup[self.indices] = torch.arange(len(self.indices), device=model.device)
        decoder = model.parent.decoder
        with torch.no_grad():
            # This inherited readout includes the actual source/travel and
            # Gabor BASE contributions; no replacement BASE approximation.
            self.base = model.parent.decode(context, self.indices, normalized=True,
                                            detail_enabled=False).detach().clone()
            active = context.features[:, :, 1:]
            _, packets = decoder.detail_transform.analyze(decoder.detail_projection(active))
            first_hidden, factors = [], []
            for head, packet, temporal, level in zip(self.heads, packets, decoder.detail_time,
                                                       decoder.detail_transform.levels):
                block = packet.permute(0, 2, 3, 1, 4).flatten(-2)
                local = torch.cat((block, level.valid_detail.to(block).expand(len(block), -1, -1, -1)), dim=-1)
                first_hidden.append(head[1](head[0](local)).detach().clone())
                rank, slots, width = decoder.detail_latent, level.valid_detail.shape[-1], head[-1].in_features
                rows = temporal.weight[self.indices]
                weight = (rows @ head[-1].weight.reshape(rank, -1)).reshape(len(rows), slots, width)
                bias = rows @ head[-1].bias.reshape(rank, slots) if head[-1].bias is not None else None
                factors.append((weight.detach().clone(), None if bias is None else bias.detach().clone()))
            self.first_hidden = tuple(first_hidden)
            self.factors = tuple(factors)
            self.source_gate = (context.source[:, 4] != 0)[:, None, None, None].detach().clone()
            self.physical_scale = (model.normalizer.metadata.pressure_scale_pa *
                                   context.source[:, 4])[:, None, None, None].detach().clone()
        self.fixed_stamp = self._stamp()
        self.preparation_host_wall_s = perf_counter() - started

    def __getattr__(self, name):
        return getattr(self.model, name)

    @classmethod
    def from_objective(cls, objective):
        rows = [objective.observation_indices, objective.observation_indices.new_tensor([0, 1, 2])]
        for centers, _ in objective.support.values():
            rows.extend((centers - 1, centers, centers + 1))
        return cls(objective.model, objective.context, torch.cat(rows))

    def _validate_roles(self):
        trainable = {id(p) for p in self.model.parameters() if p.requires_grad}
        if trainable != self.factor_ids or len(trainable) != 6:
            raise ValueError("only the same six live U,V factors may remain trainable")
        if any(module.training for module in self.model.modules()):
            raise ValueError("the frozen-part transfer cache requires all modules in eval mode")

    def _stamp(self):
        self._validate_roles()
        fixed = [(name, _version(p)) for name, p in self.model.named_parameters() if id(p) not in self.factor_ids]
        buffers = [(name, _version(b)) for name, b in self.model.named_buffers()]
        context = [(name, _version(t)) for name, t in _context_tensors(self.context)]
        return (tuple(fixed), tuple(buffers), tuple(context), repr(self.model.normalizer.metadata),
                tuple(id(head[2]) for head in self.heads))

    def check_current(self):
        if self._stamp() != self.fixed_stamp:
            raise RuntimeError("a frozen model/input tensor changed; rebuild the transfer cache")

    def decode(self, context, frame_indices=None, *, normalized=True,
               return_parts=False, detail_enabled=True):
        if context is not self.context:
            raise ValueError("cached readout belongs to the exact context used at construction")
        if return_parts:
            raise ValueError("this pressure-only cache does not provide a second parts protocol")
        self.check_current()
        rows = (torch.arange(401, device=self.model.device) if frame_indices is None else
                torch.as_tensor(frame_indices, device=self.model.device))
        if (rows.dtype not in (torch.int32, torch.int64) or rows.ndim != 1
                or bool((rows < 0).any() or (rows > 400).any())):
            raise ValueError("readout expects shared native integer row indices")
        selected = self.lookup[rows.long()]
        if bool((selected < 0).any()):
            raise ValueError("query was not included in the explicit cached row union")
        base = self.base.index_select(1, selected)
        if not detail_enabled:
            return base if normalized else base * self.physical_scale
        details = []
        for head, first, (weight, bias), level in zip(self.heads, self.first_hidden, self.factors,
                                                       self.model.detail_transform.levels):
            # Both factors remain on the live gradient path. Everything before
            # first and after second-SiLU is exactly fixed for this transfer.
            second = head[3](head[2](first))
            coeff = torch.einsum("bxyh,ksh->bkxys", second, weight.index_select(0, selected))
            if bias is not None:
                coeff = coeff + bias.index_select(0, selected)[None, :, None, None, :]
            details.append(coeff.masked_fill(~level.valid_detail, 0))
        correction = self.model.synthesize_correction(None, details, batch=1, count=len(rows),
                                                      device_tensor=self.context.features)
        pressure = base + correction * self.source_gate
        return pressure if normalized else pressure * self.physical_scale

    def receipt(self):
        def size(tensor):
            return 0 if tensor is None else tensor.numel() * tensor.element_size()
        return {"schema": "SCNO_UL_CACHED_LORA_READOUT_V1", "query_count": len(self.indices),
                "native_indices": self.indices.detach().cpu().tolist(),
                "base_bytes": size(self.base), "first_silu_bytes": sum(map(size, self.first_hidden)),
                "folded_affine_time_bytes": sum(size(w) + size(b) for w, b in self.factors),
                "preparation_host_wall_s": self.preparation_host_wall_s,
                "host_wall_timer_is_not_a_CUDA_latency_measurement": True,
                "trainable": "six U,V factors only; pressure and loss unchanged",
                "extra_pressure": "added by the existing objective, outside this cache"}
