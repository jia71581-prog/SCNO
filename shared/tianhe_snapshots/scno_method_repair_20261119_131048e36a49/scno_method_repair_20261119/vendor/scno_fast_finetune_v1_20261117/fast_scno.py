"""SCNO output-only fine-tuning. Import the original current-network classes read-only."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.autograd import forward_ad

from spectral_marmousi import SpectralMarmousiOperator, _capture

POLICY = "scno_fast_finetune_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(named_tensors):
    digest = hashlib.sha256()
    for name, value in named_tensors:
        value = value.detach().cpu().contiguous()
        digest.update(repr((name, value.shape, value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Rank4LoRALinear(nn.Module):
    def __init__(self, base, generator):
        super().__init__()
        if not isinstance(base, nn.Linear) or tuple(base.weight.shape) != (32, 32):
            raise ValueError("the registered second hidden affine map must be 32→32")
        self.base = base.requires_grad_(False)
        bound = 1 / math.sqrt(32)
        values = torch.empty((32, 4), dtype=base.weight.dtype)
        values.uniform_(-bound, bound, generator=generator)
        self.U = nn.Parameter(base.weight.new_zeros((32, 4)))
        self.V = nn.Parameter(values.to(base.weight))

    def lora_parameters(self):
        return (self.U, self.V)

    def forward(self, features):
        return self.base(features) + F.linear(F.linear(features, self.V.T), self.U)


@dataclass(frozen=True)
class DetailHead:
    module: nn.Sequential

    @property
    def fc1(self):
        return self.module[0]

    @property
    def fc2(self):
        return self.module[2]

    @property
    def out(self):
        return self.module[4]


@dataclass(frozen=True)
class FineTuneRegistry:
    pressure_readout: nn.Conv2d
    detail_heads: tuple[DetailHead, ...]


class FrozenQuery(NamedTuple):
    w: torch.Tensor
    scale: torch.Tensor
    hidden: torch.Tensor
    a1: torch.Tensor
    a2: torch.Tensor
    a3: torch.Tensor


def _jvp(function, primals, tangents):
    """Scalar-time directional derivative without reverse-over-reverse AD."""
    # Fused SDPA does not expose forward AD in the installed PyTorch.
    # The math backend evaluates the same attention and supports both AD modes.
    with sdpa_kernel(SDPBackend.MATH), forward_ad.dual_level():
        output = function(*(forward_ad.make_dual(x, dx) for x, dx in zip(primals, tangents)))
        if torch.is_tensor(output):
            primal, tangent = forward_ad.unpack_dual(output)
            return primal, torch.zeros_like(primal) if tangent is None else tangent
        results = [forward_ad.unpack_dual(value) for value in output]
        return FrozenQuery(*(p for p, _ in results)), FrozenQuery(*(
            torch.zeros_like(p) if d is None else d for p, d in results))


def _refiner_features(refiner, features, w, times, source):
    """Frozen feature segment of the inherited r22 ResidualRefiner.forward."""
    n = w.shape[0]
    scale = w.square().mean(dim=(-2, -1), keepdim=True).add(1e-6).sqrt()
    u = w / scale
    dx = F.pad(u[:, :, :, 1:] - u[:, :, :, :-1], (0, 1, 0, 0))
    dz = F.pad(u[:, :, 1:, :] - u[:, :, :-1, :], (0, 0, 0, 1))
    x = torch.cat((u, dx, dz, u.abs()), 1)
    phase = 2 * math.pi * times.reshape(n, 1) * refiner.time_hz.reshape(1, -1)
    src = source / source.new_tensor([[2000., 2000., 15., 1., 1.]])
    emb = refiner.embedding(torch.cat((times.reshape(n, 1), phase.sin(), phase.cos(),
                                       src.expand(n, -1), scale.reshape(n, 1).log()), 1))
    skips = []
    for block, condition in zip(refiner.dynamic, features):
        x = block(x, emb) + condition.expand(n, -1, -1, -1)
        skips.append(x)
    nz, nx = x.shape[-2:]
    width = x.shape[1]
    tokens = x.flatten(2).transpose(1, 2)
    qkv = refiner.qkv(refiner.token_norm(tokens))
    q, k, v = qkv.reshape(n, nz * nx, 3, 4, width // 4).permute(2, 0, 3, 1, 4).unbind(0)
    attention = ((q @ k.transpose(-2, -1)) / math.sqrt(width // 4)).softmax(-1)
    value = (attention @ v).transpose(1, 2).reshape(n, nz * nx, width)
    tokens = tokens + refiner.attention_out(value)
    x = tokens.transpose(1, 2).reshape(n, width, nz, nx)
    resize = sys.modules[type(refiner).__module__].resize_bilinear
    for block, skip in zip(refiner.up, skips[-2::-1]):
        x = block(torch.cat((resize(x, skip.shape[-2:]), skip), 1), emb)
    return scale, x


def _detail_features(branch, registry, fused, hidden, velocity, source, time_features):
    n = hidden.shape[0]
    velocity = (velocity[:, :, 1:] / 3000 - 1).expand(n, -1, -1, -1)
    fused = branch.fused_projection(fused)
    if fused.shape[-2:] != hidden.shape[-2:]:
        fused = F.interpolate(fused, size=hidden.shape[-2:], mode="bilinear", align_corners=True)
    field = torch.cat((fused[:, :, 1:], branch.refined_projection(hidden[:, :, 1:]), velocity), 1)
    _, packets = branch.transform.analyze(field)
    src = source / source.new_tensor([2000., 2000., 15., 1., 1.])
    condition = torch.cat((branch.source_projection(src).expand(n, -1),
                           branch.time_projection(time_features)), -1)
    activations = []
    for level, head, modes in zip(branch.transform.levels, registry.detail_heads, packets):
        modes = modes.permute(0, 2, 3, 1, 4).flatten(-2)
        expanded = condition[:, None, None, :].expand(-1, level.gz, level.gx, -1)
        mask = level.valid_detail[None].expand(n, -1, -1, -1)
        incoming = torch.cat((modes, expanded, mask.to(modes)), -1)
        activations.append(head.module[1](head.fc1(incoming)))
    return activations


class FastSCNO(SpectralMarmousiOperator):
    def __init__(self, config=None, device="cpu"):
        super().__init__(config, device=device)
        inherited_bicubic = self._bicubic201

        def bicubic_with_jvp(value):
            primal, tangent = forward_ad.unpack_dual(value)
            if tangent is None:
                return inherited_bicubic(value)
            # The inherited custom op has no JVP. Its forward is this
            # linear interpolation, so the identical operator maps its tangent.
            return forward_ad.make_dual(inherited_bicubic(primal), inherited_bicubic(tangent))

        self._bicubic201 = bicubic_with_jvp

    def finetune_registry(self):
        from lora_v2 import CompensatedRank4

        branch = self.refiner.spectral_detail
        if len(branch.heads) != 3 or not isinstance(self.refiner.output, nn.Conv2d):
            raise ValueError("registered three-scale SCNO architecture is required")
        heads = tuple(DetailHead(head) for head in branch.heads)
        for head, level in zip(heads, branch.transform.levels):
            if (len(head.module) != 5 or not isinstance(head.fc1, nn.Linear)
                    or head.fc1.out_features != 32 or not isinstance(head.module[1], nn.SiLU)
                    or not isinstance(head.fc2, (nn.Linear, Rank4LoRALinear, CompensatedRank4))
                    or not isinstance(head.module[3], nn.SiLU) or not isinstance(head.out, nn.Linear)
                    or head.out.in_features != 32 or head.out.out_features != level.modes - 16):
                raise ValueError("unexpected registered detail head")
        return FineTuneRegistry(self.refiner.output, heads)

    def enable_fast_finetuning(self, seed, *, checkpoint_identity=None):
        if not self.spectral_detail_enabled:
            raise ValueError("enable the spectral readout before registering fine-tuning")
        registry = self.finetune_registry()
        if any(isinstance(head.fc2, Rank4LoRALinear) for head in registry.detail_heads):
            raise ValueError("fine-tuning has already been registered")
        self.requires_grad_(False).eval()
        self.checkpoint_queries = False
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for head in registry.detail_heads:
            head.module[2] = Rank4LoRALinear(head.fc2, generator)
            head.out.requires_grad_(True)
        registry.pressure_readout.requires_grad_(True)
        self.fast_seed = int(seed)
        self.fast_checkpoint_identity = checkpoint_identity or "mechanical-check-state"
        self.fast_registry = self.finetune_registry()
        self.assert_parameter_whitelist()
        return self

    def assert_parameter_whitelist(self, optimizer=None):
        registry = self.fast_registry
        entries = list(registry.pressure_readout.parameters())
        for head in registry.detail_heads:
            entries.extend(head.fc2.lora_parameters())
            entries.extend(head.out.parameters())
        allowed = {id(p) for p in entries}
        actual = {id(p) for p in self.parameters() if p.requires_grad}
        if len(entries) != len(allowed) or actual != allowed:
            raise AssertionError("SCNO v1 trainable parameter whitelist mismatch")
        if optimizer is not None:
            params = [p for group in optimizer.param_groups for p in group["params"]]
            if len(params) != len(allowed) or {id(p) for p in params} != allowed:
                raise AssertionError("SCNO v1 optimizer parameter whitelist mismatch")
        return entries

    def finetune_receipt(self, optimizer=None):
        self.assert_parameter_whitelist(optimizer)
        trainable = {n: {"shape": list(p.shape), "scalars": p.numel()}
                     for n, p in self.named_parameters() if p.requires_grad}
        return {
            "policy_id": POLICY, "rank": 4, "gain": 1., "lora_dropout": 0.,
            "initialization_seed": self.fast_seed,
            "frozen_checkpoint_identity": self.fast_checkpoint_identity,
            "trainable_parameters": trainable,
            "trainable_scalars": sum(v["scalars"] for v in trainable.values()),
            "frozen_scalars": sum(p.numel() for p in self.parameters() if not p.requires_grad),
            "total_scalars": sum(p.numel() for p in self.parameters()),
            "frozen_parameters": [n for n, p in self.named_parameters() if not p.requires_grad],
            "pressure_normalization": "u_pa / (source_amplitude * pretrained_pressure_scale_pa)",
            "optimizer": None if optimizer is None else type(optimizer).__name__,
            "optimizer_groups": [] if optimizer is None else [
                {**{k: v for k, v in group.items() if k != "params"},
                 "scalars": sum(p.numel() for p in group["params"])}
                for group in optimizer.param_groups],
        }

    def _prepared_frozen(self, c, f):
        cv, fv = self._inputs(c, f)
        with torch.no_grad(), self.scopes():
            cv, fv = cv.detach().clone(), fv.detach().clone()
            prepared, grid, context = self._prepare_cf(cv, fv)
        return cv, fv, prepared, grid, context

    def _times(self, times):
        times = torch.as_tensor(times, dtype=torch.float32, device=self.device).reshape(-1)
        if not times.numel() or not bool(torch.isfinite(times).all()) or bool(
                (times < 0).any() or (times > 1).any()):
            raise ValueError("physical query times must be finite and lie in [0,1]")
        return times

    def _encode(self, instance, times):
        cv, fv, prepared, grid, context = instance
        features, _ = context
        decoder = getattr(self, "_decoder_base_pressure", None)
        active = dict(fusion=self.base.fusion, readout=None, frames=[], refined=[])
        token = _capture.set(active)
        try:
            if decoder is None:
                base = self.base.dense_normalized(prepared, times[None], dense_grid=grid,
                                                  time_block=self.dense_time_block, apply_correction=True)
            else:
                w = decoder(prepared, grid, fv, times, lift=self._bicubic201)
        finally:
            _capture.reset(token)
        if not active["frames"]:
            raise RuntimeError("the pressure decoder did not produce fusion activations")
        if decoder is None:
            if base.shape != (1, times.numel(), grid.height, grid.width):
                raise RuntimeError("the legacy pressure decoder returned an unexpected grid")
            base_pa = self.normalizer.decode_pressure(base, fv[:, 4])
            pa201 = self._bicubic201(base_pa[0, :, None])
            w = self.normalizer.encode_pressure(pa201[:, 0][None], fv[:, 4])[0, :, None]
        if w.shape != (times.numel(), 1, 201, 201):
            raise RuntimeError("the refiner requires normalized pressure on the 201×201 grid")
        scale, hidden = _refiner_features(self.refiner, features, w, times, fv)
        raw = torch.stack([torch.cat(part, dim=1) for part in zip(*active["frames"])], dim=-1)
        if raw.shape != (1, times.numel() * grid.height * grid.width, 2):
            raise RuntimeError("unexpected fusion stream layout")
        fused = raw.reshape(-1, grid.height, grid.width, 2).permute(0, 3, 1, 2)
        horizon = getattr(self, "duration_s", self.base.domain_t_s)
        time_features = self.base.coordinate_encoder.gabor(times, fv[0, 3], fv[0, 2], horizon)
        activations = _detail_features(self.refiner.spectral_detail, self.fast_registry,
                                       fused, hidden, cv, fv, time_features)
        return FrozenQuery(w, scale, hidden, *activations)

    def _encode_chunks(self, instance, times):
        chunks = [self._encode(instance, part) for part in times.split(self.refiner_time_block)]
        return FrozenQuery(*(torch.cat(parts, 0) for parts in zip(*chunks)))

    def encode_frozen_query(self, c, source, times):
        instance = self._prepared_frozen(c, source)
        with torch.no_grad(), self.scopes():
            return self._encode_chunks(instance, self._times(times))

    def encode_frozen_query_jvp(self, c, source, time0):
        return self._encode_jvp(self._prepared_frozen(c, source), time0)

    def _encode_jvp(self, instance, time0):
        times = self._times(time0)
        if times.numel() != 1:
            raise ValueError("the time derivative interface accepts one physical time")
        with torch.no_grad(), self.scopes():
            return _jvp(lambda t: self._encode(instance, t), (times,), (torch.ones_like(times),))

    def decode_finetune_query(self, pack, *, normalized=True, source=None, return_details=False):
        registry = self.fast_registry
        branch = self.refiner.spectral_detail
        correction = registry.pressure_readout(pack.hidden) * pack.scale
        base = pack.w + F.pad(correction[:, :, 1:], (0, 0, 1, 0))
        base = F.pad(base[:, :, 1:], (0, 0, 1, 0))
        coefficients = []
        for level, head, a in zip(branch.transform.levels, registry.detail_heads, pack[3:]):
            delta = head.out(head.module[3](head.fc2(a)))
            coefficients.append(delta.masked_fill(~level.valid_detail[None], 0)[:, None])
        coarsest = branch.transform.levels[-1]
        origin = pack.w.new_zeros((pack.w.shape[0], 1, coarsest.gz * 4, coarsest.gx * 4))
        detail = F.pad(branch.transform.synthesize(origin, coefficients), (0, 0, 1, 0))
        pressure = (base + detail)[:, 0]
        if not normalized:
            if source is None:
                raise ValueError("source parameters are required for physical Pa")
            source = torch.as_tensor(source, device=pressure.device, dtype=pressure.dtype).reshape(1, 5)
            pressure = self.normalizer.decode_pressure(pressure[None], source[:, 4])[0]
        return (pressure, base[:, 0], detail[:, 0], coefficients) if return_details else pressure

    def decode_finetune_query_jvp(self, pack, tangent, *, normalized=True, source=None):
        return _jvp(lambda *p: self.decode_finetune_query(
            FrozenQuery(*p), normalized=normalized, source=source), pack, tangent)


def load_spectral_checkpoint(config, device):
    """Load an explicit current spectral state before adding any LoRA state."""
    checkpoint = config["spectral_checkpoint"]
    if not checkpoint.get("path") or not checkpoint.get("sha256"):
        raise ValueError("provide an actual current spectral checkpoint path and its SHA-256")
    path = Path(checkpoint["path"])
    identity = sha256(path)
    if identity != checkpoint["sha256"]:
        raise ValueError("current spectral checkpoint SHA-256 mismatch")
    model = FastSCNO(config.get("model", {}), device=device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model.enable_fast_finetuning(config["lora_seed"], checkpoint_identity=identity)


class FrozenQueryCache:
    """One fixed medium/source/checkpoint. An LRU of exact FP32 query features."""
    def __init__(self, model, c, source, *, max_bytes, prepare_batch, storage="cpu"):
        if max_bytes < 0 or prepare_batch < 1 or storage not in ("cpu", "device"):
            raise ValueError("invalid frozen-prefix cache settings")
        self.model = model
        self.instance = model._prepared_frozen(c, source)
        cv, fv = self.instance[:2]
        self.identity = (
            model.fast_checkpoint_identity,
            tensor_digest((("medium", cv), ("source", fv), ("x", model.x_m), ("z", model.z_m))),
            str(cv.dtype),
        )
        self.source = fv
        self.max_bytes = int(max_bytes)
        self.prepare_batch = int(prepare_batch)
        self.storage = model.device if storage == "device" else torch.device("cpu")
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = self.unique_queries = 0
        self.sampled_queries = 0
        self.initial = None
        self.phase = {k: 0. for k in ("frozen_features", "initial_tangents", "transfers")}

    def _sync(self):
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)

    @staticmethod
    def _size(pack):
        return sum(value.numel() * value.element_size() for value in pack)

    def _move(self, pack, device):
        if all(value.device == device for value in pack):
            return pack
        from time import perf_counter
        self._sync()
        started = perf_counter()
        value = FrozenQuery(*(entry.to(device) for entry in pack))
        self._sync()
        self.phase["transfers"] += perf_counter() - started
        return value

    def _remember(self, key, pack):
        size = self._size(pack)
        if size > self.max_bytes:
            return
        while self.bytes + size > self.max_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes -= self._size(old)
            self.evictions += 1
        self.entries[key] = pack
        self.bytes += size

    def packs(self, times):
        from time import perf_counter
        started = perf_counter()
        times = self.model._times(times)
        bits = times.detach().cpu().view(torch.int32).tolist()
        self.phase["transfers"] += perf_counter() - started
        keys = [(self.identity, v) for v in bits]
        unique = dict(zip(keys, times.unbind(0)))
        self.unique_queries += len(unique)
        self.sampled_queries += len(keys)
        result, missing = {}, {}
        for key, t in unique.items():
            if key in self.entries:
                self.hits += 1
                self.entries.move_to_end(key)
                result[key] = self._move(self.entries[key], self.model.device)
            else:
                self.misses += 1
                missing[key] = t
        absent = list(missing)
        for start in range(0, len(absent), self.prepare_batch):
            batch_keys = absent[start:start + self.prepare_batch]
            batch_times = torch.stack([missing[k] for k in batch_keys])
            self._sync()
            started = perf_counter()
            with torch.no_grad(), self.model.scopes():
                packs = self.model._encode_chunks(self.instance, batch_times)
            self._sync()
            self.phase["frozen_features"] += perf_counter() - started
            for i, key in enumerate(batch_keys):
                # Clone slice storage to prevent tiny entries from retaining a whole batch.
                pack = FrozenQuery(*(p[i:i+1].detach().clone() for p in packs))
                result[key] = pack
                if self._size(pack) <= self.max_bytes:
                    self._remember(key, self._move(pack, self.storage))
        return FrozenQuery(*(torch.cat(components, 0)
                             for components in zip(*(result[k] for k in keys))))

    def initial_pack(self):
        from time import perf_counter
        if self.initial is None:
            self._sync()
            started = perf_counter()
            self.initial = self.model._encode_jvp(self.instance, [0.])
            self._sync()
            self.phase["initial_tangents"] += perf_counter() - started
        return self.initial

    def receipt(self):
        initial_bytes = 0 if self.initial is None else sum(self._size(p) for p in self.initial)
        return {"max_bytes": self.max_bytes, "cached_bytes": self.bytes,
                "entries": len(self.entries), "initial_jvp_bytes": initial_bytes,
                "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "unique_request_count": self.unique_queries,
                "total_query_request_count": self.sampled_queries,
                "duplicates_reused_within_request": self.sampled_queries - self.unique_queries,
                "storage": str(self.storage), "phases_s": dict(self.phase)}
