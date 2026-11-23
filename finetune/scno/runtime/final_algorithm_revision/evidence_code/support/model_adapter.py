"""Portable joint182 refiner adaptation, with immutable base-field caching.

No reference-pressure array is opened here. The runner alone supplies early
labels and opens later reference frames for evaluation. This checkpoint has no
final DG block, so these interfaces support the physics/AIS ablation first.
"""
from __future__ import annotations
from scno_release_paths import local_asset as _scno_local_asset

from contextlib import contextmanager
from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
import types

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.attention import sdpa_kernel, SDPBackend

ROOT = Path(_scno_local_asset('project'))
BUNDLE = ROOT / "paper_assets_main_fresh_20261013_marmousi_20261019"
CHECKPOINT = BUNDLE / "models/joint_epoch182.pt"
MODEL_CONFIG = BUNDLE / "original/runs/r28_cf_joint934_gpu01_continue200_20261011/CONFIG.json"
INITIALIZER = BUNDLE / "models/initializers/layered.pt"
DATA_PROTOCOL = BUNDLE / "original/data/joint_uniform500_layered434_raw_20261011/CONFIG.json"
CASE = BUNDLE / "paper/revisions/aurora_revision_20261021/evidence/evaluation/frozen_cases/uniform4000"
_IMPORT_LOCK = threading.RLock()

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def float32_time_keys(times):
    values = torch.as_tensor(times).detach().to(device="cpu", dtype=torch.float32)
    values = values.reshape(-1).contiguous().numpy()
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Nonempty physical query times in [0, 1] are required")
    return tuple(int(x) for x in values.view(np.uint32))


@contextmanager
def _local_protocol_path():
    """Fix only a legacy metadata path in memory; old source remains untouched."""
    from native400_runtime import r24_cf_preserved_operator as preserved
    with _IMPORT_LOCK:
        original = preserved._module
        target = (preserved.COMPAT / "r24/protocol400.py").resolve()

        def portable_module(name, path):
            path = Path(path).resolve()
            if path != target:
                return original(name, path)
            source = path.read_text()
            needle = "p=Path(META['source_R21'])/'protocol92.py'"
            if source.count(needle) != 1:
                raise ValueError("Legacy protocol path statement changed")
            replacement = "p=Path(" + repr(str(preserved.COMPAT / "r21")) + ")/'protocol92.py'"
            module = types.ModuleType(name)
            module.__file__ = str(path)
            sys.modules[name] = module
            exec(compile(source.replace(needle, replacement), str(path), "exec"), module.__dict__)
            return module

        preserved._module = portable_module
        try:
            yield
        finally:
            preserved._module = original


def load_pretrained(device="cpu", *, dense_time_block=8, refiner_time_block=8):
    """Strictly restore joint182; return tensors/paths without reading truth."""
    from native400_runtime.r25_cf_full400_operator import build_r25_cf_full400_operator
    config = json.loads(MODEL_CONFIG.read_text())
    config["parent_checkpoint"] = str(INITIALIZER)
    config.update(dense_time_block=int(dense_time_block),
                  refiner_time_block=int(refiner_time_block), checkpoint_queries=False)
    if min(dense_time_block, refiner_time_block) < 1:
        raise ValueError("Positive query blocks are required")
    with _local_protocol_path():
        model = build_r25_cf_full400_operator(config, device="cpu", kind="joint")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    if (saved.get("schema") != "R25_CF_JOINT934_ALL400_CONTINUATION_V1"
            or saved.get("model_inputs") != ["c", "f"]
            or saved.get("predictor_reads_true_pressure") is not False):
        raise ValueError("Expected the c/f-only joint pretrained checkpoint")
    model.base.load_state_dict(saved["base_model_state"], strict=True)
    model.refiner.load_state_dict(saved["refiner_state"], strict=True)
    protocol = json.loads(DATA_PROTOCOL.read_text())
    if tuple(model.full_native_frames) != tuple(protocol["full_frames"]):
        raise ValueError("Registered full400 query protocol mismatch")
    if sum(p.numel() for p in model.parameters()) != 14577239:
        raise ValueError("Unexpected joint182 architecture")
    source_metadata = json.loads((CASE / "INPUT.json").read_text())
    source = torch.tensor(source_metadata["source_model_fp32"], dtype=torch.float32)
    velocity = torch.from_numpy(np.array(np.load(CASE / "velocity201.npy", mmap_mode="r"),
                                        copy=True)).float()
    if velocity.shape != (201, 201) or not torch.all(velocity == 4000):
        raise ValueError("The prescribed uniform4000 case is required")
    model.eval().requires_grad_(False).to(device)
    checkpoint_sha = sha256(CHECKPOINT)
    assets = dict(
        velocity=velocity.to(device), source=source.to(device),
        full_times=model.full_times_s.detach().clone(),
        early_times=model.full_times_s[:8].detach().clone(),
        full_native_frames=tuple(model.full_native_frames),
        reference_path=CASE / "reference400_pa.npy",
        case_path=CASE, checkpoint_path=CHECKPOINT,
        checkpoint_sha256=checkpoint_sha,
        initializer_path=INITIALIZER, config_path=MODEL_CONFIG,
        pressure_scale_pa=float(model.normalizer.metadata.pressure_scale_pa),
        pretrained_cursor={key: saved["cursor"].get(key)
                           for key in ("epoch", "position", "steps", "exposures")},
        pretrained_parameters=14577239, checkpoint_has_final_dg=False,
        reference_pressure_opened=False,
    )
    return model, assets


class _Rank8Weight(nn.Module):
    def __init__(self, weight, generator):
        super().__init__()
        self.rank = 8
        a = torch.empty(8, weight[0].numel(), dtype=weight.dtype, device="cpu")
        a.uniform_(-1 / np.sqrt(weight[0].numel()),
                   1 / np.sqrt(weight[0].numel()), generator=generator)
        self.A = nn.Parameter(a.to(device=weight.device))
        self.B = nn.Parameter(weight.new_zeros(weight.shape[0], 8))

    def forward(self, weight):
        # alpha=rank=8, so alpha/rank is one.
        return weight + (self.B @ self.A).reshape_as(weight)


def clone_refiner_model(original, *, seed=20261105, install_adapters=True):
    """Clone the refiner; share only the immutable base and fixed buffers."""
    if any(p.requires_grad for p in original.base.parameters()):
        raise ValueError("A shared base must already be frozen")
    model = copy.copy(original)
    model._modules = dict(original._modules)
    model._modules["refiner"] = copy.deepcopy(original.refiner)
    model.eval().requires_grad_(False)
    sites = [(name, module) for name, module in model.named_modules()
             if name.startswith("refiner.") and isinstance(module, (nn.Linear, nn.Conv2d))
             and not module.weight.is_complex()
             and (not isinstance(module, nn.Conv2d) or module.groups == 1)]
    if len(sites) != 42:
        raise ValueError("Expected 42 real refiner Linear/Conv weight sites")
    parameters = []
    if install_adapters:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        for _, module in sites:
            adaptation = _Rank8Weight(module.weight, generator)
            parametrize.register_parametrization(module, "weight", adaptation)
            parameters.extend((adaptation.A, adaptation.B))
        if sum(p.numel() for p in parameters) != 121648:
            raise ValueError("Expected 121648 refiner LoRA parameters")
    allowed = {id(p) for p in parameters}
    if any(p.requires_grad != (id(p) in allowed) for p in model.parameters()):
        raise ValueError("An original weight became trainable")
    receipt = dict(targets=[name for name, _ in sites], rank=8, alpha=8,
                   trainable_parameters=sum(p.numel() for p in parameters),
                   seed=int(seed), base_frozen=True, base_shared=True,
                   checkpoint_has_final_dg=False)
    return model, parameters, receipt


class FrozenBaseCache:
    """Cache only frozen normalized base predictions, keyed by exact F32 bits."""
    def __init__(self, model, velocity, source, *, cache_device="cpu", block=8):
        if int(block) < 1:
            raise ValueError("Positive base-cache block required")
        if any(p.requires_grad for p in model.base.parameters()):
            raise ValueError("The base must remain frozen before caching")
        self.model = model
        self.device = model.device
        self.cache_device = torch.device(cache_device)
        self.block = int(block)
        self.c, self.f = model._inputs(velocity, source)
        with torch.no_grad(), model.scopes():
            self.prepared, self.grid, _ = model._prepare_cf(self.c, self.f)
        z, x = torch.meshgrid(model.z_m, model.x_m, indexing="ij")
        coordinates = torch.stack((2*x/2000-1, 2*z/2000-1,
                                   (x-self.f[0, 0])/2000, (z-self.f[0, 1])/2000))[None]
        self.static = torch.cat((self.c/3000-1, self.c.new_zeros(1, 8, 201, 201),
                                 model._source_map(self.f, 201, 10.), coordinates), 1).detach()
        self._values = {}
        self._initial = None
        self.generated_queries = 0
        self._base_versions = tuple((id(t), t._version) for t in model.base.parameters())

    def _check_base(self):
        now = tuple((id(t), t._version) for t in self.model.base.parameters())
        if now != self._base_versions:
            raise RuntimeError("Frozen base changed while its output cache was active")

    def _base_values(self, times):
        m = self.model
        base101 = m.base.dense_normalized(self.prepared, times[None], dense_grid=self.grid,
                                         time_block=m.dense_time_block, apply_correction=True)
        pa = m.normalizer.decode_pressure(base101, self.f[:, 4])
        pa201 = m._bicubic201(pa[0, :, None])
        return m.normalizer.encode_pressure(pa201[:, 0][None], self.f[:, 4])[0, :, None]

    def prefill(self, times):
        self._check_base()
        values = torch.as_tensor(times, device=self.device, dtype=torch.float32).reshape(-1)
        keys = float32_time_keys(values)
        missing = {}
        for key, value in zip(keys, values):
            if key not in self._values:
                missing.setdefault(key, value.detach())
        pairs = list(missing.items())
        for start in range(0, len(pairs), self.block):
            current = pairs[start:start+self.block]
            t = torch.stack([value for _, value in current])
            with torch.no_grad(), self.model.scopes():
                base = self._base_values(t)
            if not torch.isfinite(base).all():
                raise FloatingPointError("Nonfinite frozen base")
            for (key, _), value in zip(current, base):
                self._values[key] = value.detach().to(self.cache_device).clone()
            self.generated_queries += len(current)
        return self

    def get(self, times):
        self.prefill(times)
        return torch.stack([self._values[k] for k in float32_time_keys(times)]).to(self.device)

    def initial_base(self):
        self._check_base()
        if self._initial is None:
            t0 = torch.zeros(1, device=self.device, dtype=torch.float32)
            with torch.enable_grad(), self.model.scopes(), sdpa_kernel(SDPBackend.MATH):
                value, derivative = torch.autograd.functional.jvp(
                    self._base_values, t0, torch.ones_like(t0), create_graph=False)
            if not torch.isfinite(value).all() or not torch.isfinite(derivative).all():
                raise FloatingPointError("Nonfinite base initial-time JVP")
            self._initial = (value.detach(), derivative.detach())
        return self._initial


class UniformAdapter:
    """One independent refiner session; no future truth is an accepted input."""
    def __init__(self, model, cache, *, seed=20261105, install_adapters=True):
        self.operator, self._parameters, self.receipt = clone_refiner_model(
            model, seed=seed, install_adapters=install_adapters)
        self.cache = cache
        self.device = self.operator.device
        if self.device != cache.device:
            raise ValueError("Model and frozen base cache must use the same device")
        self.source = cache.f
        self.static = cache.static

    def parameters(self):
        return iter(self._parameters)

    def training_scope(self):
        return self.operator.scopes()

    def query(self, times, *, grad=True, initial_bridge=False):
        t = torch.as_tensor(times, device=self.device, dtype=torch.float32).reshape(-1)
        float32_time_keys(t)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            if initial_bridge or t.requires_grad:
                if not bool(torch.all(t.detach() == 0)):
                    raise ValueError("Cached time differentiation is supported only at t=0")
                base0, base_dt = self.cache.initial_base()
                base = base0 + t[:, None, None, None] * base_dt
            else:
                base = self.cache.get(t)
            # Static LoRA features must not be cached between optimizer steps.
            features = self.operator.refiner.prepare(self.static)
            value = base + self.operator.refiner(features, base, t, self.source)
            return torch.cat((torch.zeros_like(value[:, :, :1]), value[:, :, 1:]), -2)[:, 0]

    def query_physical(self, times, *, grad=False):
        value = self.query(times, grad=grad)
        return self.operator.normalizer.decode_pressure(value[None], self.source[:, 4])[0]

    def initial_state(self, *, create_graph=True):
        """Return (u(0), du/dt(0)), retaining LoRA gradients when requested."""
        t0 = torch.zeros(1, device=self.device, dtype=torch.float32)
        with torch.enable_grad(), self.training_scope(), sdpa_kernel(SDPBackend.MATH):
            return torch.autograd.functional.jvp(
                lambda t: self.query(t, grad=True, initial_bridge=True),
                t0, torch.ones_like(t0), create_graph=create_graph)

    def adapter_state_dict(self):
        allowed = {id(parameter) for parameter in self._parameters}
        return {name: value.detach().cpu().clone()
                for name, value in self.operator.named_parameters() if id(value) in allowed}

    def load_adapter_state_dict(self, state):
        """Strict, atomic factor restore without replacing Parameters.

        Optimizer and RNG states are separate runner-owned checkpoint fields.
        Validation and device staging finish before any live factor is copied.
        """
        if not isinstance(state, Mapping):
            raise TypeError("Adapter state must be a mapping of factor names to tensors")
        allowed = {id(parameter) for parameter in self._parameters}
        expected = {name: parameter for name, parameter in self.operator.named_parameters()
                    if id(parameter) in allowed}
        if set(state) != set(expected):
            missing = sorted(set(expected)-set(state))
            unexpected = sorted(map(str, set(state)-set(expected)))
            raise ValueError(f"Adapter factor keys differ: missing={missing}, unexpected={unexpected}")
        staged = {}
        for name, parameter in expected.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
                raise TypeError(f"Dense tensor required for adapter factor {name}")
            if value.shape != parameter.shape:
                raise ValueError(f"Adapter factor shape differs: {name}")
            if value.dtype != parameter.dtype:
                raise ValueError(f"Adapter factor dtype differs: {name}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Adapter factor must be finite: {name}")
            staged[name] = value.detach().to(device=parameter.device).clone()
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(staged[name])
        return dict(factors_copied=len(expected),
                    adapter_parameters=sum(parameter.numel() for parameter in expected.values()),
                    parameter_identity_preserved=True)
