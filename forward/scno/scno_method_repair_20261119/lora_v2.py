"""Checkpointable compensated rank-four factors for SCNO's second SiLU affine."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


ADAPTER_PREFIX = "refiner.spectral_detail.heads."
ADAPTER_KEYS = ("base.weight", "base.bias", "U", "V", "scaling")
PLAIN_AFFINE_KEYS = ("weight", "bias")


class CompensatedRank4(nn.Module):
    def __init__(self, linear, u, v, scaling, frozen_weight=None):
        super().__init__()
        if not isinstance(linear, nn.Linear) or tuple(linear.weight.shape) != (32, 32) or linear.bias is None:
            raise ValueError("expected an affine 32-to-32 second hidden layer")
        if tuple(u.shape) != (32, 4) or tuple(v.shape) != (32, 4):
            raise ValueError("expected two rank-four factors of shape (32, 4)")
        if not math.isfinite(float(scaling)) or float(scaling) <= 0:
            raise ValueError("low-rank scaling must be finite and positive")
        self.base = copy.deepcopy(linear).requires_grad_(False)
        if frozen_weight is not None:
            with torch.no_grad():
                self.base.weight.copy_(frozen_weight)
        self.U = nn.Parameter(u.detach().to(linear.weight).clone(), requires_grad=False)
        self.V = nn.Parameter(v.detach().to(linear.weight).clone(), requires_grad=False)
        self.register_buffer("scaling", linear.weight.new_tensor(float(scaling)))

    @classmethod
    @torch.no_grad()
    def insert(cls, linear, scaling=1.):
        scaling = float(scaling)
        if not math.isfinite(scaling) or scaling <= 0:
            raise ValueError("low-rank scaling must be finite and positive")
        weight = linear.weight.detach().double().cpu()
        left, values, right_h = torch.linalg.svd(weight, full_matrices=True)
        singular_root = (values[:4] / scaling).sqrt()
        u = (left[:, :4] * singular_root).to(dtype=linear.weight.dtype)
        v = (right_h[:4].T * singular_root).to(dtype=linear.weight.dtype)
        frozen = weight - scaling * (u.double() @ v.double().T)
        return cls(linear, u, v, scaling, frozen)

    def lora_parameters(self):
        return self.U, self.V

    def effective_weight(self):
        return self.base.weight + self.scaling * (self.U @ self.V.T)

    def forward(self, value):
        return F.linear(value, self.effective_weight(), self.base.bias)


def adapter_prefix(index):
    return f"{ADAPTER_PREFIX}{index}.2."


def second_affine_tensors(layer):
    """Return the currently effective layer's state, including a legacy unit scale."""
    state = layer.state_dict()
    if set(state) == set(ADAPTER_KEYS) - {"scaling"}:
        state["scaling"] = state["U"].new_tensor(1.)
    if set(state) not in (set(ADAPTER_KEYS), set(PLAIN_AFFINE_KEYS)):
        raise ValueError("unexpected second hidden affine state")
    return state


def prepared_second_affine(layer, state):
    """Validate a candidate and prepare its topology without rerunning SVD."""
    if not state:
        return layer, None  # Old affine-only candidates kept the original second layer.
    base = layer if isinstance(layer, nn.Linear) else layer.base
    expected = {"weight": base.weight, "bias": base.bias,
                "base.weight": base.weight, "base.bias": base.bias,
                "U": base.weight.new_empty((32, 4)), "V": base.weight.new_empty((32, 4)),
                "scaling": base.weight.new_empty(())}
    valid_sets = (set(ADAPTER_KEYS), set(PLAIN_AFFINE_KEYS), {"U", "V"})
    if (set(state) not in valid_sets
            or any(not torch.is_tensor(value) or value.shape != expected[key].shape
                   or value.dtype != expected[key].dtype or not torch.isfinite(value).all()
                   for key, value in state.items())):
        raise ValueError("incomplete or incompatible saved second hidden affine")
    if set(state) == {"U", "V"}:
        if not isinstance(layer, nn.Linear):
            raise ValueError("a legacy candidate requires its original unadapted second affine")
        state = {**state, "base.weight": base.weight.detach(), "base.bias": base.bias.detach(),
                 "scaling": base.weight.new_tensor(1.)}
    if "scaling" in state:
        if float(state["scaling"]) <= 0.:
            raise ValueError("low-rank scaling must be positive")
        if not isinstance(layer, CompensatedRank4):
            layer = CompensatedRank4(base, state["U"], state["V"], state["scaling"], state["base.weight"])
    elif not isinstance(layer, nn.Linear):
        layer = copy.deepcopy(base).requires_grad_(False)
    return layer, state


def place_saved_adapters(model, state, *, legacy=False):
    """Recreate adapter topology from parameters already on disk, without SVD."""
    replacements = []
    for index, head in enumerate(model.refiner.spectral_detail.heads):
        prefix = adapter_prefix(index)
        tensors = {key: state[prefix + key] for key in ADAPTER_KEYS if prefix + key in state}
        if not tensors:
            continue
        keys = ("U", "V") if legacy else ADAPTER_KEYS
        if set(tensors) != set(keys):
            raise ValueError(f"incomplete saved low-rank module at {prefix}")
        module, _ = prepared_second_affine(head[2], tensors)
        replacements.append((head, module))
    for head, module in replacements:
        head[2] = module
