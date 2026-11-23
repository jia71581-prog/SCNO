"""Four compensated rank-four affines and a live bounded propagation gain."""

from copy import deepcopy
from types import ModuleType

import torch
from torch import nn

from finetune_ul_transfer import detail_heads
from lora_v2 import CompensatedRank4


ADAPTER_SCHEMA = "SCNO_CURRENT_E200_U96_INSTANCE_ADAPTER_V1"
POSITION_NAMES = ("detail_0", "detail_1", "detail_2", "u96_output")
FACTOR_NAMES = ("U", "V", "U0", "V0")


def clone_current(current):
    """Clone tensors and module hierarchy while retaining imported runtime helpers."""
    memo = {}
    visited = set()

    def keep_runtime_modules(value):
        key = id(value)
        if key in visited:
            return
        visited.add(key)
        if isinstance(value, ModuleType):
            memo[key] = value
        elif isinstance(value, dict):
            for child in value.values():
                keep_runtime_modules(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                keep_runtime_modules(child)

    for layer in current.modules():
        keep_runtime_modules(vars(layer))
    return deepcopy(current, memo)


class CurrentCompensatedLoRA(CompensatedRank4):
    """Retain the six-factor E200 cache interface and support rectangular U96."""

    def __init__(self, linear, anchors=None):
        # CompensatedRank4's constructor is limited to the old 32-to-32 topology.
        nn.Module.__init__(self)
        if not isinstance(linear, nn.Linear) or linear.bias is None or min(linear.weight.shape) < 4:
            raise ValueError("Expected a biased affine with at least four singular directions")
        self.base = deepcopy(linear).requires_grad_(False)
        if anchors is None:
            left, values, right_h = torch.linalg.svd(linear.weight.detach().double().cpu(), full_matrices=False)
            roots = values[:4].sqrt()
            anchors = (left[:, :4] * roots, right_h[:4].T * roots)
        u0, v0 = (factor.detach().to(linear.weight).clone() for factor in anchors)
        if u0.shape != (linear.out_features, 4) or v0.shape != (linear.in_features, 4):
            raise ValueError("Saved LoRA anchor shape differs from the affine")
        self.register_buffer("U0", u0)
        self.register_buffer("V0", v0)
        self.U = nn.Parameter(u0.clone())
        self.V = nn.Parameter(v0.clone())
        self.eval()

    def delta_weight(self):
        return self.U @ self.V.T - self.U0 @ self.V0.T

    def effective_weight(self):
        return self.base.weight + self.delta_weight()

    def delta_output(self, hidden, frozen_anchor):
        return (hidden @ self.V) @ self.U.T - frozen_anchor


def current_layers(current):
    return (*(head[2] for head in detail_heads(current.backbone)), current.readout.spatial[4])


def insert_current_adapters(current, saved_layers=None):
    current.requires_grad_(False).eval()
    current.backbone.compiled_readout = None
    locations = [(head, 2) for head in detail_heads(current.backbone)] + [(current.readout.spatial, 4)]
    if [tuple(seq[index].weight.shape) for seq, index in locations] != [(32, 32)] * 3 + [(111, 96)]:
        raise ValueError("Unexpected E200/U96 affine layout")
    for name, (seq, index) in zip(POSITION_NAMES, locations):
        anchors = None if saved_layers is None else (saved_layers[name]["U0"], saved_layers[name]["V0"])
        seq[index] = CurrentCompensatedLoRA(seq[index], anchors)
    current.propagation_gain = nn.Parameter(current_layers(current)[0].U.new_zeros(()))
    expected = {id(p) for layer in current_layers(current) for p in layer.lora_parameters()}
    expected.add(id(current.propagation_gain))
    trainable = {id(p): p.numel() for p in current.parameters() if p.requires_grad}
    if set(trainable) != expected or sum(trainable.values()) != 1597:
        raise AssertionError("Expected exactly eight factors and one scalar gain")
    return current


def current_adapter_state(current):
    return {
        "layers": {
            name: {key: getattr(layer, key).detach().cpu().clone() for key in FACTOR_NAMES}
            for name, layer in zip(POSITION_NAMES, current_layers(current))
        },
        "gain": current.propagation_gain.detach().cpu().clone(),
    }


@torch.no_grad()
def restore_current_adapter(current, payload, identity, instance, *, expected_updates=50):
    if (payload["schema"] != ADAPTER_SCHEMA or payload["identity"] != identity
            or payload["instance"] != instance or payload["updates"] != expected_updates
            or payload["selection"] != "final"):
        raise ValueError("Adapter does not bind this model, instance and final-step protocol")
    state = payload["adapter_state"]
    if set(state["layers"]) != set(POSITION_NAMES):
        raise ValueError("Missing or extra rank-four affine state")
    if not hasattr(current, "propagation_gain"):
        insert_current_adapters(current, state["layers"])
    for name, layer in zip(POSITION_NAMES, current_layers(current)):
        saved = state["layers"][name]
        if set(saved) != set(FACTOR_NAMES):
            raise ValueError("Incomplete compensated factors")
        for key in FACTOR_NAMES:
            target, value = getattr(layer, key), saved[key]
            if target.shape != value.shape or target.dtype != value.dtype or not torch.isfinite(value).all():
                raise ValueError(f"Incompatible saved factor {name}.{key}")
            if key in ("U0", "V0") and not torch.equal(target.cpu(), value.cpu()):
                raise ValueError(f"The saved compensation anchor differs at {name}.{key}")
            if key in ("U", "V") and (target.requires_grad or not torch.equal(target, value.to(target))):
                target.copy_(value)
    gain = state["gain"]
    if gain.ndim != 0 or not 0. <= float(gain) <= 1.:
        raise ValueError("The saved propagation gain is outside [0,1]")
    current.propagation_gain.copy_(gain)
    return current
