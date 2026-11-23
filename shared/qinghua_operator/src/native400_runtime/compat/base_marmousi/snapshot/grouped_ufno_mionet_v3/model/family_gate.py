#!/usr/bin/env python3
"""Per-record family gate for the phase channel and the scattering head (20260926).

Why this exists
---------------
`uniform` records are near-homogeneous: the medium carries almost no reflectivity
contrast, so the enumerated phase channel and the Born scattering head have
nothing physical to add, while their carriers still inject structure that the
measured runs turned into degradation (v1 +6.8%, v2 +9.3% on uniform; the
mechanism was the phase carrier manufacturing spurious oscillation in the
near-zero-energy late window).  Uniform must nevertheless stay IN the pool, in
the loss, and in the round-robin, because the trunk learns from it through the
shared weights.  So the gate is applied to the BRANCHES, per record, inside the
forward -- never by dropping samples on the data side.

Exactness
---------
The gate is a float32 multiplier applied with `torch.where`, so a gated record's
branch contribution is bitwise 0.0 (not "small"): `torch.where(mask, term, 0)`
returns the literal zero tensor for the gated rows.  At the fork the branch
outputs are zero anyway (zero-initialised output layers), so both the open and
the closed path reproduce the parent bitwise -- see
`identity_check_family_gate.py`.

The gate is a function of the record's family label only.  It is bound by the
caller through `PreparedV3State.family_gate`, exactly like `coarse_field`: the
model never reads a file, and a model built without the gate ignores it.
"""
from __future__ import annotations

import torch

# Family names are the dataset's medium_type values (data/index.py).
KNOWN_FAMILIES = ("uniform", "layered", "marmousi")
# The gate is defined by EXCLUSION, not inclusion: a family not listed here is
# treated as gated (safe default -- an unknown label must not silently switch on
# a carrier whose failure mode is only measured for uniform).
DEFAULT_UNGATED = ("layered", "marmousi")


def resolve_ungated(raw, *, default=DEFAULT_UNGATED) -> tuple[str, ...]:
    """Normalise the `scatter_ungated_families` spec field."""
    if raw is None:
        return tuple(default)
    if isinstance(raw, str):
        raw = (raw,)
    values = tuple(str(v) for v in raw)
    unknown = sorted(set(values) - set(KNOWN_FAMILIES))
    if unknown:
        raise ValueError(
            f"scatter_ungated_families must be a subset of {KNOWN_FAMILIES}, got {unknown}")
    return values


def family_gate(
    families: "list[str] | tuple[str, ...]",
    *,
    ungated=DEFAULT_UNGATED,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """[record] float32 1.0 for ungated families, exactly 0.0 for gated ones.

    `families[i]` is the medium_type of record i; the caller builds it from the
    same draw metadata that already supplies `record_to_medium`.
    """
    ungated = set(resolve_ungated(ungated))
    values = [1.0 if str(f) in ungated else 0.0 for f in families]
    return torch.tensor(values, dtype=dtype, device=device)


def gate_view(term: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    """Row-wise gate: `term` is [record, ...], `gate` is [record] or None (no-op)."""
    if gate is None:
        return term
    if gate.ndim != 1:
        raise ValueError(f"family gate must be [record], got {tuple(gate.shape)}")
    if gate.shape[0] != term.shape[0]:
        raise ValueError(
            f"family gate has {gate.shape[0]} entries but the term has {term.shape[0]} records")
    view = gate.reshape((-1,) + (1,) * (term.ndim - 1))
    return torch.where(view != 0, term, torch.zeros_like(term))


def gated_family_labels(
    families: "list[str] | tuple[str, ...]",
    *,
    ungated=DEFAULT_UNGATED,
) -> list[str]:
    """Per-draw audit trail: the label and switch state of every record in the batch."""
    ungated = set(resolve_ungated(ungated))
    return [f"{f}:{'open' if str(f) in ungated else 'closed'}" for f in families]


__all__ = [
    "KNOWN_FAMILIES",
    "DEFAULT_UNGATED",
    "resolve_ungated",
    "family_gate",
    "gate_view",
    "gated_family_labels",
]
