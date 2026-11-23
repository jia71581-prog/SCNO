"""Multi-arrival phase channel features (P2, 20260925).

Per arrival a (K primary reflections followed by K source-side free-surface
multiples) with precomputed travel time T_a(z, x) (inf where the arrival does
not exist) and reflection-coefficient proxy r_a:

    tau_a  = t - t0 - T_a
    phi_a  = f0 * tau_a                      (cycles)
    [sin(2 pi m phi_a), cos(2 pi m phi_a)]   m = 1..M
    causal = sigmoid(tau_a / tau_w)           tau_w = 0.01 * domain_t_s (the direct branch's window)
    amp    = sqrt(tau_w / max(T_a, tau_w))    1/sqrt(T) spreading proxy, normalised to (0, 1]
    rc     = r_a * causal

Absent arrivals contribute exactly zero on every feature, so the channel is
inert wherever the interface extraction found nothing.
"""
from __future__ import annotations

import math

import torch


def phase_feature_count(arrivals: int, harmonics: int) -> int:
    return int(arrivals) * (2 * int(harmonics) + 3)


def multi_arrival_phase_features(
    travel_s: torch.Tensor,      # [record, arrivals, points]
    reflection: torch.Tensor,    # [record, arrivals]
    time_s: torch.Tensor,        # [record, points]
    t0_s: torch.Tensor,          # [record]
    f0_hz: torch.Tensor,         # [record]
    *,
    harmonics: int,
    domain_t_s: float,
) -> torch.Tensor:
    if travel_s.ndim != 3 or reflection.shape != travel_s.shape[:2] or time_s.shape != (travel_s.shape[0], travel_s.shape[2]):
        raise ValueError("phase feature inputs have inconsistent shapes")
    tau_w = max(domain_t_s * 0.01, 1.0e-6)
    valid = torch.isfinite(travel_s)
    T = torch.where(valid, travel_s, torch.zeros_like(travel_s))
    tau = time_s[:, None, :] - t0_s[:, None, None] - T
    phi = 2.0 * math.pi * f0_hz[:, None, None] * tau
    parts = []
    for m in range(1, int(harmonics) + 1):
        parts.append(torch.sin(m * phi))
        parts.append(torch.cos(m * phi))
    causal = torch.sigmoid(tau / tau_w)
    parts.append(causal)
    parts.append(torch.sqrt(tau_w / torch.clamp_min(T, tau_w)))
    parts.append(reflection[:, :, None] * causal)
    stacked = torch.stack(parts, dim=-1)                                  # [record, arrivals, points, F]
    stacked = torch.where(valid[..., None], stacked, torch.zeros_like(stacked))
    out = stacked.permute(0, 2, 1, 3).reshape(travel_s.shape[0], travel_s.shape[2], -1)
    if not torch.isfinite(out).all():
        raise RuntimeError("phase features produced non-finite values")
    return out


__all__ = ["multi_arrival_phase_features", "phase_feature_count"]
