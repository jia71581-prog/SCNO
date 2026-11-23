from __future__ import annotations
import math
import torch

# --- Late-window auxiliary data terms (20260929, accuracy F/G arms) -------------------
# Motivation (MAX_temporal rank-2): layered late-coda amplitude collapse is the
# conditional-mean fixed point of the L2 objective under phase uncertainty; these two
# ADDITIVE data-side terms attack the fixed point without touching the record/frame
# blend, the RAS tables, or any physics/solver component.
#   late_amp : hinge on the per-frame Frobenius amplitude ratio a_k=||u_k||/max(||y_k||,floor)
#              L_amp = sum_{k in late} (1/pi_k) relu(a_floor - a_k)^2
#   late_ncc : per-frame spatial normalized cross-correlation
#              L_ncc = sum_{k in late} (1/pi_k) (1 - <u_k,y_k>/(||u_k||*||y_k|| + eps)),
#              eps = eps_frac*||y_k||  (regularizes the 1/||u|| gradient near collapse)
# late window: frame time t_k > late_frac * T with T the record's own axis end.
# pi_k is the SAME Hansen-Hurwitz inclusion probability the frame term uses, so both
# sums are unbiased for the finite-population late-window sums. Terms are gated by the
# draw's family and linearly warmed up over warmup_updates optimizer steps.
# Absent/disabled blocks are dropped at config normalization, so the loss value of
# every pre-existing config is BITWISE unchanged (asserted in test_late_window_loss.py).

LATE_COMMON_KEYS = {'enabled', 'weight', 'families', 'late_frac', 'warmup_updates'}
LATE_TERM_SPECIFIC = {'late_amp': {'a_floor'}, 'late_ncc': {'eps_frac'}}
_LATE_FAMILIES = ('uniform', 'layered', 'marmousi')


def normalize_late_window_config(raw_loss: dict) -> dict:
    """Validate loss.late_amp / loss.late_ncc blocks (called from train.resolved_config).

    Returns {} when both blocks are absent OR enabled=false, so pre-existing semantic
    digests and the loss value are unchanged; enabled blocks return fully-defaulted
    dicts that enter c['loss'] (and hence the semantic digest)."""
    out = {}
    for name, extra_keys in LATE_TERM_SPECIFIC.items():
        blk = raw_loss.get(name)
        if blk is None:
            continue
        blk = dict(blk)
        unknown = set(blk) - (LATE_COMMON_KEYS | extra_keys)
        if unknown:
            raise ValueError(f'unknown loss.{name} keys {sorted(unknown)}')
        if not bool(blk.get('enabled', False)):
            continue  # disabled block -> dropped: bitwise-identical loss and digest
        weight = float(blk.get('weight', 0.10))
        late_frac = float(blk.get('late_frac', 0.66))
        warmup = int(blk.get('warmup_updates', 200))
        fams = list(blk.get('families', ['layered']))
        if not (weight > 0.0):
            raise ValueError(f'loss.{name}.weight must be > 0 when enabled')
        if not 0.0 < late_frac < 1.0:
            raise ValueError(f'loss.{name}.late_frac must lie in (0,1)')
        if warmup < 0:
            raise ValueError(f'loss.{name}.warmup_updates must be >= 0')
        if not fams or len(set(fams)) != len(fams) or not set(fams) <= set(_LATE_FAMILIES):
            raise ValueError(f'loss.{name}.families must be a nonempty duplicate-free subset of {_LATE_FAMILIES}')
        norm = {'enabled': True, 'weight': weight, 'late_frac': late_frac,
                'warmup_updates': warmup, 'families': sorted(fams, key=_LATE_FAMILIES.index)}
        if name == 'late_amp':
            a_floor = float(blk.get('a_floor', 0.3))
            if not 0.0 < a_floor <= 1.0:
                raise ValueError('loss.late_amp.a_floor must lie in (0,1]')
            norm['a_floor'] = a_floor
        else:
            eps_frac = float(blk.get('eps_frac', 0.05))
            if not (eps_frac > 0.0):
                raise ValueError('loss.late_ncc.eps_frac must be > 0')
            norm['eps_frac'] = eps_frac
        out[name] = norm
    return out


def _late_warmup_weight(term_cfg: dict, completed: int) -> float:
    wu = int(term_cfg['warmup_updates'])
    scale = 1.0 if wu <= 0 else min(max(float(completed) / wu, 0.0), 1.0)
    return float(term_cfg['weight']) * scale


def late_window_terms(prediction, target, pi, data, draw, loss_cfg, completed,
                      record_energy, n_future_total):
    """Compute the gated late-window terms for one draw.

    Returns (extra, row): extra is a graph-connected scalar to ADD to the data total
    (None when nothing is active -> caller leaves the total object untouched, bitwise);
    row always carries the four metric keys so the metrics.jsonl schema is stable.
    loss_late_amp / loss_late_ncc record the RAW (unweighted) HH-estimated sums; the
    *_weight_effective fields expose the warmup ramp for auditing."""
    row = {'loss_late_amp': 0.0, 'loss_late_ncc': 0.0,
           'late_amp_weight_effective': 0.0, 'late_ncc_weight_effective': 0.0}
    amp_cfg = loss_cfg.get('late_amp')
    ncc_cfg = loss_cfg.get('late_ncc')
    if amp_cfg is None and ncc_cfg is None:
        return None, row
    family = str(draw.get('family'))
    K = int(prediction.shape[1])
    t_axis = torch.as_tensor(data['time_axis']).detach().cpu()
    T = float(t_axis[-1])
    t_k = t_axis[torch.as_tensor(draw['frames'], dtype=torch.long)]
    u = prediction[0].reshape(K, -1)
    y = target[0].reshape(K, -1)
    u_norm = torch.linalg.vector_norm(u, dim=1)
    y_norm = torch.linalg.vector_norm(y, dim=1)
    extra = None
    if amp_cfg is not None and family in amp_cfg['families']:
        late = (t_k > amp_cfg['late_frac'] * T).to(prediction.device)
        if bool(late.any()):
            # Same energy floor convention as the frame term (frame_floor_rel * E / n),
            # taken as a NORM floor via sqrt.
            floor = math.sqrt(float(loss_cfg['frame_floor_rel']) * float(record_energy) / float(n_future_total))
            a_k = u_norm / torch.clamp(y_norm, min=floor)
            hinge = torch.relu(amp_cfg['a_floor'] - a_k).square()
            L_amp = (hinge / pi)[late].sum()
            w = _late_warmup_weight(amp_cfg, completed)
            row['loss_late_amp'] = float(L_amp.detach())
            row['late_amp_weight_effective'] = w
            if w > 0.0:
                extra = w * L_amp if extra is None else extra + w * L_amp
    if ncc_cfg is not None and family in ncc_cfg['families']:
        late = (t_k > ncc_cfg['late_frac'] * T).to(prediction.device)
        if bool(late.any()):
            dot = (u * y).sum(dim=1)
            # eps = eps_frac * ||y_k||: as ||u||->0 the gradient is capped at ~y/(eps_frac*||y||)
            # by construction (no explicit clip); the 1e-30 clamp only guards ||y||==0 exactly.
            denom = torch.clamp(u_norm * y_norm + ncc_cfg['eps_frac'] * y_norm, min=1e-30)
            term = 1.0 - dot / denom
            L_ncc = (term / pi)[late].sum()
            w = _late_warmup_weight(ncc_cfg, completed)
            row['loss_late_ncc'] = float(L_ncc.detach())
            row['late_ncc_weight_effective'] = w
            if w > 0.0:
                extra = w * L_ncc if extra is None else extra + w * L_ncc
    return extra, row


def frame_beta(loss_cfg: dict, completed: int) -> float:
    if loss_cfg.get("kind") != "hybrid_record_frame_energy":
        return 0.0
    target=float(loss_cfg["beta_target"])
    ramp=max(int(loss_cfg["beta_ramp_updates"]),1)
    return target * min(max(float(completed)/ramp,0.0),1.0)

def eligible_future_count(data: dict, draw: dict) -> int:
    n_future=int(len(data["time_axis"]) - int(data["future_start"]))
    if n_future <= 0:
        raise ValueError("no future frames")
    h=draw.get("causal_horizon_s")
    if h is None or bool(draw.get("causal_fallback",False)):
        return n_future
    axis=torch.as_tensor(data["time_axis"]).detach().cpu()
    start=int(data["future_start"])
    n=int((axis[start:] <= float(h)+1e-12).sum().item())
    if n <= 0:
        # Must agree with sampler fallback semantics.
        return n_future
    return n

def hybrid_data_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    record_loss: torch.Tensor,
    record_energy: torch.Tensor,
    data: dict,
    draw: dict,
    loss_cfg: dict,
    completed: int,
):
    """Blend legacy record-normalized loss with unbiased per-frame relative RMS.

    The frame estimator is Hansen-Hurwitz for the sampler's with-replacement
    draw. draw['inclusion_probabilities'][k] stores K*p_k, including both
    stratum allocation q_s and residual-adaptive within-stratum p_frame.
    Therefore sum(value_k / inclusion_probability_k) is unbiased for the
    finite-population sum over eligible frames.
    """
    if loss_cfg.get("kind") != "hybrid_record_frame_energy":
        return record_loss, {
            "loss_record": float(record_loss.detach()),
            "loss_frame_balanced": None,
            "frame_beta": 0.0,
        }
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[0] != 1:
        raise ValueError("prediction/target must be [1,K,Z,X]")
    K=int(prediction.shape[1])
    pi=torch.as_tensor(draw.get("inclusion_probabilities"),device=prediction.device,dtype=prediction.dtype)
    if pi.shape != (K,) or not bool(torch.isfinite(pi).all()) or bool((pi<=0).any()):
        raise ValueError("invalid inclusion probabilities for frame-balanced loss")
    err=(prediction-target)
    e_k=err.square().sum(dim=(2,3))[0]
    t_k=target.square().sum(dim=(2,3))[0]
    n_future_total=int(len(data["time_axis"]) - int(data["future_start"]))
    n_eligible=eligible_future_count(data,draw)
    E=record_energy.to(device=prediction.device,dtype=prediction.dtype)
    floor=float(loss_cfg["frame_floor_rel"]) * E / float(n_future_total)
    rel2=e_k/torch.clamp(t_k,min=floor)
    total_est=(rel2/pi).sum()
    frame_loss=torch.sqrt(torch.clamp(total_est/float(n_eligible),min=0.0))
    beta=frame_beta(loss_cfg,completed)
    total=(1.0-beta)*record_loss + beta*frame_loss
    # Late-window additive terms (20260929). Disabled/absent blocks return (None, zeros)
    # and leave `total` the SAME tensor object -- bitwise identical to the pre-existing
    # objective. The row keys are always present so the metrics schema is stable.
    late_extra, late_row = late_window_terms(
        prediction, target, pi, data, draw, loss_cfg, completed, E, n_future_total)
    if late_extra is not None:
        total = total + late_extra
    with torch.no_grad():
        # Unweighted diagnostic over the sampled frames only; not an estimator.
        sampled_rms=torch.sqrt(torch.clamp(rel2.mean(),min=0.0))
    return total, {
        "loss_record": float(record_loss.detach()),
        "loss_frame_balanced": float(frame_loss.detach()),
        "frame_beta": float(beta),
        "frame_rel_rms_sampled": float(sampled_rms),
        "frame_eligible_count": int(n_eligible),
        **late_row,
    }
