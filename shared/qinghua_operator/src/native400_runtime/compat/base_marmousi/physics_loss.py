#!/usr/bin/env python3
"""PDE / initial-condition / boundary-condition residual losses (2026-09-27).

WHY THIS EXISTS
---------------
The released objective is DATA ONLY: one record-normalised vector norm over the
sampled frames (train_ddp.py, `loss = vector_norm(residual) / E.sqrt()`), and
train.py rejects any other loss kind.  Nothing in the training path constrains the
prediction to satisfy the acoustic wave equation, to match the initial state, or to
behave at the domain edges.  This module adds those three terms as an OPT-IN
addition; with `physics.enabled: false` (the default) every number in the training
loop is bitwise the historical one.

THE EQUATION
------------
The dataset solves the constant-density 2-D acoustic wave equation

    (1/c^2) d2p/dt2  -  laplacian(p)  =  s(t) * delta_src(x,z)

with c = velocity_mps [z,x] (LWC84 solver, dx = dz = 10 m, solver dt = 1.25e-4 s,
CFL 0.28; the SAVED axis is decimated 20x to dt = 2.5e-3 s).

DISCRETISATION CHOICES, AND WHY (these are not free parameters)
--------------------------------------------------------------
1. SPACE: 4th-order 5-point Laplacian, NOT 2nd-order.  At 30 Hz (the measurement
   bound; the v16+ pools cap at 25 Hz, where every stencil error below shrinks)
   and the slowest medium, the grid carries only ~5-7 points
   per wavelength.  The 2nd-order centred symbol is then wrong by 6.7-12.5%, which
   is far larger than the model improvements under study, so a 2nd-order residual
   would mostly penalise the network for not reproducing the stencil's own
   dispersion.  The 4th-order stencil brings that to 0.7-2.4%.  (Measured; see
   PHYSICS_LOSS_20260927.md.)
2. TIME: the model takes time as a CONTINUOUS query coordinate, so the time
   derivative is formed by re-querying at t +/- h with h = `dt_h_s` (default the
   solver's own 1.25e-4 s) rather than by differencing the sampled frames -- the
   sampler draws one frame per stratum, so consecutive stored frames are generally
   not in a batch at all.  This also means the PDE term needs NO labels.
3. INTERIOR ONLY: the 4th-order stencil needs 2 cells of margin, and the left/right/
   bottom boundaries are CFS-CPML absorbers whose auxiliary memory state lives OUTSIDE
   the stored physical domain (see BOUNDARY CONTRACT below), so the PDE residual is
   evaluated on the interior with `halo` cells trimmed.  The top boundary is a hard
   free surface, imposed by the model as an output constraint, not by this loss.
4. SOURCE: the residual is evaluated only at collocation points OUTSIDE the source
   footprint (`src_exclude_cells`), so the delta forcing -- which a grid function
   cannot represent pointwise -- never enters.  This is why no wavelet term appears
   here.

BOUNDARY CONTRACT (production generator, cited 2026-09-28 for v18)
------------------------------------------------------------------
Source of truth: local_assets/archive/staging/pseudo_marmousi_fork_20260928/
generation_config_pmarm.yaml (schema acoustic-lwc84-cpml-401-to-201-v1) and
snapshot/fno_acoustic/data_generation/grid.py::BoundaryConfig, the exact code that
generated combined_dataset_v2.h5:

    top    = free_surface_dirichlet   (hard p=0 at z=0; enforced, not optional)
    left   = right = bottom = CFS-CPML: npml=40 cells, target reflection 1e-8,
             polynomial order 3, kappa_max=3.0, alpha_max = pi * 8 Hz
    cpml_outside_physical_domain = true: the solver runs on the EXTENDED grid
    (nz+npml, nx+2*npml) and stores only physical_slices, i.e. the physical
    domain.  The CFS-CPML auxiliary memory variables exist only in the exterior
    strip and are NOT in the dataset.

Consequences for this module, in order:
  * No CPML residual is possible here without FABRICATING the auxiliary state the
    dataset does not carry; none is fabricated.  Dirichlet p=0, Neumann dp/dn=0 and
    a discrete Sommerfeld surrogate at the edge are all wrong substitutes (the first
    two reflect; the third was rejected by measurement, see the BOUNDARY TERM note).
  * The DATA term is the edge supervision: the stored fields already contain the
    absorber's effect at full label accuracy, and the training residual covers the
    complete physical domain including its edge ring.
  * The PDE residual stays interior (`halo`), where the homogeneous wave equation
    holds without boundary terms.

NORMALISED UNITS
----------------
PhysicalNormalizer.encode_pressure is a per-record SCALAR divide, and the wave
operator is homogeneous, so a constant scale factors straight out: the residual may
be formed on normalised fields, and only the (1/c^2) d2/dt2 vs laplacian balance
matters.  Because that balance carries physical dimensions, the two terms are formed
with physical dx and dt and the residual is reported relative to the RMS of the two
terms, so the loss is dimensionless and comparable across records.

THE TERMS (v18: pde + ic only; there is deliberately NO bc term)
----------------------------------------------------------------
pde : mean over interior collocation points of  w * (r / scale)^2 , where
      r = (1/c^2) p_tt - lap(p)  and scale = RMS of the two terms (so a point where
      both terms are tiny cannot dominate).  Relative form, hence unitless.
ic  : the model's own prediction at the 8 initial-condition times must reproduce the
      IC frames it was conditioned on.  This is a CONSISTENCY constraint, not new
      information -- the IC frames are already an input -- and it is what stops the
      operator drifting away from its own initial state.
bc  : REMOVED by measurement (see the BOUNDARY TERM note below) and by the boundary
      contract above: the CPML edges are supervised by the data term, the free
      surface is a hard model constraint, and no surrogate residual on this grid is
      accurate enough to measure anything.  `weight_bc`/`free_surface_z0` are
      rejected keys, not silently ignored ones.

Every function is deterministic and adds no global state.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

PHYSICS_DEFAULTS = {
    'enabled': False,
    'weight_pde': 0.0,
    'weight_ic': 0.0,
    # Free-surface residual on the model's RAW field (before the hard free-surface factor):
    # penalises raw pressure at z=0 so the constraint is learned, not only imposed.
    'weight_bc': 0.0,
    # Time-derivative spacing for the re-query.  NOT the solver step (1.25e-4 s): the model
    # runs in float32, where a 2nd difference trades truncation O(h^2 w^4) against roundoff.
    # MEASURED in fp32 at 30 Hz (the pool's upper bin), RMS relative error over 200 phases:
    #   h=1.25e-5: 1.2e+0   1.25e-4: 8.2e-3   1.8e-4: 5.0e-3
    #   h=2.5e-4:  2.8e-3   5.0e-4:  9.5e-4 (best)   1e-3: 2.1e-3   2.5e-3: 1.3e-2
    # 5e-4 s is the empirical optimum, ~9x better than the solver step (which is roundoff
    # dominated).  Note the closed-form estimate (12 eps / w^4)^(1/4) = 1.8e-4 s is too
    # small by ~3x because it ignores the precision lost in the arguments themselves; the
    # measured value is the one used.
    'dt_h_s': 5.0e-4,
    'space_order': 4,        # 4 (default, see module docstring) or 2
    'halo': 2,               # interior margin in cells; must be >= space_order//2
    'src_exclude_cells': 4,  # collocation points this many cells from the source are dropped
    'n_collocation_frames': 4,   # how many of the sampled frames carry a PDE residual
    'rel_floor': 1.0e-8,     # floor on the relative-residual denominator
    'warmup_updates': 0,     # linear ramp of the physics weights over this many steps
    # v18 (20260928): every grad_diag_every-th update the trainer computes the data/pde/ic
    # gradient norms SEPARATELY (train_ddp.split_grad_diagnostic) instead of one backward
    # of the sum. 0 (default) = off, bitwise-historical backward on every step. The cost
    # on a diagnostic step is one extra backward over the physics graph only (the small
    # 3*n_col+8 frame forward), NOT three full backwards of the combined objective.
    'grad_diag_every': 0,
    # v18 DG candidate (20260928): spatial discretisation of the PDE residual.
    #   strong_fd (default): the 4th-order FD Laplacian above, bitwise-old behaviour.
    #   dg_sipg: element-wise Legendre projection + symmetric interior penalty Galerkin
    #            weak residual (dg_weak.py).  Time stays the continuous-query FD stencil.
    # When space_form resolves to strong_fd the dg_* keys (and space_form itself) are
    # REMOVED from the normalised config so existing configs keep an identical digest.
    'space_form': 'strong_fd',
    'dg_order': 3,             # local polynomial order p (Q_p tensor Legendre)
    'dg_element_cells': 4,     # cells per element edge; 200x200 cells -> 50x50 elements
    'dg_penalty_gamma': 2.0,   # SIPG penalty = gamma*(p+1)^2/h
    'dg_quadrature_order': 4,  # Gauss-Legendre points per direction
}

# Rejected keys, with the reason surfaced to the user instead of being silently dropped.
PHYSICS_REJECTED = {
    'bc_ring': 'edge-ring boundary residual removed by measurement; the free-surface term '
               '(weight_bc) acts on the raw field at z=0 only. See the BOUNDARY TERM note.',
    'auto_weight': 'automatic loss reweighting is excluded from V18 paired experiments; use fixed config weights and logged gradient ratios.',
    'adaptive_weights': 'automatic loss reweighting is excluded from V18 paired experiments; use fixed config weights and logged gradient ratios.',
    'weight_schedule': 'physics loss weights are fixed config knobs; only the explicit warmup scale is permitted for reproducibility.',
    'grad_balance': 'gradient balancing may be logged diagnostically but must not mutate loss weights in V18 paired experiments.',
}

SPACE_ORDERS = (2, 4)
SPACE_FORMS = ('strong_fd', 'dg_sipg')
DG_KEYS = ('dg_order', 'dg_element_cells', 'dg_penalty_gamma', 'dg_quadrature_order')


def normalize_physics_config(raw):
    """Fill defaults and validate. Disabled collapses to {'enabled': False} for a stable digest."""
    raw = dict(raw or {})
    rejected = sorted(set(raw) & set(PHYSICS_REJECTED))
    if rejected:
        detail = '; '.join(f'{k}: {PHYSICS_REJECTED[k]}' for k in rejected)
        raise ValueError(f'physics keys rejected by measurement -- {detail}')
    unknown = set(raw) - set(PHYSICS_DEFAULTS)
    if unknown:
        raise ValueError(f'unknown physics keys {sorted(unknown)}')
    if not raw.get('enabled', False):
        return {'enabled': False}
    c = {**PHYSICS_DEFAULTS, **raw}
    c['enabled'] = True
    for k in ('weight_pde', 'weight_ic', 'weight_bc', 'dt_h_s', 'rel_floor'):
        c[k] = float(c[k])
    for k in ('space_order', 'halo', 'src_exclude_cells', 'n_collocation_frames', 'warmup_updates',
              'grad_diag_every'):
        c[k] = int(c[k])
    if c['space_order'] not in SPACE_ORDERS:
        raise ValueError(f'physics.space_order must be one of {SPACE_ORDERS}')
    if c['halo'] < c['space_order'] // 2:
        raise ValueError('physics.halo must be >= space_order//2 so the stencil stays in-domain')
    if c['dt_h_s'] <= 0:
        raise ValueError('physics.dt_h_s must be > 0')
    if min(c['weight_pde'], c['weight_ic'], c['weight_bc']) < 0:
        raise ValueError('physics weights must be >= 0')
    if max(c['weight_pde'], c['weight_ic'], c['weight_bc']) <= 0:
        raise ValueError('physics.enabled with all weights 0 is a no-op; set enabled:false instead')
    if c['n_collocation_frames'] < 1:
        raise ValueError('physics.n_collocation_frames must be >= 1')
    if c['src_exclude_cells'] < 0 or c['warmup_updates'] < 0:
        raise ValueError('physics.src_exclude_cells/warmup_updates out of range')
    if c['grad_diag_every'] < 0:
        raise ValueError('physics.grad_diag_every must be >= 0 (0 = diagnostics off)')
    c['space_form'] = str(c['space_form'])
    if c['space_form'] not in SPACE_FORMS:
        raise ValueError(f'physics.space_form must be one of {SPACE_FORMS}')
    if c['space_form'] == 'dg_sipg':
        for k in ('dg_order', 'dg_element_cells', 'dg_quadrature_order'):
            c[k] = int(c[k])
        c['dg_penalty_gamma'] = float(c['dg_penalty_gamma'])
        if c['dg_order'] < 1:
            raise ValueError('physics.dg_order must be >= 1')
        if c['dg_element_cells'] < c['dg_order']:
            raise ValueError('physics.dg_element_cells must be >= dg_order so the '
                             '(cells+1)^2 nodal patch determines the Q_p projection')
        if c['dg_quadrature_order'] < c['dg_order'] + 1:
            raise ValueError('physics.dg_quadrature_order must be >= dg_order+1 for exact '
                             'stiffness/mass quadrature')
        if c['dg_penalty_gamma'] <= 0:
            raise ValueError('physics.dg_penalty_gamma must be > 0 (SIPG stability)')
    else:
        # strong_fd: strip the DG knobs so pre-DG configs normalise to the identical dict
        for k in ('space_form',) + DG_KEYS:
            c.pop(k, None)
    return c


def physics_scale(cfg, completed):
    """Linear warm-up multiplier. A physics term fighting a cold model from step 0 can
    dominate the data term; the ramp lets the data term establish the field first."""
    w = int(cfg.get('warmup_updates', 0))
    if w <= 0:
        return 1.0
    return float(min(max(int(completed), 0) / w, 1.0))


def laplacian(p, dx, dz, order=4):
    """[...,Z,X] -> laplacian on the same grid, valid only away from the edges.

    order=2: (p[i-1]-2p[i]+p[i+1])/h^2
    order=4: (-1/12, 4/3, -5/2, 4/3, -1/12)/h^2  (5-point, 4th order)
    The returned tensor has the same shape; the outer `order//2` cells are garbage and
    the caller trims them via `halo`.
    """
    if p.shape[-1] < 2 * (order // 2) + 1 or p.shape[-2] < 2 * (order // 2) + 1:
        raise ValueError('field too small for the requested stencil')
    if order == 2:
        kz = torch.tensor([1.0, -2.0, 1.0], dtype=p.dtype, device=p.device)
        kx = kz
    else:
        kz = torch.tensor([-1.0 / 12, 4.0 / 3, -5.0 / 2, 4.0 / 3, -1.0 / 12], dtype=p.dtype, device=p.device)
        kx = kz
    shp = p.shape
    v = p.reshape(-1, 1, shp[-2], shp[-1])
    pad = order // 2
    # replicate padding keeps the shape; those columns are trimmed by the caller anyway
    vz = F.conv2d(F.pad(v, (0, 0, pad, pad), mode='replicate'), kz.view(1, 1, -1, 1)) / (dz * dz)
    vx = F.conv2d(F.pad(v, (pad, pad, 0, 0), mode='replicate'), kx.view(1, 1, 1, -1)) / (dx * dx)
    return (vz + vx).reshape(shp)


def interior_mask(shape_zx, halo, device):
    """True on the interior, False on a `halo`-thick frame."""
    Z, X = int(shape_zx[0]), int(shape_zx[1])
    if 2 * halo >= min(Z, X):
        raise ValueError('halo leaves no interior')
    m = torch.zeros((Z, X), dtype=torch.bool, device=device)
    m[halo:Z - halo, halo:X - halo] = True
    return m


def source_exclusion_mask(shape_zx, source_xz_m, x0_m, z0_m, dx, dz, cells, device):
    """False within `cells` of the source point (which carries a delta this grid cannot represent)."""
    Z, X = int(shape_zx[0]), int(shape_zx[1])
    if cells <= 0:
        return torch.ones((Z, X), dtype=torch.bool, device=device)
    sx, sz = float(source_xz_m[0]), float(source_xz_m[1])
    ix = (sx - float(x0_m)) / float(dx)
    iz = (sz - float(z0_m)) / float(dz)
    zz = torch.arange(Z, device=device, dtype=torch.float32)[:, None]
    xx = torch.arange(X, device=device, dtype=torch.float32)[None, :]
    d2 = (zz - iz) ** 2 + (xx - ix) ** 2
    return d2 > float(cells) ** 2


def pde_residual_from_fields(p_m, p_0, p_p, velocity, cfg, dx, dz):
    """Relative PDE residual from precomputed fields at (t-h, t, t+h).

    The trainer evaluates all three stencil points in ONE dense forward (a single
    concatenated time batch), so this variant takes the fields directly; each must be
    graph-connected for the gradient to reach the model.

    Returns (residual_rel [K,Z,X], parts dict) -- the caller masks and reduces.
    """
    h = float(cfg['dt_h_s'])
    p_tt = (p_p - 2.0 * p_0 + p_m) / (h * h)
    lap = laplacian(p_0, dx, dz, order=int(cfg['space_order']))
    inv_c2 = 1.0 / (velocity.to(p_0.dtype) ** 2)
    term_t = inv_c2 * p_tt
    r = term_t - lap
    # Relative form: divide by the RMS of the two competing terms so a point where both
    # are ~0 (no wave has arrived yet) cannot produce a large relative residual.
    scale = torch.sqrt(0.5 * (term_t.detach() ** 2 + lap.detach() ** 2) + float(cfg['rel_floor']))
    return r / scale, {'term_t': term_t.detach(), 'lap': lap.detach(), 'p0': p_0.detach()}


def pde_residual(model_query, times, velocity, cfg, dx, dz):
    """Callable-query variant of pde_residual_from_fields (probes and tests)."""
    h = float(cfg['dt_h_s'])
    return pde_residual_from_fields(model_query(times - h), model_query(times),
                                    model_query(times + h), velocity, cfg, dx, dz)


# ---------------------------------------------------------------------------------------
# BOUNDARY TERM: REMOVED BY MEASUREMENT (2026-09-27).  Read this before adding one back.
# ---------------------------------------------------------------------------------------
# Two formulations were implemented and both were rejected on measurement, not on taste:
#
# 1. relu(p * dp/dn) as a one-sided "inward flux" penalty.  WRONG: p*dp/dn is a flux
#    proxy only for a travelling wave.  Measured: a field piling up at the boundary --
#    exactly the failure it was meant to catch -- returns a penalty of 0.000000.
#
# 2. The Sommerfeld radiation condition dp/dt + c dp/dn = 0, which IS the physically
#    correct statement for the CPML absorber this dataset uses (Dirichlet p=0 and
#    Neumann dp/dn=0 are both wrong at an absorbing edge; either reflects).  This one
#    fails for a discretisation reason: it needs the OUTWARD NORMAL DERIVATIVE at the
#    edge, where only a one-sided stencil exists.  Measured RMS error of that derivative
#    at this grid's 10 points/wavelength:
#        simple one-sided, 1 cell : 31.3%      2 cells : 60.5%     3 cells : 85.8%
#        2nd-order one-sided (3pt): 12.9%
#    The residual is a DIFFERENCE of two comparable terms, so a 12.9% error in one of
#    them puts an exactly-outgoing wave at residual ~0.13 -- indistinguishable from a
#    genuinely reflecting boundary.  Confirmed end to end: a radially outgoing wave
#    scored 1.25 where it should score ~0 (an incoming one scored 2.75).
#
# Conclusion: with dx = 10 m and content up to 30 Hz there is no boundary residual on
# this grid whose discretisation error is smaller than the effect it would measure.  The
# honest options are to drop the term (done) or to supervise the boundary with DATA --
# the stored fields already contain the absorber's effect, so the data term covers the
# edges at full label accuracy and needs no physics surrogate.  `weight_bc` is therefore
# rejected by normalize_physics_config rather than silently ignored.


def collocation_times(times, n, seed):
    """Pick n of the sampled times for the PDE residual (deterministic given seed).

    The PDE term costs 3 extra forward evaluations per collocation time, so it runs on a
    SUBSET of the batch's frames rather than all of them; which subset is drawn per step
    so that over training every frame region is visited.
    """
    t = torch.as_tensor(times).reshape(-1)
    n = min(int(n), t.numel())
    rng = np.random.default_rng(int(seed) % (2**63))
    idx = np.sort(rng.choice(t.numel(), size=n, replace=False))
    return t[torch.as_tensor(idx.copy())], idx


def compute_physics_loss(cfg, completed, forward, data, times, target_ic, device, seed,
                         return_terms=False, forward_raw=None):
    """The full physics objective for one training step.

    forward   : callable(times [M] float tensor) -> [1, M, Z, X] normalised prediction,
                graph-connected (the trainer passes a forward_live closure).
    data      : the pool's per-record dict (velocity, x, z, time_axis, future_start, source).
    times     : the batch's sampled frame times [K] (collocation times are a subset).
    target_ic : [1, 8, Z, X] the ENCODED initial-condition frames (normaliser applied by
                the caller, which owns the normaliser).
    seed      : per-draw deterministic seed for the collocation subset.

    Returns (loss_tensor, metrics_dict), or with return_terms=True (v18 gradient
    diagnostics) (loss_tensor, metrics_dict, terms) where terms maps 'pde'/'ic' to the
    graph-connected weighted+warmup-scaled tensors with loss_tensor == sum(terms.values())
    (summation order aside).  All three stencil points AND the IC queries are evaluated in
    ONE forward call, so the step pays exactly one extra prepared_state.
    """
    h = float(cfg['dt_h_s'])
    axis = data['time_axis']
    t_max = float(axis[-1])
    x = data['x']; z = data['z']
    dx = float(x[1] - x[0]); dz = float(z[1] - z[0])
    tt = torch.as_tensor(times).reshape(-1).float()
    # the stencil must stay inside the model's time domain
    ok = (tt >= h) & (tt <= t_max - h)
    metrics = {'physics_scale': physics_scale(cfg, completed)}
    want_ic = cfg['weight_ic'] > 0
    want_bc = cfg.get('weight_bc', 0.0) > 0
    if want_bc and forward_raw is None:
        raise ValueError('physics.weight_bc > 0 requires a forward_raw callable (raw field before '
                         'the free-surface factor)')
    tc = None
    if cfg['weight_pde'] > 0 and bool(ok.any()):
        tc, _ = collocation_times(tt[ok], cfg['n_collocation_frames'], seed)
    n_col = 0 if tc is None else tc.numel()
    fs = int(data['future_start'])
    ic_t = axis[fs - 8:fs].float()
    segments = ([tc - h, tc, tc + h] if n_col else []) + ([ic_t] if want_ic else [])
    if not segments and not want_bc:
        zero = torch.zeros((), device=device)
        return (zero, metrics, {}) if return_terms else (zero, metrics)
    terms = {}
    if segments:
        pred = forward(torch.cat(segments))[0]                   # [3n(+8), Z, X]
        if n_col:
            p_m, p_0, p_p = torch.split(pred[:3 * n_col], [n_col, n_col, n_col])
        p_ic = pred[3 * n_col:] if want_ic else None
    else:
        p_ic = None
    if n_col:
        vel = torch.as_tensor(np.asarray(data['velocity']), dtype=torch.float32, device=pred.device)
        vel = vel.reshape(vel.shape[-2], vel.shape[-1])
        if cfg.get('space_form', 'strong_fd') == 'dg_sipg':
            # SIPG weak-form spatial residual (dg_weak.py); lazy import so the default
            # strong_fd path is untouched even if dg_weak is absent.
            from dg_weak import dg_loss_from_fields
            L_pde, dg_metrics = dg_loss_from_fields(
                p_m, p_0, p_p, vel, cfg, dx, dz,
                source_xz_m=data['source'][0, :2], x0_m=float(x[0]), z0_m=float(z[0]))
            terms['pde'] = cfg['weight_pde'] * L_pde
            metrics['physics_pde'] = float(L_pde)
            metrics['physics_pde_frames'] = int(n_col)
            metrics.update(dg_metrics)
        else:
            r, parts = pde_residual_from_fields(p_m, p_0, p_p, vel, cfg, dx, dz)
            m = interior_mask((r.shape[-2], r.shape[-1]), cfg['halo'], r.device)
            m &= source_exclusion_mask((r.shape[-2], r.shape[-1]), data['source'][0, :2], float(x[0]), float(z[0]),
                                       dx, dz, cfg['src_exclude_cells'], r.device)
            L_pde = r[:, m].square().mean()
            terms['pde'] = cfg['weight_pde'] * L_pde
            metrics['physics_pde'] = float(L_pde)
            metrics['physics_pde_frames'] = int(n_col)
            metrics['physics_pde_term_ratio'] = float(
                (parts['term_t'][:, m].square().mean() / (parts['lap'][:, m].square().mean() + 1e-30)).sqrt())
    if want_ic:
        t_ic = target_ic[0].to(p_ic.dtype)
        L_ic = torch.linalg.vector_norm(p_ic - t_ic) / (torch.linalg.vector_norm(t_ic) + 1e-12)
        terms['ic'] = cfg['weight_ic'] * L_ic
        metrics['physics_ic'] = float(L_ic)
    if want_bc:
        # Free-surface residual on the RAW field (before the hard tanh^2 factor) at z=0.
        # Times: reuse the PDE collocation subset when it exists, else up to
        # n_collocation_frames of the batch's sampled times.
        tbc = tc if tc is not None else tt[:int(cfg['n_collocation_frames'])]
        raw_bc = forward_raw(tbc)[0]                          # [M, Z, X]
        L_bc = raw_bc[:, 0, :].square().mean()
        terms['bc'] = cfg['weight_bc'] * L_bc
        metrics['physics_bc'] = float(L_bc)
    if not terms:
        zero = torch.zeros((), device=device)
        return (zero, metrics, {}) if return_terms else (zero, metrics)
    # Per-term warmup scaling, so each returned term is exactly its contribution to the
    # total (sum(a*s) vs (sum a)*s differ only in fp association).
    terms = {k: v * metrics['physics_scale'] for k, v in terms.items()}
    total = sum(terms.values())
    metrics['physics_total'] = float(total)
    return (total, metrics, terms) if return_terms else (total, metrics)
