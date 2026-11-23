"""ST-RAS: spatio-temporal residual-adaptive sampling (20260925).

Three EMA residual tables (all initialised to 1, tau=0.9 by default, so the first
visit of every record/stratum is ~uniform and the tables sharpen with visits):

  r[i]       record level : EMA of the record's UNBIASED full-future relative L2
                            (Horvitz-Thompson form, independent of the weighting)
  Rt[i,s]    time level   : EMA of the relative L2 of the frames drawn from stratum s
  Rx[i,s,c]  space level  : EMA of the residual-energy fraction of frame(s) drawn from
                            stratum s on a GxG coarse grid (adaptive_avg_pool2d of the
                            squared-error field, normalised to sum 1 over the G*G cells)

Decisions are pure functions of (config seed, draw_index, table state):
  record : the 9-cell round robin is untouched; inside the cell
           p_i = (1-eps) r_i^alpha / sum_j r_j^alpha + eps / |cell|
  time   : n_s ~ Multinomial(n_frames, q),  q_s = (1-eps) Rt[i,s]^alpha / sum + eps / n_strata,
           uniform inside a stratum (a stratum may receive 0 or several frames)
  space  : per-pixel weight W = clip((1-eps_x) up(Rx[i,s]) / mean(Rx[i,s]) + eps_x, eps_x, cap)
           (3x3 box smoothing of the coarse table, bilinear upsampling to the grid), multiplied
           into the NUMERATOR of the record-normalised loss only.

Loss weights per sampled frame k drawn from stratum s (n_s draws, allocation prob q_s):
  importance_correct = False : w_k = |S_s|                  (deliberately biased towards high residual)
  importance_correct = True  : w_k = |S_s| / (n_frames q_s) / (|cell| p_i)  and W == 1
                               -> E[sum_k w_k |err_k|^2] equals the fixed-sampling squared objective.

The 'legacy' kind never touches the tables and reproduces data.Pool.plan bitwise.
Every function here is deterministic given its inputs; no global RNG is used.
"""
from __future__ import annotations
import hashlib, math
import numpy as np
import torch
import torch.nn.functional as F

SAMPLER_DEFAULTS = {
    'kind': 'legacy',
    'alpha': 1.0, 'epsilon': 0.3, 'tau': 0.9, 'sync_every': 50, 'importance_correct': False,
    'record': True, 'time': True, 'space': True,
    'space_grid': 26, 'epsilon_x': 0.3, 'weight_cap': 10.0, 'space_smooth': True,
    'frame_energy_floor_rel': 1.0e-3,
    'intra_stratum': 'uniform',
    # Rehearsal / replay (20260927): per-family share of the DRAW budget. None keeps the
    # historical uniform cell round-robin bitwise. When set, the cell is DRAWN (seeded,
    # per-draw) with probability share[family]/3 per that family's three frequency cells,
    # instead of the deterministic draw%n rotation. This is a curriculum/design choice in
    # the same sense as the original balanced rotation: it deliberately reweights the
    # objective across families and is NOT importance-corrected (recorded per draw as
    # ras_family_share). Use case: marmousi-focused fine-tuning with a small uniform/
    # layered replay fraction against catastrophic forgetting.
    'family_shares': None,
    # Curriculum (20260928, agent_curriculum): family/frequency schedule on top of the
    # seeded cell draw. None (default) keeps both the round-robin and the plain
    # family_shares paths bitwise AND adds no key to the normalized config, so every
    # pre-existing semantic digest is unchanged. When set, see curriculum.py: family
    # balanced (default) or difficulty-aware with a MANDATORY per-family floor, staged
    # frequency weights (8-15/15-22/22-25 Hz) with a low-band replay floor and a
    # monotone high-band ramp. family_shares additionally accepts the literal string
    # 'balanced' (exact 1/3 per family).
    'curriculum': None,
    # Causal time curriculum (20260928, agent_causal_min): staged horizon on the TIME level
    # only. None or enabled=false removes the key from the normalized config, so every
    # pre-existing sampler digest and decision stream is bitwise unchanged. When enabled,
    # Pool.decide masks each stratum to the frames with record time <= horizon_s (record's
    # own time axis), zeroes strata with no eligible frame in the existing Rt floor_mix
    # allocation and renormalizes; intra_stratum='residual' runs the existing Rf floor_mix
    # over the eligible frames only. Record and space levels, loss weights semantics, and
    # the formal evaluation are untouched.
    'causal_time': None,
}
SCHEMA = 'st_ras_tables_v2'

INTRA_STRATUM_MODES = ('uniform', 'residual')


def normalize_sampler_config(raw):
    """Fill defaults and validate; legacy collapses to {'kind':'legacy'} so its digest is stable."""
    raw = dict(raw or {})
    kind = raw.get('kind', 'legacy')
    if kind == 'legacy':
        unknown = set(raw) - {'kind'}
        if unknown:
            raise ValueError(f'sampler.kind=legacy accepts no other keys, got {sorted(unknown)}')
        return {'kind': 'legacy'}
    if kind != 'st_ras':
        raise ValueError(f"sampler.kind must be 'legacy' or 'st_ras', got {kind!r}")
    unknown = set(raw) - set(SAMPLER_DEFAULTS)
    if unknown:
        raise ValueError(f'unknown sampler keys {sorted(unknown)}')
    c = {**SAMPLER_DEFAULTS, **raw}
    c['alpha'] = float(c['alpha']); c['epsilon'] = float(c['epsilon']); c['tau'] = float(c['tau'])
    c['sync_every'] = int(c['sync_every']); c['importance_correct'] = bool(c['importance_correct'])
    for k in ('record', 'time', 'space', 'space_smooth'):
        c[k] = bool(c[k])
    c['space_grid'] = int(c['space_grid']); c['epsilon_x'] = float(c['epsilon_x'])
    c['weight_cap'] = float(c['weight_cap']); c['frame_energy_floor_rel'] = float(c['frame_energy_floor_rel'])
    c['intra_stratum'] = str(c['intra_stratum'])
    if c['intra_stratum'] not in INTRA_STRATUM_MODES:
        raise ValueError(f'sampler.intra_stratum must be one of {INTRA_STRATUM_MODES}, got {c["intra_stratum"]!r}')
    fs = c['family_shares']
    if isinstance(fs, str):
        if fs != 'balanced':
            raise ValueError(f"sampler.family_shares accepts a mapping or the literal 'balanced', got {fs!r}")
        fs = {'uniform': 1.0 / 3.0, 'layered': 1.0 / 3.0, 'marmousi': 1.0 / 3.0}
        c['family_shares'] = fs
    if fs is not None:
        if not isinstance(fs, dict) or not fs:
            raise ValueError('sampler.family_shares must be a nonempty mapping family->share or absent')
        allowed = {'uniform', 'layered', 'marmousi'}
        unknown_f = set(fs) - allowed
        if unknown_f:
            raise ValueError(f'unknown families in sampler.family_shares: {sorted(unknown_f)}')
        vals = {k: float(v) for k, v in fs.items()}
        if any(v <= 0 for v in vals.values()):
            raise ValueError('sampler.family_shares values must be > 0 (omit a family instead of 0)')
        total = sum(vals.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f'sampler.family_shares must sum to 1, got {total}')
        c['family_shares'] = vals
    if not 0.0 < c['alpha'] <= 1.0: raise ValueError('sampler.alpha must lie in (0,1]')
    if not 0.0 <= c['epsilon'] <= 1.0: raise ValueError('sampler.epsilon must lie in [0,1]')
    if not 0.0 <= c['tau'] < 1.0: raise ValueError('sampler.tau must lie in [0,1)')
    if c['sync_every'] < 1: raise ValueError('sampler.sync_every must be >=1')
    if c['space_grid'] < 1: raise ValueError('sampler.space_grid must be >=1')
    if not 0.0 <= c['epsilon_x'] <= 1.0: raise ValueError('sampler.epsilon_x must lie in [0,1]')
    if c['weight_cap'] < 1.0: raise ValueError('sampler.weight_cap must be >=1')
    if c['frame_energy_floor_rel'] < 0.0: raise ValueError('sampler.frame_energy_floor_rel must be >=0')
    # Curriculum: validated against the (numeric) family shares; absent -> the key is
    # REMOVED from the normalized dict so pre-existing sampler digests stay bitwise.
    cur = c.pop('curriculum')
    if cur is not None:
        from curriculum import normalize_curriculum_config
        c['curriculum'] = normalize_curriculum_config(cur, family_shares=c['family_shares'])
    # Causal time: absent / disabled -> key removed, digest stable.
    ct = c.pop('causal_time')
    if ct is not None:
        ct = normalize_causal_time(ct)
        if ct is not None:
            if not c['time']:
                raise ValueError('sampler.causal_time requires time=True (it acts on the stratum allocation)')
            c['causal_time'] = ct
    if not (c['record'] or c['time'] or c['space']):
        raise ValueError('st_ras with record/time/space all off is legacy; use kind: legacy')
    return c


def draw_seed(config_seed, draw_index, *tags):
    """Deterministic 64-bit seed: sha256(seed|draw_index|tag...)."""
    text = '|'.join([str(config_seed), str(draw_index), *map(str, tags)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def floor_mix(values, alpha, eps):
    """p = (1-eps) v^alpha / sum v^alpha + eps / n; exact normalisation, all entries >= eps/n."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError('floor_mix needs a nonempty finite nonnegative vector')
    w = np.power(v, alpha)
    total = float(w.sum())
    base = w / total if total > 0 else np.full(v.size, 1.0 / v.size)
    p = (1.0 - eps) * base + eps / v.size
    p = p / p.sum()
    return p


def normalize_causal_time(raw):
    """Validate the causal_time block; returns None when disabled (key then dropped).

    schedule: list of {until_update, horizon_s}; until_update strictly increasing positive
    ints, the LAST entry must have until_update=None (open-ended); horizon_s > 0 and
    non-decreasing. update = floor((draw_index - sampler_draw_offset) / draws_per_update);
    stage s applies to updates < until_update."""
    raw = dict(raw)
    unknown = set(raw) - {'enabled', 'draws_per_update', 'schedule'}
    if unknown:
        raise ValueError(f'unknown sampler.causal_time keys {sorted(unknown)}')
    if not bool(raw.get('enabled', False)):
        return None
    dpu = int(raw.get('draws_per_update', 4))
    if dpu < 1:
        raise ValueError('sampler.causal_time.draws_per_update must be >=1')
    sched = raw.get('schedule')
    if not isinstance(sched, (list, tuple)) or not sched:
        raise ValueError('sampler.causal_time.schedule must be a nonempty list')
    out = []
    prev_until = 0
    prev_h = 0.0
    for j, st in enumerate(sched):
        st = dict(st)
        if set(st) != {'until_update', 'horizon_s'}:
            raise ValueError(f'causal_time schedule entries need exactly until_update and horizon_s, got {sorted(st)}')
        u = st['until_update']
        h = float(st['horizon_s'])
        if not (h > 0.0) or not math.isfinite(h):
            raise ValueError('causal_time horizon_s must be finite and > 0')
        if h < prev_h:
            raise ValueError('causal_time horizon_s must be non-decreasing')
        if j == len(sched) - 1:
            if u is not None:
                raise ValueError('the last causal_time schedule entry must have until_update: null')
        else:
            u = int(u)
            if u <= prev_until:
                raise ValueError('causal_time until_update must be strictly increasing')
            prev_until = u
        out.append({'until_update': None if u is None else int(u), 'horizon_s': h})
        prev_h = h
    return {'enabled': True, 'draws_per_update': dpu, 'schedule': out}


def causal_horizon_s(causal_cfg, update):
    """Horizon (seconds) for a given update index; the last (open-ended) stage always matches."""
    u = max(int(update), 0)
    for st in causal_cfg['schedule']:
        if st['until_update'] is None or u < int(st['until_update']):
            return float(st['horizon_s'])
    raise AssertionError('unreachable: last schedule stage is open-ended')


def entropy_norm(p):
    """Shannon entropy of a distribution normalised to [0,1] by log(n) (n=1 -> 1)."""
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    if p.size <= 1:
        return 1.0
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / math.log(p.size))


def ess(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    return float(1.0 / (p * p).sum())


class STRASTables:
    """Per-rank residual tables + delta-synchronisation bases (see sync)."""

    def __init__(self, cfg, n_records, n_strata, device='cpu', n_time=0):
        self.cfg = cfg
        self.n_records = int(n_records); self.n_strata = int(n_strata)
        self.grid = int(cfg['space_grid']); self.device = torch.device(device)
        self.r = np.ones(self.n_records, dtype=np.float64)
        self.Rt = np.ones((self.n_records, self.n_strata), dtype=np.float64)
        self.Rx = torch.ones((self.n_records, self.n_strata, self.grid * self.grid), dtype=torch.float32, device=self.device)
        # Frame-level table (20260927), for intra_stratum='residual'.  Rt[i,s] is one
        # scalar per stratum, so it carries NO information about which frame inside a
        # stratum is worth drawing; with 51 strata of 6-8 frames each there is a real
        # choice to make there.  Rf[i,f] is the EMA of the per-frame relative L2, indexed
        # by absolute frame index on the shared saved time axis (n_time), so it is
        # comparable across records regardless of onset.  1771 x 372 x 8 B = 5.3 MB, i.e.
        # negligible next to Rx's 244 MB.  n_time=0 leaves the table absent (legacy /
        # uniform arms), which keeps their checkpoints free of it.
        self.n_time = int(n_time)
        self.Rf = np.ones((self.n_records, self.n_time), dtype=np.float64) if self.n_time else None
        self.r_base = self.r.copy(); self.Rt_base = self.Rt.copy(); self.Rx_base = self.Rx.clone()
        self.Rf_base = None if self.Rf is None else self.Rf.copy()
        self.steps = 0          # optimizer updates applied to this rank's tables (decision lag reference)
        self.updates = 0        # table updates applied locally
        self.syncs = 0

    # ---- persistence -------------------------------------------------------------------------
    def state_dict(self):
        return {'schema': SCHEMA, 'config': dict(self.cfg), 'n_records': self.n_records, 'n_strata': self.n_strata,
                'grid': self.grid, 'r': self.r.copy(), 'Rt': self.Rt.copy(), 'Rx': self.Rx.detach().cpu().clone(),
                'n_time': self.n_time, **({} if self.Rf is None else {'Rf': self.Rf.copy()}),
                'steps': self.steps, 'updates': self.updates, 'syncs': self.syncs}

    def load_state_dict(self, s):
        if s.get('schema') != SCHEMA: raise ValueError('sampler table schema mismatch')
        if (int(s['n_records']), int(s['n_strata']), int(s['grid'])) != (self.n_records, self.n_strata, self.grid):
            raise ValueError('sampler table shape mismatch')
        self.r = np.array(s['r'], dtype=np.float64); self.Rt = np.array(s['Rt'], dtype=np.float64)
        self.Rx = torch.as_tensor(s['Rx'], dtype=torch.float32).to(self.device).clone()
        assert self.r.shape == (self.n_records,) and self.Rt.shape == (self.n_records, self.n_strata) and tuple(self.Rx.shape) == (self.n_records, self.n_strata, self.grid * self.grid)
        assert np.all(np.isfinite(self.r)) and np.all(np.isfinite(self.Rt)) and bool(torch.isfinite(self.Rx).all())
        # Frame table: presence must match this run's configuration, so a resume cannot
        # silently drop the intra-stratum residual weighting (or silently inherit a table
        # the current config does not use).
        if int(s.get('n_time', 0)) != self.n_time:
            raise ValueError(f'sampler frame-table n_time mismatch: checkpoint {s.get("n_time", 0)} vs config {self.n_time}')
        if self.Rf is None:
            if 'Rf' in s: raise ValueError('checkpoint carries a frame table but this run has none')
        else:
            if 'Rf' not in s: raise ValueError('intra_stratum=residual requires Rf in the checkpoint')
            self.Rf = np.array(s['Rf'], dtype=np.float64)
            assert self.Rf.shape == (self.n_records, self.n_time) and np.all(np.isfinite(self.Rf))
        self.r_base = self.r.copy(); self.Rt_base = self.Rt.copy(); self.Rx_base = self.Rx.clone()
        self.Rf_base = None if self.Rf is None else self.Rf.copy()
        self.steps = int(s.get('steps', 0)); self.updates = int(s.get('updates', 0)); self.syncs = int(s.get('syncs', 0))

    # ---- decisions ----------------------------------------------------------------------------
    def record_probabilities(self, roster_indices):
        if not self.cfg['record']:
            return np.full(len(roster_indices), 1.0 / len(roster_indices))
        return floor_mix(self.r[np.asarray(roster_indices)], self.cfg['alpha'], self.cfg['epsilon'])

    def time_probabilities(self, roster_index):
        if not self.cfg['time']:
            return np.full(self.n_strata, 1.0 / self.n_strata)
        return floor_mix(self.Rt[roster_index], self.cfg['alpha'], self.cfg['epsilon'])

    def frame_probabilities(self, roster_index, frames):
        """Residual-adaptive importance distribution INSIDE one stratum.

        `frames`: the absolute frame indices making up the stratum (6-8 of them at 51
        strata).  Returns p over those frames, same floor_mix form as the other two
        levels, so a frame that has never been visited (table value 1) is never starved
        and the eps floor keeps every frame reachable.

        Unlike the stratum allocation, this is a WITH-replacement categorical draw per
        frame slot, so the per-frame inclusion probability is exactly p, and the caller
        divides by it to keep the Horvitz-Thompson algebra intact.
        """
        idx = np.asarray(frames, dtype=np.int64)
        if self.Rf is None or self.cfg['intra_stratum'] != 'residual':
            return np.full(idx.size, 1.0 / idx.size)
        return floor_mix(self.Rf[int(roster_index)][idx], self.cfg['alpha'], self.cfg['epsilon'])

    # ---- loss-side weights --------------------------------------------------------------------
    def spatial_weights(self, roster_index, frame_strata, out_hw):
        """[n_frames, Z, X] per-pixel weights for the sampled frames.

        None when space is off OR importance_correct is on (the corrected objective must keep
        the expectation equal to the fixed-sampling objective, so pixels stay unweighted)."""
        if not self.cfg['space'] or self.cfg['importance_correct']:
            return None
        return spatial_weights_from_rows(self.Rx[roster_index, list(frame_strata)], out_hw, self.grid,
                                         self.cfg['epsilon_x'], self.cfg['weight_cap'], self.cfg['space_smooth'])

    # ---- updates ------------------------------------------------------------------------------
    @torch.no_grad()
    def update(self, roster_index, frame_strata, err2, target2_sum, E, n_future, stratum_sizes, alloc_q, frames=None):
        """err2: [n_frames, Z, X] squared error (detached; CPU or CUDA — pooling runs on its
        device, the table math on CPU); target2_sum: [n_frames] target energies.

        Returns a dict of diagnostics (rho per frame, unbiased loss, updated strata, space entropy)."""
        tau = self.cfg['tau']; i = int(roster_index); K = err2.shape[0]
        strata = np.asarray(frame_strata, dtype=np.int64); assert strata.shape == (K,)
        err2 = err2.detach()
        e_k = err2.sum(dim=(1, 2)).double().cpu()                                  # [K]
        t_k = torch.as_tensor(target2_sum, dtype=torch.float64).cpu()
        E = float(E); floor = self.cfg['frame_energy_floor_rel'] * E / max(int(n_future), 1)
        rho = (e_k / torch.clamp(t_k, min=floor)).sqrt()                           # relative L2 per sampled frame
        sizes = np.asarray(stratum_sizes, dtype=np.float64); q = np.asarray(alloc_q, dtype=np.float64)
        ht = torch.as_tensor(sizes[strata] / (K * q[strata]), dtype=torch.float64)
        loss_unb = float(((ht * e_k).sum() / E).sqrt())
        # assignment matrix M[s,k] = 1/n_s for frames of stratum s (CPU)
        n_s = np.bincount(strata, minlength=self.n_strata).astype(np.float64)
        M = torch.zeros((self.n_strata, K), dtype=torch.float64)
        M[torch.as_tensor(strata), torch.arange(K)] = 1.0
        M = M / torch.clamp(torch.as_tensor(n_s)[:, None], min=1.0)
        visited = n_s > 0
        rho_s = (M @ rho).numpy()
        out = {'rho': [float(x) for x in rho.numpy()], 'loss_unbiased': loss_unb,
               'updated_strata': [int(s) for s in np.nonzero(visited)[0]]}
        if self.cfg['record']:
            self.r[i] = tau * self.r[i] + (1.0 - tau) * loss_unb
        if self.cfg['time']:
            self.Rt[i, visited] = tau * self.Rt[i, visited] + (1.0 - tau) * rho_s[visited]
        if self.Rf is not None and frames is not None:
            # Frame level: EMA of the per-frame relative L2 at the ABSOLUTE frame indices
            # just drawn.  A frame may be drawn more than once in a step (the stratum draw
            # is with replacement), so average the duplicates first -- applying the EMA
            # twice in one step would decay the old value by tau^2 and double-count.
            f_idx = np.asarray(frames, dtype=np.int64)
            assert f_idx.shape == (K,), (f_idx.shape, K)
            rho_np = rho.numpy()
            uniq, inv = np.unique(f_idx, return_inverse=True)
            acc = np.zeros(uniq.size, dtype=np.float64); cnt = np.zeros(uniq.size, dtype=np.float64)
            np.add.at(acc, inv, rho_np); np.add.at(cnt, inv, 1.0)
            self.Rf[i, uniq] = tau * self.Rf[i, uniq] + (1.0 - tau) * (acc / cnt)
            out['frames_updated'] = int(uniq.size)
        if self.cfg['space']:
            d = F.adaptive_avg_pool2d(err2[:, None].float(), self.grid)[:, 0].flatten(1).double().cpu()  # [K, G*G]
            tot = d.sum(dim=1, keepdim=True)
            d = torch.where(tot > 0, d / torch.clamp(tot, min=1e-300), torch.full_like(d, 1.0 / d.shape[1]))
            d_s = (M @ d)                                                                # [n_strata, G*G]
            vis = torch.from_numpy(visited)
            rows = self.Rx[i].double().cpu()
            rows[vis] = tau * rows[vis] + (1.0 - tau) * d_s[vis]
            self.Rx[i] = rows.to(self.Rx.dtype).to(self.Rx.device)
            p = rows[vis] / rows[vis].sum(dim=1, keepdim=True)
            H = -(p * torch.log(torch.clamp(p, min=1e-300))).sum(dim=1) / math.log(p.shape[1])
            out['entropy_space'] = float(H.mean())
        self.updates += 1
        return out

    # ---- distributed --------------------------------------------------------------------------
    def sync(self, world, comm_device=None, group=None):
        """Delta-sync: every rank adds the sum of all ranks' local deltas since the last sync to
        the common base, so disjoint visits combine exactly like sequential EMA updates.
        comm_device: device for the all-reduce buffers (NCCL needs CUDA); tables stay on self.device."""
        import torch.distributed as dist
        if world > 1:
            dev = torch.device(comm_device) if comm_device is not None else self.device
            dr = torch.from_numpy(self.r - self.r_base).to(dev)
            dt = torch.from_numpy(self.Rt - self.Rt_base).to(dev)
            dx = (self.Rx - self.Rx_base).to(dev)
            dist.all_reduce(dr, op=dist.ReduceOp.SUM, group=group); dist.all_reduce(dt, op=dist.ReduceOp.SUM, group=group); dist.all_reduce(dx, op=dist.ReduceOp.SUM, group=group)
            # Positivity floor: if several ranks shrank the SAME entry within one window the linear
            # delta sum can undershoot (base + sum of negative deltas); tables must stay > 0.
            self.r = np.maximum(self.r_base + dr.cpu().numpy(), 1e-8)
            self.Rt = np.maximum(self.Rt_base + dt.cpu().numpy(), 1e-8)
            self.Rx = torch.clamp(self.Rx_base + dx.to(self.Rx.device), min=1e-8)
            assert np.all(np.isfinite(self.r)) and np.all(np.isfinite(self.Rt)) and bool(torch.isfinite(self.Rx).all())
            if self.Rf is not None:
                # Same delta-sum as the other tables: without this the five ranks would each
                # steer their intra-stratum draws by a private frame table.
                df = torch.from_numpy(self.Rf - self.Rf_base).to(dev)
                dist.all_reduce(df, op=dist.ReduceOp.SUM, group=group)
                self.Rf = np.maximum(self.Rf_base + df.cpu().numpy(), 1e-8)
                assert np.all(np.isfinite(self.Rf))
        self.r_base = self.r.copy(); self.Rt_base = self.Rt.copy(); self.Rx_base = self.Rx.clone()
        if self.Rf is not None: self.Rf_base = self.Rf.copy()
        self.syncs += 1

    def table_entropies(self):
        """Global sharpness of the tables (normalised entropies in [0,1]; 1 = uniform).
        Space is the mean row entropy over a fixed subsample of rows (full table is 19M cells)."""
        pr = self.r / self.r.sum()
        rt = self.Rt / self.Rt.sum(axis=1, keepdims=True)
        out = {'table_entropy_records': entropy_norm(pr),
               'table_entropy_frames': float(np.mean(-(rt * np.log(np.clip(rt, 1e-300, None))).sum(axis=1)) / math.log(self.n_strata))}
        rows = self.Rx[:: max(self.n_records // 64, 1)].double().flatten(0, 1)
        p = rows / rows.sum(dim=1, keepdim=True)
        out['table_entropy_space'] = float((-(p * torch.log(torch.clamp(p, min=1e-300))).sum(dim=1) / math.log(p.shape[1])).mean())
        return out

    def summary(self):
        return {'r_mean': float(self.r.mean()), 'r_min': float(self.r.min()), 'r_max': float(self.r.max()),
                'Rt_mean': float(self.Rt.mean()), 'Rx_mean': float(self.Rx.mean()), 'updates': self.updates, 'syncs': self.syncs, 'steps': self.steps}


def spatial_weights_from_rows(rows, out_hw, grid, eps_x, cap, smooth=True):
    """rows: [K, grid*grid] coarse residual rows -> [K, Z, X] weights, mean ~1 before clipping."""
    K = rows.shape[0]
    R = rows.detach().float().view(K, 1, grid, grid)
    if smooth and grid >= 2:
        R = F.avg_pool2d(R, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    R = R / torch.clamp(R.mean(dim=(2, 3), keepdim=True), min=1e-30)
    up = F.interpolate(R, size=tuple(int(x) for x in out_hw), mode='bilinear', align_corners=False)[:, 0]
    return torch.clamp((1.0 - eps_x) * up + eps_x, min=eps_x, max=cap)


def decision_stats(p_cell, q, n_s, n_strata):
    late = n_strata - n_strata // 3   # late third of the strata (16 -> strata 11..15)
    return {'entropy_record': entropy_norm(p_cell), 'entropy_time': entropy_norm(q), 'ess_record': ess(p_cell), 'ess_time': ess(q),
            'late_alloc_frac': float(np.asarray(n_s)[late:].sum() / max(float(np.asarray(n_s).sum()), 1.0))}


def weight_stats(W):
    """W: [K, Z, X] -> max, mean, mass fraction in the top 10% pixels (uniform -> 0.10)."""
    flat = W.detach().float().flatten()
    k = max(int(round(0.1 * flat.numel())), 1)
    top = torch.topk(flat, k).values.sum()
    return {'space_wmax': float(flat.max()), 'space_wmean': float(flat.mean()), 'space_mass_top10': float(top / flat.sum())}
