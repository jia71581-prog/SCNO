#!/usr/bin/env python3
"""DG weak-form (SIPG) spatial residual for the acoustic PDE loss (2026-09-28).

WHAT THIS IS
------------
An OPT-IN alternative to the strong-form 4th-order FD Laplacian in physics_loss.py,
selected by `physics.space_form: dg_sipg` (default `strong_fd`, bitwise-old behaviour).
Only the SPACE discretisation changes; the time second derivative keeps the existing
continuous-query FD stencil (t-h, t, t+h) so the model's time contract is untouched.

DISCRETISATION
--------------
Regular 201x201 nodal grid = 200x200 cells.  With dg_element_cells = ec (default 4)
the mesh is (200/ec)^2 = 50x50 elements; each element owns a (ec+1)x(ec+1) = 5x5
patch of nodal predictions (patch edges shared with neighbours).  Per element the
nodal patch is least-squares projected onto the tensor Legendre basis Q_p
(p = dg_order, default 3 -> 16 orthonormal modal coefficients) with a FIXED,
precomputed pseudo-inverse; 25 samples over 16 coefficients is over-determined and
the projection reproduces any Q3 polynomial exactly.

Weak residual per element K and test mode phi_j (all integrals by fixed
Gauss-Legendre quadrature of dg_quadrature_order points per direction, exact for the
polynomial products that appear):

  R_K[j] =  \int_K (1/c^2) p_tt phi_j                                   (mass)
          + \int_K grad(p) . grad(phi_j)                                (stiffness)
          - sum_F \int_F {dp/dn} [phi_j]                                (flux / consistency)
          - sum_F \int_F {dphi_j/dn} [p]                                (symmetry / jump)
          + sum_F sigma \int_F [p] [phi_j]                              (penalty)

with F ranging over the INTERIOR faces of K only, jump [.] and average {.} the
standard SIPG operators with the face normal fixed from the minus to the plus
element, and sigma = dg_penalty_gamma * (p+1)^2 / h with h the face-perpendicular
element size.  p^-/p^+ traces and normal-derivative traces come from the two
adjacent elements' INDEPENDENT projections evaluated on the shared face.

BOUNDARIES AND SOURCE (mirrors the strong-form contract)
--------------------------------------------------------
* No DG boundary residual on the outer top/left/right/bottom faces: the top is a
  hard free surface handled by the model + BC loss, the three CPML sides are
  supervised by the data term (the auxiliary CPML state is not in the dataset).
  Because an element touching the outer boundary is missing that face's term, the
  whole outer ring of elements is marked INVALID and excluded from the loss
  (the weak analogue of the strong form's `halo`).  Their projections still exist
  and legitimately supply traces to the first interior ring.
* Elements whose cell-index footprint comes within `src_exclude_cells` of the
  source point are INVALID too (the delta forcing is not representable).

NORMALISATION
-------------
Per element/mode the five components above have physical dimensions and scale with
the 10 m/40 m grid factors; the residual is divided by the DETACHED per-element/mode
RMS of the five components (+ rel_floor) so the reduced loss is dimensionless and a
quiet element (no wave yet) cannot blow up.  Metrics report the RMS of each
normalised component over valid elements: dg_volume_rms (mass+stiffness),
dg_flux_rms, dg_jump_rms (symmetry term), dg_penalty_rms, plus valid_elements.

Everything precomputable is built once per (grid, element, order, quadrature,
device, dtype) key and cached; the runtime path is a handful of batched matmuls.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_OP_CACHE: dict = {}


def _legendre_basis(xs, nb):
    """Orthonormal Legendre values and derivatives on [-1,1] at points xs.

    Returns (V, D) each (len(xs), nb) float64; columns satisfy \int V_k V_l = delta_kl.
    """
    xs = np.asarray(xs, dtype=np.float64).reshape(-1)
    V = np.zeros((xs.size, nb))
    D = np.zeros((xs.size, nb))
    for k in range(nb):
        c = np.zeros(k + 1)
        c[k] = 1.0
        s = np.sqrt((2 * k + 1) / 2.0)
        V[:, k] = s * np.polynomial.legendre.legval(xs, c)
        dc = np.polynomial.legendre.legder(c)
        D[:, k] = s * np.polynomial.legendre.legval(xs, dc) if k > 0 else 0.0
    return V, D


def _bilinear_weights(q, ec):
    """Piecewise-linear interpolation weights from ec+1 equispaced nodes on [-1,1] to q."""
    s = (float(q) + 1.0) / 2.0 * ec
    c = min(int(np.floor(s)), ec - 1)
    c = max(c, 0)
    f = s - c
    w = np.zeros(ec + 1)
    w[c] = 1.0 - f
    w[c + 1] = f
    return w


def _face_mats(Tm, Tp, Dm, Dp, wf):
    """SIPG face-term matrices (each maps a 16-vector of modal coefs to a 16-vector
    of test-mode contributions).  Normal fixed minus -> plus; [v^-]=v^-, [v^+]=-v^+,
    {dv/dn}=0.5*dv/dn on either side.  Penalty matrices are returned WITHOUT sigma."""
    W = np.diag(wf)
    flux = {'mm': -0.5 * Tm.T @ W @ Dm, 'mp': -0.5 * Tm.T @ W @ Dp,
            'pm': +0.5 * Tp.T @ W @ Dm, 'pp': +0.5 * Tp.T @ W @ Dp}
    sym = {'mm': -0.5 * Dm.T @ W @ Tm, 'mp': +0.5 * Dm.T @ W @ Tp,
           'pm': -0.5 * Dp.T @ W @ Tm, 'pp': +0.5 * Dp.T @ W @ Tp}
    pen = {'mm': Tm.T @ W @ Tm, 'mp': -Tm.T @ W @ Tp,
           'pm': -Tp.T @ W @ Tm, 'pp': Tp.T @ W @ Tp}
    return flux, sym, pen


class DGOperator:
    """Precomputed SIPG operators for one (grid, element, order, quadrature) setup."""

    def __init__(self, Z, X, dx, dz, element_cells, order, quad_order, device, dtype):
        ec = int(element_cells)
        if (Z - 1) % ec or (X - 1) % ec:
            raise ValueError(f'grid {Z}x{X} nodes ({Z-1}x{X-1} cells) not divisible by '
                             f'dg_element_cells={ec}')
        self.Z, self.X = int(Z), int(X)
        self.ec = ec
        self.order = int(order)
        self.nb1 = self.order + 1
        self.nb = self.nb1 ** 2
        self.nn1 = ec + 1
        self.nn = self.nn1 ** 2
        self.Ez, self.Ex = (Z - 1) // ec, (X - 1) // ec
        self.dx, self.dz = float(dx), float(dz)
        self.hx, self.hz = ec * self.dx, ec * self.dz
        self.Jvol = 0.25 * self.hx * self.hz
        nq = int(quad_order)
        self.nq = nq
        if self.nn1 < self.nb1:
            raise ValueError('element_cells+1 nodal points per direction must be >= dg_order+1')

        # --- reference-element builds, all float64 ---
        xi_nodes = -1.0 + 2.0 * np.arange(self.nn1) / ec
        Vn, _ = _legendre_basis(xi_nodes, self.nb1)
        V2 = np.kron(Vn, Vn)                       # node a = az*nn1+ax, mode m = mz*nb1+mx
        proj = np.linalg.pinv(V2)                  # (nb, nn) fixed least-squares projection

        gq, gw = np.polynomial.legendre.leggauss(nq)
        Bq, Dq = _legendre_basis(gq, self.nb1)
        Phi = np.kron(Bq, Bq)                      # (nq^2, nb), row q = qz*nq+qx
        PhiD_x = np.kron(Bq, Dq)                   # d/dxi_x at volume quad points
        PhiD_z = np.kron(Dq, Bq)
        w2 = np.kron(gw, gw)

        M1 = Bq.T @ np.diag(gw) @ Bq               # ~identity (orthonormal basis)
        K1 = Dq.T @ np.diag(gw) @ Dq
        A_stiff = self.Jvol * ((4.0 / self.hx ** 2) * np.kron(M1, K1)
                               + (4.0 / self.hz ** 2) * np.kron(K1, M1))

        e_p, d_p = _legendre_basis([1.0], self.nb1)
        e_m, d_m = _legendre_basis([-1.0], self.nb1)

        # x-orientation faces (normal +x): minus = left element (its xi_x=+1 face),
        # plus = right element (xi_x=-1 face); face coordinate is xi_z.
        Tm = np.kron(Bq, e_p); Tp = np.kron(Bq, e_m)
        Dm = (2.0 / self.hx) * np.kron(Bq, d_p); Dp = (2.0 / self.hx) * np.kron(Bq, d_m)
        wf_x = 0.5 * self.hz * gw
        self._fx = _face_mats(Tm, Tp, Dm, Dp, wf_x)
        self._trace_x = (Tm, Tp)                   # kept (float64) for tests
        # z-orientation faces (normal +z): minus = top element (xi_z=+1 face).
        Tm = np.kron(e_p, Bq); Tp = np.kron(e_m, Bq)
        Dm = (2.0 / self.hz) * np.kron(d_p, Bq); Dp = (2.0 / self.hz) * np.kron(d_m, Bq)
        wf_z = 0.5 * self.hx * gw
        self._fz = _face_mats(Tm, Tp, Dm, Dp, wf_z)
        self._trace_z = (Tm, Tp)

        # nodal -> volume-quadrature bilinear interpolation (for the velocity field)
        Qv = np.zeros((nq * nq, self.nn))
        for a in range(nq):
            wzq = _bilinear_weights(gq[a], ec)
            for b in range(nq):
                Qv[a * nq + b] = np.kron(wzq, _bilinear_weights(gq[b], ec))

        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)
        self.proj = t(proj)
        self.Phi = t(Phi)
        self.PhiD_x, self.PhiD_z = t(PhiD_x), t(PhiD_z)
        self.w2 = t(w2)
        self.A_stiff = t(A_stiff)
        self.Qv = t(Qv)
        self.fx = {k: {s: t(m) for s, m in d.items()} for k, d in
                   zip(('flux', 'sym', 'pen'), self._fx)}
        self.fz = {k: {s: t(m) for s, m in d.items()} for k, d in
                   zip(('flux', 'sym', 'pen'), self._fz)}
        self.device, self.dtype = device, dtype

    # ------------------------------------------------------------------ projection
    def project(self, p):
        """[K,Z,X] nodal field -> [K,Ez,Ex,nb] modal coefficients (per-element LSQ)."""
        K = p.shape[0]
        v = F.unfold(p.reshape(K, 1, self.Z, self.X), kernel_size=self.nn1, stride=self.ec)
        u = torch.einsum('mn,kne->kem', self.proj, v)
        return u.reshape(K, self.Ez, self.Ex, self.nb)

    def _invc2_quad(self, velocity):
        """[Z,X] velocity -> [Ez,Ex,nq^2] of 1/c^2 at volume quadrature points."""
        v = F.unfold(velocity.reshape(1, 1, self.Z, self.X),
                     kernel_size=self.nn1, stride=self.ec)[0]      # [nn, E]
        vq = self.Qv @ v                                            # [nq^2, E]
        return (1.0 / (vq * vq)).T.reshape(self.Ez, self.Ex, -1)

    def _face_comp(self, u, mats, orient):
        """Accumulate one face component over all interior faces of one orientation."""
        if orient == 'x':
            uL, uR = u[:, :, :-1], u[:, :, 1:]
            rM = torch.matmul(uL, mats['mm'].T) + torch.matmul(uR, mats['mp'].T)
            rP = torch.matmul(uL, mats['pm'].T) + torch.matmul(uR, mats['pp'].T)
            return F.pad(rM, (0, 0, 0, 1)) + F.pad(rP, (0, 0, 1, 0))
        uT, uB = u[:, :-1], u[:, 1:]
        rM = torch.matmul(uT, mats['mm'].T) + torch.matmul(uB, mats['mp'].T)
        rP = torch.matmul(uT, mats['pm'].T) + torch.matmul(uB, mats['pp'].T)
        return F.pad(rM, (0, 0, 0, 0, 0, 1)) + F.pad(rP, (0, 0, 0, 0, 1, 0))

    # ------------------------------------------------------------------ residual
    def components(self, p_m, p_0, p_p, velocity, h, gamma):
        """The five weak-residual components, each [K,Ez,Ex,nb], graph-connected."""
        u0 = self.project(p_0)
        utt = (self.project(p_p) - 2.0 * u0 + self.project(p_m)) / (h * h)
        invc2 = self._invc2_quad(velocity.to(p_0.dtype))
        ptt_q = torch.matmul(utt, self.Phi.T)                       # [K,Ez,Ex,nq^2]
        mass = self.Jvol * torch.matmul(ptt_q * invc2 * self.w2, self.Phi)
        stiff = torch.matmul(u0, self.A_stiff.T)
        flux = self._face_comp(u0, self.fx['flux'], 'x') + self._face_comp(u0, self.fz['flux'], 'z')
        sym = self._face_comp(u0, self.fx['sym'], 'x') + self._face_comp(u0, self.fz['sym'], 'z')
        sig_x = float(gamma) * (self.order + 1) ** 2 / self.hx
        sig_z = float(gamma) * (self.order + 1) ** 2 / self.hz
        pen = (sig_x * self._face_comp(u0, self.fx['pen'], 'x')
               + sig_z * self._face_comp(u0, self.fz['pen'], 'z'))
        return {'mass': mass, 'stiff': stiff, 'flux': flux, 'sym': sym, 'pen': pen}

    def valid_mask(self, source_xz_m=None, x0_m=0.0, z0_m=0.0, src_exclude_cells=0):
        """[Ez,Ex] bool: outer ring of elements excluded (missing outer-face terms /
        CPML-adjacent / free-surface row), plus elements within src_exclude_cells of
        the source (cell-index distance from the element's index rectangle)."""
        m = torch.zeros((self.Ez, self.Ex), dtype=torch.bool, device=self.device)
        m[1:-1, 1:-1] = True
        if source_xz_m is not None and src_exclude_cells > 0:
            sx, sz = float(source_xz_m[0]), float(source_xz_m[1])
            ix = (sx - float(x0_m)) / self.dx
            iz = (sz - float(z0_m)) / self.dz
            lo_z = torch.arange(self.Ez, device=self.device, dtype=torch.float32) * self.ec
            lo_x = torch.arange(self.Ex, device=self.device, dtype=torch.float32) * self.ec
            dzd = torch.clamp(torch.maximum(lo_z - iz, iz - (lo_z + self.ec)), min=0.0)
            dxd = torch.clamp(torch.maximum(lo_x - ix, ix - (lo_x + self.ec)), min=0.0)
            d2 = dzd[:, None] ** 2 + dxd[None, :] ** 2
            m &= d2 > float(src_exclude_cells) ** 2
        return m


def get_dg_operator(Z, X, dx, dz, element_cells, order, quad_order, device, dtype):
    key = (int(Z), int(X), round(float(dx), 9), round(float(dz), 9), int(element_cells),
           int(order), int(quad_order), str(device), str(dtype))
    op = _OP_CACHE.get(key)
    if op is None:
        op = DGOperator(Z, X, dx, dz, element_cells, order, quad_order, device, dtype)
        _OP_CACHE[key] = op
    return op


def _rms(x, mask):
    return float(torch.sqrt(x[:, mask].detach().square().mean()))


def dg_loss_from_fields(p_m, p_0, p_p, velocity, cfg, dx, dz,
                        source_xz_m=None, x0_m=0.0, z0_m=0.0):
    """SIPG weak-form PDE loss from precomputed fields at (t-h, t, t+h).

    Drop-in replacement for the strong-form residual+mask+reduce block in
    compute_physics_loss: takes the same [K,Z,X] graph-connected fields, returns
    (loss scalar tensor, metrics dict).
    """
    Z, X = int(p_0.shape[-2]), int(p_0.shape[-1])
    p_m = p_m.reshape(-1, Z, X); p_0 = p_0.reshape(-1, Z, X); p_p = p_p.reshape(-1, Z, X)
    op = get_dg_operator(Z, X, dx, dz, cfg['dg_element_cells'], cfg['dg_order'],
                         cfg['dg_quadrature_order'], p_0.device, p_0.dtype)
    comps = op.components(p_m, p_0, p_p, velocity, float(cfg['dt_h_s']),
                          cfg['dg_penalty_gamma'])
    valid = op.valid_mask(source_xz_m, x0_m, z0_m, int(cfg.get('src_exclude_cells', 0)))
    n_valid = int(valid.sum())
    metrics = {'valid_elements': n_valid}
    if n_valid == 0:
        zero = torch.zeros((), device=p_0.device, dtype=p_0.dtype)
        metrics.update({'physics_dg': 0.0, 'dg_volume_rms': 0.0, 'dg_flux_rms': 0.0,
                        'dg_jump_rms': 0.0, 'dg_penalty_rms': 0.0})
        return zero, metrics
    r = comps['mass'] + comps['stiff'] + comps['flux'] + comps['sym'] + comps['pen']
    # detached per-element/mode component-RMS scale -> dimensionless residual
    stack = torch.stack([comps[k].detach() for k in ('mass', 'stiff', 'flux', 'sym', 'pen')])
    scale = torch.sqrt(stack.square().mean(0) + float(cfg.get('rel_floor', 1e-8)))
    rn = r / scale
    loss = rn[:, valid].square().mean()
    metrics['physics_dg'] = float(loss)
    metrics['dg_volume_rms'] = _rms((comps['mass'] + comps['stiff']) / scale, valid)
    metrics['dg_flux_rms'] = _rms(comps['flux'] / scale, valid)
    metrics['dg_jump_rms'] = _rms(comps['sym'] / scale, valid)
    metrics['dg_penalty_rms'] = _rms(comps['pen'] / scale, valid)
    return loss, metrics
