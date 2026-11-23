"""Physics and sampling for a newly preregistered homogeneous ablation.

No reference fields beyond the supplied early normalization enter this module.
The sampling defaults belong to this new ablation, not a recovered old recipe.
Training can use sampled tiles; stopping counts the complete supplied grid.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch
from torch.nn import functional as F

_COEFFICIENTS = (-205/72, 8/5, -1/5, 8/315, -1/560)


def _positive(value, name):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _field(value):
    result = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if result.ndim not in (2, 3) or result.is_complex():
        raise ValueError("Expected real [z,x] or [time,z,x] fields")
    return result.to(dtype=torch.float64)


def laplacian8(field, h=10.):
    """Eighth-order centered Laplacian, zero padding outside the supplied grid.

    The physical residual must exclude an eight-node halo after composition.
    Conversion to FP64 preserves the input's gradient connection.
    """
    h = _positive(h, "h")
    field = _field(field)
    a = _COEFFICIENTS
    stencil = field.new_tensor([a[4], a[3], a[2], a[1], a[0],
                                a[1], a[2], a[3], a[4]])
    flat = field.reshape(-1, 1, *field.shape[-2:])
    xx = F.conv2d(flat, stencil.reshape(1, 1, 1, 9), padding=(0, 4))
    zz = F.conv2d(flat, stencil.reshape(1, 1, 9, 1), padding=(4, 0))
    return ((xx+zz)/h**2).reshape_as(field)


def lwc_residual(pminus, p0, pplus, velocity, h=10., delta=.0005):
    """Return Dtt(p)-L(p)-delta² L(L(p))/12 with L=c² lap_h."""
    delta = _positive(delta, "delta")
    h = _positive(h, "h")
    minus, middle, plus = (_field(x) for x in (pminus, p0, pplus))
    if minus.shape != middle.shape or plus.shape != middle.shape:
        raise ValueError("The three pressure queries must have identical shapes")
    if minus.device != middle.device or plus.device != middle.device:
        raise ValueError("Pressure queries must be on one device")
    c = torch.as_tensor(velocity, device=middle.device, dtype=torch.float64)
    if c.shape != middle.shape[-2:] or not torch.isfinite(c).all() or not (c > 0).all():
        raise ValueError("Velocity must be finite, positive and match the spatial grid")
    c2 = c.square()
    first = c2*laplacian8(middle, h)
    second = c2*laplacian8(first, h)
    return (plus-2*middle+minus)/delta**2-first-(delta**2/12)*second


def make_valid_mask(shape=(201, 201), h=10., source=(1000., 100.),
                    halo=8, radius=90., device=None):
    """Physical mask; source is (x,z), while tensor rows index z."""
    h = _positive(h, "h")
    if len(shape) != 2 or any(int(n) != n or n < 1 for n in shape):
        raise ValueError("A positive two-dimensional grid shape is required")
    if not isinstance(halo, (int, np.integer)) or halo < 0:
        raise ValueError("Halo must be a nonnegative integer")
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("Source exclusion radius must be finite and nonnegative")
    if len(source) != 2 or not all(math.isfinite(float(x)) for x in source):
        raise ValueError("Source must contain finite x,z coordinates")
    nz, nx = map(int, shape)
    z, x = torch.meshgrid(torch.arange(nz, device=device, dtype=torch.float64)*h,
                          torch.arange(nx, device=device, dtype=torch.float64)*h,
                          indexing="ij")
    mask = (x-float(source[0])).square()+(z-float(source[1])).square() > radius**2
    if halo:
        mask[:halo] = False; mask[-halo:] = False
        mask[:, :halo] = False; mask[:, -halo:] = False
    if not mask.any():
        raise ValueError("The halo and source exclusion leave no valid nodes")
    return mask


def residual_scale_squared(early_normalized, f0=15., eps=1e-24):
    """Fixed nu²=omega⁴ mean(early pressure²), using early labels only."""
    f0 = _positive(f0, "f0")
    eps = _positive(eps, "eps")
    early = torch.as_tensor(early_normalized).detach()
    if early.is_complex():
        raise ValueError("Real early reference values are required")
    early = early.to(dtype=torch.float64)
    if not early.numel() or not torch.isfinite(early).all():
        raise ValueError("Finite nonempty early reference values are required")
    result = ((2*math.pi*f0)**4*early.square().mean()).clamp_min(eps)
    if not torch.isfinite(result):
        raise ValueError("Early residual normalization overflowed")
    return result


def _normalizer(value, reference):
    complex_value = value.is_complex() if isinstance(value, torch.Tensor) else np.iscomplexobj(value)
    if complex_value:
        raise ValueError("nu_squared must be real")
    result = torch.as_tensor(value, device=reference.device, dtype=torch.float64).detach()
    if result.numel() != 1 or not torch.isfinite(result).all() or not (result > 0).all():
        raise ValueError("nu_squared must be a finite positive scalar")
    return result.reshape(())


def full_grid_diagnostics(residual, mask, f0=15., epsilon_r=.25,
                          epsilon_p=.01, nu_squared=None):
    """Exact failure fraction on every supplied valid residual unit."""
    f0 = _positive(f0, "f0")
    epsilon_r = _positive(epsilon_r, "epsilon_r")
    epsilon_p = _positive(epsilon_p, "epsilon_p")
    if epsilon_p >= 1:
        raise ValueError("epsilon_p must lie in (0,1)")
    with torch.no_grad():
        r = _field(residual)
        if r.ndim == 2: r = r[None]
        mask = torch.as_tensor(mask, device=r.device, dtype=torch.bool)
        if mask.shape != r.shape[-2:] or not mask.any() or not r.shape[0]:
            raise ValueError("A nonempty matching valid-node mask is required")
        values = r[:, mask]
        if not torch.isfinite(values).all():
            raise ValueError("Valid residual entries must remain finite")
        rho = values.abs()/(2*math.pi*f0)**2
        count = int((rho > epsilon_r).sum().item())
        units = int(rho.numel())
        probability = count/units
        result = {
            "failure_probability": probability,
            "failure_count": count,
            "units": units,
            "threshold_satisfied": probability < epsilon_p,
            "rho_max": float(rho.max()),
            "rho_mean_square": float(rho.square().mean()),
            "residual_mean_square": float(values.square().mean()),
        }
        if nu_squared is not None:
            result["pde_mean"] = float(values.square().mean()/_normalizer(nu_squared, r))
        return result


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if np.iscomplexobj(value):
        raise ValueError("Real sampling energies are required")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Sampling energies must be finite and nonnegative")
    return result


class AISSampler:
    """One-window, node-weighted time/tile sampling with exact HT weights."""

    def __init__(self, mask, n_times=400, tile_size=16,
                 beta_uniform=.1, ema=.9, seed=0):
        mask = torch.as_tensor(mask, dtype=torch.bool, device="cpu").detach().clone()
        if mask.ndim != 2 or not mask.any():
            raise ValueError("A nonempty two-dimensional valid mask is required")
        if not isinstance(n_times, (int, np.integer)) or n_times < 1:
            raise ValueError("n_times must be a positive integer")
        if not isinstance(tile_size, (int, np.integer)) or tile_size < 1:
            raise ValueError("tile_size must be a positive integer")
        if not math.isfinite(beta_uniform) or not 0 < beta_uniform <= 1:
            raise ValueError("beta_uniform must lie in (0,1]")
        if not math.isfinite(ema) or not 0 <= ema < 1:
            raise ValueError("ema must lie in [0,1)")
        self.mask = mask
        self.n_times = int(n_times)
        self.tile_size = int(tile_size)
        self.beta_uniform = float(beta_uniform)
        self.ema = float(ema)
        self.n_valid = int(mask.sum())
        self.tile_bounds = []
        counts = []
        nz, nx = mask.shape
        for z0 in range(0, nz, self.tile_size):
            for x0 in range(0, nx, self.tile_size):
                z1, x1 = min(z0+self.tile_size, nz), min(x0+self.tile_size, nx)
                count = int(mask[z0:z1, x0:x1].sum())
                if count:
                    self.tile_bounds.append((z0, z1, x0, x1))
                    counts.append(count)
        self.n_nodes = np.asarray(counts, dtype=np.int64)
        self._n_tile_z = (nz+self.tile_size-1)//self.tile_size
        self._n_tile_x = (nx+self.tile_size-1)//self.tile_size
        self._tile_ids = torch.tensor([
            (z0//self.tile_size)*self._n_tile_x+x0//self.tile_size
            for z0, _, x0, _ in self.tile_bounds
        ], dtype=torch.long)
        self._tile_counts = torch.as_tensor(self.n_nodes.copy(), dtype=torch.float64)
        self._geometry_cache = {}
        self.history = np.zeros((self.n_times, len(counts)), dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def tile_energies(self, residual, nu_squared):
        """Differentiable per-tile means of R²/nu² for every supplied time."""
        residual = _field(residual)
        if residual.ndim == 2: residual = residual[None]
        if residual.shape[-2:] != self.mask.shape:
            raise ValueError("Residual spatial grid must match the sampler mask")
        scale = _normalizer(nu_squared, residual)
        device = residual.device
        if device not in self._geometry_cache:
            self._geometry_cache[device] = (
                self.mask.to(device=device),
                self._tile_ids.to(device=device),
                self._tile_counts.to(device=device),
            )
        mask, ids, counts = self._geometry_cache[device]
        # Mask before squaring: excluded NaN entries must neither contaminate
        # the sum nor create NaN gradients through zero-times-NaN products.
        squares = torch.where(mask[None], residual, 0.).square()
        pad_z = self._n_tile_z*self.tile_size-residual.shape[-2]
        pad_x = self._n_tile_x*self.tile_size-residual.shape[-1]
        squares = F.pad(squares, (0, pad_x, 0, pad_z))
        sums = squares.reshape(residual.shape[0], self._n_tile_z, self.tile_size,
                               self._n_tile_x, self.tile_size).sum(dim=(2, 4))
        selected = sums.flatten(1).index_select(1, ids)
        return selected/counts[None]/scale

    def initialize_history(self, energies):
        energies = _numpy(energies)
        if energies.shape != self.history.shape:
            raise ValueError("Initial history must cover every time and valid tile")
        self.history = energies.copy()

    def _q(self, mode):
        uniform = np.broadcast_to(
            self.n_nodes[None]/(self.n_times*self.n_valid), self.history.shape).copy()
        if mode == "uniform":
            return uniform
        if mode != "ais":
            raise ValueError("mode must be uniform or ais")
        scores = self.n_nodes[None]*np.sqrt(self.history)
        total = float(scores.sum())
        if total == 0:
            return uniform
        q = self.beta_uniform*uniform+(1-self.beta_uniform)*(scores/total)
        return q

    def draw(self, mode="ais", n_time_draws=8, n_tile_draws=8):
        for value in (n_time_draws, n_tile_draws):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("Draw counts must be positive integers")
        n_time_draws, n_tile_draws = int(n_time_draws), int(n_tile_draws)
        state_before = deepcopy(self.rng.bit_generator.state)
        q = self._q(mode)
        q_time = q.sum(axis=1)
        times = self.rng.choice(self.n_times, size=n_time_draws, replace=True, p=q_time)
        tiles = np.empty((n_time_draws, n_tile_draws), dtype=np.int64)
        counts = np.zeros(q.shape, dtype=np.int64)
        for j, t in enumerate(times):
            tiles[j] = self.rng.choice(len(self.n_nodes), size=n_tile_draws,
                                      replace=True, p=q[t]/q_time[t])
            np.add.at(counts[t], tiles[j], 1)
        unique = np.unique(times)
        target = self.n_nodes[None]/(self.n_times*self.n_valid)
        coefficients = counts[unique]*target/(n_time_draws*n_tile_draws*q[unique])
        return {
            "mode": mode,
            "time_indices": unique,
            "matrix_coeffs": coefficients,
            "q_joint": q,
            "time_draws": times,
            "tile_draws": tiles,
            "pair_counts": counts,
            "expected_counts": n_time_draws*n_tile_draws*q,
            "n_time_draws": n_time_draws,
            "n_tile_draws": n_tile_draws,
            "rng_state_before": state_before,
            "rng_state_after": deepcopy(self.rng.bit_generator.state),
        }

    def loss(self, energies, plan):
        if not isinstance(energies, torch.Tensor):
            energies = torch.as_tensor(energies, dtype=torch.float64)
        coefficients = torch.as_tensor(plan["matrix_coeffs"], device=energies.device,
                                        dtype=torch.float64)
        if energies.shape != coefficients.shape:
            raise ValueError("Energies must contain each distinct selected time and all valid tiles")
        return (energies.to(dtype=torch.float64)*coefficients).sum()

    def commit(self, plan, energies):
        times = np.asarray(plan["time_indices"], dtype=np.int64)
        values = _numpy(energies)
        if values.shape != (len(times), len(self.n_nodes)):
            raise ValueError("Commit energies must cover all tiles at each selected distinct time")
        if len(np.unique(times)) != len(times) or np.any(times < 0) or np.any(times >= self.n_times):
            raise ValueError("Commit times must be unique valid indices")
        self.history[times] = self.ema*self.history[times]+(1-self.ema)*values

    def state_dict(self):
        return {
            "history": self.history.copy(),
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "n_times": self.n_times,
            "tile_size": self.tile_size,
            "beta_uniform": self.beta_uniform,
            "ema": self.ema,
            "n_nodes": self.n_nodes.copy(),
        }

    def load_state_dict(self, state):
        for name in ("n_times", "tile_size", "beta_uniform", "ema"):
            if state[name] != getattr(self, name):
                raise ValueError(f"Sampler restore configuration differs for {name}")
        if not np.array_equal(state["n_nodes"], self.n_nodes):
            raise ValueError("Sampler restore mask/tile geometry differs")
        self.initialize_history(state["history"])
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
