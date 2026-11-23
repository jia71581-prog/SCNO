"""V18-B fixed high-order numerical propagation prior (fixed_wave_physics branch).

Why this module exists
----------------------
The dense decoder receives the pressure-like coarse field but, before V18-B,
its only direct view of that field is a LEARNED 3x3 lift (``coarse_lift``).
The dataset generator, however, evolves pressure with a FIXED eighth-order
centred spatial operator (LWC-84, space order 8): the discrete increment that
takes p_n to p_{n+1} is built from ``c^2 Laplacian_8(p)``.  This branch hands
the network exactly that fixed discrete structure -- the generator's own
stencil applied to the generator's own state variable -- as extra input
features, instead of hoping the learned lift rediscovers a 9-point 1D stencil
whose extreme weights span three orders of magnitude (8/5 vs 1/560).

It is a fixed FEATURE prior inside the main network, not a refiner and not a
time integrator: no second network, no FDM time stepping, no extra forward of
the field, and no new loss term.  Training remains Data + PDE + IC.

Design commitments
------------------
* Generator-consistent coefficients.  The second-derivative weights are
  IMPORTED from ``pde_residual._W8`` (the LWC-84 spatial stencil, identical to
  ``lwc84_training.W8``), so there is a single source of truth; the analytic
  sine test asserts eighth-order convergence against the continuum operator.
* Dimensionless features on the production grid.  The saved grid is 201x201
  with dx = dz = 10 m (square cells), so ``dx^2 * Laplacian_8`` is exactly the
  raw two-axis stencil sum with NO grid constant, and the optional high-order
  gradients are per-cell (``dx * d/dx``), matching how glocal.py scales its
  ``d(log c)`` channels.  No c^2 factor is applied here: the medium already
  enters the same pre-activation sum through the FiLM conditioning and the
  V18-A velocity side features, so the fused blocks can form any stable
  dimensionless combination without this branch hard-coding one.
* CPML-aware validity halo.  The three-sided CPML lives OUTSIDE the physical
  domain and its auxiliary memory is not present in these tensors, and the
  free-surface ghost extension is not reproduced here, so the fixed stencil is
  only meaningful >= 4 cells inside every edge (same radius-4 rule as
  ``pde_residual.laplacian8``).  The stencils are applied as VALID
  convolutions and the missing 4-cell halo is zero-filled: no left/right/
  bottom Dirichlet, Neumann or Sommerfeld condition is invented, and the
  field itself is never modified -- only the derived features vanish where
  they would be wrong.
* Pressure-like input only.  The branch reads the coarse field the decoder
  already receives -- the quantity the physics loss and the labels live on --
  never a hidden channel reinterpreted as pressure, never a family label,
  never an offline cache.
* Zero-initialised projection, deterministic construction.  The stencils are
  non-learnable buffers (``persistent=False``: no state_dict keys), the only
  parameters are one 1x1 projection whose weight and bias start at zero, and
  construction runs under a forked RNG.  A build with the branch off is
  bitwise identical in every shared tensor, and a V18-A checkpoint
  transplanted into an enabled build reproduces its step-0 forward exactly.
  The projection term is LINEAR in its own weights with the external fixed
  features as the other factor, so the zero init is not a gradient deadlock
  (same argument as the v17 carrier lift).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..pde_residual import _W8

# Stencil radius: features are valid only >= HALO cells from every edge.
HALO = 4

# Eighth-order centred FIRST-derivative weights (offsets -4..+4), the standard
# central-difference companion of _W8 at the same order.  Used only for the
# optional per-cell gradient channels; the propagation term itself uses _W8.
_D8 = (1.0 / 280.0, -4.0 / 105.0, 1.0 / 5.0, -4.0 / 5.0,
       0.0,
       4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0)

# Channel layout of the feature block: [coarse, dx^2*lap8, dz*d/dz, dx*d/dx].
FIXED_PHYSICS_CHANNELS = 4
_STENCIL_CHANNELS = FIXED_PHYSICS_CHANNELS - 1


def fixed_physics_stencils(dtype: torch.dtype) -> torch.Tensor:
    """The three fixed 9x9 stencils as a [3, 1, 9, 9] conv weight.

    0. ``dx^2 * Laplacian_8`` for square cells (dx = dz): the _W8 row plus the
       _W8 column (centre weight counted on both axes, as the generator does);
    1. ``dz * d/dz`` (axis -2);
    2. ``dx * d/dx`` (axis -1).

    All three are dimensionless multiples of the field, so no grid constant
    enters the module.  Deterministic: no RNG draws.
    """
    size = 2 * HALO + 1
    w8 = torch.tensor(_W8, dtype=dtype)
    d8 = torch.tensor(_D8, dtype=dtype)
    kernels = torch.zeros(_STENCIL_CHANNELS, 1, size, size, dtype=dtype)
    kernels[0, 0, HALO, :] += w8
    kernels[0, 0, :, HALO] += w8
    kernels[1, 0, :, HALO] = d8
    kernels[2, 0, HALO, :] = d8
    return kernels


def fixed_physics_features(
    coarse: torch.Tensor, stencils: torch.Tensor | None = None
) -> torch.Tensor:
    """[batch, 1, z, x] coarse frames -> [batch, FIXED_PHYSICS_CHANNELS, z, x].

    Stencil channels are computed as VALID convolutions and zero-padded back to
    the full grid, so the 4-cell halo where the fixed stencil would read CPML/
    ghost territory carries exactly 0.  The value channel is the coarse field
    itself, untouched everywhere.
    """
    if coarse.ndim != 4 or coarse.shape[1] != 1:
        raise ValueError(
            f"fixed physics features expect [batch,1,z,x], got {tuple(coarse.shape)}")
    if coarse.shape[-2] < 2 * HALO + 1 or coarse.shape[-1] < 2 * HALO + 1:
        raise ValueError("grid too small for the eighth-order stencil")
    if stencils is None:
        stencils = fixed_physics_stencils(coarse.dtype)
    interior = F.conv2d(coarse, stencils)
    padded = F.pad(interior, (HALO, HALO, HALO, HALO))
    return torch.cat((coarse, padded), dim=1)


class FixedWavePhysicsBranch(nn.Module):
    """Fixed eighth-order features of the coarse field, zero-init projected.

    Forward takes the decoder's flattened coarse frames ``[batch, 1, z, x]``
    and returns a ``[batch, width, z, x]`` additive term for the decoder's
    input hidden state (alongside ``coarse_lift``), where it is fused with the
    FiLM medium conditioning and, through the spectral blocks, with the V18-A
    local/side features.  The single learnable tensor pair is the 1x1
    projection; the stencils are fixed buffers excluded from the state_dict.
    """

    def __init__(self, *, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = int(width)
        self.register_buffer(
            "stencils", fixed_physics_stencils(torch.float32), persistent=False
        )
        # RNG-neutral construction, same rationale as glocal.py: nn.Conv2d
        # draws random numbers in __init__ even though both tensors are
        # overwritten with zeros below.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Conv2d(FIXED_PHYSICS_CHANNELS, self.width, kernel_size=1)
            with torch.no_grad():
                self.projection.weight.zero_()
                self.projection.bias.zero_()

    def features(self, coarse: torch.Tensor) -> torch.Tensor:
        return fixed_physics_features(coarse, self.stencils.to(coarse.dtype))

    def forward(self, coarse: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(coarse))


__all__ = [
    "FIXED_PHYSICS_CHANNELS",
    "HALO",
    "FixedWavePhysicsBranch",
    "fixed_physics_features",
    "fixed_physics_stencils",
]
