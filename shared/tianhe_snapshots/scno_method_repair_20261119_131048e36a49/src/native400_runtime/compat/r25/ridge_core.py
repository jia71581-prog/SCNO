"""Support-only differentiable FP64 ridge; query VJPs are replayed once."""
import torch
from torch import nn
from torch.nn import functional as F


class RidgePrior(nn.Module):
    def __init__(self, dimension=16, initial_ridge=.01):
        super().__init__()
        self.a0 = nn.Parameter(torch.zeros(dimension, dtype=torch.float64))
        self.raw_ridge = nn.Parameter(torch.log(torch.expm1(torch.tensor(initial_ridge, dtype=torch.float64))))

    def strength(self):
        return F.softplus(self.raw_ridge) + 1e-8


def solve_support(gram, rhs, count, known_energy, prior):
    """gram/rhs contain ONLY actual known-observation forward residuals."""
    if gram.shape != (16,16) or rhs.shape != (16,) or count <= 0:
        raise ValueError('16-dimensional support normal equations')
    if not torch.isfinite(gram).all() or not torch.isfinite(rhs).all():
        raise FloatingPointError('nonfinite support equations')
    if not torch.isfinite(known_energy):
        raise FloatingPointError('invalid known-only support energy')
    if float(known_energy.detach()) <= 0:
        return prior.a0*0, dict(hold=True,support_energy=0.,support_samples=count,
                               numerical_rank=0,no_future_statistics=True)
    u, b = gram.double()/count, rhs.double()/count
    # This scale is derived from known support feature columns, never query GT.
    diagonal = torch.diagonal(u)
    scale = torch.where(diagonal>1e-8,torch.sqrt(diagonal+1e-8),torch.ones_like(diagonal))
    g, d = u/(scale[:,None]*scale[None,:]), b/scale
    g = .5*(g+g.T)
    ridge = prior.strength()
    matrix = g + ridge*torch.eye(16, device=g.device, dtype=g.dtype)
    normalized = torch.linalg.solve(matrix, d + ridge*prior.a0)
    coefficient = normalized/scale
    if not torch.isfinite(coefficient).all():
        raise FloatingPointError('nonfinite ridge solution')
    residual = torch.linalg.vector_norm(matrix@normalized-(d+ridge*prior.a0))
    return coefficient, dict(hold=False,ridge=float(ridge.detach()),condition=float(torch.linalg.cond(matrix.detach())),
                            numerical_rank=int(torch.linalg.matrix_rank(g.detach())),
                            solve_residual=float(residual.detach()),support_samples=count,
                            support_energy=float(known_energy.detach()),no_future_statistics=True)


def replay_meta(coefficient, query_leaf):
    if query_leaf.grad is None or not torch.isfinite(query_leaf.grad).all():
        raise FloatingPointError('missing/nonfinite outer coefficient VJP')
    coefficient.backward(query_leaf.grad)


def detached_query_leaf(coefficient):
    return coefficient.detach().clone().requires_grad_(True)
