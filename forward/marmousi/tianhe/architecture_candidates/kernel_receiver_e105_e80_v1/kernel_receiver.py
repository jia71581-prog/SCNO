"""A differentiable fixed-source B5 residual interpolator on physical velocities."""

from math import isfinite, sqrt

import torch
from torch import nn
from torch.nn import functional as F


BASIS_SIZE = 163
RECEIVERS = 201
FRAMES = 801
GRID = (201, 201)
DESCRIPTOR_SIZE = 51 * 51


def log_velocity_descriptor(velocity):
    """Average log speed over 5×5 cells and evaluate the descriptor every 40 m."""
    if velocity.ndim == 4 and velocity.shape[1] == 1:
        velocity = velocity[:, 0]
    if velocity.ndim != 3 or tuple(velocity.shape[-2:]) != GRID:
        raise ValueError("Expected 201 × 201 physical velocity grids in metres per second")
    if not (torch.isfinite(velocity) & (velocity > 0.)).all():
        raise ValueError("All physical velocity samples must be finite and positive")
    log_speed = (velocity.to(dtype=torch.float64) / 2000.).log()
    smoothed = F.avg_pool2d(log_speed[:, None], 5, stride=4, padding=2, count_include_pad=False)
    return smoothed.flatten(1) / sqrt(DESCRIPTOR_SIZE)


def squared_distances(query, reference):
    query, reference = query.double(), reference.double()
    norms = query.square().sum(1, keepdim=True) + reference.square().sum(1)
    return torch.addmm(norms, query, reference.mT, alpha=-2.).clamp_min_(0.)


def gaussian(distance_squared, length):
    return torch.exp(distance_squared * (-.5 / length**2))


def block_factor(distance_squared, target, energy, length, ridge, median_energy):
    """Solve weighted KRR using whitened duals with compact FP32 stored targets.

    With d_i the frozen full-shot energy floored in the caller, ridge adds
    λ d_i / median(d) to the Gram diagonal. QR supplies the original temporal
    weights, so the data term measures the original shot-normalized metric.
    """
    kernel = gaussian((distance_squared + distance_squared.mT) * .5, length)
    kernel.diagonal().fill_(1.)
    diagonal = ridge * energy.double() / median_energy
    system = kernel.clone()
    system.diagonal().add_(diagonal)
    factor, info = torch.linalg.cholesky_ex(system)
    jitter = 0.
    if info.item():
        jitter = 1e-10 * system.diagonal().max().item()
        system.diagonal().add_(jitter)
        factor = torch.linalg.cholesky(system)
    eigenvalues = torch.linalg.eigvalsh(system)
    if not (torch.isfinite(eigenvalues).all() and eigenvalues[0] > 0.):
        raise ValueError("Regularized Gaussian source Gram is not positive definite")
    whitened = torch.linalg.solve_triangular(factor, target.flatten(1).double(), upper=False).float()
    if not torch.isfinite(whitened).all():
        raise ValueError("Nonfinite receiver coefficients after Cholesky")
    return factor, whitened, {"condition": float(eigenvalues[-1] / eigenvalues[0]), "jitter": jitter}


def evaluate_block(distance_squared, factor, whitened, length):
    query = gaussian(distance_squared, length)
    whitened_query = torch.linalg.solve_triangular(factor, query.mT.contiguous(), upper=False)
    values = whitened_query.mT @ whitened.double()
    return values.reshape(-1, BASIS_SIZE, RECEIVERS)


def decode_frames(decode_rows, coefficients):
    return decode_rows.to(device=coefficients.device, dtype=torch.float32) @ coefficients.float()


class SourceFactor(nn.Module):
    def __init__(self, descriptor_rows, factor, whitened):
        super().__init__()
        self.register_buffer("descriptor_rows", descriptor_rows.long())
        self.register_buffer("factor", factor.double())
        self.register_buffer("whitened", whitened.float())


class KernelB5Residual(nn.Module):
    """Map physical velocities and the registered physical source tuples to B5 traces."""

    def __init__(self, descriptors, sources, blocks, decode_rows, length):
        super().__init__()
        if (descriptors.ndim != 2 or descriptors.shape[1] != DESCRIPTOR_SIZE
                or len(blocks) != len(sources) or sources.shape[1:] != (5,)
                or decode_rows.shape != (FRAMES, BASIS_SIZE) or not isfinite(length) or length <= 0.):
            raise ValueError("Invalid kernel receiver geometry or coefficients")
        self.length = length
        self.register_buffer("train_descriptors", descriptors.double())
        self.register_buffer("physical_sources", sources.float())
        self.register_buffer("decode_rows", decode_rows.float())
        self.groups = nn.ModuleList(SourceFactor(*block) for block in blocks)

    def forward(self, velocity, source, frame_indices=None):
        descriptor = log_velocity_descriptor(velocity)
        if source.shape != (len(descriptor), 5):
            raise ValueError("Supply one physical (x, z, f0, delay, amplitude) tuple per velocity")
        matches = (source.float()[:, None] == self.physical_sources[None, :]).all(-1)
        if not torch.all(matches.sum(1) == 1):
            raise ValueError("The kernel comparator only supports the registered physical sources")
        rows = self.decode_rows
        if frame_indices is not None:
            indices = torch.as_tensor(frame_indices, device=rows.device)
            if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64) or (indices < 0).any() or (indices >= FRAMES).any():
                raise ValueError("Use one-dimensional registered time indices")
            rows = rows[indices]
        prediction = rows.new_zeros(len(descriptor), len(rows), RECEIVERS)
        for column in matches.any(0).nonzero().flatten().tolist():
            samples = matches[:, column].nonzero().flatten()
            group = self.groups[column]
            distances = squared_distances(descriptor[samples], self.train_descriptors[group.descriptor_rows])
            result = decode_frames(rows, evaluate_block(distances, group.factor, group.whitened, self.length))
            prediction.index_copy_(0, samples, result)
        return prediction


def load_kernel_residual(path, *, device="cpu"):
    artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if artifact["binding"]["schema"] != "marmousi_kernel_b5_residual_v1":
        raise ValueError("Not a committed physical-source kernel B5 comparator")
    return KernelB5Residual(artifact["descriptors"], artifact["sources"], artifact["blocks"],
                            artifact["decode_rows"], artifact["binding"]["length"]).to(device).eval()
