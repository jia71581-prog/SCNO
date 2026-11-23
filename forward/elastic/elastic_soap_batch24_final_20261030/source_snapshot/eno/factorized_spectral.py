"""Channel Tucker spectral convolution with full 3-D frequency cores.

Real, shared channel factors commute with the FFT. Applying them in physical
space makes the FFT operate on the reduced channels, without reconstructing W.
This is a channel-only Tucker parameterization, not axis-separable F-FNO.
"""
import math

import torch
from torch import nn


class ChannelTuckerSpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, D1, D2, D3,
                 modes1, modes2, modes3, rank_fraction=.25, min_rank=8):
        super().__init__()
        if not 0 < rank_fraction <= 1 or min_rank < 1:
            raise ValueError('rank_fraction must be in (0,1] and min_rank positive')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d1, self.d2, self.d3 = D1, D2, D3
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.rank_in = min(in_channels, max(min_rank, math.ceil(in_channels * rank_fraction)))
        self.rank_out = min(out_channels, max(min_rank, math.ceil(out_channels * rank_fraction)))
        self.input_factor = nn.Parameter(torch.empty(in_channels, self.rank_in))
        self.output_factor = nn.Parameter(torch.empty(out_channels, self.rank_out))
        nn.init.orthogonal_(self.input_factor)
        nn.init.orthogonal_(self.output_factor)
        scale = (1 / (2 * self.rank_in)) ** .5
        self.cores = nn.ParameterList([
            nn.Parameter(scale * torch.rand(self.rank_in, self.rank_out, modes1, modes2, modes3,
                                            dtype=torch.complex64)) for _ in range(4)
        ])

    def forward(self, x, D1=None, D2=None, D3=None):
        dims = (self.d1, self.d2, self.d3) if D1 is None else (D1, D2, D3)
        reduced = torch.einsum('bixyt,ir->brxyt', x, self.input_factor)
        spectrum = torch.fft.rfftn(reduced, dim=(-3, -2, -1), norm='forward')
        output = spectrum.new_zeros(x.shape[0], self.rank_out, dims[0], dims[1], dims[2] // 2 + 1)
        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        if min(x.shape[-3], dims[0]) < 2*m1 or min(x.shape[-2], dims[1]) < 2*m2:
            raise ValueError('spatial Fourier quadrants overlap')
        if min(x.shape[-1]//2 + 1, dims[2]//2 + 1) < m3:
            raise ValueError('too many temporal Fourier modes')
        for core, (s1, s2) in zip(self.cores, [
                (slice(0, m1), slice(0, m2)), (slice(-m1, None), slice(0, m2)),
                (slice(0, m1), slice(-m2, None)), (slice(-m1, None), slice(-m2, None))]):
            selected = spectrum[:, :, s1, s2, :m3]
            output[:, :, s1, s2, :m3] = torch.einsum('bixyt,ioxyt->boxyt', selected, core)
        reduced_output = torch.fft.irfftn(output, s=dims, norm='forward')
        return torch.einsum('brxyt,or->boxyt', reduced_output, self.output_factor).contiguous()

    @classmethod
    @torch.no_grad()
    def from_dense(cls, dense, rank_fraction=.25, min_rank=8, chunk_modes=2048,
                   input_basis=None, output_basis=None):
        """Real channel HOSVD over all four quadrants, without data or labels.

        A full-rank projection preserves the dense operator and its input
        derivative. Reduced ranks are an initialization approximation.
        """
        weights = [getattr(dense, f'weights{i}') for i in range(1, 5)]
        base = weights[0]
        real_dtype = base.real.dtype
        result = cls(dense.in_channels, dense.out_channels, dense.d1, dense.d2, dense.d3,
                     dense.modes1, dense.modes2, dense.modes3, rank_fraction, min_rank)
        result.to(device=base.device)
        result.input_factor.data = result.input_factor.data.to(real_dtype)
        result.output_factor.data = result.output_factor.data.to(real_dtype)
        for core in result.cores:
            core.data = core.data.to(base.dtype)
        ci, co = dense.in_channels, dense.out_channels
        if (input_basis is None) != (output_basis is None):
            raise ValueError('provide both channel bases or neither')
        gram_in = torch.zeros(ci, ci, device=base.device, dtype=torch.float64)
        gram_out = torch.zeros(co, co, device=base.device, dtype=torch.float64)
        total_energy = 0.
        for weight in weights:
            flat = weight.flatten(2)
            for first in range(0, flat.shape[-1], chunk_modes):
                part = flat[:, :, first:first+chunk_modes]
                if input_basis is None:
                    # FP64 accumulation; each BLAS product retains source precision.
                    gram_in.add_(torch.einsum('iok,jok->ij', part, part.conj()).real.double())
                    gram_out.add_(torch.einsum('iok,ipk->op', part, part.conj()).real.double())
                total_energy += float(part.abs().square().sum(dtype=torch.float64))
        ain = (torch.linalg.eigh(gram_in)[1][:, -result.rank_in:] if input_basis is None
               else input_basis[:, -result.rank_in:]).to(device=base.device, dtype=real_dtype)
        aout = (torch.linalg.eigh(gram_out)[1][:, -result.rank_out:] if output_basis is None
                else output_basis[:, -result.rank_out:]).to(device=base.device, dtype=real_dtype)
        result.input_factor.copy_(ain)
        result.output_factor.copy_(aout)
        retained = 0.
        for target, weight in zip(result.cores, weights):
            flat = weight.flatten(2)
            target_flat = target.flatten(2)
            for first in range(0, flat.shape[-1], chunk_modes):
                part = flat[:, :, first:first+chunk_modes]
                projected = torch.einsum('ir,iok->rok', ain.to(base.dtype), part)
                projected = torch.einsum('rok,os->rsk', projected, aout.to(base.dtype))
                target_flat[:, :, first:first+chunk_modes].copy_(projected)
                retained += float(projected.abs().square().sum(dtype=torch.float64))
        fraction = retained / total_energy if total_energy else 1.
        result.projection_report = dict(rank_in=result.rank_in, rank_out=result.rank_out,
            retained_weight_energy=fraction, relative_weight_error=math.sqrt(max(0., 1-fraction)),
            dense_parameters=sum(p.numel() for p in dense.parameters()),
            factorized_parameters=sum(p.numel() for p in result.parameters()))
        return result


def replace_spectral_modules(model, rank_fraction=.25, min_rank=8, project_weights=False, channel_bases=None):
    """Replace only the nine spectral branches, retaining all other modules."""
    reports = {}
    for name in [f'conv{i}' for i in range(9)]:
        block = getattr(model, name)
        old = block.conv
        if project_weights:
            bases = channel_bases.get(name, {}) if channel_bases else {}
            new = ChannelTuckerSpectralConv3d.from_dense(old, rank_fraction, min_rank, **bases)
            reports[name] = new.projection_report
        else:
            new = ChannelTuckerSpectralConv3d(old.in_channels, old.out_channels, old.d1, old.d2, old.d3,
                old.modes1, old.modes2, old.modes3, rank_fraction, min_rank)
        block.conv = new
    return reports
