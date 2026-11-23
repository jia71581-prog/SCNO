"""Reflection symmetry for the 1 km free-surface Trace-Spectral model."""

import numpy as np
import torch

from trace_spectral_v1.input_pack import pack_input
from trace_spectral_v1.model import TraceSpectralWave


def reflection_map(space):
    """Reverse cells and vertices. Odd cell bubbles also change sign."""
    indices = np.arange(space.dofs)
    signs = np.ones(space.dofs)
    for offset in range(space.degree):
        indices[offset::space.degree] = indices[offset::space.degree][::-1]
        if offset and (offset - 1) % 2:
            signs[offset::space.degree] = -1
    if not np.allclose(space.matrix[::-1], space.matrix[:, indices] * signs, atol=1e-12):
        raise ValueError('Spatial basis does not support the registered reflection')
    return torch.from_numpy(indices.copy()), torch.from_numpy(signs)


def reflect_inputs(velocity, source, extent):
    reflected = source.clone()
    reflected[:, 0] = extent - reflected[:, 0]
    return velocity.flip((-1,)), reflected


class ReflectionTraceSpectral(TraceSpectralWave):
    """Project predictions onto horizontal reflection equivariance."""

    def __init__(self, space, temporal):
        super().__init__(space_x=space, space_z=space, time_basis=temporal)
        indices, signs = reflection_map(space)
        self.register_buffer('reflection_indices', indices, persistent=False)
        self.register_buffer('reflection_signs', signs.float(), persistent=False)

    def reflect_coefficients(self, coefficients):
        return coefficients.index_select(-1, self.reflection_indices) * self.reflection_signs

    def predict_equivariant_coefficients(self, velocity, source, normalization):
        mirrored_velocity, mirrored_source = reflect_inputs(velocity, source, self.space_x.length)
        original = self.predict_coefficients(pack_input(velocity, source, **normalization))
        mirrored = self.predict_coefficients(pack_input(mirrored_velocity, mirrored_source, **normalization))
        return .5 * (original + self.reflect_coefficients(mirrored))

    def predict_equivariant_pressure(self, velocity, source, normalization, *, pressure_scale):
        coefficients = self.predict_equivariant_coefficients(velocity, source, normalization)
        return self.decode_full(coefficients, pressure_scale=pressure_scale)
