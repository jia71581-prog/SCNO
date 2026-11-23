"""Fixed reference bases and uniquely owned native samples for TraceSpectralWave."""

import hashlib
import json
import math

import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss


def _check_positive_int(value, name):
    if not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _coords(values, left, right):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or np.any(values < left) or np.any(values > right):
        raise ValueError(f'coordinates must lie in [{left}, {right}]')
    return values


def _raw_bubbles(coords, count, time, derivative):
    """Evaluate factored Legendre bubbles without endpoint cancellation."""
    if time:
        s = coords
        xi = 2*s-1
        window = [s*s*(1-s)*(1-s), 2*s*(1-s)*(1-2*s), 2-12*s+12*s*s]
        derivative_scale = 2
    else:
        xi = coords
        window = [1-xi*xi, -2*xi, np.full_like(xi, -2)]
        derivative_scale = 1
    columns = []
    for j in range(count):
        polynomial = Legendre.basis(j)
        value = sum(math.comb(derivative, k) * window[derivative-k] *
                    derivative_scale**k * polynomial.deriv(k)(xi)
                    for k in range(derivative+1))
        columns.append(value)
    return np.stack(columns, axis=1) if columns else np.empty((len(coords), 0))


def _endpoint_lift(coords, time, derivative):
    s = coords
    if time:
        # Derivatives are with respect to the reference s, with the time axis
        # storing the endpoint jet as (U, ΔT V).
        if derivative == 0:
            return np.stack(((1+2*s)*(1-s)**2, s*(1-s)**2,
                             s*s*(3-2*s), s*s*(s-1)), axis=1)
        if derivative == 1:
            return np.stack((6*s*(s-1), (1-s)*(1-3*s),
                             6*s*(1-s), s*(3*s-2)), axis=1)
        return np.stack((12*s-6, 6*s-4, 6-12*s, 6*s-2), axis=1)
    if derivative == 0:
        return np.stack(((1-s)/2, (1+s)/2), axis=1)
    if derivative == 1:
        return np.broadcast_to((-0.5, 0.5), (len(s), 2))
    return np.zeros((len(s), 2))


class _Reference:
    def __init__(self, count, time):
        self.count = count
        self.time = time
        xi, weights = leggauss(count + 5)
        self.points = (xi+1)/2 if time else xi
        self.weights = weights/2 if time else weights
        raw = _raw_bubbles(self.points, count, time, 0)
        _, r = np.linalg.qr(np.sqrt(self.weights[:, None])*raw, mode='reduced')
        # Fix the QR sign convention so the coordinate orientation has a
        # reproducible meaning across runs on the same numerical environment.
        self.r = np.where(np.diag(r) < 0, -1., 1.)[:, None]*r
        self.transform = np.linalg.solve(self.r, np.eye(count)) if count else self.r
        b = self._bubbles(self.points, 0)
        mass = b.T@(self.weights[:, None]*b)
        if not np.allclose(mass, np.eye(count), rtol=0, atol=1e-12):
            raise ValueError('reference bubble Gram is not orthonormal')
        self.lift_projection = b.T@(self.weights[:, None]*_endpoint_lift(self.points, time, 0))

    def _bubbles(self, coords, derivative):
        raw = _raw_bubbles(coords, self.count, self.time, derivative)
        return raw@self.transform

    def evaluate(self, coords, derivative):
        bubble = self._bubbles(coords, derivative)
        lift = _endpoint_lift(coords, self.time, derivative) - bubble@self.lift_projection
        if self.time:
            return np.concatenate((lift, bubble), axis=1)
        return np.concatenate((lift[:, :1], bubble, lift[:, 1:]), axis=1)


def _scale(reference, ids, dofs):
    """Use the assembled reference diagonal; native data use equal node weights."""
    values = reference.evaluate(reference.points, 0)
    diagonal = np.einsum('pi,p,pi->i', values, reference.weights, values)
    assembled = np.zeros(dofs, dtype=np.float64)
    np.add.at(assembled, ids.flatten(), np.tile(diagonal, len(ids)))
    if np.any(assembled <= 0) or not np.isfinite(assembled).all():
        raise ValueError('invalid assembled reference diagonal')
    return 1/np.sqrt(assembled)


def _assemble(values, cells, ids, dofs):
    """Assign one owner per evaluation coordinate; omit constrained columns."""
    columns = ids[cells]
    out = np.zeros((len(cells), dofs), dtype=np.float64)
    row, slot = np.where(columns >= 0)
    out[row, columns[row, slot]] = values[row, slot]
    return out


def _fingerprint(config, *arrays):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode('ascii'))
    for array in arrays:
        array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder('<'))
        digest.update(str((array.shape, array.dtype.str)).encode('ascii'))
        digest.update(memoryview(array))
    return digest.hexdigest()


class SpaceBasis:
    """C0 local polynomials on a 1D physical axis. Global coefficient 0 is the left trace."""

    def __init__(self, elements=25, degree=8, length=2000.):
        _check_positive_int(elements, 'elements')
        _check_positive_int(degree, 'degree')
        if not np.isfinite(length) or length <= 0:
            raise ValueError('length must be positive and finite')
        self.elements, self.degree, self.length = elements, degree, float(length)
        self.cell_width = self.length/elements
        self.dofs = elements*degree+1
        self.nodes = np.linspace(0, length, self.dofs)
        self.local_to_global = np.arange(elements, dtype=np.int64)[:, None]*degree + np.arange(degree+1)
        self.reference = _Reference(degree-1, time=False)
        self.scale = _scale(self.reference, self.local_to_global, self.dofs)
        self.inverse_scale = 1/self.scale
        self.local_matrix = self.reference.evaluate(np.linspace(-1, 1, degree+1), 0)[None, :, :] * (
            self.scale[self.local_to_global][:, None, :])
        native = np.arange(self.dofs)
        self.native_owner = np.minimum(native//degree, elements-1)
        self.local_native_row = native-self.native_owner*degree
        self.matrix = _assemble(self.local_matrix[self.native_owner, self.local_native_row], self.native_owner,
                                self.local_to_global, self.dofs)
        self.fingerprint = _fingerprint({'version': 1, 'kind': 'C0', 'elements': elements,
                                         'degree': degree, 'length': self.length},
                                        self.matrix, self.local_to_global, self.scale,
                                        self.reference.transform, self.reference.lift_projection)

    def evaluate_local(self, reference_coords, element, derivative=0, reverse=False):
        """Evaluate one cell; reverse=True lists its physical positions in opposite orientation.

        Returned columns always retain the cell's canonical local coefficient
        order. Derivatives are with respect to the canonical physical axis.
        """
        if derivative not in (0, 1, 2) or not 0 <= element < self.elements:
            raise ValueError('invalid derivative order or spatial cell')
        xi = _coords(reference_coords, -1, 1)
        if reverse:
            xi = -xi
        values = self.reference.evaluate(xi, derivative)
        return values*self.scale[self.local_to_global[element]]*(2/self.cell_width)**derivative

    def evaluate(self, coords, derivative=0):
        if derivative not in (0, 1, 2):
            raise ValueError('derivative must be 0, 1, or 2')
        position = _coords(coords, 0, self.length)/self.cell_width
        owner = np.minimum(position.astype(np.int64), self.elements-1)
        xi = np.clip(2*(position-owner)-1, -1, 1)
        values = self.reference.evaluate(xi, derivative) * self.scale[self.local_to_global[owner]]
        return _assemble(values*(2/self.cell_width)**derivative, owner, self.local_to_global, self.dofs)


class TimeBasis:
    """C1 local Hermite and Legendre modes with shared endpoint jets and zero initial jet."""

    def __init__(self, slabs=20, intervals=40, bubbles=8, duration=1.):
        for value, name in [(slabs, 'slabs'), (intervals, 'intervals'), (bubbles, 'bubbles')]:
            _check_positive_int(value, name)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('duration must be positive and finite')
        self.slabs, self.intervals, self.bubbles, self.duration = slabs, intervals, bubbles, float(duration)
        self.slab_width = self.duration/slabs
        self.frames = slabs*intervals+1
        self.dofs = slabs*(2+bubbles)
        self.nodes = np.linspace(0, duration, self.frames)
        self.reference = _Reference(bubbles, time=True)
        stride = bubbles+2
        left = np.arange(slabs, dtype=np.int64)[:, None]*stride
        full_indices = left+np.r_[0, 1, stride, stride+1, np.arange(2, stride)]
        self.full_scale = _scale(self.reference, full_indices, self.dofs+2)
        self.scale = self.full_scale[2:].copy()
        self.inverse_scale = 1/self.scale
        self.indices = np.maximum(full_indices-2, -1)
        self.local_scale = self.full_scale[full_indices]
        self.local_matrix = self.reference.evaluate(np.linspace(0, 1, intervals+1), 0)[None, :, :] * (
            self.local_scale[:, None, :])
        native = np.arange(self.frames)
        self.native_owner = np.minimum(native//intervals, slabs-1)
        self.local_native_row = native-self.native_owner*intervals
        self.matrix = _assemble(self.local_matrix[self.native_owner, self.local_native_row], self.native_owner,
                                self.indices, self.dofs)
        self.fingerprint = _fingerprint({'version': 1, 'kind': 'C1', 'slabs': slabs, 'intervals': intervals,
                                         'bubbles': bubbles, 'duration': self.duration},
                                        self.matrix, self.indices, self.scale,
                                        self.reference.transform, self.reference.lift_projection)

    def evaluate_local(self, reference_coords, slab, derivative=0):
        if derivative not in (0, 1, 2) or not 0 <= slab < self.slabs:
            raise ValueError('invalid derivative order or time slab')
        s = _coords(reference_coords, 0, 1)
        return self.reference.evaluate(s, derivative)*self.local_scale[slab]/self.slab_width**derivative

    def evaluate(self, coords, derivative=0):
        if derivative not in (0, 1, 2):
            raise ValueError('derivative must be 0, 1, or 2')
        position = _coords(coords, 0, self.duration)/self.slab_width
        owner = np.minimum(position.astype(np.int64), self.slabs-1)
        s = np.clip(position-owner, 0, 1)
        values = self.reference.evaluate(s, derivative)*self.local_scale[owner]
        return _assemble(values/self.slab_width**derivative, owner, self.indices, self.dofs)
