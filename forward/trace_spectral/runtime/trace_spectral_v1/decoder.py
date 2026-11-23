"""Fixed native-grid reconstruction and continuous point/time queries."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _Axis(nn.Module):
    def __init__(self, indices, matrices, native_owner, local_row, full, dofs):
        super().__init__()
        self.dofs = dofs
        self.register_buffer('indices', torch.from_numpy(indices.copy()) + 1)
        self.register_buffer('take', torch.from_numpy(native_owner * matrices.shape[1] + local_row))
        full64 = torch.from_numpy(full.copy())
        local64 = torch.from_numpy(matrices.copy())
        self.register_buffer('matrix64', full64)
        self.register_buffer('matrix32', full64.float())
        self.register_buffer('local64', local64)
        self.register_buffer('local32', local64.float())

    def table(self, dtype):
        return self.matrix64 if dtype == torch.float64 else self.matrix32

    def time(self, values):
        """Keep time-major pressure contiguous while contracting local slabs."""
        batch, _, nz, nx = values.shape
        flat = values.flatten(2)
        padded = F.pad(flat, (0, 0, 1, 0))
        slabs, columns = self.indices.shape
        local_input = padded.index_select(1, self.indices.flatten()).reshape(batch, slabs, columns, -1)
        matrices = self.local64 if values.dtype == torch.float64 else self.local32
        local_output = torch.matmul(matrices, local_input).flatten(1, 2)
        return local_output.index_select(1, self.take).reshape(batch, len(self.take), nz, nx)

    def forward(self, values, axis):
        moved = values.movedim(axis, -1)
        shape = moved.shape[:-1]
        # Index zero is the fixed zero coordinate, used by removed spatial
        # and temporal traces. Remaining local columns select free modes.
        local_input = F.pad(moved, (1, 0)).index_select(-1, self.indices.flatten())
        cells, columns = self.indices.shape
        local_input = local_input.reshape(-1, cells, columns).permute(1, 2, 0)
        matrices = self.local64 if values.dtype == torch.float64 else self.local32
        local_output = torch.bmm(matrices, local_input)
        native = local_output.flatten(0, 1).index_select(0, self.take).T
        return native.reshape(*shape, len(self.take)).movedim(-1, axis)


def _space_axis(basis, remove_top=False):
    indices, full = basis.local_to_global, basis.matrix
    if remove_top:
        indices, full = indices - 1, full[:, 1:]
    return _Axis(indices, basis.local_matrix, basis.native_owner, basis.local_native_row,
                 full, basis.dofs - int(remove_top))


class FixedDecoder(nn.Module):
    """Evaluate tensor-product bases without assembling a spacetime matrix."""

    def __init__(self, space_x, space_z, time_basis):
        super().__init__()
        self.space_x, self.space_z, self.time_basis = space_x, space_z, time_basis
        self.x_axis = _space_axis(space_x)
        self.z_axis = _space_axis(space_z, remove_top=True)
        self.t_axis = _Axis(time_basis.indices, time_basis.local_matrix, time_basis.native_owner,
                            time_basis.local_native_row, time_basis.matrix, time_basis.dofs)
        self.coefficient_shape = (time_basis.dofs, space_z.dofs - 1, space_x.dofs)

    def _prepare(self, coefficients):
        if coefficients.ndim != 4 or coefficients.shape[1:] != self.coefficient_shape:
            raise ValueError(f'expected coefficient tensor [B,{self.coefficient_shape}]')
        dtype = torch.float64 if coefficients.dtype == torch.float64 else torch.float32
        return coefficients.to(dtype=dtype)

    def decode_full_local(self, coefficients, *, pressure_scale=1.0):
        a = self._prepare(coefficients)
        with torch.autocast(device_type=a.device.type, enabled=False):
            a = self.x_axis(a, -1)
            a = self.z_axis(a, -2)
            pressure = self.t_axis.time(a)
            return pressure if pressure_scale == 1.0 else pressure * pressure_scale

    def decode_full(self, coefficients, *, pressure_scale=1.0):
        a = self._prepare(coefficients)
        with torch.autocast(device_type=a.device.type, enabled=False):
            x = self.x_axis.table(a.dtype)
            z = self.z_axis.table(a.dtype)
            # Each spatial axis is a small fixed GEMM. The time contraction
            # still touches only the 4 + n_bubble modes of each slab.
            spatial = z @ (a @ x.T)
            pressure = self.t_axis.time(spatial)
            return pressure if pressure_scale == 1.0 else pressure * pressure_scale

    def decode_full_reference(self, coefficients, *, pressure_scale=1.0):
        a = self._prepare(coefficients)
        with torch.autocast(device_type=a.device.type, enabled=False):
            x, z, t = (axis.table(a.dtype) for axis in (self.x_axis, self.z_axis, self.t_axis))
            return torch.einsum('kt,btzx->bkzx', t, torch.einsum('jz,btzx->btjx', z, a @ x.T)) * pressure_scale

    def decode_times(self, coefficients, times, *, pressure_scale=1.0):
        """Decode whole spatial frames at arbitrary physical times in seconds."""
        a = self._prepare(coefficients)
        times = times.detach().cpu().numpy() if torch.is_tensor(times) else times
        t = torch.as_tensor(self.time_basis.evaluate(times), device=a.device, dtype=a.dtype)
        with torch.autocast(device_type=a.device.type, enabled=False):
            queried = torch.einsum('kt,btzx->bkzx', t, a)
            return self.z_axis(self.x_axis(queried, -1), -2).contiguous() * pressure_scale

    def decode_queries(self, coefficients, xyz_t, *, pressure_scale=1.0):
        """Decode physical [x in m, z in m, t in s] coordinates in bounded chunks."""
        a = self._prepare(coefficients)
        points = xyz_t.detach().cpu().numpy() if torch.is_tensor(xyz_t) else np.asarray(xyz_t)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('queries must have shape [Q,3] in x,z,t order')
        if not len(points):
            return a.new_empty((len(a), 0))
        flat = a.flatten(1)
        outputs = []
        with torch.autocast(device_type=a.device.type, enabled=False):
            for chunk in np.array_split(points, max(1, (len(points) + 255) // 256)):
                x, z, t = chunk.T
                index_x, value_x = _query_space(self.space_x, x)
                index_z, value_z = _query_space(self.space_z, z, remove_top=True)
                index_t, value_t = _query_time(self.time_basis, t)
                _, nz, nx = self.coefficient_shape
                indices = (index_t[:, :, None, None] * nz * nx + index_z[:, None, :, None] * nx
                           + index_x[:, None, None, :])
                weight = value_t[:, :, None, None] * value_z[:, None, :, None] * value_x[:, None, None, :]
                index_gpu = torch.as_tensor(indices.reshape(-1), device=a.device)
                weight_gpu = torch.as_tensor(weight.reshape(len(chunk), -1), device=a.device, dtype=a.dtype)
                terms = flat.index_select(1, index_gpu).reshape(len(a), len(chunk), -1)
                outputs.append(torch.einsum('bqk,qk->bq', terms, weight_gpu))
        return torch.cat(outputs, dim=1) * pressure_scale


def _query_space(basis, coords, remove_top=False):
    full = basis.evaluate(coords)
    owner = np.minimum((coords / basis.cell_width).astype(np.int64), basis.elements - 1)
    indices = basis.local_to_global[owner]
    weight = np.take_along_axis(full, indices, axis=1)
    if remove_top:
        weight = np.where(indices == 0, 0., weight)
        indices = np.maximum(indices - 1, 0)
    return indices, weight


def _query_time(basis, coords):
    full = basis.evaluate(coords)
    owner = np.minimum((coords / basis.slab_width).astype(np.int64), basis.slabs - 1)
    indices = basis.indices[owner]
    safe = np.maximum(indices, 0)
    return safe, np.where(indices < 0, 0., np.take_along_axis(full, safe, axis=1))
