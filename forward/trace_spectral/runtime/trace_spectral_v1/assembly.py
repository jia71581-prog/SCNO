"""Unique entity ownership and fixed tensor-product coefficient layout."""

import numpy as np
import torch
from torch import nn


class Assembly(nn.Module):
    """Map entity coefficients to [batch, time, free z, x] with the top removed.

    Entity order uses z-major then x-minor traversal in each category.
    Horizontal edges precede vertical edges. All edges are oriented along the
    positive canonical coordinate. Cell modes flatten z first and x last.
    """

    def __init__(self, x, z):
        super().__init__()
        if x.degree != z.degree or x.degree < 2:
            raise ValueError('assembly requires a common spatial degree of at least 2')
        p = x.degree
        self.degree, self.nx, self.nz = p, x.dofs, z.dofs
        self.edge_modes, self.interior_modes = p-1, (p-1)**2
        vertex, edge, cell = [], [], []

        for iz in range(z.elements+1):
            for ix in range(x.elements+1):
                vertex.append([iz*p*self.nx+ix*p])
        for iz in range(z.elements+1):
            for ix in range(x.elements):
                edge.append([iz*p*self.nx+ix*p+i for i in range(p+1)])
        for iz in range(z.elements):
            for ix in range(x.elements+1):
                edge.append([(iz*p+i)*self.nx+ix*p for i in range(p+1)])
        for iz in range(z.elements):
            for ix in range(x.elements):
                cell.append([iz0*self.nx+ix0 for iz0 in range(iz*p, (iz+1)*p+1)
                             for ix0 in range(ix*p, (ix+1)*p+1)])

        vertex = np.asarray(vertex, dtype=np.int64)
        edge = np.asarray(edge, dtype=np.int64)
        cell = np.asarray(cell, dtype=np.int64)
        valid_vertices = np.flatnonzero(vertex[:, 0] >= self.nx)
        # Only horizontal top edges are constrained. Vertical edges incident
        # on the top vertex keep their interior modes, whose basis vanishes
        # at the constrained vertex.
        valid_edges = np.flatnonzero(edge[:, 1] >= self.nx)
        self.full_vertex_count = len(vertex)
        self.full_edge_count = len(edge)
        self.vertex_count = len(valid_vertices)
        self.edge_count = len(valid_edges)
        self.interior_count = len(cell)
        self.register_buffer('valid_vertices', torch.from_numpy(valid_vertices))
        self.register_buffer('valid_edges', torch.from_numpy(valid_edges))
        self.register_buffer('vertex_nodes', torch.from_numpy(vertex[valid_vertices]))
        self.register_buffer('edge_nodes', torch.from_numpy(edge[valid_edges]))
        self.register_buffer('interior_nodes', torch.from_numpy(cell))

        vertex_modes = vertex[valid_vertices, 0]
        edge_modes = edge[valid_edges, 1:-1]
        cell_modes = cell.reshape(-1, p+1, p+1)[:, 1:-1, 1:-1]
        # Mode coordinates are indexed in the same order as native nodes on
        # each axis, but the stored values are the scaled basis coefficients.
        mode_index = np.concatenate((vertex_modes, edge_modes.reshape(-1), cell_modes.reshape(-1))) - self.nx
        dof_index = np.argsort(mode_index)
        if not np.array_equal(mode_index[dof_index], np.arange((self.nz-1)*self.nx)):
            raise ValueError('entity map does not cover each free spatial mode exactly once')
        self.register_buffer('mode_index', torch.from_numpy(mode_index))
        self.register_buffer('dof_index', torch.from_numpy(dof_index))

    def forward(self, vertex, edge, interior):
        """Pack pruned or full entity outputs; full top entries have zero gradient."""
        if vertex.ndim != 3 or edge.ndim != 4 or interior.ndim != 4:
            raise ValueError('expected vertex [B, N, T] and edge/interior [B, N, modes, T]')
        batch, _, time = vertex.shape
        if vertex.shape[1] == self.full_vertex_count:
            vertex = vertex.index_select(1, self.valid_vertices)
        if edge.shape[1] == self.full_edge_count:
            edge = edge.index_select(1, self.valid_edges)
        if (vertex.shape != (batch, self.vertex_count, time) or
                edge.shape != (batch, self.edge_count, self.edge_modes, time) or
                interior.shape != (batch, self.interior_count, self.interior_modes, time)):
            raise ValueError('entity coefficient sizes do not match the grid')
        packed = torch.cat((vertex, edge.flatten(1, 2), interior.flatten(1, 2)), dim=1)
        return packed.transpose(1, 2).index_select(2, self.dof_index).reshape(batch, time, self.nz-1, self.nx)

    def disassemble(self, coefficients, include_constrained=False):
        """Invert the fixed permutation and optionally restore zero top entities."""
        if coefficients.ndim != 4 or coefficients.shape[-2:] != (self.nz-1, self.nx):
            raise ValueError('coefficients must have shape [B, T, free Z, X]')
        batch, time = coefficients.shape[:2]
        packed = coefficients.reshape(batch, time, -1).index_select(2, self.mode_index).transpose(1, 2)
        v, e, c = torch.split(packed, (self.vertex_count, self.edge_count*self.edge_modes,
                                       self.interior_count*self.interior_modes), dim=1)
        e = e.reshape(batch, self.edge_count, self.edge_modes, time)
        c = c.reshape(batch, self.interior_count, self.interior_modes, time)
        if include_constrained:
            v = v.new_zeros(batch, self.full_vertex_count, time).index_copy(1, self.valid_vertices, v)
            e = e.new_zeros(batch, self.full_edge_count, self.edge_modes, time).index_copy(1, self.valid_edges, e)
        return v, e, c
