"""Batched, shared MLP heads for the three unique spatial entity types."""

import torch
from torch import nn


class EntityHeads(nn.Module):
    def __init__(self, assembly, time_dofs, condition_channels, native=32, context=96, hidden=64):
        super().__init__()
        self.time_dofs = time_dofs
        self.edge_modes = assembly.edge_modes
        self.interior_modes = assembly.interior_modes
        self.register_buffer('condition_channels', torch.tensor(condition_channels, dtype=torch.long))
        common = context + len(condition_channels) + 2
        self.vertex = self._head(native + common, hidden, time_dofs)
        self.edge = self._head(native * (assembly.degree + 1) + common + 1,
                               hidden, self.edge_modes * time_dofs)
        self.interior = self._head(native * (assembly.degree + 1)**2 + common,
                                   hidden, self.interior_modes * time_dofs)
        n_horizontal = assembly.full_edge_count - (assembly.nz - 1) // assembly.degree * (
            (assembly.nx - 1) // assembly.degree + 1)
        self.register_buffer('edge_vertical', (assembly.valid_edges >= n_horizontal).float().view(1, -1, 1))
        for name in ('vertex', 'edge', 'interior'):
            nodes = getattr(assembly, f'{name}_nodes')
            position = torch.stack(((nodes % assembly.nx).float().mean(dim=1) / (assembly.nx - 1),
                                    (nodes // assembly.nx).float().mean(dim=1) / (assembly.nz - 1)), dim=-1)
            self.register_buffer(f'{name}_position', position[None])

    @staticmethod
    def _head(inputs, hidden, outputs):
        head = nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def prepare(self, native, context, packed):
        flat = native.flatten(2).transpose(1, 2).contiguous()
        conditioning = packed.index_select(1, self.condition_channels).flatten(2).transpose(1, 2)
        return flat, context, conditioning.contiguous()

    def features(self, prepared, nodes, centers, orientation=None):
        flat, context, conditioning = prepared
        batch = flat.shape[0]
        local = flat[:, nodes].flatten(-2)
        # Relative coordinates are affine, so their support mean is the entity
        # center. The pooled source stencil and constant maps keep the known
        # source accessible along with the global encoder context.
        pooled = conditioning[:, nodes].mean(dim=2).to(local.dtype)
        position = centers.expand(batch, -1, -1).to(local.dtype)
        global_context = context[:, None].expand(-1, len(nodes), -1).to(local.dtype)
        inputs = (local, global_context, position, pooled)
        if orientation is not None:
            inputs += (orientation.expand(batch, -1, -1).to(local.dtype),)
        return torch.cat(inputs, dim=-1)

    def predict_vertices(self, prepared, assembly):
        raw = self.vertex(self.features(prepared, assembly.vertex_nodes, self.vertex_position))
        return self._coefficients(raw)

    def predict_edges(self, prepared, assembly):
        raw = self.edge(self.features(prepared, assembly.edge_nodes, self.edge_position, self.edge_vertical))
        return self._coefficients(raw.reshape(len(raw), -1, self.edge_modes, self.time_dofs))

    def predict_interiors(self, prepared, assembly):
        raw = self.interior(self.features(prepared, assembly.interior_nodes, self.interior_position))
        return self._coefficients(raw.reshape(len(raw), -1, self.interior_modes, self.time_dofs))

    @staticmethod
    def _coefficients(raw):
        return raw if raw.dtype == torch.float64 else raw.float()

    def forward(self, native, context, packed, assembly):
        prepared = self.prepare(native, context, packed)
        return (self.predict_vertices(prepared, assembly), self.predict_edges(prepared, assembly),
                self.predict_interiors(prepared, assembly))
