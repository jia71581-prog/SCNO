"""Single-pass TraceSpectralWave-v1 prediction and pressure decoding."""

from torch import nn

from .assembly import Assembly
from .basis import SpaceBasis, TimeBasis
from .decoder import FixedDecoder
from .encoder import MultiScaleEncoder
from .entity_heads import EntityHeads


PARAMETER_BUDGET = 7_913_200


class TraceSpectralWave(nn.Module):
    def __init__(self, in_channels=21, space_x=None, space_z=None, time_basis=None, *,
                 entity_condition_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.space_x = SpaceBasis() if space_x is None else space_x
        self.space_z = SpaceBasis() if space_z is None else space_z
        self.time_basis = TimeBasis() if time_basis is None else time_basis
        if entity_condition_channels is None:
            entity_condition_channels = tuple(range(8, in_channels))
        if not all(0 <= channel < in_channels for channel in entity_condition_channels):
            raise ValueError('invalid conditioning channels')
        self.encoder = MultiScaleEncoder(in_channels)
        self.assemble = Assembly(self.space_x, self.space_z)
        self.entity_heads = EntityHeads(self.assemble, self.time_basis.dofs, entity_condition_channels)
        self.decoder = FixedDecoder(self.space_x, self.space_z, self.time_basis)
        count = sum(p.numel() for p in self.parameters())
        if count > PARAMETER_BUDGET:
            raise ValueError(f'{count:,} trainable parameters exceed the {PARAMETER_BUDGET:,} budget')

    def encode(self, packed):
        if packed.ndim != 4 or packed.shape[1:] != (self.in_channels, self.space_z.dofs, self.space_x.dofs):
            raise ValueError('packed inputs disagree with the velocity and basis grid')
        return self.encoder(packed)

    def prepare_entities(self, native, context, packed):
        return self.entity_heads.prepare(native, context, packed)

    def predict_entities(self, native, context, packed):
        return self.entity_heads(native, context, packed, self.assemble)

    def predict_coefficients(self, packed):
        native, context = self.encode(packed)
        return self.assemble(*self.predict_entities(native, context, packed))

    def decode_full(self, coefficients, *, pressure_scale=1.0):
        return self.decoder.decode_full(coefficients, pressure_scale=pressure_scale)

    def decode_times(self, coefficients, times, *, pressure_scale=1.0):
        return self.decoder.decode_times(coefficients, times, pressure_scale=pressure_scale)

    def decode_queries(self, coefficients, xyz_t, *, pressure_scale=1.0):
        return self.decoder.decode_queries(coefficients, xyz_t, pressure_scale=pressure_scale)

    def predict_pressure(self, packed, *, pressure_scale):
        return self.decode_full(self.predict_coefficients(packed), pressure_scale=pressure_scale)

    def forward(self, packed):
        return self.decode_full(self.predict_coefficients(packed))
