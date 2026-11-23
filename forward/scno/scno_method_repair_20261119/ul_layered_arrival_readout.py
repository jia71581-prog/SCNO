"""Learn pressure corrections on physically ordered arrivals in two flat layers."""

from typing import NamedTuple

import torch
from torch import nn


EVENTS = ('interface_reflection', 'interface_transmission', 'surface_return',
          'interface_then_surface', 'surface_then_interface')


class ArrivalGeometry(NamedTuple):
    features: torch.Tensor
    times: torch.Tensor
    mask: torch.Tensor
    direct: torch.Tensor
    frequency: torch.Tensor
    interface_depth: torch.Tensor
    eligible: torch.Tensor


def flat_two_layer_profiles(velocity):
    """Read the interfaces from the pressure grid's velocity samples."""
    profile = velocity[:, 0, :, 0]
    parallel = (velocity[:, 0] == profile[..., None]).flatten(1).all(1)
    jumps = profile[:, 1:] != profile[:, :-1]
    eligible = parallel & (jumps.sum(1) == 1)
    first_lower = 1 + jumps.to(torch.int32).argmax(1)
    return (first_lower.to(profile.dtype) - .5) * 10., profile[:, 0], profile[:, -1], eligible


def snell_time(horizontal, upper_distance, lower_distance, upper_speed, lower_speed):
    """Solve the monotone horizontal-offset equation for the ray parameter."""
    low = torch.zeros_like(horizontal)
    high = torch.ones_like(horizontal) * ((1. - 1e-6) / torch.maximum(upper_speed, lower_speed))
    for _ in range(22):
        p = (low + high) * .5
        su, sl = p * upper_speed, p * lower_speed
        cu, cl = (1. - su.square()).sqrt(), (1. - sl.square()).sqrt()
        distance = upper_distance * su / cu + lower_distance * sl / cl
        low = torch.where(distance < horizontal, p, low)
        high = torch.where(distance < horizontal, high, p)
    p = (low + high) * .5
    cu = (1. - (p * upper_speed).square()).sqrt()
    cl = (1. - (p * lower_speed).square()).sqrt()
    return upper_distance / (upper_speed * cu) + lower_distance / (lower_speed * cl)


def arrival_geometry(velocity, source):
    """Build event-specific travel times and spatial features on the active grid."""
    interface, upper, lower, eligible = flat_two_layer_profiles(velocity)
    depth = interface[:, None, None]
    cu, cl = upper[:, None, None], lower[:, None, None]
    sx, sz, f0, t0, amplitude = (column[:, None, None] for column in source.unbind(1))
    device, dtype = velocity.device, velocity.dtype
    z = (10. * torch.arange(1, 201, dtype=dtype, device=device))[None, :, None]
    x = (10. * torch.arange(201, dtype=dtype, device=device))[None, None, :]
    dx = (x - sx).expand(-1, 200, -1)
    dz = (z - sz).expand(-1, -1, 201)
    source_above, receiver_above = sz < depth, z < depth
    same_layer = (source_above == receiver_above).expand(-1, -1, 201)
    both_above = (source_above & receiver_above).expand_as(same_layer)
    same_speed = torch.where(source_above, cu, cl)
    reflection = t0 + torch.hypot(dx, z + sz - 2. * depth) / same_speed
    upper_distance = torch.where(source_above, depth - sz, depth - z).clamp_min(0.)
    lower_distance = torch.where(source_above, z - depth, sz - depth).clamp_min(0.)
    transmission = t0 + snell_time(dx.abs(), upper_distance, lower_distance, cu, cl)
    direct = torch.where(same_layer, t0 + torch.hypot(dx, dz) / same_speed, transmission)
    surface_upper = torch.where(same_layer, 2. * depth,
                                depth + torch.where(source_above, sz, z))
    surface_lower = torch.where(same_layer, (z + sz - 2. * depth).clamp_min(0.), lower_distance)
    surface = torch.where(both_above, t0 + torch.hypot(dx, z + sz) / cu,
                          t0 + snell_time(dx.abs(), surface_upper, surface_lower, cu, cl))
    reflected_then_surface = t0 + torch.hypot(dx, z - sz + 2. * depth) / cu
    surface_then_reflected = t0 + torch.hypot(dx, z - sz - 2. * depth) / cu
    arrival = torch.stack((reflection, transmission, surface, reflected_then_surface, surface_then_reflected), dim=-1)
    event_mask = torch.stack((same_layer, ~same_layer, torch.ones_like(same_layer), both_above, both_above), dim=-1)
    event_mask &= eligible[:, None, None, None] & (amplitude > 0.)[..., None]

    def broadcast(value):
        return value.expand_as(dx)

    attributes = torch.stack((dx / 2000., dz / 2000., broadcast(depth / 2000.),
                              broadcast((z - depth) / 2000.), broadcast(cu / 5000.), broadcast(cl / 5000.),
                              velocity[:, 0, 1:] / 5000., broadcast(f0 / 15.), broadcast(t0), direct), dim=-1)
    times_as_features = torch.where(event_mask, arrival, torch.zeros_like(arrival))
    # Phase, arrival time and interface depth give the hidden spatial readout
    # separate conditioning for the distinct propagating events.
    phase = (2. * torch.pi * f0[..., None]) * times_as_features
    phase_features = torch.cat((times_as_features, phase.sin() * event_mask, phase.cos() * event_mask), dim=-1)
    features = torch.cat((attributes, phase_features, event_mask.to(dtype)), dim=-1)
    return ArrivalGeometry(features.flatten(1, 2), arrival.flatten(1, 2), event_mask.flatten(1, 2),
                           direct.flatten(1), source[:, 2], interface, eligible)


class LayeredArrivalReadout(nn.Module):
    """Predict compact pressure pulses around each physical arrival."""

    geometry_channels = 10 + 4 * len(EVENTS)
    events = EVENTS
    pulse_channels = 5
    harmonics = 40

    def __init__(self, inherited_channels, width=64):
        super().__init__()
        count = len(self.events) * self.pulse_channels + (2 * self.harmonics + 1)
        self.spatial = nn.Sequential(nn.Linear(inherited_channels + self.geometry_channels, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(), nn.Linear(width, count))
        self.register_buffer('temporal_modes_hz', torch.arange(1, self.harmonics + 1, dtype=torch.float32),
                             persistent=False)
        # The added branch begins at exactly the inherited prediction.
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)

    def coefficients(self, inherited, geo_features):
        output = self.spatial(torch.cat((inherited, geo_features), dim=-1))
        pulse, late = output.split((len(self.events) * self.pulse_channels, 2 * self.harmonics + 1), dim=-1)
        return pulse.reshape(*output.shape[:-1], len(self.events), self.pulse_channels), late

    def render(self, coeff, times, arrivals, mask, f0):
        """Use one physical query list per record; reuse it at every point."""
        pulses, late = coeff
        offset = (times[..., :, None] - arrivals[..., None, :]) * f0[:, None, None, None]
        # The Gaussian Hermite pulses capture amplitude and phase with a modest
        # correction on either side of the geometrical arrival.
        x = offset * 2.6
        x2 = x.square()
        envelope = torch.exp(-x2) * mask[..., None, :]
        terms = torch.stack((torch.ones_like(x), x, 1. - 2. * x2,
                             x * (3. - 2. * x2), x2 * (3. - x2)), dim=-1)
        pressure = (terms * envelope[..., None] * pulses[..., None, :, :]).sum((-1, -2))
        native = times[:, 0, :]
        phase = (2. * torch.pi) * native[..., None] * self.temporal_modes_hz
        basis = torch.cat((torch.ones_like(phase[..., :1]), phase.cos(), phase.sin()), dim=-1)
        ramp = ((native - .25) / .15).clamp(0., 1.)
        gate = ramp.square() * (3. - 2. * ramp)
        coda = torch.einsum('bpd,bnd->bpn', late, basis)
        pressure = pressure + coda * gate[:, None, :] * mask.any(-1)[..., None]
        return pressure * (times > 0.)


class ArrivalSCNO(nn.Module):
    """The preserved SCNO plus an independently trained arrival readout."""

    def __init__(self, backbone, readout, *, geometry=arrival_geometry, profiles=flat_two_layer_profiles):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.readout = readout
        self.geometry = geometry
        self.profiles = profiles

    @property
    def device(self):
        return self.backbone.device

    @property
    def normalizer(self):
        return self.backbone.normalizer

    @property
    def full_times_s(self):
        return self.backbone.full_times_s

    def train(self, mode=True):
        self.training = mode
        self.readout.train(mode)
        self.backbone.eval()
        return self

    def forward(self, velocity, source, frame_indices=None, *, normalized=True):
        v, s = self.backbone._inputs(velocity, source)
        context = self.backbone.encode(v, s)
        if frame_indices is None:
            frame_indices = torch.arange(len(self.full_times_s), device=self.device)
        prediction = self.backbone.decode(context, frame_indices, normalized=True)
        v = v.expand(len(s), -1, -1, -1)
        if self.profiles(v)[-1].any():
            geometry = self.geometry(v, s)
            inherited = context.features[:, :, 1:].flatten(2).transpose(1, 2)
            coefficients = self.readout.coefficients(inherited, geometry.features)
            times = self.full_times_s[frame_indices]
            if times.ndim == 1:
                times = times[None].expand(len(s), -1)
            correction = []
            for part in times.split(8, dim=1):
                result = self.readout.render(coefficients, part[:, None], geometry.times, geometry.mask, geometry.frequency)
                correction.append(result.transpose(1, 2).reshape(len(s), part.shape[-1], 200, 201))
            pressure = torch.cat(correction, dim=1)
            prediction = prediction + torch.nn.functional.pad(pressure, (0, 0, 1, 0))
        return prediction if normalized else self.normalizer.decode_pressure(prediction, s[:, 4])
