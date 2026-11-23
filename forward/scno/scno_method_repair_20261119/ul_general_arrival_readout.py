"""Pressure correction on gently varying single-interface media."""

import torch

from ul_arrival_geometry_general import general_arrival_geometry
from ul_layered_arrival_readout import EVENTS, LayeredArrivalReadout


class GeneralLayeredArrivalReadout(LayeredArrivalReadout):
    events = (*EVENTS, 'direct_arrival')

    def render(self, coeff, times, arrivals, mask, f0):
        pulses, late = coeff
        x = 2.6 * (times[..., :, None] - arrivals[..., None, :]) * f0[:, None, None, None]
        x2 = x.square()
        a, b, c, d, e = (weights.unsqueeze(-2) for weights in pulses.unbind(-1))
        envelope = torch.exp(-x2) * mask[..., None, :]
        polynomial = (a + c) + x * ((b + 3. * d) + x * ((3. * e - 2. * c) - x * (2. * d + x * e)))
        pressure = (envelope * polynomial).sum(-1)

        # Shift the coda phase by direct travel time before its temporal synthesis.
        # One rotated coefficient vector serves all queries at a spatial point.
        direct = arrivals[..., -1]
        shift = (2. * torch.pi) * direct[..., None] * self.temporal_modes_hz
        cosine, sine = shift.cos(), shift.sin()
        constant, real, imag = late.split((1, self.harmonics, self.harmonics), dim=-1)
        aligned = torch.cat((constant, real * cosine - imag * sine, real * sine + imag * cosine), dim=-1)
        native = times[:, 0, :]
        phase = (2. * torch.pi) * native[..., None] * self.temporal_modes_hz
        basis = torch.cat((torch.ones_like(phase[..., :1]), phase.cos(), phase.sin()), dim=-1)
        ramp = ((native - .25) / .15).clamp(0., 1.)
        gate = ramp.square() * (3. - 2. * ramp)
        coda = torch.einsum('bpd,bnd->bpn', aligned, basis)
        pressure = pressure + coda * gate[:, None, :] * mask.any(-1)[..., None]
        # Smoothly start the correction at the leading edge of a direct pulse.
        front = (.75 + (times - direct[..., None]) * f0[:, None, None]).clamp(0., 1.)
        return pressure * front.square() * (3. - 2. * front) * (times > 0.)


def generalized_geometry(velocity, source):
    geometry = general_arrival_geometry(velocity, source)
    valid = geometry.eligible[:, None] & (source[:, 4, None] > 0.)
    direct_mask = valid.expand_as(geometry.direct)
    return geometry._replace(times=torch.cat((geometry.times, geometry.direct[..., None]), dim=-1),
                             mask=torch.cat((geometry.mask, direct_mask[..., None]), dim=-1))
