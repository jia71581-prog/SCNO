"""Lossless fine-grid layout and source conditioning for TraceSpectralWave."""

import torch
import torch.nn.functional as F


def pack_fine(values):
    if values.ndim != 3 or values.shape[-2:] != (401, 401):
        raise ValueError('Expected [B,401,401] fine grid')
    return F.pixel_unshuffle(F.pad(values[:, None], (0, 1, 0, 1)), 2)


def unpack_fine(values):
    if values.ndim != 4 or values.shape[1:] != (4, 201, 201):
        raise ValueError('Expected [B,4,201,201] packed grid')
    return F.pixel_shuffle(values, 2)[:, 0, :401, :401]


def source_mass(source_xz, extent):
    """Bilinear nodal mass in [z,x], before division by dx*dz."""
    node = source_xz*400/extent
    lower = node.floor().to(torch.long).clamp(0, 399)
    u, v = (node - lower).unbind(dim=1)
    ix, iz = lower.unbind(dim=1)
    base = iz*401 + ix
    indices = torch.stack((base, base+1, base+401, base+402), dim=1)
    weights = torch.stack(((1-u)*(1-v), u*(1-v), (1-u)*v, u*v), dim=1)
    result = weights.new_zeros((len(node), 401*401))
    return result.scatter(1, indices, weights).reshape(-1, 401, 401)


def pack_input(velocity_mps, source, *, extent_xz, duration_s, velocity_center, velocity_scale):
    """Source is [x_m,z_m,peak_hz,delay_s,amplitude] per sample."""
    if velocity_mps.ndim != 3 or velocity_mps.shape[1:] != (401, 401):
        raise ValueError('Expected velocity [B,401,401]')
    velocity = velocity_mps.to(dtype=torch.float32)
    parameters = torch.as_tensor(source, device=velocity.device, dtype=torch.float64)
    extent = torch.as_tensor(extent_xz, device=velocity.device, dtype=torch.float64)
    if parameters.shape != (len(velocity), 5) or extent.shape != (2,):
        raise ValueError('Expected exactly one five-parameter physical source per sample')
    if velocity_scale <= 0 or duration_s <= 0:
        raise ValueError('Invalid velocity scale or duration')
    valid = ((extent > 0).all() & torch.isfinite(velocity).all() & (velocity > 0).all()
             & torch.isfinite(parameters).all() & (parameters[:, :2] >= 0).all()
             & (parameters[:, :2] <= extent).all() & (parameters[:, 2] > 0).all()
             & (parameters[:, 4] != 0).all())
    torch._assert_async(valid, 'Invalid speed, domain or source parameters')
    packed_velocity = pack_fine((velocity-velocity_center)/velocity_scale)
    validity = pack_fine(velocity.new_ones((1, 401, 401))).expand(len(velocity), -1, -1, -1)
    mass = pack_fine(source_mass(parameters[:, :2], extent).to(velocity))
    coordinate = torch.linspace(0., 1., 201, dtype=velocity.dtype, device=velocity.device)
    z, x = torch.meshgrid(coordinate, coordinate, indexing='ij')
    spatial = torch.stack((x, z, (z == 0).to(velocity)), dim=0)[None].expand(len(velocity), -1, -1, -1)
    scales = torch.cat((extent/2000., extent.new_tensor([duration_s]))).expand(len(velocity), -1)
    wavelet = parameters[:, 2:]/parameters.new_tensor([15., .1, 1.])
    constants = torch.cat((scales, wavelet), dim=1).to(velocity)
    constants = constants[:, :, None, None].expand(-1, -1, 201, 201)
    return torch.cat((packed_velocity, validity, mass, spatial, constants), dim=1)
