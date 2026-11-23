"""Arrival features for a horizontally varying interface between two constant layers."""

import torch
from torch.nn import functional as F

from ul_layered_arrival_readout import ArrivalGeometry, arrival_geometry, snell_time


GRID = 201
DX_M = 10.
EXTENT_M = (GRID - 1) * DX_M


def one_interface_profiles(velocity):
    """Return interface depth at each column, both speeds and the eligibility mask."""
    field = velocity[:, 0]
    jumps = field[:, 1:] != field[:, :-1]
    upper, lower = field[:, 0, 0], field[:, -1, 0]
    depth = (jumps.to(torch.int32).argmax(dim=1).to(field.dtype) + .5) * DX_M
    eligible = (
        (jumps.sum(dim=1) == 1).all(dim=1)
        & (field[:, 0] == upper[:, None]).all(dim=1)
        & (field[:, -1] == lower[:, None]).all(dim=1)
        & (upper > 0.) & (lower > 0.)
    )
    return depth, upper, lower, eligible


def _depth_at(profile, x_m):
    grid_coordinate = x_m.clamp(0., EXTENT_M) / DX_M
    left = grid_coordinate.long().clamp_max(GRID - 2)
    left_depth = profile.gather(1, left.flatten(1)).reshape_as(x_m)
    right_depth = profile.gather(1, (left + 1).flatten(1)).reshape_as(x_m)
    return torch.lerp(left_depth, right_depth, grid_coordinate - left)


def _interface_time(profile, x, z, sx, sz, source_speed, receiver_speed):
    """Fermat travel time with one reflection or transmission at the interface."""
    def time(crossing_x):
        crossing_z = _depth_at(profile, crossing_x)
        first_leg = torch.hypot(crossing_x - sx, crossing_z - sz) / source_speed
        second_leg = torch.hypot(crossing_x - x, crossing_z - z) / receiver_speed
        return first_leg + second_leg

    coarse = torch.linspace(0., EXTENT_M, 41, device=x.device, dtype=x.dtype)
    candidates = coarse[None, None].expand(len(profile), -1, -1)
    best = coarse[time(candidates).argmin(dim=-1, keepdim=True)]
    offsets = torch.arange(-2, 3, device=x.device, dtype=x.dtype)
    for step in (15., 5., 1.):
        candidates = (best + step * offsets).clamp(0., EXTENT_M)
        best = candidates.gather(-1, time(candidates).argmin(dim=-1, keepdim=True))
    return time(best).squeeze(-1)


def _curved_geometry(velocity, source, depth, upper, lower):
    batch = len(velocity)
    axis = torch.arange(GRID, device=velocity.device, dtype=velocity.dtype) * DX_M
    z_grid, x_grid = torch.meshgrid(axis[1:], axis, indexing='ij')
    x = x_grid.flatten()[None, :, None]
    z = z_grid.flatten()[None, :, None]
    sx, sz, f0, t0, amplitude = (column[:, None, None] for column in source.unbind(1))
    cu, cl = upper[:, None, None], lower[:, None, None]

    # Smooth the one-cell staircase for the ray calculation. Label the
    # material side using the sampled grid interface before smoothing.
    kernel = depth.new_tensor([1., 2., 3., 2., 1.])[None, None] / 9.
    smooth_depth = F.conv1d(F.pad(depth[:, None], (2, 2), mode='replicate'), kernel)[:, 0]
    source_depth = _depth_at(depth, sx)
    node_depth = depth[:, None].expand(-1, GRID - 1, -1).reshape(batch, -1, 1)
    source_above, node_above = sz < source_depth, z < node_depth
    same_layer = source_above == node_above
    both_above = source_above & node_above
    source_speed = torch.where(source_above, cu, cl)
    receiver_speed = torch.where(node_above, cu, cl)

    crossed = _interface_time(smooth_depth, x, z, sx, sz, source_speed, receiver_speed)
    image_source = _interface_time(smooth_depth, x, z, sx, -sz, source_speed, receiver_speed)
    image_receiver = _interface_time(smooth_depth, x, -z, sx, sz, source_speed, receiver_speed)
    dx, dz = x - sx, z - sz
    direct = torch.where(same_layer.squeeze(-1), (torch.hypot(dx, dz) / source_speed).squeeze(-1), crossed)
    # The free-surface return for two receivers in the lower medium uses
    # a locally horizontal two-crossing approximation. The ray families
    # with one crossing use the curved interface directly.
    upper_return = _depth_at(smooth_depth, sx) + _depth_at(smooth_depth, x.expand(batch, -1, -1))
    lower_return = (sz + z - upper_return).clamp_min(0.)
    two_crossings = snell_time(dx.abs(), upper_return, lower_return, cu, cl).squeeze(-1)
    upper_surface = (torch.hypot(dx, z + sz) / cu).squeeze(-1)
    one_crossing = torch.where(source_above.squeeze(-1), image_source, image_receiver)
    surface = torch.where(same_layer.squeeze(-1), torch.where(both_above.squeeze(-1), upper_surface,
                                                              two_crossings), one_crossing)
    origin = t0.squeeze(-1)
    times = torch.stack((crossed, crossed, surface, image_receiver, image_source), dim=-1) + t0
    mask = torch.cat((same_layer, ~same_layer, torch.ones_like(same_layer), both_above, both_above), dim=-1)
    mask = mask & (amplitude > 0.)
    direct = direct + origin

    def broadcast(values):
        return values.expand(batch, x.shape[1], 1)

    attributes = torch.cat((dx / EXTENT_M, dz / EXTENT_M, node_depth / EXTENT_M,
                            (z - node_depth) / EXTENT_M, broadcast(cu / 5000.), broadcast(cl / 5000.),
                            velocity[:, 0, 1:].reshape(batch, -1, 1) / 5000.,
                            broadcast(f0 / 15.), broadcast(t0), direct[..., None]), dim=-1)
    masked_times = torch.where(mask, times, torch.zeros_like(times))
    phase = (2. * torch.pi * f0) * masked_times
    features = torch.cat((attributes, masked_times, phase.sin() * mask, phase.cos() * mask,
                          mask.to(velocity.dtype)), dim=-1)
    return ArrivalGeometry(features, times, mask, direct, source[:, 2], source_depth[:, 0, 0],
                           torch.ones(batch, dtype=torch.bool, device=velocity.device))


def general_arrival_geometry(velocity, source):
    """Evaluate legacy flat-layer or curved-layer features in the common layout."""
    depth, upper, lower, eligible = one_interface_profiles(velocity)
    curved = eligible & (depth != depth[:, :1]).any(dim=1)
    if curved.all():
        return _curved_geometry(velocity, source, depth, upper, lower)
    curved_indices = curved.nonzero().flatten()
    if not curved_indices.numel():
        return arrival_geometry(velocity, source)
    flat_indices = (~curved).nonzero().flatten()
    flat_result = arrival_geometry(velocity[flat_indices], source[flat_indices])
    curved_result = _curved_geometry(velocity[curved_indices], source[curved_indices], depth[curved_indices],
                                     upper[curved_indices], lower[curved_indices])
    order = torch.cat((flat_indices, curved_indices)).argsort()
    return ArrivalGeometry(*(torch.cat((a, b))[order] for a, b in zip(flat_result, curved_result, strict=True)))
