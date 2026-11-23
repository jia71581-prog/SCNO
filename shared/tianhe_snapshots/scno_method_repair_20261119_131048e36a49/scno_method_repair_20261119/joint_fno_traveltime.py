"""Source-conditioned direct and free-surface travel times from sampled slowness."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class MediumTravelTime(nn.Module):
    """Minimize slowness integrals over a fixed family of three-segment paths.

    Inputs are a positive velocity [B,1,H,W] in m/s and a source [B,5]
    whose first two coordinates are x and depth in metres. Output is [B,2,H,W]
    in seconds, ordered direct first arrival, then one free-surface reflection.
    The reflection uses an image source and the even extension of slowness.
    Coarse rays estimate effective slowness, which is interpolated and multiplied
    by the exact source-to-node distance on the output grid. The computation is
    independent of the number of pressure frames and uses no tensor-to-host reads.
    """

    def __init__(self, height=201, width=201, dx=10., *, dz=None, stride=4, samples_per_leg=8):
        super().__init__()
        dz = dx if dz is None else dz
        if (type(height) is not int or type(width) is not int or min(height, width) < 2
                or type(stride) is not int or stride < 1 or type(samples_per_leg) is not int or samples_per_leg < 1
                or not all(math.isfinite(step) and step > 0 for step in (dx, dz))):
            raise ValueError("travel-time grids require positive physical spacing and at least two nodes per axis")
        self.height, self.width = height, width
        self.length_x, self.length_z = (width - 1) * dx, (height - 1) * dz
        coarse_h = math.ceil((height - 1) / stride) + 1
        coarse_w = math.ceil((width - 1) / stride) + 1
        for name, size, length, shape in (
                ("x", width, self.length_x, (1, 1, 1, -1)), ("z", height, self.length_z, (1, 1, -1, 1)),
                ("ray_x", coarse_w, self.length_x, (1, 1, 1, -1)),
                ("ray_z", coarse_h, self.length_z, (1, 1, -1, 1))):
            self.register_buffer(name, torch.linspace(0, length, size).reshape(shape), persistent=False)
        # Normal displacements at the one-third and two-thirds waypoints, as fractions of ray length.
        # Opposite bends sample both sides of the chord. Asymmetric bends permit an early or late turn.
        bends = torch.tensor([(0., 0.), (-.10, -.10), (.10, .10), (-.25, -.25), (.25, .25),
                              (-.24, -.06), (.24, .06), (-.06, -.24), (.06, .24)])
        self.register_buffer("bend1", bends[:, 0].reshape(1, 1, -1, 1, 1), persistent=False)
        self.register_buffer("bend2", bends[:, 1].reshape(1, 1, -1, 1, 1), persistent=False)
        midpoint = (torch.arange(samples_per_leg, dtype=torch.float32) + .5) / samples_per_leg
        self.register_buffer("midpoints", midpoint.reshape(1, 1, 1, 1, -1, 1, 1), persistent=False)

    def forward(self, velocity, source):
        if (velocity.ndim != 4 or velocity.shape[1:] != (1, self.height, self.width)
                or source.shape != (velocity.shape[0], 5)):
            raise ValueError("supply one velocity grid and five physical source parameters per record")
        if velocity.device != self.x.device or source.device != self.x.device:
            raise ValueError("place the travel-time module, velocities, and sources on the same device")
        # Physical travel times retain FP32 under mixed-precision network training.
        with torch.autocast(device_type=velocity.device.type, enabled=False):
            return self._integrate(velocity.float().reciprocal(), source.float())

    def _integrate(self, slowness, source):
        batch = source.shape[0]
        sx = source[:, 0].reshape(batch, 1, 1, 1)
        depth = source[:, 1].reshape(batch, 1, 1, 1)
        sz = torch.cat((depth, -depth), dim=1)
        delta_x = (self.ray_x - sx).unsqueeze(2)
        delta_z = (self.ray_z - sz).unsqueeze(2)
        origin_x, origin_z = sx.unsqueeze(2), sz.unsqueeze(2)

        def waypoint(fraction, bend):
            x = (origin_x + fraction * delta_x - bend * delta_z).clamp(0., self.length_x)
            z = origin_z + fraction * delta_z + bend * delta_x
            direct = z[:, :1].clamp(0., self.length_z)
            # In unfolded coordinates a reflected ray crosses the surface exactly once.
            image = torch.minimum(torch.maximum(z[:, 1:], -depth.unsqueeze(2)), self.ray_z.unsqueeze(2))
            return x, torch.cat((direct, image), dim=1)

        x1, z1 = waypoint(1 / 3, self.bend1)
        x2, z2 = waypoint(2 / 3, self.bend2)
        image1, image2 = z1[:, 1:], z2[:, 1:]
        z1 = torch.cat((z1[:, :1], torch.minimum(image1, image2)), dim=1)
        z2 = torch.cat((z2[:, :1], torch.maximum(image1, image2)), dim=1)
        xs = torch.stack((origin_x.expand_as(x1), x1, x2, self.ray_x.unsqueeze(2).expand_as(x1)), dim=3)
        zs = torch.stack((origin_z.expand_as(z1), z1, z2, self.ray_z.unsqueeze(2).expand_as(z1)), dim=3)
        dx, dz = xs.diff(dim=3), zs.diff(dim=3)
        sample_x = xs[:, :, :, :-1].unsqueeze(4) + self.midpoints * dx.unsqueeze(4)
        sample_z = (zs[:, :, :, :-1].unsqueeze(4) + self.midpoints * dz.unsqueeze(4)).abs()
        grid = torch.stack((sample_x * (2. / self.length_x) - 1., sample_z * (2. / self.length_z) - 1.), dim=-1)
        sampled = F.grid_sample(slowness, grid.reshape(batch, -1, grid.shape[-2], 2),
                                mode="bilinear", padding_mode="border", align_corners=True)
        mean_slowness = sampled.reshape_as(sample_x).mean(dim=4)
        time = (torch.hypot(dx, dz) * mean_slowness).sum(dim=3).amin(dim=2)

        coarse_length = torch.hypot(delta_x.squeeze(2), delta_z.squeeze(2))
        source_grid = torch.stack((sx.flatten() * (2. / self.length_x) - 1.,
                                   depth.flatten() * (2. / self.length_z) - 1.), dim=-1).reshape(batch, 1, 1, 2)
        local = F.grid_sample(slowness, source_grid, padding_mode="border", align_corners=True)
        effective = torch.where(coarse_length > 0., time / coarse_length.clamp_min(1e-12), local)
        effective = F.interpolate(effective, size=(self.height, self.width), mode="bilinear", align_corners=True)
        native = effective * torch.hypot(self.x - sx, self.z - sz)
        # A folded surface path is also admissible for the first-arrival minimum.
        direct = torch.minimum(native[:, :1], native[:, 1:])
        reflected = torch.where(depth == 0., direct, native[:, 1:])
        reflected = torch.cat((direct[:, :, :1], reflected[:, :, 1:]), dim=2)
        return torch.cat((direct, reflected), dim=1)
