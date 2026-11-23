"""Helpers used by the public training entry points."""
from math import floor, isfinite
import torch
from torch import nn
from torch.nn import functional as F

STAGGERED_GRADIENT_WEIGHTS = (1225 / 1024, -245 / 3072, 49 / 5120, -5 / 7168)

SOURCE_MASS_PER_UNIT_AMPLITUDE = 1.554e-8 * 5.**2

REACH = 2 * len(STAGGERED_GRADIENT_WEIGHTS) - 1

def _composed_weights():
    # D+ is indexed on [-3,4], D- on [-4,3]. Convolving the two arrays
    # therefore gives the centered pressure operator on [-7,7].
    weights = STAGGERED_GRADIENT_WEIGHTS
    first = (*(-value for value in reversed(weights)), *weights)
    return tuple(sum(first[i] * first[k - i] for i in range(len(first)) if 0 <= k - i < len(first))
                 for k in range(2 * len(first) - 1))

def staggered_pressure_laplacian(field, spacing_m):
    """Apply D^-D^+ to odd pressure, stopping before the exterior CPML."""
    if field.ndim < 2 or field.shape[-2] <= REACH:
        raise ValueError("The pressure or test field must cover the staggered spatial stencil")
    spacing_m = float(spacing_m)
    if not isfinite(spacing_m) or spacing_m <= 0.:
        raise ValueError("Grid spacing must be finite and positive")
    q = field.double().reshape(-1, 1, *field.shape[-2:])
    weights = q.new_tensor(_composed_weights())
    xx = F.conv2d(q, weights.reshape(1, 1, 1, -1), padding=(0, REACH))
    upper = -q[:, :, 1:REACH+1, :].flip(-2)
    odd = F.pad(torch.cat((upper, q), dim=-2), (0, 0, 0, REACH))
    zz = F.conv2d(odd, weights.reshape(1, 1, -1, 1))
    return ((xx + zz) / spacing_m**2).reshape_as(field)

def bilinear_source_nodes(x_m, z_m, spacing_m):
    """Yield the four frozen nodal injection weights before multiplying source mass."""
    x, z = x_m / spacing_m, z_m / spacing_m
    ix, iz = floor(x), floor(z)
    ax, az = x - ix, z - iz
    return tuple(((ix + ox) * spacing_m, (iz + oz) * spacing_m, wx * wz)
                 for oz, wz in ((0, 1. - az), (1, az)) for ox, wx in ((0, 1. - ax), (1, ax)))

def triangular_ricker(times_s, frequency_hz, delay_s, observation_dt_s):
    """Exact triangular average of Ricker at the three-frame stencil's times."""
    times = torch.as_tensor(times_s, dtype=torch.float64)
    step = float(observation_dt_s)
    rate = torch.pi * frequency_hz
    displacement = times - delay_s

    def second_primitive(offset):
        return -torch.exp(-(rate * offset).square()) / (2. * rate**2)

    return (second_primitive(displacement + step) - 2. * second_primitive(displacement)
            + second_primitive(displacement - step)) / step**2

def _bump(position):
    return (1. - position.square()).clamp_min(0.).pow(4)

class SourceForcingMoment(nn.Module):
    """Compare local native pressure's weak source trace to the frozen Ricker.

    Outside CPML, p_t/c² = div^- u + I * primitive, and u_t = grad^+ p.
    The triangular temporal average yields Dtt(p) exactly. For its spatial
    term, the three-frame quadrature (1, 10, 1)/12 is fourth-order in the
    observation step. Spatial reduction uses odd-surface test weights and the
    actual composed staggered derivative, on the native pressure grid. Frozen
    fine-grid bilinear quadrature binds the source mass. The coarse pressure
    moment is an approximation to the fine archived solver's source response.
    """

    def __init__(self, source, *, pressure_scale, spacing_m=10., fine_spacing_m=5., observation_dt_s=.005,
                 shape_zx=(201, 201)):
        super().__init__()
        source = torch.as_tensor(source, dtype=torch.float64).cpu()
        if source.shape != (5,) or not torch.isfinite(source).all():
            raise ValueError("Expected one finite source (x, z, f0, delay, amplitude)")
        xs, zs, freq, delay, amplitude = source.tolist()
        h, fine, dt, scale = map(float, (spacing_m, fine_spacing_m, observation_dt_s, pressure_scale))
        nz, nx = shape_zx
        if min(nx, nz) <= 2 * REACH or any(not isfinite(item) or item <= 0. for item in (h, fine, dt, scale, freq, amplitude)):
            raise ValueError("Expected positive acquisition scales and physical support")
        if not (0. <= delay and 0. < zs < (nz - 1) * h):
            raise ValueError("The source must lie below the free surface and inside the pressure grid")
        side = min(xs, (nx - 1) * h - xs)
        horizontal_radius = min(80., .9 * (side - REACH * h))
        if horizontal_radius < 1.5 * h or zs + 85. + REACH * h >= (nz - 1) * h:
            raise ValueError("Weak tests and their stencils must stay out of lateral and bottom CPML")
        radius_x = source.new_tensor((.8 * horizontal_radius, horizontal_radius))[:, None, None]
        radius_z = source.new_tensor((55., 85.))[:, None, None]

        def weight(x, z):
            horizontal = _bump((x - xs) / radius_x)
            vertical = _bump((z - zs) / radius_z) - _bump((z + zs) / radius_z)
            return horizontal * vertical

        z = torch.arange(nz, dtype=torch.float64)[None, :, None] * h
        x = torch.arange(nx, dtype=torch.float64)[None, None, :] * h
        test = weight(x, z)
        laplacian = staggered_pressure_laplacian(test, h)
        support = torch.nonzero((test != 0.).any(0) | (laplacian != 0.).any(0), as_tuple=False)
        (z0, x0), (z1, x1) = support.amin(0).tolist(), support.amax(0).tolist()
        self.region = (slice(z0, z1 + 1), slice(x0, x1 + 1))
        self.full_shape = tuple(shape_zx)
        self.dt = dt
        self.h = h
        self.frequency_hz = freq
        self.delay_s = delay
        self.pressure_scale = scale
        self.half_widths_m = (radius_x.flatten().tolist(), radius_z.flatten().tolist())
        self.register_buffer("source", source)
        self.register_buffer("test", test[..., self.region[0], self.region[1]].contiguous())
        self.register_buffer("laplacian_test", laplacian[..., self.region[0], self.region[1]].contiguous())
        self.register_buffer("fine_source_moment", SOURCE_MASS_PER_UNIT_AMPLITUDE * amplitude * sum(
            bilinear * weight(source.new_tensor(sx), source.new_tensor(sz)).flatten()
            for sx, sz, bilinear in bilinear_source_nodes(xs, zs, fine)))
        self.register_buffer("native_source_moment", SOURCE_MASS_PER_UNIT_AMPLITUDE * amplitude * sum(
            bilinear * weight(source.new_tensor(sx), source.new_tensor(sz)).flatten()
            for sx, sz, bilinear in bilinear_source_nodes(xs, zs, h)))

    def pressure_moment(self, pminus, p0, pplus, velocity):
        if pminus.shape != p0.shape or pplus.shape != p0.shape or p0.ndim != 3 or tuple(p0.shape[-2:]) != self.full_shape:
            raise ValueError("Pressure must be three matched [time,z,x] arrays on the registered grid")
        c = torch.as_tensor(velocity, dtype=torch.float64, device=p0.device)
        if c.shape != self.full_shape:
            raise ValueError("Physical velocity must match the nodal pressure grid")
        z, x = self.region
        before, center, after = (value[..., z, x].double() for value in (pminus, p0, pplus))
        inverse_c2 = c[z, x].square().reciprocal()
        dtt = (before - 2. * center + after) / self.dt**2
        weighted_pressure = (before + 10. * center + after) / 12.
        mass_term = torch.einsum("tij,mij->tm", dtt * inverse_c2, self.test.to(dtt))
        laplace_term = torch.einsum("tij,mij->tm", weighted_pressure, self.laplacian_test.to(dtt))
        return self.h**2 * (mass_term - laplace_term)

    def signature(self, pminus, p0, pplus, velocity, times_s):
        """Return predicted and known dimensionless source signatures [time,test]."""
        observed = self.pressure_moment(pminus, p0, pplus, velocity)
        times = torch.as_tensor(times_s, device=p0.device, dtype=torch.float64)
        if times.shape != (p0.shape[0],):
            raise ValueError("Supply the physical center time of each consecutive pressure triplet")
        target = triangular_ricker(times, self.frequency_hz, self.delay_s, self.dt)[:, None]
        curve = observed * (self.pressure_scale / self.fine_source_moment.to(observed))[None]
        return curve, target.expand_as(curve)

    def forward(self, pminus, p0, pplus, velocity, times_s):
        prediction, target = self.signature(pminus, p0, pplus, velocity, times_s)
        energy = target.square().sum(dim=0)
        if not bool(torch.all(energy > 1.e-12)):
            raise ValueError("Source center times need to cover nonzero forcing")
        return ((prediction - target).square().sum(dim=0) / energy).mean()
