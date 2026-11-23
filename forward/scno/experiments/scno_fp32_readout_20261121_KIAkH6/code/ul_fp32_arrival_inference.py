"""Request-local U96 preparation and an isolated CUDA graph for its renderer."""

from contextlib import nullcontext
from time import perf_counter
from typing import NamedTuple

import torch
from torch.nn import functional as F

from joint_fno_decoder import _frame_indices
from ul_general_arrival_readout import GeneralLayeredArrivalReadout


class PreparedU96(NamedTuple):
    pulses: torch.Tensor
    aligned: torch.Tensor
    arrivals: torch.Tensor
    mask: torch.Tensor
    valid: torch.Tensor
    direct: torch.Tensor
    frequency: torch.Tensor


def prepare_render(readout, coefficients, arrivals, mask, frequency):
    """Rotate the coda once using the current request's direct arrivals."""
    pulses, late = coefficients
    direct = arrivals[..., -1]
    shift = (2. * torch.pi) * direct[..., None] * readout.temporal_modes_hz
    cosine, sine = shift.cos(), shift.sin()
    constant, real, imag = late.split((1, readout.harmonics, readout.harmonics), dim=-1)
    aligned = torch.cat((constant, real * cosine - imag * sine, real * sine + imag * cosine), dim=-1)
    return PreparedU96(pulses, aligned, arrivals, mask, mask.any(-1), direct, frequency)


def native_time_basis(native, modes):
    phase = (2. * torch.pi) * native[..., None] * modes
    basis = torch.cat((torch.ones_like(phase[..., :1]), phase.cos(), phase.sin()), dim=-1)
    ramp = ((native - .25) / .15).clamp(0., 1.)
    return basis, ramp.square() * (3. - 2. * ramp)


def render_prepared(prepared, times, basis, gate):
    pulses, aligned, arrivals, mask, valid, direct, f0 = prepared
    x = 2.6 * (times[..., :, None] - arrivals[..., None, :]) * f0[:, None, None, None]
    x2 = x.square()
    a, b, c, d, e = (weights.unsqueeze(-2) for weights in pulses.unbind(-1))
    envelope = torch.exp(-x2) * mask[..., None, :]
    polynomial = (a + c) + x * ((b + 3. * d) + x * ((3. * e - 2. * c) - x * (2. * d + x * e)))
    pressure = (envelope * polynomial).sum(-1)
    coda = torch.einsum("bpd,bnd->bpn", aligned, basis)
    pressure = pressure + coda * gate[:, None, :] * valid[..., None]
    front = (.75 + (times - direct[..., None]) * f0[:, None, None]).clamp(0., 1.)
    return pressure * front.square() * (3. - 2. * front) * (times > 0.)


def _layout(tensor):
    return (tensor.shape, tensor.stride(), tensor.dtype, tensor.device)


def _stamp(tensor):
    return (id(tensor), tensor.data_ptr(), tensor._version, _layout(tensor))


def _call_blocks(render, times):
    blocks = []
    for start in range(0, times.shape[-1], 8):
        stop = min(start + 8, times.shape[-1])
        block = render(times[:, None, start:stop], start, stop)
        blocks.append(block.transpose(1, 2).reshape(times.shape[0], stop - start, 200, 201))
    return torch.cat(blocks, dim=1)


class RenderGraph:
    """A fixed-shape renderer. Inputs are copied before each replay."""

    def __init__(self, fn, example, stamp):
        self.device = example[0].device
        self.owner = torch.cuda.current_stream(self.device)
        self.layouts = tuple(map(_layout, example))
        self.stamp = stamp
        self.static = tuple(torch.empty_like(t) for t in example)
        started = perf_counter()
        with torch.inference_mode(), torch.cuda.device(self.device):
            for dst, src in zip(self.static, example, strict=True):
                dst.copy_(src)
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(self.owner)
            with torch.cuda.stream(side):
                for _ in range(3):
                    warm = fn(*self.static)
                del warm
            self.owner.wait_stream(side)
            torch.cuda.synchronize(self.device)
            warm_done = perf_counter()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=side):
                self.output = fn(*self.static)
            torch.cuda.synchronize(self.device)
        self.setup_ms = {"warmup": (warm_done - started) * 1000,
                         "capture": (perf_counter() - warm_done) * 1000}

    def matches(self, inputs, stamp):
        return (self.stamp == stamp and self.layouts == tuple(map(_layout, inputs))
                and self.owner == torch.cuda.current_stream(self.device))

    def __call__(self, inputs):
        for dst, src in zip(self.static, inputs, strict=True):
            dst.copy_(src)
        self.graph.replay()
        # The caller assembles the final pressure into owned storage.
        return self.output


class U96ArrivalInference:
    """Full ArrivalSCNO backbone with an opt-in U96 execution path.

    Interface classification, geometry and the spatial head run per request.
    The graph captures only the fixed-shape renderer, with the geometry and
    coefficients passed as tensors. Arbitrary query lists use the same eager
    renderer. Model parameters and the original forward stay available.
    """

    def __init__(self, model, *, prepared=True, graph=False, stage=nullcontext):
        if type(model.readout) is not GeneralLayeredArrivalReadout:
            raise TypeError("The adapter targets the bound generalized U96 deployment")
        self.model, self.prepared, self.use_graph = model, prepared, graph
        self.stage = stage
        self._basis = self._basis_stamp = self._graph = None
        self.graph_setup = []
        self.graph_replays = 0
        self.last_eligible = False

    def _fixed_basis(self):
        times = self.model.full_times_s
        modes = self.model.readout.temporal_modes_hz
        stamp = (_stamp(times), _stamp(modes))
        if stamp != self._basis_stamp:
            self._basis = native_time_basis(times, modes)
            self._basis_stamp = stamp
        return self._basis

    def _render(self, times, pulses, late, arrivals, mask, f0):
        readout = self.model.readout
        return _call_blocks(lambda t, _i, _j: readout.render((pulses, late), t, arrivals, mask, f0), times)

    def _prepared_render(self, times, pulses, late, arrivals, mask, f0, basis, gate):
        ready = prepare_render(self.model.readout, (pulses, late), arrivals, mask, f0)
        return _call_blocks(lambda t, i, j: render_prepared(ready, t, basis[:, i:j], gate[:, i:j]), times)

    def _correction(self, rows, times, coefficients, geo, full):
        times = times if times.ndim == 2 else times[None].expand(len(geo.frequency), -1)
        args = (times, *coefficients, geo.times, geo.mask, geo.frequency)
        fn = self._render
        if self.prepared:
            with self.stage("ul.native_time_basis"):
                raw_basis, raw_gate = self._fixed_basis()
                basis, gate = raw_basis[rows], raw_gate[rows]
                batch = len(geo.frequency)
                if rows.ndim == 1:
                    basis = basis[None].expand(batch, -1, -1)
                    gate = gate[None].expand(batch, -1)
            args += (basis, gate)
            fn = self._prepared_render
        if not (self.use_graph and full and times.is_cuda and len(times) == 1):
            return fn(*args)

        # Profile/geometry calculations are outside this graph. The readout's
        # fixed modes are its only captured model storage.
        stamp = (_stamp(self.model.full_times_s), _stamp(self.model.readout.temporal_modes_hz),
                 self.prepared, torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32)
        if self._graph is None or not self._graph.matches(args, stamp):
            self._graph = RenderGraph(fn, args, stamp)
            self.graph_setup.append(self._graph.setup_ms)
        result = self._graph(args)
        self.graph_replays += 1
        return result

    @torch.inference_mode()
    def __call__(self, velocity, source, frame_indices=None, *, normalized=True):
        backbone = self.model.backbone
        with self.stage("ul.encode"):
            v, s = backbone._inputs(velocity, source)
            # Match the reference encoder and input validation in all variants.
            context = backbone.encode(v, s)
        full = frame_indices is None
        rows = (torch.arange(len(self.model.full_times_s), device=self.model.device) if full
                else _frame_indices(frame_indices, len(s), self.model.device))
        with self.stage("ul.backbone_decode"):
            prediction = backbone.decode(context, rows, normalized=True)
        v = v.expand(len(s), -1, -1, -1)
        with self.stage("ul.u96_branch"):
            self.last_eligible = bool(self.model.profiles(v)[-1].any())
        if self.last_eligible:
            with self.stage("ul.u96_geometry_and_head"):
                geometry = self.model.geometry(v, s)
                inherited = context.features[:, :, 1:].flatten(2).transpose(1, 2)
                coefficients = self.model.readout.coefficients(inherited, geometry.features)
                times = self.model.full_times_s[rows]
            with self.stage("ul.u96_render"):
                pressure = self._correction(rows, times, coefficients, geometry, full)
            prediction = prediction + F.pad(pressure, (0, 0, 1, 0))
        return prediction if normalized else self.model.normalizer.decode_pressure(prediction, s[:, 4])
