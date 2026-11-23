"""Apply the zero-observation sourced readout to one frozen Marmousi shot."""

import torch

from sourced_pressure_fit import fit_sourced_pressure


class MarmousiSourcedReadout:
    """Keep one velocity and source fixed while adapting only pressure coefficients."""

    def __init__(self, model, velocity, shot_id, physical_source):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Pass an independently loaded, frozen pressure model in evaluation mode")
        self.model = model
        self.velocity = torch.as_tensor(velocity, device=model.device, dtype=torch.float32).detach().clone()
        if self.velocity.shape != (201, 201) or not bool((self.velocity.isfinite() & (self.velocity > 0)).all()):
            raise ValueError("Velocity must be a positive, finite 201 by 201 array in m/s")
        self.times = model.times.detach().double().cpu()
        if not torch.equal(self.times, torch.arange(801, dtype=torch.float64) * .005):
            raise ValueError("Expected the registered 801 queries over four seconds")
        roster = model.sources.detach().cpu()
        if roster.shape != (38, 5) or not 0 <= shot_id < 38:
            raise ValueError("Supply one registered shot index from the 38-source acquisition")
        self.source = torch.as_tensor(physical_source, dtype=torch.float64, device="cpu").clone()
        if (self.source.shape != (5,) or not torch.equal(self.source.to(roster), roster[shot_id])
                or self.source[1:].tolist() != [40., 8., .1875, 1.]):
            raise ValueError("Expected the Tianhe 40 m, 8 Hz, 0.1875 s source")
        self.pressure_scale = float(model.receiver_rms)
        self.adaptation = None
        with torch.no_grad():
            self.context = model.encode(self.velocity, roster[shot_id:shot_id+1].to(model.device))

    @torch.no_grad()
    def base_pressure(self, frame_ids):
        """Return normalized pressure with the registered hard initial and surface values."""
        raw = self.model.decode_pressure(self.context, frame_ids, normalized=True)
        if raw.shape != (1, len(frame_ids), 201, 201):
            raise ValueError("The frozen model must return one normalized full pressure plane per query")
        raw = raw[0]
        # Match the UL startup gate on frame indices. Its first three rows
        # give zero initial pressure and zero forward-difference initial rate.
        frames = torch.as_tensor(frame_ids, device=raw.device, dtype=raw.dtype)
        ramp = ((frames - 2.) / 8.).clamp(0., 1.)
        ramp = ramp.pow(3) * (10. + ramp * (-15. + 6. * ramp))
        interior = raw[:, 1:, :] * ramp[:, None, None]
        return torch.cat((torch.zeros_like(raw[:, :1, :]), interior), dim=1)

    @torch.no_grad()
    def fit(self, *, query_block=8):
        """Use only the frozen pressure queries and their local sourced equation."""
        self.adaptation = fit_sourced_pressure(self.base_pressure, self.velocity, self.source,
                                               self.times, self.pressure_scale, query_block=query_block)
        return self.adaptation.receipt

    @torch.no_grad()
    def pressure(self, frame_ids):
        if self.adaptation is None:
            return self.base_pressure(frame_ids)
        return self.adaptation.apply(self.base_pressure, frame_ids)

    @torch.no_grad()
    def b5(self, frame_ids):
        return self.model.observe(self.pressure(frame_ids))
