"""Time-channel SCNO with a supervised source response on medium-dependent travel times."""

from typing import NamedTuple
import torch

from joint_fno_model import JointFNOSCNO
from joint_fno_propagation import JointFNOPropagation
from joint_fno_traveltime import MediumTravelTime


class TravelContext(NamedTuple):
    features: object
    source: object
    propagation: object


class TravelTimeFNOSCNO(JointFNOSCNO):
    variant = "scno_joint_fno_travel_v1"

    def __init__(self, native_model, **kwargs):
        super().__init__(native_model, **kwargs)
        self.propagation = JointFNOPropagation(self.full_times_s).to(self.device)
        self.travel_time = MediumTravelTime().to(self.device)
        self.residual_gain = torch.nn.Parameter(self.full_times_s.new_tensor(.015))
        self.initialization_receipt["new_parameters"].append("residual_gain")
        self.initialization_receipt["new_parameters"].extend(
            "propagation." + name for name, _ in self.propagation.named_parameters())

    def checkpoint_semantics(self):
        return {**super().checkpoint_semantics(), "travel_time_basis": self.propagation_semantics()}

    def propagation_semantics(self):
        return {"type": "supervised_source_response", "implementation_version": 1,
                "traveltimes": "nine_three_segment_paths_on_51_grid",
                "parameters": {name: list(p.shape) for name, p in self.propagation.named_parameters()}}

    def _encode_validated(self, velocity, source, *, inference_source=None):
        base = super()._encode_validated(velocity, source, inference_source=inference_source)
        if len(velocity) == 1 and len(source) > 1:
            velocity = velocity.expand(len(source), -1, -1, -1)
        travel_times = self.travel_time(velocity, source)
        return TravelContext(*base, self.propagation.prepare(velocity, source, travel_times=travel_times))

    def propagation_factors(self, context, slots):
        physics = context.propagation
        return self.propagation.time_rows(physics, slots), self.propagation.spatial_basis(physics)
