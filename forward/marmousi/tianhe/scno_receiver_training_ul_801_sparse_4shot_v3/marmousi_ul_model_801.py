"""Construct the uploaded UL SCNO for the saved Marmousi receiver axis."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from joint_fno_transport_model import TravelTimeFNOSCNO
from method_repair_model import CorrectedSCNOReadout
from model_v2 import build_model
from readout_conditioning_repair import absorb_neural_gain


class MediumInterfaceFeatures(nn.Module):
    """Add local velocity-gradient and curvature features to the FNO encoder."""

    version = "velocity_gradient_curvature_residual_v1"
    radii = (1, 3)
    hidden_width = 24

    def __init__(self, output_channels):
        super().__init__()
        self.spatial = nn.Conv2d(3 * len(self.radii), self.hidden_width, kernel_size=3, padding=1)
        self.project = nn.Conv2d(self.hidden_width, output_channels, kernel_size=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    @staticmethod
    def differences(velocity, radius):
        height, width = velocity.shape[-2:]
        field = F.pad(velocity, (radius,) * 4, mode="replicate")
        rows = slice(radius, radius + height)
        cols = slice(radius, radius + width)
        west = field[..., rows, :width]
        east = field[..., rows, 2 * radius:2 * radius + width]
        north = field[..., :height, cols]
        south = field[..., 2 * radius:2 * radius + height, cols]
        dx = (east - west) / (2 * radius)
        dz = (south - north) / (2 * radius)
        curvature = ((west - velocity) + (east - velocity) + (north - velocity)
                     + (south - velocity)) / radius ** 2
        return dx, dz, curvature

    def forward(self, normalized_velocity):
        channels = [value for radius in self.radii for value in self.differences(normalized_velocity, radius)]
        return self.project(F.silu(self.spatial(torch.cat(channels, dim=1))))

    def semantics(self):
        return {
            "implementation": self.version,
            "input": "training-normalized velocity",
            "features_per_radius": ["centered_dx", "centered_dz", "five_point_laplacian"],
            "radius_grid_cells": list(self.radii),
            "radius_m": [10 * radius for radius in self.radii],
            "gradient_units": "normalized velocity per grid cell",
            "curvature_units": "normalized velocity per grid cell squared",
            "padding": "replicate for finite differences",
            "hidden_width": self.hidden_width,
            "output_channels": self.project.out_channels,
            "initial_projection": "zero weights and bias",
            "ul_role": "BASE",
        }


class MarmousiTravelTimeFNOSCNO(TravelTimeFNOSCNO):
    def __init__(self, native_model, **kwargs):
        super().__init__(native_model, **kwargs)
        self.interface_features = None

    def enable_interface_features(self):
        self.interface_features = MediumInterfaceFeatures(self.width)
        self.initialization_receipt["new_parameters"].extend(
            "interface_features." + name for name, _ in self.interface_features.named_parameters())

    def checkpoint_semantics(self):
        return {
            **super().checkpoint_semantics(),
            "medium_interface_feature_correction": (
                self.interface_features.semantics() if self.interface_features is not None else None
            ),
        }

    def _encode_validated(self, velocity, source, *, inference_source=None):
        context = super()._encode_validated(velocity, source, inference_source=inference_source)
        if self.interface_features is None:
            return context
        # One correction per distinct input velocity. Broadcasting gives shared
        # media the same correction when the original model sees several shots.
        correction = self.interface_features(self.normalizer.encode_velocity(velocity))
        learned, explicit = context.features.split((self.width, 8), dim=1)
        return context._replace(features=torch.cat((learned + correction, explicit), dim=1))


def build_marmousi_ul(data, seed, device, query_block, *, interface_correction=True):
    """Build the same 800-basis, 800-BASE UL on the observed 801 physical times."""
    times = np.asarray(data.times, dtype=np.float64)
    if (times.shape != (801,) or not np.isfinite(times).all()
            or not np.allclose(times, np.arange(801) * .005, rtol=0., atol=5e-7)):
        raise ValueError("Expected 801 receiver times at 5 ms spacing from 0 to 4 seconds")
    if not isinstance(query_block, int) or query_block < 1:
        raise ValueError("The UL coefficient time block must be a positive integer")
    if not isinstance(interface_correction, bool):
        raise TypeError("interface_correction must be a boolean")
    template = SimpleNamespace(times=times.copy(), metadata=dict(data.metadata))
    config = {
        "seed": int(seed),
        "data": {"kind": "style_marm_joint201"},
        "model": {"family": "marmousi", "decoder_grid_size": 201},
    }
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        native = build_model(config, template, torch.device("cpu"))
        parent = MarmousiTravelTimeFNOSCNO.from_native(native)
        absorb_neural_gain(parent, startup_mode="fresh")
        model = CorrectedSCNOReadout(
            parent,
            nonlinear_details=False,
            coarsest_base=False,
            basis_size=800,
            base_hidden_width=800,
            time_chunk=query_block,
        )
        # Initialize only after the shared UL weights to keep seeded baselines paired.
        if interface_correction:
            parent.enable_interface_features()
    expected = torch.tensor(times, dtype=torch.float32)
    if not (torch.equal(parent.full_times_s, expected)
            and torch.equal(parent.propagation.native_times, expected)):
        raise RuntimeError("UL decoder and propagation must bind all 801 physical samples")
    model.repair_receipt["marmousi_receiver_axis"] = {
        "bound_frames": len(times),
        "spacing_s": .005,
        "end_s": float(times[-1]),
        "training_labels": "frozen physical B5 receiver samples",
    }
    model.repair_receipt["medium_interface_feature_correction"] = parent.checkpoint_semantics()[
        "medium_interface_feature_correction"
    ]
    return model.to(device)
