"""Full400 Marmousi model with an orthogonal spatial detail readout."""
from contextvars import ContextVar
from pathlib import Path

import torch

from native400_runtime.r24_cf_preserved_operator import R24CFPreservedOperator
from native400_runtime.r25_cf_full400_operator import _Full400Mixin
from spectral_detail import SpectralDetailReadout
from spectral_gabor import _install_global_queries

PARENT = (Path.home() / "Data/acoustic-operator-learning/"
          "paper_assets_main_fresh_20261013_marmousi_20261019/models/initializers/marmousi.pt")
PARENT_SHA = "2638504b1c2806c29752aed5189c1d09fb0e5f4a6c746e91d91d8eb324adb8d6"
# An activation tap is scoped to the calling query, including concurrent calls.
# Recomputations after the scope exits cannot persist activations.
_capture = ContextVar("spectral_pressure_readout", default=None)


def _retain_fusion(module, args, output):
    active = _capture.get()
    if active is not None and active["fusion"] is module:
        active["frames"].append((output.coarse, output.local_residual))


def _retain_refined(module, args):
    active = _capture.get()
    if active is not None and active["readout"] is module:
        active["refined"].append(args[0])


class SpectralMarmousiOperator(_Full400Mixin, R24CFPreservedOperator):
    """Original pressure path plus a trainable residual in the detail subspace."""

    def __init__(self, config=None, device="cpu"):
        config = dict(config or {})
        super().__init__(
            device="cpu", family="marmousi",
            checkpoint_path=config.get("parent_checkpoint", PARENT),
            expected_checkpoint_sha=config.get("parent_checkpoint_SHA", PARENT_SHA),
            dense_time_block=config.get("dense_time_block", 1),
            refiner_time_block=config.get("refiner_time_block", 1),
            checkpoint_queries=config.get("checkpoint_queries", True),
        )
        self._full400_receipt()
        self._install_spectral_readout(config)
        self.to(device)

    def _install_spectral_readout(self, config):
        _install_global_queries(self.base)
        # The surrounding forecast checkpoints the full query under the
        # deterministic native scopes. A nested dense replay runs outside
        # those scopes and may replay a different operator graph.
        self.base.dense_outer_checkpoint = False
        time_features = self.base.coordinate_encoder.gabor.out_features
        self.refiner.spectral_detail = SpectralDetailReadout(time_features=time_features)
        self.base.fusion.register_forward_hook(_retain_fusion)
        self.refiner.output.register_forward_pre_hook(_retain_refined)
        self.spectral_detail_enabled = bool(config.get("spectral_detail_enabled", True))
        self.model_config = config
        self.initialization_receipt.update(
            architecture_variant="shared_temporal_gabor_polynomial_detail",
            model_branches=["medium", "source", "coordinate", "geometry", "fusion",
                            "dense", "scno", "refiner", "spectral_detail"],
            added_parameters=sum(p.numel() for p in self.refiner.spectral_detail.parameters()),
            added_persistent_buffers=2 * len(self.refiner.spectral_detail.transform.levels),
            spectral_domain=[200, 201], spectral_coarse_coefficients=784,
            spectral_detail_coefficients=[30200, 7296, 1920],
        )

    def _prepare_cf(self, c, f):
        prepared, grid, features = super()._prepare_cf(c, f)
        return prepared, grid, (features, c)

    def _query(self, prepared, grid, context, f, times):
        features, c = context
        if not self.spectral_detail_enabled:
            return super()._query(prepared, grid, features, f, times)
        active = dict(fusion=self.base.fusion, readout=self.refiner.output, frames=[], refined=[])
        token = _capture.set(active)
        try:
            base = super()._query(prepared, grid, features, f, times)
        finally:
            _capture.reset(token)
        if len(active["refined"]) != 1 or not active["frames"]:
            raise RuntimeError("expected the native fusion and refiner pressure activations")
        raw = torch.stack([torch.cat(part, dim=1) for part in zip(*active["frames"])], dim=-1)
        expected = (1, times.numel() * grid.height * grid.width, 2)
        if raw.shape != expected:
            raise RuntimeError(f"unexpected dense fusion result {raw.shape}, expected {expected}")
        fused = raw.reshape(-1, grid.height, grid.width, 2).permute(0, 3, 1, 2)
        time = self.base.coordinate_encoder.gabor(
            times, f[0, 3], f[0, 2], self.base.domain_t_s)
        detail = self.refiner.spectral_detail(fused, active["refined"][0], c, f, time)
        return base + detail[:, 0]

    def parameter_groups(self, base_lr, refiner_lr, weight_decay=1e-6):
        result = super().parameter_groups(base_lr, refiner_lr, weight_decay)
        for group in result:
            if group["group_name"] == "base_travel":
                group["group_name"] = "base_geometry"
        spectral = [p for p in self.refiner.spectral_detail.parameters() if p.requires_grad]
        if spectral:
            ids = {id(p) for p in spectral}
            for group in result:
                group["params"] = [p for p in group["params"] if id(p) not in ids]
            result.append(dict(params=spectral, lr=float(refiner_lr),
                               weight_decay=float(weight_decay), group_name="spectral_detail"))
        return result


def build_spectral_marmousi(config=None, device="cpu"):
    return SpectralMarmousiOperator(config, device)
