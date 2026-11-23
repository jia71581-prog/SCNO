"""Strict immutable configuration schema for V3 experiments."""
from __future__ import annotations
from scno_release_paths import local_asset as _scno_local_asset

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, TypeVar


ALLOWED_MEDIUM_TYPES = ("uniform", "layered", "marmousi")
# Kernel count of the scattering head's reflectivity stack, mirrored here so a
# malformed scatter_dilations schedule is rejected at config load without importing
# torch.  Kept in sync by test_receptive_field_scatter.py, which asserts this equals
# ScatteringIntegralHead.REFLECTIVITY_KERNELS.
SCATTER_REFLECTIVITY_KERNELS = (5, 3)
_T = TypeVar("_T")


def _make(cls: type[_T], values: Mapping[str, Any] | None) -> _T:
    payload = dict(values or {})
    unknown = sorted(set(payload) - {item.name for item in fields(cls)})
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {unknown}")
    return cls(**payload)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    spectral_rank: int = 40
    modes: tuple[int, ...] = (20, 16, 12, 8)
    dense_modes: tuple[int, ...] = (20, 16)
    mionet_rank: int = 128
    token_count: int = 64
    heads: int = 4
    position_bands: int = 4
    fourier_bands: int = 6
    gabor_scales_s: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1)
    ray_samples: int = 12
    dense_time_block: int = 4
    dense_equivariant: bool = False
    dense_radial_cutoff: int = 0
    # V18-A speed branch (20260928): per-channel-gated dilation-2/4 depthwise
    # branches summed into each dense block's local path before the existing
    # pointwise.  Off (default) adds no parameters and is stripped from the
    # digest; the enabled model's initial function equals the baseline bitwise
    # (gates start at exact zero).
    local_multi_dilation: bool = False
    # v18.5 branch-heavy (20260929): U-shaped medium-pyramid refinement, zero-init
    # head on the full-resolution level.  Off (default) adds no parameters.
    medium_u_enhance: bool = False
    # V18 U-FNO: lightweight 201->101->51 U-shaped hidden-space residual path.
    # Inert defaults are stripped from the config digest for legacy compatibility.
    u_path_enabled: bool = False
    u_path_channels: tuple[int, int, int] = (16, 32, 48)
    u_path_output_scale: float = 0.01
    # V18 physics prior: fixed eighth-order pressure-space features on the saved
    # 10 m grid, projected into the dense hidden state. This is not a time solver.
    fixed_wave_physics: bool = False
    # Optional local spatial path: a narrow U-shaped encoder-decoder with skips
    # that supplies the multi-scale local coupling the coarse fusion path lacks.
    # Empty tuple (the default) adds no parameters and leaves every existing
    # state_dict key untouched, so legacy checkpoints keep migrating unchanged.
    local_unet_channels: tuple[int, ...] = ()
    local_unet_rank: int = 24
    # Multi-arrival phase channels (P2, 20260925): per-record precomputed
    # reflection / free-surface-multiple travel-time maps enter the travel
    # branch through a zero-initialised projection.  "off" (default) adds no
    # parameters and is stripped from the digest.
    phase_mode: str = "off"
    phase_interfaces: int = 3
    phase_harmonics: int = 2
    # Multi-arrival additive head (arrival_head.py, 20260925): enumerated Born
    # layer on the fusion output, zero-initialised output layer.  Off (default)
    # adds no parameters and is stripped from the digest.
    arrival_head: bool = False
    arrival_head_hidden: int = 128
    arrival_head_harmonics: int = 2
    arrival_head_arrivals: int = 8
    # Coarse-solver input channel (coarse_channel.py, 20260926): zero-initialised
    # additive term over [own prediction, coarse field].  Off (default) adds no
    # parameters and is stripped from the digest.  The field itself is bound at
    # run time from the coarse cache, not from the spec.
    coarse_channel: bool = False
    coarse_channel_hidden: int = 0
    coarse_channel_gain: float = 1.0
    # Born single-scattering head (scattering_head.py, 20260926): continuous
    # scattering integral over an ``scatter_grid`` x ``scatter_grid`` scatterer
    # lattice, in place of the enumerated arrival set.  Off (default) adds no
    # parameters and is stripped from the digest.  Travel-time tables are bound at
    # run time from the scatter cache (scatter_binding.py), not from the spec.
    scattering_head: bool = False
    scatter_grid: int = 51
    scatter_harmonics: int = 1
    scatter_channels: int = 16
    scatter_hidden: int = 32
    # Receptive field of the scattering head's reflectivity/amplitude branch
    # (20260926).  One dilation per convolution of the branch, default (1, 1) =
    # the historical 5x5 + 3x3 stack, bitwise unchanged.  Dilation adds NO
    # parameters, so this selects a window, not a capacity.  Sized in LATTICE
    # cells, so the same schedule spans a different metric window per
    # scatter_grid; see ScatteringIntegralHead.receptive_field_m.  Inert default
    # is stripped from the digest, so pre-existing checkpoints are unaffected.
    scatter_dilations: tuple[int, ...] = (1, 1)
    # Per-record family gate (family_gate.py, 20260926): the phase channel and the
    # scattering head are CLOSED for a record whose family is not listed here, so
    # e.g. uniform stays in the pool and in the loss while carrying neither branch.
    # The list is deliberate inclusion (empty = every family open); a model built
    # without the gate ignores it.
    phase_gated_families: tuple[str, ...] = ()
    scatter_ungated_families: tuple[str, ...] = ("layered", "marmousi")
    # UL31 reflection branch (dense.py ReflectionBranch, 20260929): interface-aware
    # multiplicative-interaction correction inside the dense decoder, zero-init 1ch
    # head.  Off (default) adds no parameters/keys and is stripped from the digest.
    reflection_branch: bool = False
    reflection_channels: int = 32
    # Free-surface output factor: "legacy_tanh20" keeps tanh(clamp(z,0)/20)^2;
    # "exact_row" hard-zeros ONLY z=0 ((z>0) mask: z=0 -> 0, z=10m/20m -> 1);
    # "ramp_exact_row" (boundary hotfix 20260929) migrates legacy -> exact_row
    # CONTINUOUSLY over free_surface_blend_updates optimizer updates via a
    # persistent blend buffer (alpha=0 is bitwise legacy, alpha=1 bitwise
    # exact_row).  Measured on the same UL30@1250 checkpoint, an instant
    # legacy->exact_row switch jumps uniform full relL2 0.01844 -> 0.16405 and
    # z=20 receiver relL2 0.02589 -> 0.72822, so long runs must never switch
    # discontinuously.
    free_surface_mode: str = "legacy_tanh20"
    # Ramp length in optimizer updates.  Meaningless outside ramp_exact_row and
    # stripped from the digest there (any value); under ramp it binds identity.
    free_surface_blend_updates: int = 500
    # L2 v2 single-surface mirror carrier; "off" keeps the legacy function and
    # (via digest stripping below) the legacy config digest.
    carrier_mode: str = "off"
    ic_frames: int = 0
    domain_x_m: float = 2000.0
    domain_z_m: float = 2000.0
    domain_t_s: float = 1.0


@dataclass(frozen=True)
class DataConfig:
    source_h5: str = _scno_local_asset('datasets/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5')
    manifest_json: str = _scno_local_asset('datasets/processed/grouped_v3_manifest.json')
    normalization_json: str = _scno_local_asset('datasets/processed/grouped_v3_normalization.json')
    train_cache: str = _scno_local_asset('datasets/processed/grouped_dual_head_v2_train.h5')
    validation_cache: str = _scno_local_asset('datasets/processed/grouped_dual_head_v2_validation.h5')
    allowed_medium_types: tuple[str, ...] = ALLOWED_MEDIUM_TYPES
    train_families: tuple[str, ...] = ALLOWED_MEDIUM_TYPES
    expected_train_records: int = 2240
    expected_validation_records: int = 480
    continuous_fraction: float = 0.125
    gate_sample_id: str = ""
    gate_sample_ids: tuple[str, ...] = ()
    ic_frames: int = 0
    dense_time_steps: int = 3
    active_horizon_s: float = 0.60
    time_sampling_policy: str = "legacy"
    # Anchor-time contract (L1): fraction of record visits whose IC window is
    # re-anchored at a uniformly drawn stored frame instead of the source
    # onset.  0.0 keeps every pre-existing config bit-identical (and is
    # stripped from the digest while inert).
    anchor_fraction: float = 0.0
    # Residual-adaptive sampling.  Off by default so every pre-existing config
    # keeps its exact behaviour and digest; the adaptive arm turns these on.
    residual_adaptive_space: bool = False
    residual_adaptive_time: bool = False
    residual_ema_gamma: float = 0.95
    query_uniform_floor: float = 0.2
    # Activation checkpointing on the dense spectral blocks.  DDP-safe through
    # the non-reentrant implementation; off was a workaround for the reentrant
    # "parameters marked ready twice" abort, which no longer applies.
    dense_checkpoint: bool = True
    # Split the dense supervision window into chunks of this many frames and
    # accumulate gradients across them.  The dense loss is a sum over frames, so
    # the accumulated gradient equals the single-pass one up to floating-point
    # summation order, while peak activation memory scales with the chunk rather
    # than the whole window.  0 disables chunking (single pass over all frames).
    dense_chunk_frames: int = 0
    # Also checkpoint the per-time-step dense call, not just the blocks inside
    # it.  Without this, peak memory grows by roughly 0.7 GiB per supervised
    # frame because every time step of the window keeps its activations live;
    # with it, peak is set by a single time step.  Costs recomputation on the
    # backward pass, which is what makes a 51-frame window fit in 24 GiB.
    dense_outer_checkpoint: bool = False


@dataclass(frozen=True)
class LossConfig:
    lwc84_weight: float = 0.0
    lwc84_ramp_updates: int = 374
    point: float = 1.0
    frame: float = 1.0
    complex_spectrum: float = 0.15
    spectral_phase: float = 0.1
    spatial_gradient: float = 0.1
    time_difference: float = 0.1
    consistency: float = 0.05
    phase_energy_fraction: float = 1.0e-4
    relative_energy_floor_fraction: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str = "adamw"
    batch_records: int = 1
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    epochs: int = 1
    max_steps: int = 2
    evaluation_every: int = 1
    steps_per_epoch: int = 1
    query_points_per_step: int = 512
    workers: int = 0
    prefetch_factor: int = 2
    seed: int = 17
    smoke: bool = True
    device: str = "cuda"
    micro_batch_records: int = 1
    checkpoint_dir: str = "artifacts/grouped_ufno_mionet_v3/smoke/checkpoints"
    gradient_clip: float = 1.0
    plateau_patience: int = 20
    plateau_min_delta: float = 1.0e-4
    lbfgs_learning_rate: float = 0.5
    lbfgs_max_iter: int = 10
    lbfgs_history_size: int = 20


@dataclass(frozen=True)
class V3Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "V3Config":
        payload = dict(values or {})
        unknown = sorted(set(payload) - {"model", "data", "loss", "train"})
        if unknown:
            raise ValueError(f"unknown V3Config keys: {unknown}")

        model_values = dict(payload.get("model") or {})
        if "modes" in model_values:
            model_values["modes"] = tuple(int(v) for v in model_values["modes"])
        if "dense_modes" in model_values:
            model_values["dense_modes"] = tuple(int(v) for v in model_values["dense_modes"])
        if "gabor_scales_s" in model_values:
            model_values["gabor_scales_s"] = tuple(float(v) for v in model_values["gabor_scales_s"])
        if "u_path_channels" in model_values:
            request = tuple(int(v) for v in model_values["u_path_channels"])
            if len(request) != 3 or any(v <= 0 for v in request):
                raise ValueError("u_path_channels must be exactly three positive integers")
            model_values["u_path_channels"] = request
        if "phase_gated_families" in model_values:
            requested = tuple(str(v) for v in model_values["phase_gated_families"])
            unknown = sorted(set(requested) - set(ALLOWED_MEDIUM_TYPES))
            if unknown:
                raise ValueError(
                    f"phase_gated_families must be a subset of uniform/layered/marmousi, "
                    f"received {unknown}")
            model_values["phase_gated_families"] = requested
        if "scatter_ungated_families" in model_values:
            requested = tuple(str(v) for v in model_values["scatter_ungated_families"])
            unknown = sorted(set(requested) - set(ALLOWED_MEDIUM_TYPES))
            if unknown:
                raise ValueError(
                    f"scatter_ungated_families must be a subset of uniform/layered/marmousi, "
                    f"received {unknown}")
            model_values["scatter_ungated_families"] = requested
        if "scatter_dilations" in model_values:
            # Coerce to a tuple of ints so the inert-default digest strip matches
            # (JSON/YAML hand this over as a list) and so a bad dilation is caught
            # here rather than inside the module build.  The length must match the
            # reflectivity stack's kernel count; the count is asserted against the
            # module constant in the receptive-field test rather than imported here,
            # to keep config.py free of a torch import.
            request = tuple(int(v) for v in model_values["scatter_dilations"])
            if not request or any(v < 1 for v in request):
                raise ValueError(
                    f"scatter_dilations must be a non-empty sequence of ints >= 1, "
                    f"received {request}")
            if len(request) != len(SCATTER_REFLECTIVITY_KERNELS):
                raise ValueError(
                    f"scatter_dilations must have one entry per reflectivity kernel "
                    f"({SCATTER_REFLECTIVITY_KERNELS}), received {request}")
            model_values["scatter_dilations"] = request
        model = _make(ModelConfig, model_values)

        data_values = dict(payload.get("data") or {})
        if "allowed_medium_types" in data_values:
            requested = tuple(str(v) for v in data_values["allowed_medium_types"])
            if requested != ALLOWED_MEDIUM_TYPES:
                raise ValueError(
                    "allowed_medium_types is fixed to uniform, layered, marmousi; "
                    f"received {requested}"
                )
            data_values["allowed_medium_types"] = requested
        if "train_families" in data_values:
            requested = tuple(str(v) for v in data_values["train_families"])
            unknown = set(requested) - set(ALLOWED_MEDIUM_TYPES)
            if not requested or unknown:
                raise ValueError(
                    "train_families must be a non-empty subset of uniform/layered/marmousi; "
                    f"received {requested}"
                )
            data_values["train_families"] = requested
        if "gate_sample_ids" in data_values:
            data_values["gate_sample_ids"] = tuple(
                str(value) for value in data_values["gate_sample_ids"]
            )
        data = _make(DataConfig, data_values)
        loss = _make(LossConfig, payload.get("loss"))
        train = _make(TrainConfig, payload.get("train"))
        cls._validate(model, data, loss, train)
        return cls(model=model, data=data, loss=loss, train=train)

    @staticmethod
    def _validate(
        model: ModelConfig,
        data: DataConfig,
        loss: LossConfig,
        train: TrainConfig,
    ) -> None:
        if model.width <= 0 or model.spectral_rank <= 0 or model.mionet_rank <= 0:
            raise ValueError("model widths and ranks must be positive")
        if (
            not model.modes
            or not model.dense_modes
            or any(mode <= 0 for mode in (*model.modes, *model.dense_modes))
        ):
            raise ValueError("spectral modes must be a non-empty positive sequence")
        if model.ray_samples < 2 or model.dense_time_block <= 0:
            raise ValueError("ray_samples and dense_time_block are invalid")
        if model.dense_radial_cutoff < 0:
            raise ValueError("dense_radial_cutoff must be nonnegative")
        if model.dense_equivariant and model.dense_radial_cutoff <= 0:
            raise ValueError(
                "dense_equivariant requires a positive dense_radial_cutoff"
            )
        if len(model.u_path_channels) != 3 or any(v <= 0 for v in model.u_path_channels):
            raise ValueError("u_path_channels must be exactly three positive integers")
        if not 0.0 < model.u_path_output_scale <= 1.0:
            raise ValueError("u_path_output_scale must lie in (0,1]")
        if model.free_surface_mode not in ("legacy_tanh20", "exact_row", "ramp_exact_row"):
            raise ValueError(
                "free_surface_mode must be legacy_tanh20, exact_row or ramp_exact_row"
            )
        if model.free_surface_blend_updates <= 0:
            raise ValueError("free_surface_blend_updates must be positive")
        if model.reflection_channels <= 0:
            raise ValueError("reflection_channels must be positive")
        if data.allowed_medium_types != ALLOWED_MEDIUM_TYPES:
            raise ValueError("allowed_medium_types violates the V3 data contract")
        if (
            not data.train_families
            or set(data.train_families) - set(ALLOWED_MEDIUM_TYPES)
        ):
            raise ValueError(
                "train_families must be a non-empty subset of uniform/layered/marmousi"
            )
        if not 0.0 <= data.continuous_fraction < 1.0:
            raise ValueError("continuous_fraction must be in [0,1)")
        if model.ic_frames < 0 or data.ic_frames < 0:
            raise ValueError("ic_frames must be nonnegative")
        if model.ic_frames != data.ic_frames:
            raise ValueError("model.ic_frames and data.ic_frames must agree")
        if data.dense_time_steps < 1:
            raise ValueError("dense_time_steps must be at least 1")
        if data.dense_chunk_frames < 0:
            raise ValueError("dense_chunk_frames must be nonnegative")
        if data.active_horizon_s <= 0:
            raise ValueError("active_horizon_s must be positive")
        if not 0.0 <= data.anchor_fraction <= 1.0:
            raise ValueError("anchor_fraction must lie in [0,1]")
        if data.anchor_fraction > 0.0 and data.ic_frames < 1:
            raise ValueError("anchor_fraction requires ic_frames >= 1")
        if data.time_sampling_policy not in ("legacy", "full_support", "residual_adaptive"):
            raise ValueError(
                "time_sampling_policy must be legacy, full_support or residual_adaptive"
            )
        if data.residual_adaptive_time and data.time_sampling_policy != "residual_adaptive":
            raise ValueError(
                "residual_adaptive_time requires time_sampling_policy=residual_adaptive"
            )
        if not 0.0 < data.residual_ema_gamma <= 1.0:
            raise ValueError("residual_ema_gamma must lie in (0,1]")
        if not 0.0 <= data.query_uniform_floor <= 1.0:
            raise ValueError("query_uniform_floor must lie in [0,1]")
        if not 0.0 <= loss.lwc84_weight <= 1.0 or not isinstance(loss.lwc84_ramp_updates, int) or loss.lwc84_ramp_updates < 1:
            raise ValueError("invalid LWC weight or ramp")
        if any(getattr(loss, item.name) < 0 for item in fields(loss)):
            raise ValueError("loss values must be nonnegative")
        if loss.relative_energy_floor_fraction > 1.0:
            raise ValueError("relative_energy_floor_fraction must be in [0,1]")
        if train.optimizer not in {"adamw", "lbfgs"}:
            raise ValueError(f"unsupported optimizer: {train.optimizer}")
        if (
            train.batch_records <= 0
            or train.max_steps <= 0
            or train.gradient_clip <= 0
            or train.evaluation_every <= 0
            or train.steps_per_epoch <= 0
            or train.query_points_per_step <= 0
            or train.micro_batch_records <= 0
        ):
            raise ValueError("training batch, micro-batch, and step counts must be positive")
        if train.micro_batch_records > train.batch_records:
            raise ValueError("micro_batch_records must not exceed batch_records")
        if train.device not in {"cpu", "cuda"}:
            raise ValueError("training device must be cpu or cuda")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "V3Config":
        import yaml

        with Path(path).open(encoding="utf8") as handle:
            return cls.from_mapping(yaml.safe_load(handle) or {})

    def digest(self) -> str:
        payload = asdict(self)
        if payload["loss"].get("lwc84_weight") == 0.0:
            payload["loss"].pop("lwc84_weight", None)
            if payload["loss"].get("lwc84_ramp_updates") == 374:
                payload["loss"].pop("lwc84_ramp_updates", None)
        # Fields added after the v3 checkpoints were written are omitted while
        # they hold their legacy default, so existing runs keep their digest and
        # stay resumable.  Any non-default value must change the digest.
        if payload["data"].get("time_sampling_policy") == "legacy":
            payload["data"].pop("time_sampling_policy", None)
        # Residual-adaptive sampling was added later still; its defaults are
        # off, so strip them while inert and let an enabled arm change the
        # digest (which is what marks it as a new experiment identity).
        for name, default in (
            ("residual_adaptive_space", False),
            ("residual_adaptive_time", False),
            ("residual_ema_gamma", 0.95),
            ("query_uniform_floor", 0.2),
            ("dense_checkpoint", True),
            ("dense_chunk_frames", 0),
            ("dense_outer_checkpoint", False),
            ("anchor_fraction", 0.0),
        ):
            if payload["data"].get(name) == default:
                payload["data"].pop(name, None)
        # The equivariant radial dense kernel was added later still; strip its
        # inert defaults so pre-existing checkpoints keep their digest.
        for name, default in (
            ("dense_equivariant", False),
            ("dense_radial_cutoff", 0),
            # V18-A multi-dilation local path (20260928): default-off must not
            # move old digests; enabling it is a new experiment identity.
            ("local_multi_dilation", False),
            ("u_path_enabled", False),
            ("u_path_channels", (16, 32, 48)),
            ("u_path_output_scale", 0.01),
            ("fixed_wave_physics", False),
            # L2 carrier (20260914): default-off must not move old digests;
            # any enabled mode is a new experiment identity (audit R4).
            ("carrier_mode", "off"),
            ("phase_mode", "off"),
            ("phase_interfaces", 3),
            ("phase_harmonics", 2),
            ("arrival_head", False),
            ("arrival_head_hidden", 128),
            ("arrival_head_harmonics", 2),
            ("arrival_head_arrivals", 8),
            ("coarse_channel", False),
            ("coarse_channel_hidden", 0),
            ("coarse_channel_gain", 1.0),
            # Born scattering head + family gate (20260926): inert defaults stay
            # stripped so pre-existing digests are unchanged; enabling any of them
            # is a new experiment identity.
            ("scattering_head", False),
            ("scatter_grid", 51),
            ("scatter_harmonics", 1),
            ("scatter_channels", 16),
            ("scatter_hidden", 32),
            # Inert default only: a non-default schedule IS a new experiment
            # identity (it changes the reflectivity branch's window).
            ("scatter_dilations", (1, 1)),
            ("phase_gated_families", ()),
            ("scatter_ungated_families", ("layered", "marmousi")),
            # UL31 reflection branch + free-surface mode (20260929): legacy
            # defaults stay stripped; any non-default is a new identity.
            ("reflection_branch", False),
            ("reflection_channels", 32),
            ("free_surface_mode", "legacy_tanh20"),
        ):
            if payload["model"].get(name) == default:
                payload["model"].pop(name, None)
        # Boundary ramp (20260929): the blend length only means anything under
        # ramp_exact_row, so any value is stripped in the other modes (keeping
        # every pre-existing digest); under ramp it stays and binds identity.
        if payload["model"].get("free_surface_mode") != "ramp_exact_row":
            payload["model"].pop("free_surface_blend_updates", None)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf8")
        return hashlib.sha256(encoded).hexdigest()
