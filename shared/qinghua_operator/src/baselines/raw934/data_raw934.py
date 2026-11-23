"""Shared reader for the frozen joint raw934 TRAIN cache.

Binds one Python reader to `data/joint_uniform500_layered434_raw_20261011/train400_cache/`
(934 TRAIN records = 434 preserved layered + 500 regenerated uniform, native
201x201, IC8 + future392, raw nodal NO_FILTER). The reader is the only place
that knows the cache layout, so the FNO and KNO entrypoints cannot drift apart
on normalization, frame order or roster identity.

Field contract, verified against CACHE_METADATA.json / ARRAY_BLUEPRINT.json on
2026-10-11:

  known_ic.npy          [934,1,8,201,201]  float32  physical Pa (top row exact 0)
  known_velocity.npy    [934,1,1,201,201]  float32  sharp201 c401[::2,::2] NO_B5, m/s
  known_source_map.npy  [934,1,1,201,201]  float32  unit-mass, sum(fp64)==1
  known_source.npy      [934,1,5]          float32  (x_m, z_m, f0_hz, t0_s, amplitude)
  known_x.npy           [934,201]          float32  0..2000 m
  known_z.npy           [934,201]          float32  0..2000 m
  known_time_axis.npy   [934,801]          float32  0..1 s, dt 1.25 ms
  target.npy            [934,1,392,201,201] float32 normalized (see below)
  energy.npy            [934,392]          float64  per-frame squared norm of target
  total.npy, mean.npy   [934]              float64  whole-window sums

Normalization. `known_ic` stays in physical Pa by construction; the target is
encoded once through the frozen UL `PhysicalNormalizer`, i.e.
`target = physical_Pa / (pressure_scale_pa * source_amplitude)`. With
`source_amplitude == 1.0` for all 934 records this reduces to a single scalar,
and because the metric is a relative L2 the scale cancels either way. The
constant is re-derived here from the shipped `normalization_ul.json` rather
than hard-coded, and cross-checked against the cache's own SHA.

Frame protocol. `known_frames = (88,95,102,110,117,124,131,138)` and
`future_frames = 392 stored instants` (146..800), which is the native
`IC8+392` protocol of the main method, not the nyquist15 42-frame KNO window.
Both baselines therefore predict the same 392 frames the main model does.

Split discipline. Only `TRAIN` rows are reachable: the roster at construction
is asserted to be exactly the 934-record TRAIN roster with the registered
family census, and no method here accepts a row outside it. TEST/VAL rosters
are never opened. A small in-sample dev panel drawn from TRAIN is exposed for
monitoring only.
"""
from __future__ import annotations
from scno_release_paths import local_asset as _scno_local_asset

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(_scno_local_asset('project'))
CACHE = ROOT / "data" / "joint_uniform500_layered434_raw_20261011" / "train400_cache"
DATASET = CACHE.parent
NORMALIZER_PATH = DATASET / "normalization_ul.json"
TRAIN_ROSTER = DATASET / "TRAIN_ROSTER.json"

SCHEMA = "JOINT_UNIFORM500_LAYERED434_RAW_TRAIN_CACHE_V1"
EXPECTED_RECORDS = 934
EXPECTED_FAMILIES = {"layered": 434, "uniform": 500}
KNOWN_FRAMES = (88, 95, 102, 110, 117, 124, 131, 138)
N_KNOWN = 8
N_FUTURE = 392
FUTURE_FIRST = 146
FUTURE_LAST = 800
GRID = 201
DT_S = 0.00125
SOURCE_F0_HZ = 15.0
SOURCE_T0_S = 0.1
SOURCE_AMPLITUDE = 1.0

# KNO rollout schedule.
#
# The KNO's contract is a fixed number of latent supersteps, one decoded
# pressure frame each. The main model predicts all 392 stored future instants,
# so the KNO is configured to land on all of them: `KNO_STEPS = N_FUTURE` and
# `KNO_FRAME_POSITIONS = (0, 1, ..., 391)`. That makes the two models emit the
# same predicted wavefield (201x201x392 future frames) and be scored on the same
# frames, so no common-frame block or windowed caveat is needed in the table.
#
# The cost of that choice is real and is recorded here: the rollout is 392
# recurrent steps instead of the UL run's 42, so the KNO's forward/backward is
# ~9.3x deeper and its memory is dominated by the checkpointed latents. Batch
# size is set from the measured peak, not assumed; gradient checkpointing is
# mandatory. The UL run's 42-step schedule remains available by passing
# `frame_positions` explicitly to `kno_item`.
KNO_STEPS = N_FUTURE
KNO_FRAME_POSITIONS = tuple(range(N_FUTURE))

# Velocity normalization. The UL baselines divided by a fixed 6000 m/s; the
# native400 runtime instead standardizes with the frozen normalizer's
# velocity_center_mps/velocity_scale_mps. This cache is a native201 view, so the
# frozen normalizer is the binding convention; `vel/6000` is kept as an
# alternative only so a UL-matched ablation stays possible.
VELOCITY_NORM = "frozen_normalizer_standardize"
VELOCITY_FIXED_DIVISOR = 6000.0

# In-sample TRAIN-only monitoring panel: the first and last TRAIN row of each
# family, so both media types are exercised. Resolved from the roster itself
# rather than hard-coded names, because the layered ids mix the `_n15_` and
# `_rawsrc_` spellings. Never used for selection.
DEV_PANEL_ROWS = (0, 433, 434, 933)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Normalizer:
    """Frozen UL physical<->model unit transform, bound by file SHA."""

    def __init__(self, values: dict, source_sha256: str):
        self.pressure_scale_pa = float(values["pressure_scale_pa"])
        self.velocity_center_mps = float(values["velocity_center_mps"])
        self.velocity_scale_mps = float(values["velocity_scale_mps"])
        self.source_scales = tuple(float(v) for v in values["source_scales"])
        self.algorithm = str(values["algorithm"])
        self.record_count = int(values["record_count"])
        self.source_sha256 = source_sha256
        if self.pressure_scale_pa <= 0 or self.velocity_scale_mps <= 0:
            raise ValueError("frozen normalizer scales must be positive")
        if len(self.source_scales) != 5 or min(self.source_scales) <= 0:
            raise ValueError("frozen normalizer needs five positive source scales")

    def encode_pressure(self, physical_pa, amplitude: float):
        scale = self.pressure_scale_pa * float(amplitude)
        return physical_pa / scale

    def encode_velocity(self, velocity_mps, gate: str | None = None):
        if (gate or VELOCITY_NORM) == "frozen_normalizer_standardize":
            return (velocity_mps - self.velocity_center_mps) / self.velocity_scale_mps
        return velocity_mps / VELOCITY_FIXED_DIVISOR


class Raw934TrainCache:
    """Memory-mapped TRAIN-only view of the frozen raw934 cache."""

    def __init__(self, cache: Path = CACHE, dataset: Path = DATASET):
        self.cache = Path(cache)
        self.dataset = Path(dataset)
        metadata = json.loads((self.cache / "CACHE_METADATA.json").read_text())
        if metadata.get("status") != "COMPLETE_VERIFIED_TRAIN934":
            raise PermissionError("raw934 cache is not the verified TRAIN934 payload")
        if metadata.get("pressure_encoded_once") is not True:
            raise PermissionError("raw934 target was not encoded exactly once")
        if metadata.get("raw_known_physical_Pa") is not True:
            raise PermissionError("raw934 known frames must stay physical Pa")
        self.metadata = metadata
        self.ids = list(metadata["ids"])
        self.families = list(metadata["families"])
        self.row_by_id = {sid: row for row, sid in enumerate(self.ids)}
        self.view_id = str(metadata["view_id"])

        roster = json.loads((self.dataset / "TRAIN_ROSTER.json").read_text())
        if [r["sample_id"] for r in roster["records"]] != self.ids:
            raise PermissionError("cache ids disagree with the TRAIN roster order")
        census = {family: self.families.count(family) for family in set(self.families)}
        if census != EXPECTED_FAMILIES:
            raise PermissionError(f"TRAIN family census drift: {census}")
        if len(self.ids) != EXPECTED_RECORDS:
            raise PermissionError("raw934 must expose 934 TRAIN rows")

        normalizer_values = json.loads(NORMALIZER_PATH.read_text())
        normalizer_sha = _sha256(NORMALIZER_PATH)
        if normalizer_sha != metadata["normalizer_SHA"]:
            raise PermissionError("normalizer bytes changed since the cache was built")
        self.normalizer = Normalizer(normalizer_values, normalizer_sha)

        # The stored future window is a 392-point sample of frames 146..800 at
        # 1-2 frame spacing, NOT a contiguous run, so `future_frames` has to be
        # read and checked rather than reconstructed as `first + position`.
        generation = json.loads((self.dataset / "CONFIG.json").read_text())
        known_frames = tuple(int(v) for v in generation["known_frames"])
        if known_frames != KNOWN_FRAMES:
            raise PermissionError(f"known frame protocol drift: {known_frames}")
        self.known_frames = known_frames
        self.future_frames = tuple(int(v) for v in generation["future_frames"])
        if len(self.future_frames) != N_FUTURE:
            raise PermissionError("future window is not the registered 392 frames")
        if any(b <= a for a, b in zip(self.future_frames, self.future_frames[1:])):
            raise PermissionError("future frames must strictly increase")
        if self.future_frames[0] != FUTURE_FIRST or self.future_frames[-1] != FUTURE_LAST:
            raise PermissionError("future window endpoints changed")
        if any(f <= self.known_frames[-1] for f in self.future_frames):
            raise PermissionError("future frames must follow the known frames")

        # CACHE_METADATA carries only SHA and byte counts per array; the exact
        # shape/dtype contract lives in the blueprint written alongside it.
        blueprint = json.loads((self.cache / ".." / "ARRAY_BLUEPRINT.json").resolve().read_text())
        self._blueprint = blueprint["arrays"]

        self._maps: dict[str, np.ndarray] = {}
        self._array_sha_cache: dict[str, str] = {}

    # ---- low-level array access -------------------------------------------------
    def _map(self, name: str) -> np.ndarray:
        if name not in self._maps:
            entry = self._blueprint[name]
            path = (self.cache / name).resolve()
            if not path.is_relative_to(self.cache.resolve()):
                raise PermissionError("cache array path escaped the bound root")
            # `payload_bytes` is the net array payload; the .npy container adds
            # its own header on top, so compare against `array_size` of the
            # parsed header instead of the raw file length.
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(value.shape) != list(entry["shape"]) or value.dtype.str != entry["dtype"]:
                raise PermissionError(f"{name} header changed since the cache was built")
            if value.size * value.dtype.itemsize != entry["payload_bytes"]:
                raise PermissionError(f"{name} payload length disagrees with the blueprint")
            recorded = self.metadata["arrays"].get(name)
            if recorded is not None and recorded["bytes"] != path.stat().st_size:
                raise PermissionError(f"{name} file length changed since the cache was built")
            self._maps[name] = value
        return self._maps[name]

    def verify_array(self, name: str) -> str:
        """Full-file SHA check; expensive, so only a launch-time or probe check."""
        if name not in self._array_sha_cache:
            self._array_sha_cache[name] = _sha256(self.cache / name)
        expected = self.metadata["arrays"][name]["SHA"]
        if self._array_sha_cache[name] != expected:
            raise PermissionError(f"{name} byte payload differs from the frozen cache")
        return self._array_sha_cache[name]
    def row(self, sample_id: str) -> int:
        if sample_id not in self.row_by_id:
            raise PermissionError("row is outside the frozen TRAIN934 roster")
        return self.row_by_id[sample_id]

    # ---- registered quantities --------------------------------------------------
    def known_inputs(self, sample_id: str, *, velocity_norm: str | None = None) -> dict:
        """Return the registered known quantities for one TRAIN row.

        `ic` stays physical Pa; `velocity` is normalized; `source_map` is the
        unit-mass map as stored; coordinates are in metres on the 201 grid.
        """
        row = self.row(sample_id)
        gate = velocity_norm or VELOCITY_NORM
        velocity = torch.from_numpy(
            np.array(self._map("known_velocity.npy")[row, 0, 0], copy=True))
        velocity = self.normalizer.encode_velocity(velocity)
        return {
            "sample_id": sample_id,
            "row": row,
            "family": self.families[row],
            "ic": torch.from_numpy(np.array(self._map("known_ic.npy")[row, 0], copy=True)),
            "velocity": velocity[None].contiguous(),
            "source_map": torch.from_numpy(
                np.array(self._map("known_source_map.npy")[row, 0], copy=True)),
            "source": torch.from_numpy(np.array(self._map("known_source.npy")[row, 0], copy=True)),
            "anchor": float(self._map("known_anchor.npy")[row, 0]),
            "x_m": torch.from_numpy(np.array(self._map("known_x.npy")[row], copy=True)),
            "z_m": torch.from_numpy(np.array(self._map("known_z.npy")[row], copy=True)),
            "time_axis": torch.from_numpy(np.array(self._map("known_time_axis.npy")[row], copy=True)),
        }

    def target(self, sample_id: str) -> torch.Tensor:
        """Normalized future392 target, copied out of the memmap."""
        row = self.row(sample_id)
        return torch.from_numpy(np.array(self._map("target.npy")[row, 0], copy=True))

    def frame_energy(self, sample_id: str) -> np.ndarray:
        row = self.row(sample_id)
        return np.array(self._map("energy.npy")[row], copy=True)

    def window_energy(self, sample_id: str) -> float:
        row = self.row(sample_id)
        return float(self._map("total.npy")[row])

    def all_frame_energy(self) -> np.ndarray:
        """[934,392] fp64 per-frame energy, used for the FNO inverse-energy weights."""
        return np.asarray(self._map("energy.npy"))

    # ---- composite training items ----------------------------------------------
    def fno_item(self, sample_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """FNO input stack [14,201,201] and normalized target [392,201,201].

        Channel order follows the native runtime's `build_known` convention,
        extended to the raw934 contract:

          0            velocity (normalized)
          1..8         known IC frames (normalized pressure)
          9            unit-mass source map
          10..13       (2x/2000-1, 2z/2000-1, (x-x_s)/2000, (z-z_s)/2000)

        The coordinate channels carry the source offset, because the raw934
        source x spans 204.7..1799.2 m while the UL cache had a single source.
        """
        known = self.known_inputs(sample_id)
        amplitude = float(known["source"][4])
        ic = self.normalizer.encode_pressure(known["ic"], amplitude)
        z, x = torch.meshgrid(known["z_m"], known["x_m"], indexing="ij")
        source = known["source"]
        coords = torch.stack(
            (
                2 * x / 2000.0 - 1.0,
                2 * z / 2000.0 - 1.0,
                (x - float(source[0])) / 2000.0,
                (z - float(source[1])) / 2000.0,
            ),
            dim=0,
        )
        stack = torch.cat((known["velocity"], ic, known["source_map"], coords), dim=0)
        if stack.shape != (14, GRID, GRID):
            raise RuntimeError(f"unexpected FNO input shape {tuple(stack.shape)}")
        if not torch.isfinite(stack).all():
            raise FloatingPointError("nonfinite FNO input stack")
        target = self.target(sample_id)
        if target.shape != (N_FUTURE, GRID, GRID):
            raise RuntimeError(f"unexpected target shape {tuple(target.shape)}")
        return stack, target

    def kno_item(self, sample_id: str, *, frame_positions=None):
        """KNO input stack [17,201,201], controls [n,3], target [n,201,201].

        Channels mirror the audited UL KNO entrypoint
        (`data_ul.INPUT_CHANNELS`): IC8, velocity, source map, the five
        normalized source scalars broadcast to the grid, and the two coordinate
        grids. Controls are (dt_s, absolute_time_s, mean_wavelet/amplitude),
        where the wavelet is the frozen project Ricker
        `(1-2*a*tau^2)*exp(-a*tau^2)` with `a=(pi*f0)^2`, `tau=t-t0`.
        Verified against the shipped `source_wavelet` of the frozen dataset to
        2.2e-08 max abs (float32 storage precision).

        `frame_positions` selects which of the 392 stored future instants the
        rollout lands on, as 0-based positions into `future_frames` (146..800).
        The default is `KNO_FRAME_POSITIONS`, a 42-step schedule explained at
        that constant. Positions must be strictly increasing so every control
        interval advances in time.
        """
        positions = KNO_FRAME_POSITIONS if frame_positions is None else tuple(frame_positions)
        if not positions:
            raise ValueError("at least one rollout frame is required")
        if any(b <= a for a, b in zip(positions, positions[1:])):
            raise ValueError("rollout frame positions must strictly increase")
        if positions[0] < 0 or positions[-1] >= N_FUTURE:
            raise ValueError("rollout frame position outside the stored future window")

        known = self.known_inputs(sample_id)
        amplitude = float(known["source"][4])
        ic = self.normalizer.encode_pressure(known["ic"], amplitude)
        source = known["source"]
        x, z = known["x_m"], known["z_m"]
        grid_x, grid_z = torch.meshgrid(x, z, indexing="ij")
        dims = (GRID, GRID)
        source_scales = self.normalizer.source_scales
        f0, t0 = float(source[2]), float(source[3])
        channels = [
            *ic,
            known["velocity"][0],
            known["source_map"][0],
            torch.full(dims, f0 / source_scales[2]),
            torch.full(dims, t0 / source_scales[3]),
            torch.full(dims, amplitude / source_scales[4]),
            torch.full(dims, float(source[0]) / source_scales[0]),
            torch.full(dims, float(source[1]) / source_scales[1]),
            grid_x,
            grid_z,
        ]
        stack = torch.stack(channels, dim=0)
        if stack.shape != (17, GRID, GRID):
            raise RuntimeError(f"unexpected KNO input shape {tuple(stack.shape)}")
        if not torch.isfinite(stack).all():
            raise FloatingPointError("nonfinite KNO input stack")

        time_axis = known["time_axis"]
        wavelet = self._ricker(time_axis, f0, t0) / amplitude
        previous = self.known_frames[-1]
        controls = np.empty((len(positions), 3), dtype=np.float32)
        for step, position in enumerate(positions):
            frame = self.future_frames[position]
            if frame <= previous:
                raise ValueError("recurrent target must follow the previous frame")
            controls[step, 0] = np.float32(time_axis[frame] - time_axis[previous])
            controls[step, 1] = np.float32(time_axis[frame])
            controls[step, 2] = np.float32(wavelet[previous + 1:frame + 1].mean())
            previous = frame
        if not np.isfinite(controls).all() or np.any(controls[:, 0] <= 0):
            raise FloatingPointError("invalid recurrent controls")

        target_all = self.target(sample_id)
        target = target_all[torch.from_numpy(np.asarray(positions, dtype=np.int64))]
        return stack, torch.from_numpy(controls), target

    @staticmethod
    def _ricker(time_axis: torch.Tensor, f0_hz: float, t0_s: float) -> torch.Tensor:
        a = (np.pi * f0_hz) ** 2
        tau = time_axis.double() - t0_s
        return ((1 - 2 * a * tau**2) * torch.exp(-a * tau**2)).float()


def load_cache(**kwargs) -> Raw934TrainCache:
    return Raw934TrainCache(**kwargs)
