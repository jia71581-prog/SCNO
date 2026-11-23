"""Train the spectral SCNO on the published Style A/Marmousi pressure corpus."""

import argparse
import copy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as checkpoint_query

from native400_runtime.feature_enhanced_parent_20261022 import _local_protocol_import
from native400_runtime.paper201_time_axis_20261020 import native_time_axis
from native400_runtime.r24_cf_preserved_operator import COMPAT, _module, _sha
from native400_runtime.style_marm_joint_data_20261025 import JointTrainingData
from centered_ul_data import CenteredUniformLayeredData
from direct_joint_reads import DirectJointReads
from spectral_marmousi import SpectralMarmousiOperator
from train_spectral_curriculum import run_training, supervised_data_loss


class JointPressureCache:
    """Reuse the published TRAIN VDS, retaining its exact physical time samples."""

    def __init__(self, root, duration_s, data_loader=JointTrainingData):
        self.data = data_loader(root)
        self.ids = [str(record["record_id"]) for record in self.data.records]
        self.duration_s = duration_s
        times = self.data.times.numpy()
        self.times = times[times <= duration_s]
        if not len(self.times) or self.times[0] != 0 or self.times[-1] != duration_s:
            raise ValueError("training duration must end on an archived pressure frame")
        self.native_slots = list(range(len(self.times)))
        self.stats = self.data.normalization
        self.reference_energies = {}
        self.direct_reads = None
        self.pressure_scale_pa = (self.stats["pressure_scale_pa"] if "pressure_scale_pa" in self.stats
                                  else self.stats["pressure_rms"])
        self.binding = {**self.data.binding, "pressure_time_count": len(self.times),
                        "last_pressure_time_s": duration_s, "pressure_dt_s": duration_s / (len(self.times) - 1),
                        "pressure_scale_pa": self.pressure_scale_pa,
                        "data_loader_sha256": _sha(sys.modules[data_loader.__module__].__file__)}

    def batch(self, record, slots):
        selected = [int(slot) for slot in slots]
        if not selected or selected != sorted(set(selected)) or not 0 <= selected[0] <= selected[-1] < len(self.times):
            raise ValueError("expected sorted, unique pressure frame slots")
        need_energy = record not in self.reference_energies
        frames = self.native_slots if need_energy else selected
        if self.direct_reads is None:
            raw = self.data.batch([record], frames)
        else:
            raw = self.direct_reads.batch(record, frames)
        fields = raw["labels"].values[0]
        source = raw["source"]
        scale = self.pressure_scale_pa * float(source[0, 4])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("the normalized pressure unit must be finite and positive")
        if need_energy:
            if fields[0].count_nonzero().item():
                raise ValueError(f"record {self.ids[record]} has a nonzero pressure initial frame")
            energy = sum((part.double() / scale).square().sum().item() for part in fields.split(16))
            if not np.isfinite(energy) or energy <= 0:
                raise ValueError(f"record {self.ids[record]} lacks finite positive pressure energy")
            self.reference_energies[record] = energy
            fields = fields[selected]
        return {"velocity": raw["c"], "source": source,
                "times": torch.from_numpy(self.times[selected]),
                "target": (fields.double() / scale).float(),
                "full_reference_energy": self.reference_energies[record]}


class JointSpectralSCNO(SpectralMarmousiOperator):
    """The same direct pressure readout, initialized for the joint TRAIN domain."""

    def __init__(self, config, stats, pressure_times, device="cpu"):
        # Build the inherited module shapes from their configuration. The R24
        # constructor would load a learned checkpoint tied to another corpus.
        nn.Module.__init__(self)
        self.family = config.get("family", "marmousi")
        self.duration_s = float(pressure_times[-1])
        self.dense_time_block = config.get("dense_time_block", 1)
        self.refiner_time_block = config.get("refiner_time_block", 1)
        self.checkpoint_queries = config.get("checkpoint_queries", True)
        spec = json.loads((COMPAT / "shared/PARENT_MANIFEST.json").read_text())["parents"][self.family]
        code = COMPAT / ("base_" + self.family)
        constructor = copy.deepcopy(spec["resolved_config"])
        constructor.update(seed=config["seed"], model_spec=str(code / "configs/model_spec_v185b.json"))
        constructor["data"].update(source_h5="JOINT_TRAIN", normalization_json="JOINT_TRAIN")
        if _sha(constructor["model_spec"]) != spec["model_spec_sha256"]:
            raise ValueError("the audited backbone architecture specification changed")
        for path in (code, code / "snapshot", COMPAT / "shared"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        with _local_protocol_import():
            factory = _module("joint_scno_factory", code / "model.py")
            self.base = factory.build(constructor, "cpu")
            self.refiner = _module("joint_scno_refiner", COMPAT / "r22/refiner.py").ResidualRefiner()
            boundary = _module("joint_scno_boundary", COMPAT / "r21/boundary_exact.py")
            boundary.install_exact_boundary(self.base)
            self._restriction = _module("joint_scno_restrict", COMPAT / "sourcegen/restriction.py").restrict_nodal_2x
            self._bicubic201 = _module("joint_scno_lift", COMPAT / "r24/graph24.py").bicubic201
            self._core = _module("joint_scno_core", COMPAT / "shared/shorttrain_core_r2.py")
            self.helpers = {name: _module("joint_scno_" + name, COMPAT / "shared" / filename)
                            for name, filename in {"padding": "deterministic_padding.py",
                                "sampling": "deterministic_sampling.py", "pooling": "deterministic_pooling.py",
                                "cartesian": "cartesian_adjoint.py"}.items()}
        from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES
        from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer, ScaleMetadata

        pressure_scale = stats["pressure_scale_pa"] if "pressure_scale_pa" in stats else stats["pressure_rms"]
        scales = tuple(stats.get("source_scales", (2000., 2000., 50., self.duration_s, 1.)))
        self.normalizer = PhysicalNormalizer(ScaleMetadata(
            velocity_center_mps=stats["velocity_mean_mps"], velocity_scale_mps=stats["velocity_std_mps"],
            pressure_scale_pa=pressure_scale, source_scales=scales,
            train_manifest_sha256=stats["train_manifest_sha256"], allowed_medium_types=ALLOWED_MEDIUM_TYPES,
            record_count=stats["record_count"],
            algorithm=stats.get("algorithm", "joint_TRAIN_only_velocity_mean_std_pressure_rms")))
        self.base.scno_depth = 2
        self.base.domain_t_s = self.duration_s
        self.base.dense_decoder.domain_t_s = self.duration_s
        self.base.source_encoder.ic_encoder[0].weight.requires_grad_(False)
        self.refiner.static[0].conv1.weight.register_hook(self._mask_zero_ic_gradient)
        self.register_buffer("x_m", torch.arange(201, dtype=torch.float32) * 10, persistent=False)
        self.register_buffer("z_m", torch.arange(201, dtype=torch.float32) * 10, persistent=False)
        self.register_buffer("native_times_s", native_time_axis(), persistent=False)
        self.register_buffer("pressure_times_s", torch.as_tensor(pressure_times), persistent=False)
        self.initialization_receipt = {"checkpoint_SHA": None, "fresh": True, "model_spec_sha256": spec["model_spec_sha256"],
                                       "domain_t_s": self.duration_s, "seed": config["seed"]}
        self._install_spectral_readout(config)
        self.to(device)

    def _inputs(self, c, f):
        c = torch.as_tensor(c, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(f, dtype=torch.float32, device=self.device)
        if c.shape == (201, 201):
            c = c[None, None]
        if f.shape == (5,):
            f = f[None]
        if c.shape != (1, 1, 201, 201) or f.shape != (1, 5):
            raise ValueError("expected one 201x201 velocity grid and five source parameters")
        if not (torch.isfinite(c).all() and (c > 0).all() and torch.isfinite(f).all()):
            raise ValueError("source and positive velocity must be finite")
        if (not ((0 <= f[:, :2]) & (f[:, :2] <= 2000)).all()
                or f[0, 2] <= 0 or f[0, 4] <= 0 or not 0 <= f[0, 3] <= self.duration_s):
            raise ValueError("expected in-domain source coordinates and time, positive frequency and amplitude")
        return c, f

    @property
    def full_times_s(self):
        return self.pressure_times_s

    def forecast_at_times(self, c, f, times_s, *, normalized=True):
        c, f = self._inputs(c, f)
        times = torch.as_tensor(times_s, dtype=torch.float32, device=self.device).reshape(-1)
        if not times.numel() or not (torch.isfinite(times).all() and ((0 <= times) & (times <= self.duration_s)).all()):
            raise ValueError("expected finite physical queries within the configured training duration")
        with self.scopes():
            prepared, grid, features = self._prepare_cf(c, f)

        def query(block):
            with self.scopes():
                return self._query(prepared, grid, features, f, block)

        recompute = self.training and torch.is_grad_enabled() and self.checkpoint_queries
        pieces = [checkpoint_query(query, block, use_reentrant=False) if recompute else query(block)
                  for block in times.split(self.refiner_time_block)]
        pressure = torch.cat(pieces)
        return pressure if normalized else self.normalizer.decode_pressure(pressure[None], f[:, 4])[0]

    @torch.no_grad()
    def forward_cf_full(self, c, f):
        was_training = self.training
        try:
            self.eval()
            fields = self(c, f, normalized=False)
            return fields.permute(1, 2, 0).contiguous()
        finally:
            self.train(was_training)


class RunStatus:
    def __init__(self, path, details):
        self.path = path
        self.details = details
        self.start = monotonic()
        self.written = 0.
        self.update = self.saved = self.stage = 0

    def write(self, status, **extra):
        now = monotonic()
        payload = {**self.details, **extra, "state": status, "pid": os.getpid(),
                   "updated_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": now - self.start,
                   "completed_updates": self.update, "saved_updates": self.saved, "stage": self.stage}
        temporary = self.path.with_suffix(".pending")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, self.path)
        self.written = now

    def on_update(self, update, stage, saved):
        self.update, self.stage = update, stage
        if saved:
            self.saved = update
        if saved or monotonic() - self.written > 120:
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            self.write("RUNNING", cuda_peak_allocated_bytes=peak)


def check_families(cache, model, output, expected_families):
    slots = [0, len(cache.times) // 2, len(cache.times) - 1]
    families = {}
    model.eval()
    with torch.no_grad():
        for index, record in enumerate(cache.data.records):
            family = record["family"]
            if family in families:
                continue
            batch = cache.batch(index, slots)
            predicted = model(batch["velocity"], batch["source"], batch["times"])
            if predicted.shape != batch["target"].shape or not torch.isfinite(predicted).all():
                raise ValueError(f"{family} forward failed the real-data shape/finite check")
            if predicted[:, 0].count_nonzero().item() or not predicted[:, 1:].count_nonzero().item():
                raise ValueError(f"{family} forward failed the pressure readout or exact free surface check")
            families[family] = {"id": cache.ids[index], "queries_s": batch["times"].tolist(),
                                "source": batch["source"][0].tolist(), "positive_reference_energy": True,
                                "finite_prediction": True, "exact_free_surface": True}
    if set(families) != expected_families:
        raise ValueError("the published training families differ")
    output.write_text(json.dumps(families, indent=2) + "\n")


def largest_query_batch(cache, model, config, output):
    """Check the largest fitting block without changing the optimization state."""
    count = config["queries_per_record"]
    slots = np.linspace(0, len(cache.times) - 1, count, dtype=np.int64)
    batch = cache.batch(0, slots)
    buffers = [(buf, buf.detach().clone()) for buf in model.buffers()]
    was_training = model.training
    model.dense_time_block = model.refiner_time_block = 1
    model.eval()
    with torch.no_grad():
        baseline = model(batch["velocity"], batch["source"], batch["times"]).cpu()
    field_norm = max(baseline.double().norm().item(), 1e-12)
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state(model.device)

    def backward():
        model.train()
        torch.cuda.synchronize(model.device)
        started = monotonic()
        with model.scopes():
            loss = supervised_data_loss(model, batch, model.device, len(cache.times), config["energy_floor"])
            if not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite loss in query batch check")
            loss.backward()
        torch.cuda.synchronize(model.device)
        elapsed = monotonic() - started
        gradient = torch.cat([parameter.grad.detach().flatten().cpu()
                              for parameter in model.refiner.spectral_detail.parameters()
                              if parameter.grad is not None])
        if not gradient.count_nonzero() or not torch.isfinite(gradient).all():
            raise FloatingPointError("spectral detail gradients are missing or nonfinite")
        return loss.item(), gradient.double(), elapsed

    def restore_probe_state():
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda, model.device)
        with torch.no_grad():
            for buffer, prior in buffers:
                buffer.copy_(prior)

    baseline_loss, baseline_gradient, baseline_step_s = backward()
    gradient_norm = max(baseline_gradient.norm().item(), 1e-12)
    restore_probe_state()

    def check(block):
        model.dense_time_block = model.refiner_time_block = block
        torch.cuda.reset_peak_memory_stats(model.device)
        try:
            with torch.no_grad():
                model.eval()
                result = model(batch["velocity"], batch["source"], batch["times"]).cpu()
            field_delta = (result.double() - baseline).norm().item() / field_norm
            loss, gradient, elapsed = backward()
            gradient_delta = (gradient - baseline_gradient).norm().item() / gradient_norm
            loss_delta = abs(loss - baseline_loss) / max(abs(baseline_loss), 1e-12)
            if field_delta > 1e-4 or gradient_delta > 2e-3 or loss_delta > 1e-4:
                raise AssertionError(f"relative field/gradient/loss differences {field_delta}, {gradient_delta}, {loss_delta}")
            return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device),
                    "field_relative_l2": field_delta, "spectral_gradient_relative_l2": gradient_delta,
                    "loss_relative_difference": loss_delta, "forward_backward_s": elapsed}
        except torch.cuda.OutOfMemoryError:
            return None
        except (AssertionError, RuntimeError) as error:
            if block == 1:
                raise
            print(f"Time-query block {block} was incompatible: {error}", file=sys.stderr, flush=True)
            return None
        finally:
            restore_probe_state()

    attempts = {}
    upper, lower, peak = count, 0, None
    tested = count
    with torch.random.fork_rng(devices=[model.device]):
        while upper > lower:
            gc.collect()
            torch.cuda.empty_cache()
            allocated = check(tested)
            attempts[tested] = allocated
            if allocated is None:
                upper = tested - 1
            else:
                lower, peak = tested, allocated
            tested = (upper + lower + 1) // 2
    if not lower:
        raise RuntimeError("no execution block passed GPU forward/backward")
    model.dense_time_block = model.refiner_time_block = lower
    output.write_text(json.dumps({"query_block": lower, "queries_per_update": count,
                                  **peak, "query_blocks_tested": attempts,
                                  "single_query_forward_backward_s": baseline_step_s,
                                  "single_query_agreement": True, "backward_passed": True}, indent=2) + "\n")
    return lower


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("data", "config", "run", "status"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend-budget", action="store_true")
    parser.add_argument("--revise-sampling", action="store_true")
    parser.add_argument("--maximize-query-batch", action="store_true")
    parser.add_argument("--direct-check-read", action="store_true")
    args = parser.parse_args()
    if (args.extend_budget or args.revise_sampling) and not args.resume:
        parser.error("changing a saved budget or sampling protocol requires --resume")
    config = json.loads(args.config.read_text())
    torch.set_num_threads(2)
    status = RunStatus(args.status, {"dataset": str(args.data), "budget_updates": config["total_updates"],
                                    "queries_per_update": config["queries_per_record"],
                                    "pressure_duration_s": config["pressure_duration_s"], "device": args.device})
    status.write("VALIDATING_INPUTS")
    try:
        if args.device.startswith("cuda"):
            free_bytes, _ = torch.cuda.mem_get_info(args.device)
            if free_bytes < 4 * 1024**3:
                raise RuntimeError("shared GPU has fewer than 4 GiB free")
            torch.cuda.set_per_process_memory_fraction(config["cuda_memory_fraction"], args.device)
        ul = config.get("data_kind") == "centered_uniform_layered"
        data_loader = CenteredUniformLayeredData if ul else JointTrainingData
        families = {"uniform", "layered"} if ul else {"style_a", "marmousi_crop"}
        cache = JointPressureCache(args.data, config["pressure_duration_s"], data_loader)
        status.details["pressure_frames"] = len(cache.times)
        status.details["train_records"] = len(cache.ids)
        if args.direct_check_read and not ul:
            direct = DirectJointReads(cache.data)
            receipt = direct.check_published_view()
            args.status.with_name("DIRECT_READ_CHECK.json").write_text(json.dumps(receipt, indent=2) + "\n")
            cache.direct_reads = direct

        def build(model_config, device):
            model = JointSpectralSCNO({**model_config, "seed": config["seed"]}, cache.stats, cache.times, device=device)
            if not args.resume:
                check_families(cache, model, args.status.with_name("PREFLIGHT.json"), families)
            if args.maximize_query_batch and model.device.type == "cuda":
                status.write("CHECKING_GPU_BATCH")
                status.details["parallel_time_queries"] = largest_query_batch(
                    cache, model, config, args.status.with_name("QUERY_BATCH.json"))
            return model

        status.write("INITIALIZING_MODEL")
        run_training(args, cache=cache, model_factory=build, on_update=status.on_update)
        status.update = status.saved = config["total_updates"]
        status.write("COMPLETED")
    except BaseException as error:
        status.write("FAILED", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
