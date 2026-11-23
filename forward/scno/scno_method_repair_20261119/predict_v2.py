"""Pressure inference with a reusable per-instance SCNO V2 spatial preparation."""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import statistics
from time import perf_counter

import h5py
import numpy as np
import torch


JOINT_TIME_SCHEMA = "scno_joint_fno_time_channels_v1"
JOINT_TRAVEL_SCHEMA = "scno_joint_fno_travel_v1"
JOINT_SCHEMAS = (JOINT_TIME_SCHEMA, JOINT_TRAVEL_SCHEMA)


@contextmanager
def execution_mode(model, block=None):
    modes = [(module, module.training) for module in model.modules()]
    original = model.dense_time_block, model.refiner_time_block
    try:
        model.eval()
        if block is not None:
            if type(block) is not int or block < 1:
                raise ValueError("time-query block size must be a positive integer")
            model.dense_time_block = model.refiner_time_block = block
        with torch.no_grad():
            yield
    finally:
        model.dense_time_block, model.refiner_time_block = original
        for module, training in modes:
            module.training = training


def state_stamp(model):
    """Track versioned model tensors, including nonpersistent geometry buffers."""
    versions = []
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        try:
            version = tensor._version
        except RuntimeError as error:
            raise ValueError("the reusable inference cache requires versioned model tensors") from error
        versions.append((name, id(tensor), tensor.data_ptr(), version, tensor.device, tensor.dtype))
    return (id(model), id(model.normalizer), repr(model.normalizer.metadata),
            model.duration_s, model.phase, model.spectral_detail_enabled,
            model.decoder_grid_size, tuple(versions))


def as_times(model, times):
    values = (model.full_times_s if times is None
              else torch.as_tensor(times, device=model.device, dtype=torch.float32).reshape(-1))
    if not len(values) or not (torch.isfinite(values).all() and ((0 <= values) & (values <= model.duration_s)).all()):
        raise ValueError("query seconds must lie within the configured physical domain")
    return values


class PreparedWavefield:
    """Single-consumer cache for an immutable medium, source and versioned model."""

    def __init__(self, model, velocity, source, *, reuse_dense_gabor=True):
        self.model = model
        self.reuse_dense_gabor = reuse_dense_gabor
        self.stamp = state_stamp(model)
        with execution_mode(model):
            source = torch.as_tensor(source, device=model.device, dtype=torch.float32)
            if source.shape == (5,):
                source = source[None]
            self.zero_amplitude = source.shape == (1, 5) and source[0, 4].item() == 0.
            checked_source = source.clone()
            if self.zero_amplitude:
                checked_source[0, 4] = 1.
            velocity, _ = model._inputs(velocity, checked_source)
            self.velocity = velocity.clone()
            self.source = source.clone()
            if not self.zero_amplitude:
                with model.scopes():
                    self.prepared, self.grid, self.context = model._prepare_cf(self.velocity, self.source)
        self.check_current()

    def check_current(self):
        if self.stamp != state_stamp(self.model):
            self.stamp = None
            self.prepared = self.grid = self.context = None
            raise RuntimeError("the model changed after preparation; rebuild the inference cache")

    def predict(self, times=None, *, time_block=None, detail_enabled=None):
        """Query seconds in input order and return a device-resident [T,Z,X] Pa array."""
        self.check_current()
        model = self.model
        enabled = model.spectral_detail_enabled if detail_enabled is None else bool(detail_enabled)
        with execution_mode(model, time_block):
            values = as_times(model, times)
            if self.zero_amplitude:
                result = self.velocity.new_zeros((len(values), *self.velocity.shape[-2:]))
            else:
                result = self._query_pressure(values, enabled)
        self.check_current()
        return result

    def _query_pressure(self, times, enabled):
        from spectral_gabor import share_gabor_on_dense_grid

        model = self.model
        pressure = None
        start = 0
        for block in times.split(model.refiner_time_block):
            sharing = (share_gabor_on_dense_grid(self.grid.height * self.grid.width)
                       if self.reuse_dense_gabor else nullcontext())
            with model.scopes(), sharing:
                if enabled:
                    normalized = model._parts(self.prepared, self.grid, self.context, self.source, block, True)[0]
                else:
                    normalized = model._query_base(self.prepared, self.grid, self.context[0], self.source, block)
            physical = model.normalizer.decode_pressure(normalized[None], self.source[:, 4])[0]
            if pressure is None:
                pressure = physical.new_empty((len(times), *physical.shape[1:]))
            pressure[start:start + len(block)].copy_(physical)
            start += len(block)
        return pressure


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device, function):
    synchronize(device)
    start = perf_counter()
    output = function()
    synchronize(device)
    return output, perf_counter() - start


def measure_cached(cache, times, block, repeats):
    device = cache.model.device
    forwards, transfers, peaks = [], [], []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        pressure, elapsed = timed(device, lambda: cache.predict(times, time_block=block))
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        to_host, transfer = timed(device, pressure.cpu)
        del pressure
        forwards.append(elapsed)
        transfers.append(transfer)
        peaks.append(peak)
    return to_host, {"forward_s": statistics.median(forwards), "pressure_transfer_s": statistics.median(transfers),
                     "peak_allocated_bytes": max(peaks) if device.type == "cuda" else None,
                     "forward_samples_s": forwards}


def measure_native(model, velocity, source, times, repeats):
    """Time the original full call and measure its preparation inside that call."""
    preparation, totals, transfers, peaks = [], [], [], []
    original = model._prepare_cf
    had_override = "_prepare_cf" in vars(model)
    device = model.device
    setup = 0.

    def observe_prepare(*args, **kwargs):
        nonlocal setup
        result, duration = timed(device, lambda: original(*args, **kwargs))
        setup += duration
        return result

    model._prepare_cf = observe_prepare
    try:
        with execution_mode(model):
            for _ in range(repeats):
                setup = 0.
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                pressure, elapsed = timed(device, lambda: model(velocity, source, times, normalized=False))
                peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                host, transfer = timed(device, pressure.cpu)
                del pressure
                preparation.append(setup)
                totals.append(elapsed)
                transfers.append(transfer)
                peaks.append(peak)
    finally:
        if had_override:
            model._prepare_cf = original
        else:
            del model._prepare_cf
    return host, {"preparation_s": statistics.median(preparation),
                  "forward_s": statistics.median(total - prep for total, prep in zip(totals, preparation)),
                  "resident_total_s": statistics.median(totals), "pressure_transfer_s": statistics.median(transfers),
                  "peak_allocated_bytes": max(peaks) if device.type == "cuda" else None}


def pressure_parity(candidate, native, tolerance):
    if candidate.shape != native.shape or candidate.dtype != native.dtype or candidate.ndim != 3:
        raise ValueError("candidate and native pressure must share shape, axis order and dtype")
    delta, energy, maximum = 0., 0., 0.
    for a, b in zip(candidate.split(16), native.split(16)):
        difference = a.double() - b.double()
        delta += difference.square().sum().item()
        energy += b.double().square().sum().item()
        maximum = max(maximum, difference.abs().max().item())
    if not np.isfinite(delta) or not np.isfinite(energy):
        raise ValueError("nonfinite native or candidate pressure")
    relative = (delta / energy) ** .5 if energy > 0 else (0. if delta == 0 else None)
    top_exact = (candidate[:, 0].count_nonzero().item() == native[:, 0].count_nonzero().item() == 0)
    return {"pooled_relative_l2": relative, "maximum_abs_difference_pa": maximum, "exact_zero_top": top_exact,
            "accepted": top_exact and relative is not None and relative <= tolerance}


def benchmark(model, velocity, source, times, candidates, tune_frames, repeats, full_repeats, tolerance):
    device = model.device
    plan = np.linspace(0, len(times) - 1, min(tune_frames, len(times)), dtype=np.int64)
    probe = times[torch.from_numpy(plan).to(device)]
    warm = probe[:min(4, len(probe))]
    with execution_mode(model):
        model(velocity, source, warm, normalized=False)
    native, native_receipt = measure_native(model, velocity, source, times, full_repeats)
    prepared, setup = timed(device, lambda: PreparedWavefield(model, velocity, source))
    reference_probe = native[plan]
    probes = []
    accepted = []
    for block in candidates:
        row = {"time_block": block, "queries": len(probe)}
        try:
            prepared.predict(probe, time_block=block)
            pressure, timing = measure_cached(prepared, probe, block, repeats)
            parity = pressure_parity(pressure, reference_probe, tolerance)
            row.update(**timing, parity=parity)
            if parity["accepted"]:
                accepted.append(row)
        except torch.cuda.OutOfMemoryError:
            row["error"] = "cuda_out_of_memory"
            torch.cuda.empty_cache()
        probes.append(row)

    full_trials = []
    blocks_checked = [entry["time_block"] for entry in sorted(accepted, key=lambda entry: entry["forward_s"])]
    blocks_checked.append(None)  # The unchanged configured execution is a final, measured fallback.
    for block in blocks_checked:
        attempt = {"time_block": block if block is not None else "configured"}
        try:
            pressure, timing = measure_cached(prepared, times, block, full_repeats)
            parity = pressure_parity(pressure, native, tolerance)
            attempt.update(**timing, parity=parity)
        except torch.cuda.OutOfMemoryError:
            attempt["error"] = "cuda_out_of_memory"
            torch.cuda.empty_cache()
            full_trials.append(attempt)
            continue
        full_trials.append(attempt)
        if parity["accepted"]:
            return pressure, {"timed_queries": len(times), "tuning_slots": plan.tolist(),
                              "baseline_configured_blocks": [model.dense_time_block, model.refiner_time_block],
                              "baseline_native": native_receipt, "cached_preparation_s": setup,
                              "tuning": probes, "full_candidate_trials": full_trials,
                              "selected": attempt,
                              "resident_one_shot_speedup": native_receipt["resident_total_s"] / (setup + timing["forward_s"]),
                              "resident_forward_speedup_after_prepare": native_receipt["forward_s"] / timing["forward_s"],
                              "pooled_relative_l2_tolerance": tolerance}
    raise RuntimeError("no complete-axis prediction passed native pressure and exact top parity")


def load_model(config, checkpoint, dataset, device, compatible_state):
    from initialization_v2 import checkpoint_decoder_grid_size, semantics, validate_resume_semantics
    from lora_v2 import place_saved_adapters
    from model_v2 import build_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    schema = payload.get("schema")
    if schema in JOINT_SCHEMAS:
        if payload.get("config") != config or payload.get("dataset_binding") != dataset.binding:
            raise ValueError("joint FNO prediction requires the checkpoint's exact configuration and data binding")
        if schema == JOINT_TRAVEL_SCHEMA:
            from joint_fno_transport_model import TravelTimeFNOSCNO as joint_type
        else:
            from joint_fno_model import JointFNOSCNO as joint_type
        template = build_model(config, dataset, torch.device("cpu"))
        return joint_type.from_checkpoint(template, payload).to(device).eval()
    source_grid = checkpoint_decoder_grid_size(payload)
    grid_size = config.get("model", {}).get("decoder_grid_size", 101)
    if source_grid != grid_size:
        raise ValueError(f"checkpoint decoder uses {source_grid}×{source_grid}, configured decoder uses {grid_size}×{grid_size}")
    if compatible_state and grid_size != 101:
        raise ValueError("native 201 prediction requires an exact V2 checkpoint binding")
    if compatible_state:
        state = payload.get("model", payload)
    else:
        binding = payload.get("binding", {})
        if (payload.get("schema") != "scno_training_v2" or binding.get("config") != config
                or binding.get("dataset") != dataset.binding):
            raise ValueError("the V2 checkpoint requires its exact recorded configuration and data/normalization binding")
        state = payload["model"]
    model = build_model(config, dataset, device)
    if not compatible_state:
        validate_resume_semantics(payload, semantics(model, dataset))
    place_saved_adapters(model, state)
    current = model.state_dict()
    if current.keys() != state.keys():
        missing, unexpected = sorted(current.keys() - state.keys()), sorted(state.keys() - current.keys())
        raise ValueError(f"incompatible spectral state; missing {missing[:6]}, unexpected {unexpected[:6]}")
    for name, value in current.items():
        supplied = state[name]
        if not isinstance(supplied, torch.Tensor) or supplied.shape != value.shape or supplied.dtype != value.dtype:
            raise ValueError(f"strict state shape or dtype mismatch: {name}")
    model.load_state_dict(state, strict=True)
    model.set_phase(payload.get("phase", "C"))
    return model.eval()


def native_time_rows(axis, times):
    """Match seconds to registered FP32 timestamps in query order, allowing one ULP of input rounding."""
    if times.ndim != 1 or not times.numel() or not torch.isfinite(times).all():
        raise ValueError("expected a nonempty, finite vector of physical query times")
    if torch.equal(times, axis):
        return None
    positions = torch.searchsorted(axis, times.contiguous())
    left = (positions - 1).clamp(0, axis.numel() - 1)
    right = positions.clamp(0, axis.numel() - 1)
    rows = torch.where((times - axis[left]).abs() <= (times - axis[right]).abs(), left, right)
    nearest = axis[rows]
    tolerance = (torch.nextafter(nearest, torch.full_like(nearest, torch.inf)) - nearest).abs()
    unmatched = ((times < axis[0]) | (times > axis[-1]) | ((times - nearest).abs() > tolerance))
    if torch.any(unmatched):
        slot = int(torch.nonzero(unmatched, as_tuple=True)[0][0])
        raise ValueError(f"off-grid time {float(times[slot]):.10g} s for the joint FNO time-channel "
                         f"checkpoint; nearest registered time is {float(nearest[slot]):.10g} s")
    if rows.numel() == axis.numel() and torch.equal(rows, torch.arange(axis.numel(), device=axis.device)):
        return None
    return rows


@torch.inference_mode()
def measure_joint(model, velocity, source, times, repeats, *, backend="auto", parity_rtol=2e-5):
    """Return a physical TZX array and separate capture, validation, resident forward and transfer costs."""
    if source.shape not in ((5,), (1, 5)):
        raise ValueError("the TZX prediction file takes exactly one five-parameter source")
    device = model.device
    rows, lookup = timed(device, lambda: native_time_rows(model.full_times_s, times))
    if backend not in ("auto", "eager", "graph") or repeats < 1:
        raise ValueError("expected one or more repeats and the auto, eager, or graph joint backend")
    full_cuda = device.type == "cuda" and rows is None
    if backend == "graph" and not full_cuda:
        raise ValueError("the joint CUDA graph requires CUDA and the complete ordered native time axis")
    fallback, attempted_s = None, 0.
    if full_cuda and backend != "eager":
        from joint_fno_inference import JointFNOFullGraph, verify_full_graph

        graph = None
        started = perf_counter()
        try:
            graph, capture = timed(device, lambda: JointFNOFullGraph(model, velocity, source, precision="fp32"))
            agreement, validation = timed(device, lambda: verify_full_graph(graph, velocity, source, tolerance=parity_rtol))
            torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for _ in range(repeats):
                output, elapsed = timed(device, lambda: graph(velocity, source))
                samples.append(elapsed)
            peak = torch.cuda.max_memory_allocated(device)
            host, transfer = timed(device, lambda: output[0].cpu())
            return host, {"timed_queries": len(times), "full_native_axis": True, "time_index_lookup_s": lookup,
                          "backend_requested": backend, "initial_graph_capture_warmup_s": capture,
                          "initial_graph_validation_s": validation, "graph_full_axis_parity": agreement,
                          "resident_total_s": statistics.median(samples), "resident_samples_s": samples,
                          "resident_p95_s": float(np.percentile(samples, 95)), "selected": {
                              "readout": "joint_full_graph_fp32", "forward_s": statistics.median(samples),
                              "forward_samples_s": samples, "pressure_transfer_s": transfer,
                              "peak_allocated_bytes": peak}}
        except (RuntimeError, ValueError) as error:
            if backend == "graph":
                raise
            fallback, attempted_s = str(error), perf_counter() - started
        graph = None
        torch.cuda.empty_cache()
    setup_samples, output_samples, total_samples, peaks = [], [], [], []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        context, setup = timed(device, lambda: model.encode(velocity, source))
        output, projection = timed(device, lambda: model.decode(context, rows, normalized=False))
        total_samples.append(perf_counter() - started)
        if output.shape != (1, len(times), 201, 201):
            raise RuntimeError(f"expected complete joint output [1,{len(times)},201,201], got {tuple(output.shape)}")
        setup_samples.append(setup)
        output_samples.append(projection)
        if device.type == "cuda":
            peaks.append(torch.cuda.max_memory_allocated(device))
        del context
        if len(total_samples) < repeats:
            del output
    host, transfer = timed(device, lambda: output[0].cpu())
    if host[:, 0].count_nonzero():
        raise RuntimeError("joint FNO prediction violated the exact pressure free surface")
    return host, {"timed_queries": len(times), "full_native_axis": rows is None, "backend_requested": backend,
                  "graph_fallback": fallback, "attempted_graph_s": attempted_s,
                  "time_index_lookup_s": lookup, "spatial_encoding_s": statistics.median(setup_samples),
                  "resident_total_s": statistics.median(total_samples), "resident_samples_s": total_samples,
                  "resident_p95_s": float(np.percentile(total_samples, 95)),
                  "selected": {"readout": "joint_native_time_channels", "forward_s": statistics.median(output_samples),
                               "forward_samples_s": output_samples, "pressure_transfer_s": transfer,
                               "peak_allocated_bytes": max(peaks) if peaks else None}}


def read_record_inputs(dataset, record):
    """Read medium/source metadata without loading supervised pressure frames."""
    from data_v2 import _array_sha

    registered = dataset._bound[str(record)]
    dataset._check_source_file(registered)
    if registered.velocity_path:
        velocity = np.load(registered.velocity_path, allow_pickle=False).astype(np.float32)
        if _array_sha(velocity) != registered.information["velocity_sha256"]:
            raise ValueError("held-out velocity content changed")
        source = registered.source
    else:
        with h5py.File(registered.pressure_path, "r") as handle:
            velocity, source, _ = dataset._native_metadata(handle)
        np.testing.assert_array_equal(source, registered.source)
    return velocity, source


def main():
    from data_v2 import open_corpus

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--compatible-state", action="store_true",
                        help="explicitly accept an older spectral state using exact keys, shapes and dtypes")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--record", help="record ID from the configured corpus")
    source_group.add_argument("--instance", type=Path, help="NPZ with velocity_mps and source_parameters")
    parser.add_argument("--times", type=Path, help="NPY of explicit physical seconds; defaults to the full native axis")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tune-frames", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3, help="measured repeats per short tuning candidate")
    parser.add_argument("--full-repeats", type=int, default=1)
    parser.add_argument("--parity-rtol", type=float, default=2e-5)
    parser.add_argument("--joint-backend", choices=("auto", "eager", "graph"), default="auto",
                        help="default: verified FP32 CUDA graph for a full joint time axis, otherwise eager")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--blocks", type=int, nargs="+", help="benchmark blocks for a continuous-time decoder")
    mode.add_argument("--block", type=int, help="override the block for a continuous-time decoder")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path, help="optional NPZ output with times_s and pressure_pa [T,Z,X]")
    args = parser.parse_args()
    selected_blocks = args.blocks or ([] if args.block is None else [args.block])
    if min(*selected_blocks, args.repeats, args.full_repeats, args.tune_frames) < 1:
        parser.error("time-query block counts and repeats must be positive")
    if not np.isfinite(args.parity_rtol) or args.parity_rtol < 0:
        parser.error("parity tolerance must be finite and nonnegative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(config.get("cpu_threads", 2))
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        if "cuda_memory_fraction" in config:
            torch.cuda.set_per_process_memory_fraction(float(config["cuda_memory_fraction"]), device)
        torch.backends.cuda.matmul.allow_tf32 = False
    started = perf_counter()
    dataset = open_corpus(config)
    corpus_open_s = perf_counter() - started
    with args.checkpoint.open("rb") as stream:
        model, load = timed(device, lambda: load_model(config, stream, dataset, device, args.compatible_state))
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    started = perf_counter()
    if args.record is not None:
        velocity, source = read_record_inputs(dataset, args.record)
    else:
        with np.load(args.instance, allow_pickle=False) as instance:
            velocity = instance["velocity_mps"]
            source = instance["source_parameters"]
    times = (dataset.times if args.times is None else np.load(args.times, allow_pickle=False))
    if times.ndim != 1 or not len(times) or not np.isrealobj(times):
        raise ValueError("expected a nonempty array of physical times in seconds")
    host_inputs = tuple(x.copy() if isinstance(x, np.ndarray) and not x.flags.writeable else x
                        for x in (velocity, source, times))
    input_read_s = perf_counter() - started
    inputs, input_transfer_s = timed(device, lambda: tuple(torch.as_tensor(value, dtype=torch.float32, device=device)
                                                           for value in host_inputs))
    velocity_device, source_device, times_device = inputs
    joint = getattr(model, "variant", None) in JOINT_SCHEMAS
    if joint:
        pressure, report = measure_joint(model, *inputs, args.full_repeats, backend=args.joint_backend,
                                         parity_rtol=args.parity_rtol)
        if selected_blocks:
            report["requested_continuous_time_blocks"] = selected_blocks
    elif args.blocks is not None:
        pressure, report = benchmark(model, *inputs, list(dict.fromkeys(args.blocks)), args.tune_frames,
                                     args.repeats, args.full_repeats, args.parity_rtol)
    else:
        prepared, setup = timed(device, lambda: PreparedWavefield(model, velocity_device, source_device))
        pressure, timing = measure_cached(prepared, as_times(model, times_device), args.block, 1)
        report = {"timed_queries": len(times), "cached_preparation_s": setup,
                  "selected": {"time_block": args.block if args.block is not None else "configured", **timing}}
    started = perf_counter()
    if args.output is not None:
        with args.output.open("xb") as result:
            np.savez(result, times_s=times, pressure_pa=pressure.numpy())
    output_write_s = perf_counter() - started
    policy = "explicit_compatible_spectral_state" if args.compatible_state else "exact_v2_binding"
    report.update(checkpoint=str(args.checkpoint), checkpoint_sha256=digest,
                  checkpoint_policy="exact_joint_config_dataset_semantics" if joint else policy,
                  record_id=args.record, device=str(device), pressure_units="Pa", pressure_axis_order="TZX",
                  decoder_grid_size=201 if joint else model.decoder_grid_size,
                  pressure_dtype=str(pressure.dtype), times_s=times.tolist(),
                  full_native_axis=report.get("full_native_axis", args.times is None),
                  corpus_open_s=corpus_open_s, checkpoint_model_load_s=load, input_read_s=input_read_s,
                  input_transfer_s=input_transfer_s, optional_disk_output_s=output_write_s)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report is not None:
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="", flush=True)


if __name__ == "__main__":
    main()
