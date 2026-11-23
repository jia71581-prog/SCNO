"""Resident full-trajectory inference and verification for the joint SCNO time head."""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F


@contextmanager
def precision_policy(precision):
    if precision not in ("fp32", "tf32", "bf16", "fp16"):
        raise ValueError("unknown inference precision")
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = precision != "fp32"
        torch.backends.cudnn.allow_tf32 = precision != "fp32"
        with torch.autocast("cuda", enabled=precision in ("bf16", "fp16"), cache_enabled=False,
                            dtype=torch.bfloat16 if precision == "bf16" else torch.float16):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


class _SourceFeatures(NamedTuple):
    hidden: torch.Tensor
    rank: torch.Tensor
    map_field: torch.Tensor


class _ResidentSource:
    """Inference-only SourceEncoderV3 forward with its trained zero-IC response precomputed."""

    def __init__(self, model, velocity, batch):
        self.module = model.source_encoder
        self.ic_field = None
        if self.module.ic_frames:
            zero = velocity.new_zeros((batch, self.module.ic_frames, 201, 201))
            self.ic_field = self.module.ic_encoder(zero)

    def __call__(self, source, source_map, medium, mapping, anchor):
        encoder = self.module
        grid = source[:, :2].mul(2.).sub(1.).reshape(-1, 1, 1, 2)

        def sample(field):
            return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True).flatten(1)

        map_field = encoder.map_encoder(source_map)
        hidden = (encoder.parameter_mlp(source) + encoder.local_projection(sample(medium.pyramid[0][mapping]))
                  + encoder.map_projection(sample(map_field)))
        hidden = hidden + encoder.anchor_projection(anchor)
        if self.ic_field is not None:
            map_field = map_field + self.ic_field
            hidden = hidden + encoder.ic_projection(sample(self.ic_field))
        hidden = F.gelu(encoder.hidden_norm(hidden))
        return _SourceFeatures(hidden, encoder.rank_projection(hidden), map_field)


def _stamp(model):
    tensors = (*model.parameters(), *model.buffers())
    return (id(model.normalizer), id(model.normalizer.metadata), model.duration_s,
            tuple(m.training for m in model.modules()), tuple((id(t), t.data_ptr(), t._version) for t in tensors))


def _check_capture_memory(model, batch):
    device = model.device
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    available = free + torch.cuda.memory_reserved(device) - allocated
    fraction = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    if fraction is not None:
        available = min(available, int(fraction(device)*total) - allocated)
    wavefield_bytes = batch * len(model.full_times_s) * 201 * 201 * 4
    required = 2 * wavefield_bytes + batch * 128 * 2**20
    if available < required:
        raise torch.OutOfMemoryError(f"full graph and strict verification need at least {required/2**20:.0f} "
                                     f"MiB; currently available {available/2**20:.0f} MiB")


class JointFNOFullGraph:
    """Single-stream inference on changing, same-shape resident media and sources.

    Every run computes the medium, location and source conditioning and the complete
    pressure output in Pa. The captured result is borrowed until the next run;
    pass copy_output=True for independently owned pressure. Rebuild after any
    model update. Capture and kernel warm-up are one-time per model and shape.
    """

    @torch.no_grad()
    def __init__(self, model, velocity, source, *, precision="fp32"):
        if model.training or model.device.type != "cuda":
            raise ValueError("capture requires a CUDA joint model in evaluation mode")
        self.model = model
        self.precision = precision
        self.stamp = _stamp(model)
        self.stream = torch.cuda.current_stream(model.device)
        velocity, source = model._inputs(velocity, source)
        _check_capture_memory(model, len(source))
        self.velocity, self.source = velocity.contiguous().clone(), source.contiguous().clone()
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=model.device)
        stream.wait_stream(self.stream)
        with torch.cuda.device(model.device), torch.cuda.stream(stream), precision_policy(precision):
            self.source_encoder = _ResidentSource(model, self.velocity, len(source))
            for _ in range(3):
                self._forward()
            stream.synchronize()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = self._forward()
        self.stream.wait_stream(stream)

    def _forward(self):
        context = self.model._encode_validated(self.velocity, self.source, inference_source=self.source_encoder)
        return self.model.decode(context, normalized=False).float()

    @torch.no_grad()
    def __call__(self, velocity, source, *, copy_output=False):
        if self.model.training or _stamp(self.model) != self.stamp:
            raise RuntimeError("joint model changed; rebuild the full-trajectory graph")
        if torch.cuda.current_stream(self.model.device) != self.stream:
            raise RuntimeError("use the capture owner's stream or construct another graph for this stream")
        if (not torch.is_tensor(velocity) or not torch.is_tensor(source)
                or velocity.device != self.model.device or source.device != self.model.device):
            raise ValueError("supply resident velocity and source tensors on the model device")
        velocity, source = self.model._inputs(velocity, source)
        if self.velocity.shape != velocity.shape or self.source.shape != source.shape:
            raise ValueError("medium and source shapes must match the capture")
        self.velocity.copy_(velocity)
        self.source.copy_(source)
        self.graph.replay()
        return self.output.clone() if copy_output else self.output


def _full_error(gold, actual):
    if actual.shape != gold.shape or actual.dtype != torch.float32:
        raise ValueError("incomplete native pressure or inconsistent return dtype")
    moment = torch.zeros(4, device=actual.device, dtype=torch.float64)
    for ref, candidate in zip(gold.split(32, dim=1), actual.split(32, dim=1)):
        ref = ref.double()
        error = candidate.double() - ref
        moment[0] += error.square().sum()
        moment[1] += ref.square().sum()
        moment[2] = torch.maximum(moment[2], error.abs().amax())
        moment[3] = torch.maximum(moment[3], ref.abs().amax())
    e2, r2, peak, ref_peak = moment.cpu().tolist()
    return {"relative_l2_vs_eager_fp32": (e2 / max(r2, 1e-300))**.5, "max_abs_difference_pa": peak,
            "max_abs_difference_over_reference_peak": peak/max(ref_peak, 1e-300),
            "finite": bool(torch.isfinite(actual).all()), "top_max_abs_pa": float(actual[:, :, 0].abs().max())}


@torch.no_grad()
def verify_full_graph(graph, velocity, source, *, tolerance=0.):
    """Check every output point at every native time against the same eager FP32 weights."""
    with precision_policy("fp32"):
        gold = graph.model(velocity, source, normalized=False)
    measured = _full_error(gold, graph(velocity, source))
    if (not measured["finite"] or measured["top_max_abs_pa"] != 0.
            or not measured["relative_l2_vs_eager_fp32"] <= tolerance):
        raise ValueError(f"full-trajectory graph failed FP32 reference verification: {measured}")
    return measured


def _gpu_context():
    query = "nvidia-smi --query-gpu=name,pstate,utilization.gpu,memory.used --format=csv,noheader"
    jobs = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
    return {"device": subprocess.check_output(query.split(), text=True).splitlines(),
            "compute_pids_mib": subprocess.check_output(jobs.split(), text=True).splitlines()}


def _timing(function, inputs, records, repetitions):
    device = inputs[0][0].device
    for i in range(3):
        function(*inputs[i % len(inputs)])
    torch.cuda.synchronize(device)
    wall, events = [], []
    for i in range(repetitions):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        t0 = perf_counter()
        start.record()
        value = function(*inputs[i % len(inputs)])
        stop.record()
        stop.synchronize()
        wall.append(1e3*(perf_counter()-t0))
        events.append(start.elapsed_time(stop))
        del value
    per_record = {}
    for i, record in enumerate(records):
        samples = wall[i::len(records)]
        per_record[str(record)] = {"count": len(samples), "p50_ms": float(np.median(samples)),
                                   "p95_ms": float(np.quantile(samples, .95)), "max_ms": max(samples)}
    return {"wall_ms": wall, "gpu_event_ms": events, "p50_ms": float(np.median(wall)),
            "p95_ms": float(np.quantile(wall, .95)), "max_ms": max(wall), "per_record_wall": per_record}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--joint", action="store_true", help="load trained joint time-channel weights")
    parser.add_argument("--records", nargs="+", help="two or more independent validation record IDs")
    parser.add_argument("--modes", nargs="+", default=["eager-fp32", "graph-fp32", "graph-tf32", "graph-bf16"])
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--max-gpu-fraction", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from data_v2 import open_corpus
    from joint_fno_model import JointFNOSCNO
    from predict_v2 import load_model, read_record_inputs

    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    if args.max_gpu_fraction is not None:
        if not 0 < args.max_gpu_fraction <= 1:
            raise ValueError("max GPU fraction must be in (0,1]")
        torch.cuda.set_per_process_memory_fraction(args.max_gpu_fraction, device)
    config = json.loads(args.config.read_text())
    corpus = open_corpus(config)
    with args.checkpoint.open("rb") as stream:
        model = load_model(config, stream, corpus, torch.device("cpu"), False)
        stream.seek(0)
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        metadata = torch.load(stream, map_location="meta", weights_only=True)
        progress = {k: metadata[k] for k in ("epochs_completed", "updates_completed") if k in metadata}
        del metadata
    if not args.joint:
        model = JointFNOSCNO.from_native(model, freeze_encoder=True)
    if not isinstance(model, JointFNOSCNO):
        raise ValueError("expected the trained joint checkpoint or a native parent for initialization")
    model = model.to(device).eval()
    records = args.records or list(corpus.validation_ids[:2])
    if len(set(records)) < 2 or args.repeats < len(records):
        raise ValueError("use at least two independent records and measure each record at least once")
    inputs = [tuple(torch.tensor(np.array(x, copy=True), dtype=torch.float32, device=device) for x in read_record_inputs(corpus, rid))
              for rid in records]
    report = {"family": model.family, "native_shape": [1, len(model.full_times_s), 201, 201],
              "precision_reference": "all FP32, TF32 off for matmul and convolution", "input_output": "resident velocity [m/s] and source -> full GPU pressure [Pa]",
              "weights": "trained_joint" if args.joint else "native_encoder_with_seeded_fresh_joint_head",
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha, "checkpoint_progress": progress, "seed": args.seed,
              "record_ids": records, "runtime": torch.__version__, "initial_gpu": _gpu_context(), "modes": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(args.output)

    with torch.no_grad(), precision_policy("fp32"):
        gold = [model(v, s, normalized=False).float() for v,s in inputs]
        report["independent_replay_check"] = {"first_two_velocity_inputs_differ": not torch.equal(inputs[0][0], inputs[1][0]),
                                              "first_two_source_inputs_differ": not torch.equal(inputs[0][1], inputs[1][1]),
                                              "first_two_reference_outputs_differ": not torch.equal(gold[0], gold[1])}
        if not report["independent_replay_check"]["first_two_reference_outputs_differ"]:
            raise ValueError("the replay cross-check requires two distinct complete reference predictions")
    for mode in args.modes:
        path, precision = mode.split("-")
        result = {"gpu_before": _gpu_context()}
        report["modes"][mode] = result
        graph = None
        try:
            t0 = perf_counter()
            if path == "graph":
                graph = JointFNOFullGraph(model, *inputs[0], precision=precision)
                call = graph
            elif path == "eager":
                @torch.no_grad()
                def call(v, s):
                    return model(v, s, normalized=False).float()
            else:
                raise ValueError(f"unknown execution path {path}")
            torch.cuda.synchronize(device)
            result["setup_s"] = perf_counter()-t0
            with torch.no_grad(), precision_policy(precision):
                result["records"] = {str(k): _full_error(ref, call(*pair)) for k,ref,pair in zip(records, gold, inputs)}
                torch.cuda.reset_peak_memory_stats(device)
                result.update(_timing(call, inputs, records, args.repeats))
                result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device)/2**20
            result["gpu_after"] = _gpu_context()
            print(f"{mode}: p50={result['p50_ms']:.1f} p95={result['p95_ms']:.1f} ms", flush=True)
        except (RuntimeError, ValueError) as error:
            result["error"] = str(error)
            print(f"{mode}: {error}", flush=True)
        finally:
            graph = call = None
            gc.collect()
            torch.cuda.empty_cache()
            save()


if __name__ == "__main__":
    main()
