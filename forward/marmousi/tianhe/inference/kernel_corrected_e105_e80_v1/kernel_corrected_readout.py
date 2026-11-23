"""Restore the verified Marmousi E105/E80 plus kernel B5 inference operator."""

from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple
import json

import numpy as np
import torch

import kernel_receiver
import spline_checkpoint
from tianhe_observations import CommittedMarmousi


KERNEL_SHA = "3862f4f2c486c9b6b023c4a0842b3562810eca4a8bb19b3a1c1a2ecaca51b7da"
ANCHOR_SHA = "ae2cdb47bc3a0cb30f6e832ef98ed91bdd58ac387262af402a95d90bd99e2834"
NATIVE_SHA = "56b8fe5aaf1a1ccefc5f125f013cdfc2777327a056cdef6574b4a83c1452dca8"
KERNEL_CODE_SHA = "4b90a0a4d1a058d44f4bf018be2494be7652d04962b0cd25e1dc12f22e0efe82"
SPLINE_LOADER_SHA = "0c154e9519817476b39e383f5e7a3594255ea0ca2b185c2d1db167b23c96b308"
SPLINE_HEAD_SHA = "ccd5501391d941fcb9c0ba30390b31f48147f737ce85f6da2e9497ba257b6055"
ARRIVAL_LOADER_SHA = "516d2b7c13c87f7cb9dce1792d1b68fbd439cb17ef10e98c7cfd6512990fa730"


def require(condition, message):
    if not condition:
        raise ValueError(message)


class EncodedB5(NamedTuple):
    native_context: object
    normalized_kernel_traces: torch.Tensor


class KernelCorrectedReadout:
    """Physical inputs and fixed-frame, differentiable B5 outputs."""

    def __init__(self, native_core, kernel, times, receiver_rms):
        self.native_core = native_core
        self.kernel = kernel
        self.device = native_core.device
        self.times = torch.as_tensor(times, dtype=torch.float64, device=self.device)
        self.receiver_rms = receiver_rms
        self.sources = kernel.physical_sources

    def eval(self):
        self.native_core.eval()
        self.kernel.eval()
        return self

    def encode(self, velocity, source):
        velocity = torch.as_tensor(velocity, dtype=torch.float32, device=self.device)
        source = torch.as_tensor(source, dtype=torch.float32, device=self.device)
        if velocity.ndim == 2:
            velocity = velocity[None]
        if velocity.ndim == 4 and velocity.shape[1] == 1:
            velocity = velocity[:, 0]
        if source.ndim == 1:
            source = source[None]
        require(velocity.ndim == 3 and velocity.shape[1:] == (201, 201)
                and source.ndim == 2 and source.shape[1] == 5 and len(source) > 0
                and len(velocity) in (1, len(source)), "Use one shared or per-source 201 × 201 velocity in m/s")
        correction = self.kernel(velocity.expand(len(source), -1, -1), source)
        return EncodedB5(self.native_core.encode(velocity, source), correction)

    def decode(self, context, frame_indices=None, *, normalized=True):
        stored = context.normalized_kernel_traces
        slots = (torch.arange(len(self.times), device=self.device) if frame_indices is None
                 else torch.as_tensor(frame_indices, device=self.device))
        require(slots.dtype in (torch.int32, torch.int64) and slots.ndim in (1, 2) and slots.numel() > 0
                and (slots.ndim == 1 or slots.shape[0] == len(stored))
                and (slots >= 0).all() and (slots < len(self.times)).all(), "Use registered integer frame indices 0–800")
        slots = slots.long()
        selection = stored[:, slots] if slots.ndim == 1 else stored.gather(1, slots[..., None].expand(-1, -1, stored.shape[-1]))
        baseline = torch.cat([self.native_core.decode(context.native_context, chunk, normalized=True)
                              for chunk in slots.split(128, dim=-1)], dim=1)
        result = baseline + selection
        return result if normalized else result * self.receiver_rms

    def __call__(self, velocity, source, frame_indices=None, *, normalized=True):
        return self.decode(self.encode(velocity, source), frame_indices, normalized=normalized)


def validate_kernel_release(artifact, data):
    binding = artifact["binding"]
    require(binding["schema"] == "marmousi_kernel_b5_residual_v1" and binding["based_on_train_only"] is True
            and binding["dataset_binding"] == data.binding and binding["normalization_receiver_rms"] == data.rms
            and binding["anchor_sha256"] == ANCHOR_SHA and binding["native_e80_sha256"] == NATIVE_SHA
            and binding["sources_sha256"]["kernel_receiver.py"] == KERNEL_CODE_SHA,
            "Kernel artifact does not bind the exact E105/E80 and frozen release")
    registered = torch.from_numpy(data.sources.astype(np.float32))
    rows = artifact["decode_rows"]
    require(tuple(data.trace_shape) == (801, 201) and len(data.train_ids) == 800
            and registered.shape == (38, 5) and (registered[:, 4] == 1.).all()
            and torch.equal(artifact["sources"], registered)
            and rows.shape == (801, 163) and rows.dtype == torch.float32
            and rows.isfinite().all() and not rows[0].count_nonzero()
            and artifact["descriptors"].shape == (800, kernel_receiver.DESCRIPTOR_SIZE)
            and len(artifact["blocks"]) == 38, "Kernel source, native time or B5 geometry changed")
    position = {model: index for index, model in enumerate(sorted(data.train_ids))}
    total = 0
    for source, (indices, factor, whitened) in enumerate(artifact["blocks"]):
        expected = torch.tensor([position[model] for model in data.keys["train"] if source in data.rosters[model]],
                                dtype=torch.long)
        count = len(expected)
        require(torch.equal(indices, expected) and factor.shape == (count, count) and factor.dtype == torch.float64
                and whitened.shape == (count, 163 * 201) and whitened.dtype == torch.float32,
                "Kernel per-source coefficients differ from the registered TRAIN roster")
        total += count
    require(total == 3200, "Kernel release must bind all 3200 TRAIN shots")
    return binding


def load_kernel_corrected_readout(dataset, checkpoint, *, device="cpu"):
    """Load the verified B5 comparator and its separate native pressure core."""
    digest = spline_checkpoint.digest
    require(all(digest(path) == expected for path, expected in (
        (spline_checkpoint.__file__, SPLINE_LOADER_SHA),
        (spline_checkpoint.cached_receiver_spline.__file__, SPLINE_HEAD_SHA),
        (spline_checkpoint.arrival_checkpoint.__file__, ARRIVAL_LOADER_SHA),
        (kernel_receiver.__file__, KERNEL_CODE_SHA), (checkpoint, KERNEL_SHA))),
        "Kernel checkpoint or original inference implementation differs")
    data = CommittedMarmousi(dataset)
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    binding = validate_kernel_release(artifact, data)
    native_weights = Path(binding["native_checkpoint"])
    require(digest(native_weights) == NATIVE_SHA, "Native E80 checkpoint differs")
    saved = torch.load(native_weights, map_location="cpu", weights_only=True, mmap=True)
    native_binding = saved["binding"]
    require(saved["epoch"] == 80 and saved["updates"] == 4000 and saved["offset"] == 0
            and native_binding["cache_sha256"] == binding["cache_sha256"]
            and native_binding["anchor_update"] == 5250 and native_binding["anchor_sha256"] == ANCHOR_SHA,
            "Kernel and native observations do not share the exact E105 cache")
    device = torch.device(device)
    core = spline_checkpoint.load_spline_readout(data.root, native_weights, device=device)
    require(float(core.anchor.model.normalizer.metadata.pressure_scale_pa) == data.rms,
            "Native receiver pressure normalization differs from the registered numerical RMS")
    for component in (core.anchor.model, core.anchor.residual, core.residual):
        component.requires_grad_(False)
    kernel = kernel_receiver.KernelB5Residual(artifact["descriptors"], artifact["sources"], artifact["blocks"],
                                              artifact["decode_rows"], binding["length"]).to(device).eval()
    return KernelCorrectedReadout(core, kernel, data.times, data.rms).eval()


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("dataset", "checkpoint", "velocity", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-index", required=True, type=int, help="Zero-based row of the registered physical sources")
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--numeric-pressure", action="store_true", help="Return LWC numerical pressure instead of normalized B5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    require(args.cpu_threads > 0 and 0 <= args.source_index < 38 and not args.output.exists(),
            "Use a registered source index, positive CPU threads and a fresh output file")
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("highest")
    if torch.device(args.device).type == "cuda":
        torch.backends.cudnn.allow_tf32 = False
    readout = load_kernel_corrected_readout(args.dataset, args.checkpoint, device=args.device)
    v = np.load(args.velocity, allow_pickle=False)
    source = readout.sources[args.source_index]
    indices = np.arange(801) if args.frame_indices is None else np.asarray(args.frame_indices, dtype=np.int64)
    with torch.inference_mode():
        prediction = readout(v, source, indices, normalized=not args.numeric_pressure).cpu().numpy()
    units = "lwc_numerical_pressure" if args.numeric_pressure else "frozen_train_rms_normalized"
    with args.output.open("xb") as stream:
        np.savez(stream, b5=prediction, frame_indices=indices, times_s=readout.times.cpu().numpy()[indices],
                 source=source.cpu().numpy(), units=units, receiver_rms=readout.receiver_rms,
                 kernel_checkpoint_sha256=KERNEL_SHA, spline_checkpoint_sha256=NATIVE_SHA)
    print(json.dumps({"output": str(args.output), "shape": list(prediction.shape), "units": units}, allow_nan=False))


if __name__ == "__main__":
    main()
