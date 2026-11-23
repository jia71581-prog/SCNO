"""Encode complete ordinary TRAIN pressure on the native 10 m grid."""

from argparse import ArgumentParser
from pathlib import Path
from time import monotonic
import scno_posix_lock as fcntl
import json

import numpy as np
from scipy.fft import rfft

from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
from marmousi_native_time_operator import BINS, FRAMES
from tianhe_observations import CommittedMarmousi
from tools.marmousi_target_dataset_refined.common import atomic_json, digest, read_json, sha_file
from train_no_pde_native import forbid_rollouts
from train_scno_paired_full_b128_hyper import PairedPressure


NATIVE = 201
SCHEMA = "marmousi_native201_pressure_spectra_v1"


def encode_shot(output, shot, rows, rms, identity, workers):
    names = [model for model, _, _ in rows]
    path = output / f"SHOT_{shot:02}.npy"
    receipt = path.with_suffix(".json")
    expected = (len(rows), BINS, NATIVE, NATIVE, 2)
    record = dict(shot=shot, models=names, shape=list(expected), plan=identity)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            saved = np.load(path, mmap_mode="r", allow_pickle=False)
            if read_json(receipt) != record or saved.shape != expected or saved.dtype != np.float16:
                raise ValueError(f"Saved native-grid shard fails verification: {shot}")
            return
        started = monotonic()
        stage = path.with_suffix(".tmp")
        packed = np.lib.format.open_memmap(stage, mode="w+", dtype=np.float16, shape=expected)
        for index, (_, _, folder) in enumerate(rows):
            field = np.load(folder / "pressure.npy", mmap_mode="r", allow_pickle=False)
            if field.shape != (FRAMES, NATIVE, NATIVE) or field.dtype != np.float32:
                raise ValueError(f"Unexpected raw pressure at {folder}")
            spectrum = rfft(field, axis=0, norm="ortho", workers=workers)[:BINS]
            packed[index, ..., 0] = spectrum.real / rms
            packed[index, ..., 1] = spectrum.imag / rms
            if not np.isfinite(packed[index]).all():
                raise FloatingPointError(f"Nonfinite native-grid spectrum at {folder}")
        packed.flush()
        del packed
        stage.replace(path)
        atomic_json(receipt, record)
        print(json.dumps(dict(shot=shot, samples=len(names), bytes=path.stat().st_size,
                              seconds=monotonic()-started)), flush=True)


def run(args):
    with forbid_rollouts():
        forbid_exact_label_files(args.dataset)
        release = CommittedMarmousi(args.dataset)
        pairs = PairedPressure(args.pairs, release, read_threads=1)
        plan = dict(schema=SCHEMA, release=release.binding, pairs=pairs.binding,
                    native_grid_m=10, shape=[NATIVE, NATIVE], bins=BINS, frames=FRAMES,
                    source_time_axis="physical_5ms", compression="fp16_complex_rfft_low_81_bins",
                    exact_marmousi_labels=0, source_sha=sha_file(Path(__file__)))
        identity = digest(plan)
        args.output.mkdir(parents=True, exist_ok=True)
        marker = args.output / "PLAN.json"
        if marker.exists() and read_json(marker) != plan:
            raise ValueError("Native-grid cache belongs to a different training release")
        atomic_json(marker, plan)
        print(json.dumps(dict(status="TRAIN_ROSTER_VERIFIED", samples=len(pairs.rows), sha256=identity)), flush=True)
        for shot in range(args.worker, len(release.sources), args.workers):
            rows = [row for row in pairs.rows if row[1] == shot]
            encode_shot(args.output, shot, rows, np.float32(release.rms), identity, args.threads)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("dataset", "pairs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--threads", type=int, default=12)
    run(parser.parse_args())
