"""Make a compact, resumable all-time pressure target from committed ordinary TRAIN."""

from argparse import ArgumentParser
from pathlib import Path
from time import monotonic
import json

import numpy as np
from scipy.fft import rfft
import torch

from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
from make_direction_labels import split_lock
from tianhe_observations import CommittedMarmousi
from tools.marmousi_target_dataset_refined.common import atomic_json, digest, read_json, sha_file
from train_no_pde_native import forbid_rollouts
from train_scno_paired_full_b128_hyper import PairedPressure, atomic_checkpoint


SHARD_SIZE = 64
FREQUENCIES = 57


def spectrum(path, rms, workers):
    field = np.load(path, allow_pickle=False, mmap_mode="r")
    if field.shape != (801,201,201) or field.dtype != np.float32:
        raise ValueError(f"Invalid committed pressure: {path}")
    value = np.ascontiguousarray(field[:,::4,::4])
    value /= rms
    spec = rfft(value, axis=0, norm="ortho", workers=workers)[:FREQUENCIES].copy()
    spec[1:] *= np.float32(2.**.5)
    coords = spec.view(np.float32).reshape(FREQUENCIES,51,51,2)
    if not np.isfinite(coords).all():
        raise ValueError(f"Nonfinite full-axis spectral pressure at {path}")
    return torch.from_numpy(coords)


def cache(args):
    started = monotonic()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    with forbid_rollouts():
        forbid_exact_label_files(args.dataset)
        release = CommittedMarmousi(args.dataset)
        paired = PairedPressure(args.pairs,release,read_threads=1)
        rows = [(model,shot) for model,shot,_ in paired.rows]
        binding = dict(schema="marmousi_full_axis_train_spectrum_v1",data=paired.binding,
                       rows=[list(r) for r in rows],frames=801,frequencies=FREQUENCIES,
                       coarse_spacing_m=40,shard_size=SHARD_SIZE,exact_marmousi_labels=0,
                       source_sha256=sha_file(Path(__file__)))
        plan = args.output/"PLAN.json"
        with split_lock(args.output/".plan.lock"):
            if plan.exists() and read_json(plan) != binding:
                raise ValueError("The existing Fourier cache has a different contract")
            if not plan.exists():
                atomic_json(plan,binding)
        identity = digest(binding)
        paths = [args.output/f"SHARD_{i:03}.pt" for i in range((len(rows)+SHARD_SIZE-1)//SHARD_SIZE)]
        for shard in range(args.worker, len(paths), args.workers):
            path = paths[shard]
            lo, hi = shard*SHARD_SIZE, min((shard+1)*SHARD_SIZE,len(rows))
            with split_lock(path.with_suffix(".lock")):
                if path.exists():
                    state = torch.load(path,map_location="cpu",weights_only=True,mmap=True)
                    if state["plan"] != identity or state["rows"] != binding["rows"][lo:hi]:
                        raise ValueError(f"Fourier shard order changed: {path}")
                    continue
                values = {"pressure":[], "background_pressure":[]}
                for _,_,folder in paired.rows[lo:hi]:
                    for name in values:
                        values[name].append(spectrum(folder/f"{name}.npy", paired.rms, args.threads))
                value = {name:torch.stack(v) for name,v in values.items()}
                atomic_checkpoint(path,dict(plan=identity,rows=binding["rows"][lo:hi],**value))
            print(json.dumps(dict(worker=args.worker,shard=shard,shot_pairs=hi-lo,
                                  elapsed_s=monotonic()-started)),flush=True)
        with split_lock(args.output/".status.lock"):
            count = sum(path.exists() for path in paths)
            if count == len(paths):
                atomic_json(args.output/"STATUS.json",dict(status="COMPLETE",plan=identity,pairs=len(rows),
                                                           shards=len(paths),exact_marmousi_labels=0))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("dataset","pairs","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--workers",type=int,required=True)
    parser.add_argument("--worker",type=int,required=True)
    parser.add_argument("--threads",type=int,default=2)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers or args.threads <= 0:
        parser.error("Supply a valid worker index and thread count")
    cache(args)
