"""Compare first-arrival coordinates on the existing ordinary pressure bank."""

from argparse import ArgumentParser
import ctypes
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from evaluate_no_pde_sensitivity_v2 import array_sha
from expanded_train_pressure_archive import ExpandedPressureArchive
from fit_e12_b5 import atomic_npz, integrated_error, sha, write_json
from fit_pressure_travel_carrier import gram_matrix, mixtures, native_scores
from tianhe_observations import CommittedMarmousi
from train_joint_pressure import PressureArchive


def sweep_kernel(library):
    routine = ctypes.CDLL(str(library)).first_arrival
    routine.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                        ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,
                        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")]
    routine.restype = ctypes.c_int

    def solve(velocity, source):
        times = np.empty((201,201),dtype=np.float64)
        passes = routine(np.ascontiguousarray(velocity),201,201,10.,float(source[0]),float(source[1]),times)
        if passes <= 0 or np.any(times < 0.) or not np.isfinite(times).all():
            raise ValueError(f"First-arrival coordinate failed to converge ({passes})")
        return times

    velocity = np.full((201,201),2500.,dtype=np.float32)
    source = (912.8,40.)
    measured = solve(velocity,source)
    grid = np.arange(201)*10.
    exact = np.hypot(grid[:,None]-source[1],grid[None,:]-source[0])/2500.
    error = float(np.sqrt(np.square(measured-exact).mean()))
    if error > .004:
        raise ValueError(f"First-arrival axis or units failed homogeneous verification ({error:g} s)")
    return solve,error


def run(args):
    start = monotonic()
    torch.set_num_threads(args.threads)
    solve,qc = sweep_kernel(args.library)
    release = CommittedMarmousi(args.dataset)
    archive_old = PressureArchive(args.pressure_bank,release)
    archive = ExpandedPressureArchive(archive_old,args.expanded_bank,bank_a=args.bank_a,bank_f=args.bank_f)
    info = json.loads((args.cache/"CACHE.json").read_text())
    spectra_path = args.cache/"PRESSURE_SPECTRA.npy"
    geometry_path = args.cache/"GEOMETRY.npz"
    if (info["binding"] != archive.binding or info["exact_marmousi_used"] or
            info["spectra_sha256"] != sha(spectra_path) or info["descriptor_sha256"] != sha(geometry_path)):
        raise ValueError("Use the exact completed ordinary pressure bank")
    fields = np.load(spectra_path,mmap_mode="r",allow_pickle=False)
    with np.load(geometry_path,allow_pickle=False) as geom:
        descriptors,straight,tail = [geom[k] for k in ("descriptors","delays","unseen_energy")]
    roster = [*archive.items["train"],*archive._signed.values(),*archive.items["validation"]]
    if len(roster) != len(info["rows"]) or len(info["rows"]) != len(fields):
        raise ValueError("Missing a committed training or validation shot")
    corrected, x = [], []
    for (record,_,vfile), row in zip(roster,info["rows"],strict=True):
        if not all(record[key] == row[key] for key in ("model_id","shot_id","parent_group_id","family")):
            raise ValueError("Velocity and spectrum are not the same archived shot")
        speed = np.load(vfile,allow_pickle=False)
        if array_sha(speed) != record["velocity_sha256"] or speed.dtype != np.float32 or speed.shape != (201,201):
            raise ValueError("A registered ordinary velocity changed")
        source = release.sources[row["shot_id"]]
        x.append(source[0])
        corrected.append(solve(speed,source)[::2,::2].astype(np.float32))
    eikonal = np.asarray(corrected)
    x = np.asarray(x)
    args.output.mkdir(exist_ok=True,parents=True)
    atomic_npz(args.output/"FIRST_ARRIVAL.npz",ordinary_times_s=eikonal)
    n = info["train"]
    results = {}
    for name, delays in (("first_arrival",eikonal),("equal_blend",.5*(eikonal+straight))):
        gram, energy = gram_matrix(fields,delays,True,n,tail[n:])
        candidates=((float(np.sqrt(integrated_error(w,gram).sum()/energy.sum())), p, w)
                    for p,w in mixtures(descriptors,x,n))
        score, params, weight = min(candidates,key=lambda row:row[0])
        metrics=native_scores(archive,fields,weight,delays,n,True)
        results[name]=dict(parameters=params,coarse_relative_l2=score,**metrics)
        print(json.dumps({"mode":name,**{k:v for k,v in results[name].items() if k!="shots"}}),flush=True)
    write_json(args.output/"RESULT.json",dict(status="COMPLETE",archive=archive.binding,ordinary_train=n,
               ordinary_validation=len(roster)-n,time_samples=801,spacing_m=10.,exact_marmousi_used=False,
               numeric_wave_rollouts=0,coordinate_solver_offline_only=True,homogeneous_qc_rms_s=qc,
               library_sha256=sha(args.library),results=results,seconds=monotonic()-start))


def main():
    parser=ArgumentParser(description=__doc__)
    for name in ("dataset","pressure-bank","expanded-bank","bank-a","bank-f","cache","library","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--threads",type=int,default=4)
    run(parser.parse_args())


if __name__=="__main__":
    main()
