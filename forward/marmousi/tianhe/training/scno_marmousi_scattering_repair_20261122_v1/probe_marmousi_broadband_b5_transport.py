"""Measure shallow phase and coda transport on the committed ordinary B5 traces."""

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
from probe_marmousi_aligned_pressure_bank import geometry, kriging, timescale_frequencies
from tianhe_observations import CommittedMarmousi
from tools.marmousi_target_dataset_refined.common import atomic_json
from train_joint_pressure import PressureArchive
from train_no_pde_native import forbid_rollouts
from train_scno_paired_full_b128_hyper import PairedPressure


FRAMES = 801
BINS = 81


def phase(velocity, source):
    descriptor, smooth, native = geometry(velocity, source)
    arrivals = (torch.stack((smooth[0],smooth[1],native[0],native[1]))[:,None])
    arrivals = F.interpolate(arrivals, size=201, mode="linear", align_corners=True)[:,0]
    return descriptor, .5*(arrivals[::2]+arrivals[1::2])[:, None]


def predict(spectrum, delay, target, weights, omega, train, query):
    direct = spectrum*torch.exp(1j*(delay-target)*omega)
    result = [torch.einsum("s,sfzx->fzx", weights.to(direct.dtype),direct)]
    normalized = torch.cat((spectrum[:,:1],spectrum[:,1:]*2.**.5),dim=1)
    result += [timescale_frequencies(normalized,delay,target,train,query,weights,omega,stretch)
               for stretch in (.5,1.)]
    packed = torch.stack(result)
    return torch.fft.irfft(packed,n=FRAMES,dim=1,norm="ortho")[:,:,0]


def run(args):
    torch.set_num_threads(args.threads)
    with forbid_rollouts(), torch.inference_mode():
        forbid_exact_label_files(args.dataset)
        release = CommittedMarmousi(args.dataset)
        archive = PressureArchive(args.validation,release)
        training = PairedPressure(args.pairs,release,read_threads=1)
        omega = 2.*torch.pi*torch.fft.rfftfreq(FRAMES,.005)[:BINS][None,:,None,None]
        rows = []
        for index,(case,_,_) in enumerate(archive.items["validation"]):
            shot = case["shot_id"]
            source = torch.as_tensor(release.sources[shot],dtype=torch.float32)
            _, trace, medium = archive.arrays("validation",index)
            y = torch.tensor(np.array(trace))/release.rms
            query = torch.tensor(np.array(medium))
            used, traces, speeds = set(), [], []
            for model, sid, folder in training.rows:
                if sid != shot:
                    continue
                record = release.rows[model]
                parent = record["parent_group_id"]
                if record["split"] != "train" or record["is_exact_target"]:
                    raise ValueError("B5 transport uses ordinary training parents only")
                if parent not in used:
                    used.add(parent)
                    traces.append(np.load(folder/"b5.npy",allow_pickle=False)/release.rms)
                    speeds.append(training.velocities[model])
            train = torch.from_numpy(np.stack(speeds))
            spectrum = torch.fft.rfft(torch.from_numpy(np.stack(traces)),dim=1,norm="ortho")[:, :BINS, None]
            descriptor, delays = torch.vmap(lambda v: phase(v,source))(train)
            target, t = phase(query,source)
            weights = kriging(descriptor,target,2.5)
            prediction = torch.cat([predict(spectrum,delays[:,i,None],t[i],weights,omega,train,query)
                                    for i in (0,1)])
            prediction[:,:3].zero_()
            errors = (prediction-y).square().sum((1,2),dtype=torch.float64)
            energy = float(y.square().sum(dtype=torch.float64))
            rows.append(dict(model=case["model_id"],parent=case["parent_group_id"],shot=shot,
                             energy=energy,errors=errors.tolist()))
            print(json.dumps(dict(**rows[-1],relative_l2=(errors/energy).sqrt().tolist())),flush=True)
        errors = np.sum([x["errors"] for x in rows],axis=0)
        metrics = (errors/sum(x["energy"] for x in rows))**.5
        names = [a+"_"+b for a in ("smooth","native") for b in ("shift","stretch_0.5","stretch_1")]
        result = dict(status="COMPLETE",archive=archive.binding,exact_marmousi_labels=0,
                      metrics=dict(zip(names,metrics.tolist(),strict=True)),cases=rows)
        atomic_json(args.output,result)
        print(json.dumps(result["metrics"]),flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("dataset","pairs","validation","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--threads",type=int,default=8)
    run(parser.parse_args())
