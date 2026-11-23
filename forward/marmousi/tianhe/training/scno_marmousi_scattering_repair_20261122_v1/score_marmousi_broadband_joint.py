"""Score one physically closed pressure/B5 pair from ordinary TRAIN waveforms."""

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
from scipy.fft import rfft
import torch
from torch.nn import functional as F

from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
from probe_marmousi_aligned_pressure_bank import kriging
from probe_marmousi_broadband_b5_transport import BINS, FRAMES, phase, predict
from train_marmousi_broadband_atlas import features
from tianhe_observations import CommittedMarmousi
from tools.marmousi_target_dataset_refined.common import atomic_json
from train_joint_pressure import PressureArchive
from train_no_pde_native import forbid_rollouts
from train_scno_paired_full_b128_hyper import PairedPressure


def raw_support(pairs, release, shot, stride, threads):
    used, p_spectra, b_spectra, media = set(), [], [], []
    for model, sid, folder in pairs.rows:
        if sid != shot:
            continue
        row = release.rows[model]
        group = row["parent_group_id"]
        if row["split"] != "train" or row["is_exact_target"]:
            raise ValueError("Only ordinary training references may enter the joint prediction")
        if group in used:
            continue
        used.add(group)
        p = np.load(folder/"pressure.npy",allow_pickle=False,mmap_mode="r")
        field = np.ascontiguousarray(p[:,::stride,::stride])/release.rms
        p_spectra.append(torch.from_numpy(rfft(field,axis=0,norm="ortho",workers=threads)[:BINS].copy()))
        b = np.load(folder/"b5.npy",allow_pickle=False)/release.rms
        b_spectra.append(torch.fft.rfft(torch.from_numpy(b),dim=0,norm="ortho")[:BINS,None])
        media.append(pairs.velocities[model])
    return torch.stack(p_spectra), torch.stack(b_spectra), torch.from_numpy(np.stack(media))


def transformed_mix(spectra, delays, target, ratios, weights, omega):
    n, f, nz, nx = spectra.shape
    out = spectra.new_zeros((f,nz,nx))
    physical_bins = torch.arange(f)
    for s in range(n):
        centered = spectra[s]*torch.exp(1j*delays[s]*omega)
        loc = ratios[s]*physical_bins
        first = loc.floor().long().clamp(max=f-1)
        last = (first+1).clamp(max=f-1)
        fraction = (loc-first).clamp(0.,1.)[:,None,None]
        out += (centered[first]*(1.-fraction)+centered[last]*fraction)*ratios[s]*weights[s]
    return out*torch.exp(-1j*target*omega)


def physical_pressure_spectra(fields, train, query, src, omega):
    desc, arrival, _ = torch.vmap(lambda v: features(v,src))(train)
    d, delay, _ = features(query,src)
    weights = kriging(desc,d,2.5)
    space = fields.size(-1)
    enlarged = F.interpolate(torch.cat((arrival,delay[None])).reshape(-1,1,51,51),
                             size=(space,space),mode="bicubic",align_corners=True)[:,0]
    ratios = (query[:151].reciprocal().mean()/train[:, :151].reciprocal().mean((1,2))).clamp(.88,1.14)
    return transformed_mix(fields,enlarged[:-1],enlarged[-1],ratios,weights,omega)


def matrix_lift(release):
    rows = release.observe.support_depth_rows
    matrix = release.observe.matrix.double().cpu()
    active = matrix[201:]
    lift = torch.linalg.solve(active.T @ active,active.T).float()
    return rows, matrix.float(), lift


def score_case(args, archive, pairs, case_index, omega, operator):
    release = archive.release
    record, _, _ = archive.items["validation"][case_index]
    shot = record["shot_id"]
    ref, original_b5, velocity = archive.arrays("validation",case_index)
    source = torch.as_tensor(release.sources[shot],dtype=torch.float32)
    field, receiver, media = raw_support(pairs,release,shot,args.stride,args.threads)
    v = torch.tensor(np.array(velocity))
    frequency_field = physical_pressure_spectra(field,media,v,source,omega)
    pressure = torch.fft.irfft(frequency_field,n=FRAMES,dim=0,norm="ortho")
    descriptor, delays = torch.vmap(lambda value: phase(value,source))(media)
    d, delay = phase(v,source)
    weights = kriging(descriptor,d,2.5)
    b5 = predict(receiver,delays[:,0,None],delay[0],weights,omega[None],media,v)[2]
    b5[:3].zero_()

    dims, observe, lift = operator
    metrics = torch.zeros((3,4),dtype=torch.float64)
    for slots in np.array_split(np.arange(FRAMES),26):
        reference = torch.from_numpy(ref[slots].copy())/release.rms
        ref_b5 = torch.from_numpy(original_b5[slots].copy())/release.rms
        desired = b5[slots]
        b_error = (desired-ref_b5).square().sum(dtype=torch.float64)
        b_energy = ref_b5.square().sum(dtype=torch.float64)
        for mode in range(3):
            if mode == 2:
                bilinear = F.interpolate(pressure[slots,None],size=(201,201),mode="bilinear",align_corners=True)[:,0]
                bicubic = F.interpolate(pressure[slots,None],size=(201,201),mode="bicubic",align_corners=True)[:,0]
                p = .5*(bilinear+bicubic)
            else:
                method = "bilinear" if mode == 0 else "bicubic"
                p = F.interpolate(pressure[slots,None],size=(201,201),mode=method,align_corners=True)[:,0]
            p[:,0].zero_()
            p[:np.searchsorted(slots,3)].zero_()
            current = p[:,:dims].flatten(1) @ observe
            p[:,1:dims] += ((desired-current) @ lift).reshape(len(slots),dims-1,201)
            closure = (p[:,:dims].flatten(1) @ observe - desired).abs().max()
            if float(closure) > 2.e-4:
                raise ValueError(f"The pressure/B5 closure is inaccurate: {closure}")
            metrics[mode] += torch.stack(((p-reference).square().sum(dtype=torch.float64),
                                          reference.square().sum(dtype=torch.float64),b_error,b_energy))
    return dict(model=record["model_id"],parent=record["parent_group_id"],shot=shot,
                training_parents=len(media),energies=metrics.tolist(),
                pressure_l2=(metrics[:,0]/metrics[:,1]).sqrt().tolist(),
                b5_l2=float((metrics[0,2]/metrics[0,3]).sqrt()))


def run(args):
    torch.set_num_threads(args.threads)
    with torch.inference_mode(), forbid_rollouts():
        forbid_exact_label_files(args.dataset)
        release = CommittedMarmousi(args.dataset)
        validation = PressureArchive(args.validation,release)
        pairs = PairedPressure(args.pairs,release,read_threads=1)
        omega = 2.*torch.pi*torch.fft.rfftfreq(FRAMES,.005)[:BINS,None,None]
        operator = matrix_lift(release)
        started = []
        for case in range(args.first, len(validation.items["validation"]), args.tasks):
            row = score_case(args,validation,pairs,case,omega,operator)
            started.append(row)
            atomic_json(args.output,dict(cases=started,archive=validation.binding,exact_marmousi_labels=0,
                                         compress_spacing=args.stride*10,final_spacing=10,status="RUNNING"))
            print(json.dumps(row),flush=True)
        atomic_json(args.output,dict(cases=started,archive=validation.binding,exact_marmousi_labels=0,
                                     compress_spacing=args.stride*10,final_spacing=10,status="COMPLETE"))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("dataset","pairs","validation","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--stride",type=int,choices=(2,4),default=2)
    parser.add_argument("--tasks",type=int,default=1)
    parser.add_argument("--first",type=int,default=0)
    run(parser.parse_args())
