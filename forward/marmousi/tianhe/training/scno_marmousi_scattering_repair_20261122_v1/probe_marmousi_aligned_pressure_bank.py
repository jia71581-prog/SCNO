"""Test a differentiable TRAIN-only pressure prior with velocity-dependent phase alignment."""

from argparse import ArgumentParser
from pathlib import Path
from time import monotonic
import json

import numpy as np
import torch
from torch.nn.functional import interpolate

from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
from marmousi_spectral_supervision import SpectralPairedPressure, SHARD_SIZE
from scno_forward_source_propagator import smooth_velocity, spatial_coarse
from scno_learned_background import ray_geometry
from tianhe_observations import CommittedMarmousi
from tools.marmousi_target_dataset_refined.common import atomic_json
from train_joint_pressure import PressureArchive
from train_no_pde_native import forbid_rollouts


FRAMES = 801
PROFILES = tuple((alignment, bandwidth) for alignment in ("none", "smooth", "native")
                 for bandwidth in (1., 2.5, 5.))


def geometry(velocity, source):
    coarse = spatial_coarse((velocity/2200.).log())[0]
    z = torch.linspace(0.,2000.,51,device=velocity.device)
    shallow = (.15+torch.exp(-z/500.)[:,None]).sqrt()
    descriptor = torch.cat((coarse.flatten(), (coarse*shallow).flatten(),2.*coarse.diff(dim=0).flatten())).double()
    smooth, native = (ray_geometry(v,source)[1][0] for v in (smooth_velocity(velocity,150.),velocity))
    return descriptor,smooth,native


def kriging(train, query, bandwidth):
    distances = torch.cdist(train,train)
    n = len(train)
    nearest = distances.masked_fill(torch.eye(n,device=train.device,dtype=torch.bool),float("inf")).amin(1)
    scale = nearest.median().clamp_min(1.e-6)*bandwidth
    gram = torch.exp(-.5*(distances/scale).square())+.005*torch.eye(n,device=train.device,dtype=train.dtype)
    rhs = torch.exp(-.5*(torch.cdist(train,query[None])[:,0]/scale).square())
    coefficients = torch.linalg.solve(gram,torch.stack((rhs,torch.ones_like(rhs)),dim=1))
    result,const = coefficients.T
    return (result+(1.-result.sum())*const/const.sum()).float()


def frequencies(values, train_travel, query_travel, weights, omega):
    if train_travel is not None:
        phase = (train_travel-query_travel)*omega
        values = values*torch.exp(1j*phase)
    p = torch.einsum("s,sfzx->fzx",weights.to(values.dtype),values)
    return torch.cat((p[:1], p[1:]/2.**.5))


def timescale_frequencies(values, train_travel, query_travel, train_v, query_v,
                          weights, omega, stretch):
    """Align the direct arrival and scale subsequent propagation by bulk slowness."""
    bulk_train = train_v[:, :151].reciprocal().mean(dim=(1, 2))
    bulk_query = query_v[:151].reciprocal().mean()
    ratio = (bulk_query/bulk_train).pow(stretch).clamp(.88, 1.14)
    centered = values*torch.exp(1j*train_travel*omega)
    locator = ratio[:, None]*torch.arange(values.size(1), device=values.device)
    lo = locator.floor().long().clamp(max=values.size(1)-1)
    hi = (lo+1).clamp(max=values.size(1)-1)
    cases = torch.arange(len(values), device=values.device)[:, None]
    fraction = (locator-lo).clamp(0., 1.)[:, :, None, None]
    spectrum = (1.-fraction)*centered[cases, lo] + fraction*centered[cases, hi]
    rebased = spectrum*torch.exp(-1j*query_travel*omega) * ratio[:, None, None, None]
    p = torch.einsum("s,sfzx->fzx",weights.to(rebased.dtype),rebased)
    return torch.cat((p[:1], p[1:]/2.**.5))


def collect(archive, pairs, shot, device):
    arrays, media, used = [], [], set()
    release = archive.release
    for i,(model,sid,_) in enumerate(pairs.rows):
        if sid != shot:
            continue
        info = release.rows[model]
        if info["split"] != "train" or info["is_exact_target"]:
            raise ValueError("The pressure prior can only use the released ordinary TRAIN media")
        parent = info["parent_group_id"]
        if parent in used:
            continue
        used.add(parent)
        shard, slot = divmod(i,SHARD_SIZE)
        if shard not in pairs.shards:
            entry = torch.load(pairs.cache/f"SHARD_{shard:03}.pt",weights_only=True,map_location="cpu",mmap=True)
            if entry["plan"] != pairs.receipt["plan"] or entry["rows"][slot] != [model,shot]:
                raise ValueError("An archived frequency source left its frozen TRAIN roster")
            pairs.shards[shard] = entry
        entry = pairs.shards[shard]
        if entry["rows"][slot] != [model,shot]:
            raise ValueError("A cached source is out of order")
        arrays.append(entry["pressure"][slot])
        media.append(pairs.velocities[model])
    if len(used) < 20:
        raise ValueError("Too few ordinary TRAIN parents for the physical receiver source")
    return torch.view_as_complex(torch.stack(arrays).to(device)),torch.as_tensor(np.stack(media),device=device)


def run(args):
    started = monotonic()
    torch.set_num_threads(args.threads)
    with torch.inference_mode(), forbid_rollouts():
        forbid_exact_label_files(args.dataset)
        release = CommittedMarmousi(args.dataset)
        archive = PressureArchive(args.validation,release)
        pairs = SpectralPairedPressure(args.pairs,release,args.cache,read_threads=1)
        frequencies_count = pairs.frequencies
        omega = 2.*torch.pi*torch.fft.rfftfreq(FRAMES,.005,device=args.device)[:frequencies_count]
        profiles = PROFILES + (("first_arrival", 2.5), ("first_arrival_blend", 2.5)) if args.library else PROFILES
        if args.library:
            from score_pressure_travel_coordinates import sweep_kernel
            solve, _ = sweep_kernel(args.library)
        stretch_profiles = (.5, 1.)
        names = [f"{name}_{width:g}" for name, width in profiles]
        names += [f"native_stretch_{stretch:g}" for stretch in stretch_profiles]
        errors = torch.zeros(len(names),dtype=torch.float64,device=args.device)
        energy = 0.
        records = []
        for index,(case,_,_) in enumerate(archive.items["validation"]):
            shot = case["shot_id"]
            ref,_,velocity = archive.arrays("validation",index)
            source = torch.as_tensor(release.sources[shot],dtype=torch.float32,device=args.device)
            coefficients,train = collect(archive,pairs,shot,args.device)
            quantity = lambda media: geometry(media,source)
            descriptor,smoothed_travel,native_travel = torch.vmap(quantity)(train)
            native_query = torch.tensor(np.array(velocity), device=args.device)
            query,query_sm,query_native = quantity(native_query)
            trains = {"none":None,"smooth":smoothed_travel[:,None],"native":native_travel[:,None]}
            queries = {"none":None,"smooth":query_sm,"native":query_native}
            if args.library:
                def arrivals(v):
                    return solve(v.cpu().numpy(), source.cpu().numpy())[::4, ::4].astype(np.float32)
                first = torch.tensor(np.stack([arrivals(v) for v in train]), device=args.device)[:, None]
                query_first = torch.tensor(arrivals(native_query), device=args.device)
                trains.update(first_arrival=first, first_arrival_blend=.5*(first+trains["native"]))
                queries.update(first_arrival=query_first, first_arrival_blend=.5*(query_first+query_native))
            angular = omega[None,:,None,None]
            weights = {width: kriging(descriptor, query, width) for width in (1., 2.5, 5.)}
            spectra = [frequencies(coefficients,trains[alignment],queries[alignment],
                                   weights[width],angular) for alignment,width in profiles]
            spectra.extend(timescale_frequencies(coefficients, trains["native"], query_native,
                                                  train, native_query, weights[2.5], angular, stretch)
                           for stretch in stretch_profiles)
            coarse = torch.fft.irfft(torch.stack(spectra), n=FRAMES, dim=1, norm="ortho")
            sums = torch.zeros_like(errors)
            ref_energy = 0.
            for slots in np.array_split(np.arange(FRAMES),26):
                y = torch.tensor(ref[slots].copy(),device=args.device)/release.rms
                p = interpolate(coarse[:,slots].flatten(0,1)[:,None],size=(201,201),
                                mode="bilinear",align_corners=True)[:,0]
                p = p.unflatten(0,(len(names),len(slots)))
                sums += (p-y).square().sum((1,2,3),dtype=torch.float64)
                ref_energy += float(y.square().sum(dtype=torch.float64))
            errors += sums
            energy += ref_energy
            row = dict(model=case["model_id"],shot=shot,train_medium_count=len(train),
                       relative_l2=(sums/ref_energy).sqrt().tolist())
            records.append(row)
            print(json.dumps(dict(**row,elapsed_s=monotonic()-started)),flush=True)
        metrics = dict(zip(names, (errors/energy).sqrt().tolist(), strict=True))
        result = dict(status="COMPLETE",archive=archive.binding,source_cache=pairs.receipt["plan"],
                      metrics=metrics,by_shot=records,exact_marmousi_labels=0)
        atomic_json(args.output,result)
        print(json.dumps(metrics),flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("dataset","pairs","cache","validation","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--library",type=Path)
    run(parser.parse_args())
