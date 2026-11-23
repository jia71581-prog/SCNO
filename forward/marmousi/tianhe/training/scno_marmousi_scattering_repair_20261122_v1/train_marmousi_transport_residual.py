"""Train a pressure residual around the broadband, observation-closed Marmousi carrier."""

from argparse import ArgumentParser
from pathlib import Path
from time import monotonic
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
from probe_marmousi_aligned_pressure_bank import kriging
from probe_marmousi_broadband_b5_transport import phase, predict
from score_marmousi_broadband_joint import BINS, FRAMES, matrix_lift, raw_support
from score_marmousi_broadband_joint import physical_pressure_spectra
from tianhe_observations import CommittedMarmousi
from tools.marmousi_target_dataset_refined.common import atomic_json, digest, read_json, sha_file
from train_joint_pressure import PressureArchive
from train_marmousi_broadband_atlas import features
from train_no_pde_native import forbid_rollouts
from train_scno_paired_full_b128_hyper import PairedPressure, atomic_checkpoint


SIDE = 101
POINTS = 128
TARGET = .15
STRICT_PARENT = "validation_F_parent_0000"


def medium_features(medium, source):
    d, t, x = features(medium, source)
    grid = torch.cat((t[None],x.T)).reshape(-1,51,51)
    grid = F.interpolate(grid[None],size=(SIDE,SIDE),mode="bicubic",align_corners=True)[0]
    return d, grid[0], grid[1:].flatten(1).T


def crossfit_weights(desc):
    d = torch.cdist(desc,desc)
    eye = torch.eye(len(d),dtype=torch.bool)
    scale = 2.5*d.masked_fill(eye,float("inf")).amin(1).median().clamp_min(1.e-6)
    precision = torch.linalg.inv(torch.exp(-.5*(d/scale).square())+.005*eye)
    const = precision.sum(1)
    centered = precision-const[:,None]*const[None]/const.sum()
    w = -centered/centered.diag()[:,None]
    return w.masked_fill(eye,0.).float()


def aligned_samples(spectrum, delay, local, medium, weights, omega, shot):
    freq = torch.arange(BINS)
    rng = torch.Generator().manual_seed(86708+shot)
    positions = torch.randint(SIDE,SIDE**2,(len(weights),POINTS),generator=rng)
    bulk = medium[:, :151].reciprocal().mean((1,2))
    dense = spectrum.flatten(2).transpose(1,2).contiguous()
    dense *= torch.exp(1j*delay.flatten(1)[...,None]*omega)
    bases = []
    for i, points in enumerate(positions):
        ratio = (bulk[i]/bulk).clamp(.88,1.14)
        indices = ratio[:,None]*freq
        lo = indices.floor().long().clamp(max=BINS-1)
        hi = (lo+1).clamp(max=BINS-1)
        values = dense[:,points]
        a,b = [values.gather(2,j[:,None].expand(-1,POINTS,-1)) for j in (lo,hi)]
        fraction = (indices-lo).clamp(0.,1.)[:,None]
        match = ((1.-fraction)*a+fraction*b)*(weights[i]*ratio)[:,None,None]
        bases.append(match.sum(0))
    ids = torch.arange(len(weights))[:,None]
    rows = dict(base=torch.stack(bases), label=dense[ids,positions], feature=local[ids,positions],
                delay=delay.flatten(1)[ids,positions])
    if not all(torch.isfinite(v).all() for v in rows.values()):
        raise FloatingPointError("Nonfinite training features")
    return rows


def validation_input(archive, i, query, source, train, p, b, omega):
    _, delay, local = medium_features(query,source)
    field = physical_pressure_spectra(p,train,query,source,omega[:,None,None])
    aligned = field.flatten(1).T*torch.exp(1j*delay.flatten()[...,None]*omega)
    desc, travel = torch.vmap(lambda v: phase(v,source))(train)
    d, t = phase(query,source)
    weights = kriging(desc,d,2.5)
    receiver = predict(b,travel[:,0,None],t[0],weights,omega[None,:,None,None],train,query)[2]
    receiver[:3].zero_()
    return dict(index=i,base=aligned,delay=delay.flatten(),feature=local,receiver=receiver,
                key=archive.items["validation"][i][0]["model_id"])


def prepare(args, release, archive, omega, binding):
    pairs = PairedPressure(args.pairs,release,read_threads=1)
    stamp = digest(binding)
    sources = torch.as_tensor(release.sources,dtype=torch.float32)
    for shot in range(args.worker,len(sources),args.workers):
        path = args.cache/f"SHOT_{shot:02}.pt"
        if path.is_file():
            prior = torch.load(path,weights_only=True,mmap=True,map_location="cpu")
            if prior["binding"] != stamp or prior["shot"] != shot:
                raise ValueError("An existing transport shard uses different data")
            continue
        src = sources[shot]
        p,b,media = raw_support(pairs,release,shot,2,args.threads)
        query_desc, delay, local = torch.vmap(lambda v: medium_features(v,src))(media)
        train = aligned_samples(p,delay,local,media,crossfit_weights(query_desc),omega,shot)
        validation = []
        for i,(case,_,_) in enumerate(archive.items["validation"]):
            if case["shot_id"] == shot:
                ref_velocity = torch.tensor(np.array(archive.arrays("validation",i)[2]))
                validation.append(validation_input(archive,i,ref_velocity,src,media,p,b,omega))
        atomic_checkpoint(path,dict(binding=stamp,shot=shot,train=train,validation=validation))
        print(json.dumps(dict(shot=shot,train_parents=len(media),validation=len(validation))),flush=True)


class SpectralResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2*BINS+6,128),nn.LayerNorm(128),nn.SiLU(),
                                     nn.Linear(128,128),nn.SiLU(),nn.Linear(128,4*BINS))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, base, feature):
        scale = base.abs().square().mean(-1,keepdim=True).clamp_min(1.e-7).sqrt()
        normalized = torch.view_as_real(base/scale).flatten(-2)
        code = self.network(torch.cat((normalized,feature),dim=-1)).tanh()
        gain, forcing = torch.view_as_complex(code.unflatten(-1,(2,BINS,2)).contiguous()).unbind(-2)
        return base*(1.+.045*gain)+.025*scale*forcing


def energy(value, reference):
    return torch.stack(((value-reference).square().sum(dtype=torch.float64),
                        reference.square().sum(dtype=torch.float64)))


@torch.inference_mode()
def score(model, archive, saved, omega, device):
    depth, observation, lift = (x.to(device) if torch.is_tensor(x) else x for x in matrix_lift(archive.release))
    records = []
    for row in saved:
        i = row["index"]
        record = archive.items["validation"][i][0]
        ref_field, ref_receiver, _ = archive.arrays("validation",i)
        if row["key"] != record["model_id"]:
            raise ValueError("A validation wavefield lost its source binding")
        v = model(row["base"].to(device),row["feature"].to(device))
        p = v*torch.exp(-1j*row["delay"].to(device)[:,None]*omega)
        wave = torch.fft.irfft(p.T.reshape(BINS,SIDE,SIDE),n=FRAMES,dim=0,norm="ortho")
        receiver = row["receiver"].to(device)
        norms = torch.zeros(4,dtype=torch.float64,device=device)
        worst = 0.
        for slots in np.array_split(np.arange(FRAMES),26):
            native = wave[slots,None]
            linear = F.interpolate(native,size=(201,201),mode="bilinear",align_corners=True)
            cubic = F.interpolate(native,size=(201,201),mode="bicubic",align_corners=True)
            p = .5*(linear[:,0]+cubic[:,0])
            p[:,0].zero_()
            p[:np.searchsorted(slots,3)].zero_()
            desired = receiver[slots]
            b = p[:,:depth].flatten(1) @ observation
            p[:,1:depth] += ((desired-b)@lift).reshape(len(slots),depth-1,201)
            worst = max(worst,float((p[:,:depth].flatten(1)@observation-desired).abs().max()))
            reference = torch.as_tensor(ref_field[slots].copy(),device=device)/archive.release.rms
            b5 = torch.as_tensor(ref_receiver[slots].copy(),device=device)/archive.release.rms
            norms += torch.cat((energy(p,reference),energy(desired,b5)))
        if worst > 2.e-4:
            raise ValueError("The trained pressure lost receiver closure")
        records.append((record["parent_group_id"],norms.cpu()))
    metrics = {}
    for name,subset in (("all",records),("strict",[r for r in records if r[0]!=STRICT_PARENT])):
        sums = torch.stack([r[1] for r in subset]).sum(0)
        metrics[name] = dict(shots=len(subset),pressure_l2=float((sums[0]/sums[1]).sqrt()),
                             b5_l2=float((sums[2]/sums[3]).sqrt()))
    return metrics


def accepted(row):
    return all(max(row[k]["pressure_l2"],row[k]["b5_l2"])<TARGET for k in ("all","strict"))


def fit(args, archive, omega, binding):
    started = monotonic()
    device = torch.device(args.device)
    binding = {**binding,"phase_residual_gain":.045,"phase_residual_forcing":.025}
    shards, validation = [], []
    for shot in range(38):
        part = torch.load(args.cache/f"SHOT_{shot:02}.pt",weights_only=True,map_location="cpu",mmap=True)
        cache_binding = {k:v for k,v in binding.items() if not k.startswith("phase_residual_")}
        if part["binding"] != digest(cache_binding) or part["shot"] != shot:
            raise ValueError("The transport contribution uses incompatible source data")
        shards.append(part["train"])
        validation.extend(part["validation"])
    validation.sort(key=lambda sin:sin["index"])
    if [r["index"] for r in validation] != list(range(12)):
        raise ValueError("Need all 12 validation cases")
    train = {k:torch.cat([row[k].flatten(0,1) for row in shards]).to(device) for k in shards[0]}
    del shards
    omega = omega.to(device)
    rng = torch.Generator().manual_seed(3922)
    model = SpectralResidual().to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=.00025,weight_decay=.001)
    args.output.mkdir(exist_ok=True,parents=True)
    last = args.output/"LAST.pt"
    admit = args.output/"QUALIFIED.pt"
    begin, history = 0, []
    if args.long:
        sealed = torch.load(admit,map_location="cpu",weights_only=True)
        fitted = [row for row in sealed["history"] if row["epoch"]>0]
        if sealed["binding"] != binding or len(fitted) < 2 or not all(map(accepted,fitted[-2:])):
            raise ValueError("The long run requires two consecutive successful pilot epochs")
    if last.exists():
        state = torch.load(last,map_location="cpu",weights_only=True)
        if state["binding"] != binding:
            raise ValueError("Checkpoint is from another immutable training roster")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        rng.set_state(state["rng"])
        begin, history = state["epoch"],state["history"]
    else:
        initial = score(model,archive,validation,omega,device)
        history.append(dict(epoch=0,**initial))
        print(json.dumps(history[-1]),flush=True)
    length = len(train["base"])
    for epoch in range(begin+1,201 if args.long else 7):
        clock = monotonic()
        loss_sum = 0.
        model.train()
        selected = torch.randperm(length,generator=rng).split(1536)
        for ix in selected:
            ix = ix.to(device)
            base, y, f, delay = (train[key][ix] for key in ("base","label","feature","delay"))
            residual = model(base,f)-y
            raw = residual*torch.exp(-1j*delay[:,None]*omega)
            native = torch.fft.irfft(raw,n=FRAMES,norm="ortho")
            misfit = residual.abs().square().mean()+.6*native[:,200:401].square().mean()
            opt.zero_grad(set_to_none=True)
            misfit.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            loss_sum += float(misfit.detach())
        row = dict(epoch=epoch,loss=loss_sum/len(selected),epoch_seconds=monotonic()-clock,
                   training_pairs=length//POINTS)
        if not args.long or epoch % 5 == 0 or epoch == 200:
            model.eval()
            row.update(score(model,archive,validation,omega,device))
            history.append(row)
        state = dict(model=model.state_dict(),opt=opt.state_dict(),epoch=epoch,history=history,
                     binding=binding,rng=rng.get_state())
        atomic_checkpoint(last,state)
        qualified = not args.long and epoch >= 2 and all(map(accepted,history[-2:]))
        if qualified:
            atomic_checkpoint(admit,state)
        status = "COMPLETE" if epoch == 200 else "TRAINING" if args.long else (
            "QUALIFIED" if qualified else "PILOT" if epoch < 6 else "TARGET_NOT_MET")
        atomic_json(args.output/"STATUS.json",{**row,"status":status,"long":args.long,
                                                "seconds":monotonic()-started,"exact_marmousi_labels":0})
        if "all" in row:
            atomic_json(args.output/f"VAL_{epoch:03}.json",row)
        print(json.dumps(dict(status=status,**row)),flush=True)
        if qualified:
            break


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(73404)
    args.cache.mkdir(exist_ok=True,parents=True)
    with forbid_rollouts():
        forbid_exact_label_files(args.dataset)
        release = CommittedMarmousi(args.dataset)
        archive = PressureArchive(args.validation,release)
        omega = 2.*torch.pi*torch.fft.rfftfreq(FRAMES,.005)[:BINS]
        binding = dict(data=release.binding,validation=archive.binding,pair_index=sha_file(args.pairs/"TRAIN_INDEX.json"),
                       frequencies=BINS,side=SIDE,points=POINTS,source=sha_file(Path(__file__)),exact_marmousi_labels=0)
        if args.prepare:
            with torch.inference_mode():
                prepare(args,release,archive,omega,binding)
        else:
            fit(args,archive,omega,binding)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("dataset","pairs","validation","cache","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--long",action="store_true")
    parser.add_argument("--worker",type=int,default=0)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--device",default="cuda:0")
    run(parser.parse_args())
