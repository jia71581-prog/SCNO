"""Dense joint Marmousi pretraining on the complete 800-medium, 3,200-shot release."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from time import monotonic
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
from score_marmousi_broadband_joint import matrix_lift
from tianhe_observations import CommittedMarmousi
from tools.marmousi_target_dataset_refined.common import atomic_json, digest, read_json, sha_file
from train_joint_pressure import PressureArchive
from train_marmousi_transport_residual import BINS, FRAMES, SIDE, STRICT_PARENT, SpectralResidual
from train_no_pde_native import forbid_rollouts
from train_scno_paired_full_b128_hyper import atomic_checkpoint


ENERGY_KEYS = ("pressure_base","pressure_label","b5_base","b5_label","feature","delay")
PAIRS = 3200
EPOCHS = 200


class LocalPropagation(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.spatial = nn.Conv2d(width,width,3,padding=dilation,dilation=dilation,groups=width)
        self.mix = nn.Conv2d(width,width,1)
        self.norm = nn.GroupNorm(4,width)

    def forward(self, value):
        return value+.3*self.mix(F.silu(self.norm(self.spatial(value))))


def zero_layer(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


def as_complex(value):
    return torch.view_as_complex(value.float().contiguous())


class FullJointOperator(nn.Module):
    def __init__(self, seed):
        super().__init__()
        width = 32
        self.medium = nn.Sequential(nn.Conv2d(14,width,3,padding=1),nn.GroupNorm(4,width),nn.SiLU())
        self.transport = nn.Sequential(*(LocalPropagation(width,d) for d in (1,2,4,8)))
        self.phase = nn.Conv2d(width,4,1)
        self.medium_readout = nn.Conv2d(width,6,1)
        self.pressure_readout = SpectralResidual()
        self.pressure_readout.load_state_dict(seed["model"],strict=True)
        self.receiver_readout = nn.Sequential(nn.Conv1d(width+8,width,5,padding=2),nn.SiLU(),
                                               nn.Conv1d(width,4*BINS,1))
        self.register_buffer("omega",2.*torch.pi*torch.fft.rfftfreq(FRAMES,.005)[:BINS])
        self.register_buffer("startup",(torch.arange(FRAMES)>=3)[None,:,None])
        for layer in (self.phase,self.medium_readout,self.receiver_readout[-1]):
            zero_layer(layer)

    @staticmethod
    def bands(value, axis):
        scale = value.abs().square().mean(axis,keepdim=True).clamp_min(1.e-7).sqrt()
        chunks = (value/scale).tensor_split(4,dim=axis)
        return torch.cat([v for band in chunks for v in (band.real.mean(axis,keepdim=True),
                                                         band.imag.mean(axis,keepdim=True))],dim=axis)

    def pressure(self, base, feature):
        hidden = self.transport(self.medium(torch.cat((feature,self.bands(base,1)),dim=1)))
        amplitude, amplitude_slope, time, dispersion = self.phase(hidden).float().tanh().unbind(1)
        ramp = self.omega[None,:,None,None]/90.
        gain = 1.+.055*amplitude[:,None]+.025*ramp*amplitude_slope[:,None]
        travel = self.omega[None,:,None,None]*(.0025*time[:,None]+.0008*ramp*dispersion[:,None])
        value = (base*gain*torch.exp(1j*travel)).permute(0,2,3,1)
        features = (feature+.15*self.medium_readout(hidden).float().tanh()).permute(0,2,3,1)
        scale = value.abs().square().mean(-1,keepdim=True).clamp_min(1.e-7).sqrt()
        normalized = torch.view_as_real(value/scale).flatten(-2)
        code = self.pressure_readout.network(torch.cat((normalized,features),dim=-1)).float().tanh()
        gate,forcing = as_complex(code.unflatten(-1,(2,BINS,2))).unbind(-2)
        result = value*(1.+.045*gate)+.025*scale*forcing
        return result.permute(0,3,1,2),hidden

    def receiver(self, base, hidden):
        spectrum = torch.fft.rfft(base,dim=1,norm="ortho")[:,:BINS]
        surface = F.interpolate(hidden[:,:,:4].mean(2),size=201,mode="linear",align_corners=True)
        code = self.receiver_readout(torch.cat((surface,self.bands(spectrum,1)),dim=1)).float().tanh()
        ordered = code.unflatten(1,(2,BINS,2)).permute(0,1,2,4,3)
        amplitude,extra = as_complex(ordered).unbind(1)
        rms = spectrum.abs().square().mean(1,keepdim=True).clamp_min(1.e-7).sqrt()
        correction = .10*spectrum*amplitude+.06*rms*extra
        value = base+torch.fft.irfft(correction,n=FRAMES,dim=1,norm="ortho")
        return value*self.startup

    def forward(self, carrier, receiver, feature):
        pressure, context = self.pressure(carrier,feature)
        return pressure,self.receiver(receiver,context)


def cache_contract(args, release):
    stamp = read_json(args.cache/"PLAN.json")
    if (stamp["release"]!=release.binding or stamp["exact_marmousi_labels"]
            or stamp["bins"]!=BINS or stamp["frames"]!=FRAMES):
        raise ValueError("Use the same ordinary data contract for training and evaluation")
    return stamp,digest(stamp)


def load_full_data(args, stamp, device):
    arrays, names = {}, []
    total = 0
    for shot in range(38):
        path = args.cache/f"SHOT_{shot:02}.pt"
        row = torch.load(path,weights_only=True,map_location="cpu",mmap=True)
        if row["plan"]!=stamp or row["shot"]!=shot:
            raise ValueError("The full training data shard has a different physics contract")
        group = row["rows"]
        end = total+len(group)
        if end>PAIRS or len(set(group))!=len(group):
            raise ValueError("Duplicate model/shot pair in the full training tensor")
        for key in ENERGY_KEYS:
            value = row[key]
            if key not in arrays:
                arrays[key] = torch.empty((PAIRS,*value.shape[1:]),device=device,dtype=value.dtype)
            arrays[key][total:end].copy_(value)
        total = end
        names.extend(group)
    models = Counter(names)
    if total!=PAIRS or len(models)!=800 or set(models.values())!={4}:
        raise ValueError("The epoch must use all 3,200 shots, four each for all 800 TRAIN media")
    return arrays


def validation_carriers(args, seed):
    records = []
    contract = {k:v for k,v in seed["binding"].items() if not k.startswith("phase_residual_")}
    identity = digest(contract)
    for shot in range(38):
        row = torch.load(args.validation_cache/f"SHOT_{shot:02}.pt",weights_only=True,map_location="cpu",mmap=True)
        if row["shot"]!=shot or row["binding"]!=identity:
            raise ValueError("The held-out carrier bank does not match the source checkpoint")
        records.extend(row["validation"])
    records.sort(key=lambda row:row["index"])
    if [r["index"] for r in records]!=list(range(12)) or seed["binding"]["exact_marmousi_labels"]:
        raise ValueError("The fixed 12-shot holdout and previously qualified source must match")
    return records


def relative(delta, reference, dims):
    error = delta.square().sum(dims)
    norm = reference.square().sum(dims).clamp_min(1.e-7)
    return (error/norm).mean()


def objective(model, data, index):
    base, target, receiver, b5, feature, delay = (data[key][index] for key in ENERGY_KEYS)
    base = as_complex(base)
    # Labels are stored in physical time. Compare in the carrier's arrival-aligned basis.
    arrival_phase = model.omega[None,:,None,None]*delay[:,None]
    target = as_complex(target)*torch.exp(1j*arrival_phase)
    pressure,predicted = model(base,receiver.float(),feature.float())
    one_sided = 1.+(model.omega>0).float()[None,:,None,None,None]
    # Parseval energy after projecting the full native label over space and time.
    difference = torch.view_as_real(pressure-target)
    truth = torch.view_as_real(target)
    dims = (1,2,3,4)
    absolute_p = (difference.square()*one_sided).sum(dims)
    energy_p = (truth.square()*one_sided).sum(dims).clamp_min(1.e-7)
    p = (absolute_p/energy_p).mean()
    b = relative(predicted-b5.float(),b5.float(),(1,2))
    window = slice(100,401)
    late = relative(predicted[:,window]-b5[:,window].float(),b5[:,window].float(),(1,2))
    return .60*p+.30*b+.10*late,torch.stack((p,b,late)).detach()


def native_block(wave, receiver, slots, depth, observation, lift):
    value = wave[slots,None]
    linear = F.interpolate(value,size=(201,201),mode="bilinear",align_corners=True)[:,0]
    cubic = F.interpolate(value,size=(201,201),mode="bicubic",align_corners=True)[:,0]
    p = .5*(linear+cubic)
    p[:,0].zero_()
    p[:np.searchsorted(slots,3)].zero_()
    observed = p[:,:depth].flatten(1) @ observation
    p[:,1:depth] += ((receiver[slots]-observed)@lift).reshape(len(slots),depth-1,201)
    return p


@torch.inference_mode()
def validate(model, release, archive, records, device):
    depth, observe_cpu, lift_cpu = matrix_lift(release)
    observation, lift = observe_cpu.to(device), lift_cpu.to(device)
    all_energy = torch.zeros(4,device=device,dtype=torch.float64)
    clean_energy = torch.zeros_like(all_energy)
    model.eval()
    closure = 0.
    for row in records:
        index = row["index"]
        case = archive.items["validation"][index][0]
        if case["model_id"]!=row["key"]:
            raise ValueError("The validation carrier belongs to a different medium")
        feature = row["feature"].T.reshape(1,6,SIDE,SIDE).to(device)
        aligned = row["base"].T.reshape(1,BINS,SIDE,SIDE).to(device)
        base = row["receiver"][None].to(device)
        p,b = model(aligned,base,feature)
        dt = row["delay"].reshape(1,SIDE,SIDE).to(device)
        spectrum = p[0]*torch.exp(-1j*model.omega[:,None,None]*dt)
        wave = torch.fft.irfft(spectrum,n=FRAMES,dim=0,norm="ortho")
        ref,ref_b5,_ = archive.arrays("validation",index)
        energy = torch.zeros_like(all_energy)
        for slots in np.array_split(np.arange(FRAMES),26):
            pressure = native_block(wave,b[0],slots,depth,observation,lift)
            y = torch.from_numpy(ref[slots].copy()).to(device)/release.rms
            yb = torch.from_numpy(ref_b5[slots].copy()).to(device)/release.rms
            residual = (pressure[:,:depth].flatten(1)@observation-b[0,slots]).abs().amax()
            closure = max(closure,float(residual))
            values = (pressure-y,y,b[0,slots]-yb,yb)
            energy += torch.stack([v.square().sum(dtype=torch.float64) for v in values])
        all_energy += energy
        if case["parent_group_id"]!=STRICT_PARENT:
            clean_energy += energy
    if closure>2.e-4:
        raise ValueError(f"The trained pressure failed exact B5 projection: {closure}")
    def score(energy):
        return dict(pressure_l2=float((energy[0]/energy[1]).sqrt()),
                    b5_l2=float((energy[2]/energy[3]).sqrt()))
    return dict(validation=score(all_energy),strict_validation=score(clean_energy),closure_max_abs=closure)


def lr_multiplier(epoch):
    warm = min(1.,epoch/3.)
    return warm*(.18+.82*(1.+math.cos(math.pi*(epoch-1)/(EPOCHS-1)))/2.)


def fit(args, model, archive, records, data, binding, device):
    rng = torch.Generator().manual_seed(20261122)
    optimizer = torch.optim.AdamW([
        dict(params=model.medium.parameters(),lr=.00013),
        dict(params=model.transport.parameters(),lr=.00013),
        dict(params=model.phase.parameters(),lr=.00013),
        dict(params=model.medium_readout.parameters(),lr=.00013),
        dict(params=model.pressure_readout.parameters(),lr=.000045),
        dict(params=model.receiver_readout.parameters(),lr=.00024),
    ],weight_decay=.001)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    first, steps, microbatch, best = 1,0,args.microbatch,float("inf")
    args.output.mkdir(exist_ok=True,parents=True)
    last = args.output/"LAST.pt"
    if last.exists():
        saved = torch.load(last,weights_only=True,map_location="cpu")
        if saved["binding"]!=binding:
            raise ValueError("The full training checkpoint has a different immutable contract")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"])
        first, steps, microbatch, best = saved["epoch"]+1,saved["steps"],saved["microbatch"],saved["best"]
    else:
        initial = validate(model,archive.release,archive,records,device)
        atomic_json(args.output/"VAL_000.json",initial)
        print(json.dumps(dict(stage="INITIAL_VALIDATION",**initial)),flush=True)
    started = monotonic()
    for epoch in range(first,EPOCHS+1):
        start_epoch = monotonic()
        accum = torch.zeros(3,device=device,dtype=torch.float64)
        model.train()
        for group,rate in zip(optimizer.param_groups,base_lrs,strict=True):
            group["lr"] = rate*lr_multiplier(epoch)
        order = torch.randperm(PAIRS,generator=rng).to(device)
        for index in order.split(args.batch):
            while True:
                optimizer.zero_grad(set_to_none=True)
                try:
                    components = torch.zeros_like(accum)
                    for chunk in index.split(microbatch):
                        with torch.autocast("cuda",dtype=torch.bfloat16):
                            loss,terms = objective(model,data,chunk)
                        (loss*(len(chunk)/len(index))).backward()
                        components += terms*len(chunk)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    break
                except torch.cuda.OutOfMemoryError:
                    if microbatch==1:
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    microbatch = max(1,microbatch//2)
                    torch.cuda.empty_cache()
                    print(json.dumps(dict(stage="MICROBATCH_RETRY",microbatch=microbatch)),flush=True)
            optimizer.step()
            steps += 1
            accum += components
            if steps<=2 or steps%10==0:
                print(json.dumps(dict(stage="TRAINING",epoch=epoch,steps=steps,microbatch=microbatch,
                                      effective_batch=len(index),batch_terms=(components/len(index)).tolist(),
                                      gpu_peak_reserved_mib=round(torch.cuda.max_memory_reserved()/2**20))),flush=True)
        metrics = dict(epoch=epoch,epochs=EPOCHS,steps=steps,train_pairs_this_epoch=PAIRS,
                       training_media=800,microbatch=microbatch,effective_batch=args.batch,
                       loss_pressure_b5_late=(accum/PAIRS).tolist(),epoch_seconds=monotonic()-start_epoch,
                       total_seconds=monotonic()-started,status="COMPLETE" if epoch==EPOCHS else "TRAINING")
        is_best = False
        if epoch<=2 or epoch%5==0 or epoch==EPOCHS:
            metrics.update(validate(model,archive.release,archive,records,device))
            atomic_json(args.output/f"VAL_{epoch:03}.json",metrics)
            tradeoff = max(metrics["strict_validation"].values())
            is_best = tradeoff<best
            if is_best:
                best = tradeoff
        state = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                     steps=steps,microbatch=microbatch,binding=binding,rng=rng.get_state(),best=best)
        atomic_checkpoint(last,state)
        if is_best:
            atomic_checkpoint(args.output/"BEST.pt",state)
        atomic_json(args.output/"STATUS.json",metrics)
        print(json.dumps(metrics),flush=True)


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(92704)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with forbid_rollouts():
        forbid_exact_label_files(args.dataset)
        release = CommittedMarmousi(args.dataset)
        archive = PressureArchive(args.validation,release)
        _,dataid = cache_contract(args,release)
        seed = torch.load(args.seed,weights_only=True,map_location="cpu")
        if seed["epoch"]!=200 or seed["binding"]["data"]!=release.binding or seed["binding"]["validation"]!=archive.binding:
            raise ValueError("Start from the completed, ordinary-validated broadband prior")
        device = torch.device("cuda:0")
        model = FullJointOperator(seed).to(device)
        records = validation_carriers(args,seed)
        args.output.mkdir(exist_ok=True,parents=True)
        binding = dict(schema="marmousi_full_joint_pretraining_v1",data=dataid,validation=archive.binding,
                       seed_sha256=sha_file(args.seed),source_sha256=sha_file(Path(__file__)),
                       all_pressures=True,full_native_b5=True,training_pairs=PAIRS,epochs=EPOCHS,
                       effective_batch=args.batch,trainable_parameters=sum(p.numel() for p in model.parameters()),
                       exact_marmousi_labels=0)
        if (args.output/"PLAN.json").exists():
            if read_json(args.output/"PLAN.json")!=binding:
                raise ValueError("The requested run would change the frozen training plan")
        else:
            atomic_json(args.output/"PLAN.json",binding)
        data = load_full_data(args,dataid,device)
        print(json.dumps(dict(stage="FULL_TRAIN_DATA_READY",pairs=len(data[ENERGY_KEYS[0]]),
                              memory_mib=round(torch.cuda.memory_allocated()/2**20))),flush=True)
        fit(args,model,archive,records,data,binding,device)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("dataset","cache","output","validation","validation-cache","seed"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--batch",type=int,default=128)
    parser.add_argument("--microbatch",type=int,default=128)
    parser.add_argument("--threads",type=int,default=8)
    run(parser.parse_args())
