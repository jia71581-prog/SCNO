"""Fit signed, medium-conditioned pressure transport from same-shot TRAIN parents."""

from argparse import ArgumentParser
from pathlib import Path
from time import monotonic
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
from marmousi_spectral_supervision import SpectralPairedPressure
from probe_marmousi_aligned_pressure_bank import collect, geometry
from scno_forward_source_propagator import spatial_coarse
from tianhe_observations import CommittedMarmousi
from tools.marmousi_target_dataset_refined.common import atomic_json, sha_file
from train_joint_pressure import PressureArchive
from train_no_pde_native import forbid_rollouts
from train_scno_paired_full_b128_hyper import atomic_checkpoint


POINTS = 51**2
SUPPORT = 24


def features(velocity, source):
    descriptor, _, delay = geometry(velocity, source)
    speed = spatial_coarse((velocity/2200.).log())[0]
    dz = F.pad(speed.diff(dim=0), (0, 0, 0, 1))
    average = speed.cumsum(0)/torch.arange(1, len(speed)+1, device=speed.device)[:, None]
    x = torch.linspace(0., 1., 51, device=speed.device)
    z, x = torch.meshgrid(x, x, indexing="ij")
    local = torch.stack((speed/.3, dz/.15, average/.3, delay/.5,
                         x-source[0]/2000., z), dim=-1)
    return descriptor, delay.flatten(), local.flatten(0, 1)


def support_rows(descriptors, query, exclude_self=False):
    distance = torch.cdist(descriptors, descriptors)
    n = len(distance)
    diag = torch.eye(n, device=distance.device, dtype=torch.bool)
    typical = distance.masked_fill(diag, float("inf")).amin(1).median().clamp_min(1.e-6)
    covariance = torch.exp(-.5*(distance/(2.5*typical)).square())
    precision = torch.linalg.inv(covariance+.005*diag)
    constant = precision.sum(1)
    if not exclude_self:
        desired = torch.exp(-.5*(torch.cdist(query,descriptors)/(2.5*typical)).square())
        initial = desired @ precision
        weights = initial + (1.-initial.sum(1,keepdim=True))*constant[None]/constant.sum()
    else:
        # The universal kriging weights for the model with its own row deleted.
        full = precision-constant[:, None]*constant[None]/constant.sum()
        weights = (-full/full.diag()[:, None]).masked_fill(diag, 0.)
    candidates = weights.abs().masked_fill(diag, -1.) if exclude_self else weights.abs()
    indices = candidates.topk(SUPPORT, dim=1).indices
    kept = weights.gather(1, indices)
    kept = kept+(1.-kept.sum(1, keepdim=True))*kept.abs()/kept.abs().sum(1,keepdim=True)
    return indices, kept.float()


class PressureTransport(nn.Module):
    def __init__(self, omega):
        super().__init__()
        self.register_buffer("omega", omega)
        self.compatibility = nn.Sequential(nn.Linear(13,32), nn.SiLU(),
                                           nn.Linear(32,32), nn.SiLU(), nn.Linear(32,3))
        self.gain = nn.Sequential(nn.Linear(6,32), nn.SiLU(), nn.Linear(32,6))
        spectral = torch.stack((omega/60., (omega/60.).square(), torch.ones_like(omega)))
        self.register_buffer("frequency", spectral)
        for layer in (self.compatibility[-1], self.gain[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, peer, query, weights, basis):
        query_all = query[:, None].expand_as(peer)
        prior = weights[:, :, None, None].expand(*peer.shape[:-1],1)
        parameters = self.compatibility(torch.cat((query_all,peer-query_all,prior),dim=-1))
        delta, dt, d2 = parameters.unbind(dim=-1)
        strength = prior[..., 0].abs()
        diverge = .25*delta.tanh()*strength
        matched = prior[..., 0]+diverge-diverge.sum(1,keepdim=True)*strength/strength.sum(1,keepdim=True)
        phase = (.025*dt.tanh()[...,None] + .007*d2.tanh()[...,None]*self.frequency[0]) * self.omega
        out = (basis*torch.exp(1j*phase)*matched[...,None]).sum(1)
        gain, shift = self.gain(query).unflatten(-1,(2,3)).tanh().unbind(dim=-2)
        gain_curve = 1.+.18*(gain @ self.frequency)
        phase_curve = .25*(shift @ self.frequency)
        return out*gain_curve*torch.exp(1j*phase_curve)


class ShotBank:
    def __init__(self, archive, pairs, shot, source, omega):
        coefficients, velocity = collect(archive,pairs,shot,"cpu")
        self.descriptor, self.delay, self.local = torch.vmap(lambda v: features(v, source))(velocity)
        self.coefficients = coefficients.flatten(2).transpose(1,2).contiguous()
        self.coefficients *= torch.exp(1j*self.delay[...,None]*omega)
        self.support, self.weights = support_rows(self.descriptor, self.descriptor, exclude_self=True)

    def inputs(self, indices, weights, query, points, device):
        nodes = points[:, None, :]
        peers = self.local[indices[...,None],nodes]
        basis = self.coefficients[indices[...,None],nodes]
        return tuple(t.to(device) for t in (peers, query, weights, basis))


def restore(spectrum, delay, omega):
    value = spectrum*torch.exp(-1j*delay[...,None]*omega)
    return torch.cat((value[..., :1], value[..., 1:]/2.**.5), dim=-1)


@torch.inference_mode()
def evaluate(model, archive, validation, banks, device):
    model.eval()
    groups = []
    for i,(case,_,_) in enumerate(archive.items["validation"]):
        shot = case["shot_id"]
        ref, b5, v = archive.arrays("validation",i)
        bank = banks[shot]
        d, delay, local = validation[i]
        indices, weights = support_rows(bank.descriptor,d[None])
        packed = []
        for points in torch.arange(POINTS).split(160):
            query = local[None, points]
            value = model(*bank.inputs(indices, weights, query, points[None], device))[0]
            packed.append(value.cpu())
        spectrum = restore(torch.cat(packed), delay, model.omega.cpu())
        pressure = torch.fft.irfft(spectrum.T.reshape(-1,51,51),n=801,dim=0,norm="ortho")
        release = archive.release
        observation = release.observe
        depth_rows = observation.support_depth_rows
        matrix = observation.matrix.float().cpu()
        stats = torch.zeros(4, dtype=torch.float64)
        for slots in np.array_split(np.arange(801),26):
            p = F.interpolate(pressure[slots,None],size=(201,201),mode="bilinear",align_corners=True)[:,0]
            p[:,0].zero_()
            p[:np.searchsorted(slots, 3)].zero_()
            receiver = p[:,:depth_rows].flatten(1) @ matrix
            y = torch.from_numpy(ref[slots].copy())/release.rms
            b = torch.from_numpy(b5[slots].copy())/release.rms
            stats += torch.stack(((p-y).square().sum(dtype=torch.float64), y.square().sum(dtype=torch.float64),
                                  (receiver-b).square().sum(dtype=torch.float64), b.square().sum(dtype=torch.float64)))
        groups.append(stats)
    summed = torch.stack(groups).sum(0)
    return dict(pressure_l2=float((summed[0]/summed[1]).sqrt()),b5_l2=float((summed[2]/summed[3]).sqrt()))


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(18079)
    rng = torch.Generator().manual_seed(18079)
    start = monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    with forbid_rollouts():
        forbid_exact_label_files(args.dataset)
        release = CommittedMarmousi(args.dataset)
        archive = PressureArchive(args.validation,release)
        pairs = SpectralPairedPressure(args.pairs,release,args.cache,read_threads=1)
        sources = torch.as_tensor(release.sources,dtype=torch.float32)
        omega = 2.*torch.pi*torch.fft.rfftfreq(801,.005)[:pairs.frequencies]
        banks = {}
        with torch.no_grad():
            for shot, source in enumerate(sources):
                banks[shot] = ShotBank(archive, pairs, shot, source, omega)
                print(json.dumps(dict(stage="TRAIN_BANK", shot=shot, seconds=monotonic()-start)),flush=True)
        validation = [features(torch.tensor(np.array(archive.arrays("validation",i)[2])), sources[case["shot_id"]])
                      for i,(case,_,_) in enumerate(archive.items["validation"])]
        model = PressureTransport(omega).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        binding = dict(archive=archive.binding, spectrum=pairs.receipt["plan"], support=SUPPORT,
                       train_bank=sum(len(b.descriptor) for b in banks.values()),
                       source_sha256=sha_file(Path(__file__)), exact_marmousi_labels=0)
        history, first = [], 0
        last = args.output/"LAST.pt"
        if last.exists():
            saved = torch.load(last, weights_only=True, map_location="cpu")
            if saved["binding"] != binding:
                raise ValueError("The resumable run belongs to another set of ordinary parents")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            rng.set_state(saved["rng"])
            history, first = saved["history"], saved["epoch"]
        else:
            first_eval = evaluate(model,archive,validation,banks,args.device)
            history.append(dict(epoch=0, **first_eval))
            print(json.dumps(history[-1]),flush=True)
        for epoch in range(first+1,args.epochs+1):
            losses = []
            model.train()
            for shot in torch.randperm(len(sources),generator=rng).tolist():
                bank = banks[shot]
                order = torch.randperm(len(bank.descriptor),generator=rng)
                for ids in order.split(16):
                    points = torch.randint(POINTS,(len(ids),96),generator=rng)
                    local = bank.local[ids[:,None],points]
                    target = bank.coefficients[ids[:,None],points].to(args.device)
                    delay = bank.delay[ids[:,None],points].to(args.device)
                    inputs = bank.inputs(bank.support[ids],bank.weights[ids],local,points,args.device)
                    error = model(*inputs)-target
                    native = torch.fft.irfft(restore(error,delay,model.omega),n=801,norm="ortho")
                    energy = target.abs().square().mean().clamp_min(.02)
                    loss = (.7*error.abs().square().mean()+.3*4.*native[...,200:401].square().mean())/energy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            score = evaluate(model,archive,validation,banks,args.device)
            row = dict(epoch=epoch,loss=float(np.mean(losses)),seconds=monotonic()-start,**score)
            history.append(row)
            accepted = len(history) > 2 and all(max(x["pressure_l2"],x["b5_l2"])<.15 for x in history[-2:])
            state = dict(binding=binding,epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                         history=history,rng=rng.get_state())
            atomic_checkpoint(last,state)
            if accepted:
                atomic_checkpoint(args.output/"QUALIFIED.pt",state)
            atomic_json(args.output/"RESULT.json",dict(binding=binding, accepted=accepted,history=history))
            print(json.dumps(row),flush=True)
            if accepted and args.epochs < 200:
                break


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("dataset","pairs","cache","validation","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--epochs",type=int,default=6)
    parser.add_argument("--device",default="cuda:0")
    config = parser.parse_args()
    if config.epochs >= 200 and not (config.output/"QUALIFIED.pt").is_file():
        parser.error("The 200-epoch run needs a qualified ordinary validation checkpoint")
    run(config)
