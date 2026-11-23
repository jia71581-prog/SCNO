"""Use all native time samples for phase-coherent pressure supervision."""

from pathlib import Path

import numpy as np
import torch

from cache_marmousi_full_spectrum import FREQUENCIES, SHARD_SIZE
from tools.marmousi_target_dataset_refined.common import digest
from train_scno_paired_full_b128_hyper import PairedPressure, read, relative_squared


class SpectralPairedPressure(PairedPressure):
    def __init__(self, root, release, cache, read_threads):
        super().__init__(root,release,read_threads)
        self.cache = Path(cache)
        self.plan = read(self.cache/"PLAN.json")
        self.receipt = read(self.cache/"STATUS.json")
        self.frequencies = self.plan["frequencies"]
        if (self.receipt["status"] != "COMPLETE" or self.receipt["plan"] != digest(self.plan)
                or self.plan["data"] != self.binding or self.receipt["exact_marmousi_labels"]
                or self.plan["rows"] != [[m,s] for m,s,_ in self.rows]
                or self.plan["frames"] != 801 or not FREQUENCIES <= self.frequencies <= 401
                or self.plan["coarse_spacing_m"] != 40 or self.plan["shard_size"] != SHARD_SIZE):
            raise ValueError("Full-time pressure must bind to the same committed ordinary TRAIN pairs")
        self.shards = {}

    def __getitem__(self,index):
        model,shot,folder = self.rows[index]
        shard, slot = divmod(index,SHARD_SIZE)
        if shard not in self.shards:
            entry = torch.load(self.cache/f"SHARD_{shard:03}.pt",map_location="cpu",weights_only=True,mmap=True)
            if entry["plan"] != self.receipt["plan"] or entry["rows"] != self.plan["rows"][shard*SHARD_SIZE:(shard+1)*SHARD_SIZE]:
                raise ValueError("A full-time pressure shard changed")
            self.shards[shard] = entry
        entry = self.shards[shard]
        receiver = [np.load(folder/f"{name}.npy",allow_pickle=False)/self.rms for name in ("b5","background_b5")]
        if any(r.shape != (801,201) or r.dtype != np.float32 for r in receiver):
            raise ValueError(f"Invalid signed receiver pair: {model}/{shot}")
        return (self.velocities[model],self.sources[shot],entry["pressure"][slot],
                entry["background_pressure"][slot], *receiver)


def spectral_pressure(value, frequencies=FREQUENCIES):
    spectrum = torch.fft.rfft(value,dim=0,norm="ortho")[:frequencies]
    weighted = torch.cat((spectrum[:1],spectrum[1:]*2.**.5),dim=0)
    return torch.view_as_real(weighted)


def predict(model, velocity, source, frequencies=FREQUENCIES):
    background, correction = model.propagate(model.prepare(velocity,source))
    extended = torch.cat((torch.zeros_like(background[:,:1]),background,background[:,-1:]),dim=1)
    geometry = model.geometry
    lifted = torch.lerp(extended[:,geometry.lower[::4]],extended[:,geometry.upper[::4]],
                        geometry.blend[None,::4,None].to(background))
    p0 = spectral_pressure(lifted, frequencies)
    p = p0 + spectral_pressure(correction, frequencies)
    b0 = model.background_b5(background)
    b = b0 + model.b5((torch.zeros_like(background),correction))
    return p,b,p0,b0


def spectral_objective(model, batch, slots, background_only, weights):
    if background_only:
        raise ValueError("The coherent stage trains pressure and scattering jointly")
    v, source, pressure, background, receiver, background_receiver = batch
    p,b,p0,b0 = torch.vmap(lambda vx,s: predict(model,vx,s,pressure.shape[1]))(v,source)
    quiet_p=.012*pressure.square().sum((1,2,3,4))
    quiet_b=.012*receiver.square().sum((1,2))
    background_fit=.65*relative_squared(p0,background)+.35*relative_squared(b0,background_receiver)
    components = (
        relative_squared(p,pressure),relative_squared(b,receiver),
        relative_squared(p[:,:,13:],pressure[:,:,13:],energy_floor=quiet_p),
        relative_squared(b[:,100:400],receiver[:,100:400],energy_floor=quiet_b),
        background_fit,relative_squared(p-p0,pressure-background,energy_floor=quiet_p),
        relative_squared(b-b0,receiver-background_receiver,energy_floor=quiet_b),
    )
    return sum(w*x for w,x in zip(weights, components, strict=True))
