"""Continue the native E12 pressure operator with two matched medium-routing arms."""

from argparse import ArgumentParser
from collections import defaultdict
from functools import lru_cache
from hashlib import file_digest
from pathlib import Path
from time import monotonic
import json

import numpy as np
import torch

from scno_mhc_medium import attach_mhc
from tools.marmousi_target_dataset_refined.common import atomic_json


FRAMES = 801
BANDS = (4., 6., 9.)
WINDOWS = ((40,100), (100,200), (200,400), (400,800))
SEED = 381622


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def relative(pred, ref):
    return (pred-ref).norm()/ref.norm().clamp_min(1.e-12)


def receiver(model, speed, shots, block):
    context = model.encode(speed, model.sources[shots])
    return torch.cat([model.decode_b5(context, part) for part in torch.arange(FRAMES, device=model.device).split(block)], 1)


def relative_terms(value, reference, dims, additional_floor=0.):
    energy = reference.square().sum(dims)
    # A quiet window carries little information. Keep late-window weights
    # bounded by a fraction of the energetic windows of the same TRAIN shot.
    floor = torch.maximum(.02*energy.amax(), torch.as_tensor(additional_floor, device=value.device, dtype=value.dtype))
    return ((value-reference).square().sum(dims)/torch.maximum(energy,floor).clamp_min(1.e-9)).mean()


def pressure_frames(step, fixed=False):
    if fixed:
        return [i for a,b in WINDOWS for i in np.linspace(a, b-1, 8, dtype=int).tolist()]
    rng = np.random.default_rng([SEED,step])
    return sorted([int(x) for a,b in WINDOWS for x in rng.choice(np.arange(a,b), 3, replace=False)])


def wave(model, archive, split, index, frames):
    p, trace, v = archive.arrays(split, index)
    source = archive.items[split][index][0]["shot_id"]
    context = model.encode(torch.from_numpy(v).to(model.device), model.sources[[source]])
    slots = torch.as_tensor(frames, device=model.device)
    pred = model.decode_pressure(context, slots)[0]
    gather = model.decode_b5(context, slots)[0]
    def target(array):
        return torch.as_tensor(array[frames].copy(), device=model.device)/archive.release.rms
    return pred, target(p), gather, target(trace)


def wave_loss(model, archive, index, step):
    frames = pressure_frames(step)
    p, ref, b, obs = wave(model, archive, "train", index, frames)
    # Each arrival window and each depth zone get a separate relative budget.
    pressure = p.reshape(len(WINDOWS), 3, *p.shape[-2:])
    truth = ref.reshape_as(pressure)
    whole = relative_terms(pressure, truth, (1,2,3))
    shot_energy = truth.square().sum((1,2,3)).amax()
    deep = relative_terms(pressure[:,:,60:], truth[:,:,60:], (1,2,3), .005*shot_energy)
    b5 = relative_terms(b.reshape(len(WINDOWS), 3, -1), obs.reshape(len(WINDOWS),3,-1),(1,2))
    return .45*whole + .25*deep + .30*b5


def spectrum(value):
    return torch.fft.rfft(value.double(), dim=-2, norm="ortho")


def response_bands(device):
    grid = torch.fft.rfftfreq(FRAMES, .005, device=device)
    return [grid <= k for k in BANDS]


def predict_pair(model, sample, block):
    return torch.stack([receiver(model, v, [sample["shot"]], block)[0] for v in sample["velocity"]])


def secant_loss(pred, sample, bands):
    secant = (pred[1].double()-pred[0].double())/(2.*sample["h"])
    reference = sample["direction"]
    f, g = (spectrum(v) for v in (secant, reference))
    frequency_loss = torch.stack([relative(f[b],g[b]).square() for b in bands]).mean()
    fine = relative(secant[100:400], reference[100:400]).square()
    # This label is a directional projection of an adjoint, <w,J d>.
    # The common residual is fixed at the archived checkpoint for both arms.
    projections = [((sample["cotangent"][b].conj()*(f[b]-g[b])).sum().real /
                    (sample["cotangent"][b].norm()*g[b].norm()).clamp_min(1.e-12)).square() for b in bands]
    return .65*frequency_loss + .25*fine + .10*torch.stack(projections).mean()


class Samples:
    def __init__(self, args, data, model):
        from train_sensitivity import pair_bank

        self.data, self.device = data, model.device
        self.training, self.validation, self.bindings = [], [], {}
        for split, roster in (("train", self.training), ("validation",self.validation)):
            for key in ("bank_a", "bank_f"):
                pairs, stamp = pair_bank(data, getattr(args,key), split)
                roster.extend((row, path) for row,path in pairs)
                self.bindings[f"{key}_{split}"] = stamp
        self.pilot = args.structured.resolve(strict=True)
        report = read(self.pilot/"PILOT.json")
        if report["status"] != "COMPLETE" or report["pair_count"] != 3 or not all(
                row["label_qc_passed"] for row in report["checks"]):
            raise ValueError("Require the three verified small-step structured TRAIN pairs")
        self.structured = {row["plan"]["id"]:row for row in report["checks"]}
        self.bindings["structured"] = sha(self.pilot/"PILOT.json")
        overlap = {r["parent_group_id"] for r,_ in self.training} & {r["parent_group_id"] for r,_ in self.validation}
        if overlap:
            raise ValueError("Direction TRAIN must remain separate by velocity parent")
        self.records = {r["pair_id"]:(r,p) for r,p in self.training+self.validation}

    @lru_cache(maxsize=70)
    def load(self, label, shot_slot):
        from make_direction_labels import logit, velocity_from_logit
        from train_sensitivity import read_pair

        if label in self.structured:
            item = self.structured[label]
            plan = item["plan"]
            row = self.data.rows[plan["center"]]
            shard = self.pilot/(label+".npz")
            if (row["split"] != "train" or row["is_exact_initial"] or row["is_exact_target"]
                    or row["parent_group_id"] != plan["parent"] or sha(shard) != item["shard_sha256"]):
                raise ValueError("Structured direction lost its ordinary TRAIN binding")
            with np.load(shard, allow_pickle=False) as entry:
                v, d = entry["center_velocity_mps"], entry["direction_latent"]
                raw, ids, h = entry["half"], entry["shot_ids"], float(entry["h"])/2.
            if ids.tolist() != plan["shot_ids"] or not (raw.shape == (2,len(ids),801,201)):
                raise ValueError("Structured endpoints or source roster changed")
            bounds = (self.data.contract["parameterization"]["lower_m_s"],
                      self.data.contract["parameterization"]["upper_m_s"])
            center = logit(v,bounds)
            v = np.stack([velocity_from_logit(center+sign*h*d,bounds) for sign in (-1.,1.)]).astype(np.float32)
            endpoints = raw[:,shot_slot].astype(np.float64)/self.data.rms
        else:
            record, path = self.records[label]
            minus, plus, ids, labels, _, h = read_pair(record, path, self.data)
            v, endpoints = np.stack([minus,plus]), labels[:,shot_slot].astype(np.float64)
        as_tensor = lambda data: torch.as_tensor(data, device=self.device)
        goal = as_tensor(endpoints)
        return dict(velocity=as_tensor(v), shot=int(ids[shot_slot]), goal=goal,
                    direction=(goal[1]-goal[0])/(2.*h), h=h)

    def schedules(self, steps):
        groups = defaultdict(list)
        for row,_ in self.training:
            groups[row["direction_type"]].append(row["pair_id"])
        rng = np.random.default_rng(SEED)
        for group in groups:
            rng.shuffle(groups[group])
        structured = list(self.structured)
        strata = [*structured, "deep_tapered", "smooth_multiscale", "archived_family_F_multiscale_gaussian"]
        train = []
        for i in range(steps):
            g = strata[i%len(strata)]
            labels = groups.get(g,[g])
            train.append((labels[(i//len(strata))%len(labels)], (i//2)%4))
        val = [(row["pair_id"],j) for i,(row,_) in enumerate(self.validation)
               if row["direction_type"] == "deep_tapered" for j in (0,2)]
        varied = [row for row,_ in self.validation if row["direction_type"] != "deep_tapered"]
        val += [(row["pair_id"],1) for row in (varied[0],varied[-1])]
        return train, val


@torch.no_grad()
def fixed_train_targets(model, samples, schedule, block):
    result = {}
    for key in sorted(set(schedule)):
        datum = dict(samples.load(*key))
        frozen = predict_pair(model, datum, block)
        residual = frozen.double().mean(0)-datum["goal"].mean(0)
        # Preserve the 0.5–2 s separation used in the FWI diagnostic.
        window = torch.zeros_like(residual)
        window[100:400] = residual[100:400]
        datum["cotangent"] = spectrum(window)
        result[key] = datum
        print(json.dumps(dict(stage="FROZEN_COTANGENTS", pairs_ready=len(result), pairs_total=len(set(schedule)))), flush=True)
    return result


@torch.no_grad()
def score(model, archive, samples, pairs, block):
    device = model.device
    totals = np.zeros((3,2))
    frames = pressure_frames(0, fixed=True)
    frame_groups = [("pressure", slice(None)), ("deep_pressure", slice(60,None))]
    for i in range(len(archive.items["validation"])):
        p, ref, b, obs = wave(model, archive, "validation", i, frames)
        for k,(_,sl) in enumerate(frame_groups):
            totals[k] += np.array([(p[:,sl]-ref[:,sl]).double().square().sum().item(),ref[:,sl].double().square().sum().item()])
        totals[2] += np.array([(b-obs).double().square().sum().item(),obs.double().square().sum().item()])
    bands = response_bands(device)
    direction = defaultdict(lambda: np.zeros((len(bands),4)))
    for key in pairs:
        label, _ = key
        sample = samples.load(*key)
        trial = predict_pair(model, sample, block)
        value, truth = (spectrum(x) for x in ((trial[1].double()-trial[0].double())/(2.*sample["h"]),sample["direction"]))
        group = samples.records[label][0]["direction_type"]
        for i,b in enumerate(bands):
            a,g = value[b],truth[b]
            terms = np.array([(a-g).norm().square().item(),a.norm().square().item(),g.norm().square().item(),
                              (a*g.conj()).sum().real.item()])
            direction[group][i] += terms
            direction["all"][i] += terms
    by_band = {group:{str(int(k)):dict(relative_l2=float(np.sqrt(e/ref)),
                   cosine=float(dot/np.sqrt(max(p*ref,1.e-24)))) for k,(e,p,ref,dot) in zip(BANDS,rows,strict=True)}
               for group,rows in direction.items()}
    return dict(waveform={name:float(np.sqrt(e/ref)) for name,(e,ref) in
                         zip(("pressure","deep_pressure","b5"),totals,strict=True)},
                directions=by_band, pressure_frames=frames)


def physical_probe(model, archive):
    p, _, speed = archive.arrays("train",0)
    del p
    shot = archive.items["train"][0][0]["shot_id"]
    speed = torch.tensor(speed,device=model.device,requires_grad=True)
    src = model.sources[[shot]]
    slots = torch.tensor([0,1,2,8,24,67,125,233],device=speed.device)
    context = model.encode(speed,src)
    b5 = model.decode_b5(context,slots)
    pressure = model.decode_pressure(context,slots)
    difference = relative(model.observe(pressure),b5)
    weight = torch.linspace(-1,1,b5.numel(),device=speed.device).reshape_as(b5)
    gradients = torch.autograd.grad((b5 * weight).sum(),speed)[0]
    return pressure.detach(), b5.detach(), gradients, float(difference.detach())


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run(args):
    from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
    from make_direction_labels import split_lock
    from no_pde_native_pressure import load_neural_pressure_core
    from tianhe_observations import CommittedMarmousi
    from train_joint_pressure import PressureArchive
    from train_no_pde_native import forbid_rollouts

    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(args.allocator_fraction,0)
    args.output.mkdir(parents=True,exist_ok=True)
    integrity = forbid_exact_label_files(args.dataset)
    with split_lock(args.output/".writer.lock"), forbid_rollouts():
        data = CommittedMarmousi(args.dataset)
        archive = PressureArchive(args.pressure_bank, data)
        model = load_neural_pressure_core(args.dataset,args.base_checkpoint,args.e12_checkpoint,args.selection,
                                         args.native_checkpoint,device=args.device,mode="lora_only").requires_grad_(False).eval()
        samples = Samples(args,data,model)
        train, val = samples.schedules(args.steps)
        plan = dict(model=model.identity, pressure_archive=archive.binding, direction_sources=samples.bindings,
                    train_pairs=train, val_pairs=val, arms={"identity":1., "mhc":.9}, steps_per_arm=args.steps,
                    trainer_sha256=sha(Path(__file__)), adapter_sha256=sha(Path(attach_mhc.__code__.co_filename)),
                    ordinary_only=True, numerical_rollouts=0, training_kind="signed_secants_and_projected_adjoint",
                    prior_direction_val_reused=True, update_lr=args.lr, secant_weight=args.secant_weight,
                    wave_window_energy_floor="2% of the peak same-shot window, deep also 0.5% of full-field peak")
        plan_path = args.output/"PLAN.json"
        if plan_path.exists():
            if read(plan_path) != json.loads(json.dumps(plan)):
                raise ValueError("Existing output has different input contracts or training steps")
        else:
            atomic_json(plan_path,plan)
        print(json.dumps(dict(stage="PLAN_COMMITTED", train=len(train), val=len(val))),flush=True)
        baseline_probe = physical_probe(model,archive)
        adapter = attach_mhc(model,1.)
        if hasattr(adapter.parent,"u_enhance") and adapter.parent.u_enhance is None:
            raise ValueError("Use the staged encoder with its original U enhancement")
        probe = physical_probe(model,archive)
        parity = {k:float(relative(x,y)) for k,x,y in zip(("pressure","b5","velocity_vjp"),probe[:3],baseline_probe[:3],strict=True)}
        parity["pressure_observation"] = probe[3]
        if max(parity.values()) > 2.e-5:
            raise ValueError(f"Warm-start pressure, observation or velocity VJP changed: {parity}")
        atomic_json(args.output/"WARM_START.json",dict(**parity, exact_template_integrity=integrity))
        print(json.dumps(dict(stage="WARM_START",**parity)),flush=True)
        if args.smoke:
            return
        start = monotonic()
        bands = response_bands(model.device)
        targets = fixed_train_targets(model,samples,train,args.frame_block)
        if (args.output/"BASELINE.json").is_file():
            baseline = read(args.output/"BASELINE.json")
        else:
            baseline = score(model,archive,samples,val,args.frame_block)
            atomic_json(args.output/"BASELINE.json",baseline)
        records = {}
        pressure_schedule = np.random.default_rng(SEED).permutation(len(archive.items["train"]))
        for arm,mix in plan["arms"].items():
            adapter.mix = mix
            with torch.no_grad():
                adapter.read.zero_()
            optimizer = torch.optim.AdamW([adapter.read],lr=args.lr,weight_decay=.01)
            directory = args.output/arm
            directory.mkdir(exist_ok=True)
            checkpoint = directory/"LAST.pt"
            previous_step = 0
            history = []
            if checkpoint.exists():
                state = torch.load(checkpoint,map_location=model.device,weights_only=True)
                if state["plan_sha256"] != sha(plan_path) or state["mix"] != mix:
                    raise ValueError("Incompatible optimizer continuation")
                with torch.no_grad():
                    adapter.read.copy_(state["read"])
                optimizer.load_state_dict(state["optimizer"])
                previous_step, history = state["step"], state["history"]
            for step in range(previous_step,args.steps):
                optimizer.zero_grad(set_to_none=True)
                p_index = int(pressure_schedule[step%len(pressure_schedule)])
                ordinary = wave_loss(model,archive,p_index,step)
                ordinary.backward()
                sample = targets[train[step]]
                directional = secant_loss(predict_pair(model,sample,args.frame_block),sample,bands)
                (args.secant_weight*directional).backward()
                gradient = torch.nn.utils.clip_grad_norm_([adapter.read], .5,error_if_nonfinite=True)
                optimizer.step()
                item = dict(arm=arm,step=step+1,ordinary=float(ordinary.detach()),signed=float(directional.detach()),
                            routing_gradient=float(gradient),case=train[step][0],seconds=monotonic()-start)
                history.append(item)
                atomic_checkpoint(checkpoint,dict(step=step+1,read=adapter.read.detach(),optimizer=optimizer.state_dict(),
                                  plan_sha256=sha(plan_path),mix=mix,history=history,model=model.identity))
                atomic_json(args.output/"STATUS.json",dict(status="TRAINING",arm=arm,step=step+1))
                print(json.dumps(dict(stage="UPDATE",**item)),flush=True)
            evaluation = score(model,archive,samples,val,args.frame_block)
            atomic_json(directory/"RESULT.json",evaluation)
            records[arm] = evaluation
        atomic_json(args.output/"RESULT.json",dict(status="COMPLETE",baseline=baseline,arms=records,
                    warm_start=parity,steps_per_arm=args.steps,no_test_labels=True,seconds=monotonic()-start))
        atomic_json(args.output/"STATUS.json",dict(status="COMPLETE",steps_per_arm=args.steps))


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for name in ("dataset","base-checkpoint","e12-checkpoint","selection","native-checkpoint",
                 "pressure-bank","bank-a","bank-f","structured","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--frame-block",type=int,default=128)
    p.add_argument("--steps",type=int,default=12)
    p.add_argument("--lr",type=float,default=.003)
    p.add_argument("--secant-weight",type=float,default=.35)
    p.add_argument("--allocator-fraction",type=float,default=.5)
    p.add_argument("--smoke",action="store_true")
    options=p.parse_args()
    if min(options.steps,options.threads,options.frame_block) <= 0 or not 0. < options.lr or not 0. < options.secant_weight:
        p.error("Require positive resumable training settings")
    run(options)
