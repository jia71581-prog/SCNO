#!/usr/bin/env python3
"""Train shared SCNO orthogonal-detail LoRA and coarse propagation on Marmousi."""

from argparse import ArgumentParser
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from train_sensitivity import (
    FRAMES, SPLITS, anchor_rows, canonical_sha, durable, log, normalized_loss,
    pair_bank, read_json, read_pair, require, save_json, sha256,
)


FRAMEWORK = "scno_orthogonal_detail_propagation_v1"


def source_binding():
    import coarse_propagation
    import finetune_ul_transfer
    import shared_framework
    import ul_current_lora

    result = {module.__name__: sha256(module.__file__) for module in (
        coarse_propagation, finetune_ul_transfer, shared_framework, ul_current_lora)}
    source = Path(finetune_ul_transfer.__file__).resolve().parent
    result["current_ul_runner"] = sha256(source / "finetune_ul_current_pair.py")
    result["native_residual_operator"] = sha256(source / "vendor/residual_support/operators.py")
    result["trainer"] = sha256(__file__)
    result["training_support"] = sha256(Path(__file__).with_name("train_sensitivity.py"))
    return result


def new_readout(dataset, checkpoint, device, build_dir=None):
    from shared_framework import SharedFrameworkReadout

    model = SharedFrameworkReadout(dataset, checkpoint, torch.device(device), build_dir)
    parameters = [value for value in model.parameters() if value.requires_grad]
    require(sum(value.numel() for value in parameters) == 769 and model.gain.numel() == 1
            and model.gain.requires_grad and len(parameters) == 7,
            "Expected exactly six rank-four factors and one propagation gain")
    return model


def load_sensitivity_readout(dataset, checkpoint, adapter_checkpoint, *, expected_sha256, device="cuda:0"):
    from kernel_corrected_readout import KERNEL_SHA, NATIVE_SHA
    from tianhe_observations import CommittedMarmousi

    require(sha256(adapter_checkpoint) == expected_sha256, "Frozen validation-selected adapter changed")
    saved = torch.load(adapter_checkpoint, map_location="cpu", weights_only=True)
    binding = saved["binding"]
    data = CommittedMarmousi(dataset)
    require(saved["selected"] is True and saved["epoch"] > 0 and binding["framework"] == FRAMEWORK
            and binding["dataset_binding"] == data.binding and binding["receiver_rms"] == data.rms
            and binding["kernel_sha256"] == KERNEL_SHA and binding["native_sha256"] == NATIVE_SHA
            and binding["training_used_exact_marmousi"] is False and binding["selection_used_exact_marmousi"] is False
            and binding["training_or_selection_used_fwi_audits"] is False
            and binding["sources"] == source_binding(), "Selected shared-framework candidate binding differs")
    model = new_readout(data.root, checkpoint, device)
    model.load_state_dict(saved["model"], strict=True)
    require(torch.isfinite(model.gain) and 0. <= float(model.gain) <= 1., "Invalid saved propagation gain")
    return model.requires_grad_(False).eval()


def cached(cache_dir, binding, key, construct):
    path = cache_dir / (canonical_sha(key) + ".pt")
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        require(saved["binding"] == binding and saved["key"] == key, "Coarse proposal cache binding changed")
    else:
        payload = {"binding": binding, "key": key, "batch": construct()}
        durable(path, lambda stream: torch.save(payload, stream))
    return path


def load_batch(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)["batch"]


def interior_support(pair_id, seed, count):
    rng = np.random.default_rng(int(canonical_sha([pair_id, seed])[:16], 16))
    strata = np.array_split(np.arange(1, FRAMES - 1), count)
    centers = np.array([rng.choice(group) for group in strata])
    requested = np.r_[centers - 1, centers, centers + 1]
    frames, inverse = np.unique(requested, return_inverse=True)
    return frames.tolist(), inverse.tolist()


@torch.no_grad()
def prepare_wave(model, velocity, ids, target, support=None):
    media = torch.as_tensor(velocity, dtype=torch.float32, device=model.device)
    sources = model.sources[ids]
    frames = [0] if support is None else support[0]
    proposal, pressure = model.make_proposal(media, sources, snapshot_slots=frames)
    frozen = model.encode(media, sources, proposal=proposal).propagation_difference
    require(frozen.shape == target.shape and torch.isfinite(frozen).all(), "Invalid cached coarse propagation direction")
    batch = {"velocity": media.cpu(), "ids": ids, "target": torch.as_tensor(target), "coarse_direction": frozen.cpu()}
    if support is not None:
        require(pressure.shape == (len(ids), len(frames), 201, 201) and torch.isfinite(pressure).all(),
                "Invalid native coarse pressure support")
        batch["defect"] = {"frames": frames, "inverse": support[1], "proposal": pressure[:1].cpu()}
    return batch


def prepare_all(model, data, pairs, anchors, directory, binding, args):
    prepared_pairs = {split: [] for split in SPLITS}
    prepared_anchors = {split: [] for split in SPLITS}
    for split in SPLITS:
        for i, (record, path) in enumerate(pairs[split]):
            def prepare():
                vm, vp, ids, y, secant, h = read_pair(record, path, data)
                eligible = split == "train" and i % args.defect_every == 0
                support = interior_support(record["pair_id"], args.seed, args.defect_centers) if eligible else None
                return {"minus": prepare_wave(model, vm, ids.tolist(), y[0]),
                        "plus": prepare_wave(model, vp, ids.tolist(), y[1], support),
                        "secant": torch.from_numpy(secant), "h": h}

            key = ["pair", split, record["pair_id"], record["shard_sha256"]]
            prepared_pairs[split].append(cached(directory, binding, key, prepare))
            log({"stage": "pair_coarse_cache", "split": split, "committed": i + 1, "total": len(pairs[split])})
    for split in SPLITS:
        for i, (key, shot, index) in enumerate(anchors[split], 1):
            def prepare():
                sample = data.sample(split, index, np.arange(FRAMES))
                require(sample["model_id"] == key and sample["shot_id"] == shot, "Frozen waveform anchor changed")
                return prepare_wave(model, sample["velocity"], [shot], sample["target"][None])

            prepared_anchors[split].append(cached(directory, binding, ["anchor", split, key, shot], prepare))
            if i % 8 == 0 or i == len(anchors[split]):
                log({"stage": "anchor_coarse_cache", "split": split, "committed": i, "total": len(anchors[split])})
    return prepared_pairs, prepared_anchors


def evaluate_wave(model, record, frames=None, *, native_defect=False):
    from shared_framework import Encoded, defect_loss

    velocity = record["velocity"].to(model.device)
    sources = model.sources[record["ids"]]
    with torch.no_grad():
        encoded = model.base.encode(velocity, sources)
    context = Encoded(encoded, record["coarse_direction"].to(model.device))
    values = model.decode(context, frames, normalized=True).double()
    labels = record["target"] if frames is None else record["target"][:, frames]
    defect = values.new_zeros(())
    if native_defect and "defect" in record:
        support = record["defect"]
        native_sources = sources[:1]
        with torch.no_grad():
            enc = model.base.encode(velocity, native_sources)
        native_context = Encoded(enc, context.propagation_difference[:1])
        proposal = support["proposal"].to(model.device)
        physical = model.full_pressure(native_context, support["frames"], proposal)
        slots = support["inverse"]
        observation = record["target"][:1].to(model.device)
        defect = defect_loss(model, physical[:, slots], proposal[:, slots], velocity, native_sources,
                             len(slots) // 3, observation)
    return values, labels.to(values), defect


def evaluate_pair(model, record, frames=None, *, native_defect=False):
    lower, target_lower, _ = evaluate_wave(model, record["minus"], frames)
    upper, target_upper, defect = evaluate_wave(model, record["plus"], frames, native_defect=native_defect)
    secant = record["secant"] if frames is None else record["secant"][:, frames]
    return ((upper - lower) / (2 * record["h"]), secant.to(upper),
            torch.cat((lower, upper)), torch.cat((target_lower, target_upper)), defect)


@torch.no_grad()
def validate(model, pairs, anchors):
    totals = {name: torch.zeros(4, dtype=torch.float64, device=model.device)
              for name in ("secant", "endpoints", "waveforms")}

    def accumulate(name, prediction, labels):
        totals[name] += torch.stack(((prediction - labels).square().sum(), labels.square().sum(),
                                     (prediction * labels).sum(), prediction.square().sum()))

    for path in pairs:
        secant, target, traces, labels, _ = evaluate_pair(model, load_batch(path))
        accumulate("secant", secant, target)
        accumulate("endpoints", traces, labels)
    for path in anchors:
        traces, labels, _ = evaluate_wave(model, load_batch(path))
        accumulate("waveforms", traces, labels)
    result = {}
    for name, value in totals.items():
        sse, energy, projection, predicted = value.tolist()
        require(min(energy, predicted) > 0., f"Zero held-out {name} energy")
        result[name] = {"relative_l2": float(np.sqrt(sse / energy)), "signed_cosine": float(projection / np.sqrt(energy * predicted)),
                        "rms_gain": float(np.sqrt(predicted / energy))}
    return result


def fit(model, optimizer, pairs, anchors, args, epoch):
    rng = np.random.default_rng(args.seed + epoch)
    anchor_order = rng.permutation(len(anchors))
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros(4, device=model.device, dtype=torch.float64)
    defects = 0
    for step, row in enumerate(rng.permutation(len(pairs))):
        frames = np.sort(rng.choice(FRAMES, size=args.sample_frames, replace=False))
        pair = load_batch(pairs[row])
        anchor = load_batch(anchors[anchor_order[step % len(anchors)]])
        secant, target, traces, labels, defect = evaluate_pair(model, pair, frames, native_defect=True)
        prediction, observed, _ = evaluate_wave(model, anchor, frames)
        losses = torch.stack((normalized_loss(secant, target, pair["secant"], len(frames)),
                              normalized_loss(traces, labels, torch.cat((pair["minus"]["target"], pair["plus"]["target"])), len(frames)),
                              normalized_loss(prediction, observed, anchor["target"], len(frames)), defect))
        total += losses.detach()
        defects += int("defect" in pair["plus"])
        weights = losses.new_tensor((1., args.endpoint_weight, args.anchor_weight, args.defect_weight * args.defect_every))
        scalar = losses.dot(weights) + args.trust_weight * model.trust()
        group = min(args.grad_accum, len(pairs) - (step // args.grad_accum) * args.grad_accum)
        (scalar / group).backward()
        if (step + 1) % args.grad_accum == 0 or step + 1 == len(pairs):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                model.gain.clamp_(0., 1.)
            optimizer.zero_grad(set_to_none=True)
    means = total / total.new_tensor([len(pairs)] * 3 + [max(defects, 1)])
    return {**dict(zip(("secant", "endpoints", "waveforms", "propagation_defect"), means.tolist())),
            "propagation_gain": float(model.gain.detach()), "defect_records": defects,
            "lora_trust": float(model.trust().detach())}


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("dataset", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pairs", type=Path, nargs="+", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    for name, default in (("epochs", 12), ("threads", 4), ("seed", 261120), ("sample_frames", 64),
                          ("train_anchors", 40), ("val_anchors", 30), ("grad_accum", 4),
                          ("defect_every", 4), ("defect_centers", 3)):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in (("lr", 2e-4), ("gain_lr", 5e-4), ("endpoint_weight", .5), ("anchor_weight", 1.),
                          ("defect_weight", .001), ("trust_weight", .001), ("wave_limit", .03), ("secant_gain", .005)):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    args = parser.parse_args()
    require(args.resume == args.output.is_dir() and 1 <= args.sample_frames <= FRAMES
            and 1 <= args.defect_centers <= 16 and min(args.epochs, args.threads, args.train_anchors,
                    args.val_anchors, args.grad_accum, args.defect_every) > 0
            and all(np.isfinite(x) and x > 0 for x in (args.lr, args.gain_lr, args.endpoint_weight, args.anchor_weight,
                    args.defect_weight, args.trust_weight)) and 0 <= args.wave_limit <= .25 and 0 < args.secant_gain < 1.,
            "Invalid shared training configuration or output directory")
    for original in (args.dataset, *args.pairs, args.checkpoint.parent):
        require(not args.output.resolve().is_relative_to(original.resolve()), "Expected a fresh directory outside frozen inputs")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = monotonic()

    from kernel_corrected_readout import KERNEL_SHA, NATIVE_SHA
    from tianhe_observations import CommittedMarmousi

    data = CommittedMarmousi(args.dataset)
    pairs, digests = {}, {}
    for split in SPLITS:
        entries = [pair_bank(data, root, split) for root in args.pairs]
        pairs[split] = [pair for group, _ in entries for pair in group]
        digests[split] = [digest for _, digest in entries]
        require(len({row["pair_id"] for row, _ in pairs[split]}) == len(pairs[split]), f"Duplicate {split} pair")
    parents = {split: {record["parent_group_id"] for record, _ in pairs[split]} for split in SPLITS}
    require(not (parents["train"] & parents["validation"]), "Train and validation share a directional parent")
    anchors = {split: anchor_rows(data, split, count, args.seed + j)
               for j, (split, count) in enumerate(zip(SPLITS, (args.train_anchors, args.val_anchors)))}
    binding = {"framework": FRAMEWORK, "dataset_binding": data.binding, "receiver_rms": data.rms, "sources": source_binding(),
               "kernel_sha256": KERNEL_SHA, "native_sha256": NATIVE_SHA, "directional_manifest_sha256": digests,
               "anchors": anchors, "training_used_exact_marmousi": False, "selection_used_exact_marmousi": False,
               "training_or_selection_used_fwi_audits": False,
               "configuration": {k: v for k, v in vars(args).items() if k not in
                                 {"dataset", "checkpoint", "pairs", "output", "resume", "epochs", "device", "threads"}}}
    root = args.output
    anchor_file, latest, approval = root / "BINDING.json", root / "latest.pt", root / "SELECTED.json"
    if args.resume:
        require(not approval.exists() and canonical_sha(read_json(anchor_file)) == canonical_sha(binding),
                "Preserve frozen shared framework approval and configuration")
    else:
        root.mkdir(parents=False, exist_ok=False)
        save_json(anchor_file, binding)
    cache_path = root / "cache"
    cache_path.mkdir(exist_ok=True)
    model = new_readout(data.root, args.checkpoint, args.device, root / "cuda_build")
    trainable_detail = [parameter for parameter in model.detail.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([{"params": trainable_detail, "lr": args.lr},
                                  {"params": [model.gain], "lr": args.gain_lr}], weight_decay=1e-4)
    records, best = [], None
    start_epoch = 1
    if latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        require(state["binding"] == binding and 0 <= state["epoch"] <= args.epochs, "Resume checkpoint binding differs")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        records, best, start_epoch = state["history"], state["best"], state["epoch"] + 1
        if best:
            require(sha256(root / best["file"]) == best["sha256"], "Validation candidate artifact changed")
    prepared, wave_anchors = prepare_all(model.eval(), data, pairs, anchors, cache_path, canonical_sha(binding), args)
    if not records:
        baseline = validate(model, prepared["validation"], wave_anchors["validation"])
        records.append({"epoch": 0, "validation": baseline})
    else:
        baseline = records[0]["validation"]
    log({"stage": "baseline", "framework": FRAMEWORK, "validation": baseline, "wall_seconds": monotonic() - started})
    for epoch in range(start_epoch, args.epochs + 1):
        train = fit(model.train(), optimizer, prepared["train"], wave_anchors["train"], args, epoch)
        val = validate(model.eval(), prepared["validation"], wave_anchors["validation"])
        guard = all(val[name]["relative_l2"] <= baseline[name]["relative_l2"] * (1. + args.wave_limit)
                    for name in ("endpoints", "waveforms"))
        score = val["secant"]["relative_l2"]
        selected = guard and score < min(baseline["secant"]["relative_l2"] * (1. - args.secant_gain),
                                         best["score"] if best else np.inf)
        record = {"epoch": epoch, "training": train, "validation": val, "waveform_guard": guard,
                  "selected": bool(selected), "wall_seconds": monotonic() - started}
        records.append(record)
        if selected:
            candidate = root / f"candidate_epoch_{epoch:04d}.pt"
            durable(candidate, lambda stream: torch.save({"binding": binding, "epoch": epoch, "selected": True,
                    "model": model.state_dict(), "validation": val}, stream))
            best = {"epoch": epoch, "file": candidate.name, "sha256": sha256(candidate), "score": score, "validation": val}
        durable(latest, lambda stream: torch.save({"binding": binding, "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "history": records, "best": best}, stream))
        save_json(root / "TRAINING.json", {"status": "RUNNING", "binding": binding, "baseline": baseline,
                                          "best": best, "history": records})
        log({"stage": "epoch", **record})
    save_json(root / "TRAINING.json", {"status": "COMPLETE", "binding": binding, "baseline": baseline,
                                      "best": best, "history": records})
    if best:
        held_out = digests["validation"]
        receipt = {"schema": "marmousi_b5_sensitivity_selection_v1", "status": "APPROVED", "selection_split": "validation",
                   "framework": FRAMEWORK, "candidate_checkpoint_file": best["file"], "candidate_checkpoint_sha256": best["sha256"],
                   "base_checkpoint_sha256": KERNEL_SHA, "dataset_contract_sha256": data.binding["contract_hash"],
                   "dataset_manifest_sha256": data.binding["manifest_hash"],
                   "held_out_val_pairs_sha256": held_out[0] if len(held_out) == 1 else canonical_sha(held_out),
                   "held_out_val_pair_manifest_files_sha256": held_out, "epoch": best["epoch"],
                   "training_used_exact_marmousi": False, "selection_used_exact_marmousi": False,
                   "training_or_selection_used_fwi_audits": False, "baseline_validation": baseline,
                   "candidate_validation": best["validation"], "max_allowed_waveform_relative_regression": args.wave_limit,
                   "minimum_required_relative_secant_improvement": args.secant_gain}
        save_json(approval, receipt)
        log({"stage": "selection_frozen", **receipt})
    else:
        log({"stage": "no_qualifying_validation_improvement"})


if __name__ == "__main__":
    main()
