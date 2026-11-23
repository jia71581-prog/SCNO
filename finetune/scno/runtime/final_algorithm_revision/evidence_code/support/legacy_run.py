#!/usr/bin/env python3
"""Local, isolated homogeneous loss/sampling ablation. No future-label selection."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for path in (str(HERE), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class ThresholdLedger:
    def __init__(self, epsilon_p=.01):
        if not 0 < epsilon_p < 1:
            raise ValueError("A probability threshold in (0,1) is required")
        self.epsilon_p = epsilon_p
        self.endpoint_step = None

    def observe(self, step, probability):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("A finite full-grid probability is required")
        if self.endpoint_step is None and probability < self.epsilon_p:
            self.endpoint_step = int(step)
            return True
        return False


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gradient_vector(parameters):
    import torch
    return torch.cat([(p.grad.detach() if p.grad is not None else
                       torch.zeros_like(p)).reshape(-1) for p in parameters])


def parameter_digest(parameters):
    import torch
    digest = hashlib.sha256()
    for p in parameters:
        value = p.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initial_loss(adapter, *, create_graph):
    import torch
    u0, velocity0 = adapter.initial_state(create_graph=create_graph)
    value = (u0.double().square() +
             (velocity0.double() / (2 * math.pi * 15)).square()).mean()
    if not torch.isfinite(value):
        raise FloatingPointError("Nonfinite initial-condition loss")
    return value


def train_update(adapter, optimizer, early, early_times, centers, velocity,
                 nu_squared, sampler, plan, counters, *, batch=2):
    """Only early truth is an accepted argument. Future truth is evaluation-only."""
    import numpy as np
    import torch
    from operators import lwc_residual
    params = list(adapter.parameters())
    optimizer.zero_grad(set_to_none=True)
    denominator = early.double().square().sum().clamp_min(1e-24)
    losses = {"data_loss": 0., "ic_loss": 0., "pde_loss": 0.}
    saved_energies = []
    sync()
    started = time.perf_counter()
    with adapter.training_scope():
        for start in range(0, len(early_times), batch):
            times = early_times[start:start+batch]
            predicted = adapter.query(times, grad=True)
            loss = ((predicted.double()-early[start:start+batch].double()).square().sum()
                    / denominator)
            loss.backward()
            losses["data_loss"] += float(loss.detach())
            counters["pressure_queries"] += len(times)
        data_gradient = gradient_vector(params).clone()
        ic = initial_loss(adapter, create_graph=True)
        (.01 * ic).backward()
        losses["ic_loss"] = float(ic.detach())
        after_ic = gradient_vector(params).clone()
        counters["pressure_queries"] += 1
        counters["initial_derivative_queries"] += 1
        if plan is not None:
            selected = plan["time_indices"]
            for start in range(0, len(selected), batch):
                indices = selected[start:start+batch]
                t = centers[torch.as_tensor(indices, device=centers.device)]
                times = torch.cat((t-.0005, t, t+.0005))
                values = adapter.query(times, grad=True)
                n = len(t)
                residual = lwc_residual(values[:n], values[n:2*n], values[2*n:], velocity)
                energies = sampler.tile_energies(residual, nu_squared)
                weights = torch.as_tensor(plan["matrix_coeffs"][start:start+n],
                                          device=energies.device, dtype=torch.float64)
                loss = (energies*weights).sum()
                (.001*loss).backward()
                losses["pde_loss"] += float(loss.detach())
                saved_energies.extend(energies.detach().cpu().numpy())
                counters["pressure_queries"] += 3*n
                counters["residual_node_evaluations"] += n*sampler.n_valid
            counters["time_draws"] += int(plan["n_time_draws"])
            counters["tile_draws"] += int(plan["n_time_draws"]*plan["n_tile_draws"])
        final_gradient = gradient_vector(params)
        if not torch.isfinite(final_gradient).all():
            raise FloatingPointError("Nonfinite adapter gradient")
        losses.update(data_gradient_norm=float(data_gradient.norm()),
                      weighted_ic_gradient_norm=float((after_ic-data_gradient).norm()),
                      weighted_pde_gradient_norm=float((final_gradient-after_ic).norm()))
        losses["gradient_norm_before_clip"] = float(
            torch.nn.utils.clip_grad_norm_(params, 1.))
        before = [p.detach().clone() for p in params]
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in params):
            raise FloatingPointError("Nonfinite SOAP proposal")
        changed = any(not torch.equal(p.detach(), old) for p, old in zip(params, before))
        delta = math.sqrt(sum(float((p.detach()-old).double().square().sum())
                              for p, old in zip(params, before)))
    if plan is not None:
        sampler.commit(plan, np.asarray(saved_energies))
    sync()
    losses.update(parameter_changed=changed, parameter_delta_norm=delta,
                  training_step_s=time.perf_counter()-started)
    losses["weighted_objective"] = (losses["data_loss"] + .01*losses["ic_loss"] +
                                    .001*losses["pde_loss"])
    return losses


def monitor(adapter, centers, velocity, nu_squared, sampler, counters, *, batch=32, need_energies=False):
    """Full finite grid count, without updating weights or AIS history."""
    import numpy as np
    import torch
    from operators import full_grid_diagnostics, lwc_residual
    energy_rows = []
    failures = units = 0
    residual_sum = rho_sum = 0.
    rho_max = 0.
    sync()
    started = time.perf_counter()
    with torch.no_grad(), adapter.training_scope():
        for start in range(0, len(centers), batch):
            t = centers[start:start+batch]
            values = adapter.query(torch.cat((t-.0005, t, t+.0005)), grad=False)
            n = len(t)
            residual = lwc_residual(values[:n], values[n:2*n], values[2*n:], velocity)
            diag = full_grid_diagnostics(residual, sampler.mask, nu_squared=nu_squared)
            failures += diag["failure_count"]
            units += diag["units"]
            residual_sum += diag["residual_mean_square"]*diag["units"]
            rho_sum += diag["rho_mean_square"]*diag["units"]
            rho_max = max(rho_max, diag["rho_max"])
            if need_energies:
                energy_rows.append(sampler.tile_energies(residual, nu_squared).cpu().numpy())
            counters["pressure_queries"] += 3*n
            counters["residual_node_evaluations"] += n*sampler.n_valid
    sync()
    probability = failures/units
    return dict(failure_probability=probability, failure_count=failures, units=units,
                threshold_satisfied=probability < .01, rho_max=rho_max,
                rho_mean_square=rho_sum/units,
                pde_mean=residual_sum/units/float(nu_squared),
                monitor_s=time.perf_counter()-started), (np.concatenate(energy_rows) if energy_rows else None)


def evaluate(adapter, reference_path, times, counters, *, batch=8, save_path=None):
    """Read full truth only after a state is frozen for predeclared evaluation."""
    import numpy as np
    import torch
    target = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    if target.shape != (400, 201, 201):
        raise ValueError("Expected the registered full400 reference")
    sse, energy = [], []
    prediction = None
    if save_path is not None:
        prediction = np.lib.format.open_memmap(save_path, mode="w+", dtype=np.float32,
                                               shape=(400, 201, 201))
    sync()
    started = time.perf_counter()
    with torch.no_grad(), adapter.training_scope():
        for start in range(0, len(times), batch):
            t = times[start:start+batch]
            pred = adapter.query_physical(t).detach().cpu().numpy()
            if not np.isfinite(pred).all():
                raise FloatingPointError("Nonfinite prediction evaluation")
            ref = np.asarray(target[start:start+len(t)], dtype=np.float64)
            sse.extend(np.square(pred.astype(np.float64)-ref).sum((1, 2)).tolist())
            energy.extend(np.square(ref).sum((1, 2)).tolist())
            if prediction is not None:
                prediction[start:start+len(t)] = pred
            counters["evaluation_pressure_queries"] += len(t)
    if prediction is not None:
        prediction.flush()
    sync()
    return dict(times_s=times.detach().cpu().double().tolist(),
                spatial_node_count=40401, pressure_units="Pa",
                per_frame_squared_error=sse, per_frame_reference_energy=energy,
                prediction_s=time.perf_counter()-started)


def fresh_counters():
    return dict(pressure_queries=0, evaluation_pressure_queries=0,
                residual_node_evaluations=0, time_draws=0, tile_draws=0,
                initial_derivative_queries=0)


def checkpoint(path, adapter, optimizer, sampler, *, step, attempts, ledger, counters):
    import numpy as np
    import torch
    payload = dict(schema="UNIFORM_LOSS_AIS_ABLATION_V1",
                   adapters=adapter.adapter_state_dict(), optimizer=optimizer.state_dict(),
                   sampler=sampler.state_dict(), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                   step=step, attempts=attempts, endpoint_step=ledger.endpoint_step,
                   counters=copy.deepcopy(counters))
    temporary = Path(str(path)+".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(args):
    import numpy as np
    import torch
    from model_adapter import FrozenBaseCache, UniformAdapter, load_pretrained
    from operators import AISSampler, make_valid_mask, residual_scale_squared
    from native400_runtime.vendor.soap_upstream import SOAP
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261105)
    np.random.seed(20261105)
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    code = out/"code";code.mkdir()
    for f in HERE.glob("*.py"):
        (code/f.name).write_bytes(f.read_bytes())
    write_json(out/"STATUS.json",dict(status="loading", pid=os.getpid(), started_at=utc()))
    model, assets = load_pretrained(args.device, dense_time_block=4, refiner_time_block=4)
    times = assets["full_times"].to(args.device)
    centers = times.clone().clamp_max(1-.0005)
    # Extraction of only the permitted labels; later values are read only by evaluate().
    reference = np.load(assets["reference_path"], mmap_mode="r", allow_pickle=False)
    early_np = np.array(reference[:8], copy=True)
    del reference
    early = torch.as_tensor(early_np, device=args.device)
    early = early/(assets["pressure_scale_pa"]*float(assets["source"][4]))
    nu_squared = residual_scale_squared(early)
    mask = make_valid_mask()
    milestones = [0, 1, 8, 16, 32, 64]
    milestones = sorted(set(x for x in milestones if x <= args.steps) | {args.steps})
    config = dict(schema="UNIFORM_LOSS_AIS_ABLATION_V1", created_at=utc(),
                  code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob("*.py")},
                  mode=args.mode, case_id="uniform4000", seed=args.seed,
                  arms=["NO_ADAPT", "EARLY_ONLY", "PHYS_UNIFORM", "PHYS_AIS"],
                  max_actual_updates=args.steps, max_attempts=args.steps+2,
                  per_arm_wall_limit_s=args.max_wall, monitoring_steps=milestones,
                  diagnostic_continuation=True, early_count=8, late_start_s=.4,
                  times_s=times.detach().cpu().double().tolist(),
                  residual_centers_s=centers.detach().cpu().double().tolist(),
                  epsilon_r=.25, epsilon_p=.01, stop_grid="full_admissible_grid",
                  pde_weight=.001, ic_weight=.01, data_weight=1.,
                  optimizer="SOAP", lr=1e-4, betas=[.95,.95], weight_decay=0.,
                  precondition_frequency=10, max_precond_dim=args.max_precond_dim,
                  gradient_norm_limit=1., n_time_draws=8, n_tile_draws=8,
                  tile_size=16, beta_uniform=.1, ema=.9, rank=8, alpha=8,
                  checkpoint_path=str(assets["checkpoint_path"]),
                  checkpoint_sha256=assets["checkpoint_sha256"],
                  reference_path=str(assets["reference_path"]),
                  pressure_scale_pa=assets["pressure_scale_pa"], pressure_units="Pa",
                  pretrained_parameters=assets["pretrained_parameters"],
                  checkpoint_has_final_dg=False, device=args.device,
                  hardware=torch.cuda.get_device_name() if args.device=="cuda" else "CPU",
                  environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__),
                  batch=args.batch, monitor_batch=args.monitor_batch,
                  cached_base_equivalence_required=True, timing_includes_reference_generation=False)
    # Persist design before observing baseline errors.
    write_json(out/"config.json",config)
    np.save(out/"early8_normalized.npy", early.cpu().numpy())
    np.save(out/"valid_mask.npy", mask.numpy())
    write_json(out/"STATUS.json",dict(status="preparing_frozen_base",pid=os.getpid(),updated_at=utc()))
    cache_start=time.perf_counter()
    cache = FrozenBaseCache(model, assets["velocity"], assets["source"],
                            cache_device="cpu", block=4)
    all_times = torch.cat((times, centers-.0005, centers, centers+.0005))
    cache.prefill(all_times)
    cache.initial_base()
    sync()
    common_preparation_s=time.perf_counter()-cache_start
    write_json(out/"CACHE.json",dict(frozen_base_queries=cache.generated_queries,
               preparation_s=common_preparation_s,shared_by_all_arms=True,
               no_reference_labels_used=True))
    baseline = UniformAdapter(model, cache, seed=args.seed, install_adapters=False)
    counters = fresh_counters()
    sampler = AISSampler(mask, n_times=len(centers), seed=args.seed)
    write_json(out/"STATUS.json",dict(status="baseline_full_grid_check",pid=os.getpid(),updated_at=utc()))
    baseline_diag, initial_energies = monitor(
        baseline,centers,assets["velocity"],nu_squared,sampler,counters,
        batch=args.monitor_batch,need_energies=True)
    baseline_eval = evaluate(baseline,assets["reference_path"],times,counters,
                             save_path=out/"pretrained_prediction_pa.npy")
    base_dir=out/"NO_ADAPT";base_dir.mkdir()
    base_record=dict(arm="NO_ADAPT",seed=args.seed,step=0,role="diagnostic",
                     status="not_adapted",elapsed_s=0.,trainable_parameters=0,
                     **baseline_diag,**baseline_eval,**counters)
    write_json(base_dir/"diagnostic_metrics.json",[base_record])
    write_json(out/"BASELINE.json",dict(**baseline_diag,**counters))
    del baseline
    order=["EARLY_ONLY","PHYS_UNIFORM","PHYS_AIS"]
    random.Random(args.seed).shuffle(order)
    write_json(out/"ARM_ORDER.json",order)
    initial_adapter_digest = None
    for arm in order:
        arm_dir=out/arm;arm_dir.mkdir()
        adapter=UniformAdapter(model,cache,seed=args.seed)
        params=list(adapter.parameters())
        digest=parameter_digest(params)
        if initial_adapter_digest is None:initial_adapter_digest=digest
        if digest != initial_adapter_digest:raise AssertionError("Unpaired adapter initialization")
        write_json(arm_dir/"INITIALIZATION.json",dict(**adapter.receipt,adapter_sha256=digest))
        optimizer=SOAP(params,lr=1e-4,betas=(.95,.95),weight_decay=0.,
                       precondition_frequency=10,max_precond_dim=args.max_precond_dim)
        sampling=AISSampler(mask,n_times=len(centers),seed=args.seed)
        sampling.initialize_history(initial_energies)
        counters=fresh_counters();ledger=ThresholdLedger()
        records=[]
        arm_started=time.perf_counter();eval_total_s=0.
        train_total_s=0.;monitor_total_s=0.
        step=attempts=0
        ledger.observe(0,baseline_diag["failure_probability"])
        initial_record=dict(base_record,arm=arm,status=("initial_threshold_satisfied"
                            if ledger.endpoint_step==0 else "initial_state"),
                            trainable_parameters=sum(p.numel() for p in params),
                            shared_initial_evaluation=True, **counters)
        records.append(initial_record)
        if ledger.endpoint_step==0:
            write_json(arm_dir/"endpoint_metrics.json",
                       dict(initial_record,role="threshold_endpoint",status="initial_threshold_satisfied"))
            checkpoint(arm_dir/"threshold_endpoint.pt",adapter,optimizer,sampling,
                       step=0,attempts=0,ledger=ledger,counters=counters)
        write_json(arm_dir/"diagnostic_metrics.json",records)
        status="running"
        try:
            while step<args.steps and attempts<args.steps+2:
                elapsed=time.perf_counter()-arm_started-eval_total_s
                if elapsed>=args.max_wall:
                    status="time_exhausted";break
                plan=(None if arm=="EARLY_ONLY" else sampling.draw(
                    mode="uniform" if arm=="PHYS_UNIFORM" else "ais",
                    n_time_draws=8,n_tile_draws=8))
                losses=train_update(adapter,optimizer,early,times[:8],centers,
                    assets["velocity"],nu_squared,sampling,plan,counters,batch=args.batch)
                attempts+=1;train_total_s+=losses["training_step_s"]
                if losses["parameter_changed"]:step+=1
                event=dict(arm=arm,seed=args.seed,step=step,optimizer_attempts=attempts,
                           elapsed_s=time.perf_counter()-arm_started-eval_total_s,
                           **losses,**counters)
                append_json(arm_dir/"metrics.jsonl",event)
                write_json(out/"STATUS.json",dict(status="running",pid=os.getpid(),
                           arm=arm,step=step,optimizer_attempts=attempts,updated_at=utc()))
                if losses["parameter_changed"] and step in milestones:
                    diag,_=monitor(adapter,centers,assets["velocity"],nu_squared,sampling,
                                   counters,batch=args.monitor_batch)
                    monitor_total_s+=diag["monitor_s"]
                    endpoint=ledger.observe(step,diag["failure_probability"])
                    eval_record=evaluate(adapter,assets["reference_path"],times,counters,
                       save_path=arm_dir/"final_diagnostic_prediction_pa.npy" if step==args.steps else None)
                    eval_total_s+=eval_record["prediction_s"]
                    record=dict(arm=arm,seed=args.seed,step=step,role="diagnostic",
                        status="diagnostic_continuation" if ledger.endpoint_step is not None else "threshold_not_met",
                        trainable_parameters=sum(p.numel() for p in params),
                        elapsed_s=time.perf_counter()-arm_started-eval_total_s,
                        optimizer_attempts=attempts,train_s=train_total_s,monitor_total_s=monitor_total_s,
                        **diag,**eval_record,**counters)
                    records.append(record)
                    write_json(arm_dir/"diagnostic_metrics.json",records)
                    append_json(arm_dir/"metrics.jsonl",{k:v for k,v in record.items()
                                if not k.startswith("per_frame") and k!="times_s"})
                    checkpoint(arm_dir/"last.pt",adapter,optimizer,sampling,step=step,
                               attempts=attempts,ledger=ledger,counters=counters)
                    if endpoint:
                        write_json(arm_dir/"endpoint_metrics.json",
                                   dict(record,role="threshold_endpoint",status="threshold_satisfied"))
                        checkpoint(arm_dir/"threshold_endpoint.pt",adapter,optimizer,sampling,
                                   step=step,attempts=attempts,ledger=ledger,counters=counters)
            else:
                status="diagnostic_budget_complete" if step==args.steps else "attempt_budget_exhausted"
            if ledger.endpoint_step is None:
                write_json(arm_dir/"endpoint_metrics.json",dict(arm=arm,seed=args.seed,step=step,
                    role="threshold_endpoint",status=status,threshold_satisfied=False,
                    failure_probability=(records[-1]["failure_probability"] if step==records[-1]["step"] else None),
                    last_monitored_step=records[-1]["step"],
                    last_monitored_failure_probability=records[-1]["failure_probability"],
                    elapsed_s=time.perf_counter()-arm_started-eval_total_s,
                    endpoint_not_reached=True,accuracy_not_inferred=True))
            write_json(arm_dir/"FINISHED.json",dict(status=status,step=step,
                optimizer_attempts=attempts,endpoint_step=ledger.endpoint_step,
                train_s=train_total_s,monitor_s=monitor_total_s,evaluation_s=eval_total_s,
                elapsed_s=time.perf_counter()-arm_started,**counters))
        except Exception:
            write_json(arm_dir/"FAILURE.json",dict(status="numerical_failure",step=step,
                optimizer_attempts=attempts,traceback=traceback.format_exc(),updated_at=utc()))
            raise
        finally:
            del optimizer,adapter,params
            if torch.cuda.is_available():torch.cuda.empty_cache()
    write_json(out/"STATUS.json",dict(status="training_complete",pid=os.getpid(),finished_at=utc(),
               common_base_preparation_s=common_preparation_s))
    from analyze import main as analyze_main
    analysis_status=analyze_main(["--run-dir", str(out)])
    if analysis_status not in (None,0):
        raise RuntimeError("Analysis failed after completed training")
    write_json(out/"STATUS.json",dict(status="complete",pid=os.getpid(),finished_at=utc(),analysis_status="complete"))
    print("COMPLETE",out,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--mode",choices=["probe","run"],default="run")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--steps",type=int,default=64)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--max-wall",type=float,default=1800.)
    parser.add_argument("--batch",type=int,default=2)
    parser.add_argument("--monitor-batch",type=int,default=32)
    parser.add_argument("--max-precond-dim",type=int,default=128)
    args=parser.parse_args()
    if min(args.steps,args.batch,args.monitor_batch)<1 or args.max_wall<=0:
        parser.error("Positive update/batch/time budgets are required")
    try:run(args)
    except Exception:
        path=Path(args.run_dir)
        if path.is_dir():
            write_json(path/"STATUS.json",dict(status="failed",pid=os.getpid(),
                       traceback=traceback.format_exc(),updated_at=utc()))
        raise


if __name__=="__main__":
    main()
