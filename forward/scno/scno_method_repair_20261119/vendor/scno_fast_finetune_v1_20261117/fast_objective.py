"""Four-term output-only fine-tuning using the registered LWC/AIS functions."""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from fast_scno import POLICY, FrozenQueryCache, load_spectral_checkpoint, sha256


PACKAGED_OPERATORS = Path(__file__).resolve().parents[1] / "residual_support" / "operators.py"


def load_residual_operators(path):
    path = PACKAGED_OPERATORS if path is None else Path(path).resolve()
    spec = importlib.util.spec_from_file_location("scno_registered_lwc_ais", path)
    operators = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operators)
    return operators


def _stop_reason(completed_updates, step_limit, elapsed, budget, diagnosis):
    if diagnosis is not None and diagnosis["threshold_satisfied"]:
        return "threshold_met"
    if elapsed >= budget:
        return "budget_exhausted"
    if completed_updates == step_limit:
        return "max_updates_reached"
    return None


def loss_observation(prediction, observed, energy_floor):
    p, y = prediction.double(), observed.double()
    return (p - y).square().sum() / y.square().sum().clamp_min(energy_floor)


def loss_initial(pressure, pressure_t, frequency):
    return (pressure.double().square()
            + (pressure_t.double() / (2 * math.pi * frequency)).square()).mean()


def loss_bc(final_pressure):
    """Existing post-projection normalized top-row mean-square operator."""
    if final_pressure.ndim != 3 or not final_pressure.numel():
        raise ValueError("boundary pressure must contain [time,z,x] fields")
    return final_pressure[:, 0, :].double().square().mean()


def loss_lwc(operators, sampler, plan, fields, velocity, norm, delta, *, return_residual=False):
    n = fields.shape[0] // 3
    r = operators.lwc_residual(*fields.split(n), velocity, h=10., delta=delta)
    energies = sampler.tile_energies(r, norm)
    loss = sampler.loss(energies, plan)
    if return_residual:
        return loss, energies, r.detach()
    return loss, energies


class Timings:
    def __init__(self, device):
        self.device = torch.device(device)
        self.started = perf_counter()
        self.seconds = {name: 0. for name in
                        ("instance_setup", "updates", "proposal_refresh", "full_grid_monitoring",
                         "transfers")}

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        self.sync()
        started = perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.seconds[name] += perf_counter() - started

    def elapsed(self):
        self.sync()
        return perf_counter() - self.started

    def receipt(self, cache):
        totals = self.seconds.copy()
        for key, value in cache.phase.items():
            totals[key] = value if key != "transfers" else totals[key] + value
        return {"total_s": self.elapsed(), "phases_s": totals,
                "cold_model_load_in_total": False}


def build_optimizer(model, config):
    values = config["optimizer"]
    if values["name"] != "AdamW":
        raise ValueError("SCNO fast v1 uses a fresh AdamW optimizer")
    peak, minimum = values["lr_peak"], values["lr_min"]
    warmup, horizon = values["warmup_updates"], values["cosine_horizon"]
    if not 0 <= minimum <= peak or peak <= 0 or not 0 < warmup < horizon:
        raise ValueError("invalid learning-rate schedule")
    optimizer = torch.optim.AdamW(model.assert_parameter_whitelist(), lr=peak,
        weight_decay=values["weight_decay"], betas=tuple(values["betas"]), eps=values["eps"])
    model.assert_parameter_whitelist(optimizer)

    def lr_multiplier(index):
        j = index + 1
        if j <= warmup:
            return j / warmup
        step = min(j, horizon)
        return (minimum + (peak - minimum) * .5
                * (1 + math.cos(math.pi * (step - warmup) / (horizon - warmup)))) / peak

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def restore_readout(model, payload):
    if (payload["policy_id"] != POLICY
            or payload["frozen_checkpoint_identity"] != model.fast_checkpoint_identity):
        raise ValueError("the readout update requires the matching inherited spectral checkpoint")
    leaves = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if leaves.keys() != payload["trainable"].keys():
        raise ValueError("saved output tensors differ from the current parameter registry")
    for name, delta in payload["trainable"].items():
        if leaves[name].shape != delta.shape or leaves[name].dtype != delta.dtype:
            raise ValueError(f"saved output tensor shape or dtype changed: {name}")
    with torch.no_grad():
        for name, value in payload["trainable"].items():
            leaves[name].copy_(value)


def sampler_state_for_save(sampler):
    # Keep sampler arrays loadable with the same weights-only reader as the readouts.
    return {name: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for name, value in sampler.state_dict().items()}


def stencil_times(times, delta):
    return torch.cat((times - delta, times, times + delta))


def objective_parts(model, operators, sampler, plan, query, initial, *, observed,
                    obs_times, centers, velocity, nu_squared, frequency, cfg,
                    return_residual=False):
    """Build equivalent cached and full-graph objectives from one borrowed plan."""
    selected = centers[torch.as_tensor(plan["time_indices"], device=centers.device)]
    fields = query(torch.cat((obs_times, stencil_times(selected, cfg["delta_s"]))))
    obs = fields[:len(obs_times)]
    u0, ut0 = initial()
    residual_data = loss_lwc(operators, sampler, plan, fields[len(obs_times):],
                             velocity, nu_squared, cfg["delta_s"],
                             return_residual=return_residual)
    lwc, energies = residual_data[:2]
    if cfg["bc_time_support"] != "observation_times":
        raise ValueError("Fast v1 BC uses the two already-evaluated observation times")
    parts = {"obs": loss_observation(obs, observed, cfg["observation_energy_floor"]),
             "ic": loss_initial(u0, ut0, frequency), "lwc": lwc, "bc": loss_bc(obs)}
    if return_residual:
        return parts, energies, residual_data[2]
    return parts, energies


def weighted_objective(parts, cfg):
    return sum(cfg["weights"][name] * value for name, value in parts.items())


@torch.no_grad()
def full_grid_monitor(model, operators, cache, timer, centers, velocity, sampler, nu,
                      frequency, cfg, *, initialize=False):
    n_fail = n_total = 0
    energies = []
    observe_threshold = cfg.get("proposal", "energy") == "threshold"
    exceedances = []
    for times in centers.split(cfg["monitor_batch"]):
        pack = cache.packs(stencil_times(times, cfg["delta_s"]))
        with timer.measure("full_grid_monitoring"):
            fields = model.decode_finetune_query(pack)
            residual = operators.lwc_residual(*fields.split(len(times)), velocity,
                                               h=10., delta=cfg["delta_s"])
            diag = operators.full_grid_diagnostics(residual, sampler.mask,
                f0=frequency, epsilon_r=cfg["epsilon_r"], epsilon_p=cfg["epsilon_p"])
            n_fail += diag["failure_count"]
            n_total += diag["units"]
        if initialize or observe_threshold:
            with timer.measure("proposal_refresh"):
                if initialize:
                    energies.append(sampler.tile_energies(residual, nu).cpu().numpy())
                if observe_threshold:
                    exceedances.append(sampler.tile_exceedances(
                        residual, frequency, cfg["epsilon_r"]).cpu().numpy())
    if n_total != len(centers) * sampler.n_valid:
        raise AssertionError("an incomplete grid cannot be used for threshold stopping")
    if initialize or observe_threshold:
        with timer.measure("proposal_refresh"):
            if initialize:
                sampler.initialize_history(np.concatenate(energies))
            if observe_threshold:
                sampler.refresh_exceedances(np.concatenate(exceedances))
                if abs(sampler.last_failure_probability - n_fail/n_total) > 1e-12:
                    raise AssertionError("threshold history must reproduce the full-grid failure fraction")
    return {"centers": len(centers), "valid_nodes_per_center": sampler.n_valid,
            "failure_count": n_fail, "units": n_total,
            "failure_probability": n_fail/n_total,
            "threshold_satisfied": n_fail/n_total < cfg["epsilon_p"]}


def finetune_instance(model, operators, c201_mps, source, observed_pa, config):
    """Run the configured objective on resident weights; labels are exactly two snapshots."""
    timer = Timings(model.device)
    cfg = config["objective"]
    proposal = cfg.get("proposal", "energy")
    if proposal not in ("energy", "threshold"):
        raise ValueError("proposal must be energy or threshold")
    threshold = proposal == "threshold"
    if threshold and not hasattr(operators.AISSampler, "refresh_exceedances"):
        raise ValueError("threshold sampling requires the bundled residual operators")
    budget = config["wall_time_budget_s"]
    steps = config["max_updates"]
    checks = set(config["monitor_updates"])
    if (not math.isfinite(budget) or budget <= 0 or type(steps) is not int or steps < 1
            or 0 not in checks or steps not in checks
            or any(type(i) is not int or i < 0 or i > steps for i in checks)):
        raise ValueError("positive budget and valid monitor milestones including zero and the end are required")
    if set(cfg["weights"]) != {"obs", "ic", "lwc", "bc"} or cfg["weights"]["bc"] != .01:
        raise ValueError("the four physics terms must be configured, with registered BC weight .01")
    if any(not math.isfinite(w) or w < 0 for w in cfg["weights"].values()):
        raise ValueError("objective weights must be finite and nonnegative")
    if config["residual_axis"] != "registered400_clamp_last":
        raise ValueError("registered 400-center residual support required")
    if (cfg["delta_s"] != .0005 or cfg["epsilon_r"] != .25 or cfg["epsilon_p"] != .01
            or cfg["observation_times_s"] != [.1025, .105]
            or cfg["bc_time_support"] != "observation_times"):
        raise ValueError("v1 requires the registered stencil, thresholds, and two observation times")
    with timer.measure("transfers"):
        cv, fv = model._inputs(c201_mps, source)
        observed_pa = torch.as_tensor(observed_pa, dtype=torch.float32, device=model.device)
    with timer.measure("instance_setup"):
        if observed_pa.shape != (2, 201, 201) or not bool(torch.isfinite(observed_pa).all()):
            raise ValueError("the only supplied labels must be the two finite, 201×201 Pa snapshots")
        observed = model.normalizer.encode_pressure(observed_pa[None], fv[:, 4])[0].detach()
        obs_times = model._times(cfg["observation_times_s"])
        if len(obs_times) != 2 or not bool(obs_times[0] < obs_times[1]):
            raise ValueError("two ordered observation times are required")
        frequency = float(fv[0, 2])
        nu = operators.residual_scale_squared(observed, f0=frequency, eps=cfg["pde_energy_floor"])
        centers = model.full_times_s.detach().clone().clamp_max(1 - cfg["delta_s"])
        if centers.numel() != 400 or not bool((centers[1:] > centers[:-1]).all()):
            raise AssertionError("unexpected registered full-grid monitoring axis")
        threshold_args = {"epsilon_p": cfg["epsilon_p"]} if threshold else {}
        sampler = operators.AISSampler(operators.make_valid_mask(
            shape=tuple(cv.shape[-2:]), source=fv[0, :2].cpu().tolist(), halo=8, radius=90.),
            n_times=400, tile_size=cfg["tile_size"], beta_uniform=cfg["beta_uniform"],
            ema=cfg["ema"], seed=config["sampling_seed"], **threshold_args)
        optimizer, scheduler = build_optimizer(model, config)
        cache = FrozenQueryCache(model, cv, fv, **config["cache"])
    velocity = cv[0, 0]

    monitor = []
    updates = 0
    status = None
    history = []
    while True:
        result = None
        if updates in checks:
            result = full_grid_monitor(model, operators, cache, timer, centers, velocity,
                                         sampler, nu, frequency, cfg, initialize=(updates == 0))
            monitor.append({"update": updates, **result})
        status = _stop_reason(updates, steps, timer.elapsed(), budget, result)
        if status is not None:
            break
        with timer.measure("proposal_refresh"):
            plan = sampler.draw(mode="threshold" if threshold else "ais", n_time_draws=cfg["n_time_draws"],
                                n_tile_draws=cfg["n_tile_draws"])
        selected = centers[torch.as_tensor(plan["time_indices"], device=centers.device)]
        pack = cache.packs(torch.cat((obs_times, stencil_times(selected, cfg["delta_s"]))))
        initial = cache.initial_pack()
        if timer.elapsed() >= budget:
            sampler.rng.bit_generator.state = plan["rng_state_before"]
            status = "budget_exhausted"
            break
        with timer.measure("updates"):
            optimizer.zero_grad(set_to_none=True)
            objective = objective_parts(model, operators, sampler, plan,
                lambda times: model.decode_finetune_query(pack),
                lambda: model.decode_finetune_query_jvp(*initial),
                observed=observed, obs_times=obs_times, centers=centers, velocity=velocity,
                nu_squared=nu, frequency=frequency, cfg=cfg, return_residual=threshold)
            parts, energies = objective[:2]
            loss = weighted_objective(parts, cfg)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite four-term objective")
            loss.backward()
            params = model.assert_parameter_whitelist(optimizer)
            if any(p.grad is None for p in params):
                raise RuntimeError("an output tensor is disconnected from the fine-tuning objective")
            torch.nn.utils.clip_grad_norm_(params, config["gradient_clip"], error_if_nonfinite=True)
            history.append({"update": updates + 1, "lr": optimizer.param_groups[0]["lr"],
                            "loss": float(loss.detach()),
                            "components": {k: float(v.detach()) for k, v in parts.items()}})
            optimizer.step()
            scheduler.step()
        with timer.measure("proposal_refresh"):
            sampler.commit(plan, energies.detach())
            if threshold:
                fractions = sampler.tile_exceedances(objective[2], frequency, cfg["epsilon_r"])
                sampler.refresh_exceedances(fractions, times=plan["time_indices"])
        updates += 1

    timing = timer.receipt(cache)
    receipt = model.finetune_receipt(optimizer)
    receipt.update(
        stop_reason=status, completed_updates=updates, wall_time_budget_s=budget,
        within_wall_time_budget=timing["total_s"] <= budget,
        timing=timing, cache=cache.receipt(), monitor=monitor, update_history=history,
        schedule=config["optimizer"], objective=cfg, sampling_proposal=proposal,
        pressure_unit_pa=float(model.normalizer.decode_pressure(
            torch.ones((1, 1, 1, 1), device=cv.device), fv[:, 4]).item()),
        observation_times_s=obs_times.tolist(),
        residual_centers_s=centers.tolist(), proposal_history_initialization="initial full-grid tile energies")
    training_state = {
        "policy_id": POLICY, "frozen_checkpoint_identity": model.fast_checkpoint_identity,
        "trainable": {name: p.detach().cpu().clone()
                      for name, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "sampler": sampler_state_for_save(sampler), "completed_updates": updates,
    }
    return receipt, training_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instance", type=Path, required=True,
                        help="NPZ with c201_mps, source, observation_times_s and observed_pa")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.out.exists():
        raise FileExistsError("choose a new output directory")
    operators = load_residual_operators(cfg.get("residual_operator_path"))
    start = perf_counter()
    model = load_spectral_checkpoint(cfg, args.device)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    cold_load = perf_counter() - start
    with np.load(args.instance, allow_pickle=False) as f:
        if not np.allclose(f["observation_times_s"], cfg["objective"]["observation_times_s"],
                           rtol=0, atol=1e-8):
            raise ValueError("observed pressure labels are not identified with the configured times")
        receipt, state = finetune_instance(
            model, operators, f["c201_mps"], f["source"], f["observed_pa"], cfg)
    receipt["cold_checkpoint_load_s"] = cold_load
    receipt["residual_operator_sha256"] = sha256(operators.__file__)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(state, args.out / "fast_readout.pt")
    (args.out / "RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
