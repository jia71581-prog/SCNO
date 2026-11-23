"""Fit and then independently score two ordinary zero-observation Marmousi shots."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp
from time import perf_counter

import numpy as np
import torch


FIT_SCHEMA = "marmousi_sourced_pressure_fit_cases_v1"
CASE_SCHEMA = "marmousi_sourced_pressure_fit_shot_v1"
SCORE_SCHEMA = "marmousi_sourced_pressure_archived_b5_score_v1"
NATIVE_SLOTS = np.arange(801, dtype=np.int64)
WINDOWS = ((0, 100), (100, 200), (200, 400), (400, 801))


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def forbid_waveforms_during_fit(dataset, output):
    """Allow only registered geometry and velocity arrays during the fit process."""
    from tools.marmousi_target_dataset_refined.common import read_json

    dataset, output = dataset.resolve(), output.resolve()
    contract = read_json(dataset / "acquisition_contract.json")
    arrays = {(dataset / item["path"]).resolve() for item in contract["shared_arrays"].values()}
    arrays |= {Path(contract["templates"][key + "_file"]).resolve() for key in ("truth", "initial")}
    waveform_array_formats = {".npy", ".npz", ".h5", ".hdf", ".hdf5", ".segy", ".sgy", ".zarr"}
    proof = {"guard": "allowlist of registered geometry, bound input velocities and produced B5 files",
             "waveform_or_full_field_label_opens": 0}

    def check(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        supplied = Path(os.fsdecode(args[0]))
        if supplied.suffix.lower() not in waveform_array_formats:
            return
        path = supplied.resolve()
        if path in arrays or (path.name == "B5.npz" and path.is_relative_to(output)):
            return
        proof["waveform_or_full_field_label_opens"] += 1
        raise RuntimeError(f"An array outside the zero-observation fit inputs was requested: {path}")

    sys.addaudithook(check)
    return arrays, proof


def cases_for_release(data):
    from audit_native_source_moment import select_cases

    return [
        {**{key: row[key] for key in ("model_id", "parent_group_id", "family", "split", "velocity_sha256")},
         "shot_id": shot, "source": data.sources[shot].tolist(), "key": f"{row['model_id']}_shot{shot:02d}"}
        for row, shot in select_cases(data)
    ]


def configure_device(device):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)


@torch.no_grad()
def gather(adapter, record_block):
    chunks = {"before": [], "after": []}
    for first in range(0, len(NATIVE_SLOTS), record_block):
        slots = list(range(first, min(first + record_block, len(NATIVE_SLOTS))))
        before = adapter.base_pressure(slots)
        if adapter.adaptation.receipt["accepted"]:
            after = adapter.pressure(slots)
        else:
            after = before
        for key, pressure in (("before", before), ("after", after)):
            require(not bool(pressure[:, 0].count_nonzero()) and
                    (first > 0 or not bool(pressure[:3].count_nonzero())), "Free surface or initial pressure changed")
            value = adapter.model.observe(pressure)
            require(value.dtype == torch.float32 and value.shape == (len(slots), 201) and bool(value.isfinite().all()),
                    "Invalid normalized B5 projection")
            chunks[key].append(value.cpu().numpy())
    return {key: np.concatenate(items) for key, items in chunks.items()}


def validate_case(path, case, plan_sha, times):
    from sourced_pressure_fit import SourcedPressureFit
    from tools.marmousi_target_dataset_refined.common import read_json, sha_file

    require(path.is_dir() and not path.is_symlink(), "Expected an atomically committed shot directory")
    receipt_path, prediction_path = path / "RECEIPT.json", path / "B5.npz"
    require(all(item.is_file() and not item.is_symlink() for item in (receipt_path, prediction_path)),
            "The sealed shot must contain its prediction and coefficient files")
    receipt = read_json(receipt_path)
    require(receipt["schema"] == CASE_SCHEMA and receipt["status"] == "SEALED"
            and receipt["plan_sha256"] == plan_sha and receipt["case"] == case
            and receipt["label_waveforms_used"] == receipt["numerical_wave_propagations"] == 0
            and receipt["prediction_sha256"] == sha_file(prediction_path), "Shot receipt or checksum differs")
    adaptation = receipt["sourced_coefficients"]
    require(adaptation["observed_or_reference_waveforms_used"] is False and adaptation["time_steps_advanced"] == 0
            and adaptation["source_x_z_f0_delay_amp"] == case["source"]
            and adaptation["times_s"] == times.tolist(), "Invalid zero-observation coefficients for this source")
    SourcedPressureFit.from_receipt(adaptation)
    with np.load(prediction_path, allow_pickle=False) as arrays:
        require(set(arrays.files) == {"before", "after", "times_s", "source"}
                and np.array_equal(arrays["times_s"], times)
                and np.array_equal(arrays["source"], case["source"]), "B5 time or source axis changed")
        values = {key: arrays[key] for key in ("before", "after")}
    require(all(p.shape == (801, 201) and p.dtype == np.float32 and np.isfinite(p).all() and not p[:3].any()
                for p in values.values()), "The sealed B5 prediction does not cover every native frame")
    require(adaptation["accepted"] or np.array_equal(values["before"], values["after"]),
            "A rejected correction must preserve the sealed baseline")
    return {"case": case, "receipt_sha256": sha_file(receipt_path), "prediction_sha256": receipt["prediction_sha256"]}, receipt, values


def commit_case(root, case, plan_sha, timed_fit, timed_prediction, coefficients, waveforms, times):
    from tools.marmousi_target_dataset_refined.common import atomic_json, sha_file, sync_directory

    stage = Path(mkdtemp(prefix=f".{case['key']}.partial-", dir=root))
    final = root / case["key"]
    prediction = stage / "B5.npz"
    with prediction.open("wb") as handle:
        np.savez(handle, **waveforms, times_s=times, source=np.asarray(case["source"], dtype=np.float64))
        handle.flush()
        os.fsync(handle.fileno())
    atomic_json(stage / "RECEIPT.json", {
        "schema": CASE_SCHEMA, "status": "SEALED", "plan_sha256": plan_sha, "case": case,
        "sourced_coefficients": coefficients, "label_waveforms_used": 0, "numerical_wave_propagations": 0,
        "prediction_sha256": sha_file(prediction), "prediction_shape": [801, 201],
        "prediction_units": "frozen TRAIN RMS normalized B5 projected from full native pressure",
        "timing_s": {"source_equation_and_coefficients": timed_fit, "both_b5_records": timed_prediction},
    })
    validate_case(stage, case, plan_sha, times)
    require(not final.exists(), "The requested shot already appeared in this run")
    os.rename(stage, final)
    sync_directory(root)
    return validate_case(final, case, plan_sha, times)[0]


def fit(args):
    import marmousi_sourced_adapter
    import source_forcing_moment
    import sourced_pressure_fit
    from audit_native_source_moment import validate_velocity
    from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files, native_only
    from marmousi_sourced_adapter import MarmousiSourcedReadout
    from no_pde_native_pressure import load_neural_pressure_core
    from tianhe_observations import CommittedMarmousi
    from tools.marmousi_target_dataset_refined.common import atomic_json, digest, read_json, sha_file

    output = args.output.resolve()
    require(not output.is_relative_to(args.dataset.resolve()) and output.exists() == args.resume,
            "Use a new directory, or --resume for an existing fit outside the release")
    integrity = forbid_exact_label_files(args.dataset)
    arrays, label_guard = forbid_waveforms_during_fit(args.dataset, output)
    data = CommittedMarmousi(args.dataset)
    cases = cases_for_release(data)
    for case in cases:
        arrays.add((data.root / data.rows[case["model_id"]]["velocity_path"]).resolve(strict=True))
    configure_device(args.device)
    started = perf_counter()
    with native_only(), torch.inference_mode():
        model = load_neural_pressure_core(data.root, args.base_checkpoint, args.checkpoint, args.selection,
                                         args.native_checkpoint, device=args.device).requires_grad_(False).eval()
        require(model.identity["dataset_binding"] == data.binding and model.receiver_rms == data.rms,
                "The frozen pressure reader or its scale changed")
        plan = {"schema": FIT_SCHEMA, "release_binding": data.binding, "cases": cases, "pressure_model": model.identity,
                "query_block": args.query_block, "record_block": args.record_block, "device": args.device,
                "source_radius_startup_and_boundary": "known frozen source, hard zero initial samples and free surface",
                "label_waveforms_allowed": False, "numerical_wave_propagations_allowed": False,
                "reader_source_sha256": {Path(path).name: sha_file(path) for path in (
                    __file__, marmousi_sourced_adapter.__file__, sourced_pressure_fit.__file__, source_forcing_moment.__file__)}}
        plan_sha = digest(plan)
        plan_path, shot_root = output / "PLAN.json", output / "cases"
        if args.resume:
            require(read_json(plan_path) == plan, "Resume requires the exact same freeze, devices and inputs")
        else:
            output.mkdir(parents=True, exist_ok=False)
            atomic_json(plan_path, plan)
        shot_root.mkdir(exist_ok=True)
        sealed = []
        for case in cases:
            final = shot_root / case["key"]
            if final.exists():
                sealed.append(validate_case(final, case, plan_sha, data.times)[0])
                continue
            velocity = validate_velocity(data, data.rows[case["model_id"]])
            adapter = MarmousiSourcedReadout(model, velocity, case["shot_id"], case["source"])
            start = perf_counter()
            coefficients = adapter.fit(query_block=args.query_block)
            elapsed = perf_counter() - start
            start = perf_counter()
            predictions = gather(adapter, args.record_block)
            sealed.append(commit_case(shot_root, case, plan_sha, elapsed, perf_counter()-start,
                                      coefficients, predictions, data.times))
            print(json.dumps({"stage": "SHOT_SEALED", "case": case["key"], "coefficient_update_accepted": coefficients["accepted"],
                              "fit_equations": coefficients["equations"]["fit"]["applied"]["total"],
                              "probe_equations": coefficients["equations"]["probe"]["applied"]["total"]}), flush=True)
            del adapter
    complete = {"status": "FIT_COMPLETE_BEFORE_SCORING", "schema": FIT_SCHEMA, "plan_sha256": plan_sha,
                "sealed_shots": sealed, "all_801_b5_frames_sealed": True, "label_waveforms_used": 0,
                "numerical_wave_propagations": 0, "exact_template_integrity_only": integrity,
                "array_access_guard": label_guard, "elapsed_s": perf_counter() - started}
    fit_path = output / "FIT.json"
    if fit_path.exists():
        prior = read_json(fit_path)
        require(prior["plan_sha256"] == plan_sha and prior["sealed_shots"] == sealed
                and prior["status"] == complete["status"], "Existing fit completion receipt differs")
    else:
        atomic_json(fit_path, complete)
    print(json.dumps({"stage": "FIT_COMPLETE_BEFORE_SCORING", "output": str(fit_path), "shots": len(sealed)}), flush=True)


def terms(prediction, target):
    p, y = prediction.astype(np.float64), target.astype(np.float64)
    return np.array([np.sum((p-y)**2), np.sum(y*y), np.sum(p*p), np.sum(p*y)])


def metrics(x):
    error, energy, prediction, cross = map(float, x)
    return {"relative_l2": (error / energy)**.5 if energy > 0 else None,
            "reference_energy": energy, "squared_error": error,
            "signed_cosine": float(np.clip(cross / (energy*prediction)**.5, -1., 1.)) if energy*prediction > 0 else None}


def score(args):
    from evaluate_no_pde_sensitivity_v2 import forbid_exact_label_files
    from tianhe_observations import CommittedMarmousi
    from tools.marmousi_target_dataset_refined.common import atomic_json, digest, read_json, sha_file

    run = args.fit_run.resolve(strict=True)
    output = args.output.resolve()
    require(not output.exists() and not output.is_relative_to(args.dataset.resolve()), "Report to a fresh path outside the release")
    if output.is_relative_to(run):
        require(output.parent == run and output.name not in ("FIT.json", "PLAN.json"),
                "Keep the independent report outside the sealed shot data")
    integrity = forbid_exact_label_files(args.dataset)
    data = CommittedMarmousi(args.dataset)
    fit_path, plan_path = run / "FIT.json", run / "PLAN.json"
    fit_receipt, plan = read_json(fit_path), read_json(plan_path)
    plan_sha = digest(plan)
    require(fit_receipt["schema"] == plan["schema"] == FIT_SCHEMA
            and fit_receipt["status"] == "FIT_COMPLETE_BEFORE_SCORING" and fit_receipt["plan_sha256"] == plan_sha
            and fit_receipt["all_801_b5_frames_sealed"] is True and fit_receipt["label_waveforms_used"] == 0
            and fit_receipt["numerical_wave_propagations"] == 0 and plan["release_binding"] == data.binding
            and plan["label_waveforms_allowed"] is False and plan["numerical_wave_propagations_allowed"] is False
            and plan["cases"] == cases_for_release(data), "Complete zero-observation sealing is required before any label access")
    verified = [validate_case(run / "cases" / case["key"], case, plan_sha, data.times) for case in plan["cases"]]
    require([entry[0] for entry in verified] == fit_receipt["sealed_shots"], "Sealed shot hashes differ from the fit receipt")

    results = []
    totals = {key: np.zeros((len(WINDOWS), 4), dtype=np.float64) for key in ("before", "after")}
    for _, receipt, forecasts in verified:
        case = receipt["case"]
        key, sid = case["model_id"], case["shot_id"]
        index = data.keys["train"].index(key)
        row_slot = data.rosters[key].index(sid)
        record = int(data.offsets["train"][index]) + row_slot
        directory = data.root / "models" / key
        meta = read_json(directory / "meta.json")
        reference_sha = sha_file(directory / "receivers.npy")
        require(reference_sha == meta["receivers_file_sha256"], "Ordinary archived B5 checksum changed")
        item = data.sample("train", record, NATIVE_SLOTS)
        require(item["model_id"] == key and item["shot_id"] == sid, "Scoring accessed a different recorded source")
        label = item["target"].cpu().numpy()
        require(label.shape == (801, 201) and np.isfinite(label).all(), "Invalid archived ordinary shot B5")
        by_method = {}
        for name, waveforms in forecasts.items():
            values = np.stack([terms(waveforms[a:b], label[a:b]) for a, b in WINDOWS])
            totals[name] += values
            by_method[name] = {"all_801": metrics(values.sum(0)), "by_window": [metrics(x) for x in values]}
        results.append({"case": case, "correction_accepted_on_equations": receipt["sourced_coefficients"]["accepted"],
                        "archived_receiver_file_sha256": reference_sha, "archived_b5_scores": by_method})
    pooled = {name: {"all_801": metrics(values.sum(0)), "by_window": [metrics(x) for x in values]}
              for name, values in totals.items()}
    report = {"status": "COMPLETE", "schema": SCORE_SCHEMA, "fit_receipt_sha256": sha_file(fit_path),
              "fit_plan_sha256": plan_sha, "release_binding": data.binding, "ordinary_split": "train",
              "reference_waveforms_read_after_seal": len(results), "labels_used_for_fit_or_selection": False,
              "exact_template_integrity_only": integrity, "numerical_wave_propagations": 0,
              "comparison": "hard-IC native E12 projected to B5 before/after the closed-form sourced correction",
              "time_window_bounds_s": [[0., .5], [.5, 1.], [1., 2.], [2., 4.]],
              "last_window_includes_final_endpoint": True, "cases": results, "pooled": pooled}
    atomic_json(output, report)
    print(json.dumps({"stage": "SCORE_COMPLETE", "output": str(output),
                      "relative_l2": {name: value["all_801"]["relative_l2"] for name, value in pooled.items()}}), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    fit_cli, score_cli = commands.add_parser("fit"), commands.add_parser("score")
    for name in ("dataset", "base-checkpoint", "checkpoint", "selection", "native-checkpoint", "output"):
        fit_cli.add_argument("--" + name, type=Path, required=True)
    fit_cli.add_argument("--device", default="cuda:0")
    fit_cli.add_argument("--query-block", type=int, default=8)
    fit_cli.add_argument("--record-block", type=int, default=16)
    fit_cli.add_argument("--resume", action="store_true")
    for name in ("dataset", "fit-run", "output"):
        score_cli.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "fit":
        require(1 <= args.query_block <= 32 and 3 <= args.record_block <= 64, "Choose bounded fit and record query blocks")
        fit(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
