"""Regenerate the frozen uniform/layered VAL roster with the paper LWC8-4 teacher."""

from __future__ import annotations
from scno_release_paths import local_asset as _scno_local_asset

import argparse
from collections import Counter
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from time import monotonic

import torch
import h5py
import numpy as np

from data_v2 import GRID, ROOTS, _array_sha, _file_sha, _read_json, _write_json


REPO = Path(_scno_local_asset('project'))
ARCHIVE = Path.home() / "acoustic_centered_20261110/dataset_uniform500_layered434_raw801_centered"
ARCHIVED_VAL_SHA = "db1742fbfa724b55e499880f34a28e9ea226b1e2f92a19d58b07d678c5237dfb"
SOURCE = [1000., 100., 15., .1, 1.]
FAMILIES = {"uniform": 100, "layered": 25}
FIELD_BYTES = 401 * GRID[0] * GRID[1] * np.dtype(np.float32).itemsize


def review_inputs(repo, archive, train):
    """Bind the archived roster and reconstruct the exact fine and coarse media."""
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from label_quality_fd.data_generation.grid import AcousticGrid
    from label_quality_fd.data_generation.raw_dataset import NodalLWC84Solver, two_layer_velocity
    from label_quality_fd.data_generation.velocity_models_lwc84 import generate_uniform_velocity
    from inspect import getfile

    completion = _read_json(archive / "DATASET_COMPLETE.json")
    inputs = archive / "INPUT_MANIFEST.json"
    if completion["status"] != "COMPLETE" or _file_sha(inputs) != completion["checks"]["input_manifest_sha256"]:
        raise ValueError("the archived reference input manifest is not verified")
    artifacts = _read_json(inputs)["artifacts"]
    for name in ("VAL_ROSTER.json", "CONFIG.json", "time_s.npy", "normalization_ul.json"):
        if _file_sha(archive / name) != artifacts[name]["sha256"]:
            raise ValueError(f"archived reference artifact differs: {name}")
    roster_path = archive / "VAL_ROSTER.json"
    roster_sha = _file_sha(roster_path)
    if roster_sha != ARCHIVED_VAL_SHA:
        raise ValueError(f"unexpected frozen validation roster SHA: {roster_sha}")
    rows = _read_json(roster_path)["records"]
    if (dict(Counter(row["family"] for row in rows)) != FAMILIES
            or len({row["sample_id"] for row in rows}) != len(rows)
            or any(row["split"] != "validation" or row["source"] != SOURCE for row in rows)):
        raise ValueError("unexpected VAL record membership, order, or physical source")
    configuration = _read_json(archive / "CONFIG.json")["physics_config"]
    source_spec = configuration["source"]
    if (configuration["solver"]["method"] != "LWC84_CFS_CPML"
            or source_spec["kind"] != "ricker_2d_point"
            or source_spec["injection"] != "bilinear_unit_mass_nodal"
            or source_spec["pde_delta_scaling"] != "one_over_dx_dz"):
        raise ValueError("the archived wave and source operator contract differs")

    train_complete = train / "DATASET_COMPLETE.json"
    train_marker = _read_json(train_complete)
    time_view = train / "dataset_train.h5"
    if (train_marker["status"] != "COMPLETE" or train_marker["schema"] != "SCNO_CENTERED_UL1000_V1"
            or _file_sha(time_view) != train_marker["artifacts"]["dataset_train.h5"]
            or _file_sha(train / "normalization_ul.json") != artifacts["normalization_ul.json"]["sha256"]):
        raise ValueError("the archived setup and the current training release do not match")
    with h5py.File(time_view, "r") as view:
        times = view["time_field_s"][:]
        for axis in ("x_m", "z_m"):
            np.testing.assert_array_equal(view[axis][:], np.arange(201) * 10.)
    raw_times = np.load(archive / "time_s.npy", allow_pickle=False)
    np.testing.assert_array_equal(times, raw_times[::2])
    np.testing.assert_array_equal(times, np.arange(401) * .0025)
    if raw_times.shape != (801,):
        raise ValueError("the archive must span the complete native 801-frame axis")

    grid_config = {key: configuration["grid"][key] for key in
                   ("nx", "nz", "dx_m", "dz_m", "lx_m", "lz_m", "centering")}
    grid = AcousticGrid(**grid_config)
    fields = []
    for row in rows:
        if row["family"] == "uniform":
            fine, _ = generate_uniform_velocity(grid, seed=row["seed"])
            np.testing.assert_array_equal(fine, np.float32(row["velocity_mps"]))
        else:
            fine, medium = two_layer_velocity(grid, row["seed"])
            if medium["n_layers"] != row["n_layers"] or medium["n_layers"] != 2:
                raise ValueError(f"layer count recovery failed: {row['sample_id']}")
        fine = np.ascontiguousarray(fine, dtype=np.float32)
        if (_array_sha(fine) != row["source_c401_sha256"]
                or _array_sha(fine[::2, ::2]) != row["velocity_sha256"]):
            raise ValueError(f"frozen velocity hash mismatch: {row['sample_id']}")
        if row["source_model_fp32"] != np.asarray(row["source"], np.float32).astype(float).tolist():
            raise ValueError(f"physical/model amplitude or source convention differs: {row['sample_id']}")
        fields.append(fine)
    builder = repo / "scripts/rebuild_paper_uniform_layered_centered_20261110.py"
    binding = {
        "validation_roster": str(roster_path), "validation_roster_sha256": roster_sha,
        "train_release": str(train_complete), "train_release_sha256": _file_sha(train_complete),
        "reference_config_sha256": artifacts["CONFIG.json"]["sha256"],
        "reference_normalization_sha256": artifacts["normalization_ul.json"]["sha256"],
        "paper_teacher_builder": str(builder), "paper_teacher_builder_sha256": _file_sha(builder),
        "solver_code_sha256": {str(path): _file_sha(path) for path in
                               {Path(getfile(NodalLWC84Solver)), Path(getfile(NodalLWC84Solver.simulate_device))}},
        "time_axis_sha256": _array_sha(times), "spatial_output": "401 fine nodes -> 201 exact nodal stride 2",
        "source_parameters": SOURCE, "record_count": len(rows), "families": FAMILIES,
        "recorded_time_count": 401, "teacher_dt_s": configuration["time"]["dt_used_s"],
        "normalization": "physical Pa stored; reuse the original TRAIN normalization at load",
    }
    return rows, np.stack(fields), times, configuration, binding


def paper_teacher(builder, configuration, device):
    spec = importlib.util.spec_from_file_location("scno_existing_paper_teacher", builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.teacher(configuration, device=device)


def direct_teacher(configuration, times, device):
    """Use the same class and integration settings with a sparser record-time array."""
    from label_quality_fd.data_generation.grid import AcousticGrid, BoundaryConfig
    from label_quality_fd.data_generation.raw_dataset import NodalLWC84Solver

    b = configuration["boundaries"]
    boundary = BoundaryConfig(**{key: b[key] for key in (
        "top", "left", "right", "bottom", "npml", "cpml_target_reflection",
        "cpml_polynomial_order", "cpml_outside_physical_domain"
    )})
    return NodalLWC84Solver(
        grid=AcousticGrid(**{key: configuration["grid"][key] for key in
                             ("nx", "nz", "dx_m", "dz_m", "lx_m", "lz_m", "centering")}),
        boundaries=boundary, dt_s=configuration["time"]["dt_used_s"], output_times_s=times,
        c_ref_mps=configuration["solver"]["c_ref_mps"], device=device, dtype=torch.float32,
        kappa_max=b["kappa_max"], minimum_frequency_hz=b["minimum_frequency_hz"],
        output_restriction_factor=2, cuda_graphs=True,
        block_step_count=configuration["solver"]["CUDA_graph_block_steps"],
    )


def solve(solver, media, sources, device, stride):
    keys = ("source_x_m", "source_z_m", "source_f0_hz", "source_t0_s", "source_amplitude")
    start = monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        result = solver.simulate_device(media, **dict(zip(keys, sources.T)))
        labels = result.wavefield_device[:, ::stride].cpu().numpy()
    torch.cuda.synchronize(device)
    return labels, {"solve_wall_s": monotonic() - start,
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def check_pressure(pressure):
    if pressure.shape != (401, *GRID) or pressure.dtype != np.float32:
        raise ValueError("the teacher must return all 401 physical float32 pressure frames")
    if (not np.isfinite(pressure).all() or np.any(pressure[:, 0, :])
            or np.any(pressure[0]) or not np.any(pressure)):
        raise ValueError("physical pressure violates the zero free surface, initial frame, or nonzero signal checks")


def save_npy(path, array):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def save_record(out, row, index, fine, pressure, execution):
    check_pressure(pressure)
    coarse = fine[::2, ::2]
    if _array_sha(coarse) != row["velocity_sha256"]:
        raise ValueError(f"coarse velocity changed before write: {row['sample_id']}")
    p, v = (out / "arrays" / f"{index:03d}_{name}.npy" for name in ("pressure", "velocity"))
    save_npy(p, pressure)
    save_npy(v, coarse)
    pressure_sha = _array_sha(pressure)
    if (_array_sha(np.load(p, mmap_mode="r", allow_pickle=False)) != pressure_sha
            or _array_sha(np.load(v, mmap_mode="r", allow_pickle=False)) != row["velocity_sha256"]):
        raise ValueError(f"written arrays did not round-trip: {row['sample_id']}")
    record = {"record_id": row["sample_id"], "split": "validation", "family": row["family"],
              "medium_id": row["group_id"], "source_parameters": row["source"],
              "velocity_mps": str(v.relative_to(out)), "velocity_sha256": row["velocity_sha256"],
              "pressure_pa": str(p.relative_to(out)), "pressure_array_sha256": pressure_sha,
              "frozen_roster_index": row["source_HDF_row"], "execution": execution}
    _write_json(out / "receipts" / f"{index:03d}.json", record)
    return record


def verified_receipt(out, row, index):
    path = out / "receipts" / f"{index:03d}.json"
    if not path.is_file():
        return None
    rec = _read_json(path)
    pressure = np.load(out / rec["pressure_pa"], mmap_mode="r", allow_pickle=False)
    velocity = np.load(out / rec["velocity_mps"], mmap_mode="r", allow_pickle=False)
    check_pressure(pressure)
    if (rec["record_id"] != row["sample_id"] or rec["source_parameters"] != row["source"]
            or rec["family"] != row["family"] or rec["medium_id"] != row["group_id"]
            or rec["split"] != "validation" or rec["velocity_sha256"] != row["velocity_sha256"]
            or _array_sha(velocity) != row["velocity_sha256"]
            or _array_sha(pressure) != rec["pressure_array_sha256"]):
        raise ValueError(f"committed validation receipt has changed: {row['sample_id']}")
    return rec


def generate(out, rows, media, times, configuration, binding, device, max_batch):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the published pressure regeneration requires the paper CUDA teacher")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / ".build.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("arrays", "receipts"):
            (out / name).mkdir(exist_ok=True)
        binding_path = out / "INPUT_BINDING.json"
        time_path = out / "time_s.npy"
        axis_exists = time_path.is_file()
        if binding_path.exists():
            if _read_json(binding_path) != binding:
                raise ValueError("existing output belongs to different archived inputs")
            if axis_exists:
                np.testing.assert_array_equal(np.load(time_path, allow_pickle=False), times)
        else:
            if any((out / "receipts").iterdir()) or any((out / "arrays").iterdir()) or axis_exists:
                raise ValueError("refuse to use an output with unbound arrays")
            _write_json(binding_path, binding)
        if not axis_exists:
            save_npy(time_path, times)
        done = {i: rec for i, row in enumerate(rows) if (rec := verified_receipt(out, row, i)) is not None}
        required = (len(rows) - len(done)) * (FIELD_BYTES + (1 << 20)) + (1 << 30)
        if shutil.disk_usage(out).free < required:
            raise RuntimeError(f"validation generation requires {required} bytes of free output storage")
        status_path = out / "BUILD_STATUS.json"
        preflight_path = out / "SNAPSHOT_PREFLIGHT.json"
        sources = np.array([row["source"] for row in rows], dtype=np.float64)
        if len(done) == len(rows):
            publish(out, binding, rows, done, status_path)
            return
        if preflight_path.is_file():
            preflight = _read_json(preflight_path)
        else:
            probe = [next(i for i, row in enumerate(rows) if row["family"] == family) for family in FAMILIES]
            _write_json(status_path, {"stage": "SNAPSHOT_PARITY", "completed": len(done), "records": len(rows)})
            legacy = paper_teacher(binding["paper_teacher_builder"], configuration, device)
            original, original_stats = solve(legacy, media[probe], sources[probe], device, 2)
            del legacy
            torch.cuda.empty_cache()
            direct_error = None
            short_stats = None
            try:
                direct = direct_teacher(configuration, times, device)
                short, short_stats = solve(direct, media[probe], sources[probe], device, 1)
                matches = [bool(np.array_equal(a, b)) for a, b in zip(original, short)]
                del short
            except (ValueError, RuntimeError) as error:
                matches = [False] * len(probe)
                direct_error = f"{type(error).__name__}: {error}"
            finally:
                if "direct" in locals():
                    del direct
                torch.cuda.empty_cache()
            preflight = {"record_ids": [rows[i]["sample_id"] for i in probe],
                         "direct401_equals_native801_stride2": matches,
                         "chosen_snapshot_mode": "direct401" if all(matches) else "native801_stride2",
                         "original": original_stats, "direct401": short_stats, "direct_error": direct_error,
                         "cuda_device": torch.cuda.get_device_name(device)}
            for j, i in enumerate(probe):
                if i not in done:
                    done[i] = save_record(out, rows[i], i, media[i], original[j], {
                        "snapshot_mode": "native801_stride2", **original_stats,
                    })
            del original
            _write_json(preflight_path, preflight)

        use_direct = preflight["chosen_snapshot_mode"] == "direct401"
        per_sample = preflight["direct401" if use_direct else "original"]["peak_cuda_reserved_bytes"] / 2
        remaining = [i for i in range(len(rows)) if i not in done]
        while remaining:
            gc.collect()
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
            estimate = max(1, int(free * .70 / max(per_sample, 128 << 20)))
            size = min(len(remaining), estimate, max_batch)
            batch = remaining[:size]
            _write_json(status_path, {"stage": "PROPAGATING", "completed": len(done),
                                      "records": len(rows), "batch": size})
            try:
                solver = (direct_teacher(configuration, times, device) if use_direct
                          else paper_teacher(binding["paper_teacher_builder"], configuration, device))
                fields, execution = solve(solver, media[batch], sources[batch], device, 1 if use_direct else 2)
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                max_batch = min(max_batch, size // 2)
                continue
            finally:
                if "solver" in locals():
                    del solver
            execution.update(snapshot_mode=preflight["chosen_snapshot_mode"], batch_size=size)
            torch.cuda.empty_cache()
            for j, i in enumerate(batch):
                done[i] = save_record(out, rows[i], i, media[i], fields[j], execution)
                _write_json(status_path, {"stage": "COMMITTING", "completed": len(done), "records": len(rows)})
            del fields
            remaining = remaining[size:]
        publish(out, binding, rows, done, status_path)


def publish(out, binding, rows, done, status_path):
    records = [done[i] for i in range(len(rows))]
    manifest = {"schema": "SCNO_V2_FROZEN_UL_VAL_401", "split": "validation", "times_s": "time_s.npy",
                "records": records, "binding": binding}
    manifest_path = out / "VALIDATION_MANIFEST.json"
    _write_json(manifest_path, manifest)
    result = {"stage": "COMPLETE", "completed": len(records), "records": len(rows),
              "manifest": str(manifest_path), "sha256": _file_sha(manifest_path)}
    _write_json(status_path, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--train", type=Path, default=Path(ROOTS["centered_uniform_layered"]))
    parser.add_argument("--out", "--output", dest="output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    roots = [p.resolve(strict=True) for p in (args.repo, args.archive, args.train)]
    rows, media, times, configuration, binding = review_inputs(*roots)
    if args.audit_only:
        print(json.dumps({**binding, "all_velocities_match_the_frozen_roster": True}, indent=2))
        return
    if args.output is None or args.max_batch < 1:
        parser.error("provide --out and a positive --max-batch for CUDA generation")
    out = args.output.resolve()
    if any(out.is_relative_to(root) for root in roots[1:]):
        raise ValueError("write validation data to an independent directory")
    try:
        generate(out, rows, media, times, configuration, binding, args.device, args.max_batch)
    except BaseException as error:
        if out.is_dir():
            print(json.dumps({"stage": "FAILED", "error": f"{type(error).__name__}: {error}"}), flush=True)
        raise


if __name__ == "__main__":
    main()
