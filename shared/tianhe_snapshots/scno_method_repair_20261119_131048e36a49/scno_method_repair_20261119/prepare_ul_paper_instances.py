"""Package only two registered Pa observations for the two audited paper targets.

Canonical hashes identify numerical reference fields, not analytical solutions.
Rows 41/42 are the explicitly chosen NEW experiment convention; the archived
native400 trajectory-fitting runs did not supply this observation manifest.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


SCHEMA = "SCNO_UL_PAPER_TWO_OBSERVATION_INSTANCE_V1"
GRID = (201, 201)
OBSERVATION_INDICES = np.asarray([41, 42], dtype=np.int64)
TIME_SHA = "5952d2e38557085648f34f7db3f3615641f22f6d34c1968b7a3ec091ec03a524"
SOURCE_SHA = "52c9ad389c7e32e34d3175b10965c3e13fcb2c4c2e2700fbb0c0e6053233161b"
SOLVER_SOURCE_SHA = "f84a903e6e988599c1992cef35579614c78f082e2469809e420fd8fd15c062d3"
CASES = {
    "uniform4000": {
        "reference_uniform401.npy": "f63157a67c35a95f56d2f4db5128dbe1a5d7ceafb9f6db2c9ebdb4120c9a037c",
        "velocity201.npy": "b226ddf152b884a81fce40b11c811bf2f6da5acd3b77838382d8065140018847",
        "wavefield_801.h5": "bdb25476da66a2f7d0dd977b3d0fece8af0f911a5be4dc24f453262964b163ca",
    },
    "layered3000_5000_z1200": {
        "reference_uniform401.npy": "7966f41085c916dd8b3d980f800e0c8c49c0e1fb5da2a3341403da321074c3da",
        "velocity201.npy": "7466784866c89ed36c2c8b6c58945fe14700daf3f6e7cb226085d704a7a34d61",
        "wavefield_801.h5": "1ab009294fce3859b05e6b8c45c84b1d66ab100c926917febfb72a6b7ef0b51e",
    },
}
for _case in CASES.values():
    _case.update({"time_uniform401_s.npy": TIME_SHA, "source_model.npy": SOURCE_SHA,
                  "source_solver.npy": SOLVER_SOURCE_SHA})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validated_reference(reference_root, case, *, specs=None):
    """Return authenticated mmap fields and provenance; never construct labels."""
    specifications = CASES if specs is None else specs
    if case not in specifications:
        raise ValueError("choose one of the two audited paper target identities")
    root, expected = Path(reference_root).resolve() / case, specifications[case]
    files = {}
    for name, digest in expected.items():
        path = root / name
        if name == "wavefield_801.h5" and not path.is_file():
            files[name] = {"path": str(path), "sha256": digest,
                           "verified_this_run": False, "role": "recorded upstream provenance"}
            continue
        actual = sha256(path)
        if actual != digest:
            raise ValueError(f"canonical SHA-256 mismatch: {path}")
        files[name] = {"path": str(path), "sha256": actual, "verified_this_run": True}
    origin = json.loads((root / "RECEIPT.json").read_text(encoding="utf-8"))
    complete = json.loads((root / "FULL401_RECEIPT.json").read_text(encoding="utf-8"))
    if (origin.get("status") != "PASS" or origin.get("case") != case
            or origin.get("wavefield_sha256") != expected["wavefield_801.h5"]
            or complete.get("status") != "PASS" or complete.get("case") != case
            or complete.get("reference_sha256") != expected["reference_uniform401.npy"]
            or complete.get("shape") != [401, *GRID]
            or complete.get("source") != "wavefield_801.h5 frames 0,2,...,800"
            or [complete.get(k) for k in ("time_start_s", "time_end_s", "time_step_s")] != [0., 1., .0025]):
        raise ValueError("canonical source/full401 receipts contradict the reference identity")
    field = np.load(root / "reference_uniform401.npy", mmap_mode="r", allow_pickle=False)
    velocity = np.load(root / "velocity201.npy", allow_pickle=False)
    source = np.load(root / "source_model.npy", allow_pickle=False)
    solver_source = np.load(root / "source_solver.npy", allow_pickle=False)
    times = np.load(root / "time_uniform401_s.npy", allow_pickle=False)
    if field.shape != (401, *GRID) or field.dtype != np.dtype("float32"):
        raise ValueError("expected numerical Pa reference [T,Z,X]=[401,201,201], float32")
    if velocity.shape != GRID or velocity.dtype != np.dtype("float32"):
        raise ValueError("expected the authenticated physical m/s velocity201 array")
    wanted_source = np.asarray([1000., 1000., 15., .1, 1.])
    if (source.shape != (5,) or source.dtype != np.dtype("float32")
            or not np.array_equal(source, wanted_source.astype(np.float32))
            or not np.array_equal(solver_source, wanted_source)
            or not np.array_equal(times, np.arange(401, dtype=np.float64) * .0025)):
        raise ValueError("source or native physical-time axis differs from the audited target")
    expected_velocity = np.full(GRID, 4000., dtype=np.float32)
    if case == "layered3000_5000_z1200":
        expected_velocity[:120] = 3000.; expected_velocity[120:] = 5000.
    if not np.array_equal(velocity, expected_velocity):
        raise ValueError("velocity values/interface do not implement the selected paper target")
    # Exact audited downstream file hashes bind its TZX/Pa semantics. If the
    # upstream HDF is present, also inspect its metadata (without pressure I/O).
    axis_evidence = "exact audited reference SHA and FULL401 source receipt"
    if (root / "wavefield_801.h5").is_file():
        import h5py
        with h5py.File(root / "wavefield_801.h5", "r") as upstream:
            if (upstream.attrs.get("units") != "Pa" or upstream.attrs.get("wavefield_axis_order") != "TZX"
                    or upstream["wavefield"].shape != (801, *GRID)):
                raise ValueError("upstream HDF pressure axes/units differ")
        axis_evidence += "; upstream HDF TZX/Pa attributes checked"
    return field, velocity, source, times, {
        "case": case, "reference_root": str(Path(reference_root).resolve()), "files": files,
        "reference_kind": "archived_numerical_pressure", "axis_order": "TZX", "units": "Pa",
        "axis_evidence": axis_evidence, "shape": [401, *GRID],
        "full401_source": complete["source"], "time_axis_s": {"count": 401, "start": 0., "end": 1., "dt": .0025},
        "receipt_sha256": {name: sha256(root / name) for name in ("RECEIPT.json", "FULL401_RECEIPT.json")}}


def prepare_instances(reference_root, output, *, specs=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new instance output directory; existing files are preserved")
    validated = [(case, validated_reference(reference_root, case, specs=specs)) for case in CASES]
    output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for case, (field, velocity, source, times, provenance) in validated:
        observed = np.array(field[OBSERVATION_INDICES], copy=True)
        if not np.isfinite(observed).all() or not np.any(observed != 0) or np.any(observed[:, 0] != 0):
            raise ValueError("two finite, nonzero-energy Pa observations with the exact zero top are required")
        directory = output / case; directory.mkdir()
        instance = directory / "instance.npz"
        np.savez_compressed(instance, velocity_mps=velocity, source_parameters=source,
                            observed_pa=observed, observation_indices=OBSERVATION_INDICES.copy())
        manifest = {"schema": SCHEMA, **provenance, "instance": str(instance.resolve()),
                    "instance_sha256": sha256(instance), "instance_keys": ["velocity_mps", "source_parameters", "observed_pa", "observation_indices"],
                    "observation_indices": OBSERVATION_INDICES.tolist(),
                    "observation_times_s": times[OBSERVATION_INDICES].tolist(),
                    "observation_policy": "explicit new-experiment rows 41/42; not an archived sparse-observation manifest",
                    "full_reference_in_instance": False, "input_normalization": "none; observed_pa is absolute Pa",
                    "downstream_reference_sha256": provenance["files"]["reference_uniform401.npy"]["sha256"]}
        (directory / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests[case] = manifest
    return manifests


def cpu_self_check():
    """Synthetic file-contract test; fixtures are never canonical evidence."""
    with tempfile.TemporaryDirectory(prefix="ul_instance_cpu_", dir=Path.cwd()) as temporary:
        root = Path(temporary); assert root.resolve().parent == Path.cwd().resolve()
        specs = copy.deepcopy(CASES)
        for case in CASES:
            folder = root / "reference" / case; folder.mkdir(parents=True)
            pressure = np.lib.format.open_memmap(folder / "reference_uniform401.npy", mode="w+", dtype=np.float32, shape=(401, *GRID))
            pressure[:] = 0.; pressure[41, 1:, 100] = 1e-8; pressure[42, 1:, 100] = 2e-8
            pressure.flush(); del pressure
            velocity = np.full(GRID, 4000., dtype=np.float32)
            if case != "uniform4000": velocity[:120] = 3000.; velocity[120:] = 5000.
            for name, value in {"velocity201.npy": velocity, "source_model.npy": np.array([1000,1000,15,.1,1], np.float32),
                                "source_solver.npy": np.array([1000,1000,15,.1,1], np.float64),
                                "time_uniform401_s.npy": np.arange(401)*.0025}.items(): np.save(folder / name, value)
            for name in specs[case]:
                if name != "wavefield_801.h5": specs[case][name] = sha256(folder / name)
            (folder / "RECEIPT.json").write_text(json.dumps({"status":"PASS", "case":case, "wavefield_sha256":specs[case]["wavefield_801.h5"]}))
            (folder / "FULL401_RECEIPT.json").write_text(json.dumps({"status":"PASS", "case":case,
                "reference_sha256":specs[case]["reference_uniform401.npy"], "shape":[401,*GRID],
                "source":"wavefield_801.h5 frames 0,2,...,800", "time_start_s":0., "time_end_s":1., "time_step_s":.0025}))
        result = prepare_instances(root / "reference", root / "instances", specs=specs)
        for case, receipt in result.items():
            with np.load(receipt["instance"], allow_pickle=False) as saved:
                assert set(saved.files) == {"velocity_mps","source_parameters","observed_pa","observation_indices"}
                assert saved["observed_pa"].shape == (2,*GRID)
                assert np.array_equal(saved["observation_indices"], [41,42])
                assert sha256(receipt["instance"]) == receipt["instance_sha256"]
        np.save(root / "reference/uniform4000/source_model.npy", np.array([1000,1000,20,.1,1], np.float32))
        try: prepare_instances(root / "reference", root / "must_not_exist", specs=specs)
        except ValueError: pass
        else: raise AssertionError("tampered source was accepted")
        assert not (root / "must_not_exist").exists()
    return {"passed": True, "device": "cpu", "fixture_only": True, "two_targets_four_keys_two_rows": True,
            "absolute_pa_preserved": True, "tamper_rejected_before_output": True, "canonical_data_used": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(cpu_self_check(), indent=2))
    else:
        if args.reference_root is None or args.output is None: parser.error("--reference-root and --output are required")
        result = prepare_instances(args.reference_root, args.output)
        print(json.dumps({case: {"instance": r["instance"], "sha256": r["instance_sha256"]} for case, r in result.items()}, indent=2))
