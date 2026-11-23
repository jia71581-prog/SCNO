# Training and fine-tuning workflows

Commands below start at the repository root in an activated Linux Python environment. The launcher sets the repository import paths for the selected workflow. `python run.py show NAME` identifies the source file. `python run.py run NAME -- --help` calls its own argument parser. To select a different interpreter, use `python run.py run --python /path/to/python NAME -- --help`.

Some archived experiments read a fixed arrangement of external inputs under `local_assets/` at the repository root. To place that tree elsewhere, set `SCNO_ASSET_ROOT` to the directory that directly contains the expected subdirectories, such as `project/`, `package/` or `datasets/`. For JSON training configurations, copy the template and set its data paths explicitly, as in the example below. Choose a separate writable root for new outputs.

## Train and resume SCNO

`scno-train` consumes a registered pressure corpus. The `centered_uniform_layered` reader expects the 1,000-record source corpus, its HDF5 view, underlying pressure records and manifests. A separate validation manifest identifies the validation media and their files. The `style_marm_joint201` reader obtains all three splits and the training normalization from its own archive. Its `production_marmousi` configuration and the separate paired-scattering experiment below use different data contracts.

| V2 corpus | Space | Saved pressure times | Identifying files |
| --- | --- | --- | --- |
| `centered_uniform_layered` | 201 × 201, 10 m | 401 over 0–1 s, 2.5 ms apart | `DATASET_COMPLETE.json`, `RECORDS.json`, `dataset_train.h5`, `normalization_ul.json` and the referenced sources |
| `style_marm_joint201` | 201 × 201, 10 m | 801 over 0–4 s, 5 ms apart | `DATASET_COMPLETE.json`, `RECORDS.json`, `SPLITS.json`, `dataset_{train,validation,test}.h5`, `normalization_stats_train.json` and the referenced sources |

The HDF5 view exposes `pressure201[record,time,z,x]` in Pa, `velocity201[record,z,x]` in m/s and `source_parameters[record,5]` in the order `[x_m, z_m, frequency_Hz, delay_s, amplitude]`. Pressure is float32. The reader checks the registered record order, physical axes and artifact hashes. Keep the native files named by each manifest accessible. The complete layout and split checks are in the [V2 data reader](../forward/scno/scno_method_repair_20261119/data_v2.py).

The following example uses a uniform/layered corpus already available locally. Set `SCNO_ASSET_ROOT` and `SCNO_RUNS` to your own data and output roots. In this example, `centered-ul/` holds the registered corpus, and `ul-validation/` holds its separately prepared validation manifest and referenced data. Only the configuration directory is created beforehand. The trainer creates the run directory itself.

```bash
export SCNO_ASSET_ROOT="$HOME/scno-inputs"
export SCNO_RUNS="$HOME/scno-runs"
mkdir -p "$SCNO_RUNS/config"

python - <<'PY'
import json
import os
from pathlib import Path

assets = Path(os.environ["SCNO_ASSET_ROOT"]).resolve()
runs = Path(os.environ["SCNO_RUNS"]).resolve()
templates = Path("forward/scno/scno_method_repair_20261119")
config = json.loads((templates / "production_uniform_layered_native201_200e.json").read_text())
config["data"]["root"] = str(assets / "centered-ul")
config["data"]["validation_manifest"] = str(assets / "ul-validation" / "VALIDATION_MANIFEST.json")
with (runs / "config" / "layered-01.json").open("x") as stream:
    json.dump(config, stream, indent=2)
PY

python run.py run scno-train -- \
  --config "$SCNO_RUNS/config/layered-01.json" \
  --run "$SCNO_RUNS/layered-01" --device cuda:0 --stop-after 1
```

The first run computes training-record pressure energies, which may involve reading full trajectories. A successful first update writes `checkpoint.pt`, `STATUS.json`, `RUN_BINDING.json`, `TRAIN_ENERGY.json` and `train.jsonl` under the new run. Read `completed_updates` in `STATUS.json`. Continue from that checkpoint with the same configuration, corpus, device type and number of processes.

```bash
python run.py run scno-train -- \
  --config "$SCNO_RUNS/config/layered-01.json" \
  --run "$SCNO_RUNS/layered-01" --device cuda:0 --resume
```

To initialize a *new* run with your own compatible checkpoint, choose a new output directory and use `--init-weights`. `--init-contract` can supply the associated model-semantics manifest. Inspect `INITIALIZATION.json` for the tensors actually inherited. On a resumed run the checkpoint restores the optimizer, scheduler, sampler and random-number states. The V2 validation log reports both `mean_record_relative_l2_squared`, used for model selection, and `pooled_relative_l2`, the square root of the ratio of summed squared field errors to summed reference energies across the evaluated records.

## Reproduce the paired layered pressure-head fit

`scno-pressure-heads` fits an orthogonal-detail head and a direct-pressure residual head on the same frozen E200+U96 model. The protocol requires 450 layered training records and three external layered validation records. Each training draw takes 16 distinct times from each of four strata. Validation uses all 401 times. Within each seed the two heads share the training draws, hidden-layer initialization and initial temporal factors.

Supply a deployment JSON with a top-level `ul` object containing paths under `binding`, `training_config`, `warm_protocol` and `completion_receipt`. These point to the frozen model binding, its original configuration, the registered split and the completed model record. Use an absolute, writable cache directory for decoded trajectories and a separate output directory.

```bash
python run.py run scno-pressure-heads -- \
  --deployment "$SCNO_ASSET_ROOT/layered-heads/deployment.json" \
  --cache "$SCNO_RUNS/layered-trajectory-cache" \
  --out "$SCNO_RUNS/layered-heads-01" --epochs 20 --batch 12
```

The output `PROTOCOL.json` records the split, seeds, sampling budget and number of trainable parameters. `LAYERED_READOUT_ABLATION.json` contains the per-case and pooled relative errors for each seed and head. This comparison fits the learned heads. The paper separately evaluates replacement by physical scaling coefficients.

## Fine-tune from two pressure fields

`scno-two-frame` fits an instance-specific readout using two observed full pressure snapshots, initial and boundary terms, and a discrete wave-equation residual. Supply an NPZ containing the four arrays below. The observation grid is indexed as `[z,x]`.

| NPZ key | Shape | Values |
| --- | --- | --- |
| `c201_mps` | `(201,201)` | Positive acoustic velocity in m/s |
| `source` | `(5,)` | `[x_m, z_m, frequency_Hz, delay_s, amplitude]` with positive amplitude |
| `observation_times_s` | `(2,)` | Two distinct increasing times in seconds within the checkpoint's support |
| `observed_pa` | `(2,201,201)` | Pressure fields in Pa at exactly those times, in the same order |

Use this example once the checkpoint whose weights you intend to fit is available. Keep the original training configuration and registered corpus accessible. The V2 loader checks their binding and the checkpoint SHA-256. The template enables an affine readout fit. Add `--enable-lora` to allow the optional low-rank update when its probe and time-budget criteria are met.

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

runs = Path(os.environ["SCNO_RUNS"]).resolve()
trained = runs / "layered-01"
weights = trained / "checkpoint.pt"
template = Path("forward/scno/scno_method_repair_20261119/finetune_v2.json")
config = json.loads(template.read_text())
digest = hashlib.sha256()
with weights.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
config["checkpoint"].update(path=str(weights), sha256=digest.hexdigest(),
                            training_config=str(trained / "CONFIG.json"))
with (runs / "config" / "two-frame-01.json").open("x") as stream:
    json.dump(config, stream, indent=2)
PY

python run.py run scno-two-frame -- \
  --config "$SCNO_RUNS/config/two-frame-01.json" \
  --instance "$SCNO_ASSET_ROOT/instance.npz" \
  --out "$SCNO_RUNS/two-frame-01" --device cuda:0
```

Choose a new output path for each instance. `RECEIPT.json` identifies the selected candidate, the observed times and the source checkpoint. `candidates.pt` contains the fitted readout states and refers back to the original checkpoint.

## Train on Marmousi pressure pairs

`scno-marmousi-paired` uses a committed set of 3,200 ordinary training model–shot pairs from 800 training velocity models, a 150 m smoothed background and a separate panel of 12 ordinary validation shots. The pair data include the registered background and total pressure responses. The loader verifies the pair index, acquisition and file commitments before training. Keep the exact target's reference pressure and receiver records reserved for blind evaluation after candidate selection.

```bash
python run.py run scno-marmousi-paired -- \
  --dataset "$SCNO_ASSET_ROOT/marmousi-release" \
  --pairs "$SCNO_ASSET_ROOT/ordinary-pressure-pairs" \
  --validation "$SCNO_ASSET_ROOT/ordinary-validation" \
  --background "$SCNO_ASSET_ROOT/background-run" \
  --scatter "$SCNO_ASSET_ROOT/scatter-run" \
  --output "$SCNO_RUNS/marmousi-paired-01"
```

The pair release supplies `PLAN.json`, `STATUS.json`, `TRAIN_INDEX.json` and the model–shot directories named in the index. Each model–shot directory contains `pressure.npy` and `background_pressure.npy` of shape `(801,201,201)` and the corresponding `b5.npy` and `background_b5.npy` gathers of shape `(801,201)`. The background run supplies `TRAIN_BASIS.pt`, `BACKGROUND_LAST.pt` and `PLAN.json`. The scatter run supplies `SCATTER_LAST.pt` and its own `PLAN.json`. Both sets of weights must agree with their recorded plans. For training on signed velocity perturbations, the separate `marmousi-derivative-fit` entry accepts a parent checkpoint and one or more paired-data roots.

## Workflow index

| Training entry | Data or experiment |
| --- | --- |
| `scno-train` | Staged SCNO training on registered V2 pressure fields |
| `scno-pressure-heads` | Matched layered residual-head fits on a shared frozen backbone |
| `scno-marmousi-paired` | Paired background and scattered-pressure training |
| `marmousi-native201` | Native 201 × 201 pressure and B5 receiver caches, compatible parent model and ordinary validation archive |
| `fno-train` | Uniform/layered FNO continuation over all 401 times per record using a parent runtime |
| `deeponet-train` | Joint branch and trunk continuation on the original 500-record training split over 400 times |
| `mionet-ufno-train` | Chunk prediction from known context frames, with `--data`, `--run-dir` and `--K` |
| `elastic-train` | Elastic neural-operator training on prepared data with a matching JSON configuration |
| `trace-spectral-train` | Full-pressure supervision for a spectral model using a frozen manifest and its SHA-256 |

| Fine-tuning entry | Data or experiment |
| --- | --- |
| `scno-two-frame` | Physical readout fit from two observed pressure snapshots, with optional LoRA |
| `scno-lora-pair` | Two 50-update low-rank fits from the same trained uniform/layered model to test the effect of the propagation-defect term |
| `scno-ais-01s` | Supervised 801-frame trajectory over 0–1 s with an importance-sampled discrete residual |
| `scno-finetune-4s` | Supervised 801-frame trajectory over 0–4 s with 5 ms spacing |
| `deeponet-pde-fit` | Single-source DeepONet fine-tuning with 400 reference frames and a PDE term |
| `marmousi-derivative-fit` | Paired pressure and waveform supervision with a signed finite-difference response |
| `trace-spectral-equivariant` | Reflection-equivariant pressure head, parent model, registered manifest and pressure cache |
| `trace-spectral-finetune` | Paired late-time pressure objectives on Marmousi using a checkpoint, manifest and cache |

Several comparison workflows continue a particular archived run with its original corpus and parent hashes. Use the [registry](workflows.json) to locate each trainer and inspect its required arguments. For a comparison across architectures, record the medium split, frame indices, source geometry and pressure normalization for every run.
