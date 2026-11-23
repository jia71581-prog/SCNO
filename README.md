# SCNO · Spectral Correction Neural Operator

Training and fine-tuning code for *A Neural Operator with Orthogonal Multiscale Corrections for Time-Domain Acoustic Wavefield Prediction* by **Yuhang Jia, Dinghui Yang and Zhilong Fang**.

SCNO predicts a two-dimensional pressure field from a velocity model, a source and a query time. A base decoder represents the large-scale pressure. Learned detail coefficients pass through fixed orthogonal synthesis at three spatial scales. Each registered time can be queried directly, including times far apart on the same trajectory.

## Choose an experiment

| Goal | Workflow | What to supply |
| --- | --- | --- |
| Train the staged SCNO on full pressure fields | `scno-train` | A registered uniform/layered or joint Marmousi-style corpus and its matching configuration |
| Reproduce the paired layered pressure-head comparison | `scno-pressure-heads` | The frozen E200+U96 model, its training and split records, and the layered pressure fields |
| Adapt a pretrained SCNO to two observed pressure snapshots | `scno-two-frame` | A compatible checkpoint, its matching metadata and a two-frame NPZ |
| Train the Marmousi background and scattered-pressure operator | `scno-marmousi-paired` | Paired full-field data, ordinary validation references and the two initialization runs |
| Run comparison models | `fno-train`, `deeponet-train`, `mionet-ufno-train` | The experiment-specific corpus, time support and any parent checkpoint |

The release contains training and fine-tuning source. Experiment datasets, pretrained weights and reference wavefields are supplied separately by the person running the experiment. The [workflow guide](docs/WORKFLOWS.md) gives the input formats, checkpoint requirements and output files for the main experiments and lists all 17 training and fine-tuning entries.

## Inspect the code

Run the following from a Linux shell. The registry can be inspected before installing scientific Python packages or locating the experiment data.

```bash
git clone https://github.com/jia71581-prog/SCNO.git
cd SCNO
python3 run.py check
python3 run.py list train
python3 run.py list finetune
python3 run.py show scno-train
```

The reference Linux environment uses Python 3.12.10 and PyTorch 2.7.1. Several workflows use Python 3.11's file-digest API. Install a PyTorch build compatible with your GPU driver. The main data reader uses NumPy and h5py. The Marmousi spectral and native-grid helpers use SciPy, and the paired LoRA experiment reads YAML. Matplotlib supplies the optional plots in the DeepONet evaluation. With the chosen environment active, install the scientific dependencies and inspect the two main entry points.

```bash
python -m pip install numpy h5py scipy pyyaml matplotlib
python run.py run scno-train -- --help
python run.py run scno-two-frame -- --help
```

For a first run, the [worked training example](docs/WORKFLOWS.md#train-and-resume-scno) binds the uniform/layered data to a local configuration, commits one optimizer update and resumes the same run. The [two-frame example](docs/WORKFLOWS.md#fine-tune-from-two-pressure-fields) uses the resulting checkpoint and two supplied pressure fields.

## Relating a run to the paper

- **Uniform and layered media.** The staged trainer and the paired pressure-head experiment answer different questions. The latter holds the trained backbone fixed, fits two heads on identical draws and scores both across all 401 time frames of the same held-out layered records.
- **Marmousi.** The paired neural trainer learns from ordinary model–shot pairs. The paper also reports a shot-indexed spectral library assembled from nearby training velocities. Its results concern those registered source positions. Keep the library and the neural checkpoint identified separately when reporting an error.
- **Pressure and receivers.** Report the spatial support, time axis and aggregation alongside relative error. A full-field error uses all evaluated grid nodes. A receiver error uses the specified acquisition operator and can have a different value on the very same prediction.

The [workflow registry](docs/workflows.json) records the source file and import paths for each entry. To identify the exact implementation used in a run, record the output of `git rev-parse HEAD` alongside the run configuration.

## Release checks

All 17 entry points opened their own command-line help on a CPU installation of Python 3.12.7 and PyTorch 2.8.0. A synthetic 201 × 201 layered example exercised the forward and backward passes, and the native-grid Marmousi configuration constructed a model with 14,672,980 parameters. A separate test updated the rank-four fine-tuning parameters and differentiated the discrete wave-equation residual. The paper's accuracy measurements use the registered experiment data described in the [workflow guide](docs/WORKFLOWS.md).

The elastic comparison model uses the SOAP optimizer. Its upstream [license and attribution](forward/elastic/elastic_soap_batch24_final_20261030/source_snapshot/eno/optim/vendor/SOAP_LICENSE.txt) accompany the implementation.
