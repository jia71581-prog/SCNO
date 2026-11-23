"""Helpers used by the public training entry points."""
from hashlib import sha256 as hash_bytes
from pathlib import Path
import json
import os
import tempfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from training_checksums import SHALLOW_SHA256, require, sha256

SCHEMA = "marmousi_scno_sensitivity_adapter_v1"

FRAMES = 801

RECEIVERS = 201

SPLITS = ("train", "validation")

BINDING_KEYS = ("contract_hash", "dataset_recipe_hash", "manifest_hash")

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def array_sha(value):
    return hash_bytes(np.ascontiguousarray(value).tobytes()).hexdigest()

def canonical_sha(value):
    return hash_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                 allow_nan=False).encode("utf-8")).hexdigest()

def durable(path, write, mode="wb"):
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, mode) as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)

def save_json(path, value):
    durable(path, lambda stream: json.dump(value, stream, indent=2, allow_nan=False), mode="w")

def log(payload):
    print(json.dumps(payload, allow_nan=False), flush=True)

def attach_shallow(base):
    import shallow_b5

    require(sha256(shallow_b5.__file__) == SHALLOW_SHA256, "Changed exact shallow B5 source")
    base.native_core.anchor = shallow_b5.ShallowArrivalReceivers.from_reference(base.native_core.anchor)
    return base.eval()

class HorizontalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.stencil = nn.Conv1d(width, width, 5, dilation=dilation, padding=2 * dilation,
                                 padding_mode="replicate", groups=width)
        self.mix = nn.Conv1d(width, width, 1)

    def forward(self, hidden):
        state = self.norm(hidden.transpose(1, 2)).transpose(1, 2)
        return hidden + self.mix(F.silu(self.stencil(state)))

class SensitivityAdapter(nn.Module):
    """Map frozen spatial context and physical velocity features to B5 splines."""

    def __init__(self, feature_channels, time_rows, *, width):
        super().__init__()
        require(time_rows.shape == (FRAMES, 163) and time_rows[0].count_nonzero() == 0,
                "Frozen B5 time basis changed")
        self.register_buffer("time_rows", time_rows.detach().clone(), persistent=False)
        self.input = nn.Conv1d(3 * feature_channels + 16 + 16 + 6 + 2, width, 1)
        self.blocks = nn.Sequential(*(HorizontalBlock(width, dilation) for dilation in (1, 4, 16)))
        self.output = nn.Conv1d(width, time_rows.shape[1], 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, frame_indices=None):
        coefficients = self.output(self.blocks(F.silu(self.input(features))))
        rows = self.time_rows if frame_indices is None else self.time_rows[torch.as_tensor(frame_indices, device=features.device)]
        return rows @ coefficients

def residual_features(velocity, source, native_context):
    native = native_context.features[:, :, 1:4].flatten(1, 2)
    travel = native_context.propagation.travel_times[:, :, 1:4].flatten(1, 2)
    batch = len(native)
    if velocity.ndim == 2:
        velocity = velocity[None]
    if velocity.ndim == 4:
        velocity = velocity[:, 0]
    values = (velocity[:, None] / 2500.).log()
    columns = F.adaptive_avg_pool2d(values, (16, RECEIVERS))[:, 0].expand(batch, -1, -1)
    lateral = F.adaptive_avg_pool2d(values, (1, 16)).reshape(len(values), 16, 1).expand(batch, -1, RECEIVERS)
    x = torch.linspace(0., 1., RECEIVERS, device=native.device, dtype=native.dtype)
    offset = x[None] - source[:, 0:1] / 2000.
    positions = torch.stack((offset, offset.abs()), dim=1)
    return torch.cat((native, columns, lateral, travel, positions), dim=1)

class SensitivityReadout:
    """Keep the base operator frozen and retain velocity autograd through both paths."""

    def __init__(self, base, adapter):
        self.base = base
        self.adapter = adapter
        self.device = base.device
        self.sources = base.sources
        self.times = base.times
        self.receiver_rms = base.receiver_rms

    def eval(self):
        self.base.eval()
        self.adapter.eval()
        return self

    def encode(self, velocity, source):
        velocity = torch.as_tensor(velocity, dtype=torch.float32, device=self.device)
        source = torch.as_tensor(source, dtype=torch.float32, device=self.device)
        if source.ndim == 1:
            source = source[None]
        encoded = self.base.encode(velocity, source)
        features = residual_features(velocity, source, encoded.native_context)
        return encoded, features

    def decode(self, context, frame_indices=None, *, normalized=True):
        encoded, features = context
        baseline = self.base.decode(encoded, frame_indices, normalized=True)
        signal = baseline + self.adapter(features, frame_indices)
        return signal if normalized else signal * self.receiver_rms

    def __call__(self, velocity, source, frame_indices=None, *, normalized=True):
        return self.decode(self.encode(velocity, source), frame_indices, normalized=normalized)

def adapter_for(base, width):
    feature_channels = base.native_core.residual.feature_channels
    return SensitivityAdapter(feature_channels, base.kernel.decode_rows, width=width).to(base.device)

def load_sensitivity_readout(dataset, checkpoint, adapter_checkpoint, *, expected_sha256, device="cpu"):
    """Restore an immutable selected adapter for physical velocity/source inference."""
    from kernel_corrected_readout import KERNEL_SHA, NATIVE_SHA, load_kernel_corrected_readout
    from tianhe_observations import CommittedMarmousi

    require(sha256(adapter_checkpoint) == expected_sha256, "Selected sensitivity artifact changed")
    state = torch.load(adapter_checkpoint, map_location="cpu", weights_only=True)
    binding = state["binding"]
    data = CommittedMarmousi(dataset)
    require(binding["schema"] == SCHEMA and state["selected"] is True and state["epoch"] > 0
            and binding["dataset_binding"] == data.binding and binding["receiver_rms"] == data.rms
            and binding["kernel_sha256"] == KERNEL_SHA and binding["native_sha256"] == NATIVE_SHA
            and binding["trainer_sha256"] == sha256(__file__) and binding["shallow_sha256"] == SHALLOW_SHA256,
            "Sensitivity weights are not bound to this frozen release, source, and selected baseline")
    base = attach_shallow(load_kernel_corrected_readout(data.root, checkpoint, device=device))
    adapter = adapter_for(base, binding["width"])
    adapter.load_state_dict(state["adapter"], strict=True)
    adapter.requires_grad_(False).eval()
    return SensitivityReadout(base, adapter)

def safe_shard(root, split, name):
    directory = (root / split).resolve()
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(directory):
        candidate = (directory / name).resolve()
    require(candidate.is_relative_to(directory) and candidate.is_file(), "Pair shard must be in its declared split")
    return candidate

def pair_bank(data, parent, split):
    receipt = parent / split / "manifest.json"
    value = read_json(receipt)
    require(value["split"] == split and value["status"] in ("COMPLETE", "PREFIX") and value["receiver_train_rms"] == data.rms
            and value["units"] == "raw_lwc" and value["secant_units"] == "frozen_train_rms_normalized_per_unit_bounded_sigmoid_latent"
            and value["ordinary_splits_only"] is True and value["exact_targets_or_fwi_checkpoints_used"] is False
            and canonical_sha({key: val for key, val in value.items() if key != "bank_content_sha256"}) == value["bank_content_sha256"]
            and all(value[key] == data.binding[key] for key in BINDING_KEYS),
            f"Invalid {split} directional data binding")
    records = value["records"]
    require(records and len({record["pair_id"] for record in records}) == len(records), "Missing or repeated pair IDs")
    require(value["status"] == "COMPLETE" or all(record.get("origin") == "generated_from_released_A_parent" for record in records),
            "Use the complete archived F-family bank")
    qc_file = receipt.with_name("QC.json")
    if qc_file.is_file():
        qc = read_json(qc_file)
        require(qc["status"] == "PASS" and qc["bank_content_sha256"] == value["bank_content_sha256"],
                "Directional half-step label QC failed or belongs to another bank")
    paths = []
    for record in records:
        if record.get("origin") == "generated_from_released_A_parent":
            key = record["center_model_id"]
            row = data.rows[key]
            require(row["split"] == split and row["family"] == "A"
                    and not row["is_exact_initial"] and not row["is_exact_target"]
                    and row["parent_group_id"] == record["parent_group_id"]
                    and row["velocity_sha256"] == record["center_model_release_sha256"]
                    and list(data.rosters[key]) == record["shot_ids"]
                    and record["direction_type"] in ("smooth_multiscale", "deep_tapered")
                    and all(record[f"{side}_model_id"] == f"{record['pair_id']}_{side}"
                            for side in ("minus", "plus")), f"Invalid smooth archived parent: {key}")
        else:
            endpoints = [data.rows[record[key]] for key in ("minus_model_id", "plus_model_id")]
            for side, sign, row in zip(("minus", "plus"), (-1, 1), endpoints):
                require(row["split"] == split and row["family"] == "F" and row["pair_sign"] == sign
                        and row["pair_id"] == record["pair_id"] and row["parent_group_id"] == record["parent_group_id"]
                        and row["parameters"]["h_actual"] == record["h"]
                        and row["parameters"]["direction_seed"] == record["direction_seed"]
                        and row["velocity_sha256"] == record[side + "_velocity_sha256"]
                        and list(data.rosters[row["model_id"]]) == record["shot_ids"]
                        and not row["is_exact_initial"] and not row["is_exact_target"],
                        f"Wrong split, signed family-F pair, or shot roster: {record['pair_id']}")
        path = safe_shard(parent, split, record["shard"])
        require(sha256(path) == record["shard_sha256"], f"Changed directional shard: {path}")
        paths.append((record, path))
    return paths, sha256(receipt)

def read_pair(record, path, data):
    with np.load(path, allow_pickle=False) as shard:
        minus, plus = shard["velocity_minus"], shard["velocity_plus"]
        center, direction = shard["center_velocity_mps"], shard["direction_latent"]
        raw_minus, raw_plus = shard["b5_minus"], shard["b5_plus"]
        secant, h_value, ids = shard["b5_secant_normalized"], shard["h"], shard["shot_ids"]
    h = float(h_value)
    expected = (len(record["shot_ids"]), FRAMES, RECEIVERS)
    require(h_value.ndim == 0 and np.isfinite(h) and h > 0 and h == record["h"]
            and ids.dtype == np.int64 and ids.tolist() == record["shot_ids"]
            and all(v.dtype == np.float32 and v.shape == (201, 201) and np.isfinite(v).all() for v in (minus, plus))
            and all(v.dtype == np.float64 and v.shape == (201, 201) and np.isfinite(v).all() for v in (center, direction))
            and abs(np.sqrt(np.square(direction).mean()) - 1.) <= 1e-3
            and array_sha(center) == record["center_velocity_sha256"]
            and array_sha(direction) == record["direction_latent_sha256"]
            and array_sha(minus) == record["minus_velocity_sha256"] and array_sha(plus) == record["plus_velocity_sha256"]
            and all(w.dtype == np.float32 and w.shape == expected and np.isfinite(w).all()
                    for w in (raw_minus, raw_plus, secant)), f"Corrupt pair contents: {path}")
    predicted_secant = (raw_plus.astype(np.float64) - raw_minus) / (2. * h * data.rms)
    require(np.allclose(secant, predicted_secant, rtol=2e-7, atol=1e-8), "Stored centered secant changed")
    templates = data.contract["templates"]
    forbidden = (templates["initial_array_sha256"], templates["truth_array_sha256"])
    center_f32_sha = array_sha(center.astype(np.float32))
    require(not any(checksum in forbidden for checksum in
                    (center_f32_sha, record["minus_velocity_sha256"], record["plus_velocity_sha256"]))
            and center_f32_sha == record.get("center_model_release_sha256", center_f32_sha),
            "Directional training input contains an exact audit or disagrees with the smooth release center")
    labels = np.stack((raw_minus, raw_plus)).astype(np.float64) / data.rms
    return minus, plus, ids, labels.astype(np.float32), secant, h

def anchor_rows(data, split, count, seed):
    """Fix one ordinary waveform shot per selected model, balanced across A–E."""
    rng = np.random.default_rng(seed)
    groups = {}
    for family in "ABCDE":
        choices = [i for i, key in enumerate(data.keys[split]) if data.rows[key]["family"] == family
                   and not data.rows[key]["is_exact_initial"] and not data.rows[key]["is_exact_target"]]
        groups[family] = list(rng.permutation(choices))
    chosen = []
    for index in range(max(map(len, groups.values()))):
        for family in groups:
            if index < len(groups[family]):
                key_index = groups[family][index]
                key = data.keys[split][key_index]
                shot_slot = (index + hash_bytes(key.encode()).digest()[0]) % len(data.rosters[key])
                chosen.append((key, data.rosters[key][shot_slot], int(data.offsets[split][key_index] + shot_slot)))
    require(len(chosen) >= count > 0, f"Insufficient {split} waveform anchors")
    return chosen[:count]

def normalized_loss(prediction, selected_target, full_target, count):
    energies = full_target.double().square().sum(dim=(1, 2)).to(prediction)
    floor = energies.mean().clamp_min(1e-16) * 0.01
    return ((prediction - selected_target).square().sum(dim=(1, 2)) / energies.clamp_min(floor)).mean() * (FRAMES / count)
