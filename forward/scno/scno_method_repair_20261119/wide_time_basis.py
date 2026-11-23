"""Function-preserving expansion of the actual SCNO joint time/detail factors.

This candidate does not change JointFNOTimeDecoder or a production checkpoint.
Use ``expand_decoder_time_basis`` on an already semantically restored decoder.
The old spatial coefficient rows and time columns are copied, not re-fitted.
New spatial rows are Xavier/nonzero; new time columns are zero.  Thus the new
time factors receive a first-step gradient; the new spatial factors become
active after a time-factor update.  True resume must not run this operation.

The existing decoder synthesizes one full-resolution field per scale/basis.
At R=800 this means 2,400 such fields. ``decode_coefficient_first`` instead
contracts the two affine factors before fixed synthesis.  It is the same
function (up to floating-point reduction order) and retains both gradients.
At fixed weights, ``compile_detail_readout`` caches this contraction for
inference. It is explicitly rejected after the relevant parameters change.
"""

from copy import deepcopy
from dataclasses import dataclass
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from joint_fno_decoder import JointFNOTimeDecoder, _frame_indices, _time_projection


SCHEMA = "SCNO_JOINT_DETAIL_TIME_BASIS_EXPANSION_V1"


def _tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def decoder_identity(decoder):
    """Check actual module roles, physical time axis, and stored orthogonal bases."""
    if type(decoder) is not JointFNOTimeDecoder:
        raise TypeError("expand only the reviewed JointFNOTimeDecoder computation, not a shape-compatible module")
    if (decoder.detail_transform is None or not len(decoder.detail_heads)
            or len(decoder.detail_heads) != len(decoder.detail_transform.levels)
            or len(decoder.detail_time) != len(decoder.detail_heads)):
        raise ValueError("expected one spatial coefficient predictor and time matrix per orthogonal detail scale")
    rank, times = int(decoder.detail_latent), decoder.native_times
    if times.ndim != 1 or not torch.isfinite(times).all() or not (times[1:] > times[:-1]).all():
        raise ValueError("the inherited physical time axis is invalid")
    scales = []
    for head, temporal, level in zip(decoder.detail_heads, decoder.detail_time, decoder.detail_transform.levels):
        slots = int(level.valid_detail.shape[-1])
        if (not isinstance(head, nn.Sequential) or len(head) != 5
                or any(type(head[i]) is not nn.Linear for i in (0, 2, 4))
                or any(type(head[i]) is not nn.SiLU for i in (1, 3))
                or type(temporal) is not nn.Linear or temporal.bias is not None
                or head[0].in_features != level.modes * decoder.detail_projection.out_channels + slots
                or head[4].out_features != rank * slots
                or temporal.in_features != rank or temporal.out_features != times.numel()):
            raise ValueError("coefficient/time modules do not implement the recorded factor roles")
        scales.append({
            "block_grid": [int(level.gz), int(level.gx)], "slots": slots,
            "valid_coefficients": int(level.valid_detail.sum()), "hidden_width": head[4].in_features,
            "qz_sha256": _tensor_hash(level.qz), "qx_sha256": _tensor_hash(level.qx),
            "valid_mask_sha256": _tensor_hash(level.valid_detail),
        })
    return {"decoder_class": "joint_fno_decoder.JointFNOTimeDecoder", "profile": decoder.profile,
            "in_channels": decoder.in_channels, "time_basis": rank,
            "time_frames": int(times.numel()), "native_axis_sha256": _tensor_hash(times),
            "native_axis_f32_sha256": hashlib.sha256(times.detach().cpu().float().numpy().astype("<f4").tobytes()).hexdigest(),
            "active_shape": list(decoder.detail_transform.shape), "scales": scales}


def _check_external_semantics(identity, semantics):
    if semantics is None:
        return
    checks = {"profile": identity["profile"], "detail_latent": identity["time_basis"],
              "output_frames": identity["time_frames"], "active_grid": identity["active_shape"],
              "native_axis_sha256": identity["native_axis_f32_sha256"]}
    for key, expected in checks.items():
        if key not in semantics or semantics[key] != expected:
            raise ValueError(f"source checkpoint semantic mismatch: {key}")


def expand_decoder_time_basis(decoder, time_basis=800, *, seed=261119,
                              source_checkpoint_semantics=None, startup="new_training", inplace=False):
    """Return (candidate_decoder, receipt), preserving every trained old basis.

    Rebuild the optimizer after replacement: old Parameter objects are not
    optimizer-compatible with the wider factors.  All other decoder weights,
    buffers, frozen flags, and the physical output time axis are preserved.
    The containing model's ``detail_latent`` and checkpoint/config semantics
    must be updated by the integration owner. This helper changes no dataset,
    normalization, backbone, Gabor features, loss, or pressure values at start.
    """
    if startup != "new_training":
        raise ValueError("basis expansion is a new-training transition; true resume must restore its existing state")
    if type(time_basis) is not int or time_basis < 1:
        raise ValueError("time_basis must be a positive integer")
    before = decoder_identity(decoder)
    _check_external_semantics(before, source_checkpoint_semantics)
    previous = before["time_basis"]
    if time_basis < previous:
        raise ValueError("this transition expands bases; it never prunes inherited bases")
    candidate = decoder if inplace else deepcopy(decoder)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    receipt = {"schema": SCHEMA, "operation": "new_training_basis_expansion", "seed": seed,
               "source_identity": before, "source_checkpoint_semantics": deepcopy(source_checkpoint_semantics),
               "old_basis": previous, "new_basis": time_basis, "scales": [],
               "optimizer_action": "rebuild optimizer for replaced parameters; do not restore old momentum by shape",
               "new_initialization": {"spatial_weight": "xavier_uniform_gain_1_cpu_seeded",
                                      "spatial_bias": "zero", "time_columns": "zero"},
               "gradient_schedule": "new time columns active immediately; new spatial rows after the time factors update",
               "unchanged": ["native physical output axis", "fixed orthogonal bases and valid masks",
                             "all old coefficient bases", "all old time columns", "base pressure readout",
                             "preceding spatial hidden layers"],
               "function_preservation": "old factors copied exactly and added time columns zero; reduction rounding only"}
    if time_basis == previous:
        receipt["operation"] = "identity_no_new_basis"
        receipt["optimizer_action"] = "no factor replacement"
    else:
        with torch.no_grad():
            for i, (head, temporal, level) in enumerate(zip(candidate.detail_heads, candidate.detail_time,
                                                           candidate.detail_transform.levels)):
                old = head[-1]
                slots = int(level.valid_detail.shape[-1])
                # Meta construction avoids perturbing process-global CPU/CUDA RNG.
                spatial = nn.Linear(old.in_features, time_basis * slots, bias=old.bias is not None,
                                    device="meta", dtype=old.weight.dtype).to_empty(device=old.weight.device)
                time = nn.Linear(time_basis, temporal.out_features, bias=False,
                                 device="meta", dtype=temporal.weight.dtype).to_empty(device=temporal.weight.device)
                initial = torch.empty(spatial.weight.shape, dtype=torch.float32, device="cpu")
                nn.init.xavier_uniform_(initial, gain=1., generator=generator)
                spatial.weight.copy_(initial.to(spatial.weight))
                if spatial.bias is not None:
                    spatial.bias.zero_()
                time.weight.zero_()
                rows = previous * slots
                spatial.weight[:rows].copy_(old.weight)
                if old.bias is not None:
                    spatial.bias[:rows].copy_(old.bias)
                time.weight[:, :previous].copy_(temporal.weight)
                spatial.weight.requires_grad_(old.weight.requires_grad)
                if spatial.bias is not None:
                    spatial.bias.requires_grad_(old.bias.requires_grad)
                time.weight.requires_grad_(temporal.weight.requires_grad)
                spatial.train(old.training)
                time.train(temporal.training)
                head[-1], candidate.detail_time[i] = spatial, time
                receipt["scales"].append({
                    "scale": i, "copied_spatial_rows": [0, rows], "copied_time_columns": [0, previous],
                    "old_spatial_weight_sha256": _tensor_hash(old.weight),
                    "copied_spatial_weight_sha256": _tensor_hash(spatial.weight[:rows]),
                    "old_time_weight_sha256": _tensor_hash(temporal.weight),
                    "copied_time_weight_sha256": _tensor_hash(time.weight[:, :previous]),
                    "new_spatial_weight_nonzero": bool(torch.count_nonzero(spatial.weight[rows:])),
                    "new_time_columns_zero": bool(torch.count_nonzero(time.weight[:, previous:]) == 0),
                })
        candidate.detail_latent = time_basis
    receipt["target_identity"] = decoder_identity(candidate)
    return candidate, receipt


def _factor_signature(decoder):
    return tuple((id(p), p._version, tuple(p.shape), str(p.device), str(p.dtype))
                 for head, temporal in zip(decoder.detail_heads, decoder.detail_time)
                 for p in (head[-1].weight, head[-1].bias, temporal.weight) if p is not None)


def decoder_checkpoint_payload(decoder, *, model_semantics=None):
    """Component state; the caller must also save optimizer/RNG/phase for resume."""
    return {"schema": SCHEMA, "component": "joint_detail_decoder",
            "identity": decoder_identity(decoder), "model_semantics": deepcopy(model_semantics),
            "state_dict": {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()},
            "training_state_complete": False}


def restore_decoder_checkpoint(template, payload, *, expected_model_semantics=None):
    """Allocate matching factors, then strictly restore; no fitting/calibration.

    A width-8 architecture template may allocate the saved width here; all
    initialized tensors are overwritten by saved state. A matching width-800
    template is simply cloned. This component does not claim full training
    resume: the owner restores the optimizer, scheduler, RNG and record cursor.
    """
    if payload.get("schema") != SCHEMA or payload.get("component") != "joint_detail_decoder":
        raise ValueError("not a semantic wide-basis decoder component checkpoint")
    if expected_model_semantics is not None and payload.get("model_semantics") != expected_model_semantics:
        raise ValueError("enclosing model/time/grid/normalization semantics differ")
    candidate, _ = expand_decoder_time_basis(template, int(payload["identity"]["time_basis"]))
    if decoder_identity(candidate) != payload["identity"]:
        raise ValueError("saved decoder axis, profile or orthogonal hierarchy differs from its template")
    current, saved = candidate.state_dict(), payload["state_dict"]
    if current.keys() != saved.keys() or any(not torch.is_tensor(saved[key]) or current[key].shape != saved[key].shape
                                           or current[key].dtype != saved[key].dtype for key in current):
        raise ValueError("decoder component state is missing or tensor-incompatible")
    candidate.load_state_dict(saved, strict=True)
    if decoder_identity(candidate) != payload["identity"]:
        raise ValueError("saved fixed buffers contradict the checkpoint identity")
    return candidate


def _fold_factors(decoder):
    result = []
    for head, temporal, level in zip(decoder.detail_heads, decoder.detail_time, decoder.detail_transform.levels):
        output = head[-1]
        rank, slots = decoder.detail_latent, level.valid_detail.shape[-1]
        # A[T,R] @ W[R,slot*hidden] is valid because only the last layer is affine.
        weight = (temporal.weight @ output.weight.reshape(rank, -1)).reshape(
            temporal.out_features, slots, output.in_features)
        bias = (temporal.weight @ output.bias.reshape(rank, slots)) if output.bias is not None else None
        result.append((weight, bias))
    return tuple(result)


@dataclass(frozen=True)
class CompiledDetailReadout:
    factors: tuple
    signature: tuple


@torch.no_grad()
def compile_detail_readout(decoder):
    """Fold fixed output weights once. This cache is inference-only, never a label cache."""
    decoder_identity(decoder)
    return CompiledDetailReadout(_fold_factors(decoder), _factor_signature(decoder))


def prepare_coefficient_features(decoder, features):
    """Return per-scale post-second-SiLU features [B,GZ,GX,H], with their graph.

    This function has no implicit cache or detach. Only explicitly frozen
    hidden paths may be cached by a caller. Recompute on every update when
    detail_projection or either hidden layer is trainable.
    """
    if features.ndim != 4 or features.shape[1] != decoder.in_channels:
        raise ValueError("expected the decoder's spatial conditioning features")
    active = features[:, :, 1:]
    if tuple(active.shape[-2:]) != tuple(decoder.detail_transform.shape):
        raise ValueError("feature grid differs from the inherited active orthogonal grid")
    _, packets = decoder.detail_transform.analyze(decoder.detail_projection(active))
    hidden = []
    for head, packet, level in zip(decoder.detail_heads, packets, decoder.detail_transform.levels):
        block = packet.permute(0, 2, 3, 1, 4).flatten(-2)
        value = torch.cat((block, level.valid_detail.to(block).expand(len(features), -1, -1, -1)), dim=-1)
        for layer in list(head.children())[:-1]:
            value = layer(value)
        hidden.append(value)
    return tuple(hidden)


def coefficient_packets(decoder, features_or_prepared, slots=None, *, compiled=None, time_chunk=None):
    """Predict valid detail packets [B,K,GZ,GX,S], without pressure synthesis.

    ``features_or_prepared`` is raw [B,C,Z,X] or the explicit tuple returned
    by prepare_coefficient_features. No targets or stale trainable features
    are cached. Dynamic folding selects time rows before matrix products,
    preserving gradients to both factors without computing unrequested rows.
    A squared coefficient loss must sum valid slots on a common record-energy
    denominator; separately averaging each scale changes that objective.
    """
    hidden = (prepare_coefficient_features(decoder, features_or_prepared)
              if torch.is_tensor(features_or_prepared) else tuple(features_or_prepared))
    if len(hidden) != len(decoder.detail_heads) or not hidden:
        raise ValueError("expected one explicit hidden feature tensor per scale")
    batch, device = hidden[0].shape[0], hidden[0].device
    for value, head, level in zip(hidden, decoder.detail_heads, decoder.detail_transform.levels):
        if value.shape != (batch, level.gz, level.gx, head[-1].in_features) or value.device != device:
            raise ValueError("prepared coefficient features do not match the decoder scale layout")
    indices = _frame_indices(slots, batch, device)
    count = decoder.native_times.numel() if indices is None else indices.shape[-1]
    if indices is not None and indices.numel() and (indices.min() < 0 or indices.max() >= len(decoder.native_times)):
        raise ValueError("coefficient time slots lie outside the native axis")
    if time_chunk is None:
        time_chunk = max(1, count)
    if type(time_chunk) is not int or time_chunk < 1:
        raise ValueError("time_chunk must be a positive integer")
    if compiled is not None:
        if torch.is_grad_enabled():
            raise ValueError("compiled coefficient factors are inference-only; dynamic folding retains training gradients")
        if not isinstance(compiled, CompiledDetailReadout) or compiled.signature != _factor_signature(decoder):
            raise ValueError("compiled detail readout is stale after factor replacement or update")
    result = []
    for scale, (value, head, temporal, level) in enumerate(zip(hidden, decoder.detail_heads, decoder.detail_time,
                                                            decoder.detail_transform.levels)):
        chunks = []
        width, rank, nslots = head[-1].in_features, decoder.detail_latent, level.valid_detail.shape[-1]
        for first in range(0, count, time_chunk):
            end = min(first + time_chunk, count)
            selected = (torch.arange(first, end, device=device) if indices is None else indices[..., first:end])
            if compiled is None:
                rows = temporal.weight[selected]
                weight = (rows @ head[-1].weight.reshape(rank, -1)).reshape(*rows.shape[:-1], nslots, width)
                bias = rows @ head[-1].bias.reshape(rank, nslots) if head[-1].bias is not None else None
            else:
                weights, biases = compiled.factors[scale]
                weight, bias = weights[selected], biases[selected] if biases is not None else None
            if selected.ndim == 1:
                coeff = torch.einsum("bxyh,ksh->bkxys", value, weight)
                if bias is not None:
                    coeff = coeff + bias[None, :, None, None, :]
            else:
                coeff = torch.einsum("bxyh,bksh->bkxys", value, weight)
                if bias is not None:
                    coeff = coeff + bias[:, :, None, None, :]
            chunks.append(coeff.masked_fill(~level.valid_detail, 0))
        result.append(torch.cat(chunks, dim=1) if chunks else
                      value.new_empty((batch, 0, level.gz, level.gx, nslots)))
    return tuple(result)


def decode_coefficient_first(decoder, features, slots=None, *, return_parts=False,
                             time_chunk=32, compiled=None):
    """Exact algebraic pressure readout without R full-grid fields per scale.

    Full output still contains T*201*201 physical pressure values. Chunking
    reduces transient coefficient/synthesis memory; it does not claim a
    measured latency improvement. Dynamic training contractions keep the
    graph to both factors. Static compiled contractions require no-grad and
    fail on stale factor versions. Shared or per-record frame indices work.
    """
    if type(time_chunk) is not int or time_chunk < 1:
        raise ValueError("time_chunk must be a positive integer")
    if features.ndim != 4 or features.shape[1] != decoder.in_channels:
        raise ValueError("expected the decoder's spatial conditioning features")
    active = features[:, :, 1:]
    if tuple(active.shape[-2:]) != tuple(decoder.detail_transform.shape):
        raise ValueError("feature grid differs from the inherited active orthogonal grid")
    batch = len(features)
    indices = _frame_indices(slots, batch, features.device)
    count = decoder.native_times.numel() if indices is None else indices.shape[-1]
    if indices is not None and indices.numel() and (indices.min() < 0 or indices.max() >= len(decoder.native_times)):
        raise ValueError("frame indices exceed the unchanged physical output axis")
    if compiled is not None:
        if torch.is_grad_enabled():
            raise ValueError("compiled fixed readout is inference-only; training must retain both factor gradients")
        if not isinstance(compiled, CompiledDetailReadout) or compiled.signature != _factor_signature(decoder):
            raise ValueError("compiled detail readout is stale; rebuild after parameter changes or device conversion")
    base = active
    for layer in list(decoder.proj.children())[:-1]:
        base = layer(base)
    hidden = prepare_coefficient_features(decoder, features)
    projection = decoder.proj[-1]
    predictions, bases, detail_chunks = [], [], [[] for _ in hidden]
    last = decoder.detail_transform.levels[-1]
    for first in range(0, count, time_chunk):
        end = min(first + time_chunk, count)
        selected = (torch.arange(first, end, device=features.device) if indices is None
                    else indices[..., first:end])
        coefficients = coefficient_packets(decoder, hidden, selected, compiled=compiled)
        coarse = active.new_zeros((batch, end - first, last.gz * 4, last.gx * 4))
        detail = decoder.detail_transform.synthesize(coarse, coefficients)
        base_pressure = _time_projection(base, projection.weight[:, :, 0, 0], projection.bias, selected)
        predictions.append(F.pad(base_pressure + detail, (0, 0, 1, 0)))
        if return_parts:
            bases.append(F.pad(base_pressure, (0, 0, 1, 0)))
            for holder, coefficient in zip(detail_chunks, coefficients):
                holder.append(coefficient)
    shape = (batch, 0, active.shape[-2] + 1, active.shape[-1])
    prediction = torch.cat(predictions, dim=1) if predictions else active.new_empty(shape)
    if not return_parts:
        return prediction
    return {"prediction": prediction,
            "base_pressure": torch.cat(bases, dim=1) if bases else active.new_empty(shape),
            "detail_coefficients": tuple(torch.cat(chunks, dim=1) if chunks else
                                         active.new_empty((batch, 0, level.gz, level.gx, level.valid_detail.shape[-1]))
                                         for chunks, level in zip(detail_chunks, decoder.detail_transform.levels)),
            "valid_detail_masks": tuple(level.valid_detail for level in decoder.detail_transform.levels)}


def basis_cost(decoder, time_basis=800, *, batch=1, bytes_per_value=4, time_chunk=32):
    """Count actual factor, coefficient and pressure tensors; no measured GPU claim."""
    identity = decoder_identity(decoder)
    frames, old = identity["time_frames"], identity["time_basis"]
    area = math.prod(identity["active_shape"])
    scales = []
    for level, head in zip(decoder.detail_transform.levels, decoder.detail_heads):
        slots, width = level.valid_detail.shape[-1], head[-1].in_features
        stored = int(level.gz * level.gx * slots)
        params = time_basis * (slots * (width + int(head[-1].bias is not None)) + frames)
        scales.append({
            "block_grid": [level.gz, level.gx], "stored_slots_per_basis": stored,
            "valid_slots_per_basis": int(level.valid_detail.sum()),
            "factor_parameters": params,
            "added_factor_parameters": (time_basis - old) * (slots * (width + int(head[-1].bias is not None)) + frames),
            "factor_parameter_bytes": params * bytes_per_value,
            "factor_adam_fp32_weight_grad_m_v_bytes": params * 16,
            "coefficient_basis_bytes_per_batch": batch * time_basis * stored * bytes_per_value,
            "separate_full_grid_basis_bytes_per_batch": batch * time_basis * area * bytes_per_value,
            "legacy_time_mix_macs_per_batch": batch * frames * time_basis * area,
            "fold_once_macs": frames * time_basis * slots * width,
            "folded_weight_bytes": frames * slots * (width + int(head[-1].bias is not None)) * bytes_per_value,
            "folded_predict_macs_per_batch": batch * frames * stored * width,
            "chunk_coefficient_bytes_per_batch": batch * min(frames, time_chunk) * stored * bytes_per_value,
        })
    return {"output_frames": frames, "requested_time_basis": time_basis,
            "maximum_time_matrix_rank": min(frames, time_basis),
            "time_axis_unchanged": True,
            "relation": "basis count is a factor width, not frame count; R=800 has rank<=401 at T=401 and rank<=800 at T=801",
            "scales": scales,
            "total_factor_parameters": sum(s["factor_parameters"] for s in scales),
            "total_added_factor_parameters": sum(s["added_factor_parameters"] for s in scales),
            "total_legacy_synthesized_basis_bytes": batch * len(scales) * time_basis * area * bytes_per_value,
            "legacy_padded_coefficient_packets_bytes": len(scales) * sum(s["coefficient_basis_bytes_per_batch"] for s in scales),
            "compiled_effective_weights_bytes": sum(s["folded_weight_bytes"] for s in scales),
            "chunk_coefficient_bytes": sum(s["chunk_coefficient_bytes_per_batch"] for s in scales),
            "output_pressure_bytes": batch * frames * (identity["active_shape"][0] + 1) * identity["active_shape"][1] * bytes_per_value,
            "memory_scope": "named FP32 tensors/Adam factors only; not peak allocator, synthesis workspaces, backbone or CUDA library memory"}


def _difference(candidate, reference):
    delta = (candidate.detach() - reference.detach()).double()
    return {"max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / reference.detach().double().norm().clamp_min(1e-30))}


def cpu_check():
    """Synthetic nondegenerate checks; this is not a benchmark or training result."""
    from vendor.backbone.spectral_detail import PolynomialPyramid
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(261119)
        checks = []
        for profile, frames in (("uniform_layered", 401), ("marmousi", 801)):
            old = JointFNOTimeDecoder(5, torch.linspace(0, 1 if frames == 401 else 4, frames),
                                     detail_transform=PolynomialPyramid((32, 33)), profile=profile)
            candidate, receipt = expand_decoder_time_basis(old, 800)
            features = torch.randn(2, 5, 33, 33)
            slots = torch.tensor([0, 1, frames // 2, frames - 2, frames - 1])
            reference = old(features, slots)
            prediction = decode_coefficient_first(candidate, features, slots, time_chunk=2)
            torch.testing.assert_close(prediction, reference, rtol=3e-5, atol=3e-6)
            original_packets = old(features, slots, return_parts=True)["detail_coefficients"]
            packets = coefficient_packets(candidate, prepare_coefficient_features(candidate, features), slots, time_chunk=2)
            for packet, original in zip(packets, original_packets):
                torch.testing.assert_close(packet, original, rtol=3e-5, atol=3e-6)
            per_record_slots = slots[None].expand(2, -1).clone()
            per_record_slots[1] = per_record_slots[1].flip(0)
            individual = coefficient_packets(candidate, features, per_record_slots, time_chunk=2)
            for packet, original in zip(individual, packets):
                torch.testing.assert_close(packet[0], original[0], rtol=3e-5, atol=3e-6)
                torch.testing.assert_close(packet[1], original[1].flip(0), rtol=3e-5, atol=3e-6)
            # The existing 800-basis decoder remains executable, but is expensive at 201².
            legacy_wide = candidate(features, slots)
            torch.testing.assert_close(prediction, legacy_wide, rtol=3e-5, atol=3e-6)
            cotangent = torch.randn_like(prediction)
            (prediction * cotangent).mean().backward()
            first_time = [float(layer.weight.grad[:, 8:].norm()) for layer in candidate.detail_time]
            first_spatial = [float(head[-1].weight.grad[8 * level.valid_detail.shape[-1]:].norm())
                             for head, level in zip(candidate.detail_heads, candidate.detail_transform.levels)]
            if not all(value > 0 and math.isfinite(value) for value in first_time):
                raise AssertionError("new nonzero spatial features did not activate new time-factor gradients")
            if any(value != 0 for value in first_spatial):
                raise AssertionError("zero new time columns should initially gate new spatial gradients")
            # One explicitly synthetic time-factor perturbation tests subsequent gradient flow.
            with torch.no_grad():
                for layer in candidate.detail_time:
                    grad = layer.weight.grad[:, 8:]
                    layer.weight[:, 8:].add_(grad / grad.norm(), alpha=-.01)
            candidate.zero_grad(set_to_none=True)
            updated = decode_coefficient_first(candidate, features, slots, time_chunk=2)
            (updated * cotangent).mean().backward()
            second_spatial = [float(head[-1].weight.grad[8 * level.valid_detail.shape[-1]:].norm())
                              for head, level in zip(candidate.detail_heads, candidate.detail_transform.levels)]
            if not all(value > 0 and math.isfinite(value) for value in second_spatial):
                raise AssertionError("new spatial factor gradients are missing after time factors become active")
            with torch.no_grad():
                compiled = compile_detail_readout(candidate)
                cached = decode_coefficient_first(candidate, features, slots, compiled=compiled)
                eager = decode_coefficient_first(candidate, features, slots)
                torch.testing.assert_close(cached, eager, rtol=3e-5, atol=3e-6)
                cached_difference = _difference(cached, eager)
                stream = io.BytesIO()
                torch.save(decoder_checkpoint_payload(candidate), stream)
                stream.seek(0)
                restored = restore_decoder_checkpoint(old, torch.load(stream, weights_only=True))
                reloaded = decode_coefficient_first(restored, features, slots)
                torch.testing.assert_close(reloaded, eager, rtol=0, atol=0)
                candidate.detail_time[0].weight.add_(1e-7)
                try:
                    decode_coefficient_first(candidate, features, slots, compiled=compiled)
                    raise AssertionError("stale cache was accepted")
                except ValueError as exc:
                    if "stale" not in str(exc):
                        raise
            checks.append({"profile": profile, "frames": frames, "cpu_grid": [33, 33],
                           "function_preservation": _difference(prediction, reference),
                           "coefficient_first_vs_wide_legacy": _difference(prediction, legacy_wide),
                           "new_time_gradient_norm_first_step": first_time,
                           "new_spatial_gradient_norm_first_step": first_spatial,
                           "new_spatial_gradient_norm_after_time_activation": second_spatial,
                           "coefficient_packets_and_per_record_slots_match": True,
                           "compiled_difference": cached_difference, "stale_cache_rejected": True,
                           "checkpoint_reload_exact_match": True,
                           "copied_factor_hashes_match": all(s["old_spatial_weight_sha256"] == s["copied_spatial_weight_sha256"]
                                                             and s["old_time_weight_sha256"] == s["copied_time_weight_sha256"]
                                                             for s in receipt["scales"])})
        costs = []
        for frames, profile in ((401, "uniform_layered"), (801, "marmousi")):
            decoder = JointFNOTimeDecoder(5, torch.linspace(0, 1 if frames == 401 else 4, frames),
                                         detail_transform=PolynomialPyramid((200, 201)), profile=profile)
            costs.append(basis_cost(decoder, 800))
        return {"schema": SCHEMA, "status": "PASS", "device": "cpu", "synthetic_checks": checks,
                "production_grid_analytic_costs": costs,
                "scope": "function/gradient/memory accounting only; no measured accuracy, GPU latency or production training"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.cpu_check:
        parser.error("use --cpu-check; production integration belongs to the main training entrypoint")
    result = cpu_check()
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
