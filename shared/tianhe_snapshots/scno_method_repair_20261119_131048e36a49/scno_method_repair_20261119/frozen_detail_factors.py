"""A-phase constant folding with full gradients through current spatial features.

Only permanently frozen-for-this-phase output affine/time factors are cached.
Features, hidden activations, labels and contexts are never cached or detached.
Any factor update, replacement, device/dtype move or unfreezing invalidates the
cache. The caller returns to the normal dynamic head in phases B/C.
"""

import torch

from joint_fno_decoder import _frame_indices
from wide_time_basis import _factor_signature, _fold_factors, prepare_coefficient_features


class FrozenDetailFactors:
    """Cache T x slot x hidden folded affines from frozen output factors only."""

    def __init__(self, decoder):
        self.decoder = decoder
        self._parameters = tuple(
            p for head, temporal in zip(decoder.detail_heads, decoder.detail_time, strict=True)
            for p in (head[-1].weight, head[-1].bias, temporal.weight) if p is not None)
        if not self._parameters or any(p.requires_grad for p in self._parameters):
            raise ValueError("freeze every spatial output affine and time matrix before folding")
        self._signature = _factor_signature(decoder)
        self._axis_signature = (
            id(decoder.native_times), decoder.native_times._version, tuple(decoder.native_times.shape),
            str(decoder.native_times.device), str(decoder.native_times.dtype), decoder.detail_latent)
        self._layout = tuple(
            (level.gz, level.gx, level.valid_detail.shape[-1],
             id(level.valid_detail), level.valid_detail._version)
            for level in decoder.detail_transform.levels)
        with torch.no_grad():
            self.factors = tuple((weight.detach(), bias.detach() if bias is not None else None)
                                 for weight, bias in _fold_factors(decoder))
        if any(not torch.isfinite(value).all().item()
               for pair in self.factors for value in pair if value is not None):
            raise ValueError("frozen output factors must be finite")
        self.storage_bytes = sum(
            value.numel() * value.element_size()
            for pair in self.factors for value in pair if value is not None)
        self.receipt = {
            "schema": "SCNO_FROZEN_DETAIL_FACTOR_FOLD_V1",
            "native_frames": len(decoder.native_times), "time_basis": decoder.detail_latent,
            "factor_storage_bytes": self.storage_bytes,
            "gradient_scope": "current features, feature projection and spatial hidden layers",
            "cached_scope": "frozen output/time parameter products only",
            "features_contexts_or_labels_cached": False,
        }

    def check_frozen(self):
        """Reject stale caches rather than silently dropping trainable gradients."""
        if any(p.requires_grad for p in self._parameters):
            raise RuntimeError("cached output factors were unfrozen; use the dynamic detail head")
        if _factor_signature(self.decoder) != self._signature:
            raise RuntimeError("output factor identity/version/device/dtype changed; rebuild or use the dynamic head")
        axis = self.decoder.native_times
        if (id(axis), axis._version, tuple(axis.shape), str(axis.device), str(axis.dtype),
                self.decoder.detail_latent) != self._axis_signature:
            raise RuntimeError("native time-axis or time-basis definition changed")
        current_layout = tuple(
            (level.gz, level.gx, level.valid_detail.shape[-1], id(level.valid_detail), level.valid_detail._version)
            for level in self.decoder.detail_transform.levels)
        if current_layout != self._layout:
            raise RuntimeError("fixed detail slot geometry/masks changed")
        return True

    def packets(self, features, indices=None, *, time_chunk=None):
        """Return native valid coefficient packets with the current feature graph.

        features is [B,C,Z,X] including the free surface. indices supports shared
        [K] and per-record [B,K] native integer queries. This method intentionally
        is NOT decorated no_grad: frozen linear maps still differentiate inputs.
        """
        self.check_frozen()
        if not torch.is_tensor(features):
            raise TypeError("pass current spatial features, never cached hidden/label tensors")
        hidden = prepare_coefficient_features(self.decoder, features)
        batch, device = hidden[0].shape[0], hidden[0].device
        slots = _frame_indices(indices, batch, device)
        count = len(self.decoder.native_times) if slots is None else slots.shape[-1]
        if slots is not None and slots.numel() and (
                slots.min().item() < 0 or slots.max().item() >= len(self.decoder.native_times)):
            raise ValueError("native time indices are outside the unchanged axis")
        chunk = max(1, count) if time_chunk is None else time_chunk
        if type(chunk) is not int or chunk < 1:
            raise ValueError("time_chunk must be a positive integer")
        packets = []
        for value, (weights, biases), level in zip(
                hidden, self.factors, self.decoder.detail_transform.levels, strict=True):
            pieces = []
            for first in range(0, count, chunk):
                end = min(first + chunk, count)
                selected = (slice(first, end) if slots is None else slots[..., first:end])
                weight = weights[selected]
                bias = biases[selected] if biases is not None else None
                if slots is None or slots.ndim == 1:
                    coefficient = torch.einsum("bxyh,ksh->bkxys", value, weight)
                    if bias is not None:
                        coefficient = coefficient + bias[None, :, None, None]
                else:
                    coefficient = torch.einsum("bxyh,bksh->bkxys", value, weight)
                    if bias is not None:
                        coefficient = coefficient + bias[:, :, None, None]
                pieces.append(coefficient.masked_fill(~level.valid_detail, 0))
            packets.append(torch.cat(pieces, dim=1) if pieces else value.new_empty(
                (batch, 0, level.gz, level.gx, level.valid_detail.shape[-1])))
        return tuple(packets)


def run_cpu_checks():
    """Actual wide decoder/fixed transform: pressure, input and upstream gradients."""
    import importlib.util
    from pathlib import Path
    import sys
    from torch import nn
    from torch.nn import functional as F
    from joint_fno_decoder import JointFNOTimeDecoder, _time_projection
    from wide_time_basis import expand_decoder_time_basis, coefficient_packets

    specification = importlib.util.spec_from_file_location(
        "_frozen_factors_cpu_spectral", Path(__file__).parent / "vendor/backbone/spectral_detail.py")
    spectral = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = spectral
    specification.loader.exec_module(spectral)

    def pressure(decoder, features, indices, coefficients):
        hidden = features[:, :, 1:]
        for layer in list(decoder.proj.children())[:-1]:
            hidden = layer(hidden)
        final = decoder.proj[-1]
        base = _time_projection(hidden, final.weight[:, :, 0, 0], final.bias, indices)
        last = decoder.detail_transform.levels[-1]
        coarse = features.new_zeros((len(features), coefficients[0].shape[1], last.gz * 4, last.gx * 4))
        details = decoder.detail_transform.synthesize(coarse, coefficients)
        return F.pad(base + details, (0, 0, 1, 0))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7817)
        decoder = JointFNOTimeDecoder(
            5, torch.linspace(0, 1, 17),
            detail_transform=spectral.PolynomialPyramid((16, 17), merges=1))
        decoder, _ = expand_decoder_time_basis(decoder, 800)
        decoder.requires_grad_(False)
        frozen = FrozenDetailFactors(decoder)
        feature = torch.randn(2, 5, 17, 17, requires_grad=True)
        indices = torch.tensor([[0, 3, 5, 12, 16], [1, 4, 7, 11, 15]])
        reference_packets = coefficient_packets(decoder, feature, indices)
        reference = pressure(decoder, feature, indices, reference_packets)
        cotangent = torch.randn_like(reference)
        reference_gradient = torch.autograd.grad((reference * cotangent).sum(), feature)[0]
        actual_packets = frozen.packets(feature, indices)
        actual = pressure(decoder, feature, indices, actual_packets)
        actual_gradient = torch.autograd.grad((actual * cotangent).sum(), feature)[0]
        assert torch.allclose(reference, actual, rtol=2e-5, atol=1e-6)
        assert torch.allclose(reference_gradient, actual_gradient, rtol=3e-5, atol=2e-6)
        # Non-folded feature projection and hidden parameters may be trainable;
        # the cached constants must still preserve their input-side graph.
        decoder.detail_projection.weight.requires_grad_(True)
        decoder.detail_heads[0][0].weight.requires_grad_(True)
        upstream = (feature, decoder.detail_projection.weight, decoder.detail_heads[0][0].weight)
        dynamic = coefficient_packets(decoder, feature, indices)
        dynamic_loss = sum(packet.square().sum() for packet in dynamic)
        dynamic_gradients = torch.autograd.grad(dynamic_loss, upstream)
        cached = frozen.packets(feature, indices, time_chunk=2)
        cached_loss = sum(packet.square().sum() for packet in cached)
        cached_gradients = torch.autograd.grad(cached_loss, upstream)
        assert all(torch.allclose(a, b, rtol=5e-5, atol=3e-6)
                   for a, b in zip(dynamic_gradients, cached_gradients, strict=True))
        assert all(gradient.count_nonzero().item() for gradient in cached_gradients)
        for query in (None, torch.tensor([0, 8, 16]), torch.empty(0, dtype=torch.long)):
            expected = coefficient_packets(decoder, feature, query)
            candidate = frozen.packets(feature, query)
            assert all(torch.allclose(a, b, rtol=2e-5, atol=1e-6)
                       for a, b in zip(expected, candidate, strict=True))
        factor = decoder.detail_time[0].weight
        factor.requires_grad_(True)
        try:
            frozen.check_frozen()
        except RuntimeError:
            pass
        else:
            raise AssertionError("unfrozen output factor used a detached cache")
        factor.requires_grad_(False)
        with torch.no_grad():
            factor.add_(.01)
        try:
            frozen.check_frozen()
        except RuntimeError:
            pass
        else:
            raise AssertionError("mutated factor used stale products")
        refreshed = FrozenDetailFactors(decoder)
        decoder.detail_time[0].weight = nn.Parameter(factor.detach().clone(), requires_grad=False)
        try:
            refreshed.check_frozen()
        except RuntimeError:
            pass
        else:
            raise AssertionError("replaced Parameter used stale products")
        return {
            "passed": True, "torch_version_actually_tested": torch.__version__,
            "time_basis": 800, "pressure_max_abs_difference": (reference - actual).abs().max().item(),
            "feature_gradient_max_abs_difference": (reference_gradient - actual_gradient).abs().max().item(),
            "upstream_projection_hidden_gradients_preserved": True,
            "shared_per_record_full_empty_and_chunked_queries_match": True,
            "unfreeze_mutation_and_replacement_rejected": True,
            "no_feature_context_or_label_cache": True,
            "test_factor_storage_bytes": frozen.storage_bytes, "speedup_measured": False,
        }


if __name__ == "__main__":
    import json
    import sys
    if sys.argv[1:] == ["--cpu-check"]:
        torch.set_num_threads(2)
        print(json.dumps(run_cpu_checks(), indent=2))
    else:
        print("Import FrozenDetailFactors; --cpu-check performs CPU-only wide-head checks.")
