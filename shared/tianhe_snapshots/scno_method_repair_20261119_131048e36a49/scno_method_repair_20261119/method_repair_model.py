"""Isolated, versioned SCNO readout repairs for real TRAIN-fit comparisons.

Old learned modules remain intact. New nonlinear corrections use physical-time
features and fixed orthogonal synthesis, with a separately named coarsest base
correction. A wide-basis alternative is supplied by wide_time_basis.py.
"""

from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

from time_conditioned_detail import (
    TimeConditionedDetailHead, TimeConditionedCoarsestBaseHead)


class CorrectedSCNOReadout(nn.Module):
    variant = "scno_time_conditioned_repair_v1"

    def __init__(self, parent, *, nonlinear_details=True, coarsest_base=True,
                 time_chunk=128, hidden_width=32, basis_size=None, restore_basis=False,
                 local_phase=False, base_hidden_width=None, restore_base=False):
        super().__init__()
        self.parent = parent
        self.nonlinear_details = bool(nonlinear_details)
        self.coarsest_base = bool(coarsest_base)
        self.time_chunk = int(time_chunk)
        self.hidden_width = int(hidden_width)
        self.local_phase = bool(local_phase)
        self.basis_size = basis_size
        self.base_hidden_width = base_hidden_width
        self.base_expansion_receipt = None
        self.compiled_readout = None
        self.basis_expansion_receipt = None
        if basis_size is not None:
            if nonlinear_details:
                raise ValueError("compare the factorized and nonlinear heads in separate candidates")
            if restore_basis:
                if parent.decoder.detail_latent != basis_size or parent.detail_latent != basis_size:
                    raise ValueError("restored parent has a different time-basis definition")
            else:
                from wide_time_basis import expand_decoder_time_basis
                decoder, receipt = expand_decoder_time_basis(
                    parent.decoder, basis_size, source_checkpoint_semantics=parent.checkpoint_semantics())
                parent.decoder = decoder
                parent.detail_latent = basis_size
                self.basis_expansion_receipt = receipt
        if base_hidden_width is not None:
            from wide_base_readout import expand_base_readout
            if restore_base:
                if parent.decoder.proj[-3].out_channels != base_hidden_width:
                    raise ValueError("restored BASE has a different hidden-width meaning")
            else:
                decoder, receipt = expand_base_readout(
                    parent.decoder, base_hidden_width, source_checkpoint_semantics=parent.checkpoint_semantics())
                parent.decoder = decoder
                self.base_expansion_receipt = receipt
        channels = parent.decoder.in_channels
        temporal_features = parent.gabor_embedding.in_features
        kwargs = dict(hidden_width=hidden_width, time_chunk=time_chunk)
        detail_type, coarse_type = TimeConditionedDetailHead, TimeConditionedCoarsestBaseHead
        if self.local_phase:
            if not nonlinear_details:
                raise ValueError("local phase belongs to the nonlinear candidate")
            from local_phase_features import LocalPhaseDetailHead, LocalPhaseCoarsestBaseHead
            detail_type, coarse_type = LocalPhaseDetailHead, LocalPhaseCoarsestBaseHead
            kwargs["balance_spatial_input"] = True
        self.detail_head = (detail_type(
            channels, temporal_features, parent.detail_transform, **kwargs).to(parent.device)
            if nonlinear_details else None)
        self.coarse_head = (coarse_type(
            channels, temporal_features, parent.detail_transform, **kwargs).to(parent.device)
            if coarsest_base else None)
        self.repair_receipt = {"variant": self.variant,
                               "new_modules": [name for name in ("detail_head", "coarse_head")
                                               if getattr(self, name) is not None]}

    @property
    def device(self):
        return self.parent.device

    @property
    def normalizer(self):
        return self.parent.normalizer

    @property
    def full_times_s(self):
        return self.parent.full_times_s

    @property
    def duration_s(self):
        return self.parent.duration_s

    @property
    def source_encoder(self):
        return self.parent.source_encoder

    @property
    def detail_transform(self):
        return self.parent.detail_transform

    def _inputs(self, velocity, source):
        return self.parent._inputs(velocity, source)

    def _encode_validated(self, velocity, source, **kwargs):
        return self.parent._encode_validated(velocity, source, **kwargs)

    def encode(self, velocity, source):
        return self.parent.encode(velocity, source)

    def temporal_features(self, context, slots=None):
        source = context.source
        times = self.parent._selected_times(slots, len(source))
        bank = self.parent.gabor(times, source[:, 3:4], source[:, 2:3], self.duration_s)
        if self.local_phase:
            from local_phase_features import pack_temporal_phase_features
            bank = pack_temporal_phase_features(bank, times, source)
        return bank

    def conditioned_spatial(self, context):
        if not self.local_phase:
            return context.features
        from local_phase_features import pack_spatial_phase_features
        return pack_spatial_phase_features(context.features, context.propagation.travel_times, context.source)

    def _new_coefficients(self, context, temporal):
        features = self.conditioned_spatial(context)
        if self.detail_head is not None:
            details = self.detail_head(features, temporal, time_chunk=self.time_chunk)
        else:
            details = None
        if self.basis_size is not None:
            from wide_time_basis import coefficient_packets
            details = coefficient_packets(
                self.parent.decoder, context.features, time_chunk=self.time_chunk,
                compiled=self.compiled_readout if not torch.is_grad_enabled() else None)
        coarse = (self.coarse_head(features, temporal, time_chunk=self.time_chunk)[0]
                  if self.coarse_head is not None else None)
        return coarse, details

    def synthesize_correction(self, coarse, details, *, batch, count, device_tensor):
        last = self.detail_transform.levels[-1]
        if coarse is None:
            coarse_field = device_tensor.new_zeros(batch * count, 1, last.gz * 4, last.gx * 4)
        else:
            coarse_field = coarse.reshape(batch * count, 1, last.gz, last.gx, 4, 4)
            coarse_field = coarse_field.transpose(-3, -2).reshape(batch * count, 1, last.gz * 4, last.gx * 4)
        if details is None:
            packets = [device_tensor.new_zeros(batch * count, 1, level.gz, level.gx,
                                               level.valid_detail.shape[-1])
                       for level in self.detail_transform.levels]
        else:
            packets = [value.reshape(batch * count, 1, level.gz, level.gx, -1)
                       for value, level in zip(details, self.detail_transform.levels)]
        interior = self.detail_transform.synthesize(coarse_field, packets)
        return F.pad(interior.reshape(batch, count, *self.detail_transform.shape), (0, 0, 1, 0))

    def decode(self, context, frame_indices=None, *, normalized=True,
               return_parts=False, detail_enabled=True):
        from joint_fno_decoder import _frame_indices
        slots = _frame_indices(frame_indices, len(context.source), self.device)
        inherited = self.parent.decode(context, slots, normalized=True,
                                       detail_enabled=detail_enabled and self.basis_size is None)
        temporal = self.temporal_features(context, slots)
        if not detail_enabled:
            details = None
            coarse = (self.coarse_head(self.conditioned_spatial(context), temporal,
                                       time_chunk=self.time_chunk)[0]
                      if self.coarse_head is not None and not self.coarse_head.calibration_required else None)
        elif self.basis_size is not None and slots is not None:
            from wide_time_basis import coefficient_packets
            details = coefficient_packets(self.parent.decoder, context.features, slots,
                                          time_chunk=self.time_chunk,
                                          compiled=self.compiled_readout if not torch.is_grad_enabled() else None)
            coarse = (self.coarse_head(context.features, temporal, time_chunk=self.time_chunk)[0]
                      if self.coarse_head is not None else None)
        else:
            coarse, details = self._new_coefficients(context, temporal)
        correction = self.synthesize_correction(
            coarse, details, batch=len(context.source), count=temporal.shape[1],
            device_tensor=context.features)
        source = context.source
        prediction = inherited + correction * (source[:, 4] != 0)[:, None, None, None]
        unit_factor = ((self.normalizer.metadata.pressure_scale_pa * source[:, 4])[:, None, None, None]
                       if not normalized else 1.)
        prediction = prediction * unit_factor
        if return_parts:
            return {"prediction": prediction, "inherited_pressure": inherited * unit_factor,
                    "coefficient_units": "normalized_pressure", "pressure_units": "normalized" if normalized else "Pa",
                    "base_coarsest_coefficients": coarse, "new_detail_coefficients": details}
        return prediction

    def forward(self, velocity, source, frame_indices=None, **kwargs):
        return self.decode(self.encode(velocity, source), frame_indices, **kwargs)

    def train(self, mode=True):
        if mode:
            self.compiled_readout = None
        return super().train(mode)

    @torch.no_grad()
    def prepare_inference(self):
        self.eval()
        if self.basis_size is not None:
            from wide_time_basis import compile_detail_readout
            self.compiled_readout = compile_detail_readout(self.parent.decoder)
        return self

    def checkpoint_payload(self):
        from wide_base_readout import base_decoder_checkpoint_payload
        return {"schema": self.variant, "parent_semantics": self.parent.checkpoint_semantics(),
                "parent_encoder_trainable": self.parent.encoder_trainable,
                "trainable_parameter_names": [name for name, p in self.named_parameters() if p.requires_grad],
                "repair_settings": {"nonlinear_details": self.nonlinear_details,
                                    "coarsest_base": self.coarsest_base,
                                    "time_chunk": self.time_chunk,
                                    "basis_size": self.basis_size,
                                    "hidden_width": self.hidden_width,
                                    "local_phase": self.local_phase,
                                    "base_hidden_width": self.base_hidden_width},
                "base_component": (base_decoder_checkpoint_payload(
                    self.parent.decoder, model_semantics=self.parent.checkpoint_semantics())
                    if self.base_hidden_width is not None else None),
                "base_expansion_receipt": deepcopy(self.base_expansion_receipt),
                "basis_expansion_receipt": deepcopy(self.basis_expansion_receipt),
                "conditioning_receipt": deepcopy(getattr(self.parent, "readout_conditioning_receipt", None)),
                "calibration_receipts": {name: deepcopy(getattr(getattr(self, name), "calibration_receipt", None))
                                         for name in ("detail_head", "coarse_head")},
                "repair_receipt": deepcopy(self.repair_receipt), "model": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, native_template, payload):
        """Restore the saved computation, without gain absorption or calibration."""
        from joint_fno_transport_model import TravelTimeFNOSCNO
        if payload.get("schema") != cls.variant:
            raise ValueError("expected a separately versioned SCNO readout repair")
        state = payload["model"]
        parent_state = {name[len("parent."):]: value for name, value in state.items()
                        if name.startswith("parent.")}
        if payload.get("base_component") is None:
            parent = TravelTimeFNOSCNO.from_checkpoint(
                native_template, {"model_semantics": payload["parent_semantics"], "model": parent_state,
                                  "encoder_trainable": payload.get("parent_encoder_trainable", True)})
        else:
            from wide_base_readout import restore_base_decoder_checkpoint
            semantics = payload["parent_semantics"]
            parent = TravelTimeFNOSCNO.from_native(
                native_template, freeze_encoder=not payload.get("parent_encoder_trainable", True),
                detail_width=semantics["detail_width"], detail_latent=semantics["detail_latent"])
            if parent.checkpoint_semantics() != semantics:
                raise ValueError("the expanded BASE parent has a different physical or model binding")
            parent.decoder = restore_base_decoder_checkpoint(
                parent.decoder, payload["base_component"], expected_model_semantics=semantics)
            parent.load_state_dict(parent_state, strict=True)
        if float(parent.residual_gain.detach()) != 1.:
            raise ValueError("saved repaired readout must have its already absorbed unit gain")
        parent.residual_gain.requires_grad_(False)
        settings = dict(payload["repair_settings"])
        model = cls(parent, **settings, restore_basis=settings.get("basis_size") is not None,
                    restore_base=settings.get("base_hidden_width") is not None)
        model.load_state_dict(state, strict=True)
        if "trainable_parameter_names" in payload:
            saved_names = set(payload["trainable_parameter_names"])
            if not saved_names.issubset(dict(model.named_parameters())):
                raise ValueError("saved optimizer roles refer to missing model parameters")
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name in saved_names)
        model.repair_receipt = deepcopy(payload["repair_receipt"])
        model.basis_expansion_receipt = deepcopy(payload.get("basis_expansion_receipt"))
        model.base_expansion_receipt = deepcopy(payload.get("base_expansion_receipt"))
        if payload.get("conditioning_receipt") is not None:
            model.parent.readout_conditioning_receipt = deepcopy(payload["conditioning_receipt"])
        for name, receipt in payload.get("calibration_receipts", {}).items():
            if getattr(model, name) is not None:
                getattr(model, name).calibration_receipt = deepcopy(receipt)
        return model
