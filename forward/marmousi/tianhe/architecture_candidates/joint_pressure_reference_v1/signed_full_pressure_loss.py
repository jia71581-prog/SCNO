"""Independent signed pressure-null and released B5 supervision."""

from dataclasses import dataclass
from math import prod

import torch

from objectives import lowpass_time


SCHEMA = "scno_signed_full_pressure_and_b5_energy_v1"
NATIVE_FRAMES = 801


@dataclass(frozen=True)
class SignedEnergies:
    """Additive energy sums, with an independently backpropagatable SSE."""

    squared_error_sum: torch.Tensor
    reference_squared_sum: torch.Tensor
    predicted_squared_sum: torch.Tensor
    signed_inner_product: torch.Tensor
    element_weight_sum: torch.Tensor

    def __add__(self, other):
        return SignedEnergies(self.squared_error_sum + other.squared_error_sum,
                              self.reference_squared_sum + other.reference_squared_sum,
                              self.predicted_squared_sum + other.predicted_squared_sum,
                              self.signed_inner_product + other.signed_inner_product,
                              self.element_weight_sum + other.element_weight_sum)

    @property
    def reference_mean_square(self):
        """Use pooled ordinary TRAIN sums to freeze one scale for each objective."""
        return self.reference_squared_sum/self.element_weight_sum

    def normalized(self, fixed_train_mean_square):
        """Return weighted mean SSE divided by a precomputed TRAIN mean square.

        Validation ratios can separately use the additive error/reference
        sums. This training reduction does not divide by an individual pair
        or the current minibatch's directional response energy.
        """
        scale = torch.as_tensor(fixed_train_mean_square, device=self.squared_error_sum.device,
                                dtype=torch.float64).detach()
        if scale.ndim or not (torch.isfinite(scale) & (scale > 0.)):
            raise ValueError("Use one positive finite prebound TRAIN mean square")
        return self.squared_error_sum/(self.element_weight_sum*scale)


def _metadata(value, leading_shape, device, *, description, positive):
    field = torch.as_tensor(value, dtype=torch.float64, device=device).detach()
    if field.ndim > len(leading_shape):
        raise ValueError(f"{description} needs sample axes only")
    # A single [pairs] vector applies to all shots within each pair. Use
    # explicit [1,shots] for per-shot weights shared by several pairs.
    field = field.reshape(field.shape+(1,)*(len(leading_shape)-field.ndim))
    try:
        field = torch.broadcast_to(field, leading_shape)
    except RuntimeError as exc:
        raise ValueError(f"{description} does not match the leading pair and shot axes") from exc
    valid = field > 0. if positive else field >= 0.
    if not (torch.isfinite(field) & valid).all() or not (field.sum() > 0.):
        raise ValueError(f"Use finite {description} with positive total weight")
    return field


def _paired_secants(predicted_minus, predicted_plus, reference_minus, reference_plus, half_step, trailing_dimensions):
    frames = tuple(torch.as_tensor(item) for item in (predicted_minus, predicted_plus, reference_minus, reference_plus))
    sample = frames[0]
    if (sample.ndim < trailing_dimensions or not all(sample.shape)
            or any(x.shape != sample.shape or x.device != sample.device or not x.is_floating_point() for x in frames)):
        raise ValueError("Each endpoint must use the same pressure or B5 axes, device and real pressure units")
    leading = sample.shape[:-trailing_dimensions]
    h = _metadata(half_step, leading, sample.device, description="positive perturbation half-step", positive=True)
    divisor = 2.*h.reshape((*leading, *(1,)*trailing_dimensions))
    pminus, pplus, rminus, rplus = frames
    predicted = (pplus.double()-pminus.double())/divisor
    target = (rplus.detach().double()-rminus.detach().double())/divisor
    return predicted, target


def _energy(predicted, target, trailing_dimensions, sample_weights):
    leading = predicted.shape[:-trailing_dimensions]
    samples, elements = prod(leading), prod(predicted.shape[-trailing_dimensions:])
    weight = _metadata(1. if sample_weights is None else sample_weights, leading, predicted.device,
                       description="nonnegative sample weights", positive=False).reshape(samples)
    predicted, target = (value.reshape(samples, elements) for value in (predicted, target))
    error = ((predicted-target).square().sum(-1)*weight).sum()
    with torch.no_grad():
        reference = (target.square().sum(-1)*weight).sum()
        amplitude = (predicted.square().sum(-1)*weight).sum()
        inner = ((predicted*target).sum(-1)*weight).sum()
        mass = weight.sum()*elements
        if not torch.isfinite(torch.stack((error.detach(), reference, amplitude, inner, mass))).all():
            raise ValueError("The signed observation or pressure energy is nonfinite")
    return SignedEnergies(error, reference, amplitude, inner, mass)


def body_signed_energies(lifter, predicted_minus, predicted_plus, reference_minus, reference_plus, half_step,
                         *, sample_weights=None):
    """Score only the signed full-pressure component invisible to frozen B5.

    Inputs are matching [...,time,z,x] pressure blocks from the same ordinary
    parent direction, physical shot IDs and exact frame IDs. The caller binds
    these identities. Pressure is in a common TRAIN scale. Use the actual
    archive half-step h for that direction's coordinate.

    Subtract both endpoint pairs in FP64 and use PressureObservationLift's
    exact B5 null projection on each centered secant. Consequently adding
    any frozen-B5 minimum-norm lift to the predicted pressure has zero effect
    on this loss. No raw-pressure receiver scale or pressure-to-B5 coupling is
    estimated here. Pressure blocks can be scored and accumulated in chunks.
    """
    predicted, target = _paired_secants(predicted_minus, predicted_plus, reference_minus, reference_plus,
                                        half_step, trailing_dimensions=3)
    body = lifter.project_null(predicted)
    with torch.no_grad():
        physical_body = lifter.project_null(target)
    return _energy(body, physical_body, 3, sample_weights)


def b5_signed_energies(predicted_minus, predicted_plus, released_minus, released_plus, half_step,
                       *, sample_weights=None, response=None):
    """Score a separate B5 branch against the released receiver endpoints.

    Use aligned [...,time,receivers] predictions from the actual receiver
    readout and archived B5 values, in identical TRAIN-normalized units and
    with the same perturbation half-step as the pressure pair. The receiver
    target comes from the committed physical B5 archive.

    A supplied response uses the original full 801-frame FFT contract. Any
    time slicing for a full-band local loss must occur with response=None.
    """
    predicted, target = _paired_secants(predicted_minus, predicted_plus, released_minus, released_plus,
                                        half_step, trailing_dimensions=2)
    if response is not None:
        if predicted.shape[-2] != NATIVE_FRAMES:
            raise ValueError("Band-limited B5 secants require the original complete 801-frame time axis")
        shape = predicted.shape
        weighted_response = torch.as_tensor(response, device=predicted.device, dtype=torch.float64).detach()
        predicted, target = (lowpass_time(value.reshape(-1, *shape[-2:]), weighted_response).reshape(shape)
                             for value in (predicted, target))
    return _energy(predicted, target, 2, sample_weights)
