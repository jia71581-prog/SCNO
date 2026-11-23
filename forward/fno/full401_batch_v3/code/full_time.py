"""One complete401-frame prediction and backward, with bounded loss temporaries."""
import torch
from torch.autograd.function import once_differentiable


class _WeightedFullTimeMSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction, target, weights):
        assert prediction.ndim == 4 and prediction.shape[1:] == (401, 201, 201)
        assert target.shape == prediction.shape and weights.shape == (401,)
        assert prediction.dtype == target.dtype == weights.dtype == torch.float32
        assert not target.requires_grad and not weights.requires_grad
        # One complete CPU->GPU transfer. This owned buffer becomes the residual;
        # a separate full-size squared-error tensor is unnecessary.
        residual = torch.empty_like(prediction)
        residual.copy_(target, non_blocking=True)
        residual.neg_().add_(prediction)
        per_time_norm = torch.linalg.vector_norm(residual, dim=(0, 2, 3))
        loss = (per_time_norm.square() * weights).sum() / residual.numel()
        ctx.save_for_backward(residual, weights)
        ctx.elements = residual.numel()
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, incoming):
        residual, weights = ctx.saved_tensors
        # First-order supervised Adam only; the complete temporal gradient is
        # passed to the original model in one backward call.
        residual.mul_(weights[None, :, None, None])
        residual.mul_(2. / ctx.elements).mul_(incoming)
        return residual, None, None


def backward_full401(model, inputs, target_cpu, weights):
    prediction = model(inputs)
    loss = _WeightedFullTimeMSE.apply(prediction, target_cpu, weights)
    del prediction
    loss.backward()
    return loss.detach()
