"""SOAP on explicit real/imaginary parameter views, preserving model storage.

The numerical update is the pinned upstream SOAP algorithm. Compatibility
changes: real views for complex tensors, strict view metadata on restore,
and curvature/eigensolvers preserving each parameter's FP32/FP64 dtype.
"""
import copy
import torch
from .vendor.soap_upstream import SOAP


class RealViewSOAP(SOAP):
    def __init__(self, named_parameters, **kwargs):
        if kwargs.get("merge_dims", False):
            raise ValueError("This audited adapter supports merge_dims=False")
        self.originals = list(named_parameters)
        self.proxies = []
        self.signature = []
        for name, p in self.originals:
            if not p.requires_grad:
                raise ValueError("Pass only trainable parameters")
            proxy = torch.nn.Parameter(torch.view_as_real(p.detach())) if p.is_complex() else p
            self.proxies.append(proxy)
            self.signature.append(dict(name=name, shape=list(p.shape), dtype=str(p.dtype),
                                       real_shape=list(proxy.shape), real_dtype=str(proxy.dtype)))
        super().__init__(self.proxies, **kwargs)

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for (_, p), proxy in zip(self.originals, self.proxies):
            if p is not proxy:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Use an explicit backward before SOAP.step")
        for (_, p), proxy in zip(self.originals, self.proxies):
            if p is not proxy:
                if torch.view_as_real(p).data_ptr() != proxy.data_ptr():
                    raise ValueError("Model storage changed after optimizer construction")
                proxy.grad = None if p.grad is None else torch.view_as_real(p.grad.resolve_conj())
        return super().step()

    def state_dict(self):
        result = super().state_dict()
        result["real_view_schema"] = "REAL_VIEW_SOAP_V1"
        result["real_view_signature"] = copy.deepcopy(self.signature)
        return result

    def load_state_dict(self, state_dict):
        if state_dict.get("real_view_schema") != "REAL_VIEW_SOAP_V1" or state_dict.get("real_view_signature") != self.signature:
            raise ValueError("SOAP real/complex parameter identity mismatch")
        state_dict = {k: v for k, v in state_dict.items() if k not in ("real_view_schema", "real_view_signature")}
        super().load_state_dict(state_dict)

    def init_preconditioner(self, grad, state, **kwargs):
        super().init_preconditioner(grad, state, **kwargs)
        state["GG"] = [m.to(dtype=grad.dtype) if torch.is_tensor(m) else m for m in state["GG"]]

    def get_orthogonal_matrix(self, matrices):
        result = []
        for m in matrices:
            if not torch.is_tensor(m):
                result.append([])
                continue
            eye = torch.eye(m.shape[0], device=m.device, dtype=m.dtype)
            try:
                _, q = torch.linalg.eigh(m + 1e-30 * eye)
            except torch.linalg.LinAlgError:
                # Same fallback as upstream; do not lose FP64 local-path precision.
                _, q = torch.linalg.eigh(m.double() + 1e-30 * eye.double())
                q = q.to(m.dtype)
            result.append(torch.flip(q, [1]))
        return result

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        if merge_dims:
            raise ValueError("Unsupported merged covariance")
        result, variance = [], state["exp_avg_sq"]
        for axis, (m, q) in enumerate(zip(state["GG"], state["Q"])):
            if not torch.is_tensor(m):
                result.append([])
                continue
            ordering = torch.argsort(torch.diag(q.T @ m @ q), descending=True)
            variance = variance.index_select(axis, ordering)
            q = q[:, ordering]
            q, _ = torch.linalg.qr(m @ q)
            result.append(q)
        state["exp_avg_sq"] = variance
        return result


def finite_state(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(v) for v in value)
    return True
