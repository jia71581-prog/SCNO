"""Original SOAP arithmetic with reusable pinned host state and prefetch.

At public boundaries every state is on CPU and all transfers are complete.
Only one current and one prefetched parameter state reside on CUDA. Parameters
are updated by the original upstream routine in their original order.
"""
import torch

from .offloaded_soap import OffloadedRealViewSOAP
from .vendor.soap_upstream import SOAP


def _to_cuda(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_cuda(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cuda(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cuda(v, device) for v in value)
    return value


def _copy_to_host(value, host=None, *, non_blocking=False):
    if torch.is_tensor(value):
        if (not torch.is_tensor(host) or host.shape != value.shape or host.dtype != value.dtype
                or host.device.type != 'cpu' or not host.is_pinned()):
            host = torch.empty_like(value, device='cpu', pin_memory=True)
        host.copy_(value, non_blocking=non_blocking)
        if value.device.type == 'cuda' and non_blocking:
            value.record_stream(torch.cuda.current_stream(value.device))
        return host
    if isinstance(value, dict):
        old = host if isinstance(host, dict) else {}
        return {k: _copy_to_host(v, old.get(k), non_blocking=non_blocking) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        old = host if isinstance(host, (tuple, list)) else []
        items = [_copy_to_host(v, old[i] if i < len(old) else None, non_blocking=non_blocking)
                 for i, v in enumerate(value)]
        return tuple(items) if isinstance(value, tuple) else items
    return value


class PrefetchedRealViewSOAP(OffloadedRealViewSOAP):
    def __init__(self, named_parameters, **kwargs):
        super().__init__(named_parameters, **kwargs)
        self._host_buffers = {}
        self._transfer_stream = None

    def offload_state(self, proxy):
        if proxy.device.type != 'cuda':
            return super().offload_state(proxy)
        host = _copy_to_host(self.state[proxy], self._host_buffers.get(proxy))
        self._host_buffers[proxy] = host
        self.state[proxy] = host

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Use explicit backward')
        if not self.proxies or self.proxies[0].device.type != 'cuda':
            return super().step()
        for (_, param), proxy in zip(self.originals, self.proxies):
            if param is not proxy:
                if torch.view_as_real(param).data_ptr() != proxy.data_ptr():
                    raise ValueError('Parameter storage changed')
                proxy.grad = None if param.grad is None else torch.view_as_real(param.grad.resolve_conj())
        groups = self.param_groups
        pending = [(group, proxy) for group in groups for proxy in group['params'] if proxy.grad is not None]
        if not pending:
            return
        device = pending[0][1].device
        if any(proxy.device != device for _, proxy in pending):
            raise ValueError('prefetched SOAP supports one CUDA device per optimizer')
        if self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream(device=device)
        transfer = self._transfer_stream
        compute = torch.cuda.current_stream(device)

        def prefetch(proxy):
            with torch.cuda.stream(transfer):
                gpu_state = _to_cuda(self.state[proxy], device)
                ready = torch.cuda.Event()
                ready.record(transfer)
            return gpu_state, ready

        gpu_state, ready = prefetch(pending[0][1])
        try:
            for index, (group, proxy) in enumerate(pending):
                compute.wait_event(ready)
                self.state[proxy] = gpu_state
                # H2D for the following parameter overlaps current SOAP kernels.
                following = prefetch(pending[index+1][1]) if index+1 < len(pending) else None
                self.param_groups = [dict(group, params=[proxy])]
                SOAP.step(self)
                done = torch.cuda.Event()
                done.record(compute)
                with torch.cuda.stream(transfer):
                    transfer.wait_event(done)
                    host = _copy_to_host(self.state[proxy], self._host_buffers.get(proxy), non_blocking=True)
                self._host_buffers[proxy] = host
                self.state[proxy] = host
                if following is not None:
                    gpu_state, ready = following
        finally:
            # CPU checkpoint/sampler code can never observe incomplete D2H state.
            transfer.synchronize()
            self.param_groups = groups
