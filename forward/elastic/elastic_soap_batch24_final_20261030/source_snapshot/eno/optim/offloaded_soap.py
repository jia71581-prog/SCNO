"""The original RealViewSOAP math, with inactive optimizer state held on CPU."""
import torch
from .soap_complex import RealViewSOAP
from .vendor.soap_upstream import SOAP


def _move(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:_move(v,device) for k,v in value.items()}
    if isinstance(value,list):return [_move(v,device) for v in value]
    if isinstance(value,tuple):return tuple(_move(v,device) for v in value)
    return value


class OffloadedRealViewSOAP(RealViewSOAP):
    def offload_state(self,proxy):
        self.state[proxy]=_move(self.state[proxy],'cpu')

    def load_state_dict(self,state):
        super().load_state_dict(state)
        for proxy in self.proxies:self.offload_state(proxy)

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('Use explicit backward')
        for (_,p),proxy in zip(self.originals,self.proxies):
            if p is not proxy:
                if torch.view_as_real(p).data_ptr()!=proxy.data_ptr():raise ValueError('Parameter storage changed')
                proxy.grad=None if p.grad is None else torch.view_as_real(p.grad.resolve_conj())
        groups=self.param_groups
        try:
            for group in groups:
                for proxy in group['params']:
                    if proxy.grad is None:continue
                    self.state[proxy]=_move(self.state[proxy],proxy.device)
                    self.param_groups=[dict(group,params=[proxy])]
                    SOAP.step(self)
                    self.offload_state(proxy)
        finally:self.param_groups=groups
