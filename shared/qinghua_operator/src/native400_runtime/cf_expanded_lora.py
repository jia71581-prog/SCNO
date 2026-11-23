"""Add zero-initialized low-rank weight updates to Linear and Conv2d layers.

Expose effective weights as well as forward methods: native dense/attention
paths sometimes consume a layer's weight directly. Original tensors stay frozen.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LinearLoRA(nn.Module):
    def __init__(self, base, rank, generator):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank = min(rank, self.in_features, self.out_features)
        self.A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.A.data.uniform_(-1 / math.sqrt(self.in_features), 1 / math.sqrt(self.in_features), generator=generator)
        self.B = nn.Parameter(torch.zeros(self.out_features, self.rank))

    @property
    def weight(self):
        return self.base.weight + self.B @ self.A

    @property
    def bias(self):
        return self.base.bias

    def forward(self, value):
        return F.linear(value, self.weight, self.bias)


class ConvLoRA(nn.Module):
    def __init__(self, base, rank, generator, *, active_columns=None):
        super().__init__()
        self.base = base
        self.in_channels, self.out_channels = base.in_channels, base.out_channels
        self.kernel_size, self.stride = base.kernel_size, base.stride
        self.padding, self.dilation, self.groups = base.padding, base.dilation, base.groups
        self.padding_mode = base.padding_mode
        self.flat = base.weight.numel() // self.out_channels
        legacy_rank = min(rank, self.flat, self.out_channels)
        columns = torch.arange(self.flat) if active_columns is None else torch.as_tensor(active_columns,dtype=torch.long)
        self.register_buffer('active_columns',columns,persistent=False)
        self.rank = legacy_rank if columns.numel() else 0
        # Draw the legacy tensor before selecting columns to keep all later
        # adapters under the same seed unchanged.
        initial=torch.empty(legacy_rank,self.flat)
        initial.uniform_(-1/math.sqrt(self.flat),1/math.sqrt(self.flat),generator=generator)
        self.A = nn.Parameter(initial[:self.rank].index_select(1,columns).contiguous())
        self.B = nn.Parameter(torch.zeros(self.out_channels, self.rank))

    @property
    def weight(self):
        if self.rank==0:return self.base.weight
        delta=self.B@self.A
        if self.active_columns.numel()!=self.flat:
            delta=delta.new_zeros(self.out_channels,self.flat).index_copy(1,self.active_columns,delta)
        return self.base.weight + delta.reshape_as(self.base.weight)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        a=state_dict.get(prefix+'A');b=state_dict.get(prefix+'B')
        if a is not None and a.shape!=self.A.shape and a.ndim==2 and a.shape[1]==self.flat:
            if self.rank==0:
                state_dict[prefix+'A']=a[:0,:0].contiguous()
                if b is not None:state_dict[prefix+'B']=b[:,:0].contiguous()
            elif a.shape[0]==self.rank:
                state_dict[prefix+'A']=a.index_select(1,self.active_columns.to(a.device)).contiguous()
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    @property
    def bias(self):
        return self.base.bias

    def forward(self, value):
        return self.base._conv_forward(value, self.weight, self.bias)


def expand(operator, old_phi_names, rank=32, seed=20261018):
    for parameter in operator.parameters():
        parameter.requires_grad_(False)
    sites = [(name, module) for name, module in operator.named_modules()
             if name and isinstance(module, (nn.Linear, nn.Conv2d))]
    generator = torch.Generator().manual_seed(seed)
    records = []
    contract=getattr(operator,'physical_initial_condition_contract',{})
    zero_pressure_inputs=isinstance(contract,dict) and contract.get('measured_pressure_inputs')==0
    for name, module in sites:
        parent, leaf = name.rsplit('.', 1)
        columns=None
        if zero_pressure_inputs and name=='base.source_encoder.ic_encoder.0':
            columns=torch.empty(0,dtype=torch.long)
        elif zero_pressure_inputs and name=='refiner.static.0.conv1':
            if module.groups!=1 or module.in_channels!=14:raise ValueError('zero-IC layout requires the registered 14 input channels')
            channels=torch.tensor([0,9,10,11,12,13]);area=module.kernel_size[0]*module.kernel_size[1]
            columns=(channels[:,None]*area+torch.arange(area)).flatten()
        wrapper = ConvLoRA(module, rank, generator,active_columns=columns) if isinstance(module, nn.Conv2d) else LinearLoRA(module, rank, generator)
        operator.get_submodule(parent)._modules[leaf] = wrapper
        records.append({'path': name, 'kind': type(module).__name__, 'rank': wrapper.rank,
                        'original_shape': list(module.weight.shape),
                        'parameters': wrapper.A.numel() + wrapper.B.numel(),
                        'zero_input_projection':columns is not None})
    active = []
    for name, parameter in operator.named_parameters():
        # Old meta factors retain their learned initialization. New A/B are
        # leaves of the new wrappers; underlying original weights remain frozen.
        selected=name in old_phi_names or (name.rsplit('.', 1)[-1] in ('A', 'B') and
                                    isinstance(operator.get_submodule(name.rsplit('.', 1)[0]), (LinearLoRA, ConvLoRA)))
        parameter.requires_grad_(selected and parameter.numel()>0)
        if parameter.requires_grad:
            active.append(name)
    return active, records


def cpu_contract():
    torch.manual_seed(11)
    cases = [(nn.Linear(7, 5), torch.randn(2, 7)),
             (nn.Conv2d(4, 6, 3, stride=2, padding=1), torch.randn(2, 4, 11, 9)),
             (nn.Conv2d(4, 4, 3, padding=1, groups=4), torch.randn(2, 4, 9, 9))]
    reports = []
    for base, x in cases:
        old = base.weight.detach().clone()
        expected = base(x).detach()
        for p in base.parameters():
            p.requires_grad_(False)
        gen = torch.Generator().manual_seed(19)
        wrapped = ConvLoRA(base, 3, gen) if isinstance(base, nn.Conv2d) else LinearLoRA(base, 3, gen)
        assert torch.equal(expected, wrapped(x))
        assert torch.equal(wrapped.weight, base.weight)
        opt = torch.optim.Adam([wrapped.A, wrapped.B], lr=.01)
        wrapped(x).square().mean().backward()
        assert wrapped.B.grad is not None and torch.isfinite(wrapped.B.grad).all() and wrapped.B.grad.abs().sum() > 0
        opt.step()
        assert torch.equal(old, base.weight)
        assert not torch.equal(expected, wrapped(x))
        manual = base._conv_forward(x, wrapped.weight, wrapped.bias) if isinstance(base, nn.Conv2d) else F.linear(x, wrapped.weight, wrapped.bias)
        assert torch.equal(manual, wrapped(x))
        reports.append({'kind': type(base).__name__, 'zero_delta_exact': True,
                        'base_unchanged_after_step': True, 'weight_access_matches_forward': True})
    return reports


if __name__ == '__main__':
    import json
    print(json.dumps(cpu_contract()))
