"""R25 c/f-only full400 delivery, preserving every R24 parameter/state name.

All original registered 400 instants (including the former known8) are network
queries. No measured pressure, ridge, FD solve, or observation copy is accepted.
The inherited internal zero8 channels are disabled architectural placeholders,
not observed IC. Physical targets p(0)=0 and partial_t p(0)=0 remain the existing
label-free IC loss targets; this wrapper does not introduce a hard IC projection.
The inherited exact top boundary remains hard. Internal time blocks are allowed;
one public forward_cf_full(c,f) returns the complete contiguous Pa tensor.
"""
from __future__ import annotations

import torch
from .r24_cf_preserved_operator import R24CFPreservedOperator
from .r24_cf_marmousi_dg_operator import R24CFMarmousiDGOperator
from .paper201_time_axis_20261020 import native_time_axis

SCHEMA = 'R25_CF_ALL400_PREDICTED_ZERO_PHYSICAL_IC_V1'


class _Full400Mixin:
    """No parameters or persistent buffers are added to the inherited models."""
    @property
    def full_times_s(self):
        indices = torch.as_tensor(self.full_native_frames, device=self.native_times_s.device,
                                  dtype=torch.long)
        if indices.numel() != 400 or not bool(torch.all(indices[1:] > indices[:-1])):
            raise ValueError('original strictly ordered 400 native indices required')
        if int(indices[0]) < 0 or int(indices[-1]) >= self.native_times_s.numel():
            raise ValueError('registered full400 indices outside native axis')
        return self.native_times_s.index_select(0, indices)

    @property
    def physical_initial_condition_contract(self):
        return dict(p0_Pa=0.0, pt0_Pa_per_s=0.0, measured_pressure_inputs=0,
                    enforcement='inherited label-free IC loss; no new hard projection',
                    hard_top_boundary=True)

    def _full400_receipt(self):
        self.initialization_receipt.update(delivery_schema=SCHEMA,
            all400_network_predicted=True, known8_delivery_only=False,
            measured_pressure_inputs=0, observed_pressure_copy=False,
            public_delivery_inputs=['c', 'f'], public_delivery_shape=[201,201,400],
            public_delivery_dtype='float32', public_delivery_units='Pa',
            added_parameters=0, added_persistent_buffers=0,
            full400_native_indices=list(self.full_native_frames),
            physical_initial_conditions=self.physical_initial_condition_contract)

    def forward(self, c, f, times_s=None, *, normalized=True):
        """Graph-connected TRAIN queries; all400 by default, explicit IC times allowed."""
        return self.forecast_at_times(c, f, self.full_times_s if times_s is None else times_s,
                                      normalized=normalized)

    @torch.no_grad()
    def forward_cf_full(self, c, f):
        """One API call: c/f -> contiguous float32 Pa [Z=201,X=201,T=400].

        There is intentionally no known8 argument. The inherited predictor
        prepares c/f features once and queries every registered original time.
        """
        was_training = self.training
        self.eval()
        try:
            field = self.forecast_at_times(c, f, self.full_times_s, normalized=False)
            if field.dtype != torch.float32 or field.shape != (400,201,201):
                raise ValueError('inherited decoder must return float32 Pa TZX400')
            if not bool(torch.isfinite(field).all()):
                raise FloatingPointError('nonfinite full400 prediction')
            if not bool(torch.all(field[:,0,:] == 0)):
                raise AssertionError('inherited exact top boundary violated')
            return field.permute(1,2,0).contiguous()
        finally:
            self.train(was_training)


class R25CFFull400JointOperator(_Full400Mixin, R24CFPreservedOperator):
    """Joint uniform/layered architecture and nine inherited optimizer groups."""
    def __init__(self, device='cpu', **kwargs):
        if kwargs.get('family', 'layered') != 'layered':
            raise ValueError('joint model preserves the layered R24 architecture')
        kwargs['family'] = 'layered'
        super().__init__(device=device, **kwargs)
        self._full400_receipt()


class R25CFFull400MarmousiDGOperator(_Full400Mixin, R24CFMarmousiDGOperator):
    """Native Marm MIONet/U-FNO/DG and all ten inherited optimizer groups."""
    def __init__(self, config, device='cpu'):
        super().__init__(config, device=device)
        self._full400_receipt()


def build_r25_cf_full400_joint_operator(config, device='cpu'):
    keys = ('dense_time_block','refiner_time_block','checkpoint_queries')
    kwargs = {key:config[key] for key in keys if key in config}
    if 'parent_checkpoint' in config: kwargs['checkpoint_path'] = config['parent_checkpoint']
    if 'parent_checkpoint_SHA' in config: kwargs['expected_checkpoint_sha'] = config['parent_checkpoint_SHA']
    return R25CFFull400JointOperator(device=device, **kwargs)


def build_r25_cf_full400_marmousi_dg_operator(config, device='cpu'):
    return R25CFFull400MarmousiDGOperator(config, device=device)


def build_r25_cf_full400_operator(config, device='cpu', *, kind):
    if kind == 'joint': return build_r25_cf_full400_joint_operator(config, device)
    if kind == 'marmousi': return build_r25_cf_full400_marmousi_dg_operator(config, device)
    raise ValueError('kind must be joint or marmousi')


def cpu_contract_test():
    """Cheap real-size delivery test with a stub forecast; NOT an accuracy test.

    The registered protocol and actual mixin delivery execute on CPU. Expensive
    MIONet/U-FNO/DG forecasting is replaced only here by a known Pa formula.
    Actual numerical accuracy and trained-state reload are separate TRAIN checks.
    """
    from torch import nn
    from .r24_cf_preserved_operator import _module, COMPAT
    protocol = _module('r25_full400_cpu_contract_protocol', COMPAT/'r24/protocol400.py')

    class Stub(_Full400Mixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))
            self.full_native_frames = tuple(protocol.FULL_FRAMES)
            self.native_times_s = native_time_axis()
            self.calls = []
        def forecast_at_times(self, c, f, times_s, *, normalized=True):
            if isinstance(c,dict) or isinstance(f,dict): raise TypeError('pressure dictionaries forbidden')
            self.calls.append(dict(times=times_s.clone(),normalized=normalized,training=self.training,
                                   grad_enabled=torch.is_grad_enabled()))
            z=torch.arange(201,dtype=torch.float32)[None,:,None]
            x=torch.arange(201,dtype=torch.float32)[None,None,:]
            v=self.weight*(times_s[:,None,None]+z*3+x*5)
            v=torch.cat((torch.zeros_like(v[:,:1]),v[:,1:]),1)
            return v if normalized else v*torch.as_tensor(f,dtype=torch.float32)[4]*7

    model=Stub().train();before=set(model.state_dict())
    c=torch.full((201,201),3000.,dtype=torch.float32);f=torch.tensor([1000.,100.,15.,.1,2.])
    result=model.forward_cf_full(c,f)
    assert result.shape==(201,201,400) and result.dtype==torch.float32 and result.is_contiguous()
    assert model.training and len(model.calls)==1 and not model.calls[0]['normalized']
    assert not model.calls[0]['training'] and not model.calls[0]['grad_enabled']
    assert set(model.state_dict())==before
    assert torch.equal(model.calls[0]['times'],model.native_times_s[list(protocol.FULL_FRAMES)])
    assert tuple(protocol.FULL_FRAMES[:8])==tuple(protocol.IC_FRAMES)
    for position in (0,7,8,157,399):
        t=model.full_times_s[position]
        expected=2*(t+3*17+5*29)*2*7
        assert result[17,29,position].item()==expected.item()
    assert torch.all(result[0]==0)
    for bad in (lambda:model.forward_cf_full(c,f,known8_pa=torch.zeros(8,201,201)),
                lambda:model.forward_cf_full({'known8':0},f)):
        try:bad()
        except TypeError:pass
        else:raise AssertionError('pressure input accepted')
    model.calls.clear()
    value=model.forward(c,f,torch.tensor([0.,.0005,.001]),normalized=True)
    assert value.shape==(3,201,201) and value.requires_grad
    value.square().mean().backward();assert model.weight.grad is not None
    assert model.physical_initial_condition_contract['p0_Pa']==0
    assert model.physical_initial_condition_contract['pt0_Pa_per_s']==0
    return dict(status='PASS_CPU_DELIVERY_CONTRACT',shape=list(result.shape),contiguous=True,
        dtype=str(result.dtype),units='Pa',all400_queried_once=True,former8_network_queried=True,
        original_protocol_SHA=protocol.PROTOCOL_SHA,registered_native_indices=list(protocol.FULL_FRAMES),
        pressure_argument_rejected=True,IC_query_graph_connected=True,
        expensive_forecast_stubbed=True,numerical_accuracy_tested=False,
        real_trained_checkpoint_reload_tested=False,CUDA_initialized=torch.cuda.is_initialized())


if __name__ == '__main__':
    import json
    print(json.dumps(cpu_contract_test(),sort_keys=True))
