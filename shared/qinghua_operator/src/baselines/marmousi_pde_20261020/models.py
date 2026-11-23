"""Preserve every parent tensor; add one zero-initialized known8 adapter.

Inputs are [parent8, observed8], where parent8 retains the original ordering
[c, source5, x, z]. All400 outputs come from the network; no observation is
copied to output. The adapter adds25,632 real scalars to both14,552,439-parameter
parents:14,578,071 each,32 more than the retained ours TOTAL14,578,039.

At adapter zero the parent forward is unchanged. After continuation begins,
the conditional map and objective change; no unchanged optimizer trajectory
is claimed. The separate physical velocity/source never comes from adapted
feature maps. DeepONet's explicit source argument also remains unadapted.
"""
from __future__ import annotations

import torch
from torch import nn
from baselines.marmousi_matched_20261020.models import (
    build_model as build_parent, parameter_report as parent_parameter_report,
)

PARENT_PARAMETERS = 14_552_439
OURS_TOTAL_PARAMETERS = 14_578_039
ADAPTER_PARAMETERS = 25_632
CONTINUED_PARAMETERS = PARENT_PARAMETERS + ADAPTER_PARAMETERS


class Known8Continuation(nn.Module):
    def __init__(self, parent):
        super().__init__()
        if getattr(parent, 'kind', None) not in ('fno', 'deeponet') or parent.in_channels != 8:
            raise ValueError('An original8-channel FNO or DeepONet parent is required')
        # Register parent first: its named-parameter order remains unchanged.
        self.parent = parent
        self.known8_adapter = nn.Sequential(nn.Conv2d(8,178,3,padding=1,bias=False),
                                           nn.GELU(),
                                           nn.Conv2d(178,8,3,padding=1,bias=False))
        # The first convolution is genuinely used after the zero output gate
        # opens. Its initial connected-zero gradient is expected, not a missing
        # gradient or a dummy parameter. The last layer receives gradient now.
        nn.init.zeros_(self.known8_adapter[-1].weight)
        self.kind = parent.kind
        self.in_channels = 16
        self.configuration = dict(parent=dict(parent.configuration),
                                  adapter='zero_last_known8_conv3_8_178_8_no_bias_v2')

    def adapted_inputs(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 16:
            raise ValueError('Expected [original8,normalized_observed8] in(B,16,H,W)')
        original, known = inputs[:, :8], inputs[:, 8:16]
        return original + self.known8_adapter(known)

    def load_parent_state_dict(self, state):
        if bool(self.known8_adapter[-1].weight.detach().count_nonzero()):
            raise ValueError('Parent migration requires a fresh zero adapter')
        return self.parent.load_state_dict(state, strict=True)

    def encode_branch(self, inputs, source=None):
        if self.kind != 'deeponet':
            raise ValueError('Only DeepONet has a branch encoder')
        adapted = self.adapted_inputs(inputs)
        # The parent otherwise reads source5 from channels1:6 at spatial(0,0).
        # Those adapted feature channels no longer represent physical scalars.
        if source is None:
            source = inputs[:, 1:6, 0, 0]
        return self.parent.encode_branch(adapted, source=source)

    def decode_trunk(self, branch, coords):
        return self.parent.decode_trunk(branch, coords)

    def predict_points(self, branch, coords):
        return self.decode_trunk(branch, coords)

    def forward(self, inputs, coords=None):
        if self.kind == 'fno':
            if coords is not None:
                raise ValueError('FNO directly returns the frozen400 time channels')
            return self.parent(self.adapted_inputs(inputs))
        if coords is None:
            raise ValueError('DeepONet requires explicit(t,x,z) query coordinates')
        return self.decode_trunk(self.encode_branch(inputs), coords)


def build_model(kind, parent_state=None, architecture=None, target_parameters=CONTINUED_PARAMETERS):
    parent = build_parent(kind, target_parameters=None, architecture=architecture)
    model = Known8Continuation(parent)
    if parent_state is not None:
        model.load_parent_state_dict(parent_state)
    report = parameter_report(model, target_parameters)
    if target_parameters is not None and not report['exact_match']:
        raise ValueError(f'Actual continuation parameter count differs: {report}')
    return model


def parameter_report(model, target_parameters=CONTINUED_PARAMETERS):
    report = parent_parameter_report(model, target_parameters)
    report['parent_trainable_real_scalars'] = sum(p.numel() for p in model.parent.parameters() if p.requires_grad)
    report['added_adapter_real_scalars'] = sum(p.numel() for p in model.known8_adapter.parameters())
    report['difference_to_original_ours_trainable_count'] = report['trainable_real_scalars']-PARENT_PARAMETERS
    report['difference_to_original_ours_total_count'] = report['trainable_real_scalars']-OURS_TOTAL_PARAMETERS
    report['parent_parameter_shapes_unchanged'] = True
    return report
