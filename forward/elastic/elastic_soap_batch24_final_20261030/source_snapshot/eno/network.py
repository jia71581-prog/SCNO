"""UNO with configurable spectral modules, checkpointing and explicit units."""
import types
from functools import partial
import torch
from torch.utils.checkpoint import checkpoint
from .uno import Uno3D,OperatorBlock_3D
from .core import operator_input,normalize_input,decode_output


def to_precision(model,precision):
    if precision not in ['float32','float64']:raise ValueError('unsupported precision')
    model.double() if precision=='float64' else model.float()
    complex_dtype=torch.complex128 if precision=='float64' else torch.complex64
    for p in model.parameters():
        if p.is_complex():p.data=p.data.to(complex_dtype)
    return model


def build_network(width=16,activation_checkpoint=True,spectral_module='dense',rank_fraction=.25,min_rank=8,operator_block='original',checkpoint_projection=False,projection_chunk_rows=0):
    if spectral_module not in ('dense','channel_tucker'):
        raise ValueError('unknown spectral_module: '+str(spectral_module))
    if spectral_module=='channel_tucker':
        from .factorized_spectral import ChannelTuckerSpectralConv3d
        factory=partial(ChannelTuckerSpectralConv3d,rank_fraction=rank_fraction,min_rank=min_rank)
        model=Uno3D(6,width,pad=0,spectral_factory=factory)
    else:model=Uno3D(6,width,pad=0)
    if operator_block=='fused_fft':
        from .fused_operator import fuse_operator_blocks
        fuse_operator_blocks(model)
    elif operator_block!='original':raise ValueError('unknown operator_block: '+str(operator_block))
    model.checkpoint_projection=bool(checkpoint_projection)
    if projection_chunk_rows<0:raise ValueError('projection_chunk_rows must be nonnegative')
    model.projection_chunk_rows=int(projection_chunk_rows)
    if activation_checkpoint:
        for block in model.modules():
            if isinstance(block,OperatorBlock_3D):
                block._original_forward=block.forward
                def run(self,*args):
                    if torch.is_grad_enabled():return checkpoint(self._original_forward,*args,use_reentrant=False)
                    return self._original_forward(*args)
                block.forward=types.MethodType(run,block)
    return model


def build_network_from_config(tc):
    return build_network(tc['width'],tc.get('activation_checkpoint',True),
        spectral_module=tc.get('spectral_module','dense'),rank_fraction=tc.get('rank_fraction',.25),
        min_rank=tc.get('min_rank',8),operator_block=tc.get('operator_block','original'),
        checkpoint_projection=tc.get('activation_checkpoint_projection',False),
        projection_chunk_rows=tc.get('projection_chunk_rows',0))


class PhysicalUNO(torch.nn.Module):
    def __init__(self,network,statistics):
        super().__init__();self.network=network;self.statistics=statistics
    def forward(self,velocity,source):
        x=normalize_input(operator_input(velocity,source),self.statistics)
        return decode_output(self.network(x,time_grid=source.shape[-1]),self.statistics)
