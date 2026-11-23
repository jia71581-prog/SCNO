"""Actual native201 pre-head modes and arbitrary-time complete acoustic field."""
from pathlib import Path
import importlib.util
import json
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from ridge_core import RidgePrior,solve_support,detached_query_leaf,replay_meta

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
POINTER=Path(__file__).resolve().parents[4]/'configs/native400/DATA_POINTER.json'


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path));module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name]=module;spec.loader.exec_module(module);return module


adapter=load_file('r25_readonly_sharp_adapter',json.loads(POINTER.read_text())['factory_module'])
from bootstrap24 import ResidualRefiner,build_known,RecordCursor,sha,write_json,rng_state,restore_rng,loss_sums
from protocol400 import FULL_FRAMES,IC_FRAMES,FORECAST_FRAMES,PROTOCOL_SHA,OLD92_POS,BINS
from graph24 import GraphCut,_matrix


class MetaAcoustic(nn.Module):
    def __init__(self,device):
        super().__init__();self.refiner=ResidualRefiner().to(device);self.prior=RidgePrior().to(device)

    def raw_and_modes(self,features,base,times,source):
        captured=[]
        handle=self.refiner.output.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
        try:delta=self.refiner(features,base,times,source)
        finally:handle.remove()
        if len(captured)!=1 or captured[0].shape[1]!=16:raise AssertionError('actual pre-output16 tensor')
        scale=base.square().mean((-2,-1),keepdim=True).add(1e-6).sqrt()
        modes=captured[0]*scale
        modes=torch.cat((torch.zeros_like(modes[:,:,:1,:]),modes[:,:,1:,:]),-2)
        return base+delta,modes

    def adapted(self,raw,modes,coefficient):
        return raw+torch.einsum('k,nkzx->nzx',coefficient.to(modes.dtype),modes)[:,None]


def base_at_times(engine,context,times):
    """No cached pressure. Tensor time enters original decoder then fixed bicubic."""
    prepared,grid,coarse=context
    pred=engine.model.dense_normalized(prepared,times.reshape(1,-1),dense_grid=grid,time_block=8,apply_correction=True)
    # Normalization is verified scale-only in the inherited engine. The fixed
    # separable matrices are the same bicubic kernel, and preserve higher AD.
    z=_matrix(101,201,pred.device,pred.dtype);x=_matrix(101,201,pred.device,pred.dtype)
    value=torch.matmul(z,torch.matmul(pred[0,:,None],x.T))
    if value.shape!=(times.numel(),1,201,201) or not torch.isfinite(value).all():raise FloatingPointError('continuous full201 base')
    return value


def raw_at_times(meta,engine,context,features,fine,times,checkpointed=False):
    def forward(s):
        return meta.raw_and_modes(features,base_at_times(engine,context,s),s,fine['source'])
    return checkpoint(forward,times,use_reentrant=False) if checkpointed else forward(times)


def solve_known8(meta,engine,context,features,fine,time_block=4,checkpointed=True):
    """Signature contains known inputs only: no TRAIN query/energy/score access."""
    known=engine.norm.encode_pressure(fine['ic'],fine['source'][:,4])[0]
    times=fine['time_axis'][list(IC_FRAMES)]
    gram=meta.prior.a0.new_zeros((16,16));rhs=meta.prior.a0.new_zeros(16);energy=known.double().square().sum();count=0
    for start in range(0,8,time_block):
        stop=min(8,start+time_block)
        raw,modes=raw_at_times(meta,engine,context,features,fine,times[start:stop],checkpointed)
        phi=modes.permute(0,2,3,1).reshape(-1,16).double()
        residual=(known[start:stop,None].double()-raw.double()).reshape(-1)
        gram=gram+phi.T@phi;rhs=rhs+phi.T@residual;count+=residual.numel()
    coef,receipt=solve_support(gram,rhs,count,energy,meta.prior)
    receipt.update(actual_IC_times=times.detach().cpu().tolist(),raw_support_forward=True,IC_not_delivery_copy=True,
                   modes_max_abs=float(modes.detach().abs().max()))
    return coef,receipt


def make_context(meta,engine,data,packet):
    fine,context=engine.prepare(data,packet)
    features=meta.refiner.prepare(build_known(fine,engine.norm))
    cut=GraphCut((context,features));context_leaf,feature_leaf=cut.tree
    return fine,context_leaf,feature_leaf,cut


def release_exact_meta(coefficient,leaf,cut):
    replay_meta(coefficient,leaf);cut.backward();cut.release()
