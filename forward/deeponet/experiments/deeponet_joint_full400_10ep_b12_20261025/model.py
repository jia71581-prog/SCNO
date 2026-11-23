"""Conventional branch/trunk DeepONet; POD codes are TRAIN-only auxiliary targets."""
from scno_release_paths import local_asset as _scno_local_asset
from pathlib import Path
import copy, math, sys
import torch
from torch import nn
ROOT=Path(_scno_local_asset('project'))
sys.path.insert(0,str(ROOT/'src'))
from baselines.marmousi_matched_20261020.models import FourierTrunk, RectangularAdaptiveAvgPool2d, _mlp

ARCH=dict(branch_channels=[32,64,64],pool_size=5,branch_hidden=[2305,2305],trunk_width=768,trunk_depth=4,fourier_bands=16,source_bands=32,rank=512,adapter_width=178)

def tiny_architecture():
 return dict(branch_channels=[4,8,8],pool_size=2,branch_hidden=[24,24],trunk_width=16,trunk_depth=2,fourier_bands=3,source_bands=4,rank=8,adapter_width=9)

class RepairDeepONet(nn.Module):
 def __init__(self,architecture=None):
  super().__init__();self.configuration=copy.deepcopy(architecture or ARCH);a=self.configuration
  self.rank=a['rank'];self.kind='deeponet';self.in_channels=16
  self.known8_adapter=nn.Sequential(nn.Conv2d(8,a['adapter_width'],3,padding=1,bias=False),nn.GELU(),nn.Conv2d(a['adapter_width'],8,3,padding=1,bias=False))
  conv=[];last=8
  for out in a['branch_channels']:
   conv.extend([nn.Conv2d(last,out,3,stride=2,padding=1),nn.GELU()]);last=out
  conv.extend([RectangularAdaptiveAvgPool2d(a['pool_size']),nn.Flatten(1)])
  self.branch_encoder=nn.Sequential(*conv)
  self.branch=_mlp([last*a['pool_size']**2+5+4*a['source_bands'],*a['branch_hidden'],self.rank])
  self.trunk=FourierTrunk(self.rank,a['trunk_width'],a['trunk_depth'],a['fourier_bands'])
  self.trunk.bands.copy_(2*torch.pi*torch.logspace(0,math.log10(64),a['fourier_bands']))
  self.register_buffer('source_frequencies',2*torch.pi*torch.logspace(0,math.log10(64),a['source_bands']))
  self.register_buffer('branch_mean',torch.zeros(self.rank));self.register_buffer('branch_scale',torch.ones(self.rank))
  self.output_bias=nn.Parameter(torch.zeros(()))
  with torch.no_grad():
   for layer in (self.branch[-1],self.trunk.output):layer.weight.mul_(.01);layer.bias.mul_(.01)
 def encode_branch(self,inputs):
  if inputs.ndim!=4 or inputs.shape[1]!=16:raise ValueError('Expected original8 + observed8')
  source=inputs[:,1:6,0,0]
  # The inherited codec already encodes source x,z in metres/2000.
  phase=source[:,:2,None]*self.source_frequencies
  features=torch.cat((phase.sin(),phase.cos()),dim=-1).flatten(1)
  adapted=inputs[:,:8]+self.known8_adapter(inputs[:,8:])
  raw=self.branch(torch.cat((self.branch_encoder(adapted),source,features),dim=1))
  return raw*self.branch_scale+self.branch_mean
 def decode_trunk(self,codes,coords):
  basis=self.trunk(coords)
  return (torch.einsum('br,qr->bq',codes,basis) if coords.ndim==2 else torch.einsum('br,bqr->bq',codes,basis))+self.output_bias
 def forward(self,inputs,coords):return self.decode_trunk(self.encode_branch(inputs),coords)
 def set_branch_statistics(self,codes):
  with torch.no_grad():
   self.branch_mean.copy_(codes.mean(0));scale=codes.std(0,unbiased=False)
   # Only exactly constant coordinates use unit scale; no VAL statistics.
   self.branch_scale.copy_(torch.where(scale>0,scale,torch.ones_like(scale)))

def migrate_fixed_components(net,state):
 copied=[]
 with torch.no_grad():
  for key,target in net.state_dict().items():
   source=key if key.startswith('known8_adapter.') else 'parent.'+key
   if key.startswith(('known8_adapter.','branch_encoder.')) or key=='output_bias':
    if source not in state or state[source].shape!=target.shape:raise ValueError('Migration mismatch: '+key)
    target.copy_(state[source]);copied.append(source)
 return dict(transferred=copied,fresh=['branch','trunk'],optimizer='fresh for every phase; parent moments not inherited',fresh_output_initialization='branch and trunk final weight and bias multiplied by0.01; all remain connected')

class FixedCodeModel:
 """Training-only adapter; inference always calls RepairDeepONet itself."""
 def __init__(self,model,codes):self.model,self.codes=model,codes
 def encode_branch(self,inputs):return self.codes.detach().requires_grad_(True)
 def decode_trunk(self,codes,coords):return self.model.decode_trunk(codes,coords)

@torch.no_grad()
def qr_condition(net,codes,coordinates,chunk=2048):
 device=next(net.parameters()).device
 before=torch.cat([net.trunk(q.to(device)).double().cpu() for q in coordinates.split(chunk)],0)
 a=before/math.sqrt(len(before));_,r=torch.linalg.qr(a,mode='reduced')
 singular=torch.linalg.svdvals(r);condition=float(singular[0]/singular[-1])
 if not math.isfinite(condition) or condition>1e6:raise ValueError('Unsafe trunk condition number: '+str(condition))
 inverse=torch.linalg.solve_triangular(r,torch.eye(r.shape[0],dtype=r.dtype),upper=True)
 out=net.trunk.output
 out.weight.copy_((inverse.T@out.weight.detach().double().cpu()).to(out.weight))
 out.bias.copy_((out.bias.detach().double().cpu()@inverse).to(out.bias))
 transformed=codes.detach().double().cpu()@r.T
 after=torch.cat([net.trunk(q.to(device)).double().cpu() for q in coordinates.split(chunk)],0)
 orth=float(torch.linalg.vector_norm(after.T@after/len(after)-torch.eye(net.rank,dtype=torch.float64))/math.sqrt(net.rank))
 # All500 codes and all fixed queries, in bounded record chunks.
 num=den=0.
 for c,d in zip(codes.detach().double().cpu().split(16),transformed.split(16)):
  old=c@before.T;new=d@after.T
  num+=float((new-old).square().sum());den+=float(old.square().sum())
 relative=math.sqrt(num/den) if den else 0.
 proof=dict(condition_number=condition,orthogonality_relative_error=orth,oracle_pressure_relative_change=relative,queries=len(before),train_codes=len(codes),arithmetic='FP64 QR; FP32 model weights and predictions')
 if orth>1e-3 or relative>1e-4:raise ValueError('QR numerical gate failed: '+str(proof))
 return transformed.to(codes),proof
