"""Pure learned residual-cycle network. No physical PDE assembly or solve."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class ConditionalLinear(nn.Module):
    def __init__(self,width,scale):
        super().__init__()
        self.state=nn.Conv2d(width,width,3,padding=1,bias=False)
        self.gate=nn.Sequential(nn.Conv2d(width,width,1),nn.GELU(),nn.Conv2d(width,width,1))
        with torch.no_grad():
            nn.init.dirac_(self.state.weight)
            self.state.weight.mul_(scale).add_(torch.randn_like(self.state.weight)*(.01 if scale==1. else .005))
    def forward(self,a,x):
        return self.state(x)*(1+.1*torch.tanh(self.gate(a)))

class DeterministicProlongation(nn.ConvTranspose2d):
    def forward(self, input, output_size=None):
        with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=True,allow_tf32=False):
            return super().forward(input,output_size=output_size)

class LearnedCycle(nn.Module):
    def __init__(self,width=16,levels=3,lift_mode="additive"):
        super().__init__();self.width=width;self.levels=levels
        if lift_mode not in ("additive","homogeneous"):raise ValueError("unknown lift_mode")
        self.lift_mode=lift_mode
        self.A=nn.ModuleList([ConditionalLinear(width,1.) for _ in range(levels)])
        self.S=nn.ModuleList([ConditionalLinear(width,.2) for _ in range(levels)])
        self.Rr=nn.ModuleList([nn.Conv2d(width,width,3,stride=2,padding=1,bias=False) for _ in range(levels-1)])
        self.Ra=nn.ModuleList([nn.Sequential(nn.Conv2d(width,width,3,stride=2,padding=1),nn.GELU()) for _ in range(levels-1)])
        self.P=nn.ModuleList([DeterministicProlongation(width,width,3,stride=2,padding=1,output_padding=0,bias=False) for _ in range(levels-1)])
        with torch.no_grad():
            for layer in self.Rr:
                layer.weight.zero_()
                for c in range(width):layer.weight[c,c].fill_(1/9)
            for layer in self.Ra:
                layer[0].weight.zero_()
                for c in range(width):layer[0].weight[c,c].fill_(1/9)
            kernel=torch.tensor([[.25,.5,.25],[.5,1.,.5],[.25,.5,.25]])
            for layer in self.P:
                layer.weight.zero_()
                for c in range(width):layer.weight[c,c].copy_(kernel)
        self.medium_proj=nn.Conv2d(128,width,1)
        self.source_proj=nn.Conv2d(128,width,1)
        self.Lu=nn.Conv2d(2*width+6,width,1)
        self.Lf=nn.Conv2d(2*width+6,width,1)
        self.La=nn.Sequential(nn.Conv2d(5,width,1),nn.GELU(),nn.Conv2d(width,width,1))
        self.Q=nn.Conv2d(width,1,1,bias=False);nn.init.zeros_(self.Q.weight)
        self.register_buffer('alpha',torch.tensor([.5,1.,1.5,2.]))
    def cycle(self,l,a,f,u):
        r=f-self.A[l](a,u)
        u=u+self.S[l](a,r)
        if l+1<self.levels:
            r=f-self.A[l](a,u)
            ac=self.Ra[l](a);fc=self.Rr[l](r)
            assert ac.shape[-2:]==fc.shape[-2:]
            ec=self.cycle(l+1,ac,fc,torch.zeros_like(fc))
            prolong=self.P[l](ec)
            assert prolong.shape==u.shape,(prolong.shape,u.shape)
            u=u+prolong
            r=f-self.A[l](a,u)
            u=u+self.S[l](a,r)
        else:
            r=f-self.A[l](a,u)
            u=u+self.S[l](a,r)
        return u
    def compose(self,a,f,u,depth):
        for _ in range(int(depth)):u=self.cycle(0,a,f,u)
        return u
    def forward(self,raw,medium,source,velocity,c_normalized,times,f0,t0,spacing,*,depth=1):
        t=len(raw);rel=times-t0;phase=2*math.pi*f0*rel
        tf=torch.stack([times,rel,torch.ones_like(times)*f0/50,torch.sin(phase),torch.cos(phase)],1)[:,:,None,None]
        z=torch.cat([raw,self.medium_proj(medium).expand(t,-1,-1,-1),self.source_proj(source).expand(t,-1,-1,-1),tf.expand(-1,-1,*raw.shape[-2:])],1)
        if self.lift_mode == "additive":
            u0=self.Lu(z);f=self.Lf(z)
        else:
            cond=z[:,1:]
            def lift(layer):
                raw_part=F.conv2d(raw,layer.weight[:,:1],bias=None)
                cond_part=F.conv2d(cond,layer.weight[:,1:],bias=layer.bias)
                return raw_part*(1+.1*torch.tanh(cond_part))
            u0=lift(self.Lu);f=lift(self.Lf)
        kappa=2*math.pi*self.alpha[None,:,None,None]*f0*spacing/velocity
        a=self.La(torch.cat([c_normalized,kappa],1))
        un=self.compose(a,f,u0,depth)
        return self.Q(un-u0)
