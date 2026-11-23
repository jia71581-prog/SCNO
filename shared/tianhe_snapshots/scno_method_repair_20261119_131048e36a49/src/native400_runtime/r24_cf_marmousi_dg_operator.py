"""Native Marmousi R24 MIONet/U-FNO with trainable SIPG hidden-feature exchange.
Uses the unchanged trained native Marm architecture, always-zero IC conditioning,
and output-only true8 delivery. DG is embedded before the refiner output head;
PDE weak loss is a separate TRAIN objective, not the architectural DG layer.
"""
from scno_release_paths import local_asset as _scno_local_asset
from pathlib import Path
import hashlib,copy,sys,math
import torch
from .r24_cf_preserved_operator import R24CFPreservedOperator,COMPAT

DEFAULT_PARENT=_scno_local_asset('fno-reference/a_research/experiments_101x101x100/experiments/cc_time400_r24_20261008T134744Z/marmousi/frozen/run/checkpoint_final_added000256.pt')
DEFAULT_PARENT_SHA='2638504b1c2806c29752aed5189c1d09fb0e5f4a6c746e91d91d8eb324adb8d6'
SCHEMA='R24_NATIVE_MARMOUSI_ZERO_IC_EMBEDDED_DG_FEATURE_V1'

class R24CFMarmousiDGOperator(R24CFPreservedOperator):
    def __init__(self,config,device='cpu'):
        config=copy.deepcopy(config)
        super().__init__('cpu',family='marmousi',
            checkpoint_path=config.get('parent_checkpoint',DEFAULT_PARENT),
            expected_checkpoint_sha=config.get('parent_checkpoint_SHA',DEFAULT_PARENT_SHA),
            dense_time_block=config.get('dense_time_block',48),
            refiner_time_block=config.get('refiner_time_block',48),
            checkpoint_queries=config.get('checkpoint_queries',False))
        # The legacy module imports bootstrap24 only for an unused factory/class.
        # Bind that import to the canonical r24/r22 snapshots, never an archive.
        sys.path.insert(0,str(COMPAT/'r24'))
        from .dg_architecture import DGFeatureBlock
        dg=config.get('dg') or {}
        if set(dg)-{'element_cells','order','enabled','face_enabled','learning_rate'}:
            raise ValueError('unsupported embedded DG configuration')
        if dg.get('element_cells',4)!=4 or dg.get('order',3)!=3:
            raise ValueError('registered existing embedded DG uses Q3/ec4')
        self.refiner.add_module('dg',DGFeatureBlock(channels=16,size=201,element_cells=4,order=3))
        self.refiner.dg.enabled=bool(dg.get('enabled',True))
        self.refiner.dg.face_enabled=bool(dg.get('face_enabled',True))
        self.dg_learning_rate=dg.get('learning_rate')
        if self.dg_learning_rate is not None and (not math.isfinite(float(self.dg_learning_rate)) or self.dg_learning_rate<=0):
            raise ValueError('positive finite embedded DG learning rate')
        self.model_config=config
        self.initialization_receipt.update(schema=SCHEMA,initial_model_family='marmousi',
            optimizer_policy='fresh complete Adam; parent Adam is lineage only',
            embedded_DG='DGFeatureBlock Q3 ec4 SIPG volume+face before refiner.output',
            DG_initialization='zero volume/face gates, identity modal/channel mixes',
            DG_stage0_gradient='gates active immediately; mixes become active after nonzero gates',
            full_model_state_required=True,pressure_conditioning='none; zeroIC internal',
            source_scope='registered15Hz/t0.1/positiveA,x and z within0..2000m',
            physics_times_included=[0.,.0005,.001],hard_top_BC=True)
        self.to(device)

    def _inputs(self,c,f):
        if isinstance(c,dict) or isinstance(f,dict):
            raise TypeError('only explicit c/f; pressure dictionaries forbidden')
        c=torch.as_tensor(c,device=self.device,dtype=torch.float32)
        f=torch.as_tensor(f,device=self.device,dtype=torch.float32)
        if c.shape==(201,201):c=c[None,None]
        if f.shape==(5,):f=f[None]
        if c.shape!=(1,1,201,201) or f.shape!=(1,5):raise ValueError('c201 and knownfive source parameters')
        if not bool(torch.isfinite(c).all()) or not bool((c>0).all()) or not bool(torch.isfinite(f).all()):raise ValueError('finite c/f required')
        if (bool((f[:,:2]<0).any()) or bool((f[:,:2]>2000).any()) or float(f[0,4])<=0
                or abs(float(f[0,2])-15.)>1e-5 or abs(float(f[0,3])-.1)>1e-6):
            raise ValueError('registered Marm x/z0..2000m,15Hz,t0.1,positiveknownA')
        return c,f

    def _prepare_cf(self,c,f):
        prepared,grid,features=super()._prepare_cf(c,f)
        # Cache only c-derived fixed modal mass; never trainable features or labels.
        key=hashlib.sha256(c.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        mass=self.refiner.dg.medium_mass(c,key)
        return prepared,grid,dict(conv=features,mass=mass)

    def _query(self,prepared,grid,features,f,times):
        def transform(_module,args):return (self.refiner.dg(args[0],features['mass']),)
        hook=self.refiner.output.register_forward_pre_hook(transform,prepend=True)
        try:return super()._query(prepared,grid,features['conv'],f,times)
        finally:hook.remove()

    def parameter_groups(self,base_lr,refiner_lr,weight_decay=1e-6,dg_lr=None):
        groups=super().parameter_groups(base_lr,refiner_lr,weight_decay)
        ids={id(p) for p in self.refiner.dg.parameters() if p.requires_grad}
        for group in groups:group['params']=[p for p in group['params'] if id(p) not in ids]
        rate=dg_lr if dg_lr is not None else self.dg_learning_rate
        groups.append(dict(params=[p for p in self.refiner.dg.parameters() if p.requires_grad],
            lr=float(refiner_lr if rate is None else rate),weight_decay=float(weight_decay),group_name='embedded_DG'))
        actual=[id(p) for g in groups for p in g['params']]
        expected={id(p) for p in self.parameters() if p.requires_grad}
        if len(actual)!=len(set(actual)) or set(actual)!=expected:raise AssertionError('all trainable state in exactlyone optimizer group')
        return groups

    def dg_diagnostics(self):
        d=self.refiner.dg
        return dict(embedded_before_refiner_output=True,enabled=d.enabled,faces_enabled=d.face_enabled,
            element_cells=d.ec,order=d.order,learnable_parameters=sum(p.numel() for p in d.parameters()),
            volume_gate_max=float(d.volume_gate.detach().abs().max()),face_gate_max=float(d.face_gate.detach().abs().max()),
            algebra='h_squared_Mdiagonal_inverse_times_SIPG_volume_and_face_hidden_feature_update',
            weak_PDE_loss_is_separate=True,DG_time_propagator=False)

def build_r24_cf_marmousi_dg_operator(config,device='cpu'):
    return R24CFMarmousiDGOperator(config,device)
