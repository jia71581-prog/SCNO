"""Small preregistered Marmousi nonlinear adaptation probe: early labels only."""
from scno_release_paths import local_asset as _scno_local_asset
from pathlib import Path
from contextlib import contextmanager
import argparse, copy, hashlib, json, math, os, shutil, sys, time, traceback
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/"marmousi_probe"
SUPPORT=Path(_scno_local_asset('runtime/final_algorithm_revision/evidence_code/support'))
sys.path.insert(0,str(PREVIOUS));sys.path.insert(0,str(SUPPORT))
from inspect_interfaces import load_marmousi
from model_adapter import FrozenBaseCache
from operators import AISSampler,make_valid_mask,residual_scale_squared
from legacy_run import train_update


def initial_base_bridge(times,base_zero,base_derivative):
    if not bool(torch.all(times.detach()==0)):
        raise ValueError("The exact frozen-base time jet is only valid at t=0")
    view=times.reshape((-1,)+(1,)*(base_zero.ndim-1))
    return base_zero+view*base_derivative


def write(path,value):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def cpu_state(module):
    return {name:value.detach().cpu().clone() for name,value in module.state_dict().items()}


class MarmSession:
    """Unchanged full model or frozen-base/trainable-Marm-refiner execution."""
    def __init__(self,model,inputs,cache=None):
        self.operator=model;self.device=model.device;self.cache=cache
        self.c=inputs["velocity"];self.f=inputs["source"]
        self._parameters=[p for p in model.parameters() if p.requires_grad]
        self.mass=None
        if cache is not None:
            key=hashlib.sha256(cache.c.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            self.mass=model.refiner.dg.medium_mass(cache.c,key)

    def parameters(self):return iter(self._parameters)

    @contextmanager
    def training_scope(self):
        # Only execution/memory switches change. Query graph and parameters stay intact.
        changed=[]
        for module in self.operator.modules():
            for name in ("dense_outer_checkpoint","use_checkpoint"):
                if hasattr(module,name):
                    changed.append((module,name,getattr(module,name)))
                    setattr(module,name,False)
        try:
            with self.operator.scopes(),sdpa_kernel(SDPBackend.MATH):yield
        finally:
            for module,name,value in changed:setattr(module,name,value)

    def query(self,times,*,grad=True,initial_bridge=False):
        t=torch.as_tensor(times,device=self.device,dtype=torch.float32).reshape(-1)
        context=torch.enable_grad() if grad else torch.no_grad()
        with context:
            if self.cache is None:return self.operator(self.c,self.f,t,normalized=True)
            if initial_bridge or t.requires_grad:
                b0,bt=self.cache.initial_base()
                base=initial_base_bridge(t,b0,bt)
            else:base=self.cache.get(t)
            features=self.operator.refiner.prepare(self.cache.static)
            def apply_dg(_module,args):
                return (self.operator.refiner.dg(args[0],self.mass),)
            hook=self.operator.refiner.output.register_forward_pre_hook(apply_dg,prepend=True)
            try:value=base+self.operator.refiner(features,base,t,self.cache.f)
            finally:hook.remove()
            return torch.cat((torch.zeros_like(value[:,:,:1]),value[:,:,1:]),-2)[:,0]

    def initial_state(self,*,create_graph=True):
        zero=torch.zeros(1,device=self.device,dtype=torch.float32)
        with torch.enable_grad(),self.training_scope():
            return torch.autograd.functional.jvp(
                lambda t:self.query(t,grad=True,initial_bridge=True),
                zero,torch.ones_like(zero),create_graph=create_graph)


def evaluate_early(session,early,times):
    errors=[]
    with torch.no_grad(),session.training_scope():
        for t,target in zip(times,early):
            prediction=session.query(t[None],grad=False)[0]
            errors.append(float((prediction.double()-target.double()).square().sum()))
    energies=early.double().square().flatten(1).sum(1).cpu().numpy()
    fit=sum(errors[:6])/float(energies[:6].sum())
    selection=sum(errors[6:])/float(energies[6:].sum())
    return dict(fit_mse=fit,selection_mse=selection,
        fit_error_percent=100*math.sqrt(fit),selection_error_percent=100*math.sqrt(selection),
        per_frame_sse=errors,per_frame_reference_energy=energies.tolist())


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/"probe_code.py")
    shutil.copyfile(PREVIOUS/"inspect_interfaces.py",out/"loader_code.py")
    shutil.copyfile(HERE/"PROTOCOL.md",out/"PROTOCOL.md")
    start=time.monotonic()
    def status(stage,**details):write(out/"STATUS.json",dict(stage=stage,pid=os.getpid(),elapsed_s=time.monotonic()-start,**details))
    seed=20261106
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    config=dict(schema="MARM_NONLINEAR_EARLY_ONLY_V1",seed=seed,fit_indices=list(range(6)),
        selection_indices=[6,7],max_proposals=8,primary_arm="A_refiner_LWC",
        lr=1e-4,weight_decay=1e-4,clip=1.,IC_weight=.01,PDE_weight=.001,
        early_selection_material_MSE_ratio=.64,stop="first early-selection worsening; roll back",
        sampler="same precomputed uniform node-weighted HT plans, 8 time x 8 tile draws",
        delta_s=.0005,h_m=10.,mask_halo=8,source_exclusion_radius_m=90.,
        residual_normalization="omega^4 mean(six fit normalized snapshots squared)",
        future_reference_labels_accessed=False,full_grid_Pf_checked=False,
        final_algorithm=False,unlabeled_predictions_are_not_reference_labels=True)
    write(out/"CONFIG.json",config)
    status("loading_original_model")
    model,inputs=load_marmousi("cuda")
    original_full=cpu_state(model);original_refiner=cpu_state(model.refiner)
    early=inputs["early8_pa"]/(float(inputs["source"][4])*inputs["pressure_scale_pa"])
    times=inputs["early_times"]
    centers=inputs["full_times"][8:].clone().clamp(.0005,.9995)
    assert len(centers)==392 and not bool((centers[:,None]==times[None]).any())
    mask=make_valid_mask(source=inputs["source"][:2].tolist(),halo=8,radius=90.)
    nu_squared=residual_scale_squared(early[:6],f0=float(inputs["source"][2]))
    np.save(out/"valid_mask.npy",mask.numpy())
    sampler=AISSampler(mask,n_times=len(centers),seed=seed)
    plans=[sampler.draw("uniform",n_time_draws=8,n_tile_draws=8) for _ in range(8)]
    torch.save(plans,out/"precomputed_physics_plans.pt")
    model.requires_grad_(False)
    initial_session=MarmSession(model,inputs)
    status("prepare_frozen_base")
    with initial_session.training_scope():
        cache=FrozenBaseCache(model,inputs["velocity"],inputs["source"],block=1)
        cache.prefill(times);cache.initial_base()
    cached=MarmSession(model,inputs,cache)
    status("verify_cached_DG_and_true_initial_derivative")
    with cached.training_scope(),torch.no_grad():
        direct=initial_session.query(times[-1:],grad=False)
        reconstructed=cached.query(times[-1:],grad=False)
    field_difference=float((direct-reconstructed).double().norm()/direct.double().norm())
    assert field_difference<2e-6,field_difference
    p0,pt0=initial_session.initial_state(create_graph=False)
    pc,ptc=cached.initial_state(create_graph=False)
    initial_difference=float((p0-pc).double().norm()/p0.double().norm().clamp_min(1e-20))
    initial_derivative_difference=float((pt0-ptc).double().norm()/pt0.double().norm().clamp_min(1e-20))
    assert initial_difference<2e-6 and initial_derivative_difference<2e-5,(initial_difference,initial_derivative_difference)
    del direct,reconstructed,p0,pt0,pc,ptc
    baseline=evaluate_early(cached,early,times)
    verification=dict(cached_DG_forward_relative_l2=field_difference,
        cached_IC_pressure_relative_l2=initial_difference,
        cached_total_IC_derivative_relative_l2=initial_derivative_difference,
        baseline=baseline,nu_squared=float(nu_squared),valid_nodes=int(mask.sum()),
        model_parameters=sum(p.numel() for p in model.parameters()),
        refiner_parameters=sum(p.numel() for p in model.refiner.parameters()))
    write(out/"INITIAL_VERIFICATION.json",verification)
    torch.save(original_refiner,out/"original_refiner.pt")
    results={}
    for arm in ("A_refiner_LWC","B_refiner_data_IC","C_full_LWC"):
        armout=out/arm;armout.mkdir()
        torch.manual_seed(seed);np.random.seed(seed)
        status("start_arm",arm=arm)
        if arm.startswith("C"):
            model.load_state_dict(original_full,strict=True)
            model.requires_grad_(True)
            dict(model.named_parameters())["base.source_encoder.ic_encoder.0.weight"].requires_grad_(False)
            session=MarmSession(model,inputs)
            expected=14552439
        else:
            model.refiner.load_state_dict(original_refiner,strict=True)
            model.requires_grad_(False);model.refiner.requires_grad_(True)
            session=MarmSession(model,inputs,cache);expected=772273
        assert sum(p.numel() for p in session.parameters())==expected
        groups=model.parameter_groups(1e-4,1e-4,weight_decay=1e-4)
        groups=[g for g in groups if g["params"]]
        optimizer=torch.optim.AdamW(groups)
        count=dict(pressure_queries=0,residual_node_evaluations=0,time_draws=0,tile_draws=0,initial_derivative_queries=0)
        arm_sampler=AISSampler(mask,n_times=len(centers),seed=seed)
        best=copy.deepcopy(baseline);best_step=0;trajectory=[dict(step=0,accepted=True,**baseline)]
        arm_start=time.monotonic()
        def save(step,metrics):
            state=cpu_state(model if arm.startswith("C") else model.refiner)
            payload=dict(arm=arm,step=step,base_checkpoint=inputs["checkpoint"],
                state_kind="full_model" if arm.startswith("C") else "refiner",state=state,metrics=metrics,
                seed=seed,optimizer_state=optimizer.state_dict(),future_labels_used=False)
            path=armout/f"candidate_{step:02d}.pt"
            torch.save(payload,path)
            return path
        initial_path=save(0,trajectory[0]);shutil.copyfile(initial_path,armout/"best.pt")
        reason="eight_proposal_safety_budget_exhausted"
        try:
            for step in range(1,9):
                status("training_early_only",arm=arm,step=step)
                plan=None if arm.startswith("B") else plans[step-1]
                train=train_update(session,optimizer,early[:6],times[:6],centers,inputs["velocity"],
                    nu_squared,arm_sampler,plan,count,batch=1)
                status("evaluate_early_only",arm=arm,step=step)
                evaluated=evaluate_early(session,early,times)
                count["pressure_queries"]+=8
                accepted=evaluated["selection_mse"]<=best["selection_mse"]
                row=dict(step=step,accepted=accepted,**evaluated,**train,
                    selection_MSE_ratio_to_initial=evaluated["selection_mse"]/baseline["selection_mse"],
                    elapsed_s=time.monotonic()-arm_start,**count)
                trajectory.append(row);candidate=save(step,row)
                write(armout/"TRAJECTORY.json",trajectory)
                print(json.dumps(dict(arm=arm,step=step,accepted=accepted,
                    fit_error_percent=row["fit_error_percent"],selection_error_percent=row["selection_error_percent"],
                    selection_MSE_ratio_to_initial=row["selection_MSE_ratio_to_initial"],
                    training_step_s=row["training_step_s"])),flush=True)
                if not accepted:
                    reason="first_early_selection_worsening_rollback";break
                best=copy.deepcopy(evaluated);best_step=step
                shutil.copyfile(candidate,armout/"best.pt")
            saved=torch.load(armout/"best.pt",map_location="cpu",weights_only=False)
            (model if arm.startswith("C") else model.refiner).load_state_dict(saved["state"],strict=True)
            material=best["selection_mse"]<=.64*baseline["selection_mse"]
            result=dict(arm=arm,trainable_parameters=expected,best_step=best_step,attempted_proposals=len(trajectory)-1,
                baseline=baseline,best=best,selection_MSE_ratio_to_initial=best["selection_mse"]/baseline["selection_mse"],
                material_early_gate_passed=material,stop_reason=reason,elapsed_s=time.monotonic()-arm_start,
                base_cache_queries_after_arm=cache.generated_queries,**count,physics_training_completed=not arm.startswith("B"),
                future_reference_labels_accessed=False,full_grid_Pf_checked=False)
            write(armout/"RESULT.json",result);results[arm]=result
        except Exception as exc:
            failure=dict(arm=arm,status="failed_not_a_completed_physics_arm",exception=repr(exc),
                traceback=traceback.format_exc(),trajectory=trajectory,**count,future_reference_labels_accessed=False)
            write(armout/"FAILURE.json",failure);results[arm]=failure
            print(traceback.format_exc(),flush=True)
            optimizer.zero_grad(set_to_none=True)
        del optimizer,session
        torch.cuda.empty_cache()
        write(out/"PARTIAL_RESULTS.json",results)
    result=dict(arms=results,initial_verification=verification,total_s=time.monotonic()-start,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),future_reference_labels_accessed=False,
        full_grid_Pf_checked=False,primary_arm="A_refiner_LWC",final_manuscript_algorithm=False)
    write(out/"EARLY_NONLINEAR_RESULT.json",result)
    status("complete_early_only",primary_material_gate=results.get("A_refiner_LWC",{}).get("material_early_gate_passed",False))
    print(json.dumps(dict(output=str(out),total_s=result["total_s"],status="complete_early_only")),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
