"""Full effective batches, train-only normalizers, exact optimizer/RNG restart."""
import json,time,os,gc
from pathlib import Path
import numpy as np
import torch
from .core import atomic_json,atomic_torch,sha,source_input,operator_input,normalize_input,encode_output,decode_output,relative_training_loss
from .dataset import Records,data_contract
from .network import build_network_from_config


def train(cfg,run,stop_after_updates=None):
    if cfg.get('training_acceleration',{}).get('enabled'):
        from .accelerated_train import train as accelerated
        return accelerated(cfg,run,stop_after_updates)
    run=Path(run);folder=run/'training';folder.mkdir(exist_ok=True)
    tc=cfg['train'];device=torch.device(tc['device']);torch.set_num_threads(2)
    torch.manual_seed(tc['seed']);torch.backends.cuda.matmul.allow_tf32=False
    stats=json.loads((run/'data/normalization.json').read_text());contract=data_contract(cfg)
    trainset=Records(run/'data',contract,'train');validset=Records(run/'data',contract,'validation')
    model=build_network_from_config(tc)
    opt=torch.optim.Adam(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'],foreach=False)
    gen=torch.Generator().manual_seed(tc['seed'])
    state=dict(epoch=0,position=0,updates=0,order=None,best=float('inf'),train_loss_sum=0.,train_samples=0)
    latest=folder/'latest.pt'
    if latest.exists():
        cp=torch.load(latest,map_location='cpu',weights_only=False)
        if cp['contract']!=contract or cp['train_config']!=tc:raise ValueError('training resume configuration mismatch')
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);state=cp['state']
        gen.set_state(cp['generator_rng']);torch.set_rng_state(cp['torch_rng'])
        if device.type=='cuda' and cp.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(cp['cuda_rng'])
        del cp
    model.to(device)
    for values in opt.state.values():
        for key,val in values.items():
            if isinstance(val,torch.Tensor) and key!='step':values[key]=val.to(device)
    def save():
        atomic_torch(latest,dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state.copy(),statistics=stats,
            train_config=tc,contract=contract,generator_rng=gen.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None))
    t0=time.time()
    while state['epoch']<tc['epochs']:
        model.train()
        if state['order'] is None:state['order']=torch.randperm(len(trainset),generator=gen).tolist()
        while state['position']<len(trainset):
            ids=state['order'][state['position']:state['position']+tc['effective_batch']]
            opt.zero_grad(set_to_none=True);value=0.
            for j in range(0,len(ids),tc['microbatch']):
                group=ids[j:j+tc['microbatch']];rows=[trainset[i] for i in group]
                v=torch.stack([a[0] for a in rows]).to(device);loc=torch.stack([a[1] for a in rows])
                y=encode_output(torch.stack([a[2] for a in rows]).to(device),stats)
                src=source_input(loc,cfg,device)
                x=normalize_input(operator_input(v,src),stats)
                pred=model(x,time_grid=cfg['grid']['frames'])
                loss=relative_training_loss(pred,y)*len(group)/len(ids)
                if not torch.isfinite(loss):raise FloatingPointError('training loss non-finite')
                loss.backward();value+=float(loss.detach())
                del x,pred,loss,y,src,v
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('training gradient non-finite')
            opt.step();state['position']+=len(ids);state['updates']+=1
            state['train_loss_sum']+=value*len(ids);state['train_samples']+=len(ids)
            status=dict(stage='TRAINING',epoch=state['epoch']+1,epochs=tc['epochs'],records_in_epoch=state['position'],records_per_epoch=len(trainset),updates=state['updates'],batch_loss=value,elapsed_s=time.time()-t0)
            atomic_json(folder/'STATUS.json',status)
            print(json.dumps(status),flush=True)
            if state['updates']%tc['checkpoint_every_updates']==0:save()
            if stop_after_updates is not None and state['updates']>=stop_after_updates:
                save();del opt,model;gc.collect();torch.cuda.empty_cache();return latest
        model.eval();val=0.;relative_l2=0.
        with torch.no_grad():
            for v,loc,u in validset:
                src=source_input(loc[None],cfg,device)
                pred=model(normalize_input(operator_input(v.to(device),src),stats),time_grid=cfg['grid']['frames'])
                target=u[None].to(device)
                val+=float(relative_training_loss(pred,encode_output(target,stats)))
                physical=decode_output(pred,stats)
                relative_l2+=float(torch.linalg.vector_norm(physical-target)/torch.linalg.vector_norm(target))
        val/=len(validset);relative_l2/=len(validset)
        metrics=dict(epoch=state['epoch']+1,training_loss=state['train_loss_sum']/state['train_samples'],validation_loss=val,validation_physical_relative_l2=relative_l2,updates=state['updates'])
        if val<state['best']:
            state['best']=val
            atomic_torch(folder/'best.pt',dict(model=model.state_dict(),statistics=stats,train_config=tc,contract=contract,metrics=metrics))
        with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        state.update(epoch=state['epoch']+1,position=0,order=None,train_loss_sum=0.,train_samples=0)
        save();print(json.dumps(metrics),flush=True)
    best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    atomic_json(folder/'COMPLETE.json',dict(status='BUDGET_COMPLETED',epochs=tc['epochs'],best_metrics=best['metrics'],best_sha256=sha(folder/'best.pt'),contract=contract))
    del opt,model,best
    gc.collect();torch.cuda.empty_cache()
    return folder/'best.pt'
