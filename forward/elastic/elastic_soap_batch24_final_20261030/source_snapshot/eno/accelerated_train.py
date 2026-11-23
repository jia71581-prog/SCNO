"""Full-wavefield SOAP training, CPU cache, lagged record AIS, epoch checkpoints."""
import copy,gc,json,math,os,signal,time
from pathlib import Path
import torch
from .core import atomic_json,atomic_torch,sha,source_input,operator_input,normalize_input,encode_output,decode_output
from .dataset import Records,data_contract
from .network import build_network_from_config
from .optim.soap_complex import RealViewSOAP
from .training_acceleration import RecordSampler,RecordCache,prime_soap,per_record_loss


def train(cfg,run,stop_after_updates=None):
    run=Path(run);folder=run/'training';folder.mkdir(exist_ok=True)
    tc=cfg['train'];ac=cfg['training_acceleration'];device=torch.device(tc['device'])
    torch.set_num_threads(ac.get('cpu_threads',8));torch.manual_seed(tc['seed']);torch.backends.cuda.matmul.allow_tf32=False
    stats=json.loads((run/'data/normalization.json').read_text());contract=data_contract(cfg)
    trainset=Records(run/'data',contract,'train');validset=Records(run/'data',contract,'validation');n=len(trainset)
    latest=folder/'latest.pt';cp=torch.load(latest,map_location='cpu',weights_only=False,mmap=True) if latest.exists() else None
    model=build_network_from_config(tc).to(device)
    state=dict(epoch=0,position=0,updates=0,order=None,best=float('inf'),train_loss_sum=0.,train_samples=0,soap_updates=0)
    legacy_rng=torch.Generator().manual_seed(tc['seed']).get_state()
    if cp is not None:
        if cp['contract']!=contract:raise ValueError('Training data contract mismatch')
        model.load_state_dict(cp['model']);state=copy.deepcopy(cp['state']);state.setdefault('soap_updates',0)
        legacy_rng=cp['generator_rng'].clone()
    state.setdefault('actual_microbatch',tc['microbatch'])
    # A declared larger batch takes effect at this continuation boundary.
    if cp is not None and (cp['train_config'].get('microbatch')!=tc['microbatch']
            or cp['train_config'].get('activation_checkpoint_projection',False)!=tc.get('activation_checkpoint_projection',False)
            or cp['train_config'].get('projection_chunk_rows',0)!=tc.get('projection_chunk_rows',0)):
        state['actual_microbatch']=tc['microbatch']
    if ac.get('optimizer_state_on_cpu'):
        if ac.get('optimizer_state_prefetch'):
            from .optim.prefetched_soap import PrefetchedRealViewSOAP
            optimizer_class=PrefetchedRealViewSOAP
        else:
            from .optim.offloaded_soap import OffloadedRealViewSOAP
            optimizer_class=OffloadedRealViewSOAP
    else:optimizer_class=RealViewSOAP
    opt=optimizer_class(list(model.named_parameters()),**ac['soap'])
    sampler=RecordSampler(n,seed=tc['seed']+73,**ac['sampler'])
    if cp is not None and cp.get('optimizer_kind')=='RealViewSOAP':
        changed={k for k in set(cp['train_config'])|set(tc) if cp['train_config'].get(k)!=tc.get(k)}
        if changed:
            migration_path=run/'MIGRATION.json'
            migration=json.loads(migration_path.read_text()) if migration_path.exists() else {}
            transition=migration.get('transition')
            allowed=({'microbatch','effective_batch','activation_checkpoint_projection','projection_chunk_rows'} if transition=='increase_batch' else
                     {'operator_block'} if transition=='fuse_operator_block' else set())
            if (changed-allowed or not allowed or migration.get('source_updates')!=state['updates']
                    or migration.get('source_train_config')!=cp['train_config']):
                raise ValueError('Undeclared SOAP configuration transition')
            if transition=='fuse_operator_block':
                state['actual_microbatch']=tc['microbatch']
        if cp['training_acceleration']!=ac:
            migration=json.loads((run/'MIGRATION.json').read_text())
            execution_keys={'optimizer_state_prefetch','validation_batch'}
            stripped={k:v for k,v in ac.items() if k not in execution_keys}
            original={k:v for k,v in cp['training_acceleration'].items() if k not in execution_keys}
            if (migration.get('transition')!='prefetch_soap_state' or stripped!=original
                    or not ac.get('optimizer_state_prefetch') or not ac.get('optimizer_state_on_cpu')
                    or migration.get('source_training_acceleration')!=cp['training_acceleration']
                    or migration.get('source_updates')!=state['updates']):
                raise ValueError('SOAP acceleration configuration mismatch')
        opt.load_state_dict(cp['optimizer']);sampler.load_state_dict(cp['adaptive_sampler'])
        migration_path=run/'MIGRATION.json'
        retry=json.loads(migration_path.read_text()) if migration_path.exists() else {}
        retry_declared=(retry.get('transition')=='prefetch_soap_state'
            and retry.get('source_training_acceleration')==cp['training_acceleration']) or (
            retry.get('transition')=='increase_batch' and retry.get('source_train_config')==cp['train_config'])
        if (retry.get('retry_requested_microbatch') and retry_declared
                and retry.get('source_updates')==state['updates']):
            state['actual_microbatch']=tc['microbatch']
    elif cp is not None:
        migration=json.loads((run/'MIGRATION.json').read_text())
        if state['updates']!=migration['source_updates']:raise ValueError('Legacy checkpoint differs from handoff')
        prime_soap(opt,cp['optimizer'])
        atomic_json(folder/'OPTIMIZER_MIGRATION.json',dict(source_updates=state['updates'],model_preserved=True,
            optimizer='original acoustic-operator-learning RealViewSOAP',covariance_seed='historical Adam first moment',
            soap_moments_and_clock='new zero state; not equivalent Adam continuation',source_checkpoint_retained=True))
    else:raise ValueError('This continuation requires an existing checkpoint')
    torch.set_rng_state(cp['torch_rng'])
    if device.type=='cuda' and cp.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(cp['cuda_rng'])
    del cp;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    corpus=torch.utils.data.ConcatDataset([trainset,validset])
    depth=max(tc['effective_batch']*ac['prefetch_batches'],tc['effective_batch'])
    cache=RecordCache(corpus,max_bytes=int(ac['cache_gib']*1024**3),reserve_bytes=int(ac['reserve_gib']*1024**3),
                      workers=ac['readers'],max_pending=depth*2,process_paths=trainset.paths+validset.paths,contract=contract)
    stopping=[False];old_handlers={}
    for sig in [signal.SIGTERM,signal.SIGINT]:
        old_handlers[sig]=signal.signal(sig,lambda signum,frame:stopping.__setitem__(0,True))
    start=time.time();start_exposures=state['epoch']*n+state['position'];wait_before=0.

    def save(reason):
        atomic_torch(latest,dict(model=model.state_dict(),optimizer=opt.state_dict(),state=copy.deepcopy(state),
            statistics=stats,train_config=tc,contract=contract,generator_rng=legacy_rng,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
            optimizer_kind='RealViewSOAP',training_acceleration=ac,adaptive_sampler=sampler.state_dict()))
        atomic_json(folder/'CHECKPOINT.json',dict(reason=reason,updates=state['updates'],soap_updates=state['soap_updates'],
            epoch=state['epoch'],position=state['position'],exposures=state['epoch']*n+state['position'],saved_at=time.time()))

    def batch_rows(ids):
        rows=[cache.get(i) for i in ids]
        v=torch.stack([x[0] for x in rows]);loc=torch.stack([x[1] for x in rows]);y=torch.stack([x[2] for x in rows])
        if device.type=='cuda':v=v.pin_memory();y=y.pin_memory()
        return v.to(device,non_blocking=True),loc,y.to(device,non_blocking=True)

    def accumulate(plan,physical):
        raw=[];weighted=0.;count=len(plan)
        try:
            for j in range(0,count,physical):
                group=plan[j:j+physical];v,loc,y=batch_rows([i for i,p in group])
                src=source_input(loc,cfg,device);x=normalize_input(operator_input(v,src),stats)
                pred=model(x,time_grid=cfg['grid']['frames']);losses=per_record_loss(pred,encode_output(y,stats))
                weights=torch.tensor([1/(n*p) for i,p in group],device=device,dtype=losses.dtype)
                loss=(losses*weights).sum()/count
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite importance loss')
                loss.backward();raw.extend(losses.detach().cpu().tolist());weighted+=float(loss.detach())
                del v,loc,y,src,x,pred,losses,loss,weights
            return (raw,weighted),None
        except torch.cuda.OutOfMemoryError as exc:
            return None,str(exc)

    @torch.no_grad()
    def validate_group(ids):
        try:
            v,loc,y=batch_rows(ids);src=source_input(loc,cfg,device)
            pred=model(normalize_input(operator_input(v,src),stats),time_grid=cfg['grid']['frames'])
            values=per_record_loss(pred,encode_output(y,stats))
            physical=decode_output(pred,stats)
            relative=torch.linalg.vector_norm((physical-y).flatten(1),dim=1)/torch.linalg.vector_norm(y.flatten(1),dim=1)
            return (float(values.double().sum()),float(relative.double().sum())),None
        except torch.cuda.OutOfMemoryError as exc:return None,str(exc)

    try:
        atomic_json(folder/'RESTORED.json',dict(pid=os.getpid(),updates=state['updates'],soap_updates=state['soap_updates'],
            exposures=state['epoch']*n+state['position'],physical_batch=tc['microbatch'],effective_batch=tc['effective_batch'],
            epoch_only_checkpoint=True,source='latest checkpoint',training_target='original uniform mean, importance corrected'))
        while state['epoch']<tc['epochs']:
            model.train()
            while state['position']<n:
                count=min(tc['effective_batch'],n-state['position']);sampler.ensure(depth)
                plan=sampler.peek(count);cache.prefetch(i for i,p in sampler.peek(depth))
                step_start=time.perf_counter();cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None
                while True:
                    opt.zero_grad(set_to_none=True)
                    outcome,error=accumulate(plan,state['actual_microbatch'])
                    if error is None:
                        raw,weighted=outcome;break
                    if state['actual_microbatch']==1:raise torch.cuda.OutOfMemoryError(error)
                    before=state['actual_microbatch'];state['actual_microbatch']=before-4 if before>=12 else max(1,before//2)
                    opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
                    atomic_json(folder/'BATCH_ADAPTATION.json',dict(reason='forward/backward memory pressure',
                        before=before,after=state['actual_microbatch'],effective_batch=count,
                        parameters_and_optimizer_not_updated=True,pending_draws_preserved=True,oom_error=error))
                fb_s=time.perf_counter()-step_start;finite_start=time.perf_counter()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
                finite_s=time.perf_counter()-finite_start;soap_start=time.perf_counter()
                opt.step();soap_s=time.perf_counter()-soap_start;sampler_start=time.perf_counter()
                sampler.commit(plan,raw);sampler_s=time.perf_counter()-sampler_start
                state['position']+=count;state['updates']+=1;state['soap_updates']+=1
                state['train_loss_sum']+=weighted*count;state['train_samples']+=count
                exposures=state['epoch']*n+state['position'];elapsed=time.time()-start
                status=dict(stage='TRAINING',epoch=state['epoch']+1,epochs=tc['epochs'],records_in_epoch=state['position'],records_per_epoch=n,
                    updates=state['updates'],soap_updates=state['soap_updates'],exposures=exposures,target_exposures=tc['epochs']*n,
                    batch_loss=weighted,raw_record_loss=sum(raw)/count,physical_batch=state['actual_microbatch'],effective_batch=count,
                    sampling='record EMA AIS with replacement; 1/(N*q) corrected',ais_unique_records=int((sampler.visits>0).sum()),
                    sample_ids=[i for i,p in plan],draw_probabilities=[p for i,p in plan],cache=cache.stats(),
                    step_s=time.perf_counter()-step_start,data_wait_s=cache.wait_s-wait_before,
                    forward_backward_and_transfer_s=fb_s,finite_gradient_check_s=finite_s,
                    soap_step_s=soap_s,sampler_commit_s=sampler_s,
                    elapsed_s=elapsed,records_per_s=(exposures-start_exposures)/max(elapsed,1e-9))
                wait_before=cache.wait_s;atomic_json(folder/'STATUS.json',status);print(json.dumps(status),flush=True)
                if stopping[0] or (stop_after_updates is not None and state['updates']>=stop_after_updates):
                    save('requested_stop' if stopping[0] else 'technical_resume_check');return latest
            opt.zero_grad(set_to_none=True);model.eval();cache.drain();val=0.;relative=0.
            atomic_json(folder/'STATUS.json',dict(stage='VALIDATING',epoch=state['epoch']+1,updates=state['updates']))
            validation_batch=int(ac.get('validation_batch',1));i=0;validation_start=time.perf_counter()
            while i<len(validset):
                cache.prefetch(range(n+i,min(n+i+depth,n+len(validset))))
                count=min(validation_batch,len(validset)-i)
                outcome,error=validate_group(list(range(n+i,n+i+count)))
                if error is not None:
                    if validation_batch==1:raise torch.cuda.OutOfMemoryError(error)
                    validation_batch=max(1,validation_batch//2);gc.collect();torch.cuda.empty_cache();continue
                value,rel=outcome;val+=value;relative+=rel;i+=count
                if i%256==0 or i==len(validset):
                    atomic_json(folder/'STATUS.json',dict(stage='VALIDATING',epoch=state['epoch']+1,
                        updates=state['updates'],processed_records=i,total_records=len(validset),
                        validation_batch=validation_batch,validation_elapsed_s=time.perf_counter()-validation_start))
            val/=len(validset);relative/=len(validset)
            metrics=dict(epoch=state['epoch']+1,training_loss=state['train_loss_sum']/state['train_samples'],validation_loss=val,
                validation_physical_relative_l2=relative,updates=state['updates'],soap_updates=state['soap_updates'],exposures=(state['epoch']+1)*n)
            if val<state['best']:
                state['best']=val
                atomic_torch(folder/'best.pt',dict(model=model.state_dict(),statistics=stats,train_config=tc,contract=contract,metrics=metrics))
            with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
            state.update(epoch=state['epoch']+1,position=0,order=None,train_loss_sum=0.,train_samples=0)
            cache.drain();save('epoch_complete');print(json.dumps(metrics),flush=True)
        atomic_json(folder/'COMPLETE.json',dict(status='BUDGET_COMPLETED',epochs=tc['epochs'],contract=contract,optimizer='RealViewSOAP'))
        return folder/'best.pt'
    finally:
        cache.close()
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
