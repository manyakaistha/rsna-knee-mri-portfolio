"""torchrun image/feature training with split-clean contracts and keyed global evaluation."""
import argparse,contextlib,datetime,hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler,Subset
from data import Studies,partitions,prepare,sha
from models import Recipe,metric_loss,LABELS

def seed_worker(worker):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)

def collective(value):
    if not dist.is_initialized():return [value]
    result=[None]*dist.get_world_size();dist.all_gather_object(result,value);return result

def rng_state(device):return {'mps':torch.mps.get_rng_state() if device.type=='mps' else None,'torch':torch.get_rng_state(),'python':random.getstate(),'numpy':np.random.get_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(s):
    if s.get('mps') is not None:torch.mps.set_rng_state(s['mps'])
    torch.set_rng_state(s['torch']);random.setstate(s['python']);np.random.set_state(s['numpy'])
    if s['cuda'] is not None:torch.cuda.set_rng_state_all(s['cuda'])

def aucs(y,p,observed):
    from metrics import binary_auc
    values=[]
    for j in range(y.shape[1]):
        m=observed[:,j];v=y[m,j]>.5
        values.append(float(binary_auc(v,p[m,j])) if len(np.unique(v))==2 else None)
    return {'per_target_auc':values,'macro_auc':float(np.mean(values)) if all(v is not None for v in values) else None,'available_target_mean_auc':float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else None,'scorable_targets':sum(v is not None for v in values),'labels':'weak targets thresholded >0.5; clinical truth not established'}

@torch.inference_mode()
def evaluate(model,dataset,device,c,rank,world,workers,features):
    # Evaluation must use the same saved rank-0 BatchNorm state on every rank.
    if dist.is_initialized():
        for buffer in model.buffers():dist.broadcast(buffer,src=0)
    model.eval();rows=[]
    loader=DataLoader(Subset(dataset,list(range(rank,len(dataset),world))),batch_size=1,num_workers=workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker)
    for b in loader:
        x,m,s,y,o=prepare(b,device,c,features);z,_=model(x,m,s)
        rows.append((b['uid'][0],z.float().cpu().numpy()[0],y.cpu().numpy()[0],o.cpu().numpy()[0]))
    rows=[row for group in collective(rows) for row in group];rows.sort(key=lambda r:r[0])
    if len(rows)!=len(dataset) or len({r[0] for r in rows})!=len(rows):raise ValueError('validation duplicates/missing rows')
    ids,z,y,o=zip(*rows);z=np.stack(z);p=1/(1+np.exp(-np.clip(z,-80,80)))
    return dict(ids=np.array(ids),logits=z,probabilities=p,truth=np.stack(y),observed=np.stack(o),targets=np.array(c['targets'])),aucs(np.stack(y),p,np.stack(o))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--split',type=Path,required=True);ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--roi',type=Path);ap.add_argument('--roi-manifest',type=Path);ap.add_argument('--features',type=Path);ap.add_argument('--feature-manifest',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=0);ap.add_argument('--device',default='cuda');ap.add_argument('--max-seconds',type=float,required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--evaluate-locked',action='store_true');a=ap.parse_args()
    c=json.loads(a.config.read_text());splits=partitions(a.split);world=int(os.getenv('WORLD_SIZE','1'));rank=int(os.getenv('RANK','0'));local=int(os.getenv('LOCAL_RANK','0'))
    device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:
        if device.type=='mps':raise ValueError('MPS cannot run DDP')
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=datetime.timedelta(minutes=10))
    try:
        random.seed(c['seed']+rank);np.random.seed(c['seed']+rank);torch.manual_seed(c['seed'])
        if a.max_seconds<=0:raise ValueError('explicit positive wall budget required')
        if c['global_batch']%(world*c['microbatch']):raise ValueError('global batch must equal world * microbatch * accumulation')
        accum=c['global_batch']//(world*c['microbatch']);features=bool(a.features)
        inputs={'config':sha(a.config),'split':sha(a.split),'labels':sha(a.labels),'world':world}
        if features:
            if not a.feature_manifest:raise ValueError('feature ancestry manifest required')
            fm=json.loads(a.feature_manifest.read_text());seen=set(fm['encoder_fit_ids'])
            if seen-set(splits['fit']):raise ValueError('feature encoder saw non-fit studies')
            if fm['bank_sha256']!=sha(a.features):raise ValueError('feature-bank hash mismatch')
            if c.get('require_window_geometry',True) and not fm.get('window_geometry_verified',False):raise ValueError('feature bank lacks source-window geometry verification')
            inputs['feature_manifest']=sha(a.feature_manifest)
        else:
            inputs['cache']={name:sha(a.cache/name) for name in ('ids_shard0.npy','ids_shard1.npy','masks_shard0.npy','masks_shard1.npy','meta_shard0.json','meta_shard1.json')}
            if c.get('require_window_geometry',True) and c['kind']!='roi':
                for shard in range(2):inputs['cache'][f'centers_shard{shard}.npy']=sha(a.cache/f'centers_shard{shard}.npy')
        if c['kind']=='roi':
            if not a.roi_manifest:raise ValueError('split-clean ROI prior-fit manifest required; existing full-cohort priors are not approved')
            rm=json.loads(a.roi_manifest.read_text())
            if set(rm['prior_fit_ids'])-set(splits['fit']):raise ValueError('ROI prior fit saw non-fit studies')
            if not rm['geometry_gate_passed']:raise ValueError('ROI geometry gate failed')
            if not features and rm['roi_meta_sha256']!=sha(a.roi/'roi_meta.json'):raise ValueError('ROI metadata hash mismatch')
            inputs['roi_manifest']=sha(a.roi_manifest)
        fit=Studies(splits['fit'],a.labels,c,a.cache,a.roi,True,a.features);dev=Studies(splits['dev'],a.labels,c,a.cache,a.roi,False,a.features)
        model=Recipe(c,feature_dim=fit.feature_dim if features else None,pretrained=c['pretrained']).to(device)
        if c['kind']=='metric' and c['microbatch']<2:raise ValueError('metric auxiliary needs at least two studies per rank; use microbatch=2')
        net=DDP(model,device_ids=[local] if device.type=='cuda' else None,find_unused_parameters=False) if world>1 else model
        # Weights initialized identically; stochastic training draws differ by rank.
        torch.manual_seed(c['seed']+rank)
        groups=[{'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.')],'lr':c['head_lr']}]
        if model.encoder is not None:groups.insert(0,{'params':model.encoder.parameters(),'lr':c['backbone_lr']})
        opt=torch.optim.AdamW(groups,weight_decay=c['weight_decay'])
        sampler=DistributedSampler(fit,num_replicas=world,rank=rank,shuffle=True,seed=c['seed'],drop_last=True) if world>1 else None
        loader=DataLoader(fit,batch_size=c['microbatch'],sampler=sampler,shuffle=sampler is None,drop_last=True,num_workers=a.workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker,multiprocessing_context='spawn' if a.workers else None)
        updates=len(loader)//accum
        if not updates:raise ValueError('not enough fit studies for global batch')
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step: min(1,(step+1)/max(1,int(c['warmup_fraction']*updates*c['epochs'])))*(.5+.5*math.cos(math.pi*min(1,step/(updates*c['epochs'])))))
        amp=device.type=='cuda' and c['amp']!='off';dtype=torch.bfloat16 if c['amp']=='bf16' else torch.float16
        scaler=torch.amp.GradScaler('cuda',enabled=amp and dtype==torch.float16)
        prevalence=np.nanmean(fit.y[:,c['targets']],axis=0);pw=torch.tensor(np.clip((1-np.clip(prevalence,.03,.7))/np.clip(prevalence,.03,.7),1,10),device=device,dtype=torch.float32)
        if not torch.isfinite(pw).all():raise ValueError('fit target has no observed labels')
        a.output.mkdir(parents=True,exist_ok=True);identity={'config':c,'inputs':inputs,'fit_ids':splits['fit'],'dev_ids':splits['dev'],'target_order':[LABELS[t] for t in c['targets']]};signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        start=0;history=[]
        if a.resume:
            ck=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False)
            if ck['signature']!=signature:raise ValueError('resume config/input/world mismatch')
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler']);restore_rng(ck['rng'][rank]);start=ck['epoch']+1;history=ck['history']
        elif (a.output/'last.pt').exists():raise ValueError('output has checkpoint; choose fresh output or --resume')
        if rank==0:(a.output/'contract.json').write_text(json.dumps(identity,indent=2))
        t0=time.perf_counter();stopped=False
        for ep in range(start,c['epochs']):
            fit.epoch=ep
            if sampler:sampler.set_epoch(ep)
            net.train();it=iter(loader);wait=0;step_seconds=[];losses=[];epoch0=time.perf_counter()
            for update in range(updates):
                tw=time.perf_counter();batches=[next(it) for _ in range(accum)];wait+=time.perf_counter()-tw
                denom=torch.tensor(sum(int(b['observed'].sum()) for b in batches),device=device,dtype=torch.float32)
                if world>1:dist.all_reduce(denom)
                opt.zero_grad(set_to_none=True);ts=time.perf_counter()
                for j,b in enumerate(batches):
                    x,m,s,y,o=prepare(b,device,c,features)
                    with net.no_sync() if world>1 and j<accum-1 else contextlib.nullcontext():
                        with torch.autocast(device.type,dtype=dtype,enabled=amp):z,e=net(x,m,s)
                        raw=F.binary_cross_entropy_with_logits(z.float(),y,pos_weight=pw,reduction='none')
                        loss=(raw*o).sum()*world/denom.clamp_min(1)
                        if c['metric_weight']:loss=loss+c['metric_weight']*metric_loss(e.float(),y,o)/accum
                        scaler.scale(loss).backward();losses.append(float(loss.detach()))
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),3);old=scaler.get_scale();scaler.step(opt);scaler.update()
                if scaler.get_scale()>=old:scheduler.step()
                if device.type=='cuda':torch.cuda.synchronize(device)
                step_seconds.append(time.perf_counter()-ts)
                elapsed=torch.tensor(time.perf_counter()-t0,device=device)
                if world>1:dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
                if elapsed.item()>a.max_seconds:stopped=True;break
            if stopped:
                if rank==0:(a.output/'budget_stop.json').write_text(json.dumps({'epoch':ep,'update':update,'elapsed_seconds':float(elapsed),'checkpoint':'last completed epoch only; partial epoch discarded on resume'}))
                break
            global_loss=torch.tensor(sum(losses)/updates,device=device)
            if world>1:dist.all_reduce(global_loss);global_loss/=world
            tv=time.perf_counter();pred,score=evaluate(model,dev,device,c,rank,world,a.workers,features);validation=time.perf_counter()-tv
            record={'epoch':ep,'train_seconds':tv-epoch0,'validation_seconds':validation,'data_wait_seconds_rank0':wait,'seconds_per_update_rank0':float(np.mean(step_seconds)),'studies_per_second':updates*c['global_batch']/(tv-epoch0),'scaled_micro_loss_rank0':float(np.mean(losses)),'train_loss_global':float(global_loss),'peak_vram_bytes_rank0':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,'dev':score};history.append(record)
            states=collective(rng_state(device))
            if rank==0:
                np.savez_compressed(a.output/f'dev_epoch{ep}.npz',**pred)
                payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=states,epoch=ep,history=history,signature=signature,contract=identity)
                tmp=a.output/'last.tmp';torch.save(payload,tmp);os.replace(tmp,a.output/'last.pt');(a.output/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(record),flush=True)
            if world>1:dist.barrier()
        if not stopped and a.evaluate_locked:
            locked=Studies(splits['locked'],a.labels,c,a.cache,a.roi,False,a.features)
            pred,score=evaluate(model,locked,device,c,rank,world,a.workers,features)
            if rank==0:np.savez_compressed(a.output/'locked_final.npz',**pred);(a.output/'locked_score.json').write_text(json.dumps(score,indent=2))
    finally:
        if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
