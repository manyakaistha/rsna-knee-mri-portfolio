"""Matched global-batch8 image cost probe; no validation/model adoption claims."""
import argparse,contextlib,json,os,time
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from data import Studies,prepare
from models import Recipe

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--split',type=Path,required=True);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=50);ap.add_argument('--gpu',type=int,default=0);a=ap.parse_args()
    world=int(os.getenv('WORLD_SIZE','1'));rank=int(os.getenv('RANK','0'));local=int(os.getenv('LOCAL_RANK',str(a.gpu)));torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    if world>1:dist.init_process_group('nccl')
    try:
        c=json.loads(a.config.read_text());c['require_window_geometry']=False;torch.manual_seed(42);net=Recipe(c,pretrained=True).to(device).train();model=DDP(net,device_ids=[local]) if world>1 else net;torch.manual_seed(42+rank)
        ids=pd.read_csv(a.split,dtype=str).query("split=='fit'").StudyInstanceUID.tolist()[:64];ds=Studies(ids,a.labels,c,a.cache,train=True)
        # Rank strides correspond to a shared fixed eight-study effective update.
        indices=[i for base in range(0,64,8) for i in range(base+rank*4,base+(rank+1)*4)] if world==2 else list(range(64))
        dl=DataLoader(torch.utils.data.Subset(ds,indices),batch_size=2,num_workers=0,pin_memory=True);it=iter(dl);accum=8//(world*2);opt=torch.optim.AdamW(net.parameters(),lr=3e-5);scaler=torch.amp.GradScaler('cuda');times=[];wait=0;torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        for step in range(5+a.steps):
            tw=time.perf_counter();batches=[]
            for _ in range(accum):
                try:batches.append(next(it))
                except StopIteration:it=iter(dl);batches.append(next(it))
            wait+=time.perf_counter()-tw
            if world>1:dist.barrier()
            torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
            for j,b in enumerate(batches):
                x,m,s,y,o=prepare(b,device,c)
                with model.no_sync() if world>1 and j<accum-1 else contextlib.nullcontext():
                    with torch.autocast('cuda',dtype=torch.float16):z,_=model(x,m,s)
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(z.float(),y)/accum;scaler.scale(loss).backward()
            scaler.step(opt);scaler.update();torch.cuda.synchronize()
            seconds=torch.tensor(time.perf_counter()-t,device=device)
            if world>1:dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
            if step>=5:times.append(float(seconds))
        record={'world':world,'gpu':local,'microbatch':2,'accumulation':accum,'effective_global_batch':8,'warmup':5,'steps':a.steps,'seconds_per_update':float(np.mean(times)),'studies_per_second':8/float(np.mean(times)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'data_wait_seconds_rank0':wait,'elapsed_seconds':time.perf_counter()-started,'purpose':'real-cache cost probe; slot-safe, source-SOP audit pending; fixed fit-only64studies; no validation','recipe':c['name'],'amp':'fp16'}
        if rank==0:a.output.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    finally:
        if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
