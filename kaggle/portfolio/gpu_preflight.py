"""Bounded Kaggle cost preflight: real images, no validation or submission."""
import argparse,json,time,subprocess,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data import Studies,prepare
from models import Recipe

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--split',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True)
    import pandas as pd
    torch.set_num_threads(2);hardware={'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[{'name':torch.cuda.get_device_name(i),'total_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]};(a.output/'hardware.json').write_text(json.dumps(hardware,indent=2));print(json.dumps(hardware),flush=True)
    if not hardware['gpus']:raise RuntimeError('Kaggle supplied no GPU')
    if len(hardware['gpus'])>=2:
        proc=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(Path(__file__).with_name('ddp_smoke.py'))],env=dict(__import__('os').environ,DDP_DEVICE='cuda'),capture_output=True,text=True,timeout=90)
        (a.output/'ddp_cuda.log').write_text(proc.stdout+proc.stderr)
        if proc.returncode:raise RuntimeError('CUDA DDP smoke failed; see log')
    ids=pd.read_csv(a.split,dtype=str).query("split == 'fit'").StudyInstanceUID.tolist()[:32];results=[];started=time.perf_counter()
    for file in ['05_efficientnet_global.json','01_global_convnext.json','02_global_coatnet.json']:
        c=json.loads((Path(__file__).parent/'configs'/file).read_text());c['require_window_geometry']=False
        ds=Studies(ids,a.labels,c,a.cache,train=True);dl=DataLoader(ds,batch_size=1,num_workers=0,pin_memory=True);device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats()
        try:
            model=Recipe(c,pretrained=True).to(device).train();opt=torch.optim.AdamW(model.parameters(),lr=c['backbone_lr']);scaler=torch.amp.GradScaler('cuda');timings=[];wait=0;it=iter(dl);t0=time.perf_counter()
            measured=100 if file=='01_global_convnext.json' else 10
            for step in range(5+measured):
                t=time.perf_counter()
                try:b=next(it)
                except StopIteration:it=iter(dl);b=next(it)
                wait+=time.perf_counter()-t;x,m,s,y,o=prepare(b,device,c);opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16):z,_=model(x,m,s)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(z.float(),y);scaler.scale(loss).backward();scaler.step(opt);scaler.update();torch.cuda.synchronize()
                if step>=5:timings.append(time.perf_counter()-t)
            model.eval();t=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):model(x,m,s)
            torch.cuda.synchronize();inf=time.perf_counter()-t
            results.append({'recipe':c['name'],'gpu_count_used':1,'microbatch':1,'windows':c['k_train'],'effective_batch_in_benchmark':1,'warmup':5,'steps':measured,'seconds_per_step':sum(timings)/len(timings),'studies_per_second':len(timings)/sum(timings),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'data_wait_seconds':wait,'inference_seconds_per_study':inf,'elapsed_seconds':time.perf_counter()-t0,'purpose':'cost screen only; source-SOP geometry audit pending; no validation claim'})
        except torch.cuda.OutOfMemoryError as error:results.append({'recipe':c['name'],'status':'OOM; rejected configuration','error':str(error)})
        (a.output/'benchmarks.json').write_text(json.dumps(results,indent=2));print(json.dumps(results[-1]),flush=True)
        if 'model' in locals():del model,opt
        torch.cuda.empty_cache()
        if time.perf_counter()-started>600:break
if __name__=='__main__':main()
