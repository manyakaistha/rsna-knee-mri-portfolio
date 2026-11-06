"""Short real-cache image benchmark. Random initialization tests cost, not accuracy."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import timm
from models import Recipe
from data import legal_windows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='mps');ap.add_argument('--steps',type=int,default=5);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(2);dev=torch.device(a.device);cache=Path(__file__).resolve().parents[2]/'roi_work/cache64'
    v=np.load(cache/'vols_shard0.npy',mmap_mode='r');m=np.load(cache/'masks_shard0.npy',mmap_mode='r');results=[]
    candidates=[('efficientnet_b2','tf_efficientnet_b2.ns_jft_in1k',260),('dinov2_small','vit_small_patch14_dinov2.lvd142m',224)]
    def sync():
        if dev.type=='mps':torch.mps.synchronize()
        if dev.type=='cuda':torch.cuda.synchronize()
    for name,arch,res in candidates:
        c=dict(kind='global',encoder=arch,resolution=res,eval_chunk=12);t0=time.perf_counter()
        try:
            net=Recipe(c,pretrained=False).to(dev);opt=torch.optim.AdamW(net.parameters(),lr=3e-5);centers,slots=legal_windows(m[0],12)
            raw=np.stack([v[0,ctr-1:ctr+2] for ctr in centers]);x=torch.tensor(raw,device=dev,dtype=torch.float32)/255
            x=torch.nn.functional.interpolate(x,size=(res,res),mode='bilinear',align_corners=False)
            x=(x-torch.tensor([.485,.456,.406],device=dev)[None,:,None,None])/torch.tensor([.229,.224,.225],device=dev)[None,:,None,None]
            x=x[None];valid=torch.ones(1,12,dtype=torch.bool,device=dev);slots=torch.tensor([slots],device=dev)
            timings=[]
            for step in range(2+a.steps):
                sync();start=time.perf_counter();opt.zero_grad(set_to_none=True);z,_=net(x,valid,slots);torch.nn.functional.binary_cross_entropy_with_logits(z,torch.zeros_like(z)).backward();opt.step();sync()
                if step>=2:timings.append(time.perf_counter()-start)
            net.eval();sync();start=time.perf_counter()
            with torch.no_grad():net(x,valid,slots)
            sync();inf=time.perf_counter()-start
            results.append(dict(candidate=name,encoder=arch,resolution=res,device=str(dev),initialization='random; no validation claim',studies_per_step=1,windows=12,warmup_steps=2,measured_steps=a.steps,seconds_per_step=float(np.mean(timings)),studies_per_second=1/float(np.mean(timings)),inference_seconds_per_study=inf,total_elapsed_seconds=time.perf_counter()-t0,peak_cuda_vram_bytes=torch.cuda.max_memory_allocated() if dev.type=='cuda' else None,mps_current_allocated_bytes=torch.mps.current_allocated_memory() if dev.type=='mps' else None,amp='off',data_stalls='excluded: one preloaded real study; not end-to-end throughput'))
            del net,opt,x
            if dev.type=='mps':torch.mps.empty_cache()
        except Exception as error:results.append(dict(candidate=name,error=str(error),elapsed_seconds=time.perf_counter()-t0))
        a.output.write_text(json.dumps(results,indent=2));print(json.dumps(results[-1]),flush=True)
if __name__=='__main__':main()
