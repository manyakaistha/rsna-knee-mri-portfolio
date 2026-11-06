"""Two-rank CPU/Gloo gradient normalization test; this does not exercise CUDA/NCCL."""
import contextlib,datetime,json,os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    torch.set_num_threads(1);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2
    device=torch.device('cuda',rank) if os.getenv('DDP_DEVICE')=='cuda' else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(rank)
    dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',init_method='file://'+os.environ['DDP_TEST_FILE'] if os.getenv('DDP_TEST_FILE') else 'env://',rank=rank,world_size=world,timeout=datetime.timedelta(seconds=45))
    try:
        torch.manual_seed(42);base=torch.nn.Linear(3,12).to(device);net=torch.nn.Linear(3,12).to(device);net.load_state_dict(base.state_dict());ddp=DDP(net,device_ids=[rank] if device.type=='cuda' else None)
        x=torch.arange(24).reshape(8,3).float()/24;y=(torch.arange(96).reshape(8,12)%3==0).float();mask=torch.arange(96).reshape(8,12)%4!=0
        x=x.to(device);y=y.to(device);mask=mask.to(device)
        # Reference: eight-study global batch, masked-cell normalization.
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(base(x),y,reduction='none')*mask).sum()/mask.sum();loss.backward()
        denominator=mask[rank*4:(rank+1)*4].sum().float();dist.all_reduce(denominator)
        for micro in range(2):
            sl=slice(rank*4+micro*2,rank*4+micro*2+2)
            with ddp.no_sync() if micro==0 else contextlib.nullcontext():
                local=(torch.nn.functional.binary_cross_entropy_with_logits(ddp(x[sl]),y[sl],reduction='none')*mask[sl]).sum()*world/denominator
                local.backward()
        difference=max((a.grad-b.grad).abs().max().item() for a,b in zip(base.parameters(),net.parameters()));assert difference<1e-7,difference
        if rank==0:print(json.dumps({'backend':'nccl' if device.type=='cuda' else 'gloo','device':str(device),'ranks':2,'effective_global_batch':8,'microbatch':2,'accumulation':2,'masked_loss_gradient_max_difference':difference,'status':'passed','cuda_nccl':'passed' if device.type=='cuda' else 'not exercised'}))
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
