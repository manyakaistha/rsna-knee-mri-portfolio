"""CPU architecture contracts and checkpoint round trips; no clinical training claims."""
import io,json,time
from pathlib import Path
import torch
from models import Recipe,roi_fuse,ROI_COLUMNS,metric_loss
from data import legal_windows
import numpy as np

def main():
    torch.set_num_threads(2);torch.manual_seed(42);results=[]
    centers,slots=legal_windows(np.ones(64,bool),62)
    assert len([v for v in centers if v>=0])==54
    for ctr,s in zip(centers,slots):
        if ctr>=0:assert sum((18,14,12,8,12)[:s])<ctr<sum((18,14,12,8,12)[:s+1])-1
    for path in sorted((Path(__file__).parent/'configs').glob('*.json')):
        c=json.loads(path.read_text());net=Recipe(c,tiny=True,pretrained=False)
        shape=(2,5,4,3,16,16) if c['kind']=='roi' else (2,6,3,16,16)
        x=torch.randn(shape);mask=torch.ones(shape[:-3],dtype=torch.bool);mask[0]=False
        slots=torch.tensor([[0,0,1,2,3,4]]*2) if c['kind']!='roi' else torch.zeros(mask.shape,dtype=torch.long)
        z,e=net(x,mask,slots);assert z.shape==(2,len(c['targets'])) and torch.isfinite(z).all()
        loss=z.square().mean()
        if c['kind']=='metric':loss+=metric_loss(e,torch.rand_like(z),torch.ones_like(z,dtype=torch.bool))
        loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        opt=torch.optim.AdamW(net.parameters());opt.step();net.eval()
        before=net(x,mask,slots)[0];stream=io.BytesIO();torch.save({'model':net.state_dict(),'optimizer':opt.state_dict()},stream);stream.seek(0)
        restored=Recipe(c,tiny=True,pretrained=False);ck=torch.load(stream,weights_only=False);restored.load_state_dict(ck['model']);restored.eval()
        assert torch.equal(before,restored(x,mask,slots)[0])
        # Masked pixel/feature perturbations must not change predictions.
        altered=x.clone();altered[~mask]=1000
        assert torch.allclose(before,net(altered,mask,slots)[0],atol=1e-6)
        results.append({'recipe':c['name'],'forward_backward':'passed','checkpoint_roundtrip':'bit-identical','masked_perturbation':'passed','encoder':'tiny CPU test encoder; production encoder not tested here'})
    global_z=torch.randn(2,12);roi_z=torch.randn(2,5);valid=torch.zeros(2,5,4,dtype=torch.bool)
    assert torch.equal(global_z.sigmoid(),roi_fuse(global_z,roi_z,valid))
    valid[:,0]=True;p=roi_fuse(global_z,roi_z,valid)
    assert torch.equal(p[:,[t for t in range(12) if t!=ROI_COLUMNS[0]]],global_z.sigmoid()[:,[t for t in range(12) if t!=ROI_COLUMNS[0]]])
    print(json.dumps({'tests':results,'legal_window_max':54,'roi_fallback_target_isolation':'passed','gpu_ddp':'not exercised'},indent=2))
if __name__=='__main__':main()
