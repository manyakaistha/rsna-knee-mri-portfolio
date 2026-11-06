"""Small generic-pretrained, patient-clean representation screen on weak-fit subsets."""
import json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from data import Studies,prepare,sha
from models import Recipe,MIL,LABELS
from train import aucs

def main():
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    repo=Path(__file__).resolve().parents[2];out=repo/'reports/portfolio_2026-10-13/representation_screen';out.mkdir(exist_ok=True)
    split=pd.read_csv(repo/'kaggle/model_research/output/input_audit/weak_split.csv',dtype=str)
    pool=split[split.split=='weak_fit'].copy();pool['h']=pool.patient_sha256.map(lambda p:hashlib.sha256(('portfolio-screen-20261013|'+p).encode()).hexdigest())
    pool=pool.sort_values('h');fit=pool.iloc[:96].StudyInstanceUID.tolist();dev=pool.iloc[96:160].StudyInstanceUID.tolist()
    assert not set(fit)&set(dev);assert not set(pool.iloc[:96].patient_sha256)&set(pool.iloc[96:160].patient_sha256)
    manifest={'fit_ids':fit,'dev_ids':dev,'locked':'historical weak_locked excluded entirely','labels':'report-derived; binary evaluation >0.5','initialization':'generic pretrained only; no knee-fitted checkpoint','selection_gate':'choose DINO if mean candidate blend improvement >=.002 versus B2 reference on this small dev AND measured local cost acceptable; otherwise retain B2 provisionally','caution':'64-study development screen; not independent clinical validation or full ensemble gain','split_hash':sha(repo/'kaggle/model_research/output/input_audit/weak_split.csv')}
    (out/'contract.json').write_text(json.dumps(manifest,indent=2))
    results=[];device=torch.device('mps');labels=repo/'data/processed/train_pool.csv'
    for name,arch,res in [('b2','tf_efficientnet_b2.ns_jft_in1k',260),('dinov2','vit_small_patch14_dinov2.lvd142m',224)]:
        t0=time.perf_counter();c=dict(kind='global',encoder=arch,resolution=res,eval_chunk=6,seed=42,targets=list(range(12)),k_train=6,k_eval=6,require_window_geometry=False)
        try:
            model=Recipe(c,pretrained=True).to(device).eval();ds=Studies(fit+dev,labels,c,repo/'roi_work/cache64',train=False);h=[];ms=[];ys=[]
            with torch.inference_mode():
                for i,b in enumerate(DataLoader(ds,batch_size=1,num_workers=0)):
                    x,m,s,y,o=prepare(b,device,c);features=model.encoder(x.flatten(0,1)).reshape(1,6,-1)
                    h.append(features.cpu());ms.append(m.cpu());ys.append(y.cpu())
                    if (i+1)%40==0:print(name,'feature studies',i+1,flush=True)
            bank=torch.cat(h);mask=torch.cat(ms);truth=torch.cat(ys);del model;torch.mps.empty_cache();feature_seconds=time.perf_counter()-t0
            np.savez_compressed(out/f'{name}_features.npz',ids=np.array(fit+dev),features=bank.numpy().astype(np.float16),valid=mask.numpy())
            torch.manual_seed(42);head=MIL(bank.shape[-1]);opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.02)
            p=truth[:96].mean(0).clamp(.03,.7);pw=((1-p)/p).clamp(1,10)
            head0=time.perf_counter()
            for ep in range(20):
                head.train();order=torch.randperm(96)
                for rows in order.split(8):
                    opt.zero_grad();z,_=head(bank[rows],mask[rows]);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,truth[rows],pos_weight=pw);loss.backward();opt.step()
            head.eval()
            with torch.no_grad():z,_=head(bank[96:],mask[96:]);p=z.sigmoid().numpy()
            np.savez_compressed(out/f'{name}_dev.npz',ids=np.array(dev),probabilities=p,truth=truth[96:].numpy(),observed=np.ones_like(p,bool),targets=np.arange(12))
            score=aucs(truth[96:].numpy(),p,np.ones_like(p,bool));torch.save(head.state_dict(),out/f'{name}_head.pt')
            results.append({'candidate':name,'score':score,'feature_seconds':feature_seconds,'head_seconds':time.perf_counter()-head0,'study_count':160,'windows':6,'head_epochs':20,'encoder_trainable':False})
        except Exception as e:results.append({'candidate':name,'error':str(e),'elapsed_seconds':time.perf_counter()-t0})
        (out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results[-1]),flush=True)
if __name__=='__main__':main()
