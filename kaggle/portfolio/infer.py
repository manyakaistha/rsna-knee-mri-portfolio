"""Offline inference for newly trained portfolio checkpoints; strict cache parity."""
import argparse,json,tempfile,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from data import Studies,prepare,sha
from models import Recipe,LABELS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--studies',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--roi',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=ck['contract']['config'];device=torch.device(a.device);net=Recipe(c,pretrained=False).to(device);net.load_state_dict(ck['model'],strict=True);net.eval()
    ids=pd.read_csv(a.studies,dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist()
    if len(set(ids))!=len(ids):raise ValueError('duplicate study IDs')
    start=time.perf_counter();prob=[];available=[]
    with tempfile.TemporaryDirectory() as tmp:
        labels=Path(tmp)/'no_labels.csv';frame=pd.DataFrame(np.full((len(ids),12),np.nan),columns=LABELS);frame.insert(0,'StudyInstanceUID',ids);frame.to_csv(labels,index=False)
        dataset=Studies(ids,labels,c,a.cache,a.roi,train=False)
        with torch.inference_mode():
            for b in DataLoader(dataset,batch_size=1,num_workers=0,pin_memory=device.type=='cuda'):
                x,m,s,_,_=prepare(b,device,c)
                with torch.autocast(device.type,dtype=torch.bfloat16 if c['amp']=='bf16' else torch.float16,enabled=device.type=='cuda' and c['amp']!='off'):z,_=net(x,m,s)
                prob.append(z.float().sigmoid().cpu().numpy());present=m.any(-1) if c['kind']=='roi' else m.any(-1)[:,None].expand(-1,12)
                available.append(present.cpu().numpy())
    np.savez_compressed(a.output,ids=np.array(ids),probabilities=np.concatenate(prob),available=np.concatenate(available),targets=np.array(c['targets']))
    a.output.with_suffix('.manifest.json').write_text(json.dumps({'checkpoint_sha256':sha(a.checkpoint),'config':c,'studies_sha256':sha(a.studies),'elapsed_seconds':time.perf_counter()-start,'rows':len(ids),'tta':'none','scope':'new portfolio checkpoint; no legacy checkpoint translation'},indent=2))
if __name__=='__main__':main()
