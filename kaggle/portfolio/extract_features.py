"""Extract legal-window feature banks; reject held-out ancestry in knee checkpoints."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import Studies,partitions,prepare,sha
from models import Recipe

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--split',type=Path,required=True);ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--checkpoint',type=Path);ap.add_argument('--roi',type=Path);ap.add_argument('--roi-manifest',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args()
    c=json.loads(a.config.read_text());parts=partitions(a.split);net=Recipe(c,pretrained=a.checkpoint is None);fit_ids=[]
    if a.checkpoint:
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);fit_ids=ck['contract']['fit_ids']
        if set(fit_ids)-set(parts['fit']):raise ValueError('checkpoint trained outside current fit')
        net.load_state_dict(ck['model'])
    if c['kind']=='roi':
        if not a.roi_manifest:raise ValueError('split-clean ROI manifest required')
        rm=json.loads(a.roi_manifest.read_text())
        if set(rm['prior_fit_ids'])-set(parts['fit']) or not rm['geometry_gate_passed'] or rm['roi_meta_sha256']!=sha(a.roi/'roi_meta.json'):raise ValueError('ROI prior ancestry/geometry/hash failed')
    device=torch.device(a.device);net.to(device).eval();dataset=Studies(sum(parts.values(),[]),a.labels,c,a.cache,a.roi,False)
    ids=[];features=[];valid=[];slots=[]
    with torch.inference_mode():
        for b in DataLoader(dataset,batch_size=1,num_workers=0):
            x,m,s,_,_=prepare(b,device,c);shape=x.shape[:-3];h=torch.cat([net.encoder(part) for part in x.reshape(-1,*x.shape[-3:]).split(c['eval_chunk'])])
            features.append(h.reshape(*shape,-1).cpu().numpy());valid.append(m.cpu().numpy());slots.append(s.cpu().numpy());ids.extend(b['uid'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(a.output,ids=np.array(ids),features=np.concatenate(features).astype(np.float16),valid=np.concatenate(valid),slots=np.concatenate(slots))
    manifest={'bank_sha256':sha(a.output),'encoder_fit_ids':fit_ids,'initialization':c['encoder'],'checkpoint_sha256':sha(a.checkpoint) if a.checkpoint else None,'config_sha256':sha(a.config),'split_sha256':sha(a.split),'labels_sha256':sha(a.labels),'kind':'frozen-head screen; not independent ensemble member','windows':c['k_eval'],'window_geometry_verified':c.get('require_window_geometry',True)}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
