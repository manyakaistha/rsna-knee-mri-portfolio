"""Stage a reviewed offline blend CSV from keyed predictions. Never calls Kaggle."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from metrics import ranks
from models import LABELS,ROI_COLUMNS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--template',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    plan=json.loads(a.plan.read_text());template=pd.read_csv(a.template,dtype={'StudyInstanceUID':str});ids=template.StudyInstanceUID.tolist()
    if len(set(ids))!=len(ids):raise ValueError('duplicate template IDs')
    total=np.zeros((len(ids),12));denom=np.zeros_like(total)
    for member in plan['members']:
        bank=np.load(member['predictions'],allow_pickle=False);keys=bank['ids'].astype(str)
        if len(set(keys))!=len(keys) or set(keys)!=set(ids):raise ValueError('prediction IDs differ from complete test template')
        rows={u:i for i,u in enumerate(keys)};order=[rows[u] for u in ids];cols=bank['targets'].astype(int).tolist();p=bank['probabilities'][order].astype(float)
        if member.get('kind')=='roi' and not set(cols)<=set(ROI_COLUMNS):raise ValueError('ROI affects unsupported target')
        if p.shape!=(len(ids),len(cols)) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('invalid probabilities')
        valid=bank['available'][order].astype(bool) if 'available' in bank else np.ones_like(p,bool)
        if member.get('kind')=='roi' and 'available' not in bank:raise ValueError('ROI availability mask required')
        weights=member['weights'];weights=np.full(12,float(weights)) if np.isscalar(weights) else np.array(weights,dtype=float)
        if weights.shape!=(12,) or not np.isfinite(weights).all() or (weights<0).any():raise ValueError('invalid weights')
        for j,col in enumerate(cols):
            mask=valid[:,j];v=p[:,j].copy()
            if plan['aggregation']=='rank':v[mask]=ranks(v[mask])/max(1,int(mask.sum()))
            elif plan['aggregation']!='probability':raise ValueError('use probability or rank')
            total[:,col]+=v*weights[col]*mask;denom[:,col]+=weights[col]*mask
    if (denom<=0).any():raise ValueError('missing global fallback for one or more study/target')
    result=pd.DataFrame(total/denom,columns=LABELS);result.insert(0,'StudyInstanceUID',ids)
    if a.output.exists():raise FileExistsError('refusing overwrite')
    result.to_csv(a.output,index=False);print('Staged',a.output,'rows',len(result),'No submission sent')
if __name__=='__main__':main()
