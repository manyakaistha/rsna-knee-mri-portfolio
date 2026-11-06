"""Key-aligned ensemble analysis. No weights are fitted on locked validation."""
import argparse,json
from pathlib import Path
import numpy as np
from metrics import binary_auc,ranks

def macro(y,p):
    return float(np.mean([binary_auc(y[:,j]>.5,p[:,j]) for j in range(12)]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('predictions',nargs='+',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--draws',type=int,default=1000);ap.add_argument('--context',required=True);a=ap.parse_args()
    banks=[np.load(p,allow_pickle=False) for p in a.predictions];ids=sorted(banks[0]['ids'].astype(str));aligned=[];truth=None
    for bank in banks:
        keys=bank['ids'].astype(str)
        if len(set(keys))!=len(keys) or set(keys)!=set(ids):raise ValueError('duplicate/missing study keys')
        index={u:i for i,u in enumerate(keys)};rows=[index[u] for u in ids]
        if not np.array_equal(bank['targets'],np.arange(12)):raise ValueError('global analysis requires canonical 12 targets; apply mask-aware ROI fusion first')
        y=bank['truth'][rows];p=bank['probabilities'][rows]
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('invalid probabilities')
        if truth is not None and not np.array_equal(truth,y):raise ValueError('truth mismatch')
        if not bank['observed'][rows].all():raise ValueError('all-target observed labels required for this analysis')
        truth=y;aligned.append(p)
    p=np.stack(aligned);names=[v.stem for v in a.predictions];base=p[0];uniform=p.mean(0);logit=1/(1+np.exp(-np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).mean(0)))
    rank=np.stack([np.stack([ranks(member[:,j])/len(ids) for j in range(12)],1) for member in p]).mean(0)
    rng=np.random.default_rng(42);deltas=[];attempts=0
    while len(deltas)<a.draws and attempts<a.draws*5:
        attempts+=1;ix=rng.integers(0,len(ids),len(ids));y=truth[ix]
        if any(len(np.unique(y[:,j]>.5))<2 for j in range(12)):continue
        deltas.append(macro(y,uniform[ix])-macro(y,base[ix]))
    correlations={}
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            correlations[names[i]+' vs '+names[j]]=[float(np.corrcoef(p[i,:,t]-(truth[:,t]>.5),p[j,:,t]-(truth[:,t]>.5))[0,1]) for t in range(12)]
    report={'context':a.context,'studies':len(ids),'member_auc':{n:macro(truth,v) for n,v in zip(names,p)},'per_target_auc':{n:[binary_auc(truth[:,j]>.5,v[:,j]) for j in range(12)] for n,v in zip(names,p)},'blend_auc':{'uniform_probability':macro(truth,uniform),'uniform_logit':macro(truth,logit),'uniform_rank':macro(truth,rank)},'paired_uniform_minus_first_member':{'point':macro(truth,uniform)-macro(truth,base),'ci95':np.quantile(deltas,[.025,.975]).tolist(),'win_fraction':float(np.mean(np.array(deltas)>0)),'draws':len(deltas),'rejected_single_class_draws':attempts-len(deltas)},'leave_one_out_auc':{n:macro(truth,np.delete(p,i,axis=0).mean(0)) for i,n in enumerate(names)} if len(p)>1 else {},'error_correlations_per_target':correlations,'weight_fitting':'none; uniform comparisons only; no locked tuning','limitations':'small weak dev and potentially unequal pretrained representations; incremental value against these members only'}
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
