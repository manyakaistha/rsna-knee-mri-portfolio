"""Strict keyed datasets and split/ancestry contracts; reports are never model inputs."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from models import LABELS,ROI_NAMES
CAP=(18,14,12,8,12);OFF=np.cumsum((0,)+CAP)

def legal_windows(mask,k,rng=None,center_mask=None):
    legal=[np.array([c for c in range(OFF[s]+1,OFF[s+1]-1) if mask[c-1:c+2].all() and (center_mask is None or center_mask[c])],dtype=int) for s in range(5)]
    available=[s for s,a in enumerate(legal) if len(a)]
    if not available:raise ValueError('all-invalid study: explicit global fallback required upstream')
    if k<len(available):raise ValueError('K cannot cover present slots')
    q=np.zeros(5,int);q[available]=1
    for _ in range(k-len(available)):
        candidates=[s for s in available if q[s]<len(legal[s])]
        if not candidates:break
        s=max(candidates,key=lambda s:(len(legal[s])/max(q[s],1),-s));q[s]+=1
    centers=[];slots=[]
    for s in available:
        a=legal[s];count=q[s]
        picks=rng.choice(a,count,replace=False) if rng is not None else a[np.linspace(0,len(a)-1,count).round().astype(int)]
        centers.extend(picks.tolist());slots.extend([s]*count)
    n=len(centers);return centers+[-1]*(k-n),slots+[-1]*(k-n)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def partitions(path):
    frame=pd.read_csv(path,dtype=str)
    required={'StudyInstanceUID','patient_sha256','split'}
    if not required<=set(frame):raise ValueError('split must include verified patient hashes')
    if frame.StudyInstanceUID.duplicated().any():raise ValueError('duplicate study IDs')
    if not set(frame.split)<= {'fit','dev','locked'} or set(frame.split)!= {'fit','dev','locked'}:raise ValueError('explicit fit/dev/locked required')
    if frame.groupby('patient_sha256').split.nunique().max()>1:raise ValueError('patient leakage')
    return {s:frame.loc[frame.split==s,'StudyInstanceUID'].tolist() for s in ('fit','dev','locked')}

class Studies(Dataset):
    def __init__(self,ids,labels,config,cache,roi=None,train=False,features=None):
        self.ids=ids;self.config=config;self.train=train;self.epoch=0
        self.cache_path=Path(cache);self.roi_path=Path(roi) if roi else None;self.feature_path=Path(features) if features else None
        table=pd.read_csv(labels,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
        if table.index.duplicated().any():raise ValueError('duplicate label IDs')
        self.y=table.loc[ids,list(LABELS)].to_numpy(dtype=np.float32)
        if ((self.y[np.isfinite(self.y)]<0)|(self.y[np.isfinite(self.y)]>1)).any():raise ValueError('labels outside [0,1]')
        self.features=None
        if features:
            self.features=np.load(features,allow_pickle=False);cacheids=self.features['ids'].astype(str)
            self.feature_dim=self.features['features'].shape[-1]
        else:
            cache=Path(cache);self.vol=[np.load(cache/f'vols_shard{s}.npy',mmap_mode='r') for s in range(2)]
            cacheids=np.concatenate([np.load(cache/f'ids_shard{s}.npy') for s in range(2)]).astype(str)
            self.masks=np.concatenate([np.load(cache/f'masks_shard{s}.npy') for s in range(2)]);self.n0=len(self.vol[0])
            present=all((cache/f'centers_shard{s}.npy').exists() for s in range(2))
            self.center_masks=np.concatenate([np.load(cache/f'centers_shard{s}.npy') for s in range(2)]) if present else None
            if config.get('require_window_geometry',True) and self.center_masks is None and config['kind']!='roi':raise ValueError('fresh source-SOP window geometry audit required; cached slice masks alone are insufficient')
            if config['kind']=='roi':
                if roi is None:raise ValueError('ROI directory required')
                roi=Path(roi);rid=np.load(roi/'roi_ids.npy').astype(str)
                if not np.array_equal(rid,cacheids):raise ValueError('ROI/global ID mismatch')
                self.roi=[np.load(roi/f'roi_{t}.npy',mmap_mode='r') for t in ROI_NAMES]
                self.rm=[np.load(roi/f'roi_{t}_masks.npy',mmap_mode='r') for t in ROI_NAMES]
        if len(set(cacheids))!=len(cacheids):raise ValueError('duplicate cache IDs')
        self.rows={u:i for i,u in enumerate(cacheids)}
        if not set(ids)<=set(self.rows):raise ValueError('missing studies')
    def __getstate__(self):
        state=self.__dict__.copy()
        for key in ('vol','roi','features'): state.pop(key,None)
        return state
    def __setstate__(self,state):
        self.__dict__.update(state);self.features=None
        if self.feature_path:self.features=np.load(self.feature_path,allow_pickle=False)
        else:
            self.vol=[np.load(self.cache_path/f'vols_shard{s}.npy',mmap_mode='r') for s in range(2)]
            if self.config['kind']=='roi':self.roi=[np.load(self.roi_path/f'roi_{t}.npy',mmap_mode='r') for t in ROI_NAMES]
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        uid=self.ids[i];row=self.rows[uid];c=self.config
        seed=int.from_bytes(hashlib.sha256(f'{c["seed"]}:{self.epoch}:{uid}'.encode()).digest()[:8],'little');rng=np.random.default_rng(seed)
        if self.features is not None:
            x=self.features['features'][row].astype(np.float32);valid=self.features['valid'][row].astype(bool);slots=self.features['slots'][row].astype(np.int64)
        elif c['kind']=='roi':
            k=c['k_train'] if self.train else c['k_eval']
            indices=rng.choice(4,k,replace=False) if self.train else np.arange(k)
            x=np.stack([a[row,indices] for a in self.roi]);valid=np.stack([a[row,indices] for a in self.rm]).astype(bool);slots=np.zeros(valid.shape,dtype=np.int64)
        else:
            v=self.vol[0][row] if row<self.n0 else self.vol[1][row-self.n0]
            centers,slots=legal_windows(self.masks[row],c['k_train'] if self.train else c['k_eval'],rng if self.train else None,self.center_masks[row] if self.center_masks is not None else None)
            valid=np.array(centers)>=0;x=np.zeros((len(centers),3,336,336),np.uint8)
            for j,ctr in enumerate(centers):
                if ctr>=0:x[j]=v[ctr-1:ctr+2]
            slots=np.array(slots)
        y=self.y[i,list(c['targets'])];observed=np.isfinite(y)
        if c['kind']=='roi':observed=observed&valid.any(-1)
        return {'x':torch.from_numpy(x.copy()),'valid':torch.from_numpy(valid.copy()),'slots':torch.tensor(slots),'y':torch.tensor(np.nan_to_num(y)),'observed':torch.tensor(observed),'uid':uid}

def prepare(batch,device,config,features=False):
    x=batch['x'].to(device,non_blocking=True).float()
    if not features:
        shape=x.shape;flat=x.reshape(-1,*shape[-3:])/255
        flat=torch.nn.functional.interpolate(flat,size=(config['resolution'],)*2,mode='bilinear',align_corners=False)
        mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        x=((flat-mean)/std).reshape(*shape[:-2],config['resolution'],config['resolution'])
    return x,*[batch[k].to(device) for k in ('valid','slots','y','observed')]
