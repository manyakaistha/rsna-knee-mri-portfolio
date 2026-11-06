"""Six trainable recipes; cached heads are screens, not independent final members."""
import torch
from torch import nn
import torch.nn.functional as F
LABELS=("ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture")
ROI_COLUMNS=(1,3,5,6,8)
ROI_NAMES=('mcl','lateral_meniscus','lateral_oa','pf_oa','synovitis')

def attention(scores, valid, dim):
    mask=valid
    while mask.ndim<scores.ndim: mask=mask.unsqueeze(-1)
    # Zero attention for empty inputs, never a softmax of all -inf.
    weights=scores.float().masked_fill(~mask,-1e9).softmax(dim)*mask
    return weights/weights.sum(dim,keepdim=True).clamp_min(1e-12)

class MIL(nn.Module):
    def __init__(self,dim,targets=12):
        super().__init__();self.norm=nn.LayerNorm(dim)
        self.att=nn.Sequential(nn.Linear(dim,256),nn.Tanh(),nn.Dropout(.2),nn.Linear(256,targets))
        self.readout=nn.Parameter(torch.empty(targets,dim));nn.init.normal_(self.readout,std=.02)
        self.bias=nn.Parameter(torch.zeros(targets))
    def pool(self,h,valid):
        h=self.norm(h);a=attention(self.att(h),valid,1).to(h.dtype)
        return torch.einsum('bkt,bkf->btf',a,h)
    def forward(self,h,valid,slots=None):
        pooled=self.pool(h,valid);return (pooled*self.readout).sum(-1)+self.bias,pooled

class Hierarchy(nn.Module):
    """Finding-conditioned slice -> contrast -> plane pooling, with explicit masks."""
    def __init__(self,dim):
        super().__init__();self.slice=MIL(dim);del self.slice.readout;del self.slice.bias;self.slot_embedding=nn.Parameter(torch.randn(5,dim)*.02)
        self.contrast_gate=nn.Linear(dim,1);self.plane_gate=nn.Linear(dim,1)
        self.readout=nn.Parameter(torch.randn(12,dim)*.02);self.bias=nn.Parameter(torch.zeros(12))
    def forward(self,h,valid,slots):
        slot_masks=[valid&(slots==s) for s in range(5)]
        present=torch.stack([m.any(1) for m in slot_masks],1)
        tokens=torch.stack([self.slice.pool(h,m)+self.slot_embedding[s] for s,m in enumerate(slot_masks)],1)
        planes=[];plane_masks=[]
        for ids in ((0,1),(2,3),(4,)):
            t=tokens[:,ids];m=present[:,ids,None].expand(-1,-1,12)
            a=attention(self.contrast_gate(t).squeeze(-1),m,1)
            planes.append((t*a[...,None]).sum(1));plane_masks.append(m.any(1))
        t=torch.stack(planes,1);m=torch.stack(plane_masks,1)
        a=attention(self.plane_gate(t).squeeze(-1),m,1)
        pooled=(t*a[...,None]).sum(1)
        return (pooled*self.readout).sum(-1)+self.bias,pooled

class ROIHead(nn.Module):
    def __init__(self,dim):
        super().__init__();self.heads=nn.ModuleList([MIL(dim,1) for _ in ROI_COLUMNS])
    def forward(self,h,valid,slots=None):
        z=[];p=[]
        for j,head in enumerate(self.heads):
            a,b=head(h[:,j],valid[:,j]);z.append(a);p.append(b)
        return torch.cat(z,1),torch.cat(p,1)

class Recipe(nn.Module):
    def __init__(self,config,feature_dim=None,tiny=False,pretrained=True):
        super().__init__();self.config=config
        if feature_dim is not None:self.encoder=None;dim=feature_dim
        elif tiny:
            self.encoder=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten());dim=8
        else:
            import timm
            kw={'img_size':config['resolution']} if config['encoder'].startswith('vit_') else {}
            self.encoder=timm.create_model(config['encoder'],pretrained=pretrained,num_classes=0,**kw);dim=self.encoder.num_features
            if config.get('gradient_checkpointing',False):
                from timm.layers import set_reentrant_ckpt
                set_reentrant_ckpt(False)
                if not hasattr(self.encoder,'set_grad_checkpointing'):raise ValueError('encoder does not support activation checkpointing')
                self.encoder.set_grad_checkpointing(True)
        self.head=ROIHead(dim) if config['kind']=='roi' else Hierarchy(dim) if config['kind']=='hierarchy' else MIL(dim)
        self.projection=nn.Linear(dim,64) if config['kind']=='metric' else None
    def train(self,mode=True):
        super().train(mode)
        if self.config.get('freeze_bn_stats',False) and self.encoder is not None:
            for module in self.encoder.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self
    def forward(self,x,valid,slots):
        if self.encoder is not None:
            shape=x.shape[:-3];flat=x.reshape(-1,*x.shape[-3:]);keep=valid.flatten()
            # Encode fixed shapes including masked windows: stable DDP graph, no conditional unused branches.
            if self.training:h=self.encoder(flat)
            else:h=torch.cat([self.encoder(v) for v in flat.split(self.config.get('eval_chunk',16))])
            h=h.reshape(*shape,-1);h=h*valid[...,None]
        else:h=x
        z,p=self.head(h,valid,slots)
        e=F.normalize(self.projection(p),dim=-1) if self.projection is not None else p
        return z,e

def metric_loss(e,y,mask):
    """Soft-label pairwise similarity within study batch; auxiliary to masked BCE."""
    similarity=torch.einsum('btf,ctf->bct',e,e)
    target=1-(y[:,None]-y[None,:]).abs()
    pair=mask[:,None]&mask[None,:]&~torch.eye(len(y),device=y.device,dtype=torch.bool)[...,None]
    return ((similarity-target).square()*pair).sum()/pair.sum().clamp_min(1)

def roi_fuse(global_logits,roi_logits,roi_valid,weight=.2):
    """Late probability fusion; missing ROI falls back exactly to global probabilities."""
    p=global_logits.sigmoid().clone();present=roi_valid.any(-1)
    for j,col in enumerate(ROI_COLUMNS):
        p[:,col]=torch.where(present[:,j],(1-weight)*p[:,col]+weight*roi_logits[:,j].sigmoid(),p[:,col])
    return p
