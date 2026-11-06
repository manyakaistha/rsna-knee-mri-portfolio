"""Render publication figures from aggregate, non-patient evidence."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/assets'
OUT.mkdir(parents=True, exist_ok=True)
INK = '#122B3A'; TEAL = '#007F86'; BLUE = '#3765BC'; MUTED = '#5C7280'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titleweight':'bold','savefig.facecolor':'white'})


def overview():
    fig, ax = plt.subplots(figsize=(14,6.6))
    fig.patch.set_facecolor('#F4F8FA');ax.set_facecolor('#F4F8FA')
    ax.set(xlim=(0,14),ylim=(0,6.6));ax.axis('off')
    ax.text(.5,6.06,'RSNA  /  KNEE MRI',fontsize=11,color=TEAL,weight='bold')
    ax.text(.5,5.48,'From incomplete labels to anatomical evidence.',fontsize=24,color=INK,weight='bold')
    ax.text(.5,4.99,'A competition pipeline spanning data, models, annotation and evaluation.',fontsize=12,color=MUTED)
    def card(x,y,w,h,title,body,color=INK):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',facecolor='white',edgecolor='#D6E2E7',linewidth=1))
        ax.text(x+.22,y+h-.38,title,fontsize=12.5,color=color,weight='bold')
        ax.text(x+.22,y+h-.78,body,fontsize=10.3,color=MUTED,linespacing=1.6,va='top')
    card(.5,2.15,2.7,2.2,'01  ·  SUPERVISION','4,407 MRI studies\n58 fully labeled studies\n4,349 report-labeled studies')
    card(3.65,2.15,2.85,2.2,'02  ·  GEOMETRY','Physical slice ordering\nPlane / contrast windows\nMasks + cache parity')
    card(6.95,3.29,3.0,1.06,'03  ·  GLOBAL','Encoder + finding attention',TEAL)
    card(6.95,1.95,3.0,1.06,'04  ·  LOCAL','Landmarks + anatomical crops',TEAL)
    card(10.4,2.15,3.1,2.2,'05  ·  EVALUATION','12 independent findings\nOOF + paired uncertainty\nPublic / private results')
    for a,b in [((3.22,3.25),(3.62,3.25)),((6.52,3.45),(6.92,3.78)),((6.52,2.85),(6.92,2.48)),((9.97,3.78),(10.37,3.45)),((9.97,2.48),(10.37,2.85))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#92ABB8',linewidth=1.5))
    ax.plot([.5,13.5],[1.5,1.5],color='#D6E2E7',linewidth=1)
    for x,num,label in [(0.5,'0.937','best observed team public AUC'),(5.1,'0.920','best observed team private AUC'),(9.7,'30','completed team submissions')]:
        ax.text(x,.82,num,fontsize=27,color=INK,weight='bold')
        ax.text(x,.4,label,fontsize=10,color=MUTED)
    fig.savefig(OUT/'overview.png',dpi=180,bbox_inches='tight',pad_inches=.15);plt.close(fig)


def results():
    rows={int(r['ref']):r for r in json.loads((ROOT/'evidence/submissions.json').read_text())}
    points=[(56067136,'Scratch\nResNet34'),(56087750,'Pretrained\nResNet50'),(56297411,'ConvNeXt\n2.5D'),(56320388,'Four global\nmembers'),(56532006,'Global +\ncrop'),(56535214,'+ Retrieval'),(57450686,'Final pseudo-\nfold blend')]
    x=np.arange(len(points));pub=[float(rows[i]['publicScore']) for i,_ in points];priv=[float(rows[i]['privateScore']) for i,_ in points]
    fig,ax=plt.subplots(figsize=(13.5,5.7));fig.subplots_adjust(top=.76,bottom=.2,left=.065,right=.97)
    fig.text(.065,.94,'Progress across selected competition submissions',color=INK,fontsize=21,weight='bold')
    fig.text(.065,.88,'Official displayed scores · team Abhi · refreshed 6 November 2026',color=MUTED,fontsize=11)
    ax.plot(x,pub,'o-',color=TEAL,label='Public AUC',linewidth=2.4,markersize=7)
    ax.plot(x,priv,'s-',color=BLUE,label='Private AUC',linewidth=2.4,markersize=6)
    for i,(a,b) in enumerate(zip(pub,priv)):
        ax.annotate(f'{a:.3f}',(i,a),xytext=(0,10),textcoords='offset points',ha='center',color=TEAL,fontsize=10)
        ax.annotate(f'{b:.3f}',(i,b),xytext=(0,-18),textcoords='offset points',ha='center',color=BLUE,fontsize=10)
    ax.set_xticks(x,[s for _,s in points]);ax.set_ylim(.62,.98);ax.set_ylabel('Macro ROC AUC',color=MUTED)
    ax.spines[['top','right','bottom']].set_visible(False);ax.spines['left'].set_color('#D6E2E7');ax.grid(axis='y',color='#E4EBEF');ax.set_axisbelow(True)
    ax.legend(loc='lower right',frameon=False,ncols=2)
    fig.text(.065,.04,'Recipe, inputs and inference settings change across milestones; this is progression, not a controlled ablation.',fontsize=10,color=MUTED)
    fig.savefig(OUT/'results.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    overview();results();print('Wrote docs/assets/overview.png and results.png')
