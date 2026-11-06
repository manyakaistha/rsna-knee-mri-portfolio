"""Parallel/resumable header-only source-triplet audit; never changes cache pixels."""
import argparse,concurrent.futures,glob,hashlib,importlib.util,json,os,time
from pathlib import Path
import numpy as np,pandas as pd,pydicom
TAGS=['ImageOrientationPatient','ImagePositionPatient','InstanceNumber','PixelSpacing','SOPInstanceUID','SeriesInstanceUID']

def frames_for_series(directory,k,span):
    records=[]
    # Same glob iteration and stable position sort as the verified cache builder.
    for name in glob.glob(str(directory/'*.dcm')):
        try:
            h=pydicom.dcmread(name,stop_before_pixels=True,specific_tags=TAGS)
            normal=np.cross(np.array(h.ImageOrientationPatient[:3],float),np.array(h.ImageOrientationPatient[3:],float)) if hasattr(h,'ImageOrientationPatient') else None
            position=float(np.dot(np.array(h.ImagePositionPatient,float),normal)) if normal is not None and hasattr(h,'ImagePositionPatient') else float(getattr(h,'InstanceNumber',0) or 0)
            frame={'sop':str(h.SOPInstanceUID),'series':str(h.SeriesInstanceUID),'position_mm':position,'normal':normal.tolist()} if normal is not None and hasattr(h,'ImagePositionPatient') else None
        except Exception:position=0.;frame=None
        records.append((position,frame))
    records.sort(key=lambda r:r[0]);n=len(records)
    if not n:return [None]*k
    lo=int(n*span[0]);hi=max(lo,int(n*span[1])-1);picks=np.linspace(lo,hi,k).round().astype(int) if n>1 else np.zeros(k,int)
    return [records[min(int(p),n-1)][1] for p in picks]

def study_job(args):
    i,uid,rows,mask,root,slots,span=args;used=set();offset=0;centers=np.zeros(64,bool);frames_all=[]
    try:
        for plane,fluid,k in slots:
            candidates=[r for r in rows if r['Anatomical_Plane']==plane and r['SeriesInstanceUID'] not in used]
            preferred=[r for r in candidates if int(r.get('Fluid_Sensitive',0) or 0)==fluid] if fluid in (0,1) else []
            row=(preferred or candidates or [None])[0];frames=[None]*k
            if row:
                used.add(row['SeriesInstanceUID']);frames=frames_for_series(root/uid/row['SeriesInstanceUID'],k,span)
            for j in range(1,k-1):
                triplet=frames[j-1:j+2]
                if mask[offset+j-1:offset+j+2].all() and all(f is not None for f in triplet):
                    centers[offset+j]=(len({f['sop'] for f in triplet})==3 and len({f['series'] for f in triplet})==1 and np.diff([f['position_mm'] for f in triplet]).min()>.01 and all(np.allclose(triplet[0]['normal'],f['normal'],atol=1e-4) for f in triplet))
            frames_all.extend(frames);offset+=k
        return i,uid,centers,frames_all,None
    except Exception as error:return i,uid,centers,frames_all,str(error)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--competition-root',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--split',choices=['train','test'],default='train');ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
    builder=Path(__file__).resolve().parents[1]/'cache64/build_cache64.py';spec=importlib.util.spec_from_file_location('cache64_builder',builder);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    series=pd.read_csv(a.competition_root/f'{a.split}_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str});by={u:g.to_dict('records') for u,g in series.groupby('StudyInstanceUID',sort=False)};started=time.perf_counter()
    for shard in range(2):
        ids=np.load(a.cache/f'ids_shard{shard}.npy').astype(str);masks=np.load(a.cache/f'masks_shard{shard}.npy').astype(bool);dest=a.cache/f'centers_shard{shard}.npy'
        if dest.exists():continue
        partial=a.cache/f'centers_partial_shard{shard}.npy';donefile=a.cache/f'done_shard{shard}.npy'
        centers=np.load(partial,mmap_mode='r+') if partial.exists() else np.lib.format.open_memmap(partial,mode='w+',dtype=bool,shape=masks.shape)
        done=np.load(donefile) if donefile.exists() else np.zeros(len(ids),bool)
        if centers.shape!=masks.shape or done.shape!=(len(ids),):raise ValueError('resume shape mismatch')
        errors=[];jobs=[(i,uid,by.get(uid,[]),masks[i],a.competition_root/f'{a.split}_series',module.SLOTS64,module.SPAN) for i,uid in enumerate(ids) if not done[i]]
        with (a.cache/f'window_sources_shard{shard}.jsonl').open('a') as log,concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            for count,(i,uid,valid,frames,error) in enumerate(pool.map(study_job,jobs),1):
                if error:errors.append({'uid':uid,'error':error})
                else:centers[i]=valid;done[i]=True
                log.write(json.dumps({'uid':uid,'frames':frames,'error':error})+'\n')
                if count%10==0 or count==len(jobs):
                    centers.flush();np.save(donefile,done);log.flush();elapsed=time.perf_counter()-started
                    progress={'shard':shard,'completed_studies':int(done.sum()),'total_studies':len(ids),'valid_centers':int(centers[done].sum()),'elapsed_seconds':elapsed,'workers':a.workers,'errors':errors};(a.cache/f'progress_shard{shard}.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
        if errors or not done.all():raise RuntimeError('source audit incomplete; refusing publish final center mask')
        centers.flush();del centers;os.replace(partial,dest);final=np.load(dest)
        (a.cache/f'window_geometry_shard{shard}.json').write_text(json.dumps({'builder_sha256':hashlib.sha256(builder.read_bytes()).hexdigest(),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source':'fresh DICOM header audit; existing cache masks','valid_centers':int(final.sum()),'studies_without_centers':int((final.sum(1)==0).sum()),'elapsed_seconds':time.perf_counter()-started,'workers':a.workers,'contract':'within selected series; distinct SOPs; strictly increasing physical position; not necessarily adjacent acquisition slices'},indent=2))
if __name__=='__main__':main()
