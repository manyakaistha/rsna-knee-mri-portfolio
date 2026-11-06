"""Package private, bounded research notebooks; pushing is a separate explicit action."""
import argparse,base64,hashlib,json,zlib
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['gpu-preflight','window-geometry'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=Path(__file__).resolve().parents[2];folder=Path(__file__).parent
    split=pd.read_csv(repo/'kaggle/model_research/output/input_audit/weak_split.csv',dtype=str);screen=json.loads((repo/'reports/portfolio_2026-10-13/representation_screen/contract.json').read_text());fit_screen=set(screen['fit_ids']);dev_screen=set(screen['dev_ids'])
    def assign(row):
        if row.split=='weak_locked':return 'locked'
        if row.StudyInstanceUID in fit_screen:return 'fit'
        if row.StudyInstanceUID in dev_screen:return 'dev'
        unit=int(hashlib.sha256(('portfolio-development-20261013|'+row.patient_sha256).encode()).hexdigest()[:16],16)/16**16
        return 'dev' if unit<.15 else 'fit'
    split['split']=[assign(r) for r in split.itertuples(index=False)]
    bundle={p.name:p.read_text() for p in folder.glob('*.py') if p.name!='package_kaggle.py'}
    bundle.update({'configs/'+p.name:p.read_text() for p in (folder/'configs').glob('*.json')})
    bundle['split.csv']=split.to_csv(index=False);pool=pd.read_csv(repo/'data/processed/train_pool.csv',dtype={'StudyInstanceUID':str});bundle['labels.csv']=pool.drop(columns=['Report','label_source'],errors='ignore').to_csv(index=False)
    bundle['build_cache64.py']=(repo/'kaggle/cache64/build_cache64.py').read_text()
    blob=base64.b64encode(zlib.compress(json.dumps(bundle).encode())).decode();a.output.mkdir(parents=True,exist_ok=True)
    driver=f'''import base64,json,zlib,sys,subprocess,shutil,time
from pathlib import Path
root=Path('/kaggle/working/portfolio');root.mkdir(exist_ok=True)
files=json.loads(zlib.decompress(base64.b64decode({blob!r})))
for name,source in files.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)
cache=next(Path('/kaggle/input').rglob('vols_shard0.npy')).parent
mode={a.mode!r}
if mode=='gpu-preflight':
 subprocess.run([sys.executable,'-m','pip','install','--quiet','timm==1.0.30'],check=True)
 subprocess.run([sys.executable,str(root/'gpu_preflight.py'),'--cache',str(cache),'--labels',str(root/'labels.csv'),'--split',str(root/'split.csv'),'--output','/kaggle/working/preflight'],check=True)
else:
 work=Path('/kaggle/working/window_geometry');work.mkdir(exist_ok=True)
 for s in range(2):
  for stem in ('ids','masks','meta'):
   name=f'{{stem}}_shard{{s}}'+('.json' if stem=='meta' else '.npy');shutil.copyfile(cache/name,work/name)
 comp=next(p.parent for p in Path('/kaggle/input').rglob('train_series.csv') if (p.parent/'train_series').exists())
 builder=root.parent/'cache64';builder.mkdir(exist_ok=True);shutil.copyfile(root/'build_cache64.py',builder/'build_cache64.py')
 subprocess.run([sys.executable,str(root/'build_window_geometry.py'),'--competition-root',str(comp),'--cache',str(work)],check=True)
'''
    (a.output/'run.py').write_text(driver)
    slug='rsna-six-portfolio-'+a.mode+'-20261013';meta={'id':'manyakaistha/'+slug,'title':slug.replace('-',' '),'code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':a.mode=='gpu-preflight','enable_internet':a.mode=='gpu-preflight','dataset_sources':['manyakaistha/knee-roi-data'],'competition_sources':['rsna-knee-abnormality-detection'],'kernel_sources':[]}
    (a.output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2));(a.output/'split.csv').write_text(bundle['split.csv']);print(json.dumps({'mode':a.mode,'split_counts':split.split.value_counts().to_dict(),'script_bytes':len(driver),'script_sha256':hashlib.sha256(driver.encode()).hexdigest(),'private':True}))
if __name__=='__main__':main()
