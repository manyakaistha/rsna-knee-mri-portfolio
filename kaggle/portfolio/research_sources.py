"""Archive public primary-source pages without browser dependencies."""
import concurrent.futures,datetime,json,urllib.request
from pathlib import Path
SOURCES={
'convnext':'https://arxiv.org/abs/2201.03545',
'coatnet':'https://arxiv.org/abs/2106.04803',
'efficientnet':'https://proceedings.mlr.press/v97/tan19a.html',
'mrnet':'https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699',
'mil':'https://proceedings.mlr.press/v80/ilse18a.html',
'copas':'https://www.nature.com/articles/s41467-024-51888-4',
'dinov2':'https://arxiv.org/abs/2304.07193',
'supcon':'https://arxiv.org/abs/2004.11362',
'dinov2_implementation':'https://raw.githubusercontent.com/facebookresearch/dinov2/main/README.md',
'convnext_implementation':'https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/README.md',
'mst':'https://www.nature.com/articles/s41598-025-09041-8',
}
def fetch(item):
 name,url=item; out=Path('rsna-knee-mri/reports/portfolio_2026-10-13/raw/literature');out.mkdir(parents=True,exist_ok=True)
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=35) as r: data=r.read();status=r.status
  (out/(name+'.txt')).write_bytes(data)
  return {'name':name,'url':url,'accessed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'bytes':len(data)}
 except Exception as e:return {'name':name,'url':url,'accessed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':str(e)}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:r=list(pool.map(fetch,SOURCES.items()))
 Path('rsna-knee-mri/reports/portfolio_2026-10-13/raw/research_access.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
