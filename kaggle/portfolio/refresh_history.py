"""Read-only paginated team history; never submit or mutate Kaggle artifacts."""
import argparse, csv, datetime as dt, hashlib, json, subprocess
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); raw=a.output/'raw'; raw.mkdir(exist_ok=True)
    api=KaggleApi(); api.authenticate(); rows={}; pages=[]
    token=None
    for page in range(1000):
        request=ApiListSubmissionsRequest(); request.competition_name='rsna-knee-abnormality-detection'; request.page_size=5
        if token: request.page_token=token
        with api.build_kaggle_client() as client:
            response=client.competitions.competition_api_client.list_submissions(request)
        (raw/f'response_token_{page:03d}.json').write_text(response.to_json())
        records=[dict(x.to_dict(),status=x.status.name) for x in response.submissions]
        token=response.to_dict().get('nextPageToken')
        (raw/f'page_{page:03d}.json').write_text(json.dumps(records,indent=2,default=str))
        new=[r for r in records if str(r['ref']) not in rows]
        pages.append({'page':page,'rows':len(records),'new':len(new)})
        if not records: break
        if not new: raise RuntimeError('Pagination repeated a page; coverage cannot be certified')
        rows.update({str(r['ref']):r for r in new})
        if not token: break
    else: raise RuntimeError('Pagination safety limit')
    old=Path(__file__).resolve().parents[2]/'reports/submission_history_2026-10-04/submission_ledger.json'
    oldrows=json.loads(old.read_text()) if old.exists() else []; oldrows=oldrows if isinstance(oldrows,list) else oldrows.get('records',[])
    (raw/'historical_ledger_schema.json').write_text(json.dumps(list(oldrows[0]) if oldrows else []))
    lineage_fields=['model_checkpoint_lineage','inputs_caches','backbone','aggregation','resolution','slice_window_counts','loss','initialization','seed_fold','checkpoint_selection','tta','ensemble_members_weights','training_cost','inference_cost']
    curated=[]; retrieval=[]
    for r in rows.values():
        item=dict(r); item['scored_notebook_version_url']='https://www.kaggle.com'+r['url'] if r.get('url','').startswith('/') else r.get('url','unknown')
        item.update({k:'unknown' for k in lineage_fields}); item['evidence_type']='official API metadata; description is an unverified recipe claim'
        curated.append(item)
    (a.output/'submission_ledger.json').write_text(json.dumps(curated,indent=2,default=str))
    keys=list(dict.fromkeys(k for r in curated for k in r))
    with (a.output/'submission_ledger.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(curated)
    for slug in sorted({r.get('url','').split('?')[0].removeprefix('/code/') for r in rows.values() if r.get('url')}):
        dest=raw/'notebooks'/slug.replace('/','__'); dest.mkdir(parents=True,exist_ok=True)
        proc=subprocess.run(['kaggle','kernels','pull',slug,'-m','-p',str(dest)],capture_output=True,text=True)
        log=proc.stdout+proc.stderr; (dest/'retrieval.txt').write_text(log)
        try:
            request=ApiListKernelSessionOutputRequest();request.user_name,request.kernel_slug=slug.split('/')
            with api.build_kaggle_client() as client:
                outputs=client.kernels.kernels_api_client.list_kernel_session_output(request)
            (dest/'latest.log').write_text(outputs.log or '')
            (dest/'output_inventory.json').write_text(json.dumps([{'file_name':x.file_name} for x in outputs.files],indent=2))
        except Exception as error: (dest/'log_access_error.txt').write_text(str(error))
        retrieval.append({'slug':slug,'returncode':proc.returncode,'source_version':'latest only; not certified historical scored source'})
    (raw/'source_retrieval_log.json').write_text(json.dumps(retrieval,indent=2))
    summary={'retrieved_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'scope':'authenticated accessible team submissions only','pages':pages,'unique_attempts':len(rows),'status_counts':{s:sum(r.get('status')==s for r in rows.values()) for s in {r.get('status') for r in rows.values()}},'pagination':'structured page_size=5 and nextPageToken; terminated without continuation token','private_scores_available':sum(r.get('privateScore') is not None for r in rows.values())}
    (a.output/'refresh.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
    manifest={str(p.relative_to(a.output)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in a.output.rglob('*') if p.is_file()}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
