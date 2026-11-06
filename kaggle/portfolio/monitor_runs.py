"""Read-only status/log/small-output capture for bounded portfolio jobs."""
import datetime,json,urllib.request
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

def main():
    repo=Path(__file__).resolve().parents[2];out=repo/'reports/portfolio_2026-10-13';api=KaggleApi();api.authenticate();records=[]
    for slug in ['rsna-six-portfolio-gpu-preflight-20261013','rsna-six-portfolio-window-geometry-20261013','rsna-six-portfolio-scaling-20261014']:
        dest=out/'kaggle_outputs'/slug;dest.mkdir(parents=True,exist_ok=True)
        try:
            status=api.kernels_status('manyakaistha/'+slug);status=status.to_dict() if hasattr(status,'to_dict') else str(status)
            request=ApiListKernelSessionOutputRequest();request.user_name='manyakaistha';request.kernel_slug=slug
            with api.build_kaggle_client() as client:response=client.kernels.kernels_api_client.list_kernel_session_output(request)
            (dest/'latest.log').write_text(response.log or '')
            inventory=[]
            for f in response.files:
                inventory.append({'file_name':f.file_name})
                if f.file_name.endswith(('hardware.json','benchmarks.json','ddp_cuda.log','centers_shard0.npy','centers_shard1.npy','window_geometry_shard0.json','window_geometry_shard1.json','single.json','ddp.json','independent_0.json','independent_1.json','progress_shard0.json','progress_shard1.json','centers_partial_shard0.npy','centers_partial_shard1.npy','done_shard0.npy','done_shard1.npy')):
                    target=dest/Path(f.file_name).name
                    with urllib.request.urlopen(f.url,timeout=30) as stream:target.write_bytes(stream.read())
            (dest/'inventory.json').write_text(json.dumps(inventory,indent=2));records.append({'slug':slug,'status':status,'output_files':len(inventory)})
        except Exception as e:records.append({'slug':slug,'error':str(e)})
    record={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':records};(out/'kaggle_status.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
