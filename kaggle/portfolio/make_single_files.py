"""Generate self-contained Kaggle training entrypoints for each recipe."""
import base64,json,zlib
from pathlib import Path
ROOT=Path(__file__).parent
SOURCES={name:(ROOT/name).read_text() for name in ('data.py','models.py','metrics.py','train.py')}
SPLIT=Path(ROOT/'staged/gpu_preflight/split.csv').read_text()
HEADER='''# AUTO-GENERATED SINGLE-FILE KAGGLE TRAINER.\n# Generated from kaggle/portfolio on 2026-10-14.\n# This file is a training entrypoint; it never submits to Kaggle.\n'''
TEMPLATE=r'''__HEADER__
import argparse,base64,json,os,runpy,sys,tempfile,zlib
from pathlib import Path
_PAYLOAD=__PAYLOAD__
_CONFIG=__CONFIG__
_SPLIT=__SPLIT__

def _decode_runtime():
    root=Path('/kaggle/working')/('portfolio_runtime_'+_CONFIG['name'])
    root.mkdir(parents=True,exist_ok=True)
    for name,blob in _PAYLOAD.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists() or path.read_text()!=zlib.decompress(base64.b64decode(blob)).decode():
            path.write_text(zlib.decompress(base64.b64decode(blob)).decode())
    split=root/'split.csv'
    if not split.exists():split.write_text(_SPLIT)
    sys.path.insert(0,str(root));return root,split

def _find(name, required=True):
    roots=[Path('/kaggle/input'),Path('/kaggle/working')]
    for root in roots:
        if root.exists():
            hits=list(root.rglob(name))
            if hits:return hits[0]
    if required:raise FileNotFoundError('Required Kaggle input not found: '+name)
    return None

def _cache():
    ids0=_find('ids_shard0.npy');cache=ids0.parent
    for name in ('ids_shard1.npy','masks_shard0.npy','masks_shard1.npy','vols_shard0.npy','vols_shard1.npy','meta_shard0.json','meta_shard1.json'):
        if not (cache/name).exists():raise FileNotFoundError(f'Cache input is incomplete: {cache/name}')
    if _CONFIG.get('require_window_geometry',True) and _CONFIG['kind']!='roi':
        for s in (0,1):
            if not (cache/f'centers_shard{s}.npy').exists():
                raise RuntimeError('This recipe requires source-SOP geometry masks centers_shard0/1.npy. Attach the completed geometry-audit dataset before training.')
    return cache

def main():
    ap=argparse.ArgumentParser(description=_CONFIG['name']+' self-contained Kaggle trainer')
    ap.add_argument('--output',default='/kaggle/working/'+_CONFIG['name'])
    ap.add_argument('--max-seconds',type=float,default=21600)
    ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--evaluate-locked',action='store_true')
    ap.add_argument('--split',default='')
    ap.add_argument('--labels',default='')
    ap.add_argument('--cache',default='')
    ap.add_argument('--roi',default='')
    ap.add_argument('--roi-manifest',default='')
    args=ap.parse_args()
    root,embedded_split=_decode_runtime()
    labels=Path(args.labels) if args.labels else _find('train_pool.csv')
    cache=Path(args.cache) if args.cache else _cache()
    split=Path(args.split) if args.split else embedded_split
    roi=Path(args.roi) if args.roi else (_find('roi_ids.npy').parent if _CONFIG['kind']=='roi' else None)
    roi_manifest=Path(args.roi_manifest) if args.roi_manifest else (_find('roi_manifest.json',False) or None)
    if _CONFIG['kind']=='roi' and roi_manifest is None:
        raise RuntimeError('ROI recipe requires an attached split-clean roi_manifest.json; full-pool roi_priors.json cannot certify this run.')
    config=root/'config.json';config.write_text(json.dumps(_CONFIG,indent=2))
    argv=[str(root/'train.py'),'--config',str(config),'--split',str(split),'--labels',str(labels),'--cache',str(cache),'--output',args.output,'--workers',str(args.workers),'--device',args.device,'--max-seconds',str(args.max_seconds)]
    if roi:argv += ['--roi',str(roi)]
    if roi_manifest:argv += ['--roi-manifest',str(roi_manifest)]
    if args.resume:argv += ['--resume']
    if args.evaluate_locked:argv += ['--evaluate-locked']
    old=sys.argv;sys.argv=argv
    try:runpy.run_path(str(root/'train.py'),run_name='__main__')
    finally:sys.argv=old
if __name__=='__main__':main()
'''
for config_path in sorted((ROOT/'configs').glob('*.json')):
    config=json.loads(config_path.read_text())
    payload={name:base64.b64encode(zlib.compress(source.encode(),9)).decode() for name,source in SOURCES.items()}
    script=TEMPLATE.replace('__HEADER__',HEADER).replace('__PAYLOAD__',repr(payload)).replace('__CONFIG__',repr(config)).replace('__SPLIT__',repr(SPLIT))
    out=ROOT/'single'/f'{config["name"]}.py';out.parent.mkdir(exist_ok=True);out.write_text(script)
    print(out,len(script))
