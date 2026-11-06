"""Attach historical public/source recipes without inferring private training details."""
import argparse,ast,csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PUBLIC={
'Arm2':{'backbone':'ResNet34','resolution':160,'slice_window_counts':'20 slices, five slots','aggregation':'attention MIL','loss':'report-derived multi-label BCE; source audit required for exact weighting','initialization':'scratch or knee-DINO according to original description','seed_fold':'unknown','checkpoint_selection':'unknown','training_cost':'unknown','inference_cost':'unknown'},
'Arm6':{'backbone':'ImageNet ResNet50','resolution':160,'slice_window_counts':'32 slices','aggregation':'attention MIL','loss':'report multi-label BCE; exact weighting unknown here','initialization':'ImageNet','seed_fold':'unknown','checkpoint_selection':'best checkpoint, Gold reused','training_cost':'unknown','inference_cost':'unknown'},
'Arm10':{'backbone':'convnext_small.in12k_ft_in1k_384','resolution':336,'slice_window_counts':'K12 fit/K24 validation/K42 native inference','inputs_caches':'Raptor44, span.15-.85, 140mm, uint8,44x336','aggregation':'finding-specific attention MIL','loss':'soft BCE positive weights from fit labels','initialization':'generic ImageNet12k->1k','seed_fold':'seed42/full4349pool, Gold excluded from gradients','checkpoint_selection':'Gold-selected top-three epoch weight average','training_cost':'unknown','inference_cost':'unknown'},
'Arm11':{'backbone':'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k','resolution':384,'slice_window_counts':'K12 fit/K24 validation/K42 native inference','inputs_caches':'Raptor44','aggregation':'finding-specific attention MIL','loss':'soft BCE positive weights','initialization':'generic pretrained','seed_fold':'seed42/fullpool','checkpoint_selection':'selected epoch weight average; exact epochs unknown here','training_cost':'unknown','inference_cost':'unknown'},
'Arm12':{'backbone':'convnext_small.in12k_ft_in1k_384','resolution':336,'slice_window_counts':'K12 fit/K24 validation/K62 legacy inference; windows can cross slots','inputs_caches':'cache64-maxspan18/14/12/8/12, span.02-.98, 140mm uint8','aggregation':'finding-specific attention MIL','loss':'soft BCE, fit-derived pos_weightclip1..10','initialization':'ImageNet12k->1k','seed_fold':'seed42/full4349pool','checkpoint_selection':'Gold-selected epochs9/6/8 weight average','training_cost':'unknown','inference_cost':'unknown'},
'Arm13':{'backbone':'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k','resolution':384,'slice_window_counts':'K12 fit/K24 validation/K62 legacy inference','inputs_caches':'cache64-maxspan','aggregation':'finding-specific attention MIL','loss':'soft BCE positive weights','initialization':'generic pretrained','seed_fold':'seed42/fullpool','checkpoint_selection':'best epoch8; final averaging OOM','training_cost':'unknown','inference_cost':'unknown'},
'Arm15':{'backbone':'convnext_small.in12k_ft_in1k_384','resolution':336,'slice_window_counts':'K12 fit/K24 validation/K62 legacy inference','inputs_caches':'cache64-maxspan','aggregation':'finding-specific attention MIL','loss':'.5 report BCE +.5 Arm14 cross-fitted prediction BCE','initialization':'generic pretrained','seed_fold':'seed42/fullpool','checkpoint_selection':'Gold-selected epochs6/7/9 weight average','training_cost':'unknown','inference_cost':'unknown'},
'Arm24':{'backbone':'unknown: private training source denied','resolution':'unknown until checkpoint/source provenance verified','slice_window_counts':'four triplet ROI windows per supported target at inference','inputs_caches':'five anatomical crop bags; pinned production ROI builder/priors','aggregation':'crop-only finding-specific heads; vote only MCL/lateralmeniscus/lateralOA/PFOA/synovitis','loss':'unknown','initialization':'unknown','seed_fold':'unknown; description claims870OOF validation','checkpoint_selection':'unknown; deployment loadsarm24_full_crop_swa.pt','training_cost':'unknown','inference_cost':'unknown'},
'Arm27':{'backbone':'parent arm20_f0_noaug_swa.pt; exact training recipe unknown','slice_window_counts':'K24 queries/reference features','inputs_caches':'3479 fit-study reference bank;12 finding embeddings','aggregation':'normalized cosine kNN32, softmaxsimilarityT.07; untrained retrieval index','loss':'unknown for encoder; no independently trained retrieval head','initialization':'unknown','seed_fold':'historicalfold0;870heldoutqueries','checkpoint_selection':'parentSWA filename; actual selection unknown','training_cost':'unknown','inference_cost':'unknown'},
'Arm C':{'backbone':'Arm13 CoAtNet plus five shared-encoder ROI residual heads','inputs_caches':'cache64+fiveROIbags','aggregation':'global findingMIL+bounded target-local residual','resolution':384,'slice_window_counts':'scoredK24, fourROIwindows/target','loss':'source trainingpolicy:maskedweightedBCE+globalaux; exact run flags require checkpoint audit','initialization':'Arm13 fullpool checkpoint','seed_fold':'Gold reused; new weakholdout not clean','checkpoint_selection':'selectedarmc_roi_selected_model.pt; checkpointsha in historicalreport','training_cost':'unknown for exact scored checkpoint','inference_cost':'unknown hidden-test runtime'}
}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=ROOT/'reports/submission_history_2026-10-13');a=ap.parse_args();p=a.directory;rows=json.loads((p/'submission_ledger.json').read_text())
 old=json.loads((ROOT/'reports/submission_history_2026-10-04/submission_ledger.json').read_text())['records'];old={str(r['ref']):r for r in old}
 sources={}
 for folder in (p/'raw/notebooks').iterdir():
  entries=[]
  for source in folder.glob('*.py'):
   constants={}
   for node in ast.walk(ast.parse(source.read_text())):
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in {'MEMBERS','RETR_MEMBERS','TTA_REVERSE','REWEIGHT','RETR_WEIGHT'}:
     try:constants[node.targets[0].id]=ast.literal_eval(node.value)
     except (ValueError,TypeError):pass
   entries.append({'path':str(source.relative_to(p)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_kind':'latest only; not certified historical scored source','literal_inference_constants':constants})
  sources[folder.name.replace('__','/')]=entries
 (p/'source_provenance_map.json').write_text(json.dumps(sources,indent=2))
 members={56067136:['Arm2'],56067156:['Arm2'],56087750:['Arm6'],56297411:['Arm10'],56304203:['Arm10','Arm12'],56304635:['Arm12'],56305155:['Arm11','Arm12'],56306970:['Arm11','Arm12'],56320388:['Arm10','Arm11','Arm12','Arm13'],56320411:['Arm13'],56370304:['Arm15'],56370345:['Arm10','Arm11','Arm12','Arm13','Arm15'],56403822:['Arm10','Arm11','Arm12','Arm13'],56473837:['Arm C'],56532006:['Arm10','Arm11','Arm12','Arm13','Arm24'],56535214:['Arm10','Arm11','Arm12','Arm13','Arm24','Arm27']}
 for row in rows:
  row['historical_system']=old.get(str(row['ref']),{}).get('system','unknown')
  row['latest_source_provenance']=sources.get(row['url'].split('?')[0].removeprefix('/code/'),[])
  names=members.get(row['ref'],[])
  if names:
   row['model_checkpoint_lineage']={'members':names,'authority':'historical source audit and public description; not certified exact private training execution'}
   for key in ['inputs_caches','backbone','aggregation','resolution','slice_window_counts','loss','initialization','seed_fold','checkpoint_selection','training_cost','inference_cost']:
    row[key]={name:PUBLIC[name].get(key,'unknown') for name in names}
   row['tta']='depth reversal' if 'TTA' in row['historical_system'] else 'no TTA' if row['ref'] not in [56473837,56067136,56067156,56087750] else 'unknown unless scored source audit establishes it'
   if len(names)>1:row['ensemble_members_weights']='per-finding equal rank votes over supported members; see original description; historical inference settings, not a fitted clean validation result'
 rows=json.dumps(rows,indent=2);(p/'submission_ledger.json').write_text(rows)
 values=json.loads(rows);keys=list(dict.fromkeys(k for r in values for k in r))
 with (p/'submission_ledger.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in values])
 (p/'historical_recipe_map.json').write_text(json.dumps(PUBLIC,indent=2))
if __name__=='__main__':main()
