# Six configurable research recipes

This package implements global ConvNeXt, global CoAtNet, finding-conditioned plane/contrast pooling, an independent ROI specialist, EfficientNet and metric-auxiliary ConvNeXt. These are implemented research recipes, not six proven final ensemble members.

`train.py` provides one-process-per-GPU DDP, observed-cell loss normalization, accumulation, AMP, complete-epoch checkpoints and strict resume. `data.py` enforces keyed inputs, patient splits, source-window legality and artifact ancestry. `infer.py` and `stage_submission.py` generate predictions/CSV locally; staging does not submit to Kaggle.

```bash
python kaggle/portfolio/smoke.py
torchrun --standalone --nproc_per_node=2 kaggle/portfolio/ddp_smoke.py
python kaggle/portfolio/train.py --help
```

Full input contracts and a GPU launch example are in [REPRODUCIBILITY.md](../../docs/REPRODUCIBILITY.md#6-configurable-research-runner). Legacy Arm13/Arm24 checkpoints have different schemas. Historical screen/package utilities need their original private split/caches; generated single-file bundles can be rebuilt after supplying those inputs.
