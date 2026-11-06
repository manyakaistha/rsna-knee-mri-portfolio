# Landmark annotation application

FastAPI + SQLite + a canvas-based browser interface for four knee MRI landmarks. The system preserves source-pixel and patient-space coordinates, frozen cohort hashes, assignment state and annotation events. It supports shared review, separate readers, agreement exports and explicit adjudication.

From the repository root, after installing requirements and creating an authorized pilot manifest:

```bash
python run_web_annotator.py --competition-root data/raw
# Manual administration:
PYTHONPATH=kaggle/roi_cache python -m web_annotator.cli --help
```

The default host is `127.0.0.1`. The printed login link is private. Mutable databases, tokens, backups and exports stay in ignored `roi_work/`. Guidance includes draft clinical text that must be reviewed for actual annotation work.

`run_web_annotator.py` resolves the local application explicitly so installing the Kaggle API package does not shadow it. The public package does not include any real patient cohort. Data-dependent integration tests skip when its private fixtures are absent.

See the [full workflow](../../../docs/REPRODUCIBILITY.md#5-annotation-and-roi-caches) and [architecture](../../../docs/ARCHITECTURE.md).
