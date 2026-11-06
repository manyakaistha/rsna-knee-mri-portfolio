# DICOM geometry, anatomical crops and annotation

This pipeline generates deterministic, label-free ROI bags from physical DICOM geometry and landmark priors. It keeps masks, exact study alignment and per-window provenance.

MCL, lateral meniscus and lateral OA use paired side hypotheses because decisive laterality tags cover only 49.97% of the documented corpus. PF OA and synovitis use their own overlapping/landmark windows. Each target retains four triplets.

Start with the [reproduction guide](../../docs/REPRODUCIBILITY.md#5-annotation-and-roi-caches) and [architecture](../../docs/ARCHITECTURE.md).

- `roi_geometry.py`: coordinates, physical ordering, plane selection and crop definitions.
- `build_roi_cache.py`: pilot/cohort selection, prior fitting, cache construction and validation.
- `web_annotator/`: FastAPI/SQLite annotation system.
- `annotate_roi_pilot.py`, `run_local_pilot.py`: desktop diagnostic workflow.
- `prepare_roi_revision.py`, `migrate_roi_revision.py`: explicit revision workflows.
- `test_*.py`: synthetic geometry and cache regression tests.

```bash
python -m unittest discover -v -s kaggle/roi_cache -p 'test_*.py'
python kaggle/roi_cache/build_roi_cache.py --help
```

No annotations, authentication tokens, patient manifests or image caches ship with the public repository.
