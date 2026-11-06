# Reproduction guide

## What can be reproduced now

| Level | Public checkout supports it? | Requirements |
|---|---|---|
| Source inspection and evidence charts | Yes | Python; matplotlib for charts |
| Geometry, ROI, annotation and model unit contracts | Yes | Core requirements; synthetic fixtures |
| CPU distributed gradient smoke | Yes | PyTorch Gloo and local process/network support |
| Historical model download | 26 assets cataloged | Network and approximately 45–596 MB per artifact |
| Real MRI preprocessing and training | Source available; inputs must be obtained | Authorized competition data, label sources, GPU and storage |
| Exact selected Arm C inference | Incomplete | Selected Arm C weights are missing from the confirmed release |
| Exact best final ensemble | Incomplete | Several final fold/crop checkpoints, training sources and retrieval DB are missing |

## 1. Environment and data-free checks

Python 3.11 is the CI target; source uses Python 3.10+ syntax. For GPU work, first install the torch/torchvision build appropriate to the CUDA runtime. Core dependency ranges are in `requirements.txt`; a successful local CPU environment is recorded in `docs/validation-environment.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/run_checks.py
```

The runner checks source/manifests/links, downloader integrity, DICOM geometry, browser-coordinate behavior, database behavior, residual-model invariants, analysis math, synthetic window headers, six tiny-encoder recipe contracts and CPU DDP. Integration tests needing the private annotation cohort intentionally skip. No benchmark score is reproduced by these tests.

For individual suites:

```bash
python -m unittest discover -v -s kaggle/roi_cache -p 'test_*.py'
python -m unittest discover -v -s kaggle/armc_roi -p 'test_*.py'
python kaggle/portfolio/smoke.py
```

Regenerate the report's figures with `python scripts/build_figures.py`.

## 2. Obtain data through Kaggle

Install optional tools with `python -m pip install -r requirements-data.txt`. Configure your own Kaggle credentials outside the repository and accept the competition terms. No credentials are supplied here.

The raw layout expected by preparation scripts is:

```text
data/raw/
  train.csv
  train_series.csv
  test.csv
  test_series.csv
  train_series/<study>/<series>/<instance>.dcm
  test_series/<study>/<series>/<instance>.dcm
```

For the full corpus, `python scripts/download_data.py --dest data/raw --unzip` uses the Kaggle API. The separate `download_competition.py` uses KaggleHub's resumable cache with `RSNA_DATA_DIR`; it has a different output layout. `extract_archive.py` can safely extract a downloaded archive from the chosen output directory. Budget storage for both the raw archive and extracted files; the documented DICOM corpus alone is approximately 530 GiB. Do not launch a full download merely to run the tests.

The EDA entry point is `python references/run_dataset_eda.py --help`. It exposes explicit input/output controls and regenerates dataset statistics and plots from authorized inputs. Historical counts in the report are preserved observations, not a fresh scan of the absent DICOM corpus.

## 3. Rebuild training labels

The preparation script reads all five candidate label files. Download the three source datasets into these exact local paths:

```bash
kaggle datasets download -d dreaddevelopment/rsna-knee-labels \
  -p data/external/dreaddevelopment_rsna-knee-labels --unzip
kaggle datasets download -d stevenleehans/rsna-knee-llm-report-labels \
  -p data/external/stevenleehans_rsna-knee-llm-report-labels --unzip
kaggle datasets download -d pilkwang/rsna-knee-llm-labels \
  -p data/external/pilkwang_rsna-knee-llm-labels --unzip
python scripts/prepare_dataset.py
```

With the original source versions, expected outputs are `data/processed/gold_test.csv` (58 studies), `train_pool.csv` (4,349 studies) and `label_source_validation.csv`. The original label selection used Gold58, so reproducing it does not create an independent holdout. Upstream datasets may change; record versions and hashes and do not silently treat changed labels as the original experiment.

The historical pool CSV retains report text for relabeling. Image trainers consume only IDs and numeric finding columns. Do not publish that CSV or use the report as an inference input.

## 4. Global cache and historical trainers

The historical cache64 builders are in `kaggle/cache64/`, `cache64_s0/` and `cache64_s1/`. Kaggle metadata files preserve the original input bindings and account slugs for provenance. **Copy metadata to your own staging folder and replace the account/slug before a push.** The public source does not grant access to private Kaggle inputs.

Global cache shards contain:

```text
ids_shard{0,1}.npy
vols_shard{0,1}.npy       # uint8 [N,64,336,336]
masks_shard{0,1}.npy
meta_shard{0,1}.json
```

Run a historical `train_arm*.py` only after attaching its exact corpus, validated labels, pretrained assets and any warm-start checkpoint. Most historical scripts discover files under `/kaggle/input` and put output under `/kaggle/working`; they are standalone Kaggle recipes, not interchangeable local CLIs. Preserve target order, corpus version, inference K, TTA and checkpoint selection policy.

List/download checkpoint assets using `scripts/download_models.py`. It verifies both size and SHA-256 and never overwrites a mismatching local file. It does not deserialize checkpoints. See [the model catalog](../models/README.md) for missing weights and failed SSL controls.

## 5. Annotation and ROI caches

Create a pilot manifest from authorized DICOMs:

```bash
python kaggle/roi_cache/build_roi_cache.py select-pilot \
  --competition-root data/raw --output-dir roi_work/pilot
python run_web_annotator.py --competition-root data/raw --no-open-browser
```

The server listens on localhost by default and prints the local authenticated link. It creates or resumes the frozen manifest and annotation state under ignored `roi_work/`. Its scientific guidance includes draft text; clinical review and data-sharing permissions remain prerequisites for an actual annotation study.

For manual CLI operations without clashing with the installed Kaggle package:

```bash
PYTHONPATH=kaggle/roi_cache python -m web_annotator.cli --help
```

After annotations are complete, fit priors using `build_roi_cache.py fit-priors`, then build each ROI shard using the corresponding global `ids_shard*.npy`. Consult each subcommand's `--help`. Final priors require the frozen fit/audit manifest and geometry/coverage gates. The full five-target ROI cache is roughly 27.8 GiB; use separate shards where output quotas require it.

Each target produces uint8 `[N,4,3,336,336]` bags and `[N,4]` masks. Global/ROI IDs must be exactly aligned. Validate with:

```bash
python kaggle/roi_cache/build_roi_cache.py validate \
  --cache-dir roi_work/cache --ids-dir roi_work/cache64
ARMC_ROI_INPUT=roi_work ROI_CHECK_STRICT=1 python kaggle/armc_roi/check_roi_cache.py
```

The standalone Arm C trainer needs the global cache, all five ROI targets, the two validated label CSVs and `arm13_best.pt`. Fresh residual heads preserve the Arm13 global logits. That is a structural test; recovering the historical selected residual weights requires an additional artifact not available here.

## 6. Configurable research runner

The newer portfolio requires a numeric label table and a patient partition CSV with unique `StudyInstanceUID,patient_sha256,split`, where split is `fit`, `dev` or `locked`. Keep all studies for a patient in one partition. Gold58 is excluded. Historical patient-index CSVs are not redistributed; construct and freeze your own authorized split, or recover the original private split for historical comparisons.

Verify source windows first:

```bash
python kaggle/portfolio/build_window_geometry.py \
  --competition-root data/raw --cache roi_work/cache64 --workers 4
```

This creates `centers_shard{0,1}.npy` and source-geometry records alongside the cache. A complete global training example, after preparing all inputs:

```bash
torchrun --standalone --nproc_per_node=2 kaggle/portfolio/train.py \
  --config kaggle/portfolio/configs/01_global_convnext.json \
  --split data/processed/patient_split.csv \
  --labels data/processed/train_pool.csv \
  --cache roi_work/cache64 --output runs/global_convnext \
  --max-seconds 10800
```

Use one process for one GPU. ROI recipes additionally require a cache and an accepted fit-only prior manifest via `--roi` and `--roi-manifest`. Feature-bank experiments require compatible ancestry manifests. A full-cohort production prior is not automatically eligible for a new holdout claim.

The runner fixes the final epoch, stores unwrapped model state and per-rank RNG, and refuses incompatible resume contracts. The sample three-hour budget is an execution cap, not a promise of training completion. Legacy checkpoints do not load into this runner merely because their filenames match.

Some historical packaging/analysis utilities expect private files under `roi_work/`, `kaggle/model_research/output/` or `reports/`. Their source is retained for completeness, but those paths must be restored or explicitly adapted. `make_single_files.py` requires a local staged split and can regenerate the six large standalone entrypoints; generated duplicates are excluded from the publication.

## 7. What this publication does not claim

No full GPU training, medical image inference or independent clinical evaluation was performed to prepare the public repository. Fresh CPU contracts verify the published source, not historical AUCs. The final best ensemble cannot currently be rebuilt exactly because required artifacts are missing. Aggregate historical metrics and exact scored-version URLs remain available for inspection without exposing patient data.
