# Historical model catalog

**26 checkpoints** are available in the [original team release](https://github.com/XERO47/rsna-knee-mri/releases/tag/model-weights), totaling approximately 4.97 GB. All 26 are also mirrored in [this repository’s v1.0.0 release](https://github.com/manyakaistha/rsna-knee-mri-portfolio/releases/tag/v1.0.0), after full-byte SHA-256 verification on 6 November 2026. [manifest.json](manifest.json) records exact byte sizes, URLs and SHA-256 digests.

Weights are GitHub release assets rather than embedded Git blobs. This keeps a clone small. Downloads use this repository’s mirror; the manifest also retains every original source URL. The downloader verifies each complete file before installing it and refuses to replace an existing mismatching file.

```bash
python scripts/download_models.py --list
python scripts/download_models.py --name arm13_best.pt
# Optional: downloads the full historical release, including failed SSL controls.
python scripts/download_models.py --all
```

## Available files

| Checkpoint | MB (decimal) | Role |
|---|---:|---|
| `arm13_best.pt` | 292.9 | historical checkpoint |
| `arm14_f0_attn_swa.pt` | 198.8 | historical checkpoint |
| `arm14_f0_seq_swa.pt` | 206.9 | historical checkpoint |
| `arm15_best.pt` | 198.8 | historical checkpoint |
| `arm15_last.pt` | 596.4 | historical checkpoint |
| `arm15_swa.pt` | 198.8 | historical checkpoint |
| `arm15_topk_ep6.pt` | 198.8 | historical checkpoint |
| `arm15_topk_ep7.pt` | 198.8 | historical checkpoint |
| `arm15_topk_ep9.pt` | 198.8 | historical checkpoint |
| `arm16_f0_base_swa.pt` | 198.8 | historical checkpoint |
| `arm16_f0_canon_swa.pt` | 198.8 | historical checkpoint |
| `arm18_f0_base_swa.pt` | 198.8 | historical checkpoint |
| `arm18_f0_mask_swa.pt` | 198.8 | historical checkpoint |
| `arm19_f0_aug_swa.pt` | 198.8 | historical checkpoint |
| `arm19_f0_noaug_swa.pt` | 198.8 | historical checkpoint |
| `arm1_best.pt` | 45.1 | historical checkpoint |
| `arm1_final.pt` | 45.1 | historical checkpoint |
| `arm20_f0_aug_swa.pt` | 198.8 | historical checkpoint |
| `arm20_f0_noaug_swa.pt` | 198.8 | historical checkpoint |
| `arm2_scratch_best.pt` | 85.6 | historical checkpoint |
| `arm2_ssl_best.pt` | 85.6 | historical checkpoint |
| `arm6_best.pt` | 96.6 | historical checkpoint |
| `knee_dino_checkpoint_v3_collapsed.pt` | 182.1 | negative-control checkpoint |
| `knee_dino_checkpoint_v5_easy_evalbn_collapsed.pt` | 182.1 | negative-control checkpoint |
| `knee_dino_checkpoint_v6_easy_trainbn_stable.pt` | 182.1 | historical checkpoint |
| `knee_dino_checkpoint_v7_curriculum.pt` | 182.1 | historical checkpoint |

## Mapping and limits

An `arm<N>` checkpoint maps to the corresponding `kaggle/arm<N>/train_arm<N>.py` recipe. Fold/head suffixes distinguish attention, sequence, augmentation and canonicalization variants. Verify checkpoint metadata and target order rather than loading by filename alone.

`arm13_best.pt` is the recovered best single epoch (SHA-256 `27e6f5c9481a03fbdcdb8714953252414b512c7bfb2809d3086b82e9225b4024`). Arm13's averaging step failed; there is no separate Arm13 SWA file in this release. This is the historical Arm C warm start.

The release does **not** include the selected `armc_roi_selected_model.pt`, Arm10/11/12 final weights, Arm24 full crop weights, the late pseudo/EVF fold weights or the retrieval database required by the final blend. The selected Arm C historical digest is `baef983de972fd60abbec906a7cb36793df2b20f3a4dbe7ad9ac0d5c8099e067`; metadata is not a substitute for those missing tensors.

Collapsed knee-DINO checkpoints are experimental negative controls, not recommended pretrained models. A checkpoint's presence does not confer the final ensemble's leaderboard score. Original source/model terms apply; see [NOTICE](../NOTICE.md).
