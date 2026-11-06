# Experiment catalog

Source availability and run completion are distinct. The report records measured outcomes; this catalog preserves the breadth of the available implementation.

## Development chronology

| Phase | Work | Outcome / boundary |
|---|---|---|
| Initial baseline | Arm1 scratch ResNet18, limited subset | Gold ≈0.686; no submission in the returned ledger |
| Representation study | Frozen DINOv2 and knee-DINO v3–v7 | Protocol bias, collapse controls and weak pathology transfer |
| Scratch scale-up | Arm2/3 ResNet34 and Arm4 transformer | Scratch ceiling; more epochs did not resolve it |
| Transfer learning | Arm5 medical pretraining; Arm6 generic pretraining | Stronger local/global performance; cancelled Arm5 output not retained |
| Context tests | Arm7 self-attention, Arm8 depth restriction, Arm9 slot/band changes | No convincing improvement over Arm6; several confounds/incomplete runs |
| Strong 2.5D models | Arm10–13, ConvNeXt and CoAtNet, two corpora | Public 0.920–0.923 singles; architecture/corpus diversity |
| OOF and label refinement | Arm14/14b/14c and Arm15 | Attention selected; five-fold OOF and report/OOF label blend |
| Paired ablations | Arm16 canonicalization; Arm17 fold CoAtNet; Arm18 masking; Arm19/20 augmentation | Several rejected gates; Arm17 final outcome not established |
| Anatomical residual | Arm C, annotation system, ROI cache and controls | Small uncertain local delta; public/private 0.920/0.898 |
| Independent local member | Arm21/24 crop family | September crop blend 0.932/0.915; training-source gap |
| Retrieval | Arm27 finding-specific kNN | Local gains reported; no increment established on hidden competition scores |
| Volumetric model | Multi-plane 3D submission | 0.871 public / 0.863 private; recovered inference only |
| Late folds/refinement | MR, EVF, five-fold and pseudo-fold blends | Best observed public/private 0.937/0.920; incomplete training artifacts |
| New research runner | Six recipes with clean input/split/resume contracts | Smoke/cost/screen evidence; six full trainings not established |

## Historical training source

| Script | Original recipe description |
|---|---|
| [`kaggle/arm1/train_arm1.py`](../kaggle/arm1/train_arm1.py) | RSNA Knee Abnormality Detection -- arm1, a first trained baseline. (v2) |
| [`kaggle/arm10/train_arm10.py`](../kaggle/arm10/train_arm10.py) | arm10: arm6's per-finding attention head on Raptor's inputs, trained by us. |
| [`kaggle/arm11/train_arm11.py`](../kaggle/arm11/train_arm11.py) | arm11: arm10's exact pipeline with CoAtNet-RMLP-2 at 384 -- the second ensemble member. |
| [`kaggle/arm12/train_arm12.py`](../kaggle/arm12/train_arm12.py) | arm12: arm10's exact recipe on OUR wide-span corpus (cache64) -- span is the only change. |
| [`kaggle/arm13/train_arm13.py`](../kaggle/arm13/train_arm13.py) | arm13: CoAtNet-RMLP-2 @384 on our wide-span cache64 -- arm11's backbone on arm12's corpus. |
| [`kaggle/arm14/train_arm14.py`](../kaggle/arm14/train_arm14.py) | arm14: 5-fold ConvNeXt-small on cache64 with out-of-fold (OOF) predictions, and an A/B test |
| [`kaggle/arm14b/train_arm14.py`](../kaggle/arm14b/train_arm14.py) | arm14: 5-fold ConvNeXt-small on cache64 with out-of-fold (OOF) predictions, and an A/B test |
| [`kaggle/arm14c/train_arm14.py`](../kaggle/arm14c/train_arm14.py) | arm14: 5-fold ConvNeXt-small on cache64 with out-of-fold (OOF) predictions, and an A/B test |
| [`kaggle/arm15/train_arm15.py`](../kaggle/arm15/train_arm15.py) | arm15: arm12's exact recipe, with the TARGETS refined by out-of-fold image predictions. |
| [`kaggle/arm16/train_arm16.py`](../kaggle/arm16/train_arm16.py) | arm16: does canonicalising anatomical orientation fix compartment attribution? |
| [`kaggle/arm16/train_arm16_kaggle.py`](../kaggle/arm16/train_arm16_kaggle.py) | Standalone Kaggle trainer for Arm 16 (Arm 13 + local ROI residuals). |
| [`kaggle/arm17/train_arm17.py`](../kaggle/arm17/train_arm17.py) | arm17: a real MCL measurement for CoAtNet, and the CoAtNet OOF we have never had. |
| [`kaggle/arm18/train_arm18.py`](../kaggle/arm18/train_arm18.py) | arm18: stop training on labels the report never actually supported. |
| [`kaggle/arm19/train_arm19.py`](../kaggle/arm19/train_arm19.py) | arm19: give the model some augmentation. It has had almost none. |
| [`kaggle/arm2/train_arm2.py`](../kaggle/arm2/train_arm2.py) | arm2: full-pool, 5-slot attention-MIL, SSL-init vs scratch-init A/B. |
| [`kaggle/arm20/train_arm20.py`](../kaggle/arm20/train_arm20.py) | arm20: the 20-epoch rerun of the augmentation A/B. Tests underfitting, not a new idea. |
| [`kaggle/arm3/train_arm3.py`](../kaggle/arm3/train_arm3.py) | arm3: scratch-init (tonight's A/B winner), full pool, 5-slot, resumable. |
| [`kaggle/arm4/train_arm4.py`](../kaggle/arm4/train_arm4.py) | arm4 (v2): latent case model -- slices as tokens, case latents, finding queries. |
| [`kaggle/arm5/train_arm5.py`](../kaggle/arm5/train_arm5.py) | arm5: arm3's proven attention-MIL, but with a RadImageNet-pretrained ResNet-50. |
| [`kaggle/arm6/train_arm6.py`](../kaggle/arm6/train_arm6.py) | arm6: identical to arm5, but ImageNet-pretrained ResNet-50 (CC0-licensed). |
| [`kaggle/arm7/train_arm7.py`](../kaggle/arm7/train_arm7.py) | arm7: now that perception is fixed, retest the two things arm4 got blamed for. |
| [`kaggle/arm8/train_arm8.py`](../kaggle/arm8/train_arm8.py) | arm8: arm6 with an EVIDENCE-BASED slice selection rule. 32 slices, 160px. |
| [`kaggle/arm9/train_arm9.py`](../kaggle/arm9/train_arm9.py) | arm9-B: arm6's model, slot-structured input. Signal check, not a full run. |
| [`kaggle/armc_roi/train_armc_roi_expert_kaggle_finetune.py`](../kaggle/armc_roi/train_armc_roi_expert_kaggle_finetune.py) | Standalone Kaggle trainer for Arm C ROI — SCRATCH training (no Arm 13 base). |
| [`kaggle/armc_roi/train_armc_roi_expert_kaggle_fixed_4.py`](../kaggle/armc_roi/train_armc_roi_expert_kaggle_fixed_4.py) | Standalone Kaggle trainer for Arm C ROI — SCRATCH training (no Arm 13 base). |
| [`kaggle/armc_roi/train_armc_roi_expert_kaggle_fixed_long_run.py`](../kaggle/armc_roi/train_armc_roi_expert_kaggle_fixed_long_run.py) | Standalone Kaggle trainer for Arm C ROI — SCRATCH training (no Arm 13 base). |
| [`kaggle/armc_roi/train_armc_roi_kaggle.py`](../kaggle/armc_roi/train_armc_roi_kaggle.py) | Standalone Kaggle trainer for Arm C ROI (Arm 13 + local ROI residuals). |
| [`kaggle/armc_roi/train_armc_roi_radiodino_expert_kaggle_fix.py`](../kaggle/armc_roi/train_armc_roi_radiodino_expert_kaggle_fix.py) | Standalone Kaggle trainer for Arm C ROI — RadioDINO scratch training. |
| [`kaggle/armc_roi/train_armc_roi_radiodino_expert_kaggle_fix_b8_f.py`](../kaggle/armc_roi/train_armc_roi_radiodino_expert_kaggle_fix_b8_f.py) | Standalone Kaggle trainer for Arm C ROI — RadioDINO scratch training. |
| [`kaggle/armc_roi/train_armc_roi_radiodino_expert_r336_b4.py`](../kaggle/armc_roi/train_armc_roi_radiodino_expert_r336_b4.py) | Standalone Kaggle trainer for Arm C ROI — RadioDINO scratch training. |
| [`kaggle/armc_roi/train_armc_roi_radiodino_kaggle.py`](../kaggle/armc_roi/train_armc_roi_radiodino_kaggle.py) | Standalone Kaggle trainer for Arm C ROI — RadioDINO scratch training. |
| [`kaggle/armc_roi/train_armc_roi_scratch_kaggle_fast-fix.py`](../kaggle/armc_roi/train_armc_roi_scratch_kaggle_fast-fix.py) | Standalone Kaggle trainer for Arm C ROI — SCRATCH training (no Arm 13 base). |

## Additional components

- [ROI geometry and annotation](../kaggle/roi_cache/): priors, revision tools, crop cache and browser application.
- [Model research](../kaggle/model_research/): diagnostics, permutation controls, subgroup analysis and split/provenance experiments. Several scripts require private historical inputs.
- [Self-supervised trainer](../kaggle/knee_dino/train_knee_dino.py): collapse/stability and curriculum experiments.
- [Recovered submissions](../kaggle/submissions/): latest-source snapshots, linked to scored versions in the ledger.
- [Portfolio configurations](../kaggle/portfolio/configs/): six research recipes and their explicit hyperparameters.

The source manifest fingerprints published files. The experiment summary directory retains the original numerical adoption decisions, including unsuccessful comparisons.
