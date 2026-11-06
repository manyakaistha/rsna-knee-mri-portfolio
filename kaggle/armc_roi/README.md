# Global + local residual MIL

The main architecture extends Arm13's CoAtNet global model with five gated, target-specific ROI residual heads. Fresh heads and empty ROI masks leave global predictions unchanged; tests cover exact initialization, target isolation, packed execution and gradients.

- `model.py`, `losses.py`: reusable model and objective.
- `train_armc_roi_kaggle.py`: primary standalone historical trainer.
- `infer_submission.py`: historical inference.
- `check_roi_cache.py`: input preflight.
- Other `train_*` files: retained experimental variants, not all established as completed runs.

Read the [architecture](../../docs/ARCHITECTURE.md) and [model card](../../docs/MODEL_CARD.md). The selected historical Arm C weights are not in the confirmed model release. The available Arm13 warm-start checkpoint is not a replacement for trained ROI heads.
