"""Score 100 ROI-row permutation controls at the submitted K24 setting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

import diagnose_armc as diagnostic
from analyze_blend import LABELS, macro


REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=REPO / "kaggle/model_research/output/armc_diagnostics_gpu")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO.parent / "submisison_data/armc_roi_selected_model.pt")
    args = parser.parse_args()
    data = np.load(args.input / "gold_predictions.npz", allow_pickle=False)
    meta = json.loads((args.input / "diagnostics.json").read_text())
    reconstructed = np.load(args.input / "armc_k24_reconstructed.npz", allow_pickle=False)
    ids, y = data["ids"].astype(str), data["Y"]
    if len(ids) != 58 or len(set(ids)) != 58 or y.shape != (58, len(LABELS)):
        raise RuntimeError("Expected 58 unique keyed Gold rows")
    if not np.array_equal(ids, reconstructed["ids"].astype(str)):
        raise RuntimeError("K24 reconstruction ID mismatch")
    diagnostic.model_module = diagnostic.load_production_module()
    model = diagnostic.model_module.ArmC().cpu().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True)
    global_features, roi_features, masks = [], {t: [] for t in diagnostic.TARGET_SLUGS}, {
        t: [] for t in diagnostic.TARGET_SLUGS
    }
    for number, uid in enumerate(ids, 1):
        path = args.input / "features" / f"{number:03d}.npz"
        feature = np.load(path, allow_pickle=False)
        if feature["uid"].item() != uid or feature["armc_sha256"].item() != meta["checkpoint_sha256"]["armc"]:
            raise RuntimeError(f"Feature cache checkpoint/ID mismatch: {path}")
        if feature["input_sha256"].item() != meta["study_input_sha256"][uid]:
            raise RuntimeError(f"Feature cache input mismatch: {path}")
        global_features.append(feature["global_features"])
        for target, slug in diagnostic.TARGET_SLUGS.items():
            roi_features[target].append(feature[f"roi_{slug}"])
            masks[target].append(feature[f"mask_{slug}"])
    global62 = np.stack(global_features).astype(np.float32)
    roi = {target: np.stack(values).astype(np.float32) for target, values in roi_features.items()}
    mask = {target: np.stack(values).astype(bool) for target, values in masks.items()}
    selected24 = np.linspace(0, 61, 24).round().astype(int)
    global24 = global62[:, selected24]
    targets = tuple(roi)
    _, direct24, _ = diagnostic.forward_heads(model, global24, roi, mask, targets)
    _, direct62, _ = diagnostic.forward_heads(model, global62, roi, mask, targets)
    parity24 = float(np.max(np.abs(expit(direct24) - reconstructed["armc_k24"])))
    parity62 = float(np.max(np.abs(expit(direct62) - data["armc_combined"])))
    if parity24 > 2e-6 or parity62 > 2e-6:
        raise RuntimeError(f"Direct-head parity failed: K24 {parity24}, K62 {parity62}")
    real24, real_by_target = macro(y, expit(direct24))
    rng = np.random.default_rng(20260925)
    k24_scores, k62_scores = [], []
    target_scores = {target: [] for target in targets}
    for _ in range(100):
        shuffled_roi, shuffled_masks = {}, {}
        for target in targets:
            order = rng.permutation(len(ids))
            shuffled_roi[target] = roi[target][order]
            shuffled_masks[target] = mask[target][order]
        _, logits24, _ = diagnostic.forward_heads(model, global24, shuffled_roi, shuffled_masks, targets)
        _, logits62, _ = diagnostic.forward_heads(model, global62, shuffled_roi, shuffled_masks, targets)
        score24, by_target = macro(y, expit(logits24))
        score62, _ = macro(y, expit(logits62))
        k24_scores.append(score24)
        k62_scores.append(score62)
        for target in targets:
            target_scores[target].append(by_target[target])
    recorded62 = meta["roi_permutation_control"]
    mean62_error = abs(float(np.mean(k62_scores)) - recorded62["macro_auc_mean"])
    if mean62_error > 1e-8:
        raise RuntimeError(f"K62 permutation reproduction failed: mean error {mean62_error}")
    report = {
        "source_checkpoint_sha256": meta["checkpoint_sha256"]["armc"],
        "seed": 20260925,
        "draws": 100,
        "reused_gold": True,
        "direct_head_parity_max_abs_probability_error": {"k24": parity24, "k62": parity62},
        "k62_permutation_mean_error_vs_recorded": mean62_error,
        "real_k24_macro_auc": real24,
        "shuffled_k24_macro_auc_mean": float(np.mean(k24_scores)),
        "shuffled_k24_macro_auc_ci95": [float(x) for x in np.quantile(k24_scores, [0.025, 0.975])],
        "fraction_shuffled_below_real_k24": float(np.mean(np.asarray(k24_scores) < real24)),
        "roi_target_fraction_shuffled_below_real_k24": {
            target: float(np.mean(np.asarray(target_scores[target]) < real_by_target[target]))
            for target in targets
        },
        "shuffled_k24_macro_auc_draws": k24_scores,
    }
    output = args.input / "k24_permutation_controls.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "real_k24_macro_auc", "shuffled_k24_macro_auc_mean",
        "shuffled_k24_macro_auc_ci95", "fraction_shuffled_below_real_k24",
        "roi_target_fraction_shuffled_below_real_k24"
    )}, indent=2))


if __name__ == "__main__":
    main()
