"""Paired, keyed Stage 2 summary for one independently trained Arm C seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from analyze_blend import LABELS, macro, paired_bootstrap


def load_predictions(directory: Path):
    data = np.load(directory / "gold_predictions.npz", allow_pickle=False)
    ids = data["ids"].astype(str)
    y = data["Y"].astype(np.int64)
    if len(ids) != 58 or len(set(ids)) != 58 or y.shape != (58, len(LABELS)):
        raise RuntimeError(f"Expected 58 unique keyed Gold rows in {directory}")
    return data, ids, y


def calibrated_loss(y, p):
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "macro_brier": float(np.mean([
            brier_score_loss(y[:, j], clipped[:, j]) for j in range(y.shape[1])
        ])),
        "macro_log_loss": float(np.mean([
            log_loss(y[:, j], clipped[:, j], labels=[0, 1])
            for j in range(y.shape[1])
        ])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed_data, seed_ids, y = load_predictions(args.input)
    base_data, base_ids, base_y = load_predictions(args.baseline)
    if not np.array_equal(seed_ids, base_ids) or not np.array_equal(y, base_y):
        raise RuntimeError("Repeat seed and original Arm C keyed Gold rows differ")
    diagnostic = json.loads((args.input / "diagnostics.json").read_text())
    if not diagnostic.get("repeat_seed_mode") or diagnostic.get("checkpoint_seed") != args.seed:
        raise RuntimeError("Repeat checkpoint metadata or mode mismatch")
    seed_global = seed_data["seed_global_k62"]
    seed_combined = seed_data["seed_combined"]
    original_global = base_data["arm13_k62"]
    original_combined = base_data["armc_combined"]
    for name, p in (("seed_global", seed_global), ("seed_combined", seed_combined),
                    ("original_global", original_global), ("original_combined", original_combined)):
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise RuntimeError(f"Invalid probability matrix {name}")
    for key in ("roi_disabled", "forced_zero_residual", "empty_roi"):
        if not np.array_equal(seed_global, seed_data[key]):
            raise RuntimeError(f"{key} is not an exact no-op")
    score_global, per_global = macro(y, seed_global)
    score_combined, per_combined = macro(y, seed_combined)
    score_original, per_original = macro(y, original_combined)
    recorded = diagnostic.get("checkpoint_gold_final") or {}
    if not {"global_auc", "combined_auc"}.issubset(recorded):
        raise RuntimeError("Repeat checkpoint lacks its recorded final Gold scores")
    if abs(score_global - recorded["global_auc"]) > 1e-4:
        raise RuntimeError("Seed global Gold AUC does not reproduce checkpoint record")
    if abs(score_combined - recorded["combined_auc"]) > 1e-4:
        raise RuntimeError("Seed combined Gold AUC does not reproduce checkpoint record")
    shuffle = diagnostic["roi_permutation_control"]
    summary = {
        "seed": args.seed,
        "patients": len(seed_ids),
        "reused_gold": True,
        "checkpoint_sha256": diagnostic["checkpoint_sha256"],
        "checkpoint_selection": diagnostic["checkpoint_selection"],
        "global_tensors_changed_from_arm13": diagnostic["global_tensor_equality"]["unequal"],
        "macro_auc": {
            "seed_global": score_global,
            "seed_combined": score_combined,
            "original_armc": score_original,
            "roi_lift_within_seed": score_combined - score_global,
            "seed_minus_original_armc": score_combined - score_original,
        },
        "paired_bootstrap_roi_lift": paired_bootstrap(y, seed_combined, seed_global),
        "paired_bootstrap_vs_original_armc": paired_bootstrap(y, seed_combined, original_combined),
        "paired_bootstrap_global_drift": paired_bootstrap(y, seed_global, original_global),
        "row_permutation_control": shuffle,
        "calibration": {
            "seed_global": calibrated_loss(y, seed_global),
            "seed_combined": calibrated_loss(y, seed_combined),
            "original_armc": calibrated_loss(y, original_combined),
        },
        "per_target": {
            label: {
                "seed_global_auc": per_global[label],
                "seed_combined_auc": per_combined[label],
                "original_armc_auc": per_original[label],
                "within_seed_roi_delta": per_combined[label] - per_global[label],
                "vs_original_armc_delta": per_combined[label] - per_original[label],
            }
            for label in LABELS
        },
    }
    output = args.input / "repeat_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "seed", "macro_auc", "paired_bootstrap_roi_lift",
        "paired_bootstrap_vs_original_armc", "row_permutation_control"
    )}, indent=2))


if __name__ == "__main__":
    main()
