"""Independent paired summary of completed keyed Arm C diagnostic predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from analyze_blend import LABELS, paired_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Directory with gold_predictions.npz and diagnostics.json")
    args = parser.parse_args()
    data = np.load(args.input / "gold_predictions.npz", allow_pickle=False)
    report = json.loads((args.input / "diagnostics.json").read_text())
    ids, y = data["ids"].astype(str), data["Y"].astype(np.int64)
    if len(ids) != 58 or len(set(ids)) != len(ids) or y.shape != (58, 12):
        raise RuntimeError("Expected 58 unique keyed Gold studies and 12 labels")
    arm13 = data["arm13_k62"]
    armc = data["armc_combined"]
    disabled = data["roi_disabled"]
    if not np.array_equal(arm13, disabled) or not np.array_equal(arm13, data["empty_roi"]):
        raise RuntimeError("ROI disable/empty behavior is not an exact no-op")
    per_target = {}
    for j, label in enumerate(LABELS):
        baseline_auc = float(roc_auc_score(y[:, j], arm13[:, j]))
        combined_auc = float(roc_auc_score(y[:, j], armc[:, j]))
        per_target[label] = {"arm13_auc": baseline_auc, "armc_auc": combined_auc,
                             "delta": combined_auc - baseline_auc}
    macro13 = float(np.mean([v["arm13_auc"] for v in per_target.values()]))
    macro_c = float(np.mean([v["armc_auc"] for v in per_target.values()]))
    recorded13, recorded_c = 0.8931414219220787, 0.8946623277206842
    if abs(macro13 - recorded13) > 1e-9 or abs(macro_c - recorded_c) > 1e-4:
        raise RuntimeError(f"Saved checkpoint Gold AUC differed beyond tolerance: {macro13}, {macro_c}")
    target_only = {}
    for label, slug in (("MCL", "mcl"), ("Lateral Meniscus", "lateral_meniscus"),
                        ("Lateral OA", "lateral_oa"), ("PF OA", "pf_oa"), ("Synovitis", "synovitis")):
        j = LABELS.index(label)
        target_only[label] = float(roc_auc_score(y[:, j], data["roi_only_ordering"][:, j]))
        one = data[f"only_{slug}"]
        other = [k for k in range(12) if k != j]
        if not np.array_equal(one[:, other], arm13[:, other]):
            raise RuntimeError(f"Off-target output changed for {label}")
    summary = {
        "source": str(args.input), "checkpoint_sha256": report["checkpoint_sha256"],
        "macro_auc": {"arm13_k24": report["metrics"]["arm13_k24"]["macro_auc"],
                      "arm13_k62": macro13, "armc_combined": macro_c,
                      "delta": macro_c - macro13,
                      "armc_minus_recorded": macro_c - recorded_c},
        "paired_bootstrap_armc_minus_arm13": paired_bootstrap(y, armc, arm13),
        "per_target": per_target, "roi_only_target_auc": target_only,
        "shuffled_rows_macro_auc": report["metrics"]["shuffled_roi_rows"]["macro_auc"],
        "shuffled_masks_macro_auc": report["metrics"]["shuffled_roi_masks"]["macro_auc"],
        "roi_permutation_control": report.get("roi_permutation_control"),
        "zero_roi_image_macro_auc": report["metrics"]["zero_roi_image"]["macro_auc"],
        "empty_roi_exact": report["logit_parity"]["empty_roi_exact"],
    }
    output = args.input / "paired_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["macro_auc", "paired_bootstrap_armc_minus_arm13", "shuffled_rows_macro_auc", "roi_permutation_control"]}, indent=2))


if __name__ == "__main__":
    main()
