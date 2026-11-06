"""Recover the selected Arm C K24 Gold path from saved keyed residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logit

from analyze_blend import LABELS, macro, paired_bootstrap


REPO = Path(__file__).resolve().parents[2]
ROI = {
    "MCL": "mcl", "Lateral Meniscus": "lateral_meniscus",
    "Lateral OA": "lateral_oa", "PF OA": "pf_oa", "Synovitis": "synovitis",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=REPO / "kaggle/model_research/output/armc_diagnostics_gpu")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO.parent / "submisison_data/armc_roi_selected_model.pt")
    args = parser.parse_args()
    path = args.input / "gold_predictions.npz"
    data = np.load(path, allow_pickle=False)
    y, ids = data["Y"], data["ids"].astype(str)
    if y.shape != (58, len(LABELS)) or len(set(ids)) != 58:
        raise RuntimeError("Expected 58 unique keyed Gold predictions")
    selected = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = selected["model"]

    def combined(global_p):
        logits = logit(np.clip(global_p.astype(np.float64), 1e-8, 1 - 1e-8))
        for label, slug in ROI.items():
            gate = float(torch.sigmoid(state["beta." + label]).item())
            logits[:, LABELS.index(label)] += gate * data["residual_" + slug].astype(np.float64)
        return expit(logits)

    k62 = combined(data["arm13_k62"])
    error = float(np.max(np.abs(k62 - data["armc_combined"])))
    if error > 2e-6:
        raise RuntimeError(f"K62 reconstruction differs from direct Arm C: max {error}")
    k24 = combined(data["arm13_k24"])
    scores = {}
    per_target = {}
    for name, p in (("arm13_k24", data["arm13_k24"]),
                    ("armc_k24", k24),
                    ("arm13_k62", data["arm13_k62"]),
                    ("armc_k62", data["armc_combined"])):
        scores[name], target_scores = macro(y, p)
        for label, value in target_scores.items():
            per_target.setdefault(label, {})[name] = value
    report = {
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "source_predictions_sha256": sha256(path),
        "method": "Add saved target ROI residuals times selected checkpoint sigmoid(beta) to logit(saved Arm13 K24 probabilities); verify against saved K62 ArmC probabilities.",
        "k62_reconstruction_max_abs_probability_error": error,
        "macro_auc": scores,
        "roi_delta_k24": scores["armc_k24"] - scores["arm13_k24"],
        "roi_delta_k62": scores["armc_k62"] - scores["arm13_k62"],
        "paired_bootstrap_roi_k24": paired_bootstrap(y, k24, data["arm13_k24"]),
        "per_target": per_target,
        "reused_gold": True,
    }
    np.savez_compressed(args.input / "armc_k24_reconstructed.npz", ids=ids, Y=y,
                        arm13_k24=data["arm13_k24"], armc_k24=k24)
    (args.input / "armc_k24_reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "k62_reconstruction_max_abs_probability_error", "macro_auc", "roi_delta_k24",
        "roi_delta_k62", "paired_bootstrap_roi_k24"
    )}, indent=2))


if __name__ == "__main__":
    main()
