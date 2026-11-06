"""Recompute the exact saved five-member TTA rank blend and paired uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[2]
LABELS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
MEMBERS = ("arm10", "arm11", "arm12", "arm13_best", "arm15")


def macro(y, p):
    values = {name: float(roc_auc_score(y[:, j], p[:, j])) for j, name in enumerate(LABELS)}
    return float(np.mean(list(values.values()))), values


def rank_blend(data, members):
    total = np.zeros_like(data["Y"], dtype=np.float64)
    for member in members:
        mean = np.nanmean(np.stack([data[member], data[member + "@rev"]]), axis=0)
        ranks = pd.DataFrame(mean).rank(pct=True).to_numpy()
        ranks = np.where(np.isnan(ranks), np.nanmedian(ranks, axis=0, keepdims=True), ranks)
        total += ranks
    return total / len(members)


def paired_bootstrap(y, a, b, draws=10000, seed=20260924):
    rng = np.random.default_rng(seed)
    deltas = []
    valid_targets = []
    for _ in range(draws):
        sample = rng.integers(0, len(y), len(y))
        ys = y[sample]
        target_delta = []
        for j in range(y.shape[1]):
            positives = ys[:, j].sum()
            if positives == 0 or positives == len(ys):
                continue
            ra = rankdata(a[sample, j], method="average")
            rb = rankdata(b[sample, j], method="average")
            n1 = positives
            n0 = len(ys) - n1
            auc_a = (ra[ys[:, j] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            auc_b = (rb[ys[:, j] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            target_delta.append(float(auc_a - auc_b))
        if target_delta:
            deltas.append(float(np.mean(target_delta)))
            valid_targets.append(len(target_delta))
    return {"draws": draws, "valid_draws": len(deltas),
            "mean_delta": float(np.mean(deltas)),
            "ci95": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
            "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
            "min_targets_per_draw": int(min(valid_targets))}


def main():
    path = REPO / "kaggle/submit11/output_v11/gold_member_preds.npz"
    data = np.load(path, allow_pickle=False)
    ids = data["ids"].astype(str)
    gold = pd.read_csv(REPO / "data/processed/gold_test.csv")
    if len(set(ids)) != 58 or set(ids) != set(gold.StudyInstanceUID.astype(str)):
        raise RuntimeError("Blend Gold IDs do not match label CSV")
    y = gold.set_index("StudyInstanceUID").loc[ids, LABELS].to_numpy(dtype=np.int64)
    if not np.array_equal(y, data["Y"]):
        raise RuntimeError("Blend truth differs from Gold CSV")
    five = rank_blend(data, MEMBERS)
    four = rank_blend(data, MEMBERS[:-1])
    arm13_tta = np.mean([data["arm13_best"], data["arm13_best@rev"]], axis=0)
    score5, per5 = macro(y, five)
    score4, per4 = macro(y, four)
    score13, per13 = macro(y, arm13_tta)
    recorded = json.loads((REPO / "kaggle/submit11/output_v11/validate_ensemble.json").read_text())
    if abs(score5 - recorded["ensemble_tta"]["auc"]) > 1e-12:
        raise RuntimeError("Saved blend score mismatch")
    report = {
        "source": str(path.relative_to(REPO)), "patients": len(ids), "reused_gold": True,
        "five_member_tta_rank": {"macro_auc": score5, "per_target_auc": per5},
        "four_member_tta_rank": {"macro_auc": score4, "per_target_auc": per4},
        "arm13_tta": {"macro_auc": score13, "per_target_auc": per13},
        "paired_bootstrap_five_minus_four": paired_bootstrap(y, five, four),
        "paired_bootstrap_five_minus_arm13": paired_bootstrap(y, five, arm13_tta),
    }
    output = REPO / "kaggle/model_research/output/blend_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k.endswith("rank") or k.startswith("paired_bootstrap")}, indent=2))


if __name__ == "__main__":
    main()
