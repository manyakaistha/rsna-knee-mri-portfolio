"""Descriptive Gold slices; suppress AUC where either class has <5 patients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from analyze_blend import LABELS


REPO = Path(__file__).resolve().parents[2]


def slice_metrics(y, baseline, candidate, idx):
    result = {"patients": int(np.sum(idx)), "targets": {}}
    if result["patients"] == 0:
        return result
    deltas = []
    for j, label in enumerate(LABELS):
        truth = y[idx, j]
        positives = int(truth.sum())
        negatives = len(truth) - positives
        entry = {"positives": positives, "negatives": negatives,
                 "brier_arm13": float(brier_score_loss(truth, baseline[idx, j])),
                 "brier_armc": float(brier_score_loss(truth, candidate[idx, j]))}
        if positives >= 5 and negatives >= 5:
            a = float(roc_auc_score(truth, baseline[idx, j]))
            b = float(roc_auc_score(truth, candidate[idx, j]))
            entry.update(auc_arm13=a, auc_armc=b, auc_delta=b-a)
            deltas.append(b-a)
        else:
            entry["auc_status"] = "unavailable: fewer than five in one class"
        result["targets"][label] = entry
    result["macro_auc_delta_over_eligible_targets"] = float(np.mean(deltas)) if deltas else None
    result["eligible_target_count"] = len(deltas)
    result["macro_brier_arm13"] = float(np.mean([x["brier_arm13"] for x in result["targets"].values()]))
    result["macro_brier_armc"] = float(np.mean([x["brier_armc"] for x in result["targets"].values()]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.input / "gold_predictions.npz", allow_pickle=False)
    ids = data["ids"].astype(str)
    y, baseline, candidate = data["Y"], data["arm13_k62"], data["armc_combined"]
    series = pd.read_csv(REPO / "references/eda_outputs/dicom_series_with_declared_metadata.csv.gz")
    series = series[series.split == "train"].copy()
    series.StudyInstanceUID = series.StudyInstanceUID.astype(str)
    if not set(ids).issubset(set(series.StudyInstanceUID)):
        raise RuntimeError("Missing Gold study in EDA series inventory")
    def mode(values):
        values = values.dropna().astype(str)
        return values.mode().iloc[0] if len(values) else "unknown"
    metadata = series.groupby("StudyInstanceUID").agg(
        manufacturer=("manufacturer_normalized", mode),
        field_strength=("field_strength_t", "median"),
        spacing=("pixel_spacing_row_mm", "median"),
        thickness=("slice_thickness_mm", "median"),
        series_count=("SeriesInstanceUID", "nunique"),
    )
    slots = series.groupby("StudyInstanceUID").apply(
        lambda x: len(set(zip(x.Anatomical_Plane.astype(str), x.Fluid_Sensitive.astype(str)))),
        include_groups=False,
    ).rename("slot_count")
    metadata = metadata.join(slots)
    laterality = pd.read_csv(REPO / "references/eda_outputs/study_laterality_availability.csv").set_index("StudyInstanceUID")
    laterality.index = laterality.index.astype(str)
    metadata = metadata.join(laterality)
    metadata = metadata.loc[ids]
    if metadata.drop(columns=["field_strength"]).isna().any().any():
        raise RuntimeError("Missing Gold subgroup metadata")
    groups = {
        "all": np.ones(len(ids), dtype=bool),
        "laterality_L": (metadata.laterality_status == "L").to_numpy(),
        "laterality_R": (metadata.laterality_status == "R").to_numpy(),
        "laterality_unknown_or_conflict": (~metadata.laterality_status.isin(["L", "R"])).to_numpy(),
        "manufacturer_Siemens": (metadata.manufacturer == "Siemens").to_numpy(),
        "manufacturer_Philips": (metadata.manufacturer == "Philips").to_numpy(),
        "manufacturer_other": (~metadata.manufacturer.isin(["Siemens", "Philips"])).to_numpy(),
        "field_strength_le1p5": (metadata.field_strength <= 1.5).to_numpy(),
        "field_strength_gt1p5": (metadata.field_strength > 1.5).to_numpy(),
        "field_strength_unknown": metadata.field_strength.isna().to_numpy(),
        "spacing_le0p35mm": (metadata.spacing <= 0.35).to_numpy(),
        "spacing_gt0p35mm": (metadata.spacing > 0.35).to_numpy(),
        "thickness_le3mm": (metadata.thickness <= 3.0).to_numpy(),
        "thickness_gt3mm": (metadata.thickness > 3.0).to_numpy(),
        "series_le5": (metadata.series_count <= 5).to_numpy(),
        "series_gt5": (metadata.series_count > 5).to_numpy(),
        "missing_one_or_more_five_slots": (metadata.slot_count < 5).to_numpy(),
        "all_five_slots_present": (metadata.slot_count >= 5).to_numpy(),
    }
    roi_ids = np.load(REPO / "roi_work/cache_merged/roi_ids.npy").astype(str)
    roi_lookup = {uid: i for i, uid in enumerate(roi_ids)}
    row_indices = np.array([roi_lookup[uid] for uid in ids])
    for target in ("mcl", "lateral_meniscus", "lateral_oa", "pf_oa", "synovitis"):
        mask = np.load(REPO / f"roi_work/cache_merged/roi_{target}_masks.npy")[row_indices]
        groups[f"roi_{target}_2or3"] = (mask.sum(1) < 4)
        groups[f"roi_{target}_4"] = (mask.sum(1) == 4)
    priors = json.loads((REPO / "roi_work/cache_merged/roi_priors.json").read_text())
    prior_ids = set(priors["fit_study_ids"]) | set(priors["audit_study_ids"])
    groups["excluding_roi_prior_images"] = np.array([uid not in prior_ids for uid in ids])
    result = {name: slice_metrics(y, baseline, candidate, idx) for name, idx in groups.items()}
    (args.input / "subgroups.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {"patients": x["patients"], "eligible_targets": x.get("eligible_target_count"),
                             "macro_auc_delta": x.get("macro_auc_delta_over_eligible_targets")}
                      for name, x in result.items()}, indent=2))


if __name__ == "__main__":
    main()
