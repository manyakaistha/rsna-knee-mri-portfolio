"""Audit all merged ROI provenance rows without reading image pixels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
GEOMETRY = (
    "center_x_mm", "center_y_mm", "center_z_mm",
    "crop_center_x_mm", "crop_center_y_mm", "crop_center_z_mm",
    "row0", "row1", "column0", "column1", "padding_fraction",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", type=Path,
                        default=REPO / "roi_work/cache_merged/roi_provenance.csv.gz")
    parser.add_argument("--gold", type=Path, default=REPO / "data/processed/gold_test.csv")
    parser.add_argument("--output", type=Path,
                        default=REPO / "kaggle/model_research/output/roi_geometry_audit.json")
    args = parser.parse_args()
    rows = pd.read_csv(args.provenance)
    gold = set(pd.read_csv(args.gold).StudyInstanceUID.astype(str))
    rows["gold"] = rows.StudyInstanceUID.astype(str).isin(gold)
    valid = rows.loc[rows.valid.eq(1)]
    invalid = rows.loc[rows.valid.eq(0)]
    if len(valid) + len(invalid) != len(rows):
        raise RuntimeError("ROI valid flag is not binary")
    reasons = invalid.failure_reason.fillna("missing").map(
        lambda reason: re.sub(r":[-+]?(?:\d*\.\d+|\d+)$", "", reason)
    )
    positions = valid[[f"slice_position_{i}_mm" for i in range(3)]].to_numpy(float)
    step = np.diff(positions, axis=1)
    errors = {
        "slice_triplet_reversal_or_duplicate": int((
            (step[:, 0] * step[:, 1] <= 0) |
            (np.abs(step).min(axis=1) < 1e-4)
        ).sum()),
        "nonpositive_crop_height": int((valid.row1 <= valid.row0).sum()),
        "nonpositive_crop_width": int((valid.column1 <= valid.column0).sum()),
        "nonfinite_geometry": int((~np.isfinite(valid[list(GEOMETRY)].to_numpy(float))).any(axis=1).sum()),
        "intensity_degenerate": int((valid.intensity_high <= valid.intensity_low).sum()),
        "valid_padding_above_20pct": int(valid.padding_fraction.gt(0.2 + 1e-10).sum()),
    }
    if any(errors.values()):
        raise RuntimeError(f"Valid ROI rows have geometry errors: {errors}")
    if set(reasons) - {
        "missing_required_series",
        "crop_failed:ValueError:padding_fraction_exceeds_0.20",
        "triplet_crosses_series_boundary",
    }:
        raise RuntimeError("Unexpected invalid ROI reason category")
    quantiles = (0, 0.5, 0.9, 0.95, 0.99, 1)
    report = {
        "rows": len(rows), "valid_rows": len(valid), "invalid_rows": len(invalid),
        "invalid_reason_groups": reasons.value_counts().to_dict(),
        "valid_by_target": valid.groupby("target").size().to_dict(),
        "gold_valid_by_target": valid.loc[valid.gold].groupby("target").size().to_dict(),
        "padding_quantiles": {str(q): float(x) for q, x in
                              valid.padding_fraction.quantile(quantiles).items()},
        "gold_padding_quantiles": {str(q): float(x) for q, x in
                                   valid.loc[valid.gold].padding_fraction.quantile(quantiles).items()},
        "padding_gt_0_2": int(valid.padding_fraction.gt(0.2).sum()),
        "laterality": valid.laterality.fillna("missing").value_counts().to_dict(),
        "candidate_laterality": valid.candidate_laterality.fillna("missing").value_counts().to_dict(),
        "window_strategy": valid.window_strategy.fillna("missing").value_counts().to_dict(),
        **{key: value for key, value in errors.items() if key != "valid_padding_above_20pct"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "rows", "valid_rows", "invalid_rows", "invalid_reason_groups", "padding_quantiles"
    )}, indent=2))


if __name__ == "__main__":
    main()
