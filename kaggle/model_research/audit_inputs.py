"""Freeze a patient-grouped weak-label split and audit unchanged cache inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


REPO = Path(__file__).resolve().parents[2]
TARGETS = ("mcl", "lateral_meniscus", "lateral_oa", "pf_oa", "synovitis")
SEED = "20260924"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def representative_patient_ids(series_csv: Path, dicom_root: Path):
    mapping = {}
    checked = 0
    for row in pd.read_csv(series_csv).itertuples(index=False):
        uid = str(row.StudyInstanceUID)
        series = str(row.SeriesInstanceUID)
        files = (dicom_root / uid / series).glob("*.dcm")
        first = next(files, None)
        if first is None:
            raise RuntimeError(f"Missing DICOM series {uid}/{series}")
        header = pydicom.dcmread(first, stop_before_pixels=True, specific_tags=["PatientID", "StudyInstanceUID", "SeriesInstanceUID"])
        if str(header.StudyInstanceUID) != uid or str(header.SeriesInstanceUID) != series:
            raise RuntimeError(f"Header UID mismatch {first}")
        patient = str(getattr(header, "PatientID", ""))
        if not patient:
            raise RuntimeError(f"Missing PatientID {first}")
        if uid in mapping and mapping[uid] != patient:
            raise RuntimeError(f"Inconsistent PatientID for study {uid}")
        mapping[uid] = patient
        checked += 1
    return mapping, checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO.parent / "data-root")
    parser.add_argument("--cache64", type=Path, default=REPO / "roi_work/cache64")
    parser.add_argument("--roi", type=Path, default=REPO / "roi_work/cache_merged")
    parser.add_argument("--gold", type=Path, default=REPO / "data/processed/gold_test.csv")
    parser.add_argument("--pool", type=Path, default=REPO / "data/processed/train_pool.csv")
    parser.add_argument("--output", type=Path, default=REPO / "kaggle/model_research/output/input_audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gold, pool = pd.read_csv(args.gold), pd.read_csv(args.pool)
    gold_ids, pool_ids = set(gold.StudyInstanceUID.astype(str)), set(pool.StudyInstanceUID.astype(str))
    if len(gold_ids) != len(gold) or len(pool_ids) != len(pool) or gold_ids & pool_ids:
        raise RuntimeError("Gold/pool duplicate or overlap")
    ids = np.concatenate([np.load(args.cache64 / f"ids_shard{s}.npy", allow_pickle=False) for s in (0, 1)]).astype(str)
    rid = np.load(args.roi / "roi_ids.npy", allow_pickle=False).astype(str)
    if len(set(ids)) != len(ids) or not np.array_equal(ids, rid) or set(ids) != gold_ids | pool_ids:
        raise RuntimeError("Global, ROI and label IDs are not exactly aligned")
    shapes = {}
    for shard in (0, 1):
        volume = np.load(args.cache64 / f"vols_shard{shard}.npy", mmap_mode="r")
        mask = np.load(args.cache64 / f"masks_shard{shard}.npy", mmap_mode="r")
        if volume.shape != (len(np.load(args.cache64 / f"ids_shard{shard}.npy")), 64, 336, 336) or mask.shape != volume.shape[:2]:
            raise RuntimeError(f"Global shard shape mismatch {shard}")
        shapes[f"global_shard{shard}"] = list(volume.shape)
    provenance = pd.read_csv(args.roi / "roi_provenance.csv.gz")
    if len(provenance) != len(ids) * len(TARGETS) * 4:
        raise RuntimeError("ROI provenance row count mismatch")
    id_to_row = {uid: row for row, uid in enumerate(ids)}
    roi_stats = {}
    for target in TARGETS:
        array = np.load(args.roi / f"roi_{target}.npy", mmap_mode="r")
        mask = np.load(args.roi / f"roi_{target}_masks.npy", mmap_mode="r").astype(bool)
        if array.shape != (len(ids), 4, 3, 336, 336) or mask.shape != (len(ids), 4):
            raise RuntimeError(f"ROI shape mismatch {target}")
        rows = provenance.loc[provenance.target == target]
        if len(rows) != len(ids) * 4 or not (rows.groupby("StudyInstanceUID").size() == 4).all():
            raise RuntimeError(f"ROI provenance key mismatch {target}")
        row_indices = rows.StudyInstanceUID.astype(str).map(id_to_row).to_numpy()
        if np.isnan(row_indices.astype(float)).any():
            raise RuntimeError(f"Unknown ROI provenance ID {target}")
        window_indices = rows.window_index.to_numpy()
        if not np.array_equal(mask[row_indices, window_indices], rows.valid.astype(bool).to_numpy()):
            raise RuntimeError(f"ROI provenance mask mismatch {target}")
        roi_stats[target] = {"valid_windows": int(mask.sum()), "studies_ge2_valid": int((mask.sum(1) >= 2).sum()),
                             "gold_studies_ge2_valid": int((mask[[uid in gold_ids for uid in ids]].sum(1) >= 2).sum())}
        shapes[f"roi_{target}"] = list(array.shape)
    patients, checked = representative_patient_ids(args.data_root / "train_series.csv", args.data_root / "train_series")
    if set(patients) != set(ids):
        raise RuntimeError("Patient-header audit does not cover cache IDs")
    if len(set(patients.values())) != len(patients):
        raise RuntimeError("A patient has multiple study IDs; inspect before splitting")
    if {patients[uid] for uid in gold_ids} & {patients[uid] for uid in pool_ids}:
        raise RuntimeError("Gold/pool patient overlap")
    # Hash assignment is frozen and patient grouped. Gold remains an external,
    # historically reused evaluation set and is never assigned to development.
    split_rows = []
    for uid in sorted(pool_ids):
        patient = patients[uid]
        digest = hashlib.sha256((SEED + "|" + patient).encode()).hexdigest()
        unit = int(digest[:16], 16) / 16**16
        split_rows.append({"StudyInstanceUID": uid, "patient_sha256": hashlib.sha256(patient.encode()).hexdigest(),
                           "split": "weak_locked" if unit < 0.15 else "weak_fit"})
    split = pd.DataFrame(split_rows)
    if split.split.nunique() != 2:
        raise RuntimeError("Weak split is empty")
    manifest_file = args.output / "weak_split.csv"
    if manifest_file.exists():
        old = pd.read_csv(manifest_file, dtype=str)
        if not old.equals(split.astype(str)):
            raise RuntimeError("Frozen weak split changed; refusing overwrite")
    else:
        split.to_csv(manifest_file, index=False)
    label_cols = [name for name in gold.columns if name in pool.columns and name not in {"StudyInstanceUID", "Report", "label_source"} and pd.api.types.is_numeric_dtype(gold[name])]
    pool_with_split = pool.merge(split[["StudyInstanceUID", "split"]], on="StudyInstanceUID", validate="one_to_one")
    report = {
        "seed": SEED, "gold_studies": len(gold_ids), "pool_studies": len(pool_ids),
        "series_headers_checked": checked, "unique_patients": len(set(patients.values())),
        "global_roi_ids_exact": True, "provenance_masks_exact": True,
        "shape": shapes, "roi": roi_stats,
        "weak_split_counts": split.split.value_counts().to_dict(),
        "weak_split_positive_counts": {s: {label: int((subset[label] > 0.5).sum()) for label in label_cols}
                                        for s, subset in pool_with_split.groupby("split")},
        "file_sha256": {str(path.relative_to(REPO)): hash_file(path) for path in [
            args.gold, args.pool, args.roi / "roi_ids.npy", args.roi / "roi_priors.json",
            args.roi / "roi_meta.json", args.data_root / "train_series.csv"
        ] if path.is_relative_to(REPO)},
        "split_sha256": hash_file(manifest_file),
    }
    (args.output / "audit_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["gold_studies", "pool_studies", "series_headers_checked", "unique_patients", "weak_split_counts", "split_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
