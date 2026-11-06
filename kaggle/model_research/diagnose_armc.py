"""Keyed, resumable Arm13/ArmC Gold forward diagnostics on unchanged caches.

This is diagnostic tooling, not a new architecture or training entry point.
It imports the exact self-contained ArmC class that produced the checkpoint.
On CPU each encoded study is cached separately so an interruption loses at
most one study. Never put an untrusted checkpoint into torch.load.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


REPO = Path(__file__).resolve().parents[2]
TARGET_SLUGS = {
    "MCL": "mcl",
    "Lateral Meniscus": "lateral_meniscus",
    "Lateral OA": "lateral_oa",
    "PF OA": "pf_oa",
    "Synovitis": "synovitis",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_production_module():
    path = REPO / "kaggle/armc_roi/train_armc_roi_kaggle.py"
    spec = importlib.util.spec_from_file_location("armc_production", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def map_arm13(state):
    mapped = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith(("backbone.", "norm.")):
            mapped[key] = value
        elif key.startswith("att."):
            mapped["global_head." + key] = value
        elif key in {"clsW", "clsb"}:
            mapped["global_head." + key] = value
    return mapped


def aucs(y: np.ndarray, p: np.ndarray, labels: tuple[str, ...]) -> dict:
    values = {}
    for j, label in enumerate(labels):
        values[label] = float(roc_auc_score(y[:, j], p[:, j]))
    return {"macro_auc": float(np.mean(list(values.values()))), "per_target_auc": values}


def calibration(y: np.ndarray, p: np.ndarray, labels: tuple[str, ...]) -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        label: {
            "brier": float(brier_score_loss(y[:, j], p[:, j])),
            "log_loss": float(log_loss(y[:, j], p[:, j], labels=[0, 1])),
            "mean_probability": float(np.mean(p[:, j])),
            "prevalence": float(np.mean(y[:, j])),
        }
        for j, label in enumerate(labels)
    }


def forward_heads(model, global_features, roi_features, roi_masks, targets):
    device = next(model.parameters()).device
    amp = bool(getattr(model, "_diagnostic_amp", False)) and device.type == "cuda"
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16 if amp else torch.bfloat16, enabled=amp):
        h = torch.from_numpy(global_features).to(device)
        mask = torch.ones(h.shape[:2], dtype=torch.bool, device=device)
        global_logits = model.global_head(h, mask)
        combined = global_logits.clone()
        residuals = {}
        for target in targets:
            r = model.roi_poolers[target](
                torch.from_numpy(roi_features[target]).to(device),
                torch.from_numpy(roi_masks[target]).to(device),
            )
            residuals[target] = r.float().cpu().numpy()
            combined[:, model_module.ROI_INDEX[target]] += torch.sigmoid(model.beta[target]) * r
    return global_logits.float().cpu().numpy(), combined.float().cpu().numpy(), residuals


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm13", type=Path, default=REPO.parent / "submisison_data/arm13_best.pt")
    parser.add_argument("--armc", type=Path, default=REPO.parent / "submisison_data/armc_roi_selected_model.pt")
    parser.add_argument("--global-cache", type=Path, default=REPO / "roi_work/cache64")
    parser.add_argument("--roi-cache", type=Path, default=REPO / "roi_work/cache_merged")
    parser.add_argument("--gold", type=Path, default=REPO / "data/processed/gold_test.csv")
    parser.add_argument("--output", type=Path, default=REPO / "kaggle/model_research/output/armc_diagnostics")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--amp", action="store_true", help="Match the production CUDA float16 autocast path")
    parser.add_argument("--allow-global-change", action="store_true",
                        help="Evaluate a separately trained Arm C seed whose global weights changed")
    parser.add_argument("--expected-seed", type=int, default=0,
                        help="Fail if the selected checkpoint records a different training seed")
    parser.add_argument("--max-studies", type=int, default=0, help="Smoke only; zero means all 58")
    return parser.parse_args()


def main():
    global model_module
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    model_module = load_production_module()
    labels = tuple(model_module.LABELS)
    targets = tuple(model_module.ROI_TARGETS)
    checkpoint_hashes = {"arm13": sha256(args.arm13), "armc": sha256(args.armc)}
    arm13 = torch.load(args.arm13, map_location="cpu", weights_only=False)
    armc = torch.load(args.armc, map_location="cpu", weights_only=False)
    checkpoint_seed = armc.get("config", {}).get("seed")
    if args.expected_seed and checkpoint_seed != args.expected_seed:
        raise RuntimeError(f"Selected checkpoint seed {checkpoint_seed} != expected {args.expected_seed}")
    mapped = map_arm13(arm13["model"])
    state = armc["model"]
    missing = set(mapped) - set(state)
    unequal = [key for key, value in mapped.items() if key in state and not torch.equal(value, state[key])]
    if missing or (unequal and not args.allow_global_change):
        raise RuntimeError(f"Arm13 global mismatch: missing={len(missing)}, unequal={len(unequal)}")
    if args.allow_global_change and not unequal:
        raise RuntimeError("Repeat-seed mode requested, but all global tensors still equal Arm13")
    model = model_module.ArmC().to(args.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    model._diagnostic_amp = args.amp
    ids0 = np.load(args.global_cache / "ids_shard0.npy", allow_pickle=False)
    ids1 = np.load(args.global_cache / "ids_shard1.npy", allow_pickle=False)
    ids = np.concatenate((ids0, ids1)).astype(str)
    roi_ids = np.load(args.roi_cache / "roi_ids.npy", allow_pickle=False).astype(str)
    if len(set(ids)) != len(ids) or not np.array_equal(ids, roi_ids):
        raise RuntimeError("Global/ROI ID alignment failed")
    id_to_row = {uid: row for row, uid in enumerate(ids)}
    global_vols = [np.load(args.global_cache / f"vols_shard{s}.npy", mmap_mode="r") for s in (0, 1)]
    global_masks = np.concatenate(
        [np.load(args.global_cache / f"masks_shard{s}.npy") for s in (0, 1)]
    )
    roi_vols = {
        t: np.load(args.roi_cache / f"roi_{slug}.npy", mmap_mode="r")
        for t, slug in TARGET_SLUGS.items()
    }
    roi_masks = {
        t: np.load(args.roi_cache / f"roi_{slug}_masks.npy")
        for t, slug in TARGET_SLUGS.items()
    }
    gold = pd.read_csv(args.gold)
    gold_ids = gold.StudyInstanceUID.astype(str).to_numpy(dtype="U64")
    if len(set(gold_ids)) != len(gold_ids) or any(uid not in id_to_row for uid in gold_ids):
        raise RuntimeError("Gold IDs are duplicate or absent from cache")
    if args.max_studies:
        gold_ids = gold_ids[: args.max_studies]
    truth = gold.set_index("StudyInstanceUID").loc[gold_ids, labels].to_numpy(dtype=np.int64)
    if not np.isin(truth, [0, 1]).all():
        raise RuntimeError("Gold labels must be binary")
    feature_dir = args.output / "features"
    feature_dir.mkdir(exist_ok=True)
    all_global, all_roi, all_masks, inputs = [], {t: [] for t in targets}, {t: [] for t in targets}, []
    for number, uid in enumerate(gold_ids, 1):
        row = id_to_row[uid]
        feature_file = feature_dir / f"{number:03d}.npz"
        if feature_file.exists():
            saved = np.load(feature_file, allow_pickle=False)
            saved_amp = bool(saved["amp"].item()) if "amp" in saved.files else False
            if saved["uid"].item() != uid or saved["armc_sha256"].item() != checkpoint_hashes["armc"] or saved_amp != args.amp:
                raise RuntimeError(f"Feature cache identity mismatch: {feature_file}")
            global_features = saved["global_features"]
            local_features = {t: saved[f"roi_{TARGET_SLUGS[t]}"] for t in targets}
            masks = {t: saved[f"mask_{TARGET_SLUGS[t]}"] for t in targets}
            input_hash = saved["input_sha256"].item()
        else:
            valid = np.flatnonzero(global_masks[row] > 0)
            if len(valid) < 3:
                valid = np.arange(min(3, global_masks.shape[1]))
            lo, hi = int(valid.min()), int(valid.max())
            centers = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, 61))]
            if len(centers) != 62:
                raise RuntimeError(f"K62 cannot be reproduced for {uid}: {len(centers)} centers")
            shard = 0 if row < len(ids0) else 1
            shard_row = row if shard == 0 else row - len(ids0)
            indices = np.asarray(centers)[:, None] + np.array([-1, 0, 1])
            gx = np.array(global_vols[shard][shard_row, indices], copy=True)
            digest = hashlib.sha256(gx.tobytes())
            local_features, masks = {}, {}
            with torch.inference_mode(), torch.autocast(device_type=args.device, dtype=torch.float16, enabled=args.amp and args.device == "cuda"):
                encoded = model.encode(torch.from_numpy(gx).unsqueeze(0).to(args.device))
                global_features = model.norm(encoded).cpu().numpy()[0]
                for target in targets:
                    rx = np.array(roi_vols[target][row], copy=True)
                    rm = np.array(roi_masks[target][row], copy=True).astype(bool)
                    digest.update(rx.tobytes())
                    digest.update(rm.tobytes())
                    local_features[target] = model.encode(
                        torch.from_numpy(rx).unsqueeze(0).to(args.device)
                    ).cpu().numpy()[0]
                    masks[target] = rm
            input_hash = digest.hexdigest()
            payload = {"uid": uid, "armc_sha256": checkpoint_hashes["armc"], "amp": args.amp,
                       "input_sha256": input_hash, "global_features": global_features}
            for target in targets:
                payload[f"roi_{TARGET_SLUGS[target]}"] = local_features[target]
                payload[f"mask_{TARGET_SLUGS[target]}"] = masks[target]
            temp = feature_file.with_suffix(".tmp.npz")
            np.savez_compressed(temp, **payload)
            os.replace(temp, feature_file)
        all_global.append(global_features)
        for target in targets:
            all_roi[target].append(local_features[target])
            all_masks[target].append(masks[target])
        inputs.append(input_hash)
        print(f"{number}/{len(gold_ids)} {uid} encoded", flush=True)
    global62 = np.stack(all_global).astype(np.float32)
    roi = {t: np.stack(all_roi[t]).astype(np.float32) for t in targets}
    masks = {t: np.stack(all_masks[t]).astype(bool) for t in targets}
    selected24 = np.linspace(0, 61, 24).round().astype(int)
    global24 = global62[:, selected24]
    if not args.amp:
        model = model.cpu()
    g24, _, _ = forward_heads(model, global24, roi, masks, ())
    g62, combined, residuals = forward_heads(model, global62, roi, masks, targets)
    _, zero_residual, _ = forward_heads(model, global62, roi, masks, ())
    if not np.array_equal(g62, zero_residual):
        raise RuntimeError("Zero-residual logit parity failed")
    # Fresh-head parity applies only when the encoded features and global head
    # are those of the original Arm13 checkpoint. Repeats change both.
    if not args.allow_global_change:
        fresh = model_module.ArmC().to(next(model.parameters()).device)
        fresh.load_state_dict(mapped, strict=False)
        fresh.eval()
        fresh._diagnostic_amp = args.amp
        _, fresh_combined, _ = forward_heads(fresh, global62, roi, masks, targets)
        if not np.array_equal(g62, fresh_combined):
            raise RuntimeError("Fresh ROI warm-start logit parity failed")
        del fresh
    device = next(model.parameters()).device
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
        zero_image_feature = model.encode(torch.zeros(1, 1, 3, 336, 336, dtype=torch.uint8, device=device)).float().cpu().numpy()[0, 0]
    zero_roi = {t: np.broadcast_to(zero_image_feature, roi[t].shape).copy() for t in targets}
    _, zero_input, _ = forward_heads(model, global62, zero_roi, masks, targets)
    rng = np.random.default_rng(20260924)
    shuffled_rows, shuffled_masks = {}, {}
    for target in targets:
        permutation = rng.permutation(len(gold_ids))
        shuffled_rows[target] = roi[target][permutation]
        shuffled_masks[target] = masks[target][permutation]
    _, row_shuffle, _ = forward_heads(model, global62, shuffled_rows, shuffled_masks, targets)
    _, mask_shuffle, _ = forward_heads(model, global62, roi, shuffled_masks, targets)
    empty_masks = {t: np.zeros_like(masks[t]) for t in targets}
    _, empty_roi, _ = forward_heads(model, global62, roi, empty_masks, targets)
    if not np.array_equal(empty_roi, g62):
        raise RuntimeError("Empty ROI did not preserve all global logits exactly")
    global24_name = "seed_global_k24" if args.allow_global_change else "arm13_k24"
    global62_name = "seed_global_k62" if args.allow_global_change else "arm13_k62"
    combined_name = "seed_combined" if args.allow_global_change else "armc_combined"
    predictions = {global24_name: torch.sigmoid(torch.from_numpy(g24)).numpy(),
                   global62_name: torch.sigmoid(torch.from_numpy(g62)).numpy(),
                   combined_name: torch.sigmoid(torch.from_numpy(combined)).numpy(),
                   "roi_disabled": torch.sigmoid(torch.from_numpy(g62)).numpy(),
                   "forced_zero_residual": torch.sigmoid(torch.from_numpy(zero_residual)).numpy(),
                   "zero_roi_image": torch.sigmoid(torch.from_numpy(zero_input)).numpy(),
                   "shuffled_roi_rows": torch.sigmoid(torch.from_numpy(row_shuffle)).numpy(),
                   "shuffled_roi_masks": torch.sigmoid(torch.from_numpy(mask_shuffle)).numpy(),
                   "empty_roi": torch.sigmoid(torch.from_numpy(empty_roi)).numpy()}
    # A residual alone is an ordering ablation, not a calibrated probability.
    roi_only = np.zeros_like(g62)
    for target in targets:
        roi_only[:, model_module.ROI_INDEX[target]] = (
            float(torch.sigmoid(model.beta[target]).detach()) * residuals[target]
        )
    predictions["roi_only_ordering"] = torch.sigmoid(torch.from_numpy(roi_only)).numpy()
    for target in targets:
        _, one, _ = forward_heads(model, global62, roi, masks, (target,))
        other = [j for j in range(len(labels)) if j != model_module.ROI_INDEX[target]]
        if not np.array_equal(one[:, other], g62[:, other]):
            raise RuntimeError(f"ROI target isolation failed for {target}")
        predictions[f"only_{TARGET_SLUGS[target]}"] = torch.sigmoid(torch.from_numpy(one)).numpy()
    np.savez_compressed(args.output / "gold_predictions.npz", ids=gold_ids, Y=truth,
                        **predictions, **{f"residual_{TARGET_SLUGS[t]}": residuals[t] for t in targets})
    report = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "precision_mode": "cuda_autocast_float16" if args.amp else "float32",
        "repeat_seed_mode": args.allow_global_change,
        "checkpoint_sha256": checkpoint_hashes,
        "checkpoint_seed": checkpoint_seed,
        "checkpoint_selection": armc.get("selection"),
        "checkpoint_gold_final": armc.get("gold_final"),
        "trainer_sha256": sha256(REPO / "kaggle/armc_roi/train_armc_roi_kaggle.py"),
        "global_tensor_equality": {"compared": len(mapped), "missing": len(missing), "unequal": len(unequal)},
        "gold_ids": gold_ids.tolist(), "study_input_sha256": dict(zip(gold_ids.tolist(), inputs)),
        "logit_parity": {"roi_disabled_exact": True, "fresh_roi_exact": None if args.allow_global_change else True,
                         "empty_roi_exact": True, "off_target_exact": True},
        "shuffled_mask_changed_cells": {t: int(np.count_nonzero(masks[t] != shuffled_masks[t])) for t in targets},
        "metrics": {},
    }
    if len(gold_ids) == 58:
        for name, pred in predictions.items():
            report["metrics"][name] = {**aucs(truth, pred, labels), "calibration": calibration(truth, pred, labels)}
        report["correlation"] = {
            label: {"pearson": float(np.corrcoef(predictions[global62_name][:, j], predictions[combined_name][:, j])[0, 1]),
                    "spearman": float(spearmanr(predictions[global62_name][:, j], predictions[combined_name][:, j]).statistic)}
            for j, label in enumerate(labels)
        }
        rng = np.random.default_rng(20260925)
        permutation_scores = []
        for _ in range(100):
            permuted_roi, permuted_masks = {}, {}
            for target in targets:
                order = rng.permutation(len(gold_ids))
                permuted_roi[target] = roi[target][order]
                permuted_masks[target] = masks[target][order]
            _, permuted_logits, _ = forward_heads(model, global62, permuted_roi, permuted_masks, targets)
            permutation_scores.append(aucs(truth, torch.sigmoid(torch.from_numpy(permuted_logits)).numpy(), labels)["macro_auc"])
        real = report["metrics"][combined_name]["macro_auc"]
        report["roi_permutation_control"] = {
            "seed": 20260925, "draws": 100,
            "macro_auc_mean": float(np.mean(permutation_scores)),
            "macro_auc_ci95": [float(x) for x in np.quantile(permutation_scores, [0.025, 0.975])],
            "fraction_below_real": float(np.mean(np.asarray(permutation_scores) < real)),
        }
    (args.output / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"studies": len(gold_ids), "metrics": {k: v["macro_auc"] for k, v in report["metrics"].items()},
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
