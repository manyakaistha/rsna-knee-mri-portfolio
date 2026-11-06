"""Prespecified paired Experiment 1 report from keyed raw predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment1 import LAB, legal_centers, metrics


def auc(y, p):
    y = np.asarray(y) > .5
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return None
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.empty(len(p), float)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p))+1]
    ends = np.r_[starts[1:], len(p)]
    for lo, hi in zip(starts, ends): ranks[lo:hi] = (lo+hi+1)/2
    original = np.empty_like(ranks); original[order] = ranks
    return float((original[y].sum()-n1*(n1+1)/2)/(n1*n0))


def paired_ci(y, old, new, seed=20260924, draws=10000):
    rng = np.random.default_rng(seed)
    deltas = np.full(draws, np.nan)
    valid_counts = np.zeros(draws, int)
    target_deltas = np.full((draws, y.shape[1]), np.nan)
    for i in range(draws):
        take = rng.integers(0, len(y), len(y))
        for j in range(y.shape[1]):
            a = auc(y[take, j], old[take, j]); b = auc(y[take, j], new[take, j])
            if a is not None and b is not None:
                target_deltas[i, j] = b-a
        valid_counts[i] = np.isfinite(target_deltas[i]).sum()
        if valid_counts[i]: deltas[i] = np.nanmean(target_deltas[i])
    def summarize(v):
        v = v[np.isfinite(v)]
        return {"valid_draws": len(v), "mean_delta": float(v.mean()),
                "ci95": np.quantile(v, [.025, .975]).tolist(),
                "positive_fraction": float((v > 0).mean())}
    return {"macro": summarize(deltas), "per_target": {name: summarize(target_deltas[:, j])
            for j, name in enumerate(LAB)}, "minimum_valid_targets": int(valid_counts.min())}


def load_pair(output, split, k):
    old = np.load(output/f"{split}_arm13_k{k}.npz")
    new = np.load(output/f"{split}_experiment1_k{k}.npz")
    ids = old["ids"].astype(str)
    if len(ids) != len(set(ids)) or set(ids) != set(new["ids"].astype(str)):
        raise ValueError("missing or duplicate prediction UID")
    aligned = {u: i for i, u in enumerate(new["ids"].astype(str))}
    index = [aligned[u] for u in ids]
    y = old["truth"]
    if not np.array_equal(y, new["truth"][index]):
        raise ValueError("paired labels mismatch")
    return ids, y, old["probabilities"], new["probabilities"][index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args()
    masks = np.concatenate([np.load(args.cache/f"masks_shard{s}.npy") for s in range(2)])
    cache_ids = np.concatenate([np.load(args.cache/f"ids_shard{s}.npy").astype(str) for s in range(2)])
    if len(cache_ids) != len(set(cache_ids)): raise ValueError("duplicate cache UID")
    idrow = {u: i for i, u in enumerate(cache_ids)}
    report = {"bootstrap_seed": 20260924, "bootstrap_draws": 10000, "reused_gold": True,
              "comparisons": {}}
    for split in ("weak_locked", "gold_reused"):
        for k in (24, 62):
            ids, y, old, new = load_pair(args.output, split, k)
            slots = np.array([[bool(len(a)) for a in legal_centers(masks[idrow[u]])] for u in ids])
            groups = {"all": np.ones(len(ids), bool), "all_five_slots": slots.all(1),
                      "one_or_more_missing_slots": ~slots.all(1)}
            result = {"paired_bootstrap": paired_ci(y, old, new), "subgroups": {}}
            for group, keep in groups.items():
                count = int(keep.sum())
                if not count: continue
                per_target = {}
                for j, name in enumerate(LAB):
                    positives = int((y[keep, j] > .5).sum())
                    if positives >= 5 and count-positives >= 5:
                        per_target[name] = {"old_auc": auc(y[keep, j], old[keep, j]),
                                            "new_auc": auc(y[keep, j], new[keep, j]),
                                            "positives": positives}
                    else:
                        per_target[name] = {"old_auc": None, "new_auc": None,
                                            "positives": positives}
                result["subgroups"][group] = {"studies": count, "per_target": per_target}
            result["old"] = metrics(y, old)
            result["new"] = metrics(y, new)
            report["comparisons"][f"{split}_k{k}"] = result
    (args.output/"paired_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(args.output/"paired_report.json")


if __name__ == "__main__": main()
