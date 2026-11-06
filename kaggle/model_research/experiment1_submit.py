"""Frozen raw-DICOM inference for Experiment 1, run from the repository root.

Uses the same cache64 builder as the training cache and the slot-safe legal-center
contract. Requires the competition test files, selected checkpoint and this repo.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment1 import ARCH, LAB, Arm13, choose_centers, make_bag, sha

_builder = Path(__file__).resolve().parents[1]/"cache64"/"build_cache64.py"
_spec = importlib.util.spec_from_file_location("experiment1_cache_builder", _builder)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
build_study = _module.build_study


def predict_one(model, vol, mask, k, device, reverse=False):
    centers = choose_centers(mask, k, train=False)
    x, valid = make_bag(vol, mask, centers, reverse=reverse)
    # Encode in chunks without changing the MIL bag. This bounds K62 memory.
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        if not bool(valid.any()):
            raise ValueError("all-invalid study")
        chunks = []
        for part in x[valid].split(16):
            chunks.append(model.backbone(part.to(device)))
        hreal = torch.cat(chunks)
        h = hreal.new_zeros((len(centers), hreal.shape[-1]))
        h[valid.to(device)] = hreal
        h = model.norm(h[None])
        scores = model.att(h).float().masked_fill(~valid.to(device)[None, :, None], -torch.inf)
        pooled = torch.einsum("bkn,bkf->bnf", scores.softmax(1).to(h.dtype), h)
        logits = (pooled*model.clsW).sum(-1)+model.clsb
    return torch.sigmoid(logits.float()).cpu().numpy()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, choices=(24, 62), default=24)
    ap.add_argument("--reverse-tta", action="store_true")
    args = ap.parse_args()
    root = args.competition_root
    studies = pd.read_csv(root/"test.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(root/"test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    template = pd.read_csv(root/"sample_submission.csv", dtype={"StudyInstanceUID": str})
    ids = studies.StudyInstanceUID.astype(str).tolist()
    if len(ids) != len(set(ids)) or len(template) != len(ids) or set(template.StudyInstanceUID) != set(ids):
        raise ValueError("test/sample submission UID mismatch")
    by = {uid: group.to_dict("records") for uid, group in series.groupby("StudyInstanceUID", sort=False)}
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("arch") != ARCH or tuple(ck.get("lab", ())) != LAB or ck.get("recipe", {}).get("name") != "cache64-maxspan":
        raise ValueError("wrong experiment checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Arm13()
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    rows = []
    for uid in ids:
        vol, mask = build_study(uid, by.get(uid, []), root/"test_series")
        p = predict_one(model, vol, mask, args.k, device)
        if args.reverse_tta:
            p = (p+predict_one(model, vol, mask, args.k, device, reverse=True))/2
        rows.append(p)
    predictions = pd.DataFrame(rows, columns=list(LAB))
    predictions.insert(0, "StudyInstanceUID", ids)
    out = template[["StudyInstanceUID"]].merge(predictions, on="StudyInstanceUID", how="left", validate="one_to_one")
    if out[list(LAB)].isna().any().any():
        raise ValueError("missing submission predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    (args.output.parent/(args.output.stem+"_manifest.json")).write_text(json.dumps({
        "checkpoint_sha256": sha(args.checkpoint), "k": args.k, "reverse_tta": args.reverse_tta,
        "rows": len(out), "uid_unique": True}, indent=2)+"\n")


if __name__ == "__main__": main()
