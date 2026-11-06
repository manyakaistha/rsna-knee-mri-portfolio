"""Experiment 1: slot-safe K24 adaptation of Arm13. Run --mode probe before --mode train.

Inputs are the original cache64 shards, gold_test.csv, train_pool.csv, the frozen
weak_split.csv, and arm13_best.pt. No Gold labels are used for fitting or selection.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import random
import shutil
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
LAB = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
CAPACITY = (18, 14, 12, 8, 12)
OFFSETS = np.cumsum((0,) + CAPACITY)
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
MEAN = torch.tensor((.485, .456, .406)).view(3, 1, 1)
STD = torch.tensor((.229, .224, .225)).view(3, 1, 1)
CONFIG = {"experiment": 1, "seed": 42, "epochs": 4, "k_train": 24,
          "eval_k": [24, 62], "resolution": 384, "global_batch": 8,
          "backbone_lr": 1e-5, "head_lr": 3e-5, "weight_decay": .02,
          "onecycle_pct_start": .15, "gradient_clip": 3.0, "gain_aug": .20,
          "selection": "maximum weak_locked K24 macro AUC, earliest epoch on ties",
          "allocator": "largest remainder; capacity then slot-index tie; valid-center cap; duplicate only if total legal centers < K"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unique_ids(values, name):
    ids = [str(x) for x in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {name} IDs")
    return ids


def legal_centers(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (64,):
        raise ValueError("expected 64-slice cache mask")
    return [np.array([c for c in range(int(OFFSETS[s])+1, int(OFFSETS[s+1])-1)
                      if mask[c-1:c+2].all()], dtype=np.int16) for s in range(5)]


def quota(legal, k):
    """Hamilton allocation with exact, stable ties; first cover each present slot."""
    active = [s for s, a in enumerate(legal) if len(a)]
    if not active:
        raise ValueError("study has no legal three-slice window")
    if k < len(active):
        raise ValueError("K is smaller than present-slot count")
    q = np.zeros(5, dtype=np.int64)
    q[active] = 1
    remaining = k - len(active)
    while remaining:
        eligible = [s for s in active if q[s] < len(legal[s])]
        if not eligible:
            # Only an exceptionally sparse cache requires duplicates.
            eligible = active
        weights = np.asarray([CAPACITY[s] for s in eligible], dtype=np.float64)
        exact = remaining * weights / weights.sum()
        floors = np.floor(exact).astype(int)
        if all(q[s] + n <= len(legal[s]) for s, n in zip(eligible, floors)):
            for s, n in zip(eligible, floors):
                q[s] += n
            remaining -= int(floors.sum())
            if remaining == 0:
                break
            order = sorted(range(len(eligible)), key=lambda i: (-(exact[i]-floors[i]), eligible[i]))
            for i in order:
                s = eligible[i]
                if remaining and (q[s] < len(legal[s]) or sum(map(len, legal)) < k):
                    q[s] += 1
                    remaining -= 1
        else:
            order = sorted(eligible, key=lambda s: (-CAPACITY[s] / max(q[s], 1), s))
            q[order[0]] += 1
            remaining -= 1
    assert int(q.sum()) == k
    return q


def choose_centers(mask, k, train=False, seed=0):
    legal = legal_centers(mask)
    if k == 62:  # all legal centers, pad to a fixed tensor; padded entries are masked
        centers = [int(c) for slot in legal for c in slot]
        return centers + [-1] * (k-len(centers))
    q = quota(legal, k)
    rng = np.random.default_rng(seed)
    out = []
    for s, count in enumerate(q):
        if not count:
            continue
        a = legal[s]
        if train:
            picks = rng.choice(a, size=count, replace=count > len(a))
        elif count <= len(a):
            picks = a[np.linspace(0, len(a)-1, count).round().astype(int)]
        else:
            picks = a[np.arange(count) % len(a)]
        out.extend(map(int, picks))
    return out


def make_bag(vol, mask, centers, train=False, seed=0, reverse=False):
    valid = np.asarray([c >= 0 for c in centers], dtype=bool)
    x = np.zeros((len(centers), 3, 336, 336), dtype=np.uint8)
    for i, c in enumerate(centers):
        if c >= 0:
            idx = [c+1, c, c-1] if reverse else [c-1, c, c+1]
            x[i] = vol[idx]
    x = torch.from_numpy(x).float().div_(255)
    x = F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False)
    if train:
        rng = np.random.default_rng(seed ^ 0x5A17)
        x = (x * (1 + (rng.random()-.5)*CONFIG["gain_aug"])).clamp_(0, 1)
    x = (x-MEAN)/STD
    x[~torch.from_numpy(valid)] = 0
    return x, torch.from_numpy(valid)


class Cache:
    def __init__(self, paths):
        self.vol = [np.load(paths[f"vols_shard{s}.npy"], mmap_mode="r") for s in range(2)]
        self.ids = unique_ids(np.concatenate([np.load(paths[f"ids_shard{s}.npy"], allow_pickle=False).astype(str) for s in range(2)]), "cache")
        self.masks = np.concatenate([np.load(paths[f"masks_shard{s}.npy"]) for s in range(2)])
        if len(self.ids) != len(self.masks) or self.masks.shape[1] != 64:
            raise ValueError("cache IDs/masks mismatch")
        self.row = {uid: i for i, uid in enumerate(self.ids)}
        self.n0 = len(self.vol[0])

    def get(self, uid):
        r = self.row[uid]
        return self.vol[0][r] if r < self.n0 else self.vol[1][r-self.n0], self.masks[r]


class Studies(Dataset):
    def __init__(self, uids, labels, cache_paths, k, train=False, epoch=0, seed=42):
        self.uids, self.labels, self.paths = uids, labels, cache_paths
        self.k, self.train, self.epoch, self.seed = k, train, epoch, seed
        self.cache = None

    def __len__(self): return len(self.uids)

    def __getitem__(self, index):
        if self.cache is None:
            self.cache = Cache(self.paths)
        uid = self.uids[index]
        vol, mask = self.cache.get(uid)
        draw_seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{self.epoch}:{uid}".encode()).digest()[:8], "little")
        centers = choose_centers(mask, self.k, self.train, draw_seed)
        x, valid = make_bag(vol, mask, centers, self.train, draw_seed)
        return x, valid, torch.tensor(self.labels[uid], dtype=torch.float32), uid


class Arm13(nn.Module):
    """Checkpoint-compatible Arm13 MIL; invalid windows receive zero attention."""
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model(ARCH, pretrained=False, num_classes=0, in_chans=3, global_pool="avg")
        self.backbone.set_grad_checkpointing(True)
        f = self.backbone.num_features
        self.norm = nn.LayerNorm(f)
        self.att = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.empty(12, f))
        self.clsb = nn.Parameter(torch.zeros(12))

    def forward(self, x, valid):
        if not bool(valid.any(dim=1).all()):
            raise ValueError("all-invalid study bag")
        b, k = x.shape[:2]
        # Encode only real windows. Padding never touches BatchNorm or attention.
        flat = x.flatten(0, 1)
        keep = valid.flatten()
        real = flat[keep]
        hreal = self.backbone(real) if self.training else torch.cat(
            [self.backbone(part) for part in real.split(16)], dim=0)
        h = hreal.new_zeros((b*k, hreal.shape[-1])).index_copy(0, keep.nonzero().flatten(), hreal)
        h = self.norm(h.reshape(b, k, -1))
        scores = self.att(h).float().masked_fill(~valid[..., None], -torch.inf)
        a = scores.softmax(dim=1).to(h.dtype)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled*self.clsW).sum(-1)+self.clsb


def load_data(args):
    paths = {name: args.cache / name for s in range(2) for name in (f"vols_shard{s}.npy", f"masks_shard{s}.npy", f"ids_shard{s}.npy")}
    if any(not p.is_file() for p in paths.values()):
        raise FileNotFoundError("cache64 shards missing")
    cache = Cache(paths)
    split = pd.read_csv(args.split, dtype={"StudyInstanceUID": str})
    pool = pd.read_csv(args.pool, dtype={"StudyInstanceUID": str})
    gold = pd.read_csv(args.gold, dtype={"StudyInstanceUID": str})
    for name, frame in (("split", split), ("pool", pool), ("gold", gold)):
        unique_ids(frame.StudyInstanceUID, name)
    fit = split.loc[split.split.eq("weak_fit"), "StudyInstanceUID"].tolist()
    locked = split.loc[split.split.eq("weak_locked"), "StudyInstanceUID"].tolist()
    if len(fit) != 3712 or len(locked) != 637 or set(fit) & set(locked):
        raise ValueError("frozen weak split size/overlap mismatch")
    if set(fit+locked) != set(pool.StudyInstanceUID) or set(gold.StudyInstanceUID) & set(pool.StudyInstanceUID):
        raise ValueError("split/labels mismatch or Gold overlap")
    if not set(fit+locked+gold.StudyInstanceUID.astype(str).tolist()).issubset(cache.row):
        raise ValueError("labelled study missing from cache")
    if sha(args.split) != "e54952b32e6dd87b38740c4a3ab1daf02e6dfb7a0588baa4650d20a0ec6404b1":
        raise ValueError("frozen weak split hash mismatch")
    labels = {}
    for frame in (pool, gold):
        values = frame.loc[:, list(LAB)].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("labels must be complete probabilities in [0,1]")
        if frame is gold and not np.isin(values, (0, 1)).all():
            raise ValueError("Gold labels must be binary")
        labels.update(zip(frame.StudyInstanceUID.astype(str), values))
    prev = np.clip(np.stack([labels[u] for u in fit]).mean(0), .03, .7)
    pw = np.clip((1-prev)/prev, 1, 10).astype(np.float32)
    return paths, labels, fit, locked, gold.StudyInstanceUID.astype(str).tolist(), pw


def metrics(y, p):
    result = {"per_target": {}}
    for j, name in enumerate(LAB):
        truth = y[:, j] > 0.5
        npos, nneg = int(truth.sum()), int((~truth).sum())
        if not npos or not nneg:
            raise ValueError(f"AUC undefined for {name}")
        order = np.argsort(p[:, j], kind="stable")
        values = p[order, j]
        ranks = np.empty(len(values), dtype=np.float64)
        starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
        ends = np.r_[starts[1:], len(values)]
        for a, b in zip(starts, ends):
            ranks[a:b] = (a + b + 1) / 2
        original_ranks = np.empty_like(ranks)
        original_ranks[order] = ranks
        auc = (original_ranks[truth].sum() - npos*(npos+1)/2) / (npos*nneg)
        prob = np.clip(p[:, j].astype(np.float64), 1e-7, 1-1e-7)
        binary = truth.astype(np.float64)
        result["per_target"][name] = {"auc": float(auc),
            "brier": float(np.mean((prob-binary)**2)),
            "log_loss": float(-np.mean(binary*np.log(prob)+(1-binary)*np.log1p(-prob)))}
    for key in ("auc", "brier", "log_loss"):
        result["macro_"+key] = float(np.mean([v[key] for v in result["per_target"].values()]))
    return result


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, logits, truth, masks = [], [], [], []
    for x, valid, y, uid in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            z = model(x.to(device, non_blocking=True), valid.to(device, non_blocking=True))
        ids.extend(uid); logits.append(z.float().cpu().numpy()); truth.append(y.numpy()); masks.append(valid.numpy())
    ids = unique_ids(ids, "prediction")
    return ids, np.concatenate(logits), np.concatenate(truth), np.concatenate(masks)


def save_predictions(path, ids, logits, truth, masks):
    if len(ids) != len(set(ids)) or not np.isfinite(logits).all():
        raise ValueError("invalid predictions")
    np.savez_compressed(path, ids=np.asarray(ids), logits=logits, probabilities=1/(1+np.exp(-logits)),
                        truth=truth, window_mask=masks, targets=np.asarray(LAB))


def loader(ids, labels, paths, k, batch, workers, train=False, epoch=0, seed=42, world=1, rank=0):
    ds = Studies(ids, labels, paths, k, train, epoch, seed)
    sampler = DistributedSampler(ds, world, rank, shuffle=train, seed=seed, drop_last=train) if world > 1 else None
    dl = DataLoader(ds, batch_size=batch, sampler=sampler, shuffle=train and sampler is None,
                    drop_last=train, num_workers=workers, pin_memory=True)
    return dl, sampler


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state["cuda"] is not None: torch.cuda.set_rng_state(state["cuda"])


def optimizer(model):
    return torch.optim.AdamW([{"params": list(model.backbone.parameters()), "lr": CONFIG["backbone_lr"]},
                              {"params": [p for n,p in model.named_parameters() if not n.startswith("backbone.")],
                               "lr": CONFIG["head_lr"]}], weight_decay=CONFIG["weight_decay"])


def write_json(path, obj): path.write_text(json.dumps(obj, indent=2)+"\n")


def setup_model(args, device):
    ck = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    if ck.get("arch") != ARCH or tuple(ck.get("lab", ())) != LAB or ck.get("recipe", {}).get("name") != "cache64-maxspan":
        raise ValueError("warm-start metadata is not Arm13 cache64")
    model = Arm13()
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device)


def probe_worker(rank, args, port, micro):
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=2, timeout=datetime.timedelta(minutes=10))
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    paths, labels, fit, _, _, pw = load_data(args)
    model = setup_model(args, device)
    net = DDP(model, device_ids=[rank])
    opt = optimizer(model)
    scaler = torch.amp.GradScaler("cuda")
    dl, sampler = loader(fit[:64], labels, paths, 24, micro, 0, train=True, world=2, rank=rank)
    sampler.set_epoch(0)
    pos_weight = torch.from_numpy(pw).to(device)
    accumulation = CONFIG["global_batch"]//(2*micro)
    times = []
    torch.cuda.reset_peak_memory_stats(device)
    for step, (x, valid, y, _) in enumerate(dl):
        if step >= max(6, 3*accumulation): break
        torch.cuda.synchronize(device); t = time.monotonic()
        x, valid, y = x.to(device), valid.to(device), y.to(device)
        sync = (step+1) % accumulation == 0
        with net.no_sync() if not sync else contextlib.nullcontext():
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(net(x, valid).float(), y, pos_weight=pos_weight)
            scaler.scale(loss/accumulation).backward()
        if sync:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device); times.append(time.monotonic()-t)
    rec = {"rank": rank, "hardware": torch.cuda.get_device_name(device),
           "peak_gib": torch.cuda.max_memory_allocated(device)/2**30,
           "median_microstep_seconds": float(np.median(times[2:]))}
    write_json(args.output/f"probe_micro{micro}_rank{rank}.json", rec)
    dist.destroy_process_group()


def probe(args):
    if torch.cuda.device_count() != 2:
        raise RuntimeError("probe requires the intended two T4 GPUs")
    results = []
    for micro in (2, 1):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
        try:
            mp.spawn(probe_worker, args=(args, port, micro), nprocs=2, join=True)
        except Exception as exc:
            results.append({"microbatch_per_gpu": micro, "error": f"{type(exc).__name__}: {exc}"})
            continue
        ranks = [json.loads((args.output/f"probe_micro{micro}_rank{r}.json").read_text()) for r in range(2)]
        steady = max(r["median_microstep_seconds"] for r in ranks)
        accum = CONFIG["global_batch"]//(2*micro)
        rec = {"microbatch_per_gpu": micro, "accumulation": accum, "images_per_study": 24,
               "effective_study_batch": CONFIG["global_batch"], "ranks": ranks,
               "peak_gib": max(r["peak_gib"] for r in ranks),
               "steps_per_second": 1/(steady*accum),
               "projected_training_only_hours": 4*(3712//8)*steady*accum/3600,
               "probe_type": "two_gpu_ddp_forward_backward"}
        results.append(rec)
    write_json(args.output/"probe.json", results)
    print(json.dumps(results, indent=2))


def worker(rank, world, args, port):
    if world > 1:
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                                world_size=world, timeout=datetime.timedelta(minutes=30))
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.set_num_threads(2)
    seed = CONFIG["seed"]
    random.seed(seed+rank); np.random.seed(seed+rank); torch.manual_seed(seed+rank); torch.cuda.manual_seed(seed+rank)
    paths, labels, fit, locked, gold, pw = load_data(args)
    model = setup_model(args, device)
    net = DDP(model, device_ids=[rank], find_unused_parameters=False) if world > 1 else model
    opt = optimizer(model)
    accum = CONFIG["global_batch"]//(world*args.microbatch)
    if accum < 1 or accum*world*args.microbatch != CONFIG["global_batch"]:
        raise ValueError("microbatch must preserve effective batch 8")
    train_dl, sampler = loader(fit, labels, paths, 24, args.microbatch, args.workers,
                               train=True, world=world, rank=rank)
    updates = len(train_dl)//accum
    if updates < 1: raise ValueError("empty training epoch")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-5, 3e-5],
                total_steps=CONFIG["epochs"]*updates, pct_start=CONFIG["onecycle_pct_start"])
    scaler = torch.amp.GradScaler("cuda")
    pos_weight = torch.from_numpy(pw).to(device)
    start, best, history = 0, -math.inf, []
    last = args.output/"experiment1_last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck["config_hash"] != sha(args.output/"config.json") or ck["inputs"] != json.loads((args.output/"inputs.json").read_text()):
            raise ValueError("resume config/inputs mismatch")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        restore_rng(ck["rng"][rank]); start = ck["epoch"]+1; best = ck["best"]
        history = ck["history"]
    for ep in range(start, CONFIG["epochs"]):
        if sampler: sampler.set_epoch(ep)
        train_dl.dataset.epoch = ep
        net.train(); opt.zero_grad(set_to_none=True)
        t0 = time.monotonic(); losses = []
        usable = updates*accum
        for step, (x, valid, y, _) in enumerate(train_dl):
            if step >= usable: break
            x, valid, y = x.to(device, non_blocking=True), valid.to(device), y.to(device)
            sync = (step+1)%accum == 0
            with net.no_sync() if world > 1 and not sync else contextlib.nullcontext():
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = net(x, valid)
                    loss = F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pos_weight)
                if not bool(torch.isfinite(loss)): raise FloatingPointError("nonfinite loss")
                scaler.scale(loss/accum).backward()
            losses.append(float(loss.detach()))
            if sync:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                old_scale = scaler.get_scale(); scaler.step(opt); scaler.update()
                if scaler.get_scale() >= old_scale: sched.step()
                opt.zero_grad(set_to_none=True)
        if world > 1: dist.barrier()
        record = {"epoch": ep, "train_loss_rank0": float(np.mean(losses)), "minutes": (time.monotonic()-t0)/60}
        if rank == 0:
            val_dl, _ = loader(locked, labels, paths, 24, 1, args.workers)
            ids, z, y, vm = predict(model, val_dl, device)
            if set(ids) != set(locked): raise ValueError("weak validation UID mismatch")
            p = torch.sigmoid(torch.from_numpy(z)).numpy()
            score = metrics(y, p)
            record["weak_locked"] = score
            save_predictions(args.output/f"weak_locked_epoch{ep}_k24.npz", ids, z, y, vm)
            if score["macro_auc"] > best:
                best = score["macro_auc"]
                torch.save({"model": model.state_dict(), "epoch": ep, "score": score,
                            "arch": ARCH, "lab": LAB, "recipe": {"name": "cache64-maxspan"},
                            "config_hash": sha(args.output/"config.json"),
                            "inputs": json.loads((args.output/"inputs.json").read_text())},
                           args.output/"experiment1_selected.pt")
            history.append(record)
            write_json(args.output/"history.json", history)
            print(f"epoch {ep}: weak_locked macro AUC {score['macro_auc']:.6f}; synovitis {score['per_target']['Synovitis']['auc']:.6f}", flush=True)
        states = [None]*world if rank == 0 else None
        if world > 1: dist.gather_object(rng_state(), states, dst=0)
        else: states = [rng_state()]
        if rank == 0:
            payload = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                       "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                       "rng": states, "epoch": ep, "best": best, "history": history,
                       "config_hash": sha(args.output/"config.json"),
                       "inputs": json.loads((args.output/"inputs.json").read_text())}
            tmp = last.with_suffix(".tmp")
            torch.save(payload, tmp); os.replace(tmp, last)
        if world > 1: dist.barrier()
    if rank == 0:
        selected = torch.load(args.output/"experiment1_selected.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(selected["model"])
        report = {"selected_epoch": selected["epoch"], "weak_locked_selection_score": selected["score"], "scores": {}}
        baseline = setup_model(args, device)
        for name, uids in (("weak_locked", locked), ("gold_reused", gold)):
            for k in (24, 62):
                dl, _ = loader(uids, labels, paths, k, 1, args.workers)
                b_ids, bz, by, bm = predict(baseline, dl, device)
                c_ids, cz, cy, cm = predict(model, dl, device)
                if b_ids != c_ids or not np.array_equal(by, cy): raise ValueError("paired prediction mismatch")
                save_predictions(args.output/f"{name}_arm13_k{k}.npz", b_ids, bz, by, bm)
                save_predictions(args.output/f"{name}_experiment1_k{k}.npz", c_ids, cz, cy, cm)
                report["scores"][f"{name}_k{k}"] = {"arm13": metrics(by, 1/(1+np.exp(-bz))),
                    "experiment1": metrics(cy, 1/(1+np.exp(-cz)))}
        write_json(args.output/"comparison.json", report)
        with (args.output/"experiment_log.jsonl").open("a") as f:
            f.write(json.dumps({"experiment": 1, "status": "trained", "selected_epoch": selected["epoch"],
                "output": str(args.output), "checkpoint_sha256": sha(args.output/"experiment1_selected.pt"),
                "config_sha256": sha(args.output/"config.json"), "inputs": json.loads((args.output/"inputs.json").read_text()),
                "scores": {key: {m: v["macro_auc"] for m,v in val.items()} for key,val in report["scores"].items()}})+"\n")
    if world > 1: dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("probe", "train"), required=True)
    ap.add_argument("--cache", type=Path, required=True, help="directory with both cache64 shards")
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--warm-start", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("/kaggle/working/experiment1"))
    ap.add_argument("--microbatch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {k: sha(getattr(args, k)) for k in ("pool", "gold", "split", "warm_start")}
    inputs.update({p.name: sha(p) for p in sorted(args.cache.glob("*shard*.npy"))})
    inputs["training_code"] = sha(Path(__file__))
    inputs["submission_code"] = sha(Path(__file__).with_name("experiment1_submit.py"))
    inputs["raw_dicom_builder_code"] = sha(Path(__file__).resolve().parents[1]/"cache64"/"build_cache64.py")
    config_path, input_path = args.output/"config.json", args.output/"inputs.json"
    if config_path.exists() and json.loads(config_path.read_text()) != CONFIG: raise ValueError("config changed")
    if input_path.exists() and json.loads(input_path.read_text()) != inputs: raise ValueError("inputs changed")
    write_json(config_path, CONFIG); write_json(input_path, inputs)
    split_copy = args.output/"weak_split.csv"
    if split_copy.exists() and sha(split_copy) != inputs["split"]:
        raise ValueError("saved split IDs changed")
    if not split_copy.exists():
        shutil.copyfile(args.split, split_copy)
    with (args.output/"experiment_log.jsonl").open("a") as f:
        f.write(json.dumps({"experiment": 1, "status": "started", "mode": args.mode,
                            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "config_sha256": sha(config_path), "inputs": inputs})+"\n")
    world = torch.cuda.device_count()
    if world != 2 or any("T4" not in torch.cuda.get_device_name(i) for i in range(world)):
        raise RuntimeError("Experiment 1 requires the intended two-T4 hardware")
    if args.mode == "probe": probe(args); return
    probe_path = args.output/"probe.json"
    if not probe_path.exists(): raise RuntimeError("run --mode probe first")
    matching = [r for r in json.loads(probe_path.read_text()) if r["microbatch_per_gpu"] == args.microbatch]
    if not matching or "peak_gib" not in matching[0] or matching[0]["peak_gib"] >= 16:
        raise RuntimeError("hardware probe did not establish <16 GiB")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    mp.spawn(worker, args=(world, args, port), nprocs=world, join=True)


if __name__ == "__main__": main()
