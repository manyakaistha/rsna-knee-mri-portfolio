"""Standalone Kaggle trainer for Arm C ROI — RadioDINO scratch training.

This is a standalone version of the Arm C ROI trainer that trains from scratch
using a RadioDINO-s16 backbone (ViT-small pretrained on RadImageNet via DINO),
without requiring any Arm 13 checkpoint. All ROI heads, global heads, and
backbone are trained jointly from epoch 0.

Launch it from a notebook cell as a *file on disk*, not as pasted source:

    !python /kaggle/working/arm_training_bundle/train_armc_roi_radiodino_kaggle.py

Multi-GPU needs that path. DDP ranks are spawned as fresh interpreters that
re-import this module, so the module has to exist as a real file. Running it
through ``exec(open(...).read())`` still works, and still finds the file by
basename under /kaggle/working and /kaggle/input, but if it cannot the run
falls back to one GPU and says so loudly. All configuration lives in the
CONFIGURATION block below as plain variables — edit them in the file, since
that on-disk copy is what the spawned ranks read.

The script needs the cache64 ID files (ids_shard0/1.npy) and the validated
gold_test.csv/train_pool.csv label files in attached Kaggle inputs.
Attach either the original shard files named
``roi_<slug>_shard{0,1}.npy`` with matching masks and IDs, or the merged
dataset files ``roi_<slug>.npy``, ``roi_<slug>_masks.npy``, and ``roi_ids.npy``.
Each ROI array is uint8 [N,4,3,336,336], each mask is binary uint8 [N,4].
Merged IDs must exactly equal global shard 0 IDs followed by shard 1 IDs.

This recipe uses RadioDINO-s16 (ViT-small, patch-16, 384-dim embeddings)
pretrained on RadImageNet via self-supervised DINO.  All parameters — backbone,
global head, and ROI heads — are trained jointly from epoch 0.  The ViT-small
backbone is considerably lighter than CoAtNet-2, so the encoder chunk size can
be larger and CHANNELS_LAST is not needed.  Batch-size four per GPU and two
sampled ROI windows per target keep memory within Kaggle GPU limits; evaluation
always uses every valid ROI window.

ROI EXPERT: only the five ROI_TARGETS are learned.  No global slices are
loaded or encoded; the "global" branch is a learned per-label bias, kept so
the W&B global/combined metrics stay comparable with the other arms.  The
seven non-ROI output columns are NaN so they cannot be used by accident.

Windows travel to the GPU as uint8 at cache resolution and are normalised and
resized there. All five fixed-shape ROI bags are packed into one bounded
encoder call; downstream attention masks still remove invalid slots. Loading
therefore moves a fraction of the former float traffic and does no CPU
interpolation, which matters on Kaggle's four vCPUs.

W&B logs global-only and combined metrics, per-target deltas,
gradient/LR diagnostics, and uploads complete resumable checkpoints.
The best validation checkpoint and an uninterrupted-run top-three weight
average are compared on Gold before final selection.
"""

from __future__ import annotations

import os

# These must be set before torch initialises CUDA or NCCL, so they come first.
# device_count() must not initialise CUDA in the parent: a parent that has
# touched CUDA cannot hand working contexts to spawned ranks.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
# NCCL peer-to-peer between cloud T4s can deadlock on the first all-reduce.
# Routing through host memory costs little at this model size.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")

import datetime
import faulthandler
import importlib.util
import inspect
import json
import math
import random
import re
import signal
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm


# ------------------------------ configuration ------------------------------
LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)
ROI_TARGETS = ("MCL", "Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis")
ROI_INDEX = {name: LABELS.index(name) for name in ROI_TARGETS}
RES = 224  # ViT-small native resolution
ARCH = "hf_hub:Snarcy/RadioDino-s16"  # ViT-small patch-16, 384-dim, RadImageNet DINO
PRETRAINED = True  # RadImageNet-pretrained backbone (no Arm 13 warm start)
EPOCHS = 10
ROI_K_TRAIN = 2
BATCH_SIZE = 4  # per GPU (doubled since we dropped global slices)
# One optimizer step per batch.  With two ranks the effective batch is still
# BATCH_SIZE * ACCUM_STEPS * world = 4, and the per-rank loader is half as long,
# so steps_per_epoch is unchanged from the old single-GPU 2x2 setup and the
# OneCycle horizon carries over exactly.
ACCUM_STEPS = 1
# Keep loading inside each DDP rank.  Linux DataLoader workers default to
# ``fork``; starting them after CUDA, NCCL, and W&B have created threads can
# deadlock on the first batch.  Preprocessing is GPU-side, so the remaining
# mmap slicing is cheap and the two ranks still load concurrently.
WORKERS = 0
# All parameters trained jointly from scratch — single LR tier for backbone,
# separate tiers for global head and ROI heads.
BACKBONE_LR = 1e-4
GLOBAL_HEAD_LR = 3e-4
ROI_HEAD_LR = 3e-4
GLOBAL_LOSS_WEIGHT = 0.25
WEIGHT_DECAY = 0.05
CLIP_NORM = 3.0
SEED = 42
OUTPUT_DIR = Path("/kaggle/working/armc_roi_radiodino_expert")
RESUME = "/kaggle/working/armc_roi_radiodino_expert/armc_roi_radiodino_expert_last.pt"
REQUIRE_ROI = True
TOP_K = 3
UPLOAD_EVERY = 2
WANDB_PROJECT = "rsna-knee-armc-roi"
WANDB_ENTITY = "rsna-knee-cap-abnormality" or None
WANDB_RUN_NAME = "armc-roi-radiodino-expert-seed42"
# Leave blank and supply the key through the WANDB_API_KEY environment variable
# (a Kaggle Secret): a key written here travels with every copy of the file.
WANDB_API_KEY = "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"
WANDB_MODE = ""  # e.g. "offline" or "disabled"; leave blank for normal online logging
ARMC_ROI_INPUT = "/kaggle/input"  # root directory to search for attached datasets
# Where the DDP ranks load this trainer from when it was exec()-ed or pasted.
# Blank = search /kaggle/working, then ARMC_ROI_INPUT, for a DDP-capable copy.
SELF_PATH = ""
PROGRESS = True  # tqdm bars for the otherwise silent train/eval passes
PROGRESS_SECONDS = 5.0  # bar refresh floor; keeps notebook logs small in text mode
LOG_EVERY_STEPS = 25  # optimizer steps between step/* W&B points; 0 disables them
WORLD_SIZE = 0  # 0 = every visible GPU; set 1 to force the single-GPU path
DDP_START_METHOD = "spawn"  # fork deadlocks on Kaggle once threads or CUDA exist
DDP_TIMEOUT_HOURS = 2.0
CHANNELS_LAST = False  # ViT has no convolutional stages; NHWC gives no benefit
CUDNN_BENCHMARK = True  # few distinct shapes here, so autotuning pays for itself
# ViT-small supports simple set_grad_checkpointing() without stage-based control.
GRAD_CHECKPOINTING = True
# ViT-small is much lighter than CoAtNet-2; a larger chunk keeps GPU utilisation
# high without approaching T4 memory limits.
ENCODER_CHUNK_SIZE = 48
REFERENCE_STEP_SECONDS = 0.0  # not yet benchmarked with RadioDINO
REFERENCE_PEAK_GPU_GB = 0.0   # not yet benchmarked with RadioDINO
PERF_LOG_EVERY_STEPS = 100
# Seconds without training/eval progress before every rank prints every
# thread's stack. It also signals loader children if workers are enabled in a
# future experiment; the supported DDP configuration creates none. 0 disables.
STALL_DUMP_SECONDS = 300
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Rank of the current process, set once by train_worker.  Only rank 0 prints,
# draws progress bars, talks to W&B, or writes checkpoints; the others would
# otherwise interleave duplicate output into a single Kaggle log stream.
_RANK = 0
_WORLD = 1


def is_main() -> bool:
    return _RANK == 0


def log(message: str) -> None:
    if is_main():
        print(f"[{time.strftime('%H:%M:%S')}] [ArmC ROI RadioDINO] {message}", flush=True)


_last_progress = time.monotonic()
_progress_label = "startup"


def heartbeat(label: str | None = None) -> None:
    """Record forward progress; the stall watchdog measures time since this."""

    global _last_progress, _progress_label
    _last_progress = time.monotonic()
    if label is not None:
        _progress_label = label


def child_pids() -> list[int]:
    """Direct children of this process (its data-loader workers), Linux only."""

    me, found = os.getpid(), []
    for entry in Path("/proc").glob("[0-9]*/stat"):
        try:
            # The command name may contain spaces; ppid is 2nd field after ')'.
            fields = entry.read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) == me:
                found.append(int(entry.parent.name))
        except (OSError, IndexError, ValueError):
            continue
    return found


def start_stall_watchdog(rank: int) -> None:
    """Print every thread's stack once a rank stops making progress.

    Only rank 0 draws progress bars, so a hung rank 1, a blocked input read,
    and a collective waiting on a dead peer all look identical in the log: a
    frozen bar. Phase labels and stacks distinguish them. If a future loader
    experiment enables workers, SIGUSR1 asks those children for stacks too.
    """

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    if STALL_DUMP_SECONDS <= 0:
        return

    def watch() -> None:
        reported = None
        while True:
            time.sleep(10)
            idle = time.monotonic() - _last_progress
            if idle < STALL_DUMP_SECONDS or reported == _last_progress:
                continue
            reported = _last_progress
            # Ranks usually stall together; staggering keeps their dumps from
            # interleaving line by line in the one shared log.
            time.sleep(5 * rank)
            workers = child_pids()
            print(
                f"\n[stall] rank {rank} pid {os.getpid()}: no progress for "
                f"{idle:.0f}s during '{_progress_label}'. Stacks follow, then "
                f"{len(workers)} loader worker(s): {workers}",
                file=sys.stderr,
                flush=True,
            )
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            for pid in workers:
                try:
                    os.kill(pid, signal.SIGUSR1)
                except OSError:
                    pass

    threading.Thread(target=watch, name="stall-watchdog", daemon=True).start()


def worker_init(_worker_id: int) -> None:
    """Let the rank's watchdog pull a stack dump out of this worker."""

    faulthandler.register(signal.SIGUSR1, all_threads=True)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


@lru_cache(maxsize=None)
def find_file(name: str) -> Path | None:
    roots = [Path(ARMC_ROI_INPUT), Path("/kaggle/working")]
    for root in roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.exists():
            return direct
        for directory, subdirectories, filenames in os.walk(root):
            subdirectories[:] = [
                item
                for item in subdirectories
                if item not in {"train_series", "test_series", ".git", "__pycache__"}
            ]
            if name in filenames:
                return Path(directory) / name
    return None


def require_file(name: str) -> Path:
    path = find_file(name)
    if path is None:
        raise FileNotFoundError(
            f"Could not find {name}. Attach the cache64/label Kaggle dataset or set "
            "ARMC_ROI_INPUT variable to its mounted directory."
        )
    return path


def load_index() -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    # Only the cache64 row order is needed: it is the index the ROI arrays are
    # aligned to.  The global volumes and slice masks are never read here.
    shard_ids = (
        np.load(require_file("ids_shard0.npy"), allow_pickle=True).astype(str),
        np.load(require_file("ids_shard1.npy"), allow_pickle=True).astype(str),
    )
    ids = np.concatenate(shard_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("cache64 ids are duplicated")
    return ids, shard_ids


def load_labels(ids: np.ndarray) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    gold_path, pool_path = require_file("gold_test.csv"), require_file("train_pool.csv")
    gold, pool = pd.read_csv(gold_path), pd.read_csv(pool_path)
    required_columns = {"StudyInstanceUID", *LABELS}
    for name, frame in (("gold_test.csv", gold), ("train_pool.csv", pool)):
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(f"{name} is missing columns: {missing_columns}")
        if frame.StudyInstanceUID.astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate StudyInstanceUID rows")
    idset = set(ids.astype(str))
    gold_ids = [str(x) for x in gold.StudyInstanceUID if str(x) in idset]
    if len(gold_ids) != len(gold):
        raise ValueError(
            f"only {len(gold_ids)}/{len(gold)} Gold studies exist in cache64"
        )
    gold_set = set(gold.StudyInstanceUID.astype(str))
    train_ids = [
        str(x)
        for x in pool.StudyInstanceUID
        if str(x) in idset and str(x) not in gold_set
    ]
    labels: dict[str, np.ndarray] = {}
    for frame in (pool, gold):
        for uid, values in zip(
            frame.StudyInstanceUID.astype(str),
            frame[list(LABELS)].to_numpy(np.float32),
        ):
            labels[str(uid)] = np.asarray(values, dtype=np.float32)
    missing = [
        uid
        for uid in train_ids + gold_ids
        if uid not in labels or not np.isfinite(labels[uid]).all()
    ]
    if missing:
        raise ValueError(f"missing/invalid labels for {len(missing)} studies, e.g. {missing[0]}")
    if not train_ids:
        raise ValueError("train_pool.csv did not produce any non-Gold training studies")
    return train_ids, gold_ids, labels


class OptionalROICache:
    """Loads optional per-target ROI arrays, preserving cache row order."""

    def __init__(self, shard_ids: tuple[np.ndarray, np.ndarray]):
        self.n0 = len(shard_ids[0])
        self.merged = find_file("roi_ids.npy") is not None or any(
            find_file(f"roi_{slug(target)}.npy") is not None
            for target in ROI_TARGETS
        )
        self.data: dict[str, tuple[object, object]] = {}
        any_roi_data = any(
            find_file(f"roi_{slug(target)}_shard{shard}.npy") is not None
            for target in ROI_TARGETS
            for shard in (0, 1)
        ) or self.merged
        if any_roi_data or REQUIRE_ROI:
            if self.merged:
                roi_ids_path = find_file("roi_ids.npy")
                if roi_ids_path is None:
                    raise FileNotFoundError("merged ROI data is present but roi_ids.npy is missing")
                expected_ids = np.concatenate(shard_ids)
                roi_ids = np.load(roi_ids_path, allow_pickle=True).astype(str)
                if not np.array_equal(roi_ids, expected_ids):
                    raise ValueError(
                        "roi_ids.npy does not exactly match global shard 0 IDs followed "
                        "by shard 1 IDs; refusing a potentially misaligned cache"
                    )
            else:
                for shard in (0, 1):
                    roi_ids_path = find_file(f"roi_ids_shard{shard}.npy")
                    if roi_ids_path is None:
                        raise FileNotFoundError(
                            f"ROI data is present but roi_ids_shard{shard}.npy is missing"
                        )
                    roi_ids = np.load(roi_ids_path, allow_pickle=True).astype(str)
                    if not np.array_equal(roi_ids, shard_ids[shard]):
                        raise ValueError(
                            f"roi_ids_shard{shard}.npy does not exactly match "
                            f"ids_shard{shard}.npy; refusing a potentially misaligned cache"
                        )
        for target in ROI_TARGETS:
            s = slug(target)
            if self.merged:
                data_path = find_file(f"roi_{s}.npy")
                mask_path = find_file(f"roi_{s}_masks.npy")
                if data_path is None or mask_path is None:
                    if REQUIRE_ROI:
                        raise FileNotFoundError(
                            f"merged ROI cache is required for {target}; missing "
                            f"roi_{s}.npy or roi_{s}_masks.npy"
                        )
                    log(f"merged ROI cache not found for {target}; global-only logits.")
                    continue
                array = np.load(data_path, mmap_mode="r")
                mask = np.load(mask_path, mmap_mode="r")
                if array.shape != (self.n0 + len(shard_ids[1]), 4, 3, 336, 336) or array.dtype != np.uint8:
                    raise ValueError(f"merged ROI {target} must be uint8 [N,4,3,336,336]")
                if mask.shape != (len(array), 4) or mask.dtype != np.uint8 or not np.isin(np.unique(mask), [0, 1]).all():
                    raise ValueError(f"merged ROI {target} mask must be binary uint8 [N,4]")
                self.data[target] = (array, mask.astype(bool))
                continue
            files = [find_file(f"roi_{s}_shard{i}.npy") for i in (0, 1)]
            mask_files = [find_file(f"roi_{s}_masks_shard{i}.npy") for i in (0, 1)]
            if any(x is None for x in files + mask_files):
                if REQUIRE_ROI:
                    expected = [
                        f"roi_{s}_shard{i}.npy" for i in (0, 1)
                    ] + [
                        f"roi_{s}_masks_shard{i}.npy" for i in (0, 1)
                    ]
                    missing = [
                        name
                        for name, path in zip(expected, files + mask_files)
                        if path is None
                    ]
                    raise FileNotFoundError(
                        f"ROI cache is required for {target}; missing {missing}"
                    )
                log(
                    f"ROI cache not found for {target}; "
                    "this target uses global-only logits."
                )
                continue
            arrays = [np.load(x, mmap_mode="r") for x in files if x is not None]
            masks = [np.load(x, mmap_mode="r") for x in mask_files if x is not None]
            row_counts_match = (
                arrays[0].shape[0] == len(shard_ids[0])
                and arrays[1].shape[0] == len(shard_ids[1])
            )
            if not row_counts_match:
                raise ValueError(f"ROI {target} row count does not match cache64 ids")
            expected_tail = (4, 3, 336, 336)
            invalid_array = any(
                array.shape[1:] != expected_tail or array.dtype != np.uint8
                for array in arrays
            )
            if invalid_array:
                raise ValueError(
                    f"ROI {target} must be uint8 [N,4,3,336,336]; "
                    f"received {[array.shape for array in arrays]}"
                )
            if any(
                mask.shape != (len(shard_ids[index]), 4)
                or mask.dtype != np.uint8
                or not np.isin(np.unique(mask), [0, 1]).all()
                for index, mask in enumerate(masks)
            ):
                raise ValueError(f"ROI {target} masks must be binary uint8 [N,4]")
            self.data[target] = (
                [arrays[0], arrays[1]],
                [masks[0].astype(bool), masks[1].astype(bool)],
            )

    def get(self, target: str, row: int) -> tuple[np.ndarray, np.ndarray] | None:
        if target not in self.data:
            return None
        arrays, masks = self.data[target]
        # A read-only mmap view, deliberately not copied here: the caller picks
        # its windows first and only those get faulted in and copied.  Copying
        # all four windows and then discarding half of them read twice the
        # bytes it needed to.
        if self.merged:
            return arrays[row], masks[row]
        shard = 0 if row < self.n0 else 1
        offset = row if shard == 0 else row - self.n0
        return arrays[shard][offset], masks[shard][offset]


class StudyDataset(Dataset):
    def __init__(self, uids, id2row, labels, train, roi_cache):
        self.uids, self.id2row = list(uids), id2row
        self.labels, self.train, self.roi_cache = labels, train, roi_cache

    def __len__(self):
        return len(self.uids)

    def _select_roi_windows(
        self,
        array: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample real ROI windows without duplicating anatomy.

        Training uses a fixed-size subset for predictable GPU memory.  If a
        study has fewer valid windows than requested, unused slots remain
        masked rather than repeating a slice.  Evaluation retains all four.

        Always returns owned, writable arrays: the input is an mmap view, and
        torch.from_numpy warns on a read-only buffer.  Fancy indexing already
        copies, so only the keep-everything path needs an explicit one.
        """

        if not self.train or ROI_K_TRAIN >= len(mask):
            return np.array(array), np.array(mask)
        valid = np.flatnonzero(mask)
        invalid = np.flatnonzero(~mask)
        if len(valid) > ROI_K_TRAIN:
            chosen = np.asarray(
                random.sample(valid.tolist(), ROI_K_TRAIN),
                dtype=np.int64,
            )
        else:
            needed = ROI_K_TRAIN - len(valid)
            chosen = np.concatenate([valid, invalid[:needed]]).astype(np.int64)
        return array[chosen], mask[chosen]

    def __getitem__(self, index):
        uid = self.uids[index]
        row = self.id2row[uid]
        # ROI EXPERT: no global slices are loaded; only the ROI windows.
        roi_x, roi_mask = {}, {}
        for target in ROI_TARGETS:
            item = self.roi_cache.get(target, row)
            if item is None:
                continue
            arr, mask = self._select_roi_windows(*item)
            roi_x[target] = torch.from_numpy(arr)
            roi_mask[target] = torch.from_numpy(mask.astype(bool))
        return roi_x, roi_mask, torch.from_numpy(self.labels[uid])


def collate(batch):
    roi_x, roi_mask = {}, {}
    for target in ROI_TARGETS:
        if all(target in x[0] for x in batch):
            roi_x[target] = torch.stack([x[0][target] for x in batch])
            roi_mask[target] = torch.stack([x[1][target] for x in batch])
    labels = torch.stack([x[2] for x in batch])
    return roi_x, roi_mask, labels


# ---------------------------------- model ----------------------------------
def build_backbone():
    if ARCH == "tiny_test":
        net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        net.num_features = 16
        return net
    import timm
    # RadioDINO is a DINO ViT: its checkpoint carries the final ``norm`` and a
    # trained CLS token, and no ``fc_norm``.  global_pool="avg" would make timm
    # swap ``norm`` for an ``fc_norm`` layer and the strict weight load fails,
    # so keep the checkpoint's own CLS-token pooling.
    net = timm.create_model(
        ARCH,
        pretrained=PRETRAINED,
        num_classes=0,
        in_chans=3,
        global_pool="token",
    )
    # Left off here and toggled per phase by set_training_phase.
    if hasattr(net, "set_grad_checkpointing"):
        net.set_grad_checkpointing(False)
    return net


def set_grad_checkpointing(model: nn.Module, enabled: bool) -> None:
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "set_grad_checkpointing"):
        backbone.set_grad_checkpointing(enabled)


def masked_softmax(scores, mask):
    mask = mask.to(device=scores.device, dtype=torch.bool)
    masked_scores = scores.float().masked_fill(
        ~mask,
        torch.finfo(torch.float32).min,
    )
    weights = torch.softmax(masked_scores, dim=1).to(scores.dtype)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(1, keepdim=True).clamp_min(1e-12)


class ROIPooler(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.norm = nn.LayerNorm(f)
        self.value, self.gate = nn.Linear(f, 256), nn.Linear(f, 256)
        self.score, self.readout = nn.Linear(256, 1), nn.Linear(f, 1)
        # A zero readout makes every new residual zero while still allowing
        # the ROI head to begin learning on its first optimization step.
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, h, mask):
        mask = mask.to(device=h.device, dtype=torch.bool)
        h = self.norm(h)
        scores = self.score(torch.tanh(self.value(h)) * torch.sigmoid(self.gate(h))).squeeze(-1)
        a = masked_softmax(scores, mask)
        residual = self.readout(torch.einsum("bk,bkf->bf", a, h)).squeeze(-1)
        return residual * mask.any(1).to(residual.dtype)


class ArmC(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = build_backbone()
        f = int(self.backbone.num_features)
        self.global_bias = nn.Parameter(torch.zeros(12))
        self.roi_poolers = nn.ModuleDict({t: ROIPooler(f) for t in ROI_TARGETS})
        self.beta = nn.ParameterDict({t: nn.Parameter(torch.tensor(2.0)) for t in ROI_TARGETS})
        # Buffers rather than globals: they follow .to(device) and are already
        # in the right place and dtype when a rank's forward runs.
        self.register_buffer("pixel_mean", IMAGENET_MEAN.clone(), persistent=False)
        self.register_buffer("pixel_std", IMAGENET_STD.clone(), persistent=False)
        roi_columns = torch.zeros(len(LABELS), dtype=torch.bool)
        roi_columns[[ROI_INDEX[t] for t in ROI_TARGETS]] = True
        self.register_buffer("roi_columns", roi_columns, persistent=False)

    def prepare(self, images):
        """uint8 [n,3,h,w] at cache resolution -> normalised float at RES.

        Bags stay fixed-shape through the packed encoder. Invalid cache slots
        are deliberately processed and removed by the downstream attention
        mask; this avoids five device synchronizations and tiny launches.
        """

        images = images.float().div_(255.0)
        images = (images - self.pixel_mean) / self.pixel_std
        if images.shape[-1] != RES:
            images = F.interpolate(
                images,
                size=(RES, RES),
                mode="bilinear",
                align_corners=False,
            )
        if CHANNELS_LAST:
            images = images.contiguous(memory_format=torch.channels_last)
        return images

    def backbone_forward(self, images):
        """Encode a bounded microbatch, preserving the proven 24-image cap."""

        chunks = images.split(ENCODER_CHUNK_SIZE, dim=0)
        outputs = [self.backbone(self.prepare(chunk)).float() for chunk in chunks]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    def encode(self, x):
        """Encode a fixed-shape bag without GPU-to-CPU mask synchronizations."""

        b, k = x.shape[:2]
        return self.backbone_forward(x.flatten(0, 1)).view(b, k, -1)

    def encode_roi_bags(self, roi_x):
        """Encode every target's ROI windows in one shared backbone call."""

        names = [target for target in ROI_TARGETS if target in roi_x]
        if not names:
            return {}
        counts = [roi_x[target].shape[1] for target in names]
        packed = torch.cat([roi_x[target] for target in names], dim=1)
        packed_features = self.encode(packed)
        return dict(zip(names, packed_features.split(counts, dim=1)))

    def forward(self, global_x=None, global_mask=None, roi_x=None, roi_mask=None):
        """Always returns (combined_logits, global_logits), both [B, 12].

        ROI EXPERT: global_x/global_mask are accepted for call compatibility
        with the other arms and ignored.  The global logit is the learned
        per-label bias; ROI residuals are added on top of it.  Only the five
        ROI_TARGETS columns are trained, so the other seven are NaN: anything
        that consumes them (an ensemble, an AUC, a submission) fails loudly
        instead of silently using an untrained constant.
        """
        roi_x, roi_mask = roi_x or {}, roi_mask or {}
        missing = [t for t in ROI_TARGETS if t not in roi_x or t not in roi_mask]
        if missing:
            raise ValueError(f"ROI expert needs ROI windows for every target; missing {missing}")
        B = roi_x[ROI_TARGETS[0]].shape[0]

        global_logits = self.global_bias.unsqueeze(0).expand(B, -1).clone()
        logits = global_logits.clone()

        roi_features_by_target = self.encode_roi_bags(roi_x)
        for target in ROI_TARGETS:
            residual = self.roi_poolers[target](roi_features_by_target[target], roi_mask[target])
            target_index = ROI_INDEX[target]
            logits[:, target_index] = (
                logits[:, target_index]
                + torch.sigmoid(self.beta[target]) * residual
            )
        # torch.where passes no gradient to the NaN branch, so training is
        # unaffected: the losses only ever select the ROI columns.
        untrained = torch.full_like(logits, float("nan"))
        return (
            torch.where(self.roi_columns, logits, untrained),
            torch.where(self.roi_columns, global_logits, untrained),
        )


# -------------------------------- training ---------------------------------
def macro_auc(y, p):
    values = {}
    for j, name in enumerate(LABELS):
        if name not in ROI_TARGETS:
            continue
        truth = (y[:, j] > 0.5).astype(np.int64)
        if len(np.unique(truth)) > 1:
            values[name] = float(roc_auc_score(truth, p[:, j]))
    missing = sorted(set(ROI_TARGETS) - set(values))
    if missing:
        raise ValueError(f"Gold split cannot compute AUC for ROI targets: {missing}")
    return float(np.mean(list(values.values()))), values


def autocast_context(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")


def make_scaler(device):
    try:
        return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def progress(iterable, desc: str, leave: bool = True):
    """Wrap a loader in a throttled tqdm bar, or pass it straight through.

    Both passes are otherwise silent for minutes at a time, so the bar is the
    only way to tell a slow epoch from a hang.  ``mininterval`` keeps the
    notebook log short when tqdm falls back to plain text.
    """

    if not PROGRESS or not is_main():
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        mininterval=PROGRESS_SECONDS,
    )


def gather_rows(arrays: list[np.ndarray]) -> np.ndarray:
    """Concatenate this rank's rows, then every rank's, in rank order.

    Ranks evaluate disjoint strides of the Gold split, so each one alone sees
    a different label mix and would compute a different AUC.  Gathering in a
    fixed order leaves every rank holding the identical matrix, which is what
    lets the pre-epoch-0 equality check and the selection comparisons agree
    across ranks without any of them having to broadcast a decision.
    """

    local = (
        np.concatenate(arrays)
        if arrays
        else np.zeros((0, len(LABELS)), dtype=np.float32)
    )
    if _WORLD == 1:
        return local
    gathered: list[Any] = [None] * _WORLD
    dist.all_gather_object(gathered, local)
    return np.concatenate(gathered)


@torch.no_grad()
def evaluate(model, loader, device, desc="eval"):
    model.eval()
    predictions, global_predictions, truths = [], [], []
    total_batches = len(loader)
    iterator = iter(loader)
    batches = progress(range(total_batches), desc, leave=False)
    for batch_index in batches:
        heartbeat(f"{desc}: waiting for loader batch {batch_index + 1}/{total_batches}")
        rx, rm, y = next(iterator)
        heartbeat(f"{desc}: transferring batch {batch_index + 1}/{total_batches}")
        roi_x = {k: v.to(device, non_blocking=True) for k, v in rx.items()}
        roi_mask = {k: v.to(device, non_blocking=True) for k, v in rm.items()}
        heartbeat(f"{desc}: forward batch {batch_index + 1}/{total_batches}")
        with autocast_context(device):
            logits, global_logits = model(roi_x=roi_x, roi_mask=roi_mask)
            p = torch.sigmoid(logits).float()
            global_p = torch.sigmoid(global_logits).float()
        predictions.append(p.cpu().numpy())
        global_predictions.append(global_p.cpu().numpy())
        truths.append(y.numpy())
    heartbeat(f"{desc}: gathering predictions across ranks")
    p, y = gather_rows(predictions), gather_rows(truths)
    global_p = gather_rows(global_predictions)
    combined_auc, combined_by_target = macro_auc(y, p)
    global_auc, global_by_target = macro_auc(y, global_p)
    return {
        "combined_auc": combined_auc,
        "combined_by_target": combined_by_target,
        "global_auc": global_auc,
        "global_by_target": global_by_target,
    }


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, history, config, metrics):
    return {
        "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
        "config": config, "metrics": metrics, "labels": LABELS, "roi_targets": ROI_TARGETS,
        "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def log_checkpoint(run, payload, epoch, metrics, final=False, extra_files=()):
    name = f"armc_roi_radiodino_expert_checkpoint_ep{epoch:03d}.pt"
    path = OUTPUT_DIR / name
    torch.save(payload, path)
    artifact = wandb.Artifact(
        name=f"armc-roi-radiodino-expert-{run.id}", type="model",
        metadata={
            "epoch": epoch,
            "final": final,
            **{
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            },
        },
    )
    artifact.add_file(str(path), name=name)
    for extra in extra_files:
        extra = Path(extra)
        artifact.add_file(str(extra), name=extra.name)
    run.log_artifact(artifact, aliases=["latest", "final" if final else f"epoch-{epoch}"])
    log(f"uploaded W&B artifact {name}")


def set_training_phase(model: ArmC) -> str:
    """Scratch training: everything is always trainable (joint phase)."""

    phase = "joint"
    for _name, parameter in model.named_parameters():
        parameter.requires_grad = True
    
    set_grad_checkpointing(model, GRAD_CHECKPOINTING)
    return phase


def wrap_for_training(model: ArmC, rank: int, world: int, device: torch.device) -> nn.Module:
    """Wrap the model for the current phase's trainable parameter set.

    ``static_graph=True`` is what makes DDP and gradient checkpointing coexist:
    without it the reducer sees each checkpointed parameter marked ready twice
    and aborts.  It also lets DDP skip the unused-parameter search entirely,
    which is why ``encode`` guarantees the backbone runs on every batch.
    """

    if world == 1:
        return model
    kwargs: dict[str, Any] = {
        "device_ids": [rank] if device.type == "cuda" else None,
        "output_device": rank if device.type == "cuda" else None,
        "static_graph": True,
        "gradient_as_bucket_view": True,
    }
    if "batched_grad_copy" in inspect.signature(DDP).parameters:
        kwargs["batched_grad_copy"] = True
    return DDP(model, **kwargs)


def make_optimizer(model: ArmC, device: torch.device) -> torch.optim.Optimizer:
    backbone_params = [
        param for name, param in model.named_parameters() if name.startswith("backbone.")
    ]
    global_head_params = [
        param
        for name, param in model.named_parameters()
        if name == "global_bias"
    ]
    roi_params = [
        param
        for name, param in model.named_parameters()
        if name.startswith(("roi_poolers.", "beta."))
    ]
    groups = [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": global_head_params, "lr": GLOBAL_HEAD_LR},
        {"params": roi_params, "lr": ROI_HEAD_LR},
    ]
    # Standard AdamW for resume compatibility.
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def clip_gradient_norm(model: nn.Module) -> torch.Tensor:
    """Use foreach clipping when available, with a compatibility fallback."""

    parameters = tuple(model.parameters())
    try:
        return torch.nn.utils.clip_grad_norm_(
            parameters,
            CLIP_NORM,
            foreach=True,
        )
    except (RuntimeError, TypeError):
        return torch.nn.utils.clip_grad_norm_(parameters, CLIP_NORM)


def restore_rng_state(state: dict[str, Any]) -> None:
    if state.get("rng_python") is not None:
        random.setstate(state["rng_python"])
    if state.get("rng_numpy") is not None:
        np.random.set_state(state["rng_numpy"])
    if state.get("rng_torch") is not None:
        # The resumable checkpoint is loaded with map_location=device so its
        # RNG byte tensors may arrive on CUDA. PyTorch's RNG setters require
        # CPU ByteTensors even when restoring a CUDA generator.
        torch.set_rng_state(state["rng_torch"].detach().cpu())
    if torch.cuda.is_available() and state.get("rng_cuda") is not None:
        cuda_states = state["rng_cuda"]
        state_index = min(_RANK, len(cuda_states) - 1)
        torch.cuda.set_rng_state(
            cuda_states[state_index].detach().cpu(),
            device=torch.cuda.current_device(),
        )


def candidate_path(epoch: int) -> Path:
    return OUTPUT_DIR / f"armc_roi_radiodino_expert_candidate_ep{epoch:03d}.pt"


def save_top_candidate(model: nn.Module, epoch: int, auc: float, history) -> list[Path]:
    """Keep model-only files for the best validation epochs in this run.

    Every rank calls this and every rank gets the same list back, but only
    rank 0 writes. The barrier is why the others can safely stat the files:
    all ranks hold identical history, so the ranking itself needs no exchange.
    """

    ranked = sorted(history, key=lambda item: item["gold/macro_auc"], reverse=True)[:TOP_K]
    keep_epochs = {int(item["epoch"]) for item in ranked}
    if is_main():
        if epoch in keep_epochs:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "gold_macro_auc": auc},
                candidate_path(epoch),
            )
        for path in OUTPUT_DIR.glob("armc_roi_radiodino_expert_candidate_ep*.pt"):
            match = re.search(r"ep(\d+)\.pt$", path.name)
            if match and int(match.group(1)) not in keep_epochs:
                path.unlink()
    barrier()
    return [
        candidate_path(int(item["epoch"]))
        for item in ranked
        if candidate_path(int(item["epoch"])).exists()
    ]


def average_candidates(paths: list[Path]) -> dict[str, torch.Tensor]:
    if not paths:
        raise ValueError("no candidate checkpoints are available for averaging")
    states = [
        torch.load(path, map_location="cpu", weights_only=False)["model"]
        for path in paths
    ]
    averaged = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            value = first.float().clone()
            for state in states[1:]:
                value.add_(state[key].float())
            averaged[key] = value.div_(len(states)).to(first.dtype)
        else:
            averaged[key] = first.clone()
    return averaged


def init_wandb(config):
    """Authenticate without an interactive prompt in a Kaggle cell."""
    if WANDB_MODE:
        os.environ["WANDB_MODE"] = WANDB_MODE
    if WANDB_API_KEY:
        # Compatibility only; the checked-in default is empty. Supplying a
        # literal here makes W&B correctly emit its explicit-key warning.
        wandb.login(key=WANDB_API_KEY, relogin=True)
    elif os.environ.get("WANDB_API_KEY"):
        # Let W&B read the inherited Kaggle Secret itself. Passing the same
        # value as key= would misleadingly report that it was embedded in code.
        wandb.login()
    elif WANDB_MODE.lower() not in {"offline", "disabled"}:
        # Anonymous runs still support metric and artifact uploads. A configured
        # WANDB_API_KEY is recommended when the run must appear in a team project.
        wandb.login(anonymous="allow")
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config=config,
        job_type="train",
    )


class NullRun:
    """Stand-in for the W&B run on every rank except zero.

    All ranks execute the identical loop; only rank 0 is allowed to talk to
    W&B. A silent stub keeps rank tests out of the training code itself.
    """

    id = "local"

    def __init__(self):
        self.summary: dict[str, Any] = {}

    def define_metric(self, *args, **kwargs) -> None:
        pass

    def log(self, *args, **kwargs) -> None:
        pass

    def finish(self, *args, **kwargs) -> None:
        pass


def barrier(label: str = "waiting for rank 0 to finish checkpoint/W&B work") -> None:
    if _WORLD > 1:
        heartbeat(f"barrier: {label}")
        dist.barrier()
        heartbeat("setup")


def reduce_mean(value: float, device: torch.device) -> float:
    """Average a scalar over ranks so logged loss describes the real batch."""

    if _WORLD == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / _WORLD)


def make_loader_kwargs(device: torch.device) -> dict[str, Any]:
    """Build safe loader options for the current in-rank loading policy."""

    kwargs: dict[str, Any] = {
        "num_workers": WORKERS,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
    }
    # These options are invalid or meaningless with zero workers.  Keeping
    # them conditional also makes the no-child-process invariant obvious in
    # logs and regression tests.
    if WORKERS > 0:
        kwargs.update(
            persistent_workers=True,
            worker_init_fn=worker_init,
        )
    return kwargs


def validate_config() -> None:
    if ACCUM_STEPS < 1 or ROI_K_TRAIN < 1 or TOP_K < 1 or UPLOAD_EVERY < 1:
        raise ValueError("accumulation, ROI sampling, top-k, and upload cadence must be positive")
    if ENCODER_CHUNK_SIZE < 1:
        raise ValueError("ENCODER_CHUNK_SIZE must be positive")
    if not PRETRAINED:
        raise ValueError(
            "Scratch training requires PRETRAINED=True so the backbone starts "
            "from RadImageNet-pretrained RadioDINO weights."
        )


def train_worker(rank: int, world: int, port: int) -> None:
    global _RANK, _WORLD

    _RANK, _WORLD = rank, world
    # A 4-vCPU box shared by two ranks: letting each rank open an intra-op pool
    # per core oversubscribes it badly even without loader subprocesses.
    torch.set_num_threads(2)
    start_stall_watchdog(rank)
    # Bind the GPU *before* the process group exists. NCCL builds its
    # communicator lazily on the first collective and otherwise has to guess
    # which device a rank owns; both ranks landing on GPU 0 hangs that first
    # collective forever. device_id makes the binding explicit and the
    # communicator eager, so a broken NCCL setup fails here, not mid-epoch.
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    if world > 1:
        heartbeat("process-group init")
        group_kwargs: dict[str, Any] = {}
        if device.type == "cuda":
            group_kwargs["device_id"] = device
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world,
            timeout=datetime.timedelta(hours=DDP_TIMEOUT_HOURS),
            **group_kwargs,
        )
        # One real collective now, while nothing else is running: if the
        # transport is broken this is the line the log stops after.
        log(f"process group up ({dist.get_backend()}); testing a collective")
        probe = torch.ones(1, device=device)
        dist.all_reduce(probe)
        if int(probe.item()) != world:
            raise RuntimeError(f"all_reduce returned {probe.item()}, expected {world}")
        heartbeat("setup")
        log("collective ok: all ranks connected")
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    # Identical weights on every rank, but different augmentation draws: the
    # per-rank offset is what stops two ranks sampling the same windows.
    set_seed(SEED)
    random.seed(SEED + rank)
    np.random.seed(SEED + rank)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validate_config()
    ids, shard_ids = load_index()
    train_ids, gold_ids, labels = load_labels(ids)
    roi_cache = OptionalROICache(shard_ids)
    missing_roi = sorted(set(ROI_TARGETS) - set(roi_cache.data))
    if missing_roi:
        raise ValueError(f"required ROI targets are unavailable: {missing_roi}")
    id2row = {uid: i for i, uid in enumerate(ids)}
    train_ds = StudyDataset(train_ids, id2row, labels, True, roi_cache)
    # Each rank scores a disjoint stride of Gold and the predictions are
    # gathered before any AUC is computed, so evaluation cost halves on two
    # GPUs while every rank still sees the whole split's score.
    rank_gold_ids = gold_ids[rank::world]
    gold_ds = StudyDataset(rank_gold_ids, id2row, labels, False, roi_cache)
    loader_kwargs = make_loader_kwargs(device)
    train_sampler = (
        DistributedSampler(train_ds, world, rank, shuffle=True, seed=SEED, drop_last=True)
        if world > 1
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **loader_kwargs,
    )
    gold_loader = DataLoader(
        gold_ds,
        batch_size=1,
        shuffle=False,
        **loader_kwargs,
    )
    model = ArmC().to(device)
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    # No Arm 13 initialization — scratch training from RadImageNet-pretrained backbone.
    log(f"scratch training: backbone pretrained={PRETRAINED}, no warm-start checkpoint")
    # Wrap once for DDP — all parameters are always trainable in scratch mode.
    set_training_phase(model)
    training_model = wrap_for_training(model, rank, world, device)
    optimizer = make_optimizer(model, device)
    steps_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[BACKBONE_LR, GLOBAL_HEAD_LR, ROI_HEAD_LR],
        total_steps=EPOCHS * steps_per_epoch,
        pct_start=0.3,
    )
    scaler = make_scaler(device)
    prevalence = np.clip(
        np.stack([labels[uid] for uid in train_ids]).mean(0),
        0.03,
        0.7,
    )
    pos_weight = torch.tensor(
        np.clip(np.sqrt((1 - prevalence) / prevalence), 1, 3),
        dtype=torch.float32,
        device=device,
    )
    config = {
        "arch": ARCH,
        "resolution": RES,
        "pretrained": PRETRAINED,
        "epochs": EPOCHS,
        # Only these columns are trained; the other seven outputs are NaN.
        "output_labels": list(ROI_TARGETS),
        "roi_windows_train_per_target": ROI_K_TRAIN,
        "roi_windows_eval_per_target": 4,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUM_STEPS,
        "world_size": world,
        "effective_batch_size": BATCH_SIZE * ACCUM_STEPS * world,
        "visible_gpus": torch.cuda.device_count(),
        "channels_last": CHANNELS_LAST,
        "grad_checkpointing": GRAD_CHECKPOINTING,
        "compiled_backbone": False,
        "encoder_chunk_size": ENCODER_CHUNK_SIZE,
        "performance_reference": {
            "step_seconds": REFERENCE_STEP_SECONDS,
            "peak_gpu_gb": REFERENCE_PEAK_GPU_GB,
            "hardware": "2x Kaggle T4",
        },
        "workers": WORKERS,
        "backbone_lr": BACKBONE_LR,
        "global_head_lr": GLOBAL_HEAD_LR,
        "roi_head_lr": ROI_HEAD_LR,
        "global_loss_weight": GLOBAL_LOSS_WEIGHT,
        "weight_decay": WEIGHT_DECAY,
        "optimizer_implementation": "adamw_standard_resume_safe",
        "seed": SEED,
        "required_roi": REQUIRE_ROI,
        "initialization": "scratch (RadImageNet DINO-pretrained backbone)",
        "training_mode": "scratch_joint",
    }
    run = init_wandb(config) if is_main() else NullRun()
    start_epoch, history = 0, []
    resume_path = Path(RESUME) if RESUME else None
    if resume_path and not resume_path.exists():
        resume_path = find_file(RESUME)
    if RESUME and resume_path is None:
        # Unlike the warm-start version, not finding a resume checkpoint is
        # fine for scratch training — just start from epoch 0.
        log(f"RESUME path not found ({RESUME}); starting from scratch")
        resume_path = None
    if resume_path is not None:
        state = torch.load(
            resume_path,
            # Keep checkpoint metadata and RNG states on CPU. load_state_dict
            # copies model/optimizer tensors to their parameter devices and
            # avoids retaining a second checkpoint-sized CUDA allocation.
            map_location="cpu",
            weights_only=False,
        )
        saved_config = state.get("config", {})
        resume_contract = (
            "arch",
            "resolution",
            "epochs",
            "roi_windows_train_per_target",
            "batch_size",
            "accumulation_steps",
            # steps_per_epoch, and therefore the whole OneCycle horizon, is a
            # function of how many ranks split the training set.
            "world_size",
            "backbone_lr",
            "global_head_lr",
            "roi_head_lr",
        )
        mismatches = {
            key: (saved_config.get(key), config[key])
            for key in resume_contract
            if saved_config.get(key) != config[key]
        }
        if mismatches:
            raise ValueError(f"resume configuration mismatch: {mismatches}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        history = state.get("history", [])
        restore_rng_state(state)
        del state
        log(f"resumed from epoch {start_epoch - 1}")
        # Re-wrap DDP after loading state (model weights changed).
        training_model = wrap_for_training(model, rank, world, device)
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("gold/*", step_metric="epoch")
    run.define_metric("system/*", step_metric="epoch")
    run.define_metric("model/*", step_metric="epoch")
    # step/* has its own x-axis: these points arrive between epoch rows.
    run.define_metric("step/global_step")
    run.define_metric("step/*", step_metric="step/global_step")
    run.summary.update(
        {
            "performance/compiled_backbone": False,
            "performance/reference_step_seconds": REFERENCE_STEP_SECONDS,
            "performance/reference_peak_gpu_gb": REFERENCE_PEAK_GPU_GB,
        }
    )
    log(
        f"{world} rank(s) | {len(train_ids)} train | {len(gold_ids)} gold "
        f"| {len(train_loader)} batches/rank/epoch | {steps_per_epoch} steps/epoch"
    )
    if start_epoch == 0:
        barrier("synchronizing ranks before scratch baseline")
        baseline = evaluate(model, gold_loader, device, desc="scratch baseline")
        baseline_metrics = {
            "epoch": -1,
            "gold/macro_auc": baseline["combined_auc"],
            "gold/global_macro_auc": baseline["global_auc"],
        }
        baseline_metrics.update(
            {
                f"gold/{slug(name)}_auc": value
                for name, value in baseline["combined_by_target"].items()
            }
        )
        run.log(baseline_metrics)
        log(f"scratch baseline macro AUC: {baseline['combined_auc']:.6f}")
    roi_columns = torch.tensor(
        [ROI_INDEX[name] for name in ROI_TARGETS],
        dtype=torch.long,
        device=device,
    )
    for epoch in range(start_epoch, EPOCHS):
        started = time.time()
        phase = set_training_phase(model)
        training_model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running = torch.zeros((), dtype=torch.float32, device=device)
        window_loss = torch.zeros((), dtype=torch.float32, device=device)
        count, window_count, optimizer_steps = 0, 0, 0
        grad_norm = torch.full((), float("nan"), device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        total_batches = len(train_loader)
        iterator = iter(train_loader)
        batches = progress(range(total_batches), f"epoch {epoch} {phase}")
        for batch_index in batches:
            measure_timing = (
                device.type == "cuda"
                and PERF_LOG_EVERY_STEPS > 0
                and (batch_index + 1) % PERF_LOG_EVERY_STEPS == 0
            )
            data_started = time.perf_counter() if measure_timing else 0.0
            heartbeat(
                f"epoch {epoch} {phase}: waiting for loader batch "
                f"{batch_index + 1}/{total_batches}"
            )
            rx, rm, y = next(iterator)
            data_wait_ms = (
                (time.perf_counter() - data_started) * 1000.0
                if measure_timing
                else 0.0
            )
            timing_events = None
            if measure_timing:
                timing_events = {
                    name: torch.cuda.Event(enable_timing=True)
                    for name in ("transfer_start", "transfer_end", "backward_end", "step_end")
                }
                timing_events["transfer_start"].record()
            heartbeat(
                f"epoch {epoch} {phase}: transferring batch "
                f"{batch_index + 1}/{total_batches}"
            )
            y = y.to(device, non_blocking=True)
            rx = {
                key: value.to(device, non_blocking=True)
                for key, value in rx.items()
            }
            rm = {
                key: value.to(device, non_blocking=True)
                for key, value in rm.items()
            }
            if timing_events is not None:
                timing_events["transfer_end"].record()
            heartbeat(
                f"epoch {epoch} {phase}: forward/backward batch "
                f"{batch_index + 1}/{total_batches}"
            )
            with autocast_context(device):
                logits, global_logits = training_model(roi_x=rx, roi_mask=rm)
                # Scratch training always uses combined loss (no frozen phase).
                combined_loss = F.binary_cross_entropy_with_logits(
                    logits.index_select(1, roi_columns).float(),
                    y.index_select(1, roi_columns).float(),
                    pos_weight=pos_weight.index_select(0, roi_columns),
                )
                global_roi_loss = F.binary_cross_entropy_with_logits(
                    global_logits.index_select(1, roi_columns).float(),
                    y.index_select(1, roi_columns).float(),
                    pos_weight=pos_weight.index_select(0, roi_columns),
                )
                loss = combined_loss + GLOBAL_LOSS_WEIGHT * global_roi_loss
            group_start = (batch_index // ACCUM_STEPS) * ACCUM_STEPS
            accumulation_divisor = min(
                ACCUM_STEPS,
                len(train_loader) - group_start,
            )
            scaler.scale(loss / accumulation_divisor).backward()
            if timing_events is not None:
                timing_events["backward_end"].record()
            should_step = (
                (batch_index + 1) % ACCUM_STEPS == 0
                or batch_index + 1 == len(train_loader)
            )
            running.add_(loss.detach())
            window_loss.add_(loss.detach())
            count += 1
            window_count += 1
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = clip_gradient_norm(model)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
                if timing_events is not None:
                    timing_events["step_end"].record()
                # A window mean, not the epoch-to-date mean: a late divergence
                # stays visible instead of being averaged away.
                if LOG_EVERY_STEPS and optimizer_steps % LOG_EVERY_STEPS == 0:
                    if is_main():
                        step_loss = float((window_loss / max(window_count, 1)).item())
                        step_grad_norm = float(grad_norm.detach().item())
                        run.log(
                            {
                                "step/global_step": epoch * steps_per_epoch + optimizer_steps,
                                "step/loss": step_loss,
                                "step/grad_norm": step_grad_norm,
                                "step/loss_scale": float(scaler.get_scale()),
                                "step/lr_roi_head": optimizer.param_groups[2]["lr"],
                            }
                        )
                        if PROGRESS:
                            batches.set_postfix(
                                loss=f"{step_loss:.4f}",
                                grad=f"{step_grad_norm:.2f}",
                                refresh=False,
                            )
                    window_loss.zero_()
                    window_count = 0
            if timing_events is not None and not should_step:
                timing_events["step_end"].record()
            if timing_events is not None and is_main():
                timing_events["step_end"].synchronize()
                transfer_ms = timing_events["transfer_start"].elapsed_time(
                    timing_events["transfer_end"]
                )
                forward_backward_ms = timing_events["transfer_end"].elapsed_time(
                    timing_events["backward_end"]
                )
                optimizer_ms = timing_events["backward_end"].elapsed_time(
                    timing_events["step_end"]
                )
                run.log(
                    {
                        "step/global_step": epoch * steps_per_epoch + optimizer_steps,
                        "step/data_wait_ms": data_wait_ms,
                        "step/transfer_ms": transfer_ms,
                        "step/forward_backward_ms": forward_backward_ms,
                        "step/optimizer_ms": optimizer_ms,
                        "step/total_profiled_ms": (
                            data_wait_ms + transfer_ms + forward_backward_ms + optimizer_ms
                        ),
                    }
                )
        local_epoch_loss = float((running / max(count, 1)).item())
        epoch_loss = reduce_mean(local_epoch_loss, device)
        grad_norm_last = float(grad_norm.detach().item())
        evaluation = evaluate(model, gold_loader, device, desc="gold")
        metrics = {
            "epoch": epoch,
            "train/loss": epoch_loss,
            "gold/macro_auc": evaluation["combined_auc"],
            "gold/global_macro_auc": evaluation["global_auc"],
            "gold/roi_gain_macro_auc": (
                evaluation["combined_auc"] - evaluation["global_auc"]
            ),
            "system/epoch_minutes": (time.time() - started) / 60,
            "system/optimizer_steps": optimizer_steps,
            "system/gpu_peak_gb": (
                torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
            "train/grad_norm_last": grad_norm_last,
            "system/lr_backbone": optimizer.param_groups[0]["lr"],
            "system/lr_global_head": optimizer.param_groups[1]["lr"],
            "system/lr_roi_head": optimizer.param_groups[2]["lr"],
        }
        metrics.update(
            {
                f"gold/{slug(key)}_auc": value
                for key, value in evaluation["combined_by_target"].items()
            }
        )
        metrics.update(
            {
                f"gold/{slug(key)}_roi_gain": (
                    value - evaluation["global_by_target"][key]
                )
                for key, value in evaluation["combined_by_target"].items()
                if key in ROI_TARGETS
            }
        )
        metrics.update(
            {
                f"model/gate_{slug(key)}": float(
                    torch.sigmoid(model.beta[key]).detach().cpu()
                )
                for key in ROI_TARGETS
            }
        )
        run.log(metrics)
        history.append(metrics)
        candidate_paths = save_top_candidate(
            model,
            epoch,
            evaluation["combined_auc"],
            history,
        )
        if is_main():
            print(
                json.dumps(
                    {
                        key: round(float(value), 5)
                        if isinstance(value, (float, int))
                        else value
                        for key, value in metrics.items()
                    }
                ),
                flush=True,
            )
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                history,
                config,
                metrics,
            )
            torch.save(payload, OUTPUT_DIR / "armc_roi_radiodino_expert_last.pt")
            if (epoch + 1) % UPLOAD_EVERY == 0:
                extras = candidate_paths[:1]
                log_checkpoint(run, payload, epoch, metrics, extra_files=extras)
        barrier()
    final_epoch = max(start_epoch, EPOCHS) - 1
    final_metrics = history[-1] if history else {"epoch": final_epoch}
    resumable_payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        scaler,
        final_epoch,
        history,
        config,
        final_metrics,
    )
    candidate_paths = save_top_candidate(
        model,
        final_epoch,
        float(final_metrics.get("gold/macro_auc", float("nan"))),
        history,
    )
    if not candidate_paths:
        raise RuntimeError("no top validation checkpoint survived for final selection")
    best_state = torch.load(
        candidate_paths[0],
        map_location="cpu",
        weights_only=False,
    )["model"]
    model.load_state_dict(best_state)
    # Every rank runs the same comparison on the same gathered predictions, so
    # they reach the same verdict without anyone broadcasting a decision.
    best_evaluation = evaluate(model, gold_loader, device, desc="select best single")
    selected_name = "best_single"
    selected_state = best_state
    # Evaluation is deterministic and always uses every ROI window, so the
    # selection score already is the final Gold score: no second pass.
    final_evaluation = best_evaluation
    if len(candidate_paths) > 1:
        averaged_state = average_candidates(candidate_paths)
        model.load_state_dict(averaged_state)
        averaged_evaluation = evaluate(model, gold_loader, device, desc="select top-k average")
        if averaged_evaluation["combined_auc"] >= best_evaluation["combined_auc"]:
            selected_name = f"top_{len(candidate_paths)}_average"
            selected_state = averaged_state
            final_evaluation = averaged_evaluation
        else:
            model.load_state_dict(best_state)
    if not is_main():
        barrier()
        if world > 1:
            dist.destroy_process_group()
        return
    selected_path = OUTPUT_DIR / "armc_roi_radiodino_expert_selected_model.pt"
    torch.save(
        {
            "model": selected_state,
            "arch": ARCH,
            "res": RES,
            "labels": LABELS,
            "roi_targets": ROI_TARGETS,
            # Only these columns are trained; the model emits NaN for the rest.
            "output_labels": ROI_TARGETS,
            "selection": selected_name,
            "candidate_epochs": [
                int(re.search(r"ep(\d+)\.pt$", path.name).group(1))
                for path in candidate_paths
            ],
            "gold_final": final_evaluation,
            "config": config,
        },
        selected_path,
    )
    # The "_k62" suffix is kept only so these keys line up in W&B with the
    # global-slice arms; this ROI expert never reads global windows.
    final_log = {
        "epoch": EPOCHS,
        "gold/final_macro_auc_k62": final_evaluation["combined_auc"],
        "gold/final_global_macro_auc_k62": final_evaluation["global_auc"],
    }
    final_log.update(
        {
            f"gold/final_{slug(key)}_auc_k62": value
            for key, value in final_evaluation["combined_by_target"].items()
        }
    )
    run.log(final_log)
    log_checkpoint(
        run,
        resumable_payload,
        final_epoch,
        {**final_metrics, **final_log},
        final=True,
        extra_files=[selected_path, *candidate_paths],
    )
    run.summary.update(
        {
            "best_gold_macro_auc": max(
                (item["gold/macro_auc"] for item in history),
                default=float("nan"),
            ),
            "selected_model": selected_name,
            "final_gold_macro_auc_k62": final_evaluation["combined_auc"],
            "completed_epoch": final_epoch,
        }
    )
    run.finish()
    barrier()
    if world > 1:
        dist.destroy_process_group()


# --------------------------------- launch ----------------------------------
SELF_BASENAME = "train_armc_roi_radiodino_expert_kaggle.py"


def spawn_canary(rank: int) -> None:
    print(
        f"[canary] rank {rank} alive, pid {os.getpid()}, torch {torch.__version__}, "
        f"gpus visible {torch.cuda.device_count()}",
        flush=True,
    )


def find_all_files(name: str) -> list[Path]:
    """Every copy of ``name``, /kaggle/working first, in stable walk order."""

    found: list[Path] = []
    for root in (Path("/kaggle/working"), Path(ARMC_ROI_INPUT)):
        if not root.exists():
            continue
        for directory, subdirectories, filenames in os.walk(root):
            subdirectories[:] = sorted(
                item
                for item in subdirectories
                if item not in {"train_series", "test_series", ".git", "__pycache__"}
            )
            if name in filenames:
                path = (Path(directory) / name).resolve()
                if path not in found:
                    found.append(path)
    return found


def is_ddp_capable(path: Path) -> bool:
    """True if this copy of the trainer defines the DDP entry point.

    Checked on the text, never by importing: a stale copy would otherwise
    execute its own top level just to be rejected.
    """

    try:
        return "\ndef train_worker(" in path.read_text(errors="ignore")
    except OSError:
        return False


def resolve_launch_module():
    """Find a module object whose functions a spawned child can re-import.

    Returns ``(module, reason)``; ``module`` is None when DDP cannot launch,
    and ``reason`` then says why, naming every copy that was considered.

    Spawn starts each rank as a fresh interpreter that imports the module
    holding the target function. Running this file with ``python file.py``
    gives that for free. Running it with ``exec(open(...).read())`` does not:
    ``__main__`` is then the notebook kernel, which a child cannot usefully
    re-import, so fall back to importing this file from disk by basename.
    Kaggle inputs routinely hold older uploads of the same file, so the first
    basename match is not good enough: a pre-DDP copy has no ``train_worker``.
    """

    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    # Judge ``__main__`` by its file's contents, not its namespace. exec() of
    # this source defines train_worker inside ``__main__`` whatever file that
    # is, and a child re-runs that file as a script: under IPython it is the
    # kernel launcher, which would never reach the trainer.
    if main_file and Path(main_file).exists() and is_ddp_capable(Path(main_file)):
        return main_module, f"running as {main_file}"
    if SELF_PATH:
        candidates = [Path(SELF_PATH)]
        if not candidates[0].exists():
            return None, f"SELF_PATH does not exist: {SELF_PATH}"
    else:
        candidates = find_all_files(SELF_BASENAME)
    if not candidates:
        return None, f"no copy of {SELF_BASENAME} under /kaggle/working or {ARMC_ROI_INPUT}"
    usable = [path for path in candidates if is_ddp_capable(path)]
    if not usable:
        listed = "\n  ".join(str(path) for path in candidates)
        return None, (
            f"found {len(candidates)} copy/copies of {SELF_BASENAME}, but none defines "
            f"train_worker (they predate DDP):\n  {listed}"
        )
    path = usable[0]
    skipped = [str(item) for item in candidates if item not in usable]
    directory = str(path.parent.resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        return None, f"cannot build an import spec for {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    reason = f"ranks will load {path}"
    if skipped:
        reason += " (ignored pre-DDP copies: " + ", ".join(skipped) + ")"
    if len(usable) > 1:
        reason += " (other DDP-capable copies: " + ", ".join(map(str, usable[1:])) + ")"
    return module, reason


def run_children(target, args, nprocs: int, deadline_s: float, label: str) -> bool:
    """start_processes with a hard deadline. True only on clean completion."""

    context = mp.start_processes(
        target, args=args, nprocs=nprocs, join=False, start_method=DDP_START_METHOD
    )
    started = time.time()
    try:
        while not context.join(timeout=15):
            if time.time() - started > deadline_s:
                log(f"{label}: exceeded {deadline_s / 60:.0f} min; terminating ranks")
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                return False
    except Exception as error:  # a child raised; its traceback is already above
        log(f"{label} failed: {type(error).__name__}: {str(error)[:2000]}")
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        return False
    return True


def main():
    validate_config()
    visible = torch.cuda.device_count()
    if WORLD_SIZE == 0:
        world = max(visible, 1)
    else:
        # Never ask for more ranks than there are GPUs, but allow several ranks
        # on a GPU-less box: that is how the gloo smoke test runs.
        world = WORLD_SIZE if visible == 0 else min(WORLD_SIZE, visible)
    world = max(world, 1)
    if world == 1:
        log(f"single process on {visible or 0} visible GPU(s)")
        train_worker(0, 1, 0)
        return
    module, reason = resolve_launch_module()
    if module is None:
        log(
            f"{visible} GPUs visible but DDP ranks cannot be spawned: {reason}\n"
            "Falling back to one GPU. Fix: run `!python -u <path>/" + SELF_BASENAME
            + "` on the current file, or set SELF_PATH to it."
        )
        train_worker(0, 1, 0)
        return
    # The ranks run the on-disk copy, not this cell: say which one.
    log(reason)
    # Three-minute canary before hours of work: if children cannot start at all
    # under this start method, fail now rather than after the caches are warm.
    if not run_children(module.spawn_canary, (), 1, 180, "spawn canary"):
        raise SystemExit(f"child processes cannot start under {DDP_START_METHOD}")
    log(
        f"spawn canary passed ({REFERENCE_STEP_SECONDS:.3f}s/step reference)"
    )
    port = 29500 + random.randint(0, 999)
    log(f"launching {world} DDP ranks on port {port}")
    if not run_children(
        module.train_worker,
        (world, port),
        world,
        12 * 3600,
        "train",
    ):
        raise SystemExit("training failed or stalled; see the rank tracebacks above")


if __name__ == "__main__":
    main()
