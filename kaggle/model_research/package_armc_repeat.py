"""Create private offline-W&B Arm C repeat kernels for Stage 2 only.

The generated trainer keeps every model/data/loss/optimizer setting fixed and
changes only the random seed, output path, W&B run name/mode and self-path.
It is not a candidate architecture implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "kaggle/armc_roi/train_armc_roi_kaggle.py"


def substitute_once(source: str, pattern: str, replacement: str) -> str:
    result, count = re.subn(pattern, lambda _: replacement, source, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {pattern!r}, found {count}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed not in {2027, 2028}:
        raise ValueError("Stage 2 seeds are frozen as 2027 and 2028")
    args.output.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text()
    if 'WANDB_API_KEY = ""' not in source:
        raise RuntimeError("Hardcoded W&B credential cleanup has not passed")
    base_sha = hashlib.sha256(source.encode()).hexdigest()
    path = f"/kaggle/working/armc_roi_s{args.seed}"
    changes = [
        (r"^SEED = 42$", f"SEED = {args.seed}"),
        (r'^OUTPUT_DIR = Path\("/kaggle/working/armc_roi"\)$', f'OUTPUT_DIR = Path("{path}")'),
        # A fresh run must warm-start from Arm 13. The trainer treats every
        # nonempty RESUME path as required, even when the file does not exist.
        # Its last.pt is still saved each epoch; a resumed envelope can point
        # RESUME at that artifact explicitly.
        (r'^RESUME = "/kaggle/working/armc_roi/armc_roi_last.pt"$', 'RESUME = ""'),
        (r'^WANDB_RUN_NAME = "armc-roi-arm13-seed42"$', f'WANDB_RUN_NAME = "armc-roi-repeat-seed{args.seed}"'),
        (r'^WANDB_MODE = ""(.*)$', 'WANDB_MODE = "offline"\\1'),
        (r'^SELF_PATH = ""$', 'SELF_PATH = __file__'),
        (r'^SELF_BASENAME = "train_armc_roi_kaggle_op3.py"$', 'SELF_BASENAME = "run.py"'),
    ]
    for pattern, replacement in changes:
        if '\\1' in replacement:
            result, count = re.subn(pattern, replacement, source, flags=re.MULTILINE)
            if count != 1:
                raise RuntimeError(f"Expected one occurrence of {pattern!r}, found {count}")
            source = result
        else:
            source = substitute_once(source, pattern, replacement)
    # No online token is injected. Offline W&B preserves metrics locally and
    # checkpoint files remain in Kaggle output for recovery.
    (args.output / "run.py").write_text(source)
    kernel_id = f"manyakaistha/armc-roi-repeat-seed-{args.seed}"
    metadata = {
        "id": kernel_id, "title": kernel_id.split("/", 1)[1],
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["manyakaistha/knee-roi-data"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"seed": args.seed, "kernel_id": kernel_id,
                      "production_sha256": base_sha,
                      "repeat_script_sha256": hashlib.sha256(source.encode()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
