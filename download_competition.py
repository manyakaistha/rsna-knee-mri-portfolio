"""Download the RSNA competition data beside this project, with resumption."""

from __future__ import annotations

import os
from pathlib import Path


COMPETITION = "rsna-knee-abnormality-detection"
PROJECT_DIR = Path(__file__).resolve().parent
DOWNLOAD_ROOT = Path(
    os.environ.get("RSNA_DATA_DIR", PROJECT_DIR / "data")
).expanduser().resolve()

# This must be configured before importing kagglehub. Using its cache location,
# rather than competition_download(output_dir=...), preserves the partially
# downloaded .archive and lets kagglehub continue it with an HTTP Range request.
os.environ["KAGGLEHUB_CACHE"] = str(DOWNLOAD_ROOT)

import kagglehub  # noqa: E402


def main() -> None:
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    partial_archive = DOWNLOAD_ROOT / "competitions" / f"{COMPETITION}.archive"

    print(f"Download drive: {DOWNLOAD_ROOT.anchor}")
    print(f"Data root: {DOWNLOAD_ROOT}")
    print("An interrupted download can be resumed by running this file again.")

    try:
        path = Path(kagglehub.competition_download(COMPETITION)).resolve()
    except KeyboardInterrupt:
        print("\nDownload interrupted.")
        print(f"Partial download retained at: {partial_archive}")
        print("Run main.py again to resume it.")
        raise SystemExit(130) from None
    except Exception:
        print("\nDownload did not finish; any partial archive has been retained.")
        print(f"Resume location: {partial_archive}")
        print("Fix the reported problem, then run main.py again to resume.")
        raise

    print(f"Competition files are ready at: {path}")


if __name__ == "__main__":
    main()
