"""Resumably extract a Kaggle download archive into the current directory.

Usage:
    uv run python extract_archive.py /path/to/downloaded.archive

The archive may be stored on another drive. Output is always written beneath
the directory from which this command is run. Completed files are skipped on a
later invocation, and each new file is made visible only after it is complete.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, NamedTuple


COPY_BUFFER_BYTES = 8 * 1024 * 1024
PROGRESS_INTERVAL_SECONDS = 10


class ArchiveEntry(NamedTuple):
    name: str
    size: int
    is_directory: bool
    is_link: bool
    source: object


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def safe_output_path(root: Path, member_name: str) -> Path:
    """Map an archive member to ``root`` without allowing path traversal."""
    normalized = member_name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive path: {member_name!r}")
    output = (root / Path(*relative.parts)).resolve()
    if os.path.commonpath((str(root), str(output))) != str(root):
        raise ValueError(f"archive path escapes the output directory: {member_name!r}")
    return output


def temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.extracting")


def is_complete_file(destination: Path, expected_size: int) -> bool:
    return destination.is_file() and destination.stat().st_size == expected_size


def remaining_size(entries: Iterable[ArchiveEntry], output_root: Path) -> int:
    total = 0
    for entry in entries:
        if entry.is_directory:
            continue
        destination = safe_output_path(output_root, entry.name)
        if not is_complete_file(destination, entry.size):
            stale_temporary = temporary_path(destination)
            if stale_temporary.exists():
                stale_temporary.unlink()
            total += entry.size
    return total


def check_available_space(required: int, output_root: Path) -> None:
    available = shutil.disk_usage(output_root).free
    print(f"Space required for remaining files: {format_bytes(required)}")
    print(f"Space currently available:          {format_bytes(available)}")
    if required > available:
        shortfall = required - available
        raise OSError(
            f"Not enough free space in {output_root}. "
            f"Free at least {format_bytes(shortfall)} more, or run the script "
            "from a directory on a larger drive."
        )


def copy_entry(source: BinaryIO, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(destination)
    with temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=COPY_BUFFER_BYTES)
        output.flush()
        os.fsync(output.fileno())
    actual_size = temporary.stat().st_size
    if actual_size != expected_size:
        raise OSError(
            f"incomplete extraction for {destination}: "
            f"expected {expected_size} bytes, wrote {actual_size}"
        )
    os.replace(temporary, destination)


def extract_entries(
    entries: list[ArchiveEntry],
    output_root: Path,
    open_entry,
) -> None:
    required = remaining_size(entries, output_root)
    check_available_space(required, output_root)
    completed_bytes = 0
    skipped_files = 0
    extracted_files = 0
    last_report = time.monotonic()

    for index, entry in enumerate(entries, 1):
        destination = safe_output_path(output_root, entry.name)
        if entry.is_link:
            raise ValueError(f"archive links are not supported: {entry.name!r}")
        if entry.is_directory:
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if is_complete_file(destination, entry.size):
            skipped_files += 1
            completed_bytes += entry.size
            continue

        source = open_entry(entry)
        if source is None:
            raise OSError(f"could not read archive member: {entry.name}")
        with source:
            copy_entry(source, destination, entry.size)
        extracted_files += 1
        completed_bytes += entry.size

        now = time.monotonic()
        if now - last_report >= PROGRESS_INTERVAL_SECONDS or index == len(entries):
            print(
                f"Processed {index}/{len(entries)} entries; "
                f"{format_bytes(completed_bytes)} verified; "
                f"{extracted_files} extracted, {skipped_files} resumed/skipped",
                flush=True,
            )
            last_report = now

    print(
        f"Extraction complete: {extracted_files} files extracted, "
        f"{skipped_files} existing files reused."
    )


def extract_zip(archive_path: Path, output_root: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        entries = []
        for info in archive.infolist():
            mode = info.external_attr >> 16
            entries.append(
                ArchiveEntry(
                    name=info.filename,
                    size=info.file_size,
                    is_directory=info.is_dir(),
                    is_link=stat.S_ISLNK(mode),
                    source=info,
                )
            )
        extract_entries(
            entries,
            output_root,
            lambda entry: archive.open(entry.source, "r"),
        )


def extract_tar(archive_path: Path, output_root: Path) -> None:
    with tarfile.open(archive_path) as archive:
        entries = [
            ArchiveEntry(
                name=member.name,
                size=member.size,
                is_directory=member.isdir(),
                is_link=member.issym() or member.islnk(),
                source=member,
            )
            for member in archive.getmembers()
        ]
        extract_entries(entries, output_root, lambda entry: archive.extractfile(entry.source))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to the downloaded .archive file")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    archive_path = arguments.archive.expanduser().resolve()
    output_root = Path.cwd().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive_path}")

    print(f"Archive: {archive_path}")
    print(f"Extracting into: {output_root}")
    print("If interrupted, run the same command again to resume.")

    try:
        if zipfile.is_zipfile(archive_path):
            extract_zip(archive_path, output_root)
        elif tarfile.is_tarfile(archive_path):
            extract_tar(archive_path, output_root)
        else:
            raise ValueError(f"unsupported or corrupt archive: {archive_path}")
    except KeyboardInterrupt:
        print("\nExtraction interrupted. Completed files were retained.")
        print("Run the same command again to resume.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
