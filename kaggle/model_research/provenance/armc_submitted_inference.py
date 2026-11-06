from __future__ import annotations
# =============================================================================
# Arm C ROI -- hidden-test inference -> /kaggle/working/submission.csv
#
# ONE self-contained cell. Nothing to attach beyond the competition data and
# the trained checkpoint: the ROI geometry primitives, the ROI crop builder and
# the accepted final-r1 landmark priors are all inlined below.
#
# Run it AFTER train_armc_roi_kaggle.py has been run in the same notebook: it
# reuses ArmC, LABELS, ROI_TARGETS and autocast_context from that module.
#
# Test studies have no cached arrays, so BOTH model inputs are rebuilt from raw
# DICOM with the exact recipes the caches were built with:
#   global : kaggle/cache64/build_cache64.py  (64 x 336, span .02-.98, 140 mm)
#   ROI    : kaggle/roi_cache/build_roi_cache.py build_study (final-r1 priors)
# Both feed uint8 at 336; ArmC.prepare does /255, ImageNet normalisation and
# the resize to 384 on-device, exactly as in training.
# =============================================================================
import glob
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from pydicom.pixel_data_handlers.util import apply_modality_lut

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
CKPT = Path("/kaggle/input/datasets/manyakaistha/knee-roi-model/armc_roi_selected_model.pt")
K_SUBMIT = 24          # eval windows per study; K_EVAL=24 is the trained-on setting
BATCH_STUDIES = 2
BUILD_THREADS = 4

# --- frozen cache64 recipe (must stay identical to build_cache64.py) ---------
IMG = 336
CROP_MM = 140.0
SPAN = (0.02, 0.98)
SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
           ("Coronal", 0, 8), ("Axial", -1, 12)]
DEPTH = sum(k for _, _, k in SLOTS64)

# Accepted final-phase priors, copied from roi_work/cache_merged/roi_priors.json
# (accepted: true, phase: "final", 80 fit studies, zero mirror errors). These are
# the only priors values build_study reads.
ROI_PRIORS = {
    "landmarks": {
        "medial_joint_line": {
            "median_offset_mm": [18.667204950167765, 5.050869171716734, -19.870357392285733],
        },
        "lateral_joint_line": {
            "median_offset_mm": [-24.622100814733898, 1.260128529127364, -16.81546335696781],
        },
        "patellofemoral_center": {
            "median_offset_mm": [-5.433603266075679, 32.45693375683133, 12.868268042862667],
        },
        "suprapatellar_recess": {
            "median_offset_mm": [-9.671791724353696, 36.07318801770221, 58.749264289851105],
        },
    },
}

cv2.setNumThreads(1)


def say(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# =============================================================================
# Inlined verbatim from kaggle/roi_cache/roi_geometry.py (lines 18-409).
# Do not edit here: the crops must stay identical to the cached training crops.
# =============================================================================
POSITION_TOLERANCE_MM = 0.1
ORIENTATION_TOLERANCE_DEGREES = 1.0

LANDMARKS = (
    "medial_joint_line",
    "lateral_joint_line",
    "patellofemoral_center",
    "suprapatellar_recess",
)

TARGET_SPECS = {
    "mcl": {
        "name": "MCL",
        "landmark": "medial_joint_line",
        "crop_mm": (90.0, 120.0),
        "laterality_required": True,
        "allow_fluid_fallback": False,
        "slots": (("Coronal", 1), ("Axial", 1)),
    },
    "lateral_meniscus": {
        "name": "Lateral Meniscus",
        "landmark": "lateral_joint_line",
        "crop_mm": (80.0, 80.0),
        "laterality_required": True,
        "allow_fluid_fallback": False,
        "slots": (("Sagittal", 0), ("Coronal", 0)),
    },
    "lateral_oa": {
        "name": "Lateral OA",
        "landmark": "lateral_joint_line",
        "crop_mm": (85.0, 75.0),
        "laterality_required": True,
        "allow_fluid_fallback": False,
        "slots": (("Coronal", 0), ("Sagittal", 0)),
    },
    "pf_oa": {
        "name": "PF OA",
        "landmark": "patellofemoral_center",
        "crop_mm": (95.0, 85.0),
        "laterality_required": False,
        "allow_fluid_fallback": True,
        "slots": (("Axial", 1), ("Sagittal", 1)),
    },
    "synovitis": {
        "name": "Synovitis",
        "landmark": "suprapatellar_recess",
        "crop_mm": (100.0, 100.0),
        "laterality_required": False,
        "allow_fluid_fallback": False,
        "slots": (("Axial", 1), ("Sagittal", 1)),
        # The clinically approved policy retains the complete 100 x 100 mm
        # field but positions the suprapatellar landmark 25 mm below the
        # sagittal crop centre.  It leaves a 25 mm superior margin for an
        # expanded pouch and adds useful inferior joint context.
        "crop_center_offset_lps_mm_by_plane": {"Sagittal": (0.0, 0.0, -25.0)},
        "side_independent_window_strategy": "nearest_distinct_triplets",
    },
}


def _unit(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("direction vector has zero or non-finite length")
    return value / norm


def _finite_vector(value: object, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@dataclass(frozen=True)
class SliceGeometry:
    path: Path
    sop_uid: str
    ipp: np.ndarray
    row_dir: np.ndarray
    col_dir: np.ndarray
    normal: np.ndarray
    pixel_spacing: tuple[float, float]
    rows: int
    columns: int
    position: float

    @classmethod
    def from_header(cls, path: Path, header: object) -> "SliceGeometry":
        iop = _finite_vector(getattr(header, "ImageOrientationPatient"), 6, "IOP")
        ipp = _finite_vector(getattr(header, "ImagePositionPatient"), 3, "IPP")
        spacing = _finite_vector(getattr(header, "PixelSpacing"), 2, "PixelSpacing")
        if np.any(spacing <= 0):
            raise ValueError("PixelSpacing must be positive")
        rows = int(getattr(header, "Rows"))
        columns = int(getattr(header, "Columns"))
        if rows <= 0 or columns <= 0:
            raise ValueError("Rows and Columns must be positive")
        row_dir = _unit(iop[:3])
        col_dir = _unit(iop[3:])
        if abs(float(np.dot(row_dir, col_dir))) > 1e-3:
            raise ValueError("IOP row and column directions are not orthogonal")
        normal = _unit(np.cross(row_dir, col_dir))
        return cls(
            path=Path(path),
            sop_uid=str(getattr(header, "SOPInstanceUID", Path(path).stem)),
            ipp=ipp,
            row_dir=row_dir,
            col_dir=col_dir,
            normal=normal,
            pixel_spacing=(float(spacing[0]), float(spacing[1])),
            rows=rows,
            columns=columns,
            position=float(np.dot(ipp, normal)),
        )

    def pixel_to_patient(self, row: float, column: float) -> np.ndarray:
        row_spacing, column_spacing = self.pixel_spacing
        return (
            self.ipp
            + float(column) * column_spacing * self.row_dir
            + float(row) * row_spacing * self.col_dir
        )

    def patient_to_pixel(self, point: Sequence[float]) -> tuple[float, float]:
        delta = _finite_vector(point, 3, "patient point") - self.ipp
        row_spacing, column_spacing = self.pixel_spacing
        # ImageOrientationPatient vectors should be orthogonal, but real
        # scanner headers are often rounded enough that their dot product is
        # non-zero.  Independent dot-product projections are only an inverse
        # in the perfectly orthogonal case; their error grows with distance
        # from ImagePositionPatient.  Solve the two in-plane basis vectors
        # together instead, which is the stable inverse of pixel_to_patient
        # for both ideal and accepted near-orthogonal headers.
        basis = np.column_stack((
            column_spacing * self.row_dir,
            row_spacing * self.col_dir,
        ))
        coordinates, _residuals, rank, _singular_values = np.linalg.lstsq(
            basis, delta, rcond=None
        )
        if rank != 2 or not np.isfinite(coordinates).all():
            raise ValueError("image orientation does not define an invertible in-plane basis")
        column = float(coordinates[0])
        row = float(coordinates[1])
        return row, column

    def center(self) -> np.ndarray:
        return self.pixel_to_patient((self.rows - 1) / 2, (self.columns - 1) / 2)


@dataclass(frozen=True)
class SeriesGeometry:
    study_uid: str
    series_uid: str
    plane: str
    fluid_sensitive: int
    slices: tuple[SliceGeometry, ...]
    laterality_votes: tuple[str, ...] = ()
    manufacturer: str = ""
    field_strength: float | None = None
    slice_thickness: float | None = None

    @property
    def middle(self) -> SliceGeometry:
        return self.slices[len(self.slices) // 2]

    @property
    def center(self) -> np.ndarray:
        return self.middle.center()

    def nearest_slice_index(self, point: Sequence[float]) -> int:
        value = _finite_vector(point, 3, "patient point")
        distances = [abs(float(np.dot(value - item.ipp, item.normal))) for item in self.slices]
        return int(np.argmin(distances))


def orientation_angle_degrees(a: Sequence[float], b: Sequence[float]) -> float:
    cosine = float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def sort_deduplicate_slices(
    slices: Iterable[SliceGeometry],
    tolerance_mm: float = POSITION_TOLERANCE_MM,
    orientation_tolerance_degrees: float = ORIENTATION_TOLERANCE_DEGREES,
) -> tuple[SliceGeometry, ...]:
    """Sort by physical position and remove positions within ``tolerance_mm``.

    Duplicate selection is deterministic: SOP UID and path break position ties.
    A series is rejected unless at least three unique, consistently oriented
    slices remain.
    """
    candidates = list(slices)
    if not candidates:
        raise ValueError("series contains no geometrically valid slices")
    # Position every slice on one series-level normal.  Using each slice's own
    # (slightly noisy) normal can perturb ordering in otherwise valid oblique
    # acquisitions even when all orientations are within the 1-degree gate.
    reference = min(candidates, key=lambda item: (item.sop_uid, str(item.path)))
    for item in candidates:
        if orientation_angle_degrees(reference.row_dir, item.row_dir) > orientation_tolerance_degrees:
            raise ValueError("row orientation varies by more than 1 degree")
        if orientation_angle_degrees(reference.col_dir, item.col_dir) > orientation_tolerance_degrees:
            raise ValueError("column orientation varies by more than 1 degree")
    projected = [
        replace(item, position=float(np.dot(item.ipp, reference.normal)))
        for item in candidates
    ]
    ordered = sorted(projected, key=lambda item: (item.position, item.sop_uid, str(item.path)))
    unique: list[SliceGeometry] = []
    for item in ordered:
        if not unique or item.position - unique[-1].position > tolerance_mm:
            unique.append(item)
    if len(unique) < 3:
        raise ValueError("series contains fewer than three unique slice positions")
    positions = np.asarray([item.position for item in unique])
    if not np.all(np.diff(positions) > 0):
        raise ValueError("slice positions are not strictly monotonic")
    return tuple(unique)


def normalize_laterality(value: object) -> str | None:
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT"}:
        return "L"
    if text in {"R", "RIGHT"}:
        return "R"
    return None


def resolve_laterality(votes: Iterable[object]) -> tuple[str, str]:
    normalized = [value for value in (normalize_laterality(vote) for vote in votes) if value]
    left, right = normalized.count("L"), normalized.count("R")
    if left == right:
        return "UNKNOWN", "absent" if not normalized else "conflict"
    return ("L", "dicom_majority") if left > right else ("R", "dicom_majority")


def canonical_axes(laterality: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = normalize_laterality(laterality)
    if side is None:
        raise ValueError("canonical medial axis requires known laterality")
    medial = np.array([-1.0, 0.0, 0.0]) if side == "L" else np.array([1.0, 0.0, 0.0])
    anterior = np.array([0.0, -1.0, 0.0])
    superior = np.array([0.0, 0.0, 1.0])
    return medial, anterior, superior


def study_origin(series: Iterable[SeriesGeometry]) -> np.ndarray:
    centers = np.asarray([item.center for item in series], dtype=np.float64)
    if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] != 3:
        raise ValueError("at least one valid series is required for a study origin")
    return np.median(centers, axis=0)


def patient_to_canonical(point: Sequence[float], origin: Sequence[float], laterality: str) -> np.ndarray:
    delta = _finite_vector(point, 3, "patient point") - _finite_vector(origin, 3, "origin")
    return np.asarray([float(np.dot(delta, axis)) for axis in canonical_axes(laterality)])


def canonical_to_patient(offset: Sequence[float], origin: Sequence[float], laterality: str) -> np.ndarray:
    values = _finite_vector(offset, 3, "canonical offset")
    axes = canonical_axes(laterality)
    return _finite_vector(origin, 3, "origin") + sum(value * axis for value, axis in zip(values, axes))


def centerline_offset_to_patient(offset: Sequence[float], origin: Sequence[float]) -> np.ndarray:
    """Place a side-independent PF/suprapatellar prior for unknown laterality."""
    values = _finite_vector(offset, 3, "canonical offset")
    result = _finite_vector(origin, 3, "origin").copy()
    result += values[1] * np.array([0.0, -1.0, 0.0])
    result += values[2] * np.array([0.0, 0.0, 1.0])
    return result


def anatomical_crop_axes(plane: str, laterality: str | None) -> tuple[np.ndarray, np.ndarray]:
    # The sign of an extent axis is irrelevant. Patient X is therefore valid
    # for side-independent ML extents when laterality is unavailable.
    medial = canonical_axes(laterality)[0] if normalize_laterality(laterality) else np.array([1.0, 0.0, 0.0])
    anterior = np.array([0.0, -1.0, 0.0])
    superior = np.array([0.0, 0.0, 1.0])
    normalized = plane.strip().lower()
    if normalized == "coronal":
        return medial, superior
    if normalized == "sagittal":
        return anterior, superior
    if normalized == "axial":
        return medial, anterior
    raise ValueError(f"unsupported anatomical plane: {plane}")


def crop_bounds(
    geometry: SliceGeometry,
    center_patient: Sequence[float],
    plane: str,
    crop_mm: tuple[float, float],
    laterality: str | None,
) -> tuple[int, int, int, int]:
    """Return requested pixel bounds as ``row0,row1,col0,col1`` (end exclusive)."""
    width_axis, height_axis = anatomical_crop_axes(plane, laterality)
    half_width, half_height = float(crop_mm[0]) / 2, float(crop_mm[1]) / 2
    center = _finite_vector(center_patient, 3, "crop center")
    corners = [
        center + width_sign * half_width * width_axis + height_sign * half_height * height_axis
        for width_sign in (-1.0, 1.0)
        for height_sign in (-1.0, 1.0)
    ]
    pixels = np.asarray([geometry.patient_to_pixel(point) for point in corners])
    row0, col0 = np.floor(pixels.min(axis=0)).astype(int)
    row1, col1 = np.ceil(pixels.max(axis=0)).astype(int)
    if row1 <= row0 or col1 <= col0:
        raise ValueError("projected crop has zero area")
    return int(row0), int(row1), int(col0), int(col1)


def padding_fraction(bounds: tuple[int, int, int, int], rows: int, columns: int) -> float:
    row0, row1, col0, col1 = bounds
    requested = max(row1 - row0, 0) * max(col1 - col0, 0)
    if requested <= 0:
        return 1.0
    overlap_rows = max(0, min(row1, rows) - max(row0, 0))
    overlap_columns = max(0, min(col1, columns) - max(col0, 0))
    return float(1.0 - (overlap_rows * overlap_columns) / requested)


def crop_with_padding(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    row0, row1, col0, col1 = bounds
    output = np.zeros((row1 - row0, col1 - col0), dtype=image.dtype)
    src_row0, src_row1 = max(row0, 0), min(row1, image.shape[0])
    src_col0, src_col1 = max(col0, 0), min(col1, image.shape[1])
    if src_row1 > src_row0 and src_col1 > src_col0:
        dst_row0, dst_col0 = src_row0 - row0, src_col0 - col0
        output[
            dst_row0:dst_row0 + src_row1 - src_row0,
            dst_col0:dst_col0 + src_col1 - src_col0,
        ] = image[src_row0:src_row1, src_col0:src_col1]
    return output


def triplet_indices(
    center: int,
    length: int,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    candidates = ((center - 2, center - 1, center), (center, center + 1, center + 2))
    return tuple(
        item if min(item) >= 0 and max(item) < length else None
        for item in candidates
    )  # type: ignore[return-value]


def nearest_distinct_triplet_indices(
    center: int,
    length: int,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    """Return two distinct, in-bounds triplets closest to ``center``.

    Candidate triplets are all physically consecutive slices.  Ranking by the
    middle index's distance from the physically nearest landmark slice keeps
    the selection deterministic while avoiding boundary clamping or duplicate
    slices.
    """
    candidates = [(index - 1, index, index + 1) for index in range(1, length - 1)]
    ranked = sorted(candidates, key=lambda item: (abs(item[1] - center), item[1]))
    selected = ranked[:2]
    while len(selected) < 2:
        selected.append(None)
    return selected[0], selected[1]


def centered_triplet_indices(center: int, length: int) -> tuple[int, int, int] | None:
    """Return ``center-1, center, center+1`` without edge clamping."""
    candidate = (center - 1, center, center + 1)
    return candidate if min(candidate) >= 0 and max(candidate) < length else None


def pick_series(
    rows: Sequence[dict],
    plane: str,
    fluid_sensitive: int,
    allow_fluid_fallback: bool = True,
) -> dict | None:
    """Existing deterministic first-match rule, with no plane substitution."""
    candidates = [row for row in rows if str(row.get("Anatomical_Plane", "")) == plane]
    preferred = [
        row for row in candidates
        if int(float(row.get("Fluid_Sensitive", 0) or 0)) == fluid_sensitive
    ]
    if preferred:
        return preferred[0]
    return candidates[0] if candidates and allow_fluid_fallback else None

# =============================================================================
# Inlined verbatim from kaggle/roi_cache/build_roi_cache.py
# (constants, then _safe_float .. build_study).
# =============================================================================
IMAGE_SIZE = 336
WINDOWS_PER_TARGET = 4
PADDING_LIMIT = 0.20
PROVENANCE_FIELDS = (
    "StudyInstanceUID",
    "target",
    "window_index",
    "SeriesInstanceUID",
    "plane",
    "fluid_sensitive",
    "sop_uid_0",
    "sop_uid_1",
    "sop_uid_2",
    "slice_position_0_mm",
    "slice_position_1_mm",
    "slice_position_2_mm",
    "center_x_mm",
    "center_y_mm",
    "center_z_mm",
    "crop_center_x_mm",
    "crop_center_y_mm",
    "crop_center_z_mm",
    "row0",
    "row1",
    "column0",
    "column1",
    "padding_fraction",
    "intensity_low",
    "intensity_high",
    "laterality",
    "laterality_source",
    "candidate_laterality",
    "window_strategy",
    "valid",
    "failure_reason",
)

def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def read_series_geometry(
    study_uid: str,
    row: dict[str, Any],
    series_root: Path,
) -> SeriesGeometry:
    series_uid = str(row["SeriesInstanceUID"])
    directory = series_root / study_uid / series_uid
    files = sorted(directory.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no DICOMs in {directory}")
    slices: list[SliceGeometry] = []
    laterality_votes: list[str] = []
    manufacturer = ""
    field_strength = None
    slice_thickness = None
    for path in files:
        header = pydicom.dcmread(path, stop_before_pixels=True)
        slices.append(SliceGeometry.from_header(path, header))
        laterality_votes.extend([
            str(getattr(header, "ImageLaterality", "") or ""),
            str(getattr(header, "Laterality", "") or ""),
        ])
        manufacturer = manufacturer or str(getattr(header, "Manufacturer", "") or "")
        field_strength = field_strength or _safe_float(getattr(header, "MagneticFieldStrength", None))
        slice_thickness = slice_thickness or _safe_float(getattr(header, "SliceThickness", None))
    return SeriesGeometry(
        study_uid=study_uid,
        series_uid=series_uid,
        plane=str(row.get("Anatomical_Plane", "")),
        fluid_sensitive=int(float(row.get("Fluid_Sensitive", 0) or 0)),
        slices=sort_deduplicate_slices(slices),
        laterality_votes=tuple(laterality_votes),
        manufacturer=manufacturer,
        field_strength=field_strength,
        slice_thickness=slice_thickness,
    )


def _read_pixels(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(path)
    image = apply_modality_lut(dataset.pixel_array, dataset).astype(np.float32)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = image.max() - image
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"invalid pixel array in {path}")
    return image


def intensity_limits(series: SeriesGeometry) -> tuple[float, float]:
    positions = np.asarray([item.position for item in series.slices])
    low_position = positions[0] + 0.02 * (positions[-1] - positions[0])
    high_position = positions[0] + 0.98 * (positions[-1] - positions[0])
    desired_positions = np.linspace(low_position, high_position, 16)
    sample_indices = np.unique([
        int(np.argmin(np.abs(positions - position)))
        for position in desired_positions
    ])
    sample = np.concatenate([_read_pixels(series.slices[index].path).ravel() for index in sample_indices])
    low, high = np.percentile(sample, [2.0, 98.0])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("series intensity percentiles are degenerate")
    return float(low), float(high)


def _scale_uint8(image: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def load_study_geometry(
    study_uid: str,
    rows: list[dict[str, Any]],
    series_root: Path,
) -> tuple[dict[str, SeriesGeometry], dict[str, str]]:
    """Read every declared study series for a stable, acquisition-wide origin.

    Target series are still chosen by :func:`pick_series`.  Loading all series
    geometry prevents the canonical origin from shifting merely because a
    preferred slot is absent in an imperfect study.
    """
    geometries: dict[str, SeriesGeometry] = {}
    failures: dict[str, str] = {}
    unique_rows = {
        str(row["SeriesInstanceUID"]): row
        for row in rows
    }
    for series_uid in sorted(unique_rows):
        row = unique_rows[series_uid]
        try:
            geometries[series_uid] = read_series_geometry(study_uid, row, series_root)
        except Exception as error:
            failures[series_uid] = f"invalid_series:{type(error).__name__}:{error}"[:500]
    return geometries, failures


def _blank_provenance(
    study_uid: str,
    target: str,
    window_index: int,
    laterality: str,
    laterality_source: str,
    failure_reason: str,
    row: dict[str, Any] | None = None,
    candidate_laterality: str = "",
    window_strategy: str = "",
) -> dict[str, Any]:
    result = {field: "" for field in PROVENANCE_FIELDS}
    result.update({
        "StudyInstanceUID": study_uid,
        "target": target,
        "window_index": window_index,
        "SeriesInstanceUID": "" if row is None else str(row.get("SeriesInstanceUID", "")),
        "plane": "" if row is None else str(row.get("Anatomical_Plane", "")),
        "fluid_sensitive": "" if row is None else int(float(row.get("Fluid_Sensitive", 0) or 0)),
        "laterality": laterality,
        "laterality_source": laterality_source,
        "candidate_laterality": candidate_laterality,
        "window_strategy": window_strategy,
        "valid": 0,
        "failure_reason": failure_reason,
    })
    return result


def _landmark_point(
    landmark: str,
    priors: dict[str, Any],
    origin: np.ndarray,
    laterality: str,
    laterality_required: bool,
) -> np.ndarray:
    offset = priors["landmarks"][landmark]["median_offset_mm"]
    if not laterality_required:
        return centerline_offset_to_patient(offset, origin)
    return canonical_to_patient(offset, origin, laterality)


def _window_candidates(
    specification: dict[str, Any],
    priors: dict[str, Any],
    origin: np.ndarray,
    laterality: str,
) -> list[tuple[str, np.ndarray]]:
    """Return the fixed candidate centers used in every requested plane.

    Side-specific targets always receive both left- and right-knee hypotheses.
    This keeps the four-window tensor useful for the 50% of studies whose
    laterality metadata is absent, without trusting a noisy inferred side.
    """
    landmark = specification["landmark"]
    if specification["laterality_required"]:
        return [
            (side, _landmark_point(landmark, priors, origin, side, True))
            for side in ("L", "R")
        ]
    return [("CENTER", _landmark_point(landmark, priors, origin, laterality, False))]


def _side_independent_window_strategy(specification: dict[str, Any]) -> str:
    return str(specification.get("side_independent_window_strategy", "center_overlapping"))


def _crop_center(
    specification: dict[str, Any],
    plane: str,
    landmark_center: np.ndarray,
) -> np.ndarray:
    """Return target/plane-specific physical crop centre in LPS millimetres."""
    offsets = specification.get("crop_center_offset_lps_mm_by_plane", {})
    offset = offsets.get(plane, (0.0, 0.0, 0.0))
    value = np.asarray(offset, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"invalid crop centre offset for {specification['name']}/{plane}")
    return landmark_center + value


def build_study(
    study_uid: str,
    rows: list[dict[str, Any]],
    series_root: Path,
    priors: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    data = {
        target: np.zeros((WINDOWS_PER_TARGET, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for target in TARGET_SPECS
    }
    masks = {target: np.zeros(WINDOWS_PER_TARGET, dtype=np.uint8) for target in TARGET_SPECS}
    provenance: list[dict[str, Any]] = []
    geometries, series_failures = load_study_geometry(study_uid, rows, series_root)
    series_votes = [resolve_laterality(geometry.laterality_votes)[0] for geometry in geometries.values()]
    laterality, laterality_source = resolve_laterality(series_votes)
    if not geometries:
        for target in TARGET_SPECS:
            for window_index in range(WINDOWS_PER_TARGET):
                provenance.append(_blank_provenance(
                    study_uid, target, window_index, laterality, laterality_source,
                    "no_valid_selected_series",
                ))
        return data, masks, provenance
    origin = study_origin(geometries.values())
    intensity_cache: dict[str, tuple[float, float] | Exception] = {}
    pixel_cache: dict[tuple[str, int], np.ndarray] = {}

    for target, specification in TARGET_SPECS.items():
        try:
            candidates = _window_candidates(
                specification,
                priors,
                origin,
                laterality,
            )
        except Exception as error:
            for window_index in range(WINDOWS_PER_TARGET):
                provenance.append(_blank_provenance(
                    study_uid, target, window_index, laterality, laterality_source,
                    f"landmark_projection_failed:{type(error).__name__}:{error}"[:500],
                ))
            continue
        for slot_index, (plane, fluid) in enumerate(specification["slots"]):
            row = pick_series(
                rows,
                plane,
                fluid,
                allow_fluid_fallback=specification["allow_fluid_fallback"],
            )
            base_window = slot_index * 2
            if row is None:
                for local_index in range(2):
                    candidate_side = (
                        candidates[local_index][0]
                        if specification["laterality_required"]
                        else "CENTER"
                    )
                    provenance.append(_blank_provenance(
                        study_uid, target, base_window + local_index, laterality,
                        laterality_source, "missing_required_series", None,
                        candidate_laterality=candidate_side,
                        window_strategy=(
                            "paired_side_centered" if specification["laterality_required"]
                        else _side_independent_window_strategy(specification)
                        ),
                    ))
                continue
            series_uid = str(row["SeriesInstanceUID"])
            series = geometries.get(series_uid)
            if series is None:
                reason = series_failures.get(series_uid, "invalid_required_series")
                for local_index in range(2):
                    candidate_side = (
                        candidates[local_index][0]
                        if specification["laterality_required"]
                        else "CENTER"
                    )
                    provenance.append(_blank_provenance(
                        study_uid, target, base_window + local_index, laterality,
                        laterality_source, reason, row,
                        candidate_laterality=candidate_side,
                        window_strategy=(
                            "paired_side_centered" if specification["laterality_required"]
                        else _side_independent_window_strategy(specification)
                        ),
                    ))
                continue
            if series_uid not in intensity_cache:
                try:
                    intensity_cache[series_uid] = intensity_limits(series)
                except Exception as error:
                    intensity_cache[series_uid] = error
            limits = intensity_cache[series_uid]
            if isinstance(limits, Exception):
                for local_index in range(2):
                    candidate_side = (
                        candidates[local_index][0]
                        if specification["laterality_required"]
                        else "CENTER"
                    )
                    provenance.append(_blank_provenance(
                        study_uid, target, base_window + local_index, laterality,
                        laterality_source,
                        f"intensity_failed:{type(limits).__name__}:{limits}"[:500], row,
                        candidate_laterality=candidate_side,
                        window_strategy=(
                            "paired_side_centered" if specification["laterality_required"]
                        else _side_independent_window_strategy(specification)
                        ),
                    ))
                continue
            if specification["laterality_required"]:
                planned_windows = [
                    (
                        candidate_side,
                        center,
                        centered_triplet_indices(
                            series.nearest_slice_index(center),
                            len(series.slices),
                        ),
                        "paired_side_centered",
                    )
                    for candidate_side, center in candidates
                ]
            else:
                center = candidates[0][1]
                strategy = _side_independent_window_strategy(specification)
                nearest_index = series.nearest_slice_index(center)
                if strategy == "nearest_distinct_triplets":
                    selected_triplets = nearest_distinct_triplet_indices(
                        nearest_index,
                        len(series.slices),
                    )
                elif strategy == "center_overlapping":
                    selected_triplets = triplet_indices(nearest_index, len(series.slices))
                else:
                    raise ValueError(f"unsupported side-independent strategy: {strategy}")
                planned_windows = [
                    ("CENTER", center, indices, strategy)
                    for indices in selected_triplets
                ]
            for local_index, (candidate_side, center, indices, strategy) in enumerate(planned_windows):
                window_index = base_window + local_index
                if indices is None:
                    provenance.append(_blank_provenance(
                        study_uid, target, window_index, laterality,
                        laterality_source, "triplet_crosses_series_boundary", row,
                        candidate_laterality=candidate_side,
                        window_strategy=strategy,
                    ))
                    continue
                middle = series.slices[indices[1]]
                try:
                    crop_center = _crop_center(specification, plane, center)
                    bounds = crop_bounds(
                        middle, crop_center, plane, specification["crop_mm"], laterality,
                    )
                    pad = padding_fraction(bounds, middle.rows, middle.columns)
                    if pad > PADDING_LIMIT + 1e-12:
                        raise ValueError(f"padding_fraction_exceeds_0.20:{pad:.6f}")
                    channels = []
                    triplet_padding = []
                    for index in indices:
                        item = series.slices[index]
                        cache_key = (series_uid, index)
                        if cache_key not in pixel_cache:
                            pixel_cache[cache_key] = _scale_uint8(
                                _read_pixels(item.path), limits[0], limits[1],
                            )
                        image = pixel_cache[cache_key]
                        item_bounds = crop_bounds(
                            item, crop_center, plane, specification["crop_mm"], laterality,
                        )
                        item_pad = padding_fraction(item_bounds, item.rows, item.columns)
                        if item_pad > PADDING_LIMIT + 1e-12:
                            raise ValueError(f"padding_fraction_exceeds_0.20:{item_pad:.6f}")
                        triplet_padding.append(item_pad)
                        crop = crop_with_padding(image, item_bounds)
                        channels.append(cv2.resize(
                            crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR,
                        ))
                    data[target][window_index] = np.stack(channels)
                    masks[target][window_index] = 1
                    provenance.append({
                        "StudyInstanceUID": study_uid,
                        "target": target,
                        "window_index": window_index,
                        "SeriesInstanceUID": series_uid,
                        "plane": plane,
                        "fluid_sensitive": fluid,
                        "sop_uid_0": series.slices[indices[0]].sop_uid,
                        "sop_uid_1": series.slices[indices[1]].sop_uid,
                        "sop_uid_2": series.slices[indices[2]].sop_uid,
                        "slice_position_0_mm": series.slices[indices[0]].position,
                        "slice_position_1_mm": series.slices[indices[1]].position,
                        "slice_position_2_mm": series.slices[indices[2]].position,
                        "center_x_mm": float(center[0]),
                        "center_y_mm": float(center[1]),
                        "center_z_mm": float(center[2]),
                        "crop_center_x_mm": float(crop_center[0]),
                        "crop_center_y_mm": float(crop_center[1]),
                        "crop_center_z_mm": float(crop_center[2]),
                        "row0": bounds[0],
                        "row1": bounds[1],
                        "column0": bounds[2],
                        "column1": bounds[3],
                        "padding_fraction": max(triplet_padding),
                        "intensity_low": limits[0],
                        "intensity_high": limits[1],
                        "laterality": laterality,
                        "laterality_source": laterality_source,
                        "candidate_laterality": candidate_side,
                        "window_strategy": strategy,
                        "valid": 1,
                        "failure_reason": "",
                    })
                except Exception as error:
                    provenance.append(_blank_provenance(
                        study_uid, target, window_index, laterality, laterality_source,
                        f"crop_failed:{type(error).__name__}:{error}"[:500], row,
                        candidate_laterality=candidate_side,
                        window_strategy=strategy,
                    ))
    if len(provenance) != len(TARGET_SPECS) * WINDOWS_PER_TARGET:
        raise RuntimeError(f"internal provenance count error for {study_uid}: {len(provenance)}")
    return data, masks, provenance


# =============================================================================
# Global path: the cache64 recipe, inlined from kaggle/cache64/build_cache64.py
# =============================================================================
def order_and_meta(series_dir):
    records, spacings = [], []
    for path in glob.glob(series_dir + "/*.dcm"):
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True)
            iop = getattr(header, "ImageOrientationPatient", None)
            ipp = getattr(header, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                position = float(np.dot(np.array(ipp, float), normal))
            else:
                position = float(getattr(header, "InstanceNumber", 0) or 0)
            spacing = getattr(header, "PixelSpacing", None)
            spacing = float(spacing[0]) if spacing is not None else 0.5
            spacings.append(spacing)
            records.append((position, path, spacing))
        except Exception:
            records.append((0.0, path, 0.5))
    records.sort(key=lambda x: x[0])
    median = float(np.median(spacings)) if spacings else 0.5
    return [(path, spacing) for _, path, spacing in records], median


def read_px(path):
    dataset = pydicom.dcmread(path)
    image = apply_modality_lut(dataset.pixel_array, dataset).astype(np.float32)
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        image = image.max() - image
    return image


def mm_crop_resize(image, spacing):
    h, w = image.shape
    side = min(int(round(CROP_MM / max(spacing, 1e-3))), min(h, w))
    y0, x0 = (h - side) // 2, (w - side) // 2
    return cv2.resize(image[y0:y0 + side, x0:x0 + side], (IMG, IMG),
                      interpolation=cv2.INTER_AREA)


def pick_global_series(rows, plane, fluid, used):
    candidates = [r for r in rows if r["Anatomical_Plane"] == plane
                  and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        preferred = [r for r in candidates
                     if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def build_global_study(study_uid, rows, series_root):
    volume = np.zeros((DEPTH, IMG, IMG), np.uint8)
    index, used = 0, set()
    for plane, fluid, count in SLOTS64:
        row = pick_global_series(rows, plane, fluid, used)
        if row is None:
            index += count
            continue
        used.add(row["SeriesInstanceUID"])
        files, median = order_and_meta(f"{series_root}/{study_uid}/{row['SeriesInstanceUID']}")
        if not files:
            index += count
            continue
        n = len(files)
        lo, hi = int(n * SPAN[0]), int(n * SPAN[1]) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, count).round().astype(int) if n > 1 else [0] * count
        images, spacings = [], []
        for pick in picks:
            path, spacing = files[min(pick, n - 1)]
            try:
                images.append(read_px(path))
                spacings.append(spacing)
            except Exception:
                images.append(None)
                spacings.append(median)
        valid = [x for x in images if x is not None]
        if valid:
            low, high = np.percentile(np.concatenate([x.ravel() for x in valid]), [2.0, 98.0])
        else:
            low, high = 0.0, 1.0
        for image, spacing in zip(images, spacings):
            if index >= DEPTH:
                break
            if image is None:
                index += 1
                continue
            scaled = np.clip((image - low) / (high - low + 1e-6), 0, 1)
            volume[index] = (mm_crop_resize(scaled, spacing if spacing > 0 else median) * 255).astype(np.uint8)
            index += 1
    mask = (volume.reshape(DEPTH, -1).sum(1) > 0).astype(np.uint8)
    return volume, mask


def eval_centers(mask, k):
    """StudyDataset._centers(train=False) plus __getitem__'s clamp."""
    valid = np.flatnonzero(mask > 0)
    if len(valid) < 3:
        valid = np.arange(min(3, DEPTH))
    lo, hi = int(valid.min()), int(valid.max())
    centers = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, DEPTH - 2))]
    picked = [centers[i] for i in np.linspace(0, len(centers) - 1, k).round().astype(int)]
    return [max(1, min(c, DEPTH - 2)) for c in picked]


# =============================================================================
# Inference
# =============================================================================
def build_inputs(study_uid, rows, series_root):
    """uint8 tensors in exactly the layout StudyDataset yields at eval time."""
    volume, mask = build_global_study(study_uid, rows, series_root)
    if not mask.any():
        return None
    centers = eval_centers(mask, K_SUBMIT)
    triplets = np.stack([volume[[c - 1, c, c + 1]] for c in centers])
    global_x = torch.from_numpy(triplets)
    global_mask = torch.ones(len(centers), dtype=torch.bool)

    roi_data, roi_masks, _ = build_study(study_uid, rows, Path(series_root), ROI_PRIORS)
    roi_x, roi_mask = {}, {}
    for target_slug, specification in TARGET_SPECS.items():
        name = specification["name"]
        if name not in ROI_TARGETS:
            continue
        roi_x[name] = torch.from_numpy(roi_data[target_slug])
        roi_mask[name] = torch.from_numpy(roi_masks[target_slug].astype(bool))
    return global_x, global_mask, roi_x, roi_mask


def load_model(device):
    model = ArmC()
    state = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    say(f"loaded {CKPT.name} (0 missing / 0 unexpected)")
    return model.to(device).eval()


@torch.no_grad()
def predict(model, device, study_uids, rows_by_study, series_root):
    out = np.full((len(study_uids), len(LABELS)), np.nan, np.float32)
    started, unreadable = time.time(), 0
    with ThreadPoolExecutor(BUILD_THREADS) as pool:
        for start in range(0, len(study_uids), BATCH_STUDIES):
            batch = list(range(start, min(start + BATCH_STUDIES, len(study_uids))))
            built = list(pool.map(
                lambda i: build_inputs(study_uids[i], rows_by_study.get(study_uids[i], []),
                                       series_root),
                batch,
            ))
            keep = [(i, item) for i, item in zip(batch, built) if item is not None]
            unreadable += len(batch) - len(keep)
            if not keep:
                continue
            global_x = torch.stack([item[0] for _, item in keep]).to(device, non_blocking=True)
            global_mask = torch.stack([item[1] for _, item in keep]).to(device, non_blocking=True)
            roi_x = {t: torch.stack([item[2][t] for _, item in keep]).to(device, non_blocking=True)
                     for t in ROI_TARGETS}
            roi_mask = {t: torch.stack([item[3][t] for _, item in keep]).to(device, non_blocking=True)
                        for t in ROI_TARGETS}
            with autocast_context(device):
                logits, _ = model(global_x, global_mask, roi_x, roi_mask)
                probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            for (i, _), probability in zip(keep, probabilities):
                out[i] = probability
            done = batch[-1] + 1
            if done % 50 < BATCH_STUDIES or done == len(study_uids):
                elapsed = time.time() - started
                say(f"  {done}/{len(study_uids)} | {elapsed / 60:.1f} min | "
                    f"eta {elapsed / done * (len(study_uids) - done) / 60:.1f} min | "
                    f"unreadable {unreadable}")
    return out


def run_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = pd.read_csv(COMP / "test.csv")
    series = pd.read_csv(COMP / "test_series.csv")
    series["StudyInstanceUID"] = series.StudyInstanceUID.astype(str)
    series["SeriesInstanceUID"] = series.SeriesInstanceUID.astype(str)
    rows_by_study = {k: g.to_dict("records")
                     for k, g in series.groupby("StudyInstanceUID", sort=False)}
    study_uids = test.StudyInstanceUID.astype(str).tolist()
    say(f"device {device} | {len(study_uids)} test studies | K={K_SUBMIT}")

    model = load_model(device)
    probabilities = predict(model, device, study_uids, rows_by_study,
                            str(COMP / "test_series"))

    submission = pd.DataFrame(probabilities, columns=list(LABELS))
    submission.insert(0, "StudyInstanceUID", study_uids)
    sample = pd.read_csv(COMP / "sample_submission.csv")
    submission = sample[["StudyInstanceUID"]].astype(str).merge(
        submission, on="StudyInstanceUID", how="left",
    )
    filled = int(submission[list(LABELS)].isna().any(axis=1).sum())
    submission[list(LABELS)] = submission[list(LABELS)].fillna(0.5)
    if list(submission.columns) != list(sample.columns):
        raise RuntimeError("submission columns do not match sample_submission.csv")
    if not np.isfinite(submission[list(LABELS)].to_numpy()).all():
        raise RuntimeError("submission contains non-finite probabilities")
    submission.to_csv("/kaggle/working/submission.csv", index=False)
    say(f"wrote submission.csv {submission.shape} | {filled} studies fell back to 0.5")


run_submission()
