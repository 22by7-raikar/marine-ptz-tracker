"""Hardware-free validation and session-safe preparation for YOLO datasets."""

from __future__ import annotations

import hashlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

MARINE_TARGET_CLASS = "marine_target"
MARINE_TARGET_CLASS_ID = 0
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
SPLIT_NAMES = ("train", "val", "test")
TUGBOAT_DATA_YAML = """train: images/train
val: images/val
test: images/test

names:
  - marine_target
"""


class DatasetPreparationError(ValueError):
    """Raised when staging data cannot safely form a YOLO dataset."""


@dataclass(frozen=True, slots=True)
class YoloBox:
    """One normalized YOLO object label."""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One field-qualified staging or label failure."""

    path: Path
    message: str
    line_number: int | None = None

    def __str__(self) -> str:
        location = str(self.path)
        if self.line_number is not None:
            location = f"{location}:{self.line_number}"
        return f"{location}: {self.message}"


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """A validated image/label pair belonging to one recording session."""

    session: str
    image_path: Path
    label_path: Path
    destination_stem: str


@dataclass(frozen=True, slots=True)
class PlannedCopy:
    """One non-destructive copy into a dataset split."""

    split: str
    session: str
    source_image: Path
    source_label: Path
    destination_image: Path
    destination_label: Path


@dataclass(frozen=True, slots=True)
class DatasetPlan:
    """Complete validated copy plan with whole-session split assignments."""

    session_splits: tuple[tuple[str, str], ...]
    copies: tuple[PlannedCopy, ...]

    def session_count(self, split: str) -> int:
        return sum(assigned == split for _, assigned in self.session_splits)

    def sample_count(self, split: str) -> int:
        return sum(copy.split == split for copy in self.copies)


@dataclass(frozen=True, slots=True)
class TugboatDatasetDefinition:
    """Validated paths from the one-class checked-in Ultralytics data file."""

    yaml_path: Path
    train_images: Path
    val_images: Path
    test_images: Path
    train_labels: Path
    val_labels: Path
    test_labels: Path


def validate_yolo_label(path: Path) -> tuple[YoloBox, ...]:
    """Validate a YOLO label file for the single ``marine_target`` class."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetPreparationError(f"{path}: unable to read label: {exc}") from exc

    boxes: list[YoloBox] = []
    issues: list[ValidationIssue] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        parsed, issue = _parse_label_line(path, line_number, raw_line)
        if issue is not None:
            issues.append(issue)
        elif parsed is not None:
            boxes.append(parsed)
    if issues:
        raise DatasetPreparationError("\n".join(str(issue) for issue in issues))
    return tuple(boxes)


def validate_yolo_box(
    class_id: object,
    center_x: object,
    center_y: object,
    width: object,
    height: object,
) -> YoloBox:
    """Validate one typed, normalized box for the single tugboat class."""
    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id != 0:
        raise DatasetPreparationError("class id must be exactly 0 for marine_target")
    raw_values = (center_x, center_y, width, height)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_values):
        raise DatasetPreparationError("coordinates must be numeric")
    values = tuple(float(value) for value in raw_values)
    if not all(math.isfinite(value) for value in values):
        raise DatasetPreparationError("coordinates must be finite")
    normalized_center_x, normalized_center_y, normalized_width, normalized_height = values
    if not 0.0 <= normalized_center_x <= 1.0 or not 0.0 <= normalized_center_y <= 1.0:
        raise DatasetPreparationError("box center must be normalized to [0, 1]")
    if not 0.0 < normalized_width <= 1.0 or not 0.0 < normalized_height <= 1.0:
        raise DatasetPreparationError("box width and height must be in (0, 1]")
    if (
        normalized_center_x - normalized_width / 2.0 < 0.0
        or normalized_center_x + normalized_width / 2.0 > 1.0
        or normalized_center_y - normalized_height / 2.0 < 0.0
        or normalized_center_y + normalized_height / 2.0 > 1.0
    ):
        raise DatasetPreparationError("normalized box extends outside the image")
    return YoloBox(
        class_id,
        normalized_center_x,
        normalized_center_y,
        normalized_width,
        normalized_height,
    )


def validate_tugboat_data_yaml(path: Path) -> TugboatDatasetDefinition:
    """Require exactly one ``marine_target`` class and local split directories."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetPreparationError(f"unable to read dataset YAML {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetPreparationError("dataset YAML root must be a mapping")
    names = _normalized_names(raw.get("names"))
    if names != (MARINE_TARGET_CLASS,):
        raise DatasetPreparationError(
            "dataset YAML must define exactly one class named marine_target"
        )
    resolved: dict[str, Path] = {}
    for split in SPLIT_NAMES:
        value = raw.get(split)
        if not isinstance(value, str) or not value.strip():
            raise DatasetPreparationError(f"dataset YAML {split} must be a local path")
        split_path = Path(value)
        if split_path.is_absolute() or ".." in split_path.parts:
            raise DatasetPreparationError(
                f"dataset YAML {split} must be relative to the dataset directory"
            )
        resolved[split] = path.parent / split_path
    return TugboatDatasetDefinition(
        yaml_path=path,
        train_images=resolved["train"],
        val_images=resolved["val"],
        test_images=resolved["test"],
        train_labels=path.parent / "labels" / "train",
        val_labels=path.parent / "labels" / "val",
        test_labels=path.parent / "labels" / "test",
    )


def create_tugboat_data_yaml(path: Path) -> TugboatDatasetDefinition:
    """Create the canonical one-class YAML without overwriting an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(TUGBOAT_DATA_YAML)
    except FileExistsError as exc:
        raise DatasetPreparationError(f"refusing to overwrite dataset YAML {path}") from exc
    return validate_tugboat_data_yaml(path)


def inspect_staging(
    staging_root: Path,
) -> tuple[tuple[DatasetSample, ...], tuple[ValidationIssue, ...]]:
    """Inspect ``session/images`` and ``session/labels`` trees without writing."""
    if not staging_root.is_dir():
        return (), (ValidationIssue(staging_root, "staging root is not a directory"),)

    samples: list[DatasetSample] = []
    issues: list[ValidationIssue] = []
    sessions = sorted(path for path in staging_root.iterdir() if path.is_dir())
    if not sessions:
        return (), (ValidationIssue(staging_root, "no recording-session directories found"),)

    destination_stems: dict[str, Path] = {}
    for session_path in sessions:
        images_root = session_path / "images"
        labels_root = session_path / "labels"
        if not images_root.is_dir():
            issues.append(ValidationIssue(images_root, "session images directory is missing"))
            continue
        if not labels_root.is_dir():
            issues.append(ValidationIssue(labels_root, "session labels directory is missing"))
            continue

        images = sorted(
            path
            for path in images_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        )
        if not images:
            issues.append(ValidationIssue(images_root, "session contains no supported images"))

        expected_labels: set[Path] = set()
        for image_path in images:
            relative = image_path.relative_to(images_root)
            label_path = labels_root / relative.with_suffix(".txt")
            expected_labels.add(label_path)
            if not label_path.is_file():
                issues.append(ValidationIssue(label_path, "matching label file is missing"))
                continue
            try:
                validate_yolo_label(label_path)
            except DatasetPreparationError as exc:
                issues.extend(_issues_from_error(exc, label_path))
                continue

            destination_stem = _destination_stem(session_path.name, relative.with_suffix(""))
            previous = destination_stems.get(destination_stem)
            if previous is not None:
                issues.append(
                    ValidationIssue(
                        image_path,
                        f"destination name collides with {previous}",
                    )
                )
                continue
            destination_stems[destination_stem] = image_path
            samples.append(
                DatasetSample(
                    session=session_path.name,
                    image_path=image_path,
                    label_path=label_path,
                    destination_stem=destination_stem,
                )
            )

        for label_path in sorted(labels_root.rglob("*.txt")):
            if label_path.is_file() and label_path not in expected_labels:
                issues.append(ValidationIssue(label_path, "label has no matching image"))

    return tuple(samples), tuple(issues)


def deterministic_session_split(
    sessions: tuple[str, ...] | list[str],
    *,
    seed: int = 20260802,
    train_fraction: float = 0.7,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
) -> dict[str, str]:
    """Assign complete sessions deterministically to train, val, and test."""
    unique = tuple(sorted(set(sessions)))
    if len(unique) != len(sessions):
        raise DatasetPreparationError("recording-session names must be unique")
    if len(unique) < len(SPLIT_NAMES):
        raise DatasetPreparationError("at least three recording sessions are required")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DatasetPreparationError("seed must be an integer")

    fractions = (train_fraction, val_fraction, test_fraction)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in fractions
    ):
        raise DatasetPreparationError("split fractions must be finite and greater than zero")
    if not math.isclose(sum(float(value) for value in fractions), 1.0, abs_tol=1e-9):
        raise DatasetPreparationError("split fractions must sum to 1")

    counts = _split_counts(len(unique), tuple(float(value) for value in fractions))
    ranked = sorted(
        unique,
        key=lambda session: hashlib.sha256(f"{seed}\0{session}".encode()).digest(),
    )
    assignments: dict[str, str] = {}
    offset = 0
    for split, count in zip(SPLIT_NAMES, counts):
        for session in ranked[offset : offset + count]:
            assignments[session] = split
        offset += count
    return assignments


def plan_tugboat_dataset(
    staging_root: Path,
    output_root: Path,
    *,
    seed: int = 20260802,
    train_fraction: float = 0.7,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
) -> DatasetPlan:
    """Validate staging data and build a whole-session copy plan."""
    samples, issues = inspect_staging(staging_root)
    if issues:
        raise DatasetPreparationError("\n".join(str(issue) for issue in issues))
    sessions = tuple(sorted({sample.session for sample in samples}))
    assignments = deterministic_session_split(
        sessions,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    copies = tuple(
        PlannedCopy(
            split=assignments[sample.session],
            session=sample.session,
            source_image=sample.image_path,
            source_label=sample.label_path,
            destination_image=(
                output_root
                / "images"
                / assignments[sample.session]
                / f"{sample.destination_stem}{sample.image_path.suffix.casefold()}"
            ),
            destination_label=(
                output_root
                / "labels"
                / assignments[sample.session]
                / f"{sample.destination_stem}.txt"
            ),
        )
        for sample in samples
    )
    return DatasetPlan(tuple(sorted(assignments.items())), copies)


def execute_dataset_plan(plan: DatasetPlan, *, dry_run: bool = False) -> None:
    """Copy a validated plan without deleting or overwriting any file."""
    conflicts = sorted(
        path
        for copy in plan.copies
        for path in (copy.destination_image, copy.destination_label)
        if path.exists()
    )
    if conflicts:
        rendered = "\n".join(str(path) for path in conflicts)
        raise DatasetPreparationError(f"dataset destinations already exist:\n{rendered}")
    if dry_run:
        return
    for copy in plan.copies:
        copy.destination_image.parent.mkdir(parents=True, exist_ok=True)
        copy.destination_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(copy.source_image, copy.destination_image)
        shutil.copy2(copy.source_label, copy.destination_label)


def _parse_label_line(
    path: Path,
    line_number: int,
    raw_line: str,
) -> tuple[YoloBox | None, ValidationIssue | None]:
    fields = raw_line.split()
    if len(fields) != 5:
        return None, ValidationIssue(
            path,
            "expected: 0 center_x center_y width height",
            line_number,
        )
    if fields[0] != str(MARINE_TARGET_CLASS_ID):
        return None, ValidationIssue(
            path,
            "class id must be exactly 0 for marine_target",
            line_number,
        )
    try:
        center_x, center_y, width, height = (float(value) for value in fields[1:])
    except ValueError:
        return None, ValidationIssue(path, "coordinates must be numeric", line_number)
    try:
        return validate_yolo_box(0, center_x, center_y, width, height), None
    except DatasetPreparationError as exc:
        return None, ValidationIssue(path, str(exc), line_number)


def _normalized_names(raw: object) -> tuple[str, ...]:
    if isinstance(raw, list) and all(isinstance(value, str) for value in raw):
        return tuple(raw)
    if isinstance(raw, dict) and set(raw) in ({0}, {"0"}):
        return (str(raw.get(0, raw.get("0"))),)
    return ()


def _issues_from_error(error: DatasetPreparationError, path: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    prefix = f"{path}:"
    for rendered in str(error).splitlines():
        message = rendered
        line_number: int | None = None
        if rendered.startswith(prefix):
            remainder = rendered[len(prefix) :]
            possible_line, separator, rest = remainder.partition(":")
            if separator and possible_line.isdigit():
                line_number = int(possible_line)
                message = rest.strip()
            else:
                message = remainder.strip()
        issues.append(ValidationIssue(path, message, line_number))
    return issues


def _destination_stem(session: str, relative_stem: Path) -> str:
    session_name = _safe_name(session)
    relative_names = tuple(_safe_name(part) for part in relative_stem.parts)
    if relative_names and (
        relative_names[0] == session_name or relative_names[0].startswith(f"{session_name}__")
    ):
        pieces = relative_names
    else:
        pieces = (session_name, *relative_names)
    return "__".join(piece for piece in pieces if piece)


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )
    cleaned = cleaned.strip("_-")
    return cleaned or "unnamed"


def _split_counts(total: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    exact = tuple(total * fraction for fraction in fractions)
    counts = [math.floor(value) for value in exact]
    for index in sorted(
        range(len(counts)),
        key=lambda item: (exact[item] - counts[item], fractions[item], -item),
        reverse=True,
    )[: total - sum(counts)]:
        counts[index] += 1
    for empty_index, count in enumerate(tuple(counts)):
        if count != 0:
            continue
        donor = max(range(len(counts)), key=lambda index: counts[index])
        if counts[donor] <= 1:
            raise DatasetPreparationError("not enough sessions to populate every split")
        counts[donor] -= 1
        counts[empty_index] += 1
    return counts[0], counts[1], counts[2]
