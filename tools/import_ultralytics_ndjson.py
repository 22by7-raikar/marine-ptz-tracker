#!/usr/bin/env python3
"""Import annotated Ultralytics NDJSON rows into a new local staging tree."""

# ruff: noqa: E402, I001 -- add the src layout before importing the local package.

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_ptz.dataset import (
    IMAGE_SUFFIXES,
    MARINE_TARGET_CLASS,
    DatasetPreparationError,
    YoloBox,
    validate_yolo_box,
)

MANIFEST_NAME = "ultralytics-ndjson-import.json"
VALID_SPLITS = frozenset({"train", "val", "test"})


class NdjsonImportError(ValueError):
    """Raised when an export cannot be imported without ambiguity."""

    def __init__(self, failures: tuple[str, ...] | list[str] | str) -> None:
        normalized = (failures,) if isinstance(failures, str) else tuple(failures)
        self.failures = normalized or ("unknown import failure",)
        super().__init__("\n".join(self.failures))


@dataclass(frozen=True, slots=True)
class SourceImage:
    session: str
    path: Path
    relative_path: Path


@dataclass(frozen=True, slots=True)
class ImportItem:
    line_number: int
    exported_filename: str
    split: str
    width: int
    height: int
    source: SourceImage
    destination_image: Path
    destination_label: Path
    boxes: tuple[YoloBox, ...]

    @property
    def annotation_kind(self) -> str:
        return "positive" if self.boxes else "reviewed_negative"


@dataclass(frozen=True, slots=True)
class ImportPlan:
    ndjson_path: Path
    source_root: Path
    output_root: Path
    image_rows: int
    skipped_unannotated_rows: int
    items: tuple[ImportItem, ...]

    @property
    def session_count(self) -> int:
        return len({item.source.session for item in self.items})

    @property
    def positive_count(self) -> int:
        return sum(bool(item.boxes) for item in self.items)

    @property
    def negative_count(self) -> int:
        return sum(not item.boxes for item in self.items)


def plan_import(
    ndjson_path: Path,
    source_staging_root: Path,
    output_staging_root: Path,
    *,
    include_unannotated_as_negatives: bool = False,
) -> ImportPlan:
    """Fully validate one local export and return a non-writing import plan."""
    failures: list[str] = []
    if not ndjson_path.is_file():
        failures.append(f"NDJSON export does not exist: {ndjson_path}")
    if not source_staging_root.is_dir():
        failures.append(f"source staging root is not a directory: {source_staging_root}")
    if output_staging_root.exists():
        failures.append(f"output staging root must be new: {output_staging_root}")
    if failures:
        raise NdjsonImportError(failures)

    source_resolved = source_staging_root.resolve()
    output_resolved = output_staging_root.resolve()
    if source_resolved == output_resolved:
        raise NdjsonImportError("source and output staging roots must differ")
    if source_resolved in output_resolved.parents or output_resolved in source_resolved.parents:
        raise NdjsonImportError("source and output staging roots must not contain one another")

    records = _read_ndjson(ndjson_path)
    metadata_line, metadata = records[0]
    _validate_metadata(metadata_line, metadata)
    source_index = _index_source_images(source_staging_root)

    items: list[ImportItem] = []
    seen_basenames: dict[str, int] = {}
    seen_destinations: dict[Path, int] = {}
    skipped = 0
    image_rows = 0
    for line_number, row in records[1:]:
        image_rows += 1
        try:
            parsed = _parse_image_row(line_number, row)
        except NdjsonImportError as exc:
            failures.extend(exc.failures)
            continue

        basename, split, width, height, boxes = parsed
        previous_line = seen_basenames.get(basename)
        if previous_line is not None:
            failures.append(
                f"line {line_number}: duplicate filename {basename!r}; "
                f"first appeared on line {previous_line}"
            )
            continue
        seen_basenames[basename] = line_number
        if not boxes and not include_unannotated_as_negatives:
            skipped += 1
            continue

        matches = source_index.get(basename, ())
        if not matches:
            failures.append(
                f"line {line_number}: annotated source image {basename!r} was not found"
            )
            continue
        if len(matches) != 1:
            locations = ", ".join(str(match.relative_path) for match in matches)
            failures.append(
                f"line {line_number}: annotated source image {basename!r} is ambiguous: {locations}"
            )
            continue

        source = matches[0]
        destination_image = output_staging_root / source.session / "images" / basename
        destination_label = (
            output_staging_root / source.session / "labels" / f"{Path(basename).stem}.txt"
        )
        for destination in (destination_image, destination_label):
            previous_destination_line = seen_destinations.get(destination)
            if previous_destination_line is not None:
                failures.append(
                    f"line {line_number}: destination {destination} collides with "
                    f"line {previous_destination_line}"
                )
            else:
                seen_destinations[destination] = line_number
        items.append(
            ImportItem(
                line_number=line_number,
                exported_filename=basename,
                split=split,
                width=width,
                height=height,
                source=source,
                destination_image=destination_image,
                destination_label=destination_label,
                boxes=boxes,
            )
        )

    if image_rows == 0:
        failures.append("NDJSON export contains no image rows")
    if not items:
        failures.append("NDJSON export contains no importable detection rows")
    if failures:
        raise NdjsonImportError(failures)
    return ImportPlan(
        ndjson_path=ndjson_path,
        source_root=source_staging_root,
        output_root=output_staging_root,
        image_rows=image_rows,
        skipped_unannotated_rows=skipped,
        items=tuple(items),
    )


def execute_import(plan: ImportPlan, *, dry_run: bool = False) -> None:
    """Publish a fully planned import without modifying any source file."""
    if plan.output_root.exists():
        raise NdjsonImportError(f"output staging root must be new: {plan.output_root}")
    if dry_run:
        return

    plan.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            dir=plan.output_root.parent,
            prefix=f".{plan.output_root.name}.import-",
        )
    )
    published = False
    try:
        for item in plan.items:
            relative_image = item.destination_image.relative_to(plan.output_root)
            relative_label = item.destination_label.relative_to(plan.output_root)
            temporary_image = temporary_root / relative_image
            temporary_label = temporary_root / relative_label
            temporary_image.parent.mkdir(parents=True, exist_ok=True)
            temporary_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.source.path, temporary_image)
            temporary_label.write_text(_render_labels(item.boxes), encoding="utf-8")

        manifest = _manifest(plan)
        manifest_path = temporary_root / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _verify_temporary_output(plan, temporary_root, manifest)
        os.replace(temporary_root, plan.output_root)
        published = True
    except (OSError, ValueError, TypeError) as exc:
        raise NdjsonImportError(f"failed to publish imported staging tree: {exc}") from exc
    finally:
        if not published:
            shutil.rmtree(temporary_root, ignore_errors=True)


def _read_ndjson(path: Path) -> tuple[tuple[int, dict[str, Any]], ...]:
    records: list[tuple[int, dict[str, Any]]] = []
    failures: list[str] = []
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise NdjsonImportError(f"unable to read NDJSON export {path}: {exc}") from exc
    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(
                    raw_line,
                    parse_constant=lambda constant: _reject_json_constant(line_number, constant),
                    object_pairs_hook=lambda pairs: _object_without_duplicates(line_number, pairs),
                )
            except (json.JSONDecodeError, NdjsonImportError) as exc:
                failures.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(value, dict):
                failures.append(f"line {line_number}: record must be a JSON object")
                continue
            records.append((line_number, value))
    if failures:
        raise NdjsonImportError(failures)
    if not records:
        raise NdjsonImportError("NDJSON export is empty")
    return tuple(records)


def _object_without_duplicates(
    line_number: int,
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NdjsonImportError(f"line {line_number}: duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(line_number: int, constant: str) -> None:
    raise NdjsonImportError(f"line {line_number}: non-finite JSON number {constant}")


def _validate_metadata(line_number: int, metadata: dict[str, Any]) -> None:
    failures: list[str] = []
    if metadata.get("type", "dataset") != "dataset":
        failures.append(f"line {line_number}: first record must be dataset metadata")
    if metadata.get("task") != "detect":
        failures.append(f"line {line_number}: metadata task must be exactly 'detect'")
    if metadata.get("class_names") != {"0": MARINE_TARGET_CLASS}:
        failures.append(f"line {line_number}: class_names must be exactly {{'0': 'marine_target'}}")
    if failures:
        raise NdjsonImportError(failures)


def _parse_image_row(
    line_number: int,
    row: dict[str, Any],
) -> tuple[str, str, int, int, tuple[YoloBox, ...]]:
    failures: list[str] = []
    if row.get("type", "image") != "image":
        failures.append(f"line {line_number}: expected an image record")
    exported_filename = row.get("file")
    if not isinstance(exported_filename, str) or not exported_filename.strip():
        failures.append(f"line {line_number}: file must be a non-empty string")
        basename = ""
    else:
        basename = Path(exported_filename.replace("\\", "/")).name
        if basename in {"", ".", ".."} or Path(basename).suffix.casefold() not in IMAGE_SUFFIXES:
            failures.append(f"line {line_number}: file has no supported image basename")
    split = row.get("split")
    if split not in VALID_SPLITS:
        failures.append(f"line {line_number}: split must be train, val, or test")
    width = row.get("width")
    height = row.get("height")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        failures.append(f"line {line_number}: width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        failures.append(f"line {line_number}: height must be a positive integer")

    boxes: tuple[YoloBox, ...] = ()
    annotations = row.get("annotations")
    if annotations not in (None, {}):
        if not isinstance(annotations, dict):
            failures.append(f"line {line_number}: annotations must be an object")
        else:
            unexpected = sorted(set(annotations) - {"boxes"})
            if unexpected:
                failures.append(
                    f"line {line_number}: unexpected annotation types: {', '.join(unexpected)}"
                )
            raw_boxes = annotations.get("boxes", [])
            if not isinstance(raw_boxes, list):
                failures.append(f"line {line_number}: annotations.boxes must be a list")
            else:
                parsed_boxes: list[YoloBox] = []
                for box_index, raw_box in enumerate(raw_boxes):
                    if not isinstance(raw_box, list) or len(raw_box) != 5:
                        failures.append(
                            f"line {line_number} box {box_index}: expected "
                            "[class_id, center_x, center_y, width, height]"
                        )
                        continue
                    try:
                        parsed_boxes.append(validate_yolo_box(*raw_box))
                    except DatasetPreparationError as exc:
                        failures.append(f"line {line_number} box {box_index}: {exc}")
                boxes = tuple(parsed_boxes)
    if failures:
        raise NdjsonImportError(failures)
    return basename, str(split), int(width), int(height), boxes


def _index_source_images(root: Path) -> dict[str, tuple[SourceImage, ...]]:
    indexed: dict[str, list[SourceImage]] = {}
    for session_root in sorted(path for path in root.iterdir() if path.is_dir()):
        images_root = session_root / "images"
        if not images_root.is_dir():
            continue
        for image_path in sorted(images_root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            indexed.setdefault(image_path.name, []).append(
                SourceImage(
                    session=session_root.name,
                    path=image_path,
                    relative_path=image_path.relative_to(root),
                )
            )
    return {name: tuple(images) for name, images in indexed.items()}


def _render_labels(boxes: tuple[YoloBox, ...]) -> str:
    if not boxes:
        return ""
    lines = [
        " ".join(
            (
                "0",
                format(box.center_x, ".17g"),
                format(box.center_y, ".17g"),
                format(box.width, ".17g"),
                format(box.height, ".17g"),
            )
        )
        for box in boxes
    ]
    return "\n".join(lines) + "\n"


def _manifest(plan: ImportPlan) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_export": plan.ndjson_path.name,
        "class_names": {"0": MARINE_TARGET_CLASS},
        "counts": {
            "image_rows": plan.image_rows,
            "imported_images": len(plan.items),
            "positive_images": plan.positive_count,
            "negative_images": plan.negative_count,
            "label_files": len(plan.items),
            "sessions": plan.session_count,
            "skipped_unannotated_rows": plan.skipped_unannotated_rows,
            "failures": 0,
        },
        "rows": [
            {
                "ndjson_line": item.line_number,
                "exported_filename": item.exported_filename,
                "platform_split": item.split,
                "width": item.width,
                "height": item.height,
                "box_count": len(item.boxes),
                "annotation_kind": item.annotation_kind,
                "local_source": item.source.relative_path.as_posix(),
                "destination_image": item.destination_image.relative_to(
                    plan.output_root
                ).as_posix(),
                "destination_label": item.destination_label.relative_to(
                    plan.output_root
                ).as_posix(),
            }
            for item in plan.items
        ],
    }


def _verify_temporary_output(
    plan: ImportPlan,
    temporary_root: Path,
    manifest: dict[str, object],
) -> None:
    images = tuple(
        path
        for path in temporary_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )
    labels = tuple(temporary_root.rglob("*.txt"))
    rows = manifest.get("rows")
    expected = len(plan.items)
    if len(images) != expected or len(labels) != expected or not isinstance(rows, list):
        raise NdjsonImportError("annotation/image count mismatch while verifying temporary import")
    if len(rows) != expected:
        raise NdjsonImportError("annotation/image count mismatch while verifying import manifest")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndjson", type=Path, required=True)
    parser.add_argument("--source-staging-root", type=Path, required=True)
    parser.add_argument("--output-staging-root", type=Path, required=True)
    parser.add_argument(
        "--include-unannotated-as-negatives",
        action="store_true",
        help=(
            "import zero-box rows as reviewed negatives; use only after manually "
            "confirming every unboxed image contains no marine_target"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = plan_import(
            args.ndjson,
            args.source_staging_root,
            args.output_staging_root,
            include_unannotated_as_negatives=args.include_unannotated_as_negatives,
        )
        execute_import(plan, dry_run=args.dry_run)
    except NdjsonImportError as exc:
        print(f"NDJSON import failed failures={len(exc.failures)}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    mode = "validated" if args.dry_run else "imported"
    print(
        f"NDJSON {mode} positive_images={plan.positive_count} "
        f"negative_images={plan.negative_count} "
        f"label_files={len(plan.items)} sessions={plan.session_count} "
        f"skipped_unannotated_rows={plan.skipped_unannotated_rows} failures=0"
    )
    if not args.dry_run:
        print(f"manifest={plan.output_root / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
