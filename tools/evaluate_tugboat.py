#!/usr/bin/env python3
"""Evaluate custom marine_target weights on the held-out local test split."""

# ruff: noqa: E402, I001 -- add the src layout before importing the local package.

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_ptz.benchmark import local_path_marker, write_report
from marine_ptz.dataset import (
    IMAGE_SUFFIXES,
    DatasetPreparationError,
    validate_tugboat_data_yaml,
    validate_yolo_label,
)


class TugboatEvaluationError(RuntimeError):
    """Raised when held-out custom-model evaluation cannot be completed."""


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    model: Path
    data: Path
    device: str
    image_size: int
    confidence_threshold: float
    precision: float
    recall: float
    map50: float
    map50_95: float
    prediction_count: int
    prediction_directory: Path
    positive_image_count: int
    reviewed_negative_image_count: int
    false_positive_negative_image_count: int | None
    negative_image_false_positive_rate: float | None
    maximum_negative_confidence: float | None
    false_positive_directory: Path | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "evaluation_kind": "custom_model_held_out_test",
            "class_name": "marine_target",
            "model": local_path_marker(str(self.model)),
            "data": local_path_marker(str(self.data)),
            "device": self.device,
            "image_size": self.image_size,
            "confidence_threshold": self.confidence_threshold,
            "metrics": {
                "precision": self.precision,
                "recall": self.recall,
                "map50": self.map50,
                "map50_95": self.map50_95,
            },
            "prediction_count": self.prediction_count,
            "prediction_directory": local_path_marker(str(self.prediction_directory)),
            "held_out_composition": {
                "positive_images": self.positive_image_count,
                "reviewed_negative_images": self.reviewed_negative_image_count,
            },
            "reviewed_negative_metrics": {
                "evaluation_performed": self.reviewed_negative_image_count > 0,
                "status": (
                    "measured"
                    if self.reviewed_negative_image_count > 0
                    else "not_measured_no_reviewed_negative_images"
                ),
                "false_positive_images": self.false_positive_negative_image_count,
                "false_positive_image_rate": self.negative_image_false_positive_rate,
                "maximum_confidence": self.maximum_negative_confidence,
                "annotated_false_positive_directory": (
                    None
                    if self.false_positive_directory is None
                    else local_path_marker(str(self.false_positive_directory))
                ),
            },
            "baseline_context": {
                "model": "YOLO11n COCO",
                "target_class": "boat",
                "physical_target": "Green Toys tugboat",
                "observed_result": "zero detections",
                "comparable_to_custom_test_metrics": False,
            },
        }


def _evaluation_batch(value: str) -> int:
    if value.casefold() == "auto":
        raise argparse.ArgumentTypeError(
            "evaluation batch does not support auto; use a positive integer such as 8"
        )
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "evaluation batch must be a positive integer such as 8"
        ) from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("evaluation batch must be a positive integer such as 8")
    return number


def _load_yolo() -> Any:
    try:
        return import_module("ultralytics").YOLO
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        raise TugboatEvaluationError(
            "Ultralytics is unavailable; use the verified constrained vision environment"
        ) from exc


def _finite_metric(box: Any, attribute: str, label: str) -> float:
    value = getattr(box, attribute, None)
    if isinstance(value, bool):
        raise TugboatEvaluationError(f"Ultralytics did not report finite {label}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TugboatEvaluationError(f"Ultralytics did not report finite {label}") from exc
    if not math.isfinite(result):
        raise TugboatEvaluationError(f"Ultralytics did not report finite {label}")
    return result


def _confidence_values(result: Any) -> tuple[float, ...]:
    boxes = getattr(result, "boxes", None)
    values = getattr(boxes, "conf", ()) if boxes is not None else ()
    for method_name in ("detach", "cpu"):
        method = getattr(values, method_name, None)
        if callable(method):
            values = method()
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TugboatEvaluationError("Ultralytics returned invalid confidence values") from exc
    confidences: list[float] = []
    for value in raw_values:
        if isinstance(value, bool):
            raise TugboatEvaluationError("Ultralytics returned invalid confidence values")
        try:
            confidence = float(value)
        except (TypeError, ValueError) as exc:
            raise TugboatEvaluationError("Ultralytics returned invalid confidence values") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise TugboatEvaluationError("Ultralytics returned invalid confidence values")
        confidences.append(confidence)
    return tuple(confidences)


def _classify_test_images(
    definition: Any,
    test_images: tuple[Path, ...],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    positive: list[Path] = []
    negative: list[Path] = []
    for image_path in test_images:
        relative = image_path.relative_to(definition.test_images)
        label_path = definition.test_labels / relative.with_suffix(".txt")
        if not label_path.is_file():
            raise TugboatEvaluationError(f"held-out image has no label file: {image_path}")
        try:
            boxes = validate_yolo_label(label_path)
        except DatasetPreparationError as exc:
            raise TugboatEvaluationError(str(exc)) from exc
        (positive if boxes else negative).append(image_path)
    return tuple(positive), tuple(negative)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/tugboat/yolo11n-marine-target/weights/best.pt"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/tugboat/data.yaml"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=_evaluation_batch, default=8, metavar="N")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--project", type=Path, default=Path("runs/tugboat-evaluation"))
    parser.add_argument("--name", default="heldout-test")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/evaluation/tugboat-test.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metrics_run = args.project / args.name
    predictions_run = args.project / f"{args.name}-predictions"
    false_positives_run = args.project / f"{args.name}-negative-false-positives"
    try:
        definition = validate_tugboat_data_yaml(args.data)
        if not args.model.is_file():
            raise TugboatEvaluationError(f"custom model does not exist: {args.model}")
        if not definition.test_images.is_dir():
            raise TugboatEvaluationError(
                f"prepared held-out test directory is missing: {definition.test_images}"
            )
        if not definition.test_labels.is_dir():
            raise TugboatEvaluationError(
                f"prepared held-out test label directory is missing: {definition.test_labels}"
            )
        test_images = tuple(
            path
            for path in definition.test_images.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        )
        if not test_images:
            raise TugboatEvaluationError("held-out test split contains no supported images")
        positive_images, negative_images = _classify_test_images(definition, test_images)
        if args.image_size <= 0:
            raise TugboatEvaluationError("image-size must be a positive integer")
        if not math.isfinite(args.confidence) or not 0.0 <= args.confidence <= 1.0:
            raise TugboatEvaluationError("confidence must be finite and between 0 and 1")
        if not args.device.strip() or not args.name.strip():
            raise TugboatEvaluationError("device and name must be non-empty")
        for run_path in (metrics_run, predictions_run):
            if run_path.exists():
                raise TugboatEvaluationError(f"refusing to reuse existing output {run_path}")

        yolo = _load_yolo()
        model = yolo(str(args.model))
        metrics = model.val(
            data=str(args.data.resolve()),
            split="test",
            imgsz=args.image_size,
            device=args.device,
            batch=args.batch,
            project=str(args.project),
            name=args.name,
            exist_ok=False,
        )
        box = getattr(metrics, "box", None)
        if box is None:
            raise TugboatEvaluationError("Ultralytics returned no box metrics")
        precision = _finite_metric(box, "mp", "precision")
        recall = _finite_metric(box, "mr", "recall")
        map50 = _finite_metric(box, "map50", "mAP50")
        map50_95 = _finite_metric(box, "map", "mAP50-95")

        predictions = tuple(
            model.predict(
                source=str(definition.test_images),
                imgsz=args.image_size,
                device=args.device,
                conf=args.confidence,
                save=True,
                save_txt=True,
                save_conf=True,
                project=str(args.project),
                name=f"{args.name}-predictions",
                exist_ok=False,
                verbose=False,
            )
        )
        negative_results = (
            tuple(
                model.predict(
                    source=[str(path) for path in negative_images],
                    imgsz=args.image_size,
                    device=args.device,
                    conf=0.0,
                    save=False,
                    verbose=False,
                    classes=[0],
                )
            )
            if negative_images
            else ()
        )
        if len(negative_results) != len(negative_images):
            raise TugboatEvaluationError(
                "Ultralytics negative-result count does not match reviewed-negative images"
            )
        negative_confidences = tuple(_confidence_values(result) for result in negative_results)
        maximum_negative_confidence = (
            max(
                (confidence for values in negative_confidences for confidence in values),
                default=0.0,
            )
            if negative_images
            else None
        )
        false_positive_images = tuple(
            image_path
            for image_path, confidences in zip(
                negative_images,
                negative_confidences,
                strict=True,
            )
            if any(confidence >= args.confidence for confidence in confidences)
        )
        if false_positive_images:
            if false_positives_run.exists():
                raise TugboatEvaluationError(
                    f"refusing to reuse existing output {false_positives_run}"
                )
            annotated = tuple(
                model.predict(
                    source=[str(path) for path in false_positive_images],
                    imgsz=args.image_size,
                    device=args.device,
                    conf=args.confidence,
                    save=True,
                    project=str(args.project),
                    name=f"{args.name}-negative-false-positives",
                    exist_ok=False,
                    verbose=False,
                    classes=[0],
                )
            )
            if len(annotated) != len(false_positive_images):
                raise TugboatEvaluationError(
                    "annotated false-positive count does not match detected negatives"
                )
        report = EvaluationReport(
            model=args.model,
            data=args.data,
            device=args.device,
            image_size=args.image_size,
            confidence_threshold=args.confidence,
            precision=precision,
            recall=recall,
            map50=map50,
            map50_95=map50_95,
            prediction_count=len(predictions),
            prediction_directory=predictions_run,
            positive_image_count=len(positive_images),
            reviewed_negative_image_count=len(negative_images),
            false_positive_negative_image_count=(
                len(false_positive_images) if negative_images else None
            ),
            negative_image_false_positive_rate=(
                len(false_positive_images) / len(negative_images) if negative_images else None
            ),
            maximum_negative_confidence=maximum_negative_confidence,
            false_positive_directory=(false_positives_run if false_positive_images else None),
        )
        write_report(report, args.report)
    except KeyboardInterrupt:
        print("evaluation stopped by operator", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2

    print(
        f"custom marine_target test precision={precision:.6f} recall={recall:.6f} "
        f"mAP50={map50:.6f} mAP50-95={map50_95:.6f}"
    )
    if negative_images:
        print(
            f"held-out positives={len(positive_images)} "
            f"reviewed_negatives={len(negative_images)} "
            f"negative_false_positives={len(false_positive_images)} "
            f"negative_false_positive_rate="
            f"{report.negative_image_false_positive_rate:.6f} "
            f"maximum_negative_confidence={maximum_negative_confidence:.6f}"
        )
    else:
        print(
            f"held-out positives={len(positive_images)} reviewed_negatives=0 "
            "negative_false_positive_rate=not_measured "
            "maximum_negative_confidence=not_measured"
        )
    print(f"held-out predictions={predictions_run} report={args.report}")
    print("COCO yolo11n boat baseline: zero physical tugboat detections; not a custom metric")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
