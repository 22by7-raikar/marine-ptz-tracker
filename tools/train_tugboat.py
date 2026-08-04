#!/usr/bin/env python3
"""Launch one local Ultralytics fine-tuning run for marine_target."""

# ruff: noqa: E402, I001 -- add the src layout before importing the local package.

from __future__ import annotations

import argparse
import math
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_ptz.dataset import validate_tugboat_data_yaml


class TugboatTrainingError(RuntimeError):
    """Raised when a requested local training run is unsafe or invalid."""


def _batch_value(value: str) -> int | float:
    if value.casefold() == "auto":
        return -1
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch must be auto, a positive integer, or 0..1") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("batch must be auto, a positive integer, or 0..1")
    if number.is_integer() and number >= 1.0:
        return int(number)
    if number <= 1.0:
        return number
    raise argparse.ArgumentTypeError("non-integer batch values must be in (0, 1]")


def _load_yolo() -> Any:
    try:
        return import_module("ultralytics").YOLO
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        raise TugboatTrainingError(
            "Ultralytics is unavailable; use the verified constrained vision environment"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tugboat/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=_batch_value, default=-1, metavar="AUTO|N|FRACTION")
    parser.add_argument("--project", type=Path, default=Path("runs/tugboat"))
    parser.add_argument("--name", default="yolo11n-marine-target")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        definition = validate_tugboat_data_yaml(args.data)
        for split, path in (
            ("train", definition.train_images),
            ("val", definition.val_images),
            ("train labels", definition.train_labels),
            ("val labels", definition.val_labels),
        ):
            if not path.is_dir():
                raise TugboatTrainingError(f"prepared {split} directory is missing: {path}")
        if args.epochs <= 0:
            raise TugboatTrainingError("epochs must be a positive integer")
        if args.image_size <= 0:
            raise TugboatTrainingError("image-size must be a positive integer")
        if not args.device.strip() or not args.name.strip():
            raise TugboatTrainingError("device and name must be non-empty")
        run_directory = args.project / args.name
        if run_directory.exists():
            raise TugboatTrainingError(f"refusing to reuse existing training run {run_directory}")
        model_path = Path(args.model)
        if (model_path.is_absolute() or len(model_path.parts) > 1) and not model_path.is_file():
            raise TugboatTrainingError(f"local starting model does not exist: {model_path}")

        yolo = _load_yolo()
        model = yolo(args.model)
        results = model.train(
            data=str(args.data.resolve()),
            epochs=args.epochs,
            imgsz=args.image_size,
            device=args.device,
            batch=args.batch,
            project=str(args.project),
            name=args.name,
            exist_ok=False,
        )
    except KeyboardInterrupt:
        print("training stopped by operator", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"training error: {exc}", file=sys.stderr)
        return 2
    save_dir = getattr(results, "save_dir", run_directory)
    print(f"custom training complete class=marine_target run={save_dir}")
    print(f"best weights expected at {Path(save_dir) / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
