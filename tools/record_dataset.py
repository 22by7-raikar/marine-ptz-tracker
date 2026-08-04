#!/usr/bin/env python3
"""Record a bounded local UVC dataset clip without touching PTZ hardware."""

# ruff: noqa: E402, I001 -- add the src layout before importing the local package.

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_ptz.config import ConfigError, load_config


class DatasetRecordingError(RuntimeError):
    """Raised when a local camera clip cannot be recorded safely."""


@dataclass(frozen=True, slots=True)
class RecordingOptions:
    camera: str
    output: Path
    duration_s: float
    width: int
    height: int
    fps: float
    display: bool = False

    def __post_init__(self) -> None:
        if not self.camera.strip():
            raise ValueError("camera must be a non-empty explicit path")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration must be a finite number greater than zero")
        if self.output.suffix.casefold() != ".avi":
            raise ValueError("MJPEG dataset recordings must use an .avi output path")
        if self.output.exists():
            raise ValueError(f"refusing to overwrite existing recording {self.output}")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("width, height, and fps must be greater than zero")


def record_clip(
    options: RecordingOptions,
    *,
    cv2_module: Any | None = None,
    monotonic: Any = time.monotonic,
    emit: Any = print,
) -> int:
    """Record one MJPEG clip and return the number of frames written."""
    cv2 = cv2_module if cv2_module is not None else _load_cv2()
    capture: Any | None = None
    writer: Any | None = None
    frames_written = 0
    try:
        backend = getattr(cv2, "CAP_V4L2", None)
        capture = (
            cv2.VideoCapture(options.camera, backend)
            if backend is not None
            else cv2.VideoCapture(options.camera)
        )
        if not bool(capture.isOpened()):
            raise DatasetRecordingError(f"unable to open camera {options.camera!r}")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        requests = (
            (cv2.CAP_PROP_FOURCC, fourcc),
            (cv2.CAP_PROP_FRAME_WIDTH, float(options.width)),
            (cv2.CAP_PROP_FRAME_HEIGHT, float(options.height)),
            (cv2.CAP_PROP_FPS, float(options.fps)),
        )
        for property_id, value in requests:
            capture.set(property_id, value)

        options.output.parent.mkdir(parents=True, exist_ok=True)
        started = monotonic()
        while monotonic() - started < options.duration_s:
            result = capture.read()
            if not isinstance(result, tuple) or len(result) != 2:
                raise DatasetRecordingError("camera returned an invalid read result")
            ok, image = result
            if not ok or image is None:
                raise DatasetRecordingError("camera failed while recording a frame")
            height, width = _image_dimensions(image)
            if (width, height) != (options.width, options.height):
                raise DatasetRecordingError(
                    "camera did not provide the requested frame size: "
                    f"expected {options.width}x{options.height}, received {width}x{height}"
                )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(options.output),
                    fourcc,
                    float(options.fps),
                    (options.width, options.height),
                )
                if not bool(writer.isOpened()):
                    raise DatasetRecordingError(f"unable to open MJPEG writer for {options.output}")
            writer.write(image)
            frames_written += 1
            if options.display:
                cv2.imshow("Tugboat dataset recording - q/Esc to stop", image)
                key = int(cv2.waitKey(1)) & 0xFF
                if key in {ord("q"), 27}:
                    break
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        if options.display:
            cv2.destroyAllWindows()
    if frames_written == 0:
        raise DatasetRecordingError("recording ended before any frame was written")
    emit(
        f"recorded frames={frames_written} output={options.output} "
        f"requested={options.width}x{options.height}@{options.fps:g} MJPEG"
    )
    return frames_written


def _image_dimensions(image: object) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if not isinstance(shape, tuple) or len(shape) not in {2, 3}:
        raise DatasetRecordingError("camera image must have a 2D or 3D shape")
    height, width = shape[:2]
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise DatasetRecordingError("camera image dimensions must be positive integers")
    return height, width


def _load_cv2() -> Any:
    try:
        return import_module("cv2")
    except (ImportError, ModuleNotFoundError) as exc:
        raise DatasetRecordingError(
            "OpenCV is unavailable; install the constrained vision dependencies"
        ) from exc


def _default_output() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path("captures/tugboat") / f"tugboat-{timestamp}.avi"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hardware.yaml"))
    parser.add_argument("--camera", help="explicit UVC path; defaults to camera.device")
    parser.add_argument("--output", type=Path, help="new .avi path; never overwritten")
    parser.add_argument("--duration", type=float, default=30.0, help="recording seconds")
    parser.add_argument("--display", action="store_true", help="show opt-in local preview")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if config.camera.device is None and args.camera is None:
            raise ConfigError("camera.device is required when --camera is omitted")
        options = RecordingOptions(
            camera=args.camera or str(config.camera.device),
            output=args.output or _default_output(),
            duration_s=args.duration,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            display=args.display,
        )
        record_clip(options)
    except KeyboardInterrupt:
        print("\nrecording stopped by operator; resources released", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"recording error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
