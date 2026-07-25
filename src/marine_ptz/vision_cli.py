"""Real OpenCV + Ultralytics vision loop with simulated PTZ actuation."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actuators import SimulatedPTZActuator
from .config import AppConfig, ConfigError, load_config
from .opencv_source import OpenCVSource, OpenCVSourceError, parse_source_argument
from .tracking import MarineTargetSelector, ProportionalPTZController
from .types import Detection, Frame, Target
from .yolo_detector import UltralyticsDetector, UltralyticsDetectorError


@dataclass(frozen=True, slots=True)
class VisionOptions:
    source: int | str
    model: str
    device: str
    target_classes: tuple[str, ...]
    confidence_threshold: float
    iou_threshold: float
    image_size: int
    max_frames: int | None
    display: bool
    output: Path | None


def run_vision(
    config: AppConfig,
    options: VisionOptions,
    *,
    source_factory: Callable[..., Any] = OpenCVSource,
    detector_factory: Callable[..., Any] = UltralyticsDetector,
    monotonic: Callable[[], float] = time.monotonic,
    emit: Callable[[str], None] = print,
) -> int:
    """Run the vision pipeline and return the number of processed frames."""
    if options.max_frames is not None and options.max_frames <= 0:
        raise ValueError("max_frames must be greater than zero when provided")

    source: Any | None = None
    actuator: SimulatedPTZActuator | None = None
    writer: Any | None = None
    cv2: Any | None = None
    processed = 0
    frame_times: deque[float] = deque(maxlen=31)

    try:
        source = source_factory(options.source)
        detector = detector_factory(
            options.model,
            device=options.device,
            confidence_threshold=options.confidence_threshold,
            iou_threshold=options.iou_threshold,
            image_size=options.image_size,
            class_names=options.target_classes,
        )
        selector = MarineTargetSelector(options.target_classes)
        controller = ProportionalPTZController(
            config.tracking,
            config.actuator.pan_limits,
            config.actuator.tilt_limits,
            initial_pan_deg=config.actuator.initial_pan_deg,
            initial_tilt_deg=config.actuator.initial_tilt_deg,
        )
        actuator = SimulatedPTZActuator(
            config.actuator.pan_limits,
            config.actuator.tilt_limits,
            initial_pan_deg=config.actuator.initial_pan_deg,
            initial_tilt_deg=config.actuator.initial_tilt_deg,
        )
        if options.display or options.output is not None:
            cv2 = source.cv2

        while options.max_frames is None or processed < options.max_frames:
            frame = source.read()
            if frame is None:
                break
            detections = detector.detect(frame)
            target = selector.select(frame, detections)
            command = controller.command_for_target(frame, target)
            actuator.apply(command)

            now = monotonic()
            frame_times.append(now)
            rolling_fps = _rolling_fps(frame_times)
            error = _target_error(frame, target)
            emit(
                _telemetry_line(
                    frame,
                    detections,
                    target,
                    error,
                    actuator,
                    detector.last_inference_ms,
                    rolling_fps,
                    detector.active_device,
                )
            )

            if cv2 is not None:
                rendered = _annotate(
                    cv2,
                    frame,
                    detections,
                    target,
                    error,
                    actuator,
                    detector.last_inference_ms,
                    rolling_fps,
                    detector.active_device,
                )
                if options.output is not None:
                    if writer is None:
                        writer = _open_writer(
                            cv2,
                            options.output,
                            frame.width,
                            frame.height,
                            config.camera.fps,
                        )
                    writer.write(rendered)
                if options.display:
                    cv2.imshow("Marine PTZ Tracker", rendered)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        processed += 1
                        break
            processed += 1
    finally:
        cleanup_errors: list[Exception] = []
        for cleanup in (
            getattr(writer, "release", None),
            getattr(source, "close", None),
            getattr(actuator, "close", None),
            (
                getattr(cv2, "destroyAllWindows", None)
                if cv2 is not None and options.display
                else None
            ),
        ):
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors and sys.exc_info()[0] is None:
            raise RuntimeError(f"resource cleanup failed: {cleanup_errors[0]}") from cleanup_errors[0]
    return processed


def _rolling_fps(frame_times: deque[float]) -> float:
    if len(frame_times) < 2:
        return 0.0
    elapsed = frame_times[-1] - frame_times[0]
    return (len(frame_times) - 1) / elapsed if elapsed > 0 else 0.0


def _target_error(frame: Frame, target: Target | None) -> tuple[float, float] | None:
    if target is None:
        return None
    center_x, center_y = target.detection.center
    return center_x - frame.width / 2, center_y - frame.height / 2


def _telemetry_line(
    frame: Frame,
    detections: Sequence[Detection],
    target: Target | None,
    error: tuple[float, float] | None,
    actuator: SimulatedPTZActuator,
    inference_ms: float,
    fps: float,
    device: str,
) -> str:
    target_text = "none" if target is None else target.detection.label
    error_text = "none" if error is None else f"{error[0]:.1f},{error[1]:.1f}"
    return (
        f"seq={frame.sequence:06d} detections={len(detections)} target={target_text} "
        f"error_px={error_text} pan={actuator.pan_deg:.2f} tilt={actuator.tilt_deg:.2f} "
        f"inference_ms={inference_ms:.1f} fps={fps:.1f} device={device}"
    )


def _annotate(
    cv2: Any,
    frame: Frame,
    detections: Sequence[Detection],
    target: Target | None,
    error: tuple[float, float] | None,
    actuator: SimulatedPTZActuator,
    inference_ms: float,
    fps: float,
    device: str,
) -> Any:
    if frame.image is None:
        raise ValueError("cannot annotate a frame without an image")
    rendered = frame.image.copy()
    selected = target.detection if target is not None else None
    for detection in detections:
        color = (0, 255, 255) if detection == selected else (0, 200, 0)
        start = (round(detection.left), round(detection.top))
        end = (round(detection.right), round(detection.bottom))
        cv2.rectangle(rendered, start, end, color, 2)
        cv2.putText(
            rendered,
            f"{detection.label} {detection.confidence:.2f}",
            (start[0], max(15, start[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    frame_center = (round(frame.width / 2), round(frame.height / 2))
    cv2.drawMarker(rendered, frame_center, (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
    if target is not None:
        target_center = tuple(round(value) for value in target.detection.center)
        cv2.drawMarker(rendered, target_center, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.line(rendered, frame_center, target_center, (0, 255, 255), 1)

    error_text = "none" if error is None else f"({error[0]:.1f}, {error[1]:.1f}) px"
    lines = (
        f"PTZ pan={actuator.pan_deg:.2f} tilt={actuator.tilt_deg:.2f}",
        f"error={error_text}",
        f"inference={inference_ms:.1f} ms  fps={fps:.1f}  device={device}",
    )
    for index, text in enumerate(lines):
        cv2.putText(
            rendered,
            text,
            (12, 24 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return rendered


def _open_writer(
    cv2: Any,
    output: Path,
    width: int,
    height: int,
    fps: float,
) -> Any:
    writer: Any | None = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    except Exception as exc:
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        raise RuntimeError(f"unable to create annotated video output {output}: {exc}") from exc
    try:
        opened = bool(writer.isOpened())
    except Exception as exc:
        try:
            writer.release()
        except Exception:
            pass
        raise RuntimeError(
            f"unable to inspect annotated video output {output}: {exc}"
        ) from exc
    if not opened:
        writer.release()
        raise RuntimeError(f"unable to open annotated video output {output}")
    return writer


def _target_classes(values: Sequence[str] | None, defaults: Sequence[str]) -> tuple[str, ...]:
    raw = values if values else defaults
    classes = tuple(
        name.strip()
        for value in raw
        for name in value.split(",")
        if name.strip()
    )
    if not classes:
        raise ValueError("at least one target class is required")
    return classes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hardware.yaml"))
    parser.add_argument(
        "--source",
        required=True,
        help="camera index (0/camera:0) or media path (use path:0 for numeric paths)",
    )
    parser.add_argument("--model", help="Ultralytics model name or local path")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--target-class",
        action="append",
        help="repeat or use commas for target class names",
    )
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--output", type=Path, help="annotated video output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        options = VisionOptions(
            source=parse_source_argument(args.source),
            model=args.model or config.detection.model,
            device=args.device or config.detection.device,
            target_classes=_target_classes(
                args.target_class,
                config.detection.target_labels,
            ),
            confidence_threshold=(
                args.confidence
                if args.confidence is not None
                else config.detection.confidence_threshold
            ),
            iou_threshold=(
                args.iou if args.iou is not None else config.detection.iou_threshold
            ),
            image_size=(
                args.image_size
                if args.image_size is not None
                else config.detection.image_size
            ),
            max_frames=args.max_frames,
            display=args.display,
            output=args.output,
        )
        run_vision(config, options)
    except (
        ConfigError,
        OpenCVSourceError,
        UltralyticsDetectorError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"vision error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
