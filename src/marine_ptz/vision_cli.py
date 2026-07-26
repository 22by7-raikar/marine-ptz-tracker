"""Unified OpenCV + Ultralytics PTZ runtime with explicit hardware arming."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
import signal
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from .actuators import SimulatedPTZActuator
from .cancellation import OperationCancelled
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
    actuator_backend: str = "simulated"
    serial_port: str | None = None
    arm_hardware: bool = False


class RuntimeCleanupError(RuntimeError):
    """Raised when bounded cleanup fails, optionally after a primary failure."""


class UnexpectedCancellationError(RuntimeError):
    """Raised when a component cancels without a runtime termination request."""


@dataclass(slots=True)
class _RuntimeResources:
    display: bool
    source: Any | None = None
    actuator: Any | None = None
    writer: Any | None = None
    cv2: Any | None = None
    disable_needed: bool = False
    closed: bool = False

    def close(self) -> tuple[Exception, ...]:
        """Stop motion first, then release every resource exactly once."""
        if self.closed:
            return ()
        self.closed = True
        errors: list[Exception] = []

        if self.disable_needed:
            self.disable_needed = False
            disable = getattr(self.actuator, "disable", None)
            if callable(disable):
                _capture_cleanup_error(disable, errors)

        for cleanup in (
            getattr(self.actuator, "close", None),
            getattr(self.writer, "release", None),
            getattr(self.source, "close", None),
            (
                getattr(self.cv2, "destroyAllWindows", None)
                if self.cv2 is not None and self.display
                else None
            ),
        ):
            if callable(cleanup):
                _capture_cleanup_error(cleanup, errors)
        return tuple(errors)


class _CommandRateLimiter:
    def __init__(
        self,
        update_rate_hz: float,
        *,
        monotonic: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        self._interval_s = 1.0 / update_rate_hz
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_command_s: float | None = None

    def wait(self) -> None:
        now = self._monotonic()
        if not math.isfinite(now):
            raise RuntimeError("monotonic clock returned a non-finite value")
        if self._last_command_s is None:
            self._last_command_s = now
            return
        deadline = self._last_command_s + self._interval_s
        remaining = deadline - now
        if remaining > 0.0:
            self._sleep(remaining)
            self._last_command_s = deadline
        else:
            self._last_command_s = now


class _TerminationRequest:
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def __call__(self) -> bool:
        return self.requested

    def request(self, signal_number: int) -> None:
        self.requested = True
        self.signal_number = signal_number


def _never_terminated() -> bool:
    return False


def run_vision(
    config: AppConfig,
    options: VisionOptions,
    *,
    source_factory: Callable[..., Any] = OpenCVSource,
    detector_factory: Callable[..., Any] = UltralyticsDetector,
    actuator_factory: Callable[[AppConfig, VisionOptions], Any] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    rate_monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    stop_requested: Callable[[], bool] = _never_terminated,
    emit: Callable[[str], None] = print,
) -> int:
    """Run the unified pipeline and return the number of processed frames."""
    _validate_options(config, options)

    resources = _RuntimeResources(display=options.display)
    processed = 0
    frame_times: deque[float] = deque(maxlen=31)
    rate_limiter = _CommandRateLimiter(
        config.tracking.update_rate_hz,
        monotonic=rate_monotonic,
        sleep=sleep,
    )

    try:
        # Startup ordering is safety-critical. Hardware construction and opening
        # happen only after capture, inference, selection, and control are ready.
        source = source_factory(options.source)
        resources.source = source
        _raise_if_terminated(stop_requested)
        detector = detector_factory(
            options.model,
            device=options.device,
            confidence_threshold=options.confidence_threshold,
            iou_threshold=options.iou_threshold,
            image_size=options.image_size,
            class_names=options.target_classes,
        )
        _raise_if_terminated(stop_requested)
        selector = MarineTargetSelector(options.target_classes)
        controller = ProportionalPTZController(
            config.tracking,
            config.actuator.pan_limits,
            config.actuator.tilt_limits,
            initial_pan_deg=config.actuator.initial_pan_deg,
            initial_tilt_deg=config.actuator.initial_tilt_deg,
        )
        _raise_if_terminated(stop_requested)
        if options.display or options.output is not None:
            resources.cv2 = source.cv2

        if actuator_factory is None:
            actuator = _build_actuator(
                config,
                options,
                cancellation_requested=stop_requested,
            )
        else:
            actuator = actuator_factory(config, options)
        resources.actuator = actuator
        if options.actuator_backend == "arduino_serial":
            _bind_cancellation_predicate(actuator, stop_requested)
            _raise_if_terminated(stop_requested)
            _hardware_method(actuator, "open")()
            # Mark before ENABLE: a lost acknowledgement may still mean that
            # firmware attached the servos and must be disabled during cleanup.
            resources.disable_needed = True
            # The handshake succeeded; observe a termination request before
            # progressing to the distinct ENABLE safety boundary.
            _raise_if_terminated(stop_requested)
            # Keep the final check adjacent to the operation that attaches
            # outputs. A request after this point is handled by the actuator's
            # own cancellation predicate before its transport write.
            _raise_if_terminated(stop_requested)
            _hardware_method(actuator, "enable")()

        while options.max_frames is None or processed < options.max_frames:
            _raise_if_terminated(stop_requested)
            frame = source.read()
            if frame is None:
                break
            _raise_if_terminated(stop_requested)
            _validate_runtime_frame(frame)
            detections = detector.detect(frame)
            _raise_if_terminated(stop_requested)
            target = selector.select(frame, detections)
            _raise_if_terminated(stop_requested)
            rate_limiter.wait()
            _raise_if_terminated(stop_requested)
            command = controller.command_for_target(frame, target)
            _raise_if_terminated(stop_requested)
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
                    options.actuator_backend,
                )
            )

            if resources.cv2 is not None:
                rendered = _annotate(
                    resources.cv2,
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
                    if resources.writer is None:
                        resources.writer = _open_writer(
                            resources.cv2,
                            options.output,
                            frame.width,
                            frame.height,
                            config.camera.fps,
                        )
                    resources.writer.write(rendered)
                if options.display:
                    resources.cv2.imshow("Marine PTZ Tracker", rendered)
                    if resources.cv2.waitKey(1) & 0xFF == ord("q"):
                        processed += 1
                        break
            processed += 1
    except OperationCancelled as cancellation:
        cleanup_errors = resources.close()
        if stop_requested():
            if cleanup_errors:
                raise RuntimeCleanupError(
                    f"termination requested; cleanup failed: {cleanup_errors[0]}"
                ) from cleanup_errors[0]
            return processed
        if cleanup_errors:
            raise RuntimeCleanupError(
                "unexpected OperationCancelled without a recorded termination "
                f"request: {cancellation}; cleanup also failed: {cleanup_errors[0]}"
            ) from cancellation
        raise UnexpectedCancellationError(
            "unexpected OperationCancelled without a recorded termination "
            f"request: {cancellation}"
        ) from cancellation
    except BaseException as primary:
        cleanup_errors = resources.close()
        if cleanup_errors:
            raise RuntimeCleanupError(
                f"runtime failed with {type(primary).__name__}: {primary}; "
                f"cleanup also failed: {cleanup_errors[0]}"
            ) from primary
        raise

    cleanup_errors = resources.close()
    if cleanup_errors:
        raise RuntimeCleanupError(
            f"resource cleanup failed: {cleanup_errors[0]}"
        ) from cleanup_errors[0]
    return processed


def _capture_cleanup_error(
    cleanup: Callable[[], Any],
    errors: list[Exception],
) -> None:
    try:
        cleanup()
    except Exception as exc:
        errors.append(exc)


def _raise_if_terminated(stop_requested: Callable[[], bool]) -> None:
    if stop_requested():
        raise OperationCancelled("runtime termination requested")


def _bind_cancellation_predicate(
    actuator: Any,
    stop_requested: Callable[[], bool],
) -> None:
    bind = getattr(actuator, "set_cancellation_predicate", None)
    if callable(bind):
        bind(stop_requested)


def _validate_options(config: AppConfig, options: VisionOptions) -> None:
    if isinstance(options.source, bool):
        raise ValueError("source camera index must not be bool")
    if isinstance(options.source, int):
        if options.source < 0:
            raise ValueError("source camera index must be non-negative")
    elif not isinstance(options.source, str) or not options.source.strip():
        raise ValueError("source must be a camera index or non-empty media path")
    if not isinstance(options.model, str) or not options.model.strip():
        raise ValueError("model must be a non-empty name or local path")
    if not options.target_classes or any(
        not isinstance(name, str) or not name.strip()
        for name in options.target_classes
    ):
        raise ValueError("at least one non-empty target class is required")
    _probability(options.confidence_threshold, "confidence")
    _probability(options.iou_threshold, "IoU")
    if (
        isinstance(options.image_size, bool)
        or not isinstance(options.image_size, int)
        or options.image_size <= 0
    ):
        raise ValueError("image size must be a positive integer")
    if options.max_frames is not None and (
        isinstance(options.max_frames, bool)
        or not isinstance(options.max_frames, int)
        or options.max_frames <= 0
    ):
        raise ValueError("max_frames must be greater than zero when provided")
    if options.output is not None and options.output.exists() and options.output.is_dir():
        raise ValueError("output path must name a file, not a directory")
    if options.actuator_backend not in {"simulated", "arduino_serial"}:
        raise ValueError("actuator backend must be simulated or arduino_serial")
    if options.actuator_backend == "simulated":
        if options.arm_hardware:
            raise ValueError("--arm-hardware is incompatible with simulated actuation")
        if options.serial_port is not None:
            raise ValueError("--serial-port is incompatible with simulated actuation")
        return
    if config.actuator.backend != "arduino_serial":
        raise ConfigError(
            "Arduino actuation requires actuator.backend: arduino_serial "
            "in the validated configuration"
        )
    if not options.arm_hardware:
        raise ValueError("Arduino actuation requires explicit --arm-hardware")
    if config.serial is None:
        raise ConfigError("Arduino actuation requires a validated serial section")
    _resolved_serial_port(config, options)


def _probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return number


def _resolved_serial_port(config: AppConfig, options: VisionOptions) -> str:
    configured = config.serial.device if config.serial is not None else None
    value = options.serial_port if options.serial_port is not None else configured
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Arduino actuation requires an explicit serial port from "
            "--serial-port or validated configuration"
        )
    return value.strip()


def _build_actuator(
    config: AppConfig,
    options: VisionOptions,
    *,
    cancellation_requested: Callable[[], bool] = _never_terminated,
) -> Any:
    if options.actuator_backend == "simulated":
        return SimulatedPTZActuator(
            config.actuator.pan_limits,
            config.actuator.tilt_limits,
            initial_pan_deg=config.actuator.initial_pan_deg,
            initial_tilt_deg=config.actuator.initial_tilt_deg,
        )

    serial = config.serial
    if serial is None:
        raise ConfigError("Arduino actuation requires a validated serial section")
    serial = replace(serial, device=_resolved_serial_port(config, options))
    from .serial_actuator import ArduinoSerialActuator

    return ArduinoSerialActuator(
        serial,
        config.actuator.pan_limits,
        config.actuator.tilt_limits,
        initial_pan_deg=config.actuator.initial_pan_deg,
        initial_tilt_deg=config.actuator.initial_tilt_deg,
        cancellation_requested=cancellation_requested,
    )


def _hardware_method(actuator: Any, method_name: str) -> Callable[[], Any]:
    method = getattr(actuator, method_name, None)
    if not callable(method):
        raise TypeError(
            f"Arduino actuator must provide callable {method_name}()"
        )
    return method


def _validate_runtime_frame(frame: object) -> None:
    if not isinstance(frame, Frame):
        raise RuntimeError("camera source returned a value that is not a Frame")
    if (
        isinstance(frame.width, bool)
        or isinstance(frame.height, bool)
        or not isinstance(frame.width, int)
        or not isinstance(frame.height, int)
        or frame.width <= 0
        or frame.height <= 0
    ):
        raise RuntimeError("camera frame dimensions must be positive integers")
    if (
        isinstance(frame.sequence, bool)
        or not isinstance(frame.sequence, int)
        or frame.sequence < 0
    ):
        raise RuntimeError("camera frame sequence must be a non-negative integer")
    if (
        isinstance(frame.timestamp_s, bool)
        or not isinstance(frame.timestamp_s, Real)
        or not math.isfinite(float(frame.timestamp_s))
    ):
        raise RuntimeError("camera frame timestamp must be finite")
    if frame.image is None:
        raise RuntimeError("camera frame must carry an image")
    shape = getattr(frame.image, "shape", None)
    if not isinstance(shape, tuple) or len(shape) not in {2, 3}:
        raise RuntimeError("camera frame image must have a 2D or 3D shape")
    height, width = shape[:2]
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, Integral)
        or not isinstance(width, Integral)
        or int(height) != frame.height
        or int(width) != frame.width
    ):
        raise RuntimeError("camera frame metadata does not match image shape")


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
    actuator: Any,
    inference_ms: float,
    fps: float,
    device: str,
    actuator_backend: str,
) -> str:
    target_text = "none" if target is None else target.detection.label
    error_text = "none" if error is None else f"{error[0]:.1f},{error[1]:.1f}"
    return (
        f"seq={frame.sequence:06d} detections={len(detections)} target={target_text} "
        f"error_px={error_text} pan={actuator.pan_deg:.2f} tilt={actuator.tilt_deg:.2f} "
        f"inference_ms={inference_ms:.1f} fps={fps:.1f} device={device} "
        f"actuator={actuator_backend}"
    )


def _annotate(
    cv2: Any,
    frame: Frame,
    detections: Sequence[Detection],
    target: Target | None,
    error: tuple[float, float] | None,
    actuator: Any,
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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/development.yaml"),
        help="validated YAML configuration path (default: configs/development.yaml)",
    )
    parser.add_argument(
        "--source",
        help=(
            "camera index (0/camera:0) or media path; defaults to camera.device "
            "from configuration (use path:0 for numeric paths)"
        ),
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
    parser.add_argument(
        "--actuator-backend",
        choices=("simulated", "arduino_serial"),
        help=(
            "actuator selection; defaults to validated configuration "
            "(development defaults to simulated)"
        ),
    )
    parser.add_argument(
        "--serial-port",
        help=(
            "explicit Arduino serial device; never auto-discovered and valid only "
            "with arduino_serial"
        ),
    )
    parser.add_argument(
        "--arm-hardware",
        action="store_true",
        help=(
            "explicitly allow ENABLE after camera, model, controller, and serial "
            "handshake initialization"
        ),
    )
    return parser


def _source_option(cli_value: str | None, config: AppConfig) -> int | str:
    if cli_value is not None:
        return parse_source_argument(cli_value)
    value = config.camera.device
    if value is None:
        raise ValueError(
            "--source is required when camera.device is absent from configuration"
        )
    if config.camera.source == "v4l2" and value.startswith("/dev/video"):
        index = value.removeprefix("/dev/video")
        if index.isdecimal():
            return int(index)
    return parse_source_argument(value)


@contextmanager
def _termination_signals() -> Iterator[_TerminationRequest]:
    request = _TerminationRequest()
    previous: dict[int, Any] = {}

    def handler(signal_number: int, _frame: Any) -> None:
        request.request(signal_number)

    try:
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, handler)
            except (ValueError, OSError):
                previous.pop(signal_number, None)
        yield request
    finally:
        for signal_number, old_handler in previous.items():
            try:
                signal.signal(signal_number, old_handler)
            except (ValueError, OSError):
                pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        options = VisionOptions(
            source=_source_option(args.source, config),
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
            actuator_backend=(
                args.actuator_backend
                if args.actuator_backend is not None
                else config.actuator.backend
            ),
            serial_port=args.serial_port,
            arm_hardware=args.arm_hardware,
        )
        with _termination_signals() as termination:
            run_vision(config, options, stop_requested=termination)
        if termination.signal_number == signal.SIGINT:
            return 130
        if termination.signal_number == signal.SIGTERM:
            return 143
        if termination.signal_number is not None:
            return 128 + termination.signal_number
    except KeyboardInterrupt:
        print("vision interrupted", file=sys.stderr)
        return 130
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
