"""Unified OpenCV + Ultralytics PTZ runtime with explicit hardware arming."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn, Protocol

from .actuators import SimulatedPTZActuator
from .cancellation import OperationCancelled
from .concurrent_runtime import (
    ConcurrentRuntimeMetrics,
    DetectionPacket,
    FramePacket,
    LatestValueChannel,
    MonotonicControlSchedule,
    MonotonicPlaybackSchedule,
    RuntimeMetricsCollector,
    SourceMetadata,
    WorkerFailureBox,
)
from .config import AppConfig, ConfigError, load_config
from .digital_zoom import DigitalZoomController, ZoomTransform, identity_transform
from .opencv_source import OpenCVSource, OpenCVSourceError, parse_source_argument
from .tracking import MarineTargetSelector, ProportionalPTZController
from .types import AngleLimits, Detection, Frame, PTZCommand, Target
from .video_recorder import WallClockVideoRecorder
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
    digital_zoom: bool = False
    max_digital_zoom: float = 2.0
    output_fps: float | None = None
    runtime_mode: str = "single"
    result_freshness_s: float | None = None


class RuntimeCleanupError(RuntimeError):
    """Raised when bounded cleanup fails, optionally after a primary failure."""


class UnexpectedCancellationError(RuntimeError):
    """Raised when a component cancels without a runtime termination request."""


class StalePerceptionError(RuntimeError):
    """Raised when concurrent perception can no longer safely drive motion."""


class RuntimeSecondaryFailures(RuntimeError):
    """Inspectable observer and cleanup failures attached to a primary error.

    Python 3.10 has neither ``ExceptionGroup`` nor ``BaseException.add_note``.
    This exception is therefore used as the chained diagnostic context while
    the control-path exception remains the exception raised to the caller.
    """

    def __init__(
        self,
        *,
        observer_error: BaseException | None,
        cleanup_errors: tuple[Exception, ...],
    ) -> None:
        self.observer_error = observer_error
        self.cleanup_errors = cleanup_errors
        message = (
            f"processing-end observer failed: {observer_error}"
            if observer_error is not None
            else "resource cleanup failed"
        )
        if cleanup_errors:
            message += f"; cleanup also failed: {cleanup_errors[0]}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RuntimeStartEvent:
    """Immutable production-runtime snapshot emitted before frame processing."""

    processing_started_s: float
    source_type: str
    total_frames_available: int | None
    resolved_device: str
    actuator_backend: str
    initial_pan_deg: float
    initial_tilt_deg: float
    pan_limits: AngleLimits
    tilt_limits: AngleLimits


@dataclass(frozen=True, slots=True)
class RuntimeFrameEvent:
    """Immutable snapshot of one successfully applied production-loop command."""

    processing_timestamp_s: float
    frame_capture_timestamp_s: float
    frame_sequence: int
    frame_width: int
    frame_height: int
    detection_count: int
    selected_detection: Detection | None
    command: PTZCommand
    actuator_pan_deg: float
    actuator_tilt_deg: float
    inference_ms: float
    resolved_device: str


@dataclass(frozen=True, slots=True)
class RuntimeProcessingEndEvent:
    """Immutable boundary between frame processing and bounded cleanup."""

    processing_ended_s: float
    processed_frames: int
    exit_reason: str


@dataclass(frozen=True, slots=True)
class RuntimeCompleteEvent:
    """Immutable normal-completion snapshot emitted after cleanup succeeds."""

    processed_frames: int
    exit_reason: str


class RuntimeObserver(Protocol):
    """Observe immutable snapshots without access to runtime components."""

    def on_start(self, event: RuntimeStartEvent) -> None:
        """Observe the processing-start boundary."""

    def on_frame(self, event: RuntimeFrameEvent) -> None:
        """Observe one successfully applied command."""

    def on_processing_end(self, event: RuntimeProcessingEndEvent) -> None:
        """Observe the pre-cleanup processing-end boundary."""

    def on_complete(self, event: RuntimeCompleteEvent) -> None:
        """Observe normal completion after bounded cleanup succeeds."""


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


@dataclass(frozen=True, slots=True)
class _RenderedPacket:
    image: Any
    timestamp_s: float


@dataclass(slots=True)
class _CaptureWorkerState:
    source: Any | None = None
    cv2: Any | None = None
    metadata: SourceMetadata | None = None
    outcome: str = "not_started"
    playback_started_s: float | None = None
    detector_readiness_s: float | None = None


@dataclass(slots=True)
class _RecorderWorkerState:
    recorder: Any
    started_s: float | None = None
    finished_s: float | None = None
    consumed_frames: int = 0
    encoded_frames: int = 0
    duplicated_frames: int = 0
    consumed_version: int = 0
    completed: threading.Event = field(default_factory=threading.Event, repr=False)
    condition: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def mark_consumed(self, version: int) -> None:
        with self.condition:
            self.consumed_version = version
            self.consumed_frames = int(self.recorder.submitted_frame_count)
            self.condition.notify_all()

    def wait_until_consumed(self, version: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while self.consumed_version < version and not self.completed.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self.condition.wait(remaining)
            return self.consumed_version >= version

    def mark_completed(self) -> None:
        with self.condition:
            self.encoded_frames = int(self.recorder.encoded_frame_count)
            self.duplicated_frames = int(self.recorder.duplicated_frame_count)
            self.consumed_frames = int(self.recorder.submitted_frame_count)
            self.completed.set()
            self.condition.notify_all()


@dataclass(frozen=True, slots=True)
class _ConcurrentRates:
    unique_inference_fps: float
    display_fps: float
    control_hz: float
    encoded_output_fps: float | None


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
    observer: RuntimeObserver | None = None,
    concurrent_metrics_sink: Callable[[ConcurrentRuntimeMetrics], None] | None = None,
    worker_join_timeout_s: float = 2.0,
) -> int:
    """Run the selected pipeline and return its processed-frame count."""
    _validate_options(config, options)
    if options.runtime_mode == "concurrent":
        return _run_vision_concurrent(
            config,
            options,
            source_factory=source_factory,
            detector_factory=detector_factory,
            actuator_factory=actuator_factory,
            monotonic=monotonic,
            rate_monotonic=rate_monotonic,
            sleep=sleep,
            stop_requested=stop_requested,
            emit=emit,
            observer=observer,
            metrics_sink=concurrent_metrics_sink,
            worker_join_timeout_s=worker_join_timeout_s,
        )
    return _run_vision_single(
        config,
        options,
        source_factory=source_factory,
        detector_factory=detector_factory,
        actuator_factory=actuator_factory,
        monotonic=monotonic,
        rate_monotonic=rate_monotonic,
        sleep=sleep,
        stop_requested=stop_requested,
        emit=emit,
        observer=observer,
    )


def _run_vision_single(
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
    observer: RuntimeObserver | None = None,
) -> int:
    """Run the verified single-threaded pipeline."""

    resources = _RuntimeResources(display=options.display)
    processed = 0
    frame_times: deque[float] = deque(maxlen=31)
    processing_started_s: float | None = None
    processing_end_notified = False
    observer_started = False
    prepared_frame: tuple[Frame, Sequence[Detection], Target | None] | None = None
    pre_enable_eof = False
    rate_limiter = _CommandRateLimiter(
        config.tracking.update_rate_hz,
        monotonic=rate_monotonic,
        sleep=sleep,
    )
    zoom_controller = DigitalZoomController(
        enabled=options.digital_zoom,
        max_zoom=options.max_digital_zoom,
    )
    output_fps = _output_fps(config, options)

    def finish_processing(reason: str) -> None:
        nonlocal processing_end_notified
        if processing_end_notified or processing_started_s is None:
            return
        processing_end_notified = True
        if observer is not None and observer_started:
            observer.on_processing_end(
                RuntimeProcessingEndEvent(
                    processing_ended_s=_runtime_timestamp(monotonic),
                    processed_frames=processed,
                    exit_reason=reason,
                )
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

        if options.actuator_backend == "arduino_serial":
            # The first Ultralytics prediction performs cold initialization and
            # may exceed the firmware watchdog. Keep outputs detached until one
            # real frame has validated the camera and completed that prediction.
            _raise_if_terminated(stop_requested)
            frame = source.read()
            if frame is None:
                pre_enable_eof = True
                emit("source reached EOF during pre-enable validation; actuator remains disabled")
            else:
                _raise_if_terminated(stop_requested)
                _validate_runtime_frame(frame)
                detections = detector.detect(frame)
                _raise_if_terminated(stop_requested)
                target = selector.select(frame, detections)
                _raise_if_terminated(stop_requested)
                prepared_frame = (frame, detections, target)
                if resources.cv2 is not None:
                    if options.display:
                        resources.cv2.namedWindow("Marine PTZ Tracker")
                    if options.output is not None:
                        resources.writer = _open_writer(
                            resources.cv2,
                            options.output,
                            frame.width,
                            frame.height,
                            output_fps,
                        )
                emit(
                    "initialization complete: camera validated, model warm-up complete, "
                    "display/output ready"
                )

        actuator: Any | None = None
        if not pre_enable_eof:
            if actuator_factory is None:
                actuator = _build_actuator(
                    config,
                    options,
                    cancellation_requested=stop_requested,
                )
            else:
                actuator = actuator_factory(config, options)
            resources.actuator = actuator
        if observer is not None:
            processing_started_s = _runtime_timestamp(monotonic)
            observer.on_start(
                RuntimeStartEvent(
                    processing_started_s=processing_started_s,
                    source_type=_source_type(source, options.source),
                    total_frames_available=_total_frames_available(source),
                    resolved_device=str(detector.active_device),
                    actuator_backend=options.actuator_backend,
                    initial_pan_deg=float(
                        controller.pan_deg if actuator is None else actuator.pan_deg
                    ),
                    initial_tilt_deg=float(
                        controller.tilt_deg if actuator is None else actuator.tilt_deg
                    ),
                    pan_limits=config.actuator.pan_limits,
                    tilt_limits=config.actuator.tilt_limits,
                )
            )
            observer_started = True
        if options.actuator_backend == "arduino_serial" and actuator is not None:
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
            emit("actuator handshake complete; sending ENABLE")
            _hardware_method(actuator, "enable")()
            _raise_if_terminated(stop_requested)
            if prepared_frame is None:
                raise RuntimeError("hardware actuator enabled without a prepared camera frame")
            first_frame = prepared_frame[0]
            first_hold = controller.command_for_target(first_frame, None)
            _raise_if_terminated(stop_requested)
            rate_limiter.wait()
            _raise_if_terminated(stop_requested)
            emit(
                f"sending first hold SET pan={first_hold.pan_deg:.2f} "
                f"tilt={first_hold.tilt_deg:.2f}"
            )
            actuator.apply(first_hold)

        while True:
            if options.max_frames is not None and processed >= options.max_frames:
                exit_reason = "max_frames"
                break
            if pre_enable_eof:
                exit_reason = "eof"
                break
            _raise_if_terminated(stop_requested)
            if prepared_frame is not None:
                frame, detections, target = prepared_frame
                prepared_frame = None
            else:
                frame = source.read()
                if frame is None:
                    exit_reason = "eof"
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
            if actuator is None:
                raise RuntimeError("runtime actuator is unavailable")
            actuator.apply(command)

            now = _runtime_timestamp(monotonic) if observer is not None else monotonic()
            event = RuntimeFrameEvent(
                processing_timestamp_s=now,
                frame_capture_timestamp_s=float(frame.timestamp_s),
                frame_sequence=frame.sequence,
                frame_width=frame.width,
                frame_height=frame.height,
                detection_count=len(detections),
                selected_detection=None if target is None else target.detection,
                command=command,
                actuator_pan_deg=float(actuator.pan_deg),
                actuator_tilt_deg=float(actuator.tilt_deg),
                inference_ms=float(detector.last_inference_ms),
                resolved_device=str(detector.active_device),
            )
            if observer is not None:
                observer.on_frame(event)

            frame_times.append(now)
            rolling_fps = _rolling_fps(frame_times)
            error = _target_error(frame, target)
            zoom_transform = zoom_controller.update(
                frame.width,
                frame.height,
                None if target is None else target.detection,
            )
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
                    config.actuator.pan_limits,
                    config.actuator.tilt_limits,
                    zoom_transform.zoom_factor,
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
                    config.actuator.pan_limits,
                    config.actuator.tilt_limits,
                    zoom_transform,
                )
                if options.output is not None:
                    if resources.writer is None:
                        resources.writer = _open_writer(
                            resources.cv2,
                            options.output,
                            frame.width,
                            frame.height,
                            output_fps,
                        )
                    resources.writer.write(rendered, timestamp_s=now)
                if options.display:
                    resources.cv2.imshow("Marine PTZ Tracker", rendered)
                    if resources.cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                        processed += 1
                        exit_reason = "display_q"
                        break
            processed += 1
    except OperationCancelled as cancellation:
        observer_error: BaseException | None = None
        try:
            finish_processing("cancelled" if stop_requested() else "failure")
        except BaseException as error:
            observer_error = error
        cleanup_errors = resources.close()
        if stop_requested():
            # Cancellation is expected only while the shared predicate is
            # true. An observer failure is still a lifecycle failure and is
            # deliberately stronger than that otherwise-normal cancellation.
            if observer_error is not None:
                if cleanup_errors:
                    _raise_primary_with_secondary_failures(
                        observer_error,
                        observer_error.__traceback__,
                        None,
                        cleanup_errors,
                    )
                raise observer_error from cancellation
            if cleanup_errors:
                raise RuntimeCleanupError(
                    f"termination requested; cleanup failed: {cleanup_errors[0]}"
                ) from cleanup_errors[0]
            return processed
        if observer_error is not None:
            unexpected = UnexpectedCancellationError(
                "unexpected OperationCancelled without a recorded termination "
                f"request: {cancellation}"
            )
            _raise_primary_with_secondary_failures(
                unexpected,
                unexpected.__traceback__,
                observer_error,
                cleanup_errors,
            )
        if cleanup_errors:
            raise RuntimeCleanupError(
                "unexpected OperationCancelled without a recorded termination "
                f"request: {cancellation}; cleanup also failed: {cleanup_errors[0]}"
            ) from cancellation
        raise UnexpectedCancellationError(
            f"unexpected OperationCancelled without a recorded termination request: {cancellation}"
        ) from cancellation
    except BaseException as primary:
        primary_traceback = primary.__traceback__
        observer_error: BaseException | None = None
        try:
            finish_processing("failure")
        except BaseException as error:
            observer_error = error
        cleanup_errors = resources.close()
        if observer_error is not None:
            _raise_primary_with_secondary_failures(
                primary,
                primary_traceback,
                observer_error,
                cleanup_errors,
            )
        if cleanup_errors:
            raise RuntimeCleanupError(
                f"runtime failed with {type(primary).__name__}: {primary}; "
                f"cleanup also failed: {cleanup_errors[0]}"
            ) from primary
        raise

    try:
        finish_processing(exit_reason)
    except BaseException as observer_error:
        cleanup_errors = resources.close()
        if cleanup_errors:
            _raise_primary_with_secondary_failures(
                observer_error,
                observer_error.__traceback__,
                None,
                cleanup_errors,
            )
        raise
    cleanup_errors = resources.close()
    if cleanup_errors:
        raise RuntimeCleanupError(
            f"resource cleanup failed: {cleanup_errors[0]}"
        ) from cleanup_errors[0]
    if observer is not None:
        observer.on_complete(RuntimeCompleteEvent(processed, exit_reason))
    return processed


def _run_vision_concurrent(
    config: AppConfig,
    options: VisionOptions,
    *,
    source_factory: Callable[..., Any],
    detector_factory: Callable[..., Any],
    actuator_factory: Callable[[AppConfig, VisionOptions], Any] | None,
    monotonic: Callable[[], float],
    rate_monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    stop_requested: Callable[[], bool],
    emit: Callable[[str], None],
    observer: RuntimeObserver | None,
    metrics_sink: Callable[[ConcurrentRuntimeMetrics], None] | None,
    worker_join_timeout_s: float,
) -> int:
    """Run capture, inference, rendering, and control with bounded handoffs."""
    if observer is not None:
        raise ValueError(
            "runtime observers currently require --runtime-mode single; "
            "concurrent metrics use concurrent_metrics_sink"
        )
    if (
        isinstance(worker_join_timeout_s, bool)
        or not isinstance(worker_join_timeout_s, Real)
        or not math.isfinite(float(worker_join_timeout_s))
        or worker_join_timeout_s <= 0.0
    ):
        raise ValueError("worker join timeout must be a finite positive number")

    freshness_s = (
        config.runtime.result_freshness_s
        if options.result_freshness_s is None
        else float(options.result_freshness_s)
    )
    cancellation = threading.Event()
    detector_ready = threading.Event()
    frame_channel: LatestValueChannel[FramePacket] = LatestValueChannel()
    detection_channel: LatestValueChannel[DetectionPacket] = LatestValueChannel()
    render_channel: LatestValueChannel[_RenderedPacket] = LatestValueChannel()
    failures = WorkerFailureBox()
    metrics = RuntimeMetricsCollector()
    capture_state = _CaptureWorkerState()
    recorder_state: _RecorderWorkerState | None = None
    threads: list[threading.Thread] = []
    actuator: Any | None = None
    cv2: Any | None = None
    display_initialized = False
    disable_needed = False
    processed = 0
    recorder_submitted_frames = 0
    primary: BaseException | None = None
    primary_traceback: TracebackType | None = None
    cleanup_errors: list[Exception] = []

    selector = MarineTargetSelector(options.target_classes)
    controller = ProportionalPTZController(
        config.tracking,
        config.actuator.pan_limits,
        config.actuator.tilt_limits,
        initial_pan_deg=config.actuator.initial_pan_deg,
        initial_tilt_deg=config.actuator.initial_tilt_deg,
    )
    zoom_controller = DigitalZoomController(
        enabled=options.digital_zoom,
        max_zoom=options.max_digital_zoom,
    )
    frame_times: deque[float] = deque(maxlen=31)
    output_fps = _output_fps(config, options)

    try:
        capture_thread = threading.Thread(
            name="marine-ptz-capture",
            target=_capture_worker,
            kwargs={
                "configured_source": options.source,
                "source_factory": source_factory,
                "channel": frame_channel,
                "state": capture_state,
                "failures": failures,
                "cancellation": cancellation,
                "detector_ready": detector_ready,
                "monotonic": monotonic,
                "sleep": sleep,
                "metrics": metrics,
            },
        )
        threads.append(capture_thread)
        capture_thread.start()

        # Validate one camera frame before the detector worker constructs the
        # model. Physical actuation still does not exist at this point.
        first_frame_value = _wait_for_channel_value(
            frame_channel,
            failures,
            cancellation,
            stop_requested,
            mark_consumed=False,
        )
        if first_frame_value is None:
            pass
        else:
            inference_thread = threading.Thread(
                name="marine-ptz-inference",
                target=_inference_worker,
                kwargs={
                    "options": options,
                    "detector_factory": detector_factory,
                    "frames": frame_channel,
                    "results": detection_channel,
                    "failures": failures,
                    "cancellation": cancellation,
                    "detector_ready": detector_ready,
                    "monotonic": monotonic,
                    "metrics": metrics,
                },
            )
            threads.append(inference_thread)
            inference_thread.start()

            first_result_value = _wait_for_channel_value(
                detection_channel,
                failures,
                cancellation,
                stop_requested,
            )
            if first_result_value is None:
                pass
            else:
                current_result = first_result_value.value
                result_version = first_result_value.version
                first_frame = current_result.frame_packet.frame
                cv2 = capture_state.cv2
                if options.display or options.output is not None:
                    if cv2 is None:
                        raise RuntimeError("camera source did not expose OpenCV rendering support")
                    if options.display:
                        cv2.namedWindow("Marine PTZ Tracker")
                        display_initialized = True
                    if options.output is not None:
                        recorder_state = _RecorderWorkerState(
                            _open_writer(
                                cv2,
                                options.output,
                                first_frame.width,
                                first_frame.height,
                                output_fps,
                            )
                        )
                        recorder_thread = threading.Thread(
                            name="marine-ptz-recorder",
                            target=_recorder_worker,
                            kwargs={
                                "channel": render_channel,
                                "state": recorder_state,
                                "failures": failures,
                                "cancellation": cancellation,
                            },
                        )
                        threads.append(recorder_thread)
                        recorder_thread.start()

                emit(
                    "initialization complete: camera validated, model warm-up complete, "
                    "display/output ready; runtime=concurrent"
                )
                _raise_concurrent_failure(failures, stop_requested)
                _raise_if_concurrent_cancelled(cancellation, stop_requested)

                def cancellation_predicate() -> bool:
                    return cancellation.is_set() or stop_requested()

                actuator = (
                    _build_actuator(
                        config,
                        options,
                        cancellation_requested=cancellation_predicate,
                    )
                    if actuator_factory is None
                    else actuator_factory(config, options)
                )
                _bind_cancellation_predicate(actuator, cancellation_predicate)

                now_rate = _runtime_timestamp(rate_monotonic)
                control_schedule = MonotonicControlSchedule(
                    config.tracking.update_rate_hz,
                    first_deadline_s=now_rate,
                )
                if options.actuator_backend == "arduino_serial":
                    _raise_if_concurrent_cancelled(cancellation, stop_requested)
                    _hardware_method(actuator, "open")()
                    disable_needed = True
                    _raise_if_concurrent_cancelled(cancellation, stop_requested)
                    emit("actuator handshake complete; sending ENABLE")
                    _hardware_method(actuator, "enable")()
                    _raise_if_concurrent_cancelled(cancellation, stop_requested)
                    first_hold = controller.command_for_target(first_frame, None)
                    emit(
                        f"sending first hold SET pan={first_hold.pan_deg:.2f} "
                        f"tilt={first_hold.tilt_deg:.2f}"
                    )
                    _apply_concurrent_command(
                        actuator,
                        first_hold,
                        current_result,
                        cancellation,
                        stop_requested,
                        failures,
                        monotonic,
                        metrics,
                        record_session_fault=True,
                    )
                    control_schedule.command_completed(_runtime_timestamp(rate_monotonic))

                last_rendered_sequence: int | None = None
                while True:
                    if stop_requested():
                        cancellation.set()
                        raise OperationCancelled("runtime termination requested")
                    _raise_concurrent_failure(failures, stop_requested)
                    if cancellation.is_set():
                        raise OperationCancelled("concurrent runtime cancelled")

                    newest = detection_channel.peek()
                    if newest is not None and newest.version > result_version:
                        current_result = newest.value
                        result_version = newest.version

                    stop_after_control = False
                    if current_result.frame_sequence != last_rendered_sequence:
                        frame = current_result.frame_packet.frame
                        detections = current_result.detections
                        target = selector.select(frame, detections)
                        now = _runtime_timestamp(monotonic)
                        frame_times.append(now)
                        rolling_fps = _rolling_fps(frame_times)
                        concurrent_rates = _ConcurrentRates(
                            unique_inference_fps=(metrics.event_rate("inference").rate_hz or 0.0),
                            display_fps=rolling_fps,
                            control_hz=metrics.event_rate("command").rate_hz or 0.0,
                            encoded_output_fps=(output_fps if options.output is not None else None),
                        )
                        error = _target_error(frame, target)
                        zoom_transform = zoom_controller.update(
                            frame.width,
                            frame.height,
                            None if target is None else target.detection,
                        )
                        emit(
                            _telemetry_line(
                                frame,
                                detections,
                                target,
                                error,
                                actuator,
                                current_result.inference_ms,
                                rolling_fps,
                                current_result.resolved_device,
                                options.actuator_backend,
                                config.actuator.pan_limits,
                                config.actuator.tilt_limits,
                                zoom_transform.zoom_factor,
                                concurrent_rates,
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
                                current_result.inference_ms,
                                rolling_fps,
                                current_result.resolved_device,
                                config.actuator.pan_limits,
                                config.actuator.tilt_limits,
                                zoom_transform,
                                concurrent_rates,
                            )
                            if options.output is not None:
                                render_version = render_channel.publish(
                                    _RenderedPacket(rendered, now)
                                )
                                recorder_submitted_frames += 1
                                if (
                                    capture_state.outcome == "normal_eof"
                                    and recorder_state is not None
                                    and not recorder_state.wait_until_consumed(
                                        render_version,
                                        float(worker_join_timeout_s),
                                    )
                                ):
                                    _raise_concurrent_failure(failures, stop_requested)
                                    raise RuntimeError(
                                        "recorder drain deadline expired before the final "
                                        "EOF frame was consumed"
                                    )
                            if options.display:
                                cv2.imshow("Marine PTZ Tracker", rendered)
                                if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                                    metrics.record_event("display", now)
                                    processed += 1
                                    last_rendered_sequence = current_result.frame_sequence
                                    cancellation.set()
                                    break
                        metrics.record_event("display", now)
                        processed += 1
                        last_rendered_sequence = current_result.frame_sequence
                        if options.max_frames is not None and processed >= options.max_frames:
                            stop_after_control = True

                    # Inference may complete while annotation/display runs. Use
                    # that newest result for the scheduled control update, but
                    # leave it for the next render iteration as well.
                    if not stop_after_control:
                        newest = detection_channel.peek()
                        if newest is not None and newest.version > result_version:
                            current_result = newest.value
                            result_version = newest.version

                    now_rate = _runtime_timestamp(rate_monotonic)
                    if control_schedule.is_due(now_rate):
                        now = _runtime_timestamp(monotonic)
                        result_age_s = now - current_result.inference_completed_s
                        if result_age_s < 0.0:
                            raise RuntimeError("monotonic clock moved backwards")
                        if not current_result.healthy or result_age_s > freshness_s:
                            metrics.increment_stale()
                            raise StalePerceptionError(
                                "latest perception result is stale or unhealthy; "
                                "stopping SET commands and disabling actuation"
                            )
                        frame = current_result.frame_packet.frame
                        target = selector.select(frame, current_result.detections)
                        command = controller.command_for_target(frame, target)
                        _apply_concurrent_command(
                            actuator,
                            command,
                            current_result,
                            cancellation,
                            stop_requested,
                            failures,
                            monotonic,
                            metrics,
                            record_session_fault=(options.actuator_backend == "arduino_serial"),
                        )
                        # Anchor to completion, not the pre-SET deadline. A slow
                        # acknowledgement skips missed slots and cannot cause a
                        # burst or a second rate-limit sleep on the next SET.
                        control_schedule.command_completed(_runtime_timestamp(rate_monotonic))

                    if stop_after_control:
                        cancellation.set()
                        break

                    wait_s = control_schedule.bounded_wait(
                        _runtime_timestamp(rate_monotonic),
                        0.05,
                    )
                    next_value = detection_channel.latest(
                        after_version=result_version,
                        timeout_s=wait_s,
                    )
                    if next_value is not None:
                        current_result = next_value.value
                        result_version = next_value.version
                        continue
                    if detection_channel.closed:
                        if current_result.frame_sequence != last_rendered_sequence:
                            continue
                        break
    except BaseException as error:
        primary = error
        primary_traceback = error.__traceback__

    shutdown_started_s = time.monotonic()
    drain_normal_eof = (
        primary is None
        and capture_state.outcome == "normal_eof"
        and recorder_state is not None
        and not cancellation.is_set()
    )
    if not drain_normal_eof:
        cancellation.set()
    frame_channel.close()
    detection_channel.close()
    render_channel.close()

    if disable_needed and actuator is not None:
        disable_needed = False
        disable = getattr(actuator, "disable", None)
        if callable(disable):
            _capture_cleanup_error(disable, cleanup_errors)
    close_actuator = getattr(actuator, "close", None)
    if callable(close_actuator):
        _capture_cleanup_error(close_actuator, cleanup_errors)

    if (
        drain_normal_eof
        and recorder_state is not None
        and not recorder_state.completed.wait(float(worker_join_timeout_s))
    ):
        cleanup_errors.append(RuntimeError("recorder EOF drain deadline expired"))
    cancellation.set()

    _join_concurrent_workers(
        threads,
        capture_state,
        float(worker_join_timeout_s),
        cleanup_errors,
    )
    worker_failure = failures.get()
    if primary is None and worker_failure is not None:
        if isinstance(worker_failure.error, OperationCancelled) and stop_requested():
            primary = worker_failure.error
        elif isinstance(worker_failure.error, OperationCancelled):
            primary = UnexpectedCancellationError(
                f"{worker_failure.stage} worker cancelled without a recorded "
                f"termination request: {worker_failure.error}"
            )
            primary.__cause__ = worker_failure.error
        else:
            primary = RuntimeError(f"{worker_failure.stage} worker failed: {worker_failure.error}")
            primary.__cause__ = worker_failure.error
        primary_traceback = primary.__traceback__
    elif (
        primary is not None
        and worker_failure is not None
        and not _exception_chain_contains(primary, worker_failure.error)
    ):
        cleanup_errors.append(
            RuntimeError(f"{worker_failure.stage} worker also failed: {worker_failure.error}")
        )
    if cv2 is not None and display_initialized:
        destroy = getattr(cv2, "destroyAllWindows", None)
        if callable(destroy):
            _capture_cleanup_error(destroy, cleanup_errors)
    shutdown_duration_s = time.monotonic() - shutdown_started_s

    consumed_frames = 0 if recorder_state is None else recorder_state.consumed_frames
    encoded_frames = 0 if recorder_state is None else recorder_state.encoded_frames
    duplicated_frames = 0 if recorder_state is None else recorder_state.duplicated_frames
    report = metrics.snapshot(
        dropped_capture_frames=frame_channel.overwritten_count,
        dropped_render_frames=detection_channel.overwritten_count,
        recorder_submitted_frames=recorder_submitted_frames,
        recorder_consumed_frames=consumed_frames,
        dropped_recorder_frames=render_channel.overwritten_count,
        encoded_frame_count=encoded_frames,
        encoded_duplicate_frames=duplicated_frames,
        encoded_started_s=None if recorder_state is None else recorder_state.started_s,
        encoded_finished_s=None if recorder_state is None else recorder_state.finished_s,
        shutdown_duration_s=shutdown_duration_s,
        source_type=(
            "unknown" if capture_state.metadata is None else capture_state.metadata.source_type
        ),
        declared_source_fps=(
            None if capture_state.metadata is None else capture_state.metadata.declared_fps
        ),
        pacing_applied=(
            False if capture_state.metadata is None else capture_state.metadata.pacing_applied
        ),
        capture_outcome=capture_state.outcome,
        capture_playback_started_s=capture_state.playback_started_s,
        detector_readiness_s=capture_state.detector_readiness_s,
    )
    try:
        emit(f"runtime_metrics={json.dumps(report.to_dict(), sort_keys=True)}")
        if metrics_sink is not None:
            metrics_sink(report)
    except Exception as error:
        if primary is None:
            primary = error
            primary_traceback = error.__traceback__
        else:
            cleanup_errors.append(error)

    if primary is not None:
        if isinstance(primary, OperationCancelled) and stop_requested():
            if cleanup_errors:
                raise RuntimeCleanupError(
                    f"termination requested; cleanup failed: {cleanup_errors[0]}"
                ) from cleanup_errors[0]
            return processed
        if isinstance(primary, OperationCancelled):
            unexpected = UnexpectedCancellationError(
                f"unexpected OperationCancelled without a recorded termination request: {primary}"
            )
            if cleanup_errors:
                raise RuntimeCleanupError(
                    f"{unexpected}; cleanup also failed: {cleanup_errors[0]}"
                ) from primary
            raise unexpected from primary
        if cleanup_errors:
            raise RuntimeCleanupError(
                f"runtime failed with {type(primary).__name__}: {primary}; "
                f"cleanup also failed: {cleanup_errors[0]}"
            ) from primary
        raise primary.with_traceback(primary_traceback)
    if cleanup_errors:
        raise RuntimeCleanupError(
            f"resource cleanup failed: {cleanup_errors[0]}"
        ) from cleanup_errors[0]
    return processed


def _capture_worker(
    *,
    configured_source: int | str,
    source_factory: Callable[..., Any],
    channel: LatestValueChannel[FramePacket],
    state: _CaptureWorkerState,
    failures: WorkerFailureBox,
    cancellation: threading.Event,
    detector_ready: threading.Event,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    metrics: RuntimeMetricsCollector,
) -> None:
    source: Any | None = None
    previous_sequence: int | None = None
    published_frames = 0
    schedule: MonotonicPlaybackSchedule | None = None
    try:
        source = source_factory(configured_source)
        state.source = source
        state.cv2 = getattr(source, "cv2", None)
        source_type = _source_type(source, configured_source)
        declared_fps = _source_fps(source)
        pacing_applied = source_type == "video" and declared_fps is not None
        state.metadata = SourceMetadata(
            source_type,
            _total_frames_available(source),
            declared_fps,
            pacing_applied,
        )
        state.outcome = "running"
        while not cancellation.is_set():
            if schedule is not None:
                if not _wait_for_playback_deadline(
                    schedule,
                    cancellation,
                    monotonic,
                    sleep,
                ):
                    state.outcome = "cancelled"
                    break
            frame = source.read()
            if frame is None:
                if state.metadata.source_type == "camera" and not cancellation.is_set():
                    raise RuntimeError("live camera source stopped without a frame")
                state.outcome = "cancelled" if cancellation.is_set() else "normal_eof"
                break
            if cancellation.is_set():
                break
            _validate_runtime_frame(frame)
            if previous_sequence is not None and frame.sequence <= previous_sequence:
                raise RuntimeError("camera frame sequence must increase monotonically")
            previous_sequence = frame.sequence
            captured_s = _runtime_timestamp(monotonic)
            metrics.record_event("capture", captured_s)
            packet = FramePacket(
                sequence=frame.sequence,
                frame=frame,
                captured_s=captured_s,
                source=state.metadata,
            )
            try:
                channel.publish(packet)
            except RuntimeError:
                if cancellation.is_set():
                    break
                raise
            published_frames += 1
            if published_frames == 1 and state.metadata.source_type != "camera":
                while not cancellation.is_set() and not detector_ready.wait(0.05):
                    pass
                if cancellation.is_set():
                    state.outcome = "cancelled"
                    break
                if state.metadata.pacing_applied:
                    if state.metadata.declared_fps is None:
                        raise RuntimeError("paced video source is missing declared FPS")
                    playback_started_s = _runtime_timestamp(monotonic)
                    state.playback_started_s = playback_started_s
                    state.detector_readiness_s = playback_started_s - packet.captured_s
                    if state.detector_readiness_s < 0.0:
                        raise RuntimeError("monotonic clock moved backwards")
                    schedule = MonotonicPlaybackSchedule(
                        state.metadata.declared_fps,
                        started_s=playback_started_s,
                    )
        if state.outcome == "running":
            state.outcome = "cancelled" if cancellation.is_set() else "stopped"
    except BaseException as error:
        state.outcome = "capture_fault"
        failures.record("capture", error)
        cancellation.set()
    finally:
        if source is not None:
            try:
                source.close()
            except BaseException as error:
                failures.record("capture cleanup", error)
                cancellation.set()
        channel.close()


def _inference_worker(
    *,
    options: VisionOptions,
    detector_factory: Callable[..., Any],
    frames: LatestValueChannel[FramePacket],
    results: LatestValueChannel[DetectionPacket],
    failures: WorkerFailureBox,
    cancellation: threading.Event,
    detector_ready: threading.Event,
    monotonic: Callable[[], float],
    metrics: RuntimeMetricsCollector,
) -> None:
    version = 0
    try:
        detector = detector_factory(
            options.model,
            device=options.device,
            confidence_threshold=options.confidence_threshold,
            iou_threshold=options.iou_threshold,
            image_size=options.image_size,
            class_names=options.target_classes,
        )
        first_inference = True
        while not cancellation.is_set():
            value = frames.latest(after_version=version, timeout_s=0.05)
            if value is None:
                if frames.closed:
                    break
                continue
            version = value.version
            frame_packet = value.value
            if first_inference:
                _synchronize_detector(detector)
            started_s = _runtime_timestamp(monotonic)
            detections = tuple(detector.detect(frame_packet.frame))
            if first_inference:
                _synchronize_detector(detector)
            completed_s = _runtime_timestamp(monotonic)
            if completed_s < started_s or started_s < frame_packet.captured_s:
                raise RuntimeError("monotonic clock moved backwards")
            metrics.record_event("inference", completed_s)
            metrics.record_latency(
                "capture_to_inference_start",
                started_s - frame_packet.captured_s,
            )
            metrics.record_latency(
                "capture_to_inference_complete",
                completed_s - frame_packet.captured_s,
            )
            packet = DetectionPacket(
                frame_packet=frame_packet,
                inference_started_s=started_s,
                inference_completed_s=completed_s,
                detections=detections,
                healthy=True,
                inference_ms=float(detector.last_inference_ms),
                resolved_device=str(detector.active_device),
            )
            try:
                results.publish(packet)
            except RuntimeError:
                if cancellation.is_set():
                    break
                raise
            if first_inference:
                first_inference = False
                detector_ready.set()
    except BaseException as error:
        failures.record("inference", error)
        cancellation.set()
    finally:
        results.close()


def _synchronize_detector(detector: Any) -> None:
    synchronize = getattr(detector, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _wait_for_playback_deadline(
    schedule: MonotonicPlaybackSchedule,
    cancellation: threading.Event,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> bool:
    """Wait interruptibly for one frame slot and skip every missed slot."""
    while not cancellation.is_set():
        now = _runtime_timestamp(monotonic)
        remaining = schedule.wait_seconds(now)
        if remaining <= 0.0:
            schedule.frame_started(now)
            return True
        sleep(min(remaining, 0.05))
    return False


def _recorder_worker(
    *,
    channel: LatestValueChannel[_RenderedPacket],
    state: _RecorderWorkerState,
    failures: WorkerFailureBox,
    cancellation: threading.Event,
) -> None:
    version = 0
    try:
        while True:
            value = channel.latest(after_version=version, timeout_s=0.05)
            if value is None:
                if channel.closed or cancellation.is_set():
                    break
                continue
            version = value.version
            packet = value.value
            if state.started_s is None:
                state.started_s = packet.timestamp_s
            state.finished_s = packet.timestamp_s
            state.recorder.write(packet.image, timestamp_s=packet.timestamp_s)
            state.mark_consumed(version)
    except BaseException as error:
        failures.record("recorder", error)
        cancellation.set()
    finally:
        try:
            state.recorder.release()
        except BaseException as error:
            failures.record("recorder cleanup", error)
            cancellation.set()
        state.mark_completed()


def _wait_for_channel_value(
    channel: LatestValueChannel[Any],
    failures: WorkerFailureBox,
    cancellation: threading.Event,
    stop_requested: Callable[[], bool],
    *,
    mark_consumed: bool = True,
) -> Any:
    while True:
        if stop_requested():
            cancellation.set()
            raise OperationCancelled("runtime termination requested")
        _raise_concurrent_failure(failures, stop_requested)
        value = channel.latest(timeout_s=0.05, mark_consumed=mark_consumed)
        if value is not None:
            return value
        if channel.closed:
            _raise_concurrent_failure(failures, stop_requested)
            return None


def _raise_concurrent_failure(
    failures: WorkerFailureBox,
    stop_requested: Callable[[], bool],
) -> None:
    failure = failures.get()
    if failure is None:
        return
    if isinstance(failure.error, OperationCancelled) and stop_requested():
        raise failure.error
    if isinstance(failure.error, OperationCancelled):
        raise UnexpectedCancellationError(
            f"{failure.stage} worker cancelled without a recorded termination "
            f"request: {failure.error}"
        ) from failure.error
    raise RuntimeError(f"{failure.stage} worker failed: {failure.error}") from failure.error


def _raise_if_concurrent_cancelled(
    cancellation: threading.Event,
    stop_requested: Callable[[], bool],
) -> None:
    if stop_requested():
        cancellation.set()
        raise OperationCancelled("runtime termination requested")
    if cancellation.is_set():
        raise OperationCancelled("concurrent runtime cancelled")


def _apply_concurrent_command(
    actuator: Any,
    command: PTZCommand,
    result: DetectionPacket,
    cancellation: threading.Event,
    stop_requested: Callable[[], bool],
    failures: WorkerFailureBox,
    monotonic: Callable[[], float],
    metrics: RuntimeMetricsCollector,
    *,
    record_session_fault: bool,
) -> None:
    _raise_concurrent_failure(failures, stop_requested)
    _raise_if_concurrent_cancelled(cancellation, stop_requested)
    try:
        actuator.apply(command)
    except BaseException as error:
        if record_session_fault:
            metrics.increment_session_fault(error)
        cancellation.set()
        raise
    commanded_s = _runtime_timestamp(monotonic)
    if commanded_s < result.frame_packet.captured_s:
        raise RuntimeError("monotonic clock moved backwards")
    metrics.record_event("command", commanded_s)
    metrics.record_latency(
        "result_age_at_control",
        commanded_s - result.inference_completed_s,
    )
    metrics.record_latency(
        "capture_to_command",
        commanded_s - result.frame_packet.captured_s,
    )


def _join_concurrent_workers(
    threads: Sequence[threading.Thread],
    capture_state: _CaptureWorkerState,
    timeout_s: float,
    errors: list[Exception],
) -> None:
    started_s = time.monotonic()
    initial_deadline = started_s + timeout_s / 2.0
    for thread in threads:
        remaining = max(0.0, initial_deadline - time.monotonic())
        thread.join(remaining)
    alive = [thread for thread in threads if thread.is_alive()]
    if alive and capture_state.source is not None:
        # OpenCV release is the bounded shutdown escape hatch for a capture
        # read that did not observe cancellation. The capture worker remains
        # the only context that calls read().
        try:
            capture_state.source.close()
        except Exception as error:
            errors.append(error)
        final_deadline = started_s + timeout_s
        for thread in alive:
            remaining = max(0.0, final_deadline - time.monotonic())
            thread.join(remaining)
        alive = [thread for thread in alive if thread.is_alive()]
    if alive:
        errors.append(
            RuntimeError(
                "worker join deadline expired: " + ", ".join(thread.name for thread in alive)
            )
        )


def _raise_primary_with_secondary_failures(
    primary: BaseException,
    primary_traceback: TracebackType | None,
    observer_error: BaseException | None,
    cleanup_errors: tuple[Exception, ...],
) -> NoReturn:
    """Raise the control-path failure with every later failure inspectable.

    The direct cause deliberately holds structured diagnostics rather than an
    observer error alone, so an observer and one or more cleanup failures are
    all retained without replacing the original runtime failure.
    """
    diagnostic = RuntimeSecondaryFailures(
        observer_error=observer_error,
        cleanup_errors=cleanup_errors,
    )
    raise primary.with_traceback(primary_traceback) from diagnostic


def _exception_chain_contains(error: BaseException, target: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if current is target:
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


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


def _runtime_timestamp(monotonic: Callable[[], float]) -> float:
    value = monotonic()
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise RuntimeError("monotonic clock returned a non-finite value")
    return float(value)


def _source_type(source: Any, configured_source: int | str) -> str:
    if bool(getattr(source, "is_live", isinstance(configured_source, int))):
        return "camera"
    return "image" if bool(getattr(source, "is_image", False)) else "video"


def _total_frames_available(source: Any) -> int | None:
    total = getattr(source, "total_frames", None)
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return total
    return None


def _source_fps(source: Any) -> float | None:
    value = getattr(source, "fps", None)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RuntimeError("source FPS must be a finite positive number")
    return float(value)


def _bind_cancellation_predicate(
    actuator: Any,
    stop_requested: Callable[[], bool],
) -> None:
    bind = getattr(actuator, "set_cancellation_predicate", None)
    if callable(bind):
        bind(stop_requested)


def _validate_options(config: AppConfig, options: VisionOptions) -> None:
    if options.runtime_mode not in {"single", "concurrent"}:
        raise ValueError("runtime mode must be single or concurrent")
    freshness = (
        config.runtime.result_freshness_s
        if options.result_freshness_s is None
        else options.result_freshness_s
    )
    if (
        isinstance(freshness, bool)
        or not isinstance(freshness, Real)
        or not math.isfinite(float(freshness))
        or float(freshness) <= 0.0
    ):
        raise ValueError("result freshness must be a finite positive number")
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
        not isinstance(name, str) or not name.strip() for name in options.target_classes
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
    if not isinstance(options.digital_zoom, bool):
        raise ValueError("digital zoom must be bool")
    if (
        isinstance(options.max_digital_zoom, bool)
        or not isinstance(options.max_digital_zoom, Real)
        or not math.isfinite(float(options.max_digital_zoom))
        or float(options.max_digital_zoom) < 1.0
    ):
        raise ValueError("maximum digital zoom must be finite and at least 1.0")
    _output_fps(config, options)
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
    if float(freshness) >= config.serial.watchdog_timeout_s:
        raise ValueError(
            "result freshness must be less than the configured firmware watchdog timeout"
        )
    _resolved_serial_port(config, options)


def _probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return number


def _output_fps(config: AppConfig, options: VisionOptions) -> float:
    value = config.camera.fps if options.output_fps is None else options.output_fps
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("encoded output FPS must be a finite positive number")
    return float(value)


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
        raise TypeError(f"Arduino actuator must provide callable {method_name}()")
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
    pan_limits: AngleLimits,
    tilt_limits: AngleLimits,
    digital_zoom: float = 1.0,
    concurrent_rates: _ConcurrentRates | None = None,
) -> str:
    target_text = "none" if target is None else target.detection.label
    error_text = "none" if error is None else f"{error[0]:.1f},{error[1]:.1f}"
    physical_pan, physical_tilt = _physical_angles(actuator)
    physical_pan_text = "n/a" if physical_pan is None else f"{physical_pan:.0f}"
    physical_tilt_text = "n/a" if physical_tilt is None else f"{physical_tilt:.0f}"
    rate_text = (
        f"fps={fps:.1f}"
        if concurrent_rates is None
        else (
            f"unique_inference_fps={concurrent_rates.unique_inference_fps:.1f} "
            f"display_fps={concurrent_rates.display_fps:.1f} "
            f"control_hz={concurrent_rates.control_hz:.1f} "
            "encoded_output_fps="
            + (
                "n/a"
                if concurrent_rates.encoded_output_fps is None
                else f"{concurrent_rates.encoded_output_fps:.1f}"
            )
        )
    )
    return (
        f"seq={frame.sequence:06d} detections={len(detections)} target={target_text} "
        f"error_px={error_text} logical_pan={actuator.pan_deg:.2f} "
        f"logical_tilt={actuator.tilt_deg:.2f} physical_pan={physical_pan_text} "
        f"physical_tilt={physical_tilt_text} "
        f"saturation_pan={_axis_saturation(actuator.pan_deg, pan_limits)} "
        f"saturation_tilt={_axis_saturation(actuator.tilt_deg, tilt_limits)} "
        f"inference_ms={inference_ms:.1f} {rate_text} device={device} "
        f"actuator={actuator_backend} digital_zoom={digital_zoom:.2f}x"
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
    pan_limits: AngleLimits,
    tilt_limits: AngleLimits,
    zoom_transform: ZoomTransform | None = None,
    concurrent_rates: _ConcurrentRates | None = None,
) -> Any:
    if frame.image is None:
        raise ValueError("cannot annotate a frame without an image")
    transform = zoom_transform or identity_transform(frame.width, frame.height)
    if transform.is_identity:
        rendered = frame.image.copy()
    else:
        cropped = frame.image[
            transform.crop_top : transform.crop_bottom,
            transform.crop_left : transform.crop_right,
        ]
        rendered = cv2.resize(
            cropped,
            (frame.width, frame.height),
            interpolation=cv2.INTER_LINEAR,
        )
    selected = target.detection if target is not None else None
    for detection in detections:
        box = transform.transform_box(detection)
        if box is None:
            continue
        color = (0, 255, 255) if detection == selected else (0, 200, 0)
        start = (round(box[0]), round(box[1]))
        end = (round(box[2]), round(box[3]))
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

    full_center_values = transform.transform_point(frame.width / 2, frame.height / 2)
    full_center = tuple(round(value) for value in full_center_values)
    full_center_visible = 0 <= full_center[0] < frame.width and 0 <= full_center[1] < frame.height
    if full_center_visible:
        cv2.drawMarker(rendered, full_center, (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
    crop_center = (round(frame.width / 2), round(frame.height / 2))
    cv2.drawMarker(rendered, crop_center, (255, 0, 255), cv2.MARKER_CROSS, 12, 1)
    if target is not None:
        target_x, target_y = transform.transform_point(*target.detection.center)
        target_center = (
            round(min(frame.width - 1, max(0.0, target_x))),
            round(min(frame.height - 1, max(0.0, target_y))),
        )
        cv2.drawMarker(rendered, target_center, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        if full_center_visible:
            cv2.line(rendered, full_center, target_center, (0, 255, 255), 1)

    error_text = "none" if error is None else f"({error[0]:.1f}, {error[1]:.1f}) px"
    physical_pan, physical_tilt = _physical_angles(actuator)
    physical_text = (
        "physical pan=n/a tilt=n/a"
        if physical_pan is None or physical_tilt is None
        else f"physical pan={physical_pan:.0f} tilt={physical_tilt:.0f}"
    )
    rate_text = (
        f"fps={fps:.1f}"
        if concurrent_rates is None
        else (
            f"unique_inference_fps={concurrent_rates.unique_inference_fps:.1f}  "
            f"display_fps={concurrent_rates.display_fps:.1f}  "
            f"control_hz={concurrent_rates.control_hz:.1f}  "
            "encoded_output_fps="
            + (
                "n/a"
                if concurrent_rates.encoded_output_fps is None
                else f"{concurrent_rates.encoded_output_fps:.1f}"
            )
        )
    )
    lines = (
        f"tracking={'locked' if target is not None else 'lost'}",
        (
            "target_confidence=none"
            if target is None
            else f"target_confidence={target.detection.confidence:.2f}"
        ),
        f"logical pan={actuator.pan_deg:.2f} tilt={actuator.tilt_deg:.2f}",
        physical_text,
        f"saturation pan={_axis_saturation(actuator.pan_deg, pan_limits)} "
        f"tilt={_axis_saturation(actuator.tilt_deg, tilt_limits)}",
        f"error={error_text}",
        f"inference={inference_ms:.1f} ms  {rate_text}  device={device}",
        f"digital_zoom={transform.zoom_factor:.2f}x (software)",
        "centers: full=white crop=magenta target=yellow",
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


def _physical_angles(actuator: Any) -> tuple[float | None, float | None]:
    values: list[float | None] = []
    for name in ("physical_pan_deg", "physical_tilt_deg"):
        value = getattr(actuator, name, None)
        if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
            values.append(float(value))
        else:
            values.append(None)
    return values[0], values[1]


def _axis_saturation(value: float, limits: AngleLimits) -> str:
    if value <= limits.minimum_deg:
        return "min"
    if value >= limits.maximum_deg:
        return "max"
    return "none"


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
        raise RuntimeError(f"unable to inspect annotated video output {output}: {exc}") from exc
    if not opened:
        writer.release()
        raise RuntimeError(f"unable to open annotated video output {output}")
    try:
        return WallClockVideoRecorder(writer, fps)
    except Exception:
        writer.release()
        raise


def _target_classes(values: Sequence[str] | None, defaults: Sequence[str]) -> tuple[str, ...]:
    raw = values if values else defaults
    classes = tuple(name.strip() for value in raw for name in value.split(",") if name.strip())
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
        "--output-fps",
        type=float,
        help="fixed encoded video FPS; defaults to configured camera FPS",
    )
    parser.add_argument(
        "--digital-zoom",
        action="store_true",
        help="enable target-following software crop in display/output only",
    )
    parser.add_argument(
        "--max-digital-zoom",
        type=float,
        default=2.0,
        help="maximum software crop magnification (default: 2.0)",
    )
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
            "explicitly allow ENABLE after camera validation, first inference, "
            "display/output setup, and serial handshake"
        ),
    )
    parser.add_argument(
        "--runtime-mode",
        choices=("single", "concurrent"),
        help="execution topology; defaults to runtime.execution_mode from configuration",
    )
    parser.add_argument(
        "--result-freshness",
        type=float,
        help=(
            "maximum concurrent detection-result age in seconds; defaults to "
            "runtime.result_freshness_s"
        ),
    )
    return parser


def _source_option(cli_value: str | None, config: AppConfig) -> int | str:
    if cli_value is not None:
        return parse_source_argument(cli_value)
    value = config.camera.device
    if value is None:
        raise ValueError("--source is required when camera.device is absent from configuration")
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
            iou_threshold=(args.iou if args.iou is not None else config.detection.iou_threshold),
            image_size=(
                args.image_size if args.image_size is not None else config.detection.image_size
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
            digital_zoom=args.digital_zoom,
            max_digital_zoom=args.max_digital_zoom,
            output_fps=args.output_fps,
            runtime_mode=args.runtime_mode or config.runtime.execution_mode,
            result_freshness_s=args.result_freshness,
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
