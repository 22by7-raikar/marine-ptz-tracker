"""Hardware-free offline replay metrics built on the production runtime."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmark import (
    allowlisted_environment_metadata,
    local_path_marker,
    percentile,
    sanitize_command,
    summarize_samples,
    write_report,
)
from .config import AppConfig
from .evaluation import ClosedLoopMetricsCollector, SettlingEvent
from .vision_cli import (
    RuntimeCompleteEvent,
    RuntimeFrameEvent,
    RuntimeObserver,
    RuntimeProcessingEndEvent,
    RuntimeStartEvent,
    VisionOptions,
    run_vision,
)

REPLAY_SCHEMA_VERSION = 3
SUPPORTED_REPLAY_SCHEMA_VERSIONS = frozenset({2, REPLAY_SCHEMA_VERSION})
_PATH_ARGUMENTS = frozenset(
    {
        "--source",
        "--model",
        "--config",
        "--report",
        "--annotated-output",
        "--tracker-config",
    }
)


class ReplayError(ValueError):
    """Raised when replay settings or collected replay evidence are invalid."""


class ReplayCancelled(RuntimeError):
    """Raised after bounded runtime cleanup when a replay is cancelled."""


@dataclass(frozen=True, slots=True)
class AcceptanceThresholds:
    """Optional replay acceptance limits; absent limits mean report-only mode."""

    minimum_selected_target_ratio: float | None = None
    maximum_p95_inference_ms: float | None = None
    maximum_mean_normalized_error: float | None = None
    maximum_p95_normalized_error: float | None = None
    maximum_pan_saturation_count: int | None = None
    maximum_tilt_saturation_count: int | None = None

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            (
                "minimum_selected_target_ratio",
                self.minimum_selected_target_ratio,
                0.0,
                1.0,
            ),
            ("maximum_p95_inference_ms", self.maximum_p95_inference_ms, 0.0, None),
            (
                "maximum_mean_normalized_error",
                self.maximum_mean_normalized_error,
                0.0,
                None,
            ),
            (
                "maximum_p95_normalized_error",
                self.maximum_p95_normalized_error,
                0.0,
                None,
            ),
        ):
            if value is None:
                continue
            _finite_number(value, name)
            if value < minimum or (maximum is not None and value > maximum):
                bound = (
                    f"between {minimum:g} and {maximum:g}"
                    if maximum is not None
                    else f">= {minimum:g}"
                )
                raise ReplayError(f"{name} must be finite and {bound}")
        for name, value in (
            ("maximum_pan_saturation_count", self.maximum_pan_saturation_count),
            ("maximum_tilt_saturation_count", self.maximum_tilt_saturation_count),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReplayError(f"{name} must be a non-negative integer")

    @property
    def supplied(self) -> bool:
        return any(
            value is not None
            for value in (
                self.minimum_selected_target_ratio,
                self.maximum_p95_inference_ms,
                self.maximum_mean_normalized_error,
                self.maximum_p95_normalized_error,
                self.maximum_pan_saturation_count,
                self.maximum_tilt_saturation_count,
            )
        )


@dataclass(frozen=True, slots=True)
class ReplayOptions:
    """Validated, non-hardware settings for one finite-media replay."""

    source: Path
    model: str
    device: str
    target_classes: tuple[str, ...]
    confidence_threshold: float
    iou_threshold: float
    image_size: int
    max_frames: int | None
    report_path: Path
    annotated_output: Path | None = None
    thresholds: AcceptanceThresholds = AcceptanceThresholds()
    tracker_backend: str = "none"
    tracker_config: Path | None = None
    tracker_input_confidence: float = 0.20
    tracker_min_confirmation_hits: int = 2
    tracker_max_unsupported_age_s: float = 0.30
    settling_events: tuple[SettlingEvent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, Path) or not str(self.source):
            raise ReplayError("source must be a local filesystem path")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ReplayError("model must be a non-empty name or local path")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ReplayError("device must be a non-empty string")
        if not self.target_classes or any(
            not isinstance(name, str) or not name.strip() for name in self.target_classes
        ):
            raise ReplayError("at least one non-empty target class is required")
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("iou_threshold", self.iou_threshold),
        ):
            number = _finite_number(value, name)
            if not 0.0 <= number <= 1.0:
                raise ReplayError(f"{name} must be finite and between 0 and 1")
        if (
            isinstance(self.image_size, bool)
            or not isinstance(self.image_size, int)
            or self.image_size <= 0
        ):
            raise ReplayError("image_size must be a positive integer")
        if self.max_frames is not None and (
            isinstance(self.max_frames, bool)
            or not isinstance(self.max_frames, int)
            or self.max_frames <= 0
        ):
            raise ReplayError("max_frames must be greater than zero when provided")
        if not isinstance(self.report_path, Path) or not str(self.report_path):
            raise ReplayError("report_path must be a filesystem path")
        paths = [self.source, self.report_path]
        if self.annotated_output is not None:
            paths.append(self.annotated_output)
        resolved = [path.resolve(strict=False) for path in paths]
        if len(set(resolved)) != len(resolved):
            raise ReplayError("source, report, and annotated output paths must differ")
        if self.tracker_backend not in {"none", "botsort"}:
            raise ReplayError("tracker_backend must be 'none' or 'botsort'")
        tracker_input = _finite_number(
            self.tracker_input_confidence,
            "tracker_input_confidence",
        )
        if not 0.0 <= tracker_input <= 1.0 or (
            self.tracker_backend == "botsort" and tracker_input > self.confidence_threshold
        ):
            raise ReplayError(
                "tracker_input_confidence must be a probability and, for botsort, "
                "must not exceed confidence_threshold"
            )
        if (
            isinstance(self.tracker_min_confirmation_hits, bool)
            or not isinstance(self.tracker_min_confirmation_hits, int)
            or self.tracker_min_confirmation_hits <= 0
        ):
            raise ReplayError("tracker_min_confirmation_hits must be a positive integer")
        unsupported_age = _finite_number(
            self.tracker_max_unsupported_age_s,
            "tracker_max_unsupported_age_s",
        )
        if unsupported_age <= 0.0:
            raise ReplayError("tracker_max_unsupported_age_s must be greater than zero")
        if self.tracker_config is not None and not self.tracker_config.is_file():
            raise ReplayError("tracker_config must name an existing file when provided")
        if self.tracker_backend == "botsort" and self.tracker_config is None:
            raise ReplayError(
                "tracker_config must name an existing file when tracker_backend is 'botsort'"
            )
        if not isinstance(self.settling_events, tuple) or any(
            not isinstance(item, SettlingEvent) for item in self.settling_events
        ):
            raise ReplayError("settling_events must contain SettlingEvent values")


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """One explicit replay threshold evaluation."""

    name: str
    actual: float | int | None
    limit: float | int
    comparison: str
    passed: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "actual": self.actual,
            "limit": self.limit,
            "comparison": self.comparison,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """Versioned acceptance evidence generated after a completed replay."""

    timestamp_utc: str
    command: tuple[str, ...]
    options: ReplayOptions
    source_type: str
    total_frames_available: int | None
    frame_dimensions: tuple[int, int] | None
    resolved_device: str | None
    environment: Mapping[str, object]
    processed_frames: int
    frames_with_detection: int
    frames_with_target: int
    first_target_frame: int | None
    acquisition_processing_s: float | None
    lost_target_episodes: tuple[int, ...]
    inference_ms: tuple[float, ...]
    horizontal_errors: tuple[float, ...]
    vertical_errors: tuple[float, ...]
    absolute_errors: tuple[float, ...]
    command_count: int
    processing_elapsed_s: float | None
    pan_saturation_count: int
    tilt_saturation_count: int
    final_pan_deg: float | None
    final_tilt_deg: float | None
    exit_reason: str
    threshold_results: tuple[ThresholdResult, ...]
    closed_loop_metrics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        inference = _timing_summary(self.inference_ms)
        absolute_error = _error_summary(self.absolute_errors)
        horizontal = _axis_summary(self.horizontal_errors)
        vertical = _axis_summary(self.vertical_errors)
        accepted = (
            None
            if not self.options.thresholds.supplied
            else all(result.passed for result in self.threshold_results)
        )
        sanitized = sanitize_replay_command(self.command)
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "completion": {"status": "completed", "exit_reason": self.exit_reason},
            "timestamp_utc": self.timestamp_utc,
            "invocation": {
                "program": sanitized[0] if sanitized else None,
                "arguments": list(sanitized[1:]),
            },
            "input": {
                "source_path": _safe_path(self.options.source),
                "source_type": self.source_type,
                "total_frames_available": self.total_frames_available,
                "frame_dimensions": None
                if self.frame_dimensions is None
                else {"width": self.frame_dimensions[0], "height": self.frame_dimensions[1]},
            },
            "configuration": {
                "model": _sanitize_replay_model(self.options.model),
                "requested_device": self.options.device,
                "resolved_device": self.resolved_device,
                "target_classes": list(self.options.target_classes),
                "confidence_threshold": self.options.confidence_threshold,
                "iou_threshold": self.options.iou_threshold,
                "image_size": self.options.image_size,
                "max_frames": self.options.max_frames,
                "tracker_backend": self.options.tracker_backend,
                "tracker_config": (
                    None
                    if self.options.tracker_config is None
                    else _safe_path(self.options.tracker_config)
                ),
                "tracker_input_confidence": self.options.tracker_input_confidence,
                "tracker_min_confirmation_hits": self.options.tracker_min_confirmation_hits,
                "tracker_max_unsupported_age_s": self.options.tracker_max_unsupported_age_s,
                "settling_events": [
                    {
                        "name": item.name,
                        "start_frame": item.start_frame,
                        "end_frame": item.end_frame,
                        "tolerance": item.tolerance,
                        "dwell_s": item.dwell_s,
                    }
                    for item in self.options.settling_events
                ],
            },
            "environment": _sanitize_environment(self.environment),
            "metrics": {
                "processed_frames": self.processed_frames,
                "frames_with_valid_detection": self.frames_with_detection,
                "frames_with_selected_target": self.frames_with_target,
                "selected_target_ratio": _ratio(self.frames_with_target, self.processed_frames),
                "first_selected_target_frame": self.first_target_frame,
                "target_acquisition_processing_s": self.acquisition_processing_s,
                "lost_target_episode_count": len(self.lost_target_episodes),
                "lost_target_episode_lengths_frames": list(self.lost_target_episodes),
                "inference_ms": inference,
                "approximate_processing_fps": _rate(
                    self.processed_frames, self.processing_elapsed_s
                ),
                "normalized_center_error": {
                    "horizontal": horizontal,
                    "vertical": vertical,
                    "absolute_euclidean": absolute_error,
                },
                "simulated_actuator_command_count": self.command_count,
                "effective_command_rate_hz": _rate(self.command_count, self.processing_elapsed_s),
                "resulting_pan_saturation_count": self.pan_saturation_count,
                "resulting_tilt_saturation_count": self.tilt_saturation_count,
                "final_pan_deg": self.final_pan_deg,
                "final_tilt_deg": self.final_tilt_deg,
            },
            "closed_loop": dict(self.closed_loop_metrics),
            "thresholds": {
                "results": [result.to_dict() for result in self.threshold_results],
                "acceptance": accepted,
            },
        }


@dataclass(frozen=True, slots=True)
class ReplayRunResult:
    """Completed report and its optional acceptance outcome."""

    report: ReplayReport

    @property
    def accepted(self) -> bool | None:
        results = self.report.threshold_results
        if not self.report.options.thresholds.supplied:
            return None
        return all(item.passed for item in results)


class ReplayMetrics(RuntimeObserver):
    """Replay-only observer over the production loop's completed commands."""

    def __init__(
        self,
    ) -> None:
        self._started_s: float | None = None
        self._finished_s: float | None = None
        self._pan_limits: tuple[float, float] | None = None
        self._tilt_limits: tuple[float, float] | None = None
        self.source_type = "video"
        self.total_frames_available: int | None = None
        self.frame_dimensions: tuple[int, int] | None = None
        self.resolved_device: str | None = None
        self.processed_frames = 0
        self.frames_with_detection = 0
        self.frames_with_target = 0
        self.first_target_frame: int | None = None
        self.acquisition_processing_s: float | None = None
        self._seen_target = False
        self._lost_length = 0
        self.lost_target_episodes: list[int] = []
        self.inference_ms: list[float] = []
        self.horizontal_errors: list[float] = []
        self.vertical_errors: list[float] = []
        self.absolute_errors: list[float] = []
        self.command_count = 0
        self.pan_saturation_count = 0
        self.tilt_saturation_count = 0
        self.final_pan_deg: float | None = None
        self.final_tilt_deg: float | None = None
        self.exit_reason: str | None = None
        self.completed = False

    def on_start(self, event: RuntimeStartEvent) -> None:
        if event.actuator_backend != "simulated":
            raise ReplayError("offline replay requires SimulatedPTZActuator")
        self._started_s = event.processing_started_s
        self.source_type = event.source_type
        self.total_frames_available = event.total_frames_available
        self.resolved_device = event.resolved_device
        self._pan_limits = (event.pan_limits.minimum_deg, event.pan_limits.maximum_deg)
        self._tilt_limits = (event.tilt_limits.minimum_deg, event.tilt_limits.maximum_deg)
        self.final_pan_deg = event.initial_pan_deg
        self.final_tilt_deg = event.initial_tilt_deg

    def on_frame(self, event: RuntimeFrameEvent) -> None:
        self.processed_frames += 1
        self.frame_dimensions = (event.frame_width, event.frame_height)
        self.resolved_device = event.resolved_device
        if event.detection_count:
            self.frames_with_detection += 1
        inference = _finite_number(event.inference_ms, "detector.last_inference_ms")
        if inference < 0.0:
            raise ReplayError("detector.last_inference_ms must be non-negative")
        self.inference_ms.append(inference)
        self.command_count += 1
        self.final_pan_deg = event.actuator_pan_deg
        self.final_tilt_deg = event.actuator_tilt_deg

        if self._pan_limits is not None and event.actuator_pan_deg in self._pan_limits:
            self.pan_saturation_count += 1
        if self._tilt_limits is not None and event.actuator_tilt_deg in self._tilt_limits:
            self.tilt_saturation_count += 1

        if event.selected_detection is None:
            if self._seen_target:
                self._lost_length += 1
            return

        if self._lost_length:
            self.lost_target_episodes.append(self._lost_length)
            self._lost_length = 0
        self.frames_with_target += 1
        if not self._seen_target:
            self._seen_target = True
            self.first_target_frame = event.frame_sequence
            if self._started_s is not None:
                self.acquisition_processing_s = max(
                    0.0, event.processing_timestamp_s - self._started_s
                )
        detection = event.selected_detection
        x_error = (detection.center[0] - event.frame_width / 2) / (event.frame_width / 2)
        y_error = (detection.center[1] - event.frame_height / 2) / (event.frame_height / 2)
        self.horizontal_errors.append(x_error)
        self.vertical_errors.append(y_error)
        self.absolute_errors.append(math.hypot(x_error, y_error))

    def on_processing_end(self, event: RuntimeProcessingEndEvent) -> None:
        if event.processed_frames > self.processed_frames:
            raise ReplayError("runtime and replay processed-frame counts disagree")
        if self._lost_length:
            self.lost_target_episodes.append(self._lost_length)
            self._lost_length = 0
        self.exit_reason = event.exit_reason
        self._finished_s = event.processing_ended_s

    def on_complete(self, event: RuntimeCompleteEvent) -> None:
        if event.processed_frames != self.processed_frames:
            raise ReplayError("runtime and replay processed-frame counts disagree")
        if self.exit_reason != event.exit_reason:
            raise ReplayError("runtime completion reason disagrees with replay processing end")
        self.completed = True

    @property
    def elapsed_s(self) -> float | None:
        if self._started_s is None or self._finished_s is None:
            return None
        return max(0.0, self._finished_s - self._started_s)


def run_replay(
    config: AppConfig,
    options: ReplayOptions,
    *,
    source_factory: Callable[..., Any] | None = None,
    detector_factory: Callable[..., Any] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    rate_monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    stop_requested: Callable[[], bool] = lambda: False,
    environment: Callable[[str], Mapping[str, object]] = lambda _device: {},
    timestamp: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    command: Sequence[str] = (),
) -> ReplayRunResult:
    """Run finite media through ``run_vision`` and atomically write its report."""
    metrics = ReplayMetrics()
    closed_loop = ClosedLoopMetricsCollector(options.settling_events)
    vision_options = VisionOptions(
        source=str(options.source),
        model=options.model,
        device=options.device,
        target_classes=options.target_classes,
        confidence_threshold=options.confidence_threshold,
        iou_threshold=options.iou_threshold,
        image_size=options.image_size,
        max_frames=options.max_frames,
        display=False,
        output=options.annotated_output,
        actuator_backend="simulated",
        tracker_backend=options.tracker_backend,
        tracker_config=options.tracker_config,
        tracker_input_confidence=options.tracker_input_confidence,
        tracker_min_confirmation_hits=options.tracker_min_confirmation_hits,
        tracker_max_unsupported_age_s=options.tracker_max_unsupported_age_s,
    )
    run_vision(
        config,
        vision_options,
        **({} if source_factory is None else {"source_factory": source_factory}),
        **({} if detector_factory is None else {"detector_factory": detector_factory}),
        monotonic=monotonic,
        rate_monotonic=rate_monotonic,
        sleep=sleep,
        stop_requested=stop_requested,
        emit=lambda _line: None,
        observer=metrics,
        evaluation_observer=closed_loop,
    )
    if metrics.exit_reason == "cancelled" or stop_requested():
        raise ReplayCancelled("offline replay cancelled before acceptance completion")
    if not metrics.completed or metrics.exit_reason is None:
        raise ReplayError("replay completed without a normal exit reason")
    threshold_results = _evaluate_thresholds(options.thresholds, metrics)
    resolved = metrics.resolved_device
    report = ReplayReport(
        timestamp_utc=timestamp().astimezone(timezone.utc).isoformat(),
        command=tuple(command),
        options=options,
        source_type=metrics.source_type,
        total_frames_available=metrics.total_frames_available,
        frame_dimensions=metrics.frame_dimensions,
        resolved_device=resolved,
        environment={} if resolved is None else environment(resolved),
        processed_frames=metrics.processed_frames,
        frames_with_detection=metrics.frames_with_detection,
        frames_with_target=metrics.frames_with_target,
        first_target_frame=metrics.first_target_frame,
        acquisition_processing_s=metrics.acquisition_processing_s,
        lost_target_episodes=tuple(metrics.lost_target_episodes),
        inference_ms=tuple(metrics.inference_ms),
        horizontal_errors=tuple(metrics.horizontal_errors),
        vertical_errors=tuple(metrics.vertical_errors),
        absolute_errors=tuple(metrics.absolute_errors),
        command_count=metrics.command_count,
        processing_elapsed_s=metrics.elapsed_s,
        pan_saturation_count=metrics.pan_saturation_count,
        tilt_saturation_count=metrics.tilt_saturation_count,
        final_pan_deg=metrics.final_pan_deg,
        final_tilt_deg=metrics.final_tilt_deg,
        exit_reason=metrics.exit_reason,
        threshold_results=threshold_results,
        closed_loop_metrics=closed_loop.to_dict(),
    )
    write_report(report, options.report_path)
    return ReplayRunResult(report)


def normalize_replay_report(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a non-mutating compatibility view of schema-v2 or schema-v3 reports."""
    version = payload.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ReplayError("replay report schema_version must be an integer")
    if version not in SUPPORTED_REPLAY_SCHEMA_VERSIONS:
        raise ReplayError(f"unsupported replay report schema_version: {version}")
    normalized = dict(payload)
    if version == 2:
        normalized["closed_loop"] = {
            "available": False,
            "reason": "schema version 2 did not record closed-loop frame data",
        }
    return normalized


def _evaluate_thresholds(
    thresholds: AcceptanceThresholds,
    metrics: ReplayMetrics,
) -> tuple[ThresholdResult, ...]:
    inference_p95 = _percentile_or_none(metrics.inference_ms, 0.95)
    mean_error = _mean_or_none(metrics.absolute_errors)
    p95_error = _percentile_or_none(metrics.absolute_errors, 0.95)
    values: tuple[tuple[str, float | int | None, float | int | None, str], ...] = (
        (
            "minimum_selected_target_ratio",
            _ratio(metrics.frames_with_target, metrics.processed_frames),
            thresholds.minimum_selected_target_ratio,
            ">=",
        ),
        ("maximum_p95_inference_ms", inference_p95, thresholds.maximum_p95_inference_ms, "<="),
        (
            "maximum_mean_normalized_error",
            mean_error,
            thresholds.maximum_mean_normalized_error,
            "<=",
        ),
        (
            "maximum_p95_normalized_error",
            p95_error,
            thresholds.maximum_p95_normalized_error,
            "<=",
        ),
        (
            "maximum_pan_saturation_count",
            metrics.pan_saturation_count,
            thresholds.maximum_pan_saturation_count,
            "<=",
        ),
        (
            "maximum_tilt_saturation_count",
            metrics.tilt_saturation_count,
            thresholds.maximum_tilt_saturation_count,
            "<=",
        ),
    )
    results: list[ThresholdResult] = []
    for name, actual, limit, comparison in values:
        if limit is None:
            continue
        if actual is None:
            results.append(
                ThresholdResult(name, None, limit, comparison, False, "metric unavailable")
            )
            continue
        passed = actual >= limit if comparison == ">=" else actual <= limit
        results.append(ThresholdResult(name, actual, limit, comparison, passed))
    return tuple(results)


def _timing_summary(samples: Sequence[float]) -> dict[str, float | None]:
    if not samples:
        return {"mean": None, "median": None, "p95": None, "p99": None}
    summary = summarize_samples(samples)
    return {
        "mean": summary.mean_ms,
        "median": summary.median_ms,
        "p95": summary.p95_ms,
        "p99": summary.p99_ms,
    }


def _axis_summary(samples: Sequence[float]) -> dict[str, float | None]:
    absolute = [abs(value) for value in samples]
    return {
        "mean": _mean_or_none(samples),
        "p95_absolute": _percentile_or_none(absolute, 0.95),
    }


def _error_summary(samples: Sequence[float]) -> dict[str, float | None]:
    return {"mean": _mean_or_none(samples), "p95": _percentile_or_none(samples, 0.95)}


def _percentile_or_none(samples: Sequence[float], fraction: float) -> float | None:
    return None if not samples else percentile(samples, fraction)


def _mean_or_none(samples: Sequence[float]) -> float | None:
    return None if not samples else statistics.fmean(samples)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _rate(count: int, elapsed_s: float | None) -> float | None:
    return None if elapsed_s is None or elapsed_s <= 0.0 else count / elapsed_s


def _finite_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ReplayError(f"{name} must be a finite number")
    return float(value)


def _clock_value(clock: Callable[[], float]) -> float:
    value = _finite_number(clock(), "monotonic clock")
    return value


def _safe_path(path: Path) -> str:
    return local_path_marker(str(path))


def sanitize_replay_command(command: Sequence[str]) -> tuple[str, ...]:
    """Redact secrets/URLs and replace local paths in persisted replay argv."""
    generic = sanitize_command(command)
    sanitized: list[str] = []
    path_value_next = False
    for index, raw_value in enumerate(command):
        raw = str(raw_value)
        generic_value = generic[index]
        if path_value_next:
            sanitized.append(local_path_marker(raw))
            path_value_next = False
            continue
        if raw in _PATH_ARGUMENTS:
            sanitized.append(generic_value)
            path_value_next = True
            continue
        if "=" in raw:
            name, value = raw.split("=", 1)
            if name in _PATH_ARGUMENTS:
                sanitized.append(f"{name}={local_path_marker(value)}")
                continue
        sanitized.append(
            _sanitize_replay_path(generic_value) if _looks_local_path(raw) else generic_value
        )
    return tuple(sanitized)


def _sanitize_replay_model(value: str) -> str:
    return sanitize_command(("--model", value))[1]


def _sanitize_replay_path(value: str) -> str:
    if value == "<redacted>" or value.startswith(("<local-path:", "<remote-resource:")):
        return value
    return local_path_marker(value)


def _looks_local_path(value: str) -> bool:
    if value.startswith("path:"):
        return True
    if "://" in value:
        return False
    return value.startswith(("/", "./", "../", "~", "\\")) or "/" in value or "\\" in value


def _sanitize_environment(values: Mapping[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for name, value in allowlisted_environment_metadata(values).items():
        if isinstance(value, str):
            sanitized[name] = sanitize_command((value,))[0]
        else:
            sanitized[name] = value
    return sanitized
