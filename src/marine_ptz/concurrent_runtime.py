"""Hardware-neutral coordination primitives for the concurrent vision runtime."""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from .types import Detection, Frame

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Immutable source facts copied into captured frame packets."""

    source_type: str
    total_frames_available: int | None
    declared_fps: float | None
    pacing_applied: bool


class MonotonicPlaybackSchedule:
    """Deadline-based finite-media pacing that skips missed frame slots."""

    def __init__(self, fps: float, *, started_s: float) -> None:
        rate = _finite_number(fps, "source FPS")
        if rate <= 0.0:
            raise ValueError("source FPS must be greater than zero")
        self._interval_s = 1.0 / rate
        self._next_deadline_s = _finite_number(started_s, "playback start") + self._interval_s

    @property
    def next_deadline_s(self) -> float:
        return self._next_deadline_s

    def wait_seconds(self, now_s: float) -> float:
        """Return the non-negative delay until the next source-frame slot."""
        now = _finite_number(now_s, "playback clock")
        return max(0.0, self._next_deadline_s - now)

    def frame_started(self, now_s: float) -> None:
        """Advance to the first future slot, dropping every missed deadline."""
        now = _finite_number(now_s, "playback clock")
        if now < self._next_deadline_s:
            raise ValueError("playback frame started before its deadline")
        missed = math.floor((now - self._next_deadline_s) / self._interval_s)
        self._next_deadline_s += (missed + 1) * self._interval_s


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One captured frame and the monotonic instant at which it was returned."""

    sequence: int
    frame: Frame
    captured_s: float
    source: SourceMetadata


@dataclass(frozen=True, slots=True)
class DetectionPacket:
    """One completed, healthy inference result for one captured frame."""

    frame_packet: FramePacket
    inference_started_s: float
    inference_completed_s: float
    detections: tuple[Detection, ...]
    healthy: bool
    inference_ms: float
    resolved_device: str

    @property
    def frame_sequence(self) -> int:
        return self.frame_packet.sequence


@dataclass(frozen=True, slots=True)
class ChannelValue(Generic[T]):
    """A versioned snapshot returned from a latest-value channel."""

    version: int
    value: T


class LatestValueChannel(Generic[T]):
    """A condition-backed capacity-one channel that replaces stale values."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: T | None = None
        self._version = 0
        self._consumed_version = 0
        self._closed = False
        self._overwritten = 0

    @property
    def overwritten_count(self) -> int:
        with self._condition:
            return self._overwritten

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def publish(self, value: T) -> int:
        """Publish a value, replacing an unread older value if necessary."""
        with self._condition:
            if self._closed:
                raise RuntimeError("cannot publish to a closed latest-value channel")
            if self._version > self._consumed_version:
                self._overwritten += 1
            self._version += 1
            self._value = value
            self._condition.notify_all()
            return self._version

    def latest(
        self,
        *,
        after_version: int = 0,
        timeout_s: float | None = None,
        mark_consumed: bool = True,
    ) -> ChannelValue[T] | None:
        """Return the newest value after a version, or ``None`` on close/timeout."""
        if after_version < 0:
            raise ValueError("after_version must be non-negative")
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or timeout_s < 0.0
        ):
            raise ValueError("timeout_s must be a finite non-negative number or None")
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        with self._condition:
            while self._version <= after_version and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            if self._version <= after_version or self._value is None:
                return None
            if mark_consumed:
                self._consumed_version = max(self._consumed_version, self._version)
            return ChannelValue(self._version, self._value)

    def peek(self) -> ChannelValue[T] | None:
        """Return and mark the current value consumed without waiting."""
        with self._condition:
            if self._value is None:
                return None
            self._consumed_version = max(self._consumed_version, self._version)
            return ChannelValue(self._version, self._value)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    """The first exception raised by a named worker."""

    stage: str
    error: BaseException


class WorkerFailureBox:
    """Thread-safe first-failure mailbox used to surface worker exceptions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failure: WorkerFailure | None = None

    def record(self, stage: str, error: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = WorkerFailure(stage, error)

    def get(self) -> WorkerFailure | None:
        with self._lock:
            return self._failure


class MonotonicControlSchedule:
    """A nonblocking periodic deadline that skips, rather than replays, delays."""

    def __init__(self, update_rate_hz: float, *, first_deadline_s: float) -> None:
        rate = _finite_number(update_rate_hz, "control update rate")
        if rate <= 0.0:
            raise ValueError("control update rate must be greater than zero")
        self._interval_s = 1.0 / rate
        self._next_deadline_s = _finite_number(first_deadline_s, "control deadline")

    @property
    def next_deadline_s(self) -> float:
        return self._next_deadline_s

    def is_due(self, now_s: float) -> bool:
        return _finite_number(now_s, "control clock") >= self._next_deadline_s

    def command_completed(self, completed_s: float) -> None:
        completed = _finite_number(completed_s, "control completion time")
        self._next_deadline_s = completed + self._interval_s

    def bounded_wait(self, now_s: float, maximum_s: float) -> float:
        now = _finite_number(now_s, "control clock")
        maximum = _finite_nonnegative(maximum_s, "maximum control wait")
        return max(0.0, min(maximum, self._next_deadline_s - now))


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Finite sample count and common distribution statistics."""

    count: int
    mean: float | None
    p50: float | None
    p95: float | None
    p99: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass(frozen=True, slots=True)
class RateSummary:
    """Unique event count, observed span, and derived rate."""

    count: int
    elapsed_s: float
    rate_hz: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "elapsed_s": self.elapsed_s,
            "rate_hz": self.rate_hz,
        }


@dataclass(frozen=True, slots=True)
class ConcurrentRuntimeMetrics:
    """Structured measurements from one completed concurrent runtime session."""

    capture: RateSummary
    unique_inference: RateSummary
    display: RateSummary
    encoded_output: RateSummary
    actuator_commands: RateSummary
    capture_to_inference_start_ms: DistributionSummary
    capture_to_inference_complete_ms: DistributionSummary
    result_age_at_control_ms: DistributionSummary
    capture_to_command_ms: DistributionSummary
    dropped_capture_frames: int
    dropped_render_frames: int
    recorder_submitted_frames: int
    recorder_consumed_frames: int
    dropped_recorder_frames: int
    encoded_duplicate_frames: int
    encoded_duplicate_ratio: float | None
    stale_result_events: int
    watchdog_faults: int
    session_faults: int
    shutdown_duration_ms: float
    source_type: str
    declared_source_fps: float | None
    pacing_applied: bool
    capture_outcome: str
    detector_readiness_ms: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": {
                "kind": self.source_type,
                "declared_fps": self.declared_source_fps,
                "pacing_applied": self.pacing_applied,
                "capture_outcome": self.capture_outcome,
                "detector_readiness_ms": self.detector_readiness_ms,
            },
            "captured_frames": self.capture.count,
            "inferred_frames": self.unique_inference.count,
            "displayed_frames": self.display.count,
            "capture_fps": self.capture.to_dict(),
            "unique_inference_fps": self.unique_inference.to_dict(),
            "display_fps": self.display.to_dict(),
            "encoded_output_fps": self.encoded_output.to_dict(),
            "control_hz": self.actuator_commands.to_dict(),
            "latency_ms": {
                "capture_to_inference_start": self.capture_to_inference_start_ms.to_dict(),
                "capture_to_inference_complete": (self.capture_to_inference_complete_ms.to_dict()),
                "result_age_at_control": self.result_age_at_control_ms.to_dict(),
                "capture_to_command": self.capture_to_command_ms.to_dict(),
            },
            "dropped_capture_frames": self.dropped_capture_frames,
            "dropped_render_frames": self.dropped_render_frames,
            "recorder_submitted_frames": self.recorder_submitted_frames,
            "recorder_consumed_frames": self.recorder_consumed_frames,
            "dropped_recorder_frames": self.dropped_recorder_frames,
            "encoded_duplicate_frames": self.encoded_duplicate_frames,
            "encoded_duplicate_ratio": self.encoded_duplicate_ratio,
            "stale_result_events": self.stale_result_events,
            "watchdog_faults": self.watchdog_faults,
            "session_faults": self.session_faults,
            "shutdown_duration_ms": self.shutdown_duration_ms,
        }


class RuntimeMetricsCollector:
    """Lock-protected event and latency collector shared across runtime workers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event_times: dict[str, list[float]] = {
            "capture": [],
            "inference": [],
            "display": [],
            "command": [],
        }
        self._latencies: dict[str, list[float]] = {
            "capture_to_inference_start": [],
            "capture_to_inference_complete": [],
            "result_age_at_control": [],
            "capture_to_command": [],
        }
        self.stale_result_events = 0
        self.watchdog_faults = 0
        self.session_faults = 0

    def record_event(self, name: str, timestamp_s: float) -> None:
        timestamp = _finite_number(timestamp_s, f"{name} timestamp")
        with self._lock:
            self._event_times[name].append(timestamp)

    def record_latency(self, name: str, seconds: float) -> None:
        value = _finite_nonnegative(seconds, name)
        with self._lock:
            self._latencies[name].append(value * 1000.0)

    def event_rate(self, name: str) -> RateSummary:
        """Return a thread-safe live rate summary for one known event stream."""
        with self._lock:
            timestamps = tuple(self._event_times[name])
        return _rate_summary(timestamps)

    def increment_stale(self) -> None:
        with self._lock:
            self.stale_result_events += 1

    def increment_session_fault(self, error: BaseException) -> None:
        text = str(error).upper()
        with self._lock:
            self.session_faults += 1
            if "NOT_ENABLED" in text or "WATCHDOG" in text:
                self.watchdog_faults += 1

    def snapshot(
        self,
        *,
        dropped_capture_frames: int,
        dropped_render_frames: int,
        recorder_submitted_frames: int,
        recorder_consumed_frames: int,
        dropped_recorder_frames: int,
        encoded_frame_count: int,
        encoded_duplicate_frames: int,
        encoded_started_s: float | None,
        encoded_finished_s: float | None,
        shutdown_duration_s: float,
        source_type: str = "unknown",
        declared_source_fps: float | None = None,
        pacing_applied: bool = False,
        capture_outcome: str = "unknown",
        capture_playback_started_s: float | None = None,
        detector_readiness_s: float | None = None,
    ) -> ConcurrentRuntimeMetrics:
        with self._lock:
            event_times = {name: tuple(values) for name, values in self._event_times.items()}
            latencies = {name: tuple(values) for name, values in self._latencies.items()}
            stale = self.stale_result_events
            watchdog = self.watchdog_faults
            session = self.session_faults
        duplicate_ratio = (
            None
            if encoded_frame_count == 0
            else _finite_ratio(encoded_duplicate_frames, encoded_frame_count)
        )
        if not isinstance(source_type, str) or not source_type:
            raise ValueError("source type must be a non-empty string")
        if declared_source_fps is not None:
            declared_source_fps = _finite_number(declared_source_fps, "declared source FPS")
            if declared_source_fps <= 0.0:
                raise ValueError("declared source FPS must be greater than zero")
        if not isinstance(pacing_applied, bool):
            raise ValueError("pacing applied must be bool")
        if capture_outcome not in {
            "not_started",
            "running",
            "normal_eof",
            "capture_fault",
            "cancelled",
            "stopped",
            "unknown",
        }:
            raise ValueError("capture outcome is invalid")
        capture_summary = (
            _rate_summary(event_times["capture"])
            if capture_playback_started_s is None
            else _rate_summary_from_start(
                event_times["capture"],
                capture_playback_started_s,
            )
        )
        readiness_ms = (
            None
            if detector_readiness_s is None
            else _finite_nonnegative(
                detector_readiness_s,
                "detector readiness duration",
            )
            * 1000.0
        )
        return ConcurrentRuntimeMetrics(
            capture=capture_summary,
            unique_inference=_rate_summary(event_times["inference"]),
            display=_rate_summary(event_times["display"]),
            encoded_output=_counted_rate(
                encoded_frame_count,
                encoded_started_s,
                encoded_finished_s,
            ),
            actuator_commands=_rate_summary(event_times["command"]),
            capture_to_inference_start_ms=summarize_distribution(
                latencies["capture_to_inference_start"]
            ),
            capture_to_inference_complete_ms=summarize_distribution(
                latencies["capture_to_inference_complete"]
            ),
            result_age_at_control_ms=summarize_distribution(latencies["result_age_at_control"]),
            capture_to_command_ms=summarize_distribution(latencies["capture_to_command"]),
            dropped_capture_frames=dropped_capture_frames,
            dropped_render_frames=dropped_render_frames,
            recorder_submitted_frames=recorder_submitted_frames,
            recorder_consumed_frames=recorder_consumed_frames,
            dropped_recorder_frames=dropped_recorder_frames,
            encoded_duplicate_frames=encoded_duplicate_frames,
            encoded_duplicate_ratio=duplicate_ratio,
            stale_result_events=stale,
            watchdog_faults=watchdog,
            session_faults=session,
            shutdown_duration_ms=_finite_nonnegative(shutdown_duration_s, "shutdown duration")
            * 1000.0,
            source_type=source_type,
            declared_source_fps=declared_source_fps,
            pacing_applied=pacing_applied,
            capture_outcome=capture_outcome,
            detector_readiness_ms=readiness_ms,
        )


def summarize_distribution(samples: Sequence[float]) -> DistributionSummary:
    """Summarize finite non-negative samples using linear interpolation."""
    values = tuple(_finite_nonnegative(value, "metric sample") for value in samples)
    if not values:
        return DistributionSummary(0, None, None, None, None)
    return DistributionSummary(
        count=len(values),
        mean=statistics.fmean(values),
        p50=_percentile(values, 0.50),
        p95=_percentile(values, 0.95),
        p99=_percentile(values, 0.99),
    )


def _percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rate_summary(timestamps: Sequence[float]) -> RateSummary:
    if len(timestamps) < 2:
        return RateSummary(len(timestamps), 0.0, None)
    elapsed = max(timestamps) - min(timestamps)
    return RateSummary(
        len(timestamps),
        elapsed,
        None if elapsed <= 0.0 else (len(timestamps) - 1) / elapsed,
    )


def _rate_summary_from_start(
    timestamps: Sequence[float],
    started_s: float,
) -> RateSummary:
    """Count every event while timing from an explicit active-phase start."""
    started = _finite_number(started_s, "rate start")
    if len(timestamps) < 2:
        return RateSummary(len(timestamps), 0.0, None)
    finished = _finite_number(timestamps[-1], "rate finish")
    elapsed = finished - started
    if elapsed < 0.0:
        raise ValueError("rate finish must not precede rate start")
    return RateSummary(
        len(timestamps),
        elapsed,
        None if elapsed == 0.0 else (len(timestamps) - 1) / elapsed,
    )


def _counted_rate(
    count: int,
    started_s: float | None,
    finished_s: float | None,
) -> RateSummary:
    if count < 0:
        raise ValueError("encoded frame count must be non-negative")
    if started_s is None or finished_s is None or count < 2:
        return RateSummary(count, 0.0, None)
    elapsed = _finite_nonnegative(finished_s - started_s, "encoded elapsed time")
    return RateSummary(count, elapsed, None if elapsed == 0.0 else (count - 1) / elapsed)


def _finite_nonnegative(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _finite_number(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _finite_ratio(numerator: int, denominator: int) -> float:
    if numerator < 0 or denominator <= 0 or numerator > denominator:
        raise ValueError("encoded duplicate counts must be bounded by encoded frame count")
    return numerator / denominator


def metrics_json_mapping(metrics: ConcurrentRuntimeMetrics) -> Mapping[str, object]:
    """Return the stable JSON-compatible runtime metrics mapping."""
    return metrics.to_dict()
