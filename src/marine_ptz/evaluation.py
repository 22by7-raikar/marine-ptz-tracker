"""Passive closed-loop tracking and control evaluation primitives."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from .config import TrackingConfig
from .tracking import TargetSelectionStatus
from .types import AngleLimits, Detection, Frame, PTZCommand, Target


@dataclass(frozen=True, slots=True)
class ObservationSample:
    """One immutable selector observation in original full-frame coordinates."""

    frame_sequence: int
    timestamp_s: float
    frame_width: int
    frame_height: int
    relevant_detection_count: int
    observed_track_ids: tuple[int, ...]
    missing_track_id_count: int
    malformed_detection_count: int
    tracker: str
    active_track_count: int
    observed_track_id: int | None
    locked_track_id: int | None
    selector_state: str
    confirmation_hits: int
    selected_confidence: float | None
    unsupported_age_s: float | None
    target_present: bool
    center_x_px: float | None
    center_y_px: float | None
    error_x_px: float | None
    error_y_px: float | None
    normalized_error_x: float | None
    normalized_error_y: float | None
    radial_normalized_error: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_sequence,
            "monotonic_timestamp_s": self.timestamp_s,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "tracker_mode": self.tracker,
            "relevant_detection_count": self.relevant_detection_count,
            "active_track_count": self.active_track_count,
            "observed_track_ids": list(self.observed_track_ids),
            "observed_track_id": self.observed_track_id,
            "locked_track_id": self.locked_track_id,
            "selector_state": self.selector_state,
            "confirmation_hits": self.confirmation_hits,
            "selected_current_observation_confidence": self.selected_confidence,
            "unsupported_age_s": self.unsupported_age_s,
            "missing_track_id_count": self.missing_track_id_count,
            "malformed_detection_count": self.malformed_detection_count,
            "target_present_current_observation": self.target_present,
            "observed_box_center_px": (
                None
                if self.center_x_px is None or self.center_y_px is None
                else {"x": self.center_x_px, "y": self.center_y_px}
            ),
            "signed_center_error_px": (
                None
                if self.error_x_px is None or self.error_y_px is None
                else {"horizontal": self.error_x_px, "vertical": self.error_y_px}
            ),
            "normalized_center_error": (
                None
                if self.normalized_error_x is None or self.normalized_error_y is None
                else {
                    "horizontal": self.normalized_error_x,
                    "vertical": self.normalized_error_y,
                    "radial": self.radial_normalized_error,
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class ControlSample:
    """One actual control decision, including stale decisions that send no command."""

    frame_sequence: int
    timestamp_s: float
    fresh: bool
    result_age_ms: float | None
    target_supplied: bool
    within_deadband: bool | None
    lost_target_action: str | None
    command: PTZCommand | None
    pan_delta_deg: float | None
    tilt_delta_deg: float | None
    pan_max_step_limited: bool
    tilt_max_step_limited: bool
    pan_limit_saturated: bool
    tilt_limit_saturated: bool
    pan_limit_hit: bool
    tilt_limit_hit: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_sequence,
            "monotonic_timestamp_s": self.timestamp_s,
            "fresh_for_control": self.fresh,
            "result_age_ms": self.result_age_ms,
            "target_supplied_to_controller": self.target_supplied,
            "controller_deadband": self.within_deadband,
            "lost_target_action": self.lost_target_action,
            "logical_command": (
                None
                if self.command is None
                else {"pan_deg": self.command.pan_deg, "tilt_deg": self.command.tilt_deg}
            ),
            "command_delta_deg": (
                None
                if self.pan_delta_deg is None or self.tilt_delta_deg is None
                else {"pan": self.pan_delta_deg, "tilt": self.tilt_delta_deg}
            ),
            "maximum_step_limited": {
                "pan": self.pan_max_step_limited,
                "tilt": self.tilt_max_step_limited,
            },
            "logical_limit_saturated": {
                "pan": self.pan_limit_saturated,
                "tilt": self.tilt_limit_saturated,
            },
            "logical_limit_hit": {"pan": self.pan_limit_hit, "tilt": self.tilt_limit_hit},
        }


class EvaluationObserver(Protocol):
    """Receive passive immutable observations without owning runtime components."""

    def on_observation(self, sample: ObservationSample) -> None:
        """Observe one newly consumed inference result."""

    def on_control(self, sample: ControlSample) -> None:
        """Observe one scheduled control decision."""


@dataclass(frozen=True, slots=True)
class SettlingEvent:
    """Explicit frame window and tolerance for one declared target-motion event."""

    name: str
    start_frame: int
    end_frame: int
    tolerance: float
    dwell_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("settling event name must be non-empty")
        if (
            isinstance(self.start_frame, bool)
            or not isinstance(self.start_frame, int)
            or self.start_frame < 0
        ):
            raise ValueError("settling event start frame must be a non-negative integer")
        if (
            isinstance(self.end_frame, bool)
            or not isinstance(self.end_frame, int)
            or self.end_frame < self.start_frame
        ):
            raise ValueError("settling event end frame must be at least its start frame")
        tolerance = _finite(self.tolerance, "settling event tolerance")
        dwell = _finite(self.dwell_s, "settling event dwell")
        if tolerance <= 0.0:
            raise ValueError("settling event tolerance must be greater than zero")
        if dwell < 0.0:
            raise ValueError("settling event dwell must be non-negative")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "dwell_s", dwell)


def make_observation_sample(
    frame: Frame,
    detections: Sequence[object],
    target: Target | None,
    status: TargetSelectionStatus,
    target_classes: Sequence[str],
) -> ObservationSample:
    """Build one current-observation-only evaluation sample."""
    targets = {name.strip().casefold() for name in target_classes if name.strip()}
    relevant: list[Detection] = []
    malformed = 0
    for value in detections:
        if not isinstance(value, Detection):
            malformed += 1
            continue
        if value.label.casefold() in targets:
            relevant.append(value)
    observed_ids = tuple(sorted({item.track_id for item in relevant if item.track_id is not None}))
    missing_ids = (
        sum(item.track_id is None for item in relevant) if status.tracker == "botsort" else 0
    )
    current = (
        None if target is None or target.frame_sequence != frame.sequence else target.detection
    )
    if current is None:
        center_x = center_y = error_x = error_y = normalized_x = normalized_y = radial = None
    else:
        center_x, center_y = current.center
        error_x = center_x - frame.width / 2.0
        error_y = center_y - frame.height / 2.0
        normalized_x = error_x / (frame.width / 2.0)
        normalized_y = error_y / (frame.height / 2.0)
        radial = math.hypot(normalized_x, normalized_y)
    return ObservationSample(
        frame_sequence=frame.sequence,
        timestamp_s=_finite(frame.timestamp_s, "frame timestamp"),
        frame_width=frame.width,
        frame_height=frame.height,
        relevant_detection_count=len(relevant),
        observed_track_ids=observed_ids,
        missing_track_id_count=missing_ids,
        malformed_detection_count=malformed,
        tracker=status.tracker,
        active_track_count=status.active_track_count,
        observed_track_id=None if current is None else current.track_id,
        locked_track_id=status.locked_track_id,
        selector_state=status.state,
        confirmation_hits=status.confirmation_hits,
        selected_confidence=status.selected_confidence if current is not None else None,
        unsupported_age_s=status.unsupported_age_s,
        target_present=current is not None,
        center_x_px=center_x,
        center_y_px=center_y,
        error_x_px=error_x,
        error_y_px=error_y,
        normalized_error_x=normalized_x,
        normalized_error_y=normalized_y,
        radial_normalized_error=radial,
    )


def make_control_sample(
    *,
    frame: Frame,
    target: Target | None,
    settings: TrackingConfig,
    pan_limits: AngleLimits,
    tilt_limits: AngleLimits,
    neutral_pan_deg: float,
    neutral_tilt_deg: float,
    previous_pan_deg: float,
    previous_tilt_deg: float,
    command: PTZCommand | None,
    timestamp_s: float,
    fresh: bool,
    result_age_s: float | None,
) -> ControlSample:
    """Describe an existing controller decision without changing its output."""
    timestamp = _finite(timestamp_s, "control timestamp")
    age_ms = None if result_age_s is None else _finite(result_age_s, "result age") * 1000.0
    if age_ms is not None and age_ms < 0.0:
        raise ValueError("result age must be non-negative")
    current = (
        None if target is None or target.frame_sequence != frame.sequence else target.detection
    )
    within_deadband: bool | None = None
    pan_step_limited = tilt_step_limited = False
    pan_saturated = tilt_saturated = False
    lost_action: str | None = None
    if current is not None:
        error_x = current.center[0] - frame.width / 2.0
        error_y = current.center[1] - frame.height / 2.0
        within_deadband = (
            abs(error_x) <= settings.deadband_px and abs(error_y) <= settings.deadband_px
        )
        raw_pan = (
            0.0
            if abs(error_x) <= settings.deadband_px
            else (error_x / (frame.width / 2.0) * settings.proportional_gain)
        )
        raw_tilt = (
            0.0
            if abs(error_y) <= settings.deadband_px
            else (error_y / (frame.height / 2.0) * settings.proportional_gain)
        )
        pan_step_limited = abs(raw_pan) > settings.max_step_deg
        tilt_step_limited = abs(raw_tilt) > settings.max_step_deg
        bounded_pan = max(-settings.max_step_deg, min(settings.max_step_deg, raw_pan))
        bounded_tilt = max(-settings.max_step_deg, min(settings.max_step_deg, raw_tilt))
        pan_saturated = not pan_limits.contains(previous_pan_deg + bounded_pan)
        tilt_saturated = not tilt_limits.contains(previous_tilt_deg + bounded_tilt)
    else:
        lost_action = settings.lost_target_behavior
        if lost_action == "return_to_neutral":
            pan_step_limited = abs(neutral_pan_deg - previous_pan_deg) > settings.max_step_deg
            tilt_step_limited = abs(neutral_tilt_deg - previous_tilt_deg) > settings.max_step_deg

    pan_delta = None if command is None else command.pan_deg - previous_pan_deg
    tilt_delta = None if command is None else command.tilt_deg - previous_tilt_deg
    return ControlSample(
        frame_sequence=frame.sequence,
        timestamp_s=timestamp,
        fresh=fresh,
        result_age_ms=age_ms,
        target_supplied=current is not None and fresh,
        within_deadband=within_deadband if fresh else None,
        lost_target_action=lost_action if fresh else None,
        command=command,
        pan_delta_deg=pan_delta,
        tilt_delta_deg=tilt_delta,
        pan_max_step_limited=pan_step_limited if fresh else False,
        tilt_max_step_limited=tilt_step_limited if fresh else False,
        pan_limit_saturated=pan_saturated if fresh else False,
        tilt_limit_saturated=tilt_saturated if fresh else False,
        pan_limit_hit=(
            command is not None
            and command.pan_deg in (pan_limits.minimum_deg, pan_limits.maximum_deg)
        ),
        tilt_limit_hit=(
            command is not None
            and command.tilt_deg in (tilt_limits.minimum_deg, tilt_limits.maximum_deg)
        ),
    )


class ClosedLoopMetricsCollector:
    """Aggregate passive observation and control samples deterministically."""

    def __init__(self, settling_events: Sequence[SettlingEvent] = ()) -> None:
        self.observations: list[ObservationSample] = []
        self.controls: list[ControlSample] = []
        self._settling_events = tuple(settling_events)

    def on_observation(self, sample: ObservationSample) -> None:
        self.observations.append(sample)

    def on_control(self, sample: ControlSample) -> None:
        self.controls.append(sample)

    def to_dict(self) -> Mapping[str, object]:
        observations = tuple(self.observations)
        controls = tuple(self.controls)
        supported = tuple(item for item in controls if item.target_supplied)
        command_samples = tuple(item for item in controls if item.command is not None)
        current = tuple(item for item in observations if item.target_present)
        identity = _identity_metrics(observations)
        states = Counter(item.selector_state for item in observations)
        state_seconds: Counter[str] = Counter()
        for first, second in zip(observations, observations[1:]):
            state_seconds[first.selector_state] += max(0.0, second.timestamp_s - first.timestamp_s)
        return {
            "available": True,
            "definitions": {
                "observation_denominator": "newly consumed inference results",
                "control_denominator": "scheduled control decisions",
                "center_coordinates": "original full-frame pixels from current locked observations",
            },
            "target_availability": {
                "observation_frames": len(observations),
                "detected_frames": sum(item.relevant_detection_count > 0 for item in observations),
                "detected_frame_ratio": _ratio(
                    sum(item.relevant_detection_count > 0 for item in observations),
                    len(observations),
                ),
                "selected_target_frames": len(current),
                "selected_target_ratio": _ratio(len(current), len(observations)),
                "controller_supported_updates": len(supported),
                "controller_supported_target_ratio": _ratio(len(supported), len(controls)),
                "selector_state_frames": dict(sorted(states.items())),
                "selector_state_seconds": dict(sorted(state_seconds.items())),
            },
            "acquisition": _acquisition_metrics(observations),
            "reacquisition": identity["reacquisition"],
            "identity": identity["identity"],
            "centering": {
                "sample_count": len(supported),
                "horizontal": _axis_metrics(
                    _supported_errors(supported, observations, "horizontal")
                ),
                "vertical": _axis_metrics(_supported_errors(supported, observations, "vertical")),
                "radial": _distribution(
                    _supported_errors(supported, observations, "radial"), maximum=True
                ),
            },
            "control": _control_metrics(controls, command_samples, supported),
            "freshness": {
                "control_decisions": len(controls),
                "stale_result_count": sum(not item.fresh for item in controls),
                "stale_result_rate": _ratio(
                    sum(not item.fresh for item in controls), len(controls)
                ),
                "result_age_at_control_ms": _distribution(
                    tuple(
                        item.result_age_ms for item in controls if item.result_age_ms is not None
                    ),
                    maximum=True,
                ),
            },
            "settling": {
                "available": bool(self._settling_events),
                "events": [
                    _settling_result(event, observations) for event in self._settling_events
                ],
                "reason": None if self._settling_events else "no settling events declared",
            },
            "frame_samples": [sample.to_dict() for sample in observations],
            "control_samples": [sample.to_dict() for sample in controls],
        }


def _acquisition_metrics(samples: Sequence[ObservationSample]) -> dict[str, object]:
    first_observation = next((item for item in samples if item.relevant_detection_count), None)
    first_lock = next(
        (item for item in samples if item.selector_state == "locked" and item.target_present),
        None,
    )
    if first_observation is None or first_lock is None:
        return {
            "available": False,
            "first_observation_frame": None
            if first_observation is None
            else first_observation.frame_sequence,
            "initial_lock_frame": None,
            "frames_to_lock": None,
            "milliseconds_to_lock": None,
            "confirmation_observations": None,
        }
    return {
        "available": True,
        "first_observation_frame": first_observation.frame_sequence,
        "initial_lock_frame": first_lock.frame_sequence,
        "frames_to_lock": first_lock.frame_sequence - first_observation.frame_sequence,
        "milliseconds_to_lock": max(
            0.0, (first_lock.timestamp_s - first_observation.timestamp_s) * 1000.0
        ),
        "confirmation_observations": first_lock.confirmation_hits,
    }


def _identity_metrics(samples: Sequence[ObservationSample]) -> dict[str, object]:
    unique_ids = sorted({track_id for item in samples for track_id in item.observed_track_ids})
    protected_changes = 0
    post_expiration_changes = 0
    locked_changes = 0
    previous: ObservationSample | None = None
    protected_id: int | None = None
    expired_id: int | None = None
    occlusion_start: ObservationSample | None = None
    recoveries: list[dict[str, object]] = []
    for sample in samples:
        if previous is not None and previous.selector_state == "occluded":
            if (
                sample.selector_state == "locked"
                and sample.locked_track_id == previous.locked_track_id
            ):
                start = occlusion_start or previous
                recoveries.append(_recovery_record("same_id_recovery", start, sample, True))
                occlusion_start = None
            elif sample.locked_track_id is None:
                expired_id = previous.locked_track_id
                protected_id = None
        if sample.selector_state == "occluded" and occlusion_start is None:
            occlusion_start = sample
        if sample.locked_track_id is not None:
            if protected_id is None:
                if expired_id is not None:
                    locked_changes += int(sample.locked_track_id != expired_id)
                    post_expiration_changes += int(sample.locked_track_id != expired_id)
                    start = occlusion_start or sample
                    recoveries.append(
                        _recovery_record("post_expiration_reacquisition", start, sample, True)
                    )
                    expired_id = None
                    occlusion_start = None
                protected_id = sample.locked_track_id
            elif sample.locked_track_id != protected_id:
                locked_changes += 1
                protected_changes += 1
                protected_id = sample.locked_track_id
        previous = sample
    if occlusion_start is not None:
        recoveries.append(
            {
                "kind": "unrecovered",
                "start_frame": occlusion_start.frame_sequence,
                "end_frame": None,
                "duration_frames": None,
                "duration_ms": None,
                "recovered": False,
            }
        )
    durations = tuple(
        float(item["duration_ms"]) for item in recoveries if item["duration_ms"] is not None
    )
    return {
        "identity": {
            "unique_observed_track_ids": unique_ids,
            "unique_observed_track_id_count": len(unique_ids),
            "locked_id_change_count": locked_changes,
            "protected_lock_id_change_count": protected_changes,
            "post_expiration_id_change_count": post_expiration_changes,
            "missing_id_event_count": sum(item.missing_track_id_count for item in samples),
            "malformed_detection_event_count": sum(
                item.malformed_detection_count for item in samples
            ),
        },
        "reacquisition": {
            "event_count": len(recoveries),
            "same_id_recovery_count": sum(
                item["kind"] == "same_id_recovery" for item in recoveries
            ),
            "post_expiration_reacquisition_count": sum(
                item["kind"] == "post_expiration_reacquisition" for item in recoveries
            ),
            "unrecovered_count": sum(not bool(item["recovered"]) for item in recoveries),
            "duration_ms": _distribution(durations, maximum=True),
            "events": recoveries,
        },
    }


def _recovery_record(
    kind: str,
    start: ObservationSample,
    end: ObservationSample,
    recovered: bool,
) -> dict[str, object]:
    return {
        "kind": kind,
        "start_frame": start.frame_sequence,
        "end_frame": end.frame_sequence,
        "duration_frames": end.frame_sequence - start.frame_sequence,
        "duration_ms": max(0.0, (end.timestamp_s - start.timestamp_s) * 1000.0),
        "recovered": recovered,
    }


def _supported_errors(
    controls: Sequence[ControlSample],
    observations: Sequence[ObservationSample],
    axis: str,
) -> tuple[float, ...]:
    by_sequence = {item.frame_sequence: item for item in observations}
    values: list[float] = []
    for control in controls:
        if not control.target_supplied:
            continue
        observation = by_sequence.get(control.frame_sequence)
        if observation is None or not observation.target_present:
            continue
        value = {
            "horizontal": observation.normalized_error_x,
            "vertical": observation.normalized_error_y,
            "radial": observation.radial_normalized_error,
        }[axis]
        if value is not None:
            values.append(value)
    return tuple(values)


def _axis_metrics(values: Sequence[float]) -> dict[str, object]:
    absolute = tuple(abs(value) for value in values)
    return {
        "signed_mean": None if not values else statistics.fmean(values),
        "absolute": _distribution(absolute, maximum=True),
    }


def _control_metrics(
    controls: Sequence[ControlSample],
    commands: Sequence[ControlSample],
    supported: Sequence[ControlSample],
) -> dict[str, object]:
    pan_deltas = tuple(
        abs(item.pan_delta_deg) for item in commands if item.pan_delta_deg is not None
    )
    tilt_deltas = tuple(
        abs(item.tilt_delta_deg) for item in commands if item.tilt_delta_deg is not None
    )
    deadband_count = sum(item.within_deadband is True for item in supported)
    max_step_count = sum(
        item.pan_max_step_limited or item.tilt_max_step_limited for item in commands
    )
    timestamps = tuple(item.timestamp_s for item in commands)
    lost_actions = Counter(
        item.lost_target_action for item in commands if item.lost_target_action is not None
    )
    return {
        "commanded_update_count": len(commands),
        "effective_update_rate_hz": _event_rate(timestamps),
        "command_delta_deg": {
            "pan": _distribution(pan_deltas, maximum=True),
            "tilt": _distribution(tilt_deltas, maximum=True),
        },
        "deadband_update_count": deadband_count,
        "deadband_frame_ratio": _ratio(deadband_count, len(supported)),
        "maximum_step_limited_count": max_step_count,
        "maximum_step_limited_rate": _ratio(max_step_count, len(commands)),
        "maximum_step_limited_by_axis": {
            "pan": sum(item.pan_max_step_limited for item in commands),
            "tilt": sum(item.tilt_max_step_limited for item in commands),
        },
        "logical_limit_saturation_count": {
            "pan": sum(item.pan_limit_saturated for item in commands),
            "tilt": sum(item.tilt_limit_saturated for item in commands),
        },
        "logical_limit_hit_count": {
            "pan": sum(item.pan_limit_hit for item in commands),
            "tilt": sum(item.tilt_limit_hit for item in commands),
        },
        "direction_reversal_count": {
            "pan": _direction_reversals(
                tuple(item.pan_delta_deg for item in commands if item.pan_delta_deg is not None)
            ),
            "tilt": _direction_reversals(
                tuple(item.tilt_delta_deg for item in commands if item.tilt_delta_deg is not None)
            ),
        },
        "lost_target_action_count": dict(sorted(lost_actions.items())),
    }


def _settling_result(
    event: SettlingEvent,
    samples: Sequence[ObservationSample],
) -> dict[str, object]:
    window = tuple(
        item for item in samples if event.start_frame <= item.frame_sequence <= event.end_frame
    )
    base = {
        "name": event.name,
        "start_frame": event.start_frame,
        "end_frame": event.end_frame,
        "tolerance": event.tolerance,
        "dwell_s": event.dwell_s,
    }
    if not window:
        return {**base, "available": False, "reason": "declared frame window was not observed"}
    supported = tuple(item for item in window if item.radial_normalized_error is not None)
    if not supported:
        return {**base, "available": False, "reason": "no current observations in window"}
    event_time = window[0].timestamp_s
    candidate_start: float | None = None
    settled_at: float | None = None
    for sample in window:
        radial = sample.radial_normalized_error
        if radial is None or radial > event.tolerance:
            candidate_start = None
            continue
        if candidate_start is None:
            candidate_start = sample.timestamp_s
        if sample.timestamp_s - candidate_start >= event.dwell_s:
            settled_at = candidate_start
            break
    radial_values = tuple(float(item.radial_normalized_error) for item in supported)
    overshoot = max(0.0, max(radial_values) - radial_values[0])
    return {
        **base,
        "available": True,
        "settled": settled_at is not None,
        "settling_time_ms": (
            None if settled_at is None else max(0.0, (settled_at - event_time) * 1000.0)
        ),
        "overshoot_radial_normalized": overshoot,
        "peak_radial_normalized_error": max(radial_values),
        "reason": None if settled_at is not None else "tolerance dwell not achieved",
    }


def _distribution(values: Sequence[float], *, maximum: bool) -> dict[str, object]:
    numbers = tuple(_finite(value, "metric sample") for value in values)
    result: dict[str, object] = {
        "count": len(numbers),
        "p50": _percentile(numbers, 0.50),
        "p95": _percentile(numbers, 0.95),
        "p99": _percentile(numbers, 0.99),
    }
    if maximum:
        result["max"] = None if not numbers else max(numbers)
    return result


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _direction_reversals(values: Sequence[float]) -> int:
    previous = 0
    reversals = 0
    for value in values:
        sign = 1 if value > 0.0 else -1 if value < 0.0 else 0
        if sign == 0:
            continue
        if previous and sign != previous:
            reversals += 1
        previous = sign
    return reversals


def _event_rate(timestamps: Sequence[float]) -> float | None:
    if len(timestamps) < 2:
        return None
    elapsed = timestamps[-1] - timestamps[0]
    return None if elapsed <= 0.0 else (len(timestamps) - 1) / elapsed


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)
