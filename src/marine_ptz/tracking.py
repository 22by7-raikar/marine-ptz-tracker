"""Target selection and hardware-independent PTZ control policy."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Callable

from .config import TrackingConfig
from .types import AngleLimits, Detection, Frame, PTZCommand, Target


class HighestConfidenceTargetSelector:
    """Select the highest-confidence detection with stable tie-breaking."""

    def select(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if not detections:
            return None
        detection = max(enumerate(detections), key=lambda item: (item[1].confidence, -item[0]))[1]
        return Target(detection=detection, frame_sequence=frame.sequence)


@dataclass(frozen=True, slots=True)
class TargetSelectionStatus:
    """Current single-target lock state for telemetry and diagnostics."""

    tracker: str
    active_track_count: int
    locked_track_id: int | None
    state: str
    confirmation_hits: int
    selected_confidence: float | None
    unsupported_age_s: float | None


class MarineTargetSelector:
    """Select one marine target, optionally retaining one tracked identity."""

    def __init__(
        self,
        target_classes: Sequence[str] = ("boat",),
        *,
        tracker_backend: str = "none",
        acquisition_confidence: float = 0.60,
        support_confidence: float = 0.20,
        minimum_confirmation_hits: int = 2,
        maximum_unsupported_age_s: float = 0.30,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        normalized = tuple(name.strip().casefold() for name in target_classes if name.strip())
        if not normalized:
            raise ValueError("at least one target class is required")
        tracker = tracker_backend.strip().casefold()
        if tracker not in {"none", "botsort"}:
            raise ValueError("tracker_backend must be 'none' or 'botsort'")
        for value, name in (
            (acquisition_confidence, "acquisition_confidence"),
            (support_confidence, "support_confidence"),
            (maximum_unsupported_age_s, "maximum_unsupported_age_s"),
        ):
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= support_confidence <= 1.0 or not 0.0 <= acquisition_confidence <= 1.0:
            raise ValueError("support_confidence and acquisition_confidence must be probabilities")
        if tracker == "botsort" and support_confidence > acquisition_confidence:
            raise ValueError(
                "support_confidence and acquisition_confidence must satisfy "
                "0 <= support <= acquisition <= 1"
            )
        if (
            isinstance(minimum_confirmation_hits, bool)
            or not isinstance(minimum_confirmation_hits, int)
            or minimum_confirmation_hits <= 0
        ):
            raise ValueError("minimum_confirmation_hits must be a positive integer")
        if maximum_unsupported_age_s <= 0.0:
            raise ValueError("maximum_unsupported_age_s must be greater than zero")
        self._target_classes = frozenset(normalized)
        self._tracker_backend = tracker
        self._acquisition_confidence = float(acquisition_confidence)
        self._support_confidence = float(support_confidence)
        self._minimum_confirmation_hits = minimum_confirmation_hits
        self._maximum_unsupported_age_s = float(maximum_unsupported_age_s)
        self._monotonic = monotonic
        self.reset()

    @property
    def status(self) -> TargetSelectionStatus:
        return self._status

    def reset(self) -> None:
        """Release all acquisition/lock state for a new source or runtime."""
        self._pending_track_id: int | None = None
        self._confirmation_hits = 0
        self._locked_track_id: int | None = None
        self._last_supported_s: float | None = None
        self._last_observed_s: float | None = None
        self._last_frame_sequence: int | None = None
        self._cached_target: Target | None = None
        self._status = TargetSelectionStatus(
            tracker=self._tracker_backend,
            active_track_count=0,
            locked_track_id=None,
            state="none",
            confirmation_hits=0,
            selected_confidence=None,
            unsupported_age_s=None,
        )

    def select(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        candidates = [
            detection
            for detection in detections
            if _is_valid_detection(detection) and detection.label.casefold() in self._target_classes
        ]
        if self._tracker_backend == "none":
            selected = _ranked_detection(frame, candidates)
            target = (
                None
                if selected is None
                else Target(detection=selected, frame_sequence=frame.sequence)
            )
            self._status = TargetSelectionStatus(
                tracker="none",
                active_track_count=0,
                locked_track_id=None,
                state="none" if target is None else "locked",
                confirmation_hits=0 if target is None else 1,
                selected_confidence=None if target is None else target.detection.confidence,
                unsupported_age_s=None,
            )
            return target

        if self._last_frame_sequence == frame.sequence:
            return self._selection_for_repeated_frame()
        if self._last_frame_sequence is not None and frame.sequence < self._last_frame_sequence:
            self.reset()

        now = self._timestamp()
        if self._last_observed_s is not None and now < self._last_observed_s:
            self.reset()
        self._last_observed_s = now
        self._last_frame_sequence = frame.sequence
        tracked = [
            detection
            for detection in candidates
            if detection.track_id is not None and detection.confidence >= self._support_confidence
        ]
        active_track_count = len({detection.track_id for detection in tracked})

        if self._locked_track_id is not None:
            supported = [
                detection for detection in tracked if detection.track_id == self._locked_track_id
            ]
            selected = _ranked_detection(frame, supported)
            if selected is not None:
                self._last_supported_s = now
                target = Target(detection=selected, frame_sequence=frame.sequence)
                self._cached_target = target
                self._status = TargetSelectionStatus(
                    tracker="botsort",
                    active_track_count=active_track_count,
                    locked_track_id=self._locked_track_id,
                    state="locked",
                    confirmation_hits=self._confirmation_hits,
                    selected_confidence=selected.confidence,
                    unsupported_age_s=0.0,
                )
                return target

            unsupported_age = self._unsupported_age(now)
            if unsupported_age <= self._maximum_unsupported_age_s:
                self._cached_target = None
                self._status = TargetSelectionStatus(
                    tracker="botsort",
                    active_track_count=active_track_count,
                    locked_track_id=self._locked_track_id,
                    state="occluded",
                    confirmation_hits=self._confirmation_hits,
                    selected_confidence=None,
                    unsupported_age_s=unsupported_age,
                )
                return None
            self._release_lock()

        acquisition = [
            detection
            for detection in tracked
            if detection.confidence >= self._acquisition_confidence
        ]
        selected = _ranked_detection(frame, acquisition)
        if selected is None or selected.track_id is None:
            self._pending_track_id = None
            self._confirmation_hits = 0
            self._cached_target = None
            self._status = TargetSelectionStatus(
                tracker="botsort",
                active_track_count=active_track_count,
                locked_track_id=None,
                state="none",
                confirmation_hits=0,
                selected_confidence=None,
                unsupported_age_s=None,
            )
            return None

        if selected.track_id == self._pending_track_id:
            self._confirmation_hits += 1
        else:
            self._pending_track_id = selected.track_id
            self._confirmation_hits = 1
        if self._confirmation_hits < self._minimum_confirmation_hits:
            self._cached_target = None
            self._status = TargetSelectionStatus(
                tracker="botsort",
                active_track_count=active_track_count,
                locked_track_id=None,
                state="acquiring",
                confirmation_hits=self._confirmation_hits,
                selected_confidence=None,
                unsupported_age_s=None,
            )
            return None

        self._locked_track_id = selected.track_id
        self._last_supported_s = now
        target = Target(detection=selected, frame_sequence=frame.sequence)
        self._cached_target = target
        self._status = TargetSelectionStatus(
            tracker="botsort",
            active_track_count=active_track_count,
            locked_track_id=self._locked_track_id,
            state="locked",
            confirmation_hits=self._confirmation_hits,
            selected_confidence=selected.confidence,
            unsupported_age_s=0.0,
        )
        return target

    def _timestamp(self) -> float:
        value = self._monotonic()
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
            self._release_lock()
            raise ValueError("selector monotonic clock must return a finite number")
        return float(value)

    def _unsupported_age(self, now: float) -> float:
        if self._last_supported_s is None:
            return self._maximum_unsupported_age_s + 1.0
        return max(0.0, now - self._last_supported_s)

    def _selection_for_repeated_frame(self) -> Target | None:
        if self._locked_track_id is None:
            return self._cached_target
        now = self._timestamp()
        if self._last_observed_s is not None and now < self._last_observed_s:
            self.reset()
            return None
        unsupported_age = self._unsupported_age(now)
        if unsupported_age > self._maximum_unsupported_age_s:
            active_track_count = self._status.active_track_count
            self._release_lock()
            self._cached_target = None
            self._status = TargetSelectionStatus(
                tracker="botsort",
                active_track_count=active_track_count,
                locked_track_id=None,
                state="none",
                confirmation_hits=0,
                selected_confidence=None,
                unsupported_age_s=None,
            )
            return None
        self._status = TargetSelectionStatus(
            tracker="botsort",
            active_track_count=self._status.active_track_count,
            locked_track_id=self._locked_track_id,
            state=self._status.state,
            confirmation_hits=self._confirmation_hits,
            selected_confidence=self._status.selected_confidence,
            unsupported_age_s=unsupported_age,
        )
        return self._cached_target

    def _release_lock(self) -> None:
        self._pending_track_id = None
        self._confirmation_hits = 0
        self._locked_track_id = None
        self._last_supported_s = None


def _ranked_detection(frame: Frame, candidates: Sequence[Detection]) -> Detection | None:
    if not candidates:
        return None
    frame_center_x = frame.width / 2
    frame_center_y = frame.height / 2

    def rank(detection: Detection) -> tuple[float, ...]:
        center_x, center_y = detection.center
        distance_squared = (center_x - frame_center_x) ** 2 + (center_y - frame_center_y) ** 2
        return (
            -detection.confidence,
            -detection.area,
            distance_squared,
            detection.left,
            detection.top,
            detection.right,
            detection.bottom,
            -1.0 if detection.track_id is None else float(detection.track_id),
        )

    return min(candidates, key=rank)


class ProportionalPTZController:
    """Convert image-plane error into bounded absolute servo commands.

    Positive image x and y errors increase pan and tilt respectively. Real
    actuator implementations may map these logical directions to servo wiring.
    """

    def __init__(
        self,
        settings: TrackingConfig,
        pan_limits: AngleLimits,
        tilt_limits: AngleLimits,
        *,
        initial_pan_deg: float,
        initial_tilt_deg: float,
    ) -> None:
        if not pan_limits.contains(initial_pan_deg):
            raise ValueError("initial pan angle must be within pan limits")
        if not tilt_limits.contains(initial_tilt_deg):
            raise ValueError("initial tilt angle must be within tilt limits")
        self._settings = settings
        self._pan_limits = pan_limits
        self._tilt_limits = tilt_limits
        self._neutral_pan_deg = initial_pan_deg
        self._neutral_tilt_deg = initial_tilt_deg
        self._pan_deg = initial_pan_deg
        self._tilt_deg = initial_tilt_deg

    @property
    def pan_deg(self) -> float:
        return self._pan_deg

    @property
    def tilt_deg(self) -> float:
        return self._tilt_deg

    def command_for(self, frame: Frame, target: Target) -> PTZCommand:
        """Return the next command for a visible target."""
        if frame.width <= 0 or frame.height <= 0:
            raise ValueError("frame dimensions must be greater than zero")
        center = _finite_target_center(target)
        if center is None:
            return self._command_for_lost_target(frame.sequence)
        target_x, target_y = center
        error_x = target_x - frame.width / 2
        error_y = target_y - frame.height / 2
        if not isfinite(error_x) or not isfinite(error_y):
            return self._command_for_lost_target(frame.sequence)
        pan_step = self._correction(error_x, frame.width)
        tilt_step = self._correction(error_y, frame.height)
        return self._set_command(
            self._pan_deg + pan_step,
            self._tilt_deg + tilt_step,
            frame.sequence,
        )

    def command_for_target(self, frame: Frame, target: Target | None) -> PTZCommand:
        """Handle either a visible target or the configured lost-target policy."""
        if target is not None:
            return self.command_for(frame, target)
        return self._command_for_lost_target(frame.sequence)

    def _command_for_lost_target(self, sequence: int) -> PTZCommand:
        if self._settings.lost_target_behavior == "return_to_neutral":
            pan = self._approach(self._pan_deg, self._neutral_pan_deg)
            tilt = self._approach(self._tilt_deg, self._neutral_tilt_deg)
            return self._set_command(pan, tilt, sequence)
        return self._set_command(self._pan_deg, self._tilt_deg, sequence)

    def _correction(self, error_px: float, frame_extent: int) -> float:
        if abs(error_px) <= self._settings.deadband_px:
            return 0.0
        normalized_error = error_px / (frame_extent / 2)
        proportional_step = normalized_error * self._settings.proportional_gain
        return _clamp(
            proportional_step,
            -self._settings.max_step_deg,
            self._settings.max_step_deg,
        )

    def _approach(self, current: float, target: float) -> float:
        delta = _clamp(
            target - current,
            -self._settings.max_step_deg,
            self._settings.max_step_deg,
        )
        return current + delta

    def _set_command(self, pan_deg: float, tilt_deg: float, sequence: int) -> PTZCommand:
        self._pan_deg = self._pan_limits.clamp(pan_deg)
        self._tilt_deg = self._tilt_limits.clamp(tilt_deg)
        return PTZCommand(self._pan_deg, self._tilt_deg, sequence)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _is_valid_detection(value: object) -> bool:
    if not isinstance(value, Detection):
        return False
    values = (
        value.confidence,
        value.left,
        value.top,
        value.right,
        value.bottom,
    )
    return (
        bool(value.label.strip())
        and all(isfinite(number) for number in values)
        and 0.0 <= value.confidence <= 1.0
        and value.right > value.left
        and value.bottom > value.top
    )


def _finite_target_center(target: object) -> tuple[float, float] | None:
    try:
        center = target.detection.center  # type: ignore[attr-defined]
        if not isinstance(center, Sequence) or len(center) != 2:
            return None
        x, y = center
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, Real)
            or not isinstance(y, Real)
        ):
            return None
        center_x = float(x)
        center_y = float(y)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if not isfinite(center_x) or not isfinite(center_y):
        return None
    return center_x, center_y
