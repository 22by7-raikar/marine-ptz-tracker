"""Target selection and hardware-independent PTZ control policy."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real

from .config import TrackingConfig
from .types import AngleLimits, Detection, Frame, PTZCommand, Target


class HighestConfidenceTargetSelector:
    """Select the highest-confidence detection with stable tie-breaking."""

    def select(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if not detections:
            return None
        detection = max(enumerate(detections), key=lambda item: (item[1].confidence, -item[0]))[1]
        return Target(detection=detection, frame_sequence=frame.sequence)


class MarineTargetSelector:
    """Select configured marine classes using a deterministic ranking policy."""

    def __init__(self, target_classes: Sequence[str] = ("boat",)) -> None:
        normalized = tuple(name.strip().casefold() for name in target_classes if name.strip())
        if not normalized:
            raise ValueError("at least one target class is required")
        self._target_classes = frozenset(normalized)

    def select(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        candidates = [
            detection
            for detection in detections
            if _is_valid_detection(detection) and detection.label.casefold() in self._target_classes
        ]
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
            )

        selected = min(candidates, key=rank)
        return Target(detection=selected, frame_sequence=frame.sequence)


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
