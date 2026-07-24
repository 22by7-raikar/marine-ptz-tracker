"""Target selection and hardware-independent PTZ control policy."""

from __future__ import annotations

from collections.abc import Sequence

from .config import TrackingConfig
from .types import AngleLimits, Detection, Frame, PTZCommand, Target


class HighestConfidenceTargetSelector:
    """Select the highest-confidence detection with stable tie-breaking."""

    def select(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if not detections:
            return None
        detection = max(enumerate(detections), key=lambda item: (item[1].confidence, -item[0]))[1]
        return Target(detection=detection, frame_sequence=frame.sequence)


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
        target_x, target_y = target.detection.center
        error_x = target_x - frame.width / 2
        error_y = target_y - frame.height / 2
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
        if self._settings.lost_target_behavior == "return_to_neutral":
            pan = self._approach(self._pan_deg, self._neutral_pan_deg)
            tilt = self._approach(self._tilt_deg, self._neutral_tilt_deg)
            return self._set_command(pan, tilt, frame.sequence)
        return self._set_command(self._pan_deg, self._tilt_deg, frame.sequence)

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
