"""Hardware-free actuator implementations."""

from __future__ import annotations

from .types import AngleLimits, PTZCommand


class SimulatedPTZActuator:
    """Maintain bounded in-memory PTZ state."""

    def __init__(
        self,
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
        self._pan_limits = pan_limits
        self._tilt_limits = tilt_limits
        self._pan_deg = initial_pan_deg
        self._tilt_deg = initial_tilt_deg
        self._last_sequence: int | None = None
        self._closed = False

    @property
    def pan_deg(self) -> float:
        return self._pan_deg

    @property
    def tilt_deg(self) -> float:
        return self._tilt_deg

    @property
    def last_sequence(self) -> int | None:
        return self._last_sequence

    def apply(self, command: PTZCommand) -> None:
        if self._closed:
            raise RuntimeError("simulated actuator is closed")
        self._pan_deg = self._pan_limits.clamp(command.pan_deg)
        self._tilt_deg = self._tilt_limits.clamp(command.tilt_deg)
        self._last_sequence = command.sequence

    def close(self) -> None:
        self._closed = True
