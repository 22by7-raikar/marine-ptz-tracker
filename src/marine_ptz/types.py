"""Hardware-neutral values exchanged between tracker components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AngleLimits:
    """Inclusive minimum and maximum angle for one servo axis."""

    minimum_deg: float
    maximum_deg: float

    def clamp(self, value: float) -> float:
        """Clamp an angle to this range."""
        return min(self.maximum_deg, max(self.minimum_deg, value))

    def contains(self, value: float) -> bool:
        """Return whether an angle is inside this range."""
        return self.minimum_deg <= value <= self.maximum_deg


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured image and its capture-time metadata."""

    image: Any
    timestamp_s: float
    width: int
    height: int
    sequence: int


@dataclass(frozen=True, slots=True)
class Detection:
    """A labeled bounding box in pixel coordinates."""

    label: str
    confidence: float
    left: float
    top: float
    right: float
    bottom: float

    @property
    def center(self) -> tuple[float, float]:
        """Return the bounding-box center in pixels."""
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)


@dataclass(frozen=True, slots=True)
class Target:
    """The detection selected for control in a particular frame."""

    detection: Detection
    frame_sequence: int


@dataclass(frozen=True, slots=True)
class PTZCommand:
    """A bounded absolute pan/tilt request, expressed in degrees."""

    pan_deg: float
    tilt_deg: float
    sequence: int
