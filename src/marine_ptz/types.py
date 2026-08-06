"""Hardware-neutral values exchanged between tracker components."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    return number


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
    """One captured frame and its capture-time metadata.

    ``image`` is intentionally opaque so synthetic callers can use a lightweight
    marker while real sources can supply an OpenCV/NumPy array without making
    NumPy a base-package dependency. Image identity and content are excluded
    from equality, hashing, and repr; frame metadata defines value equality.
    """

    image: object | None = field(compare=False, hash=False, repr=False)
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
    track_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("detection.label must be a non-empty string")
        object.__setattr__(self, "label", self.label.strip())
        for field_name in ("confidence", "left", "top", "right", "bottom"):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), f"detection.{field_name}"),
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection.confidence must be between 0 and 1")
        if self.right <= self.left:
            raise ValueError("detection bounding box width must be greater than zero")
        if self.bottom <= self.top:
            raise ValueError("detection bounding box height must be greater than zero")
        if self.track_id is not None and (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, int)
            or self.track_id < 0
        ):
            raise ValueError("detection.track_id must be a non-negative integer when provided")

    @property
    def center(self) -> tuple[float, float]:
        """Return the bounding-box center in pixels."""
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    @property
    def area(self) -> float:
        """Return bounding-box area in square pixels."""
        return max(0.0, self.right - self.left) * max(0.0, self.bottom - self.top)


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
