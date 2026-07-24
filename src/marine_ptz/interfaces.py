"""Protocol boundaries for implementations that may touch hardware."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .types import Detection, Frame, PTZCommand, Target


@runtime_checkable
class CameraSource(Protocol):
    """Supplies frames; concrete implementations may own camera resources."""

    def read(self) -> Frame | None:
        """Return the next frame, or None when no frame is currently available."""

    def close(self) -> None:
        """Release resources owned by this source."""


@runtime_checkable
class Detector(Protocol):
    """Finds candidate objects without making actuator decisions."""

    def detect(self, frame: Frame) -> Sequence[Detection]:
        """Return detections for one frame."""


@runtime_checkable
class TargetSelector(Protocol):
    """Chooses at most one control target from current candidates."""

    def select(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        """Return the selected target or None when no target is suitable."""


@runtime_checkable
class PTZController(Protocol):
    """Maps a selected target to a bounded absolute PTZ command."""

    def command_for(self, frame: Frame, target: Target) -> PTZCommand:
        """Calculate a command without performing hardware I/O."""


@runtime_checkable
class Actuator(Protocol):
    """Applies commands to a real or simulated PTZ mechanism."""

    def apply(self, command: PTZCommand) -> None:
        """Apply a previously validated bounded command."""

    def close(self) -> None:
        """Place the actuator in its implementation-defined safe state."""
