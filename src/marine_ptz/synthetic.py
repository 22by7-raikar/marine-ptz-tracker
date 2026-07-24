"""Deterministic, hardware-free camera and detection sources."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .types import Detection, Frame


@dataclass(frozen=True, slots=True)
class SyntheticImage:
    """Small payload identifying a generated frame without allocating pixels."""

    sequence: int


class SyntheticCamera:
    """Produce metadata-only frames using a monotonic clock."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("synthetic frame dimensions must be greater than zero")
        self._width = width
        self._height = height
        self._clock = clock
        self._sequence = 0
        self._closed = False

    def read(self) -> Frame | None:
        if self._closed:
            return None
        sequence = self._sequence
        self._sequence += 1
        return Frame(
            image=SyntheticImage(sequence),
            timestamp_s=self._clock(),
            width=self._width,
            height=self._height,
            sequence=sequence,
        )

    def close(self) -> None:
        self._closed = True


class SyntheticTargetDetector:
    """Generate one bounding box that follows a repeatable two-axis path."""

    def __init__(
        self,
        *,
        label: str = "boat",
        confidence: float = 1.0,
        box_width_ratio: float = 0.15,
        box_height_ratio: float = 0.15,
        travel_frames: int = 30,
    ) -> None:
        if not label:
            raise ValueError("synthetic target label must not be empty")
        if not 0 <= confidence <= 1:
            raise ValueError("synthetic confidence must be between 0 and 1")
        if not 0 < box_width_ratio <= 1 or not 0 < box_height_ratio <= 1:
            raise ValueError("synthetic box ratios must be between 0 and 1")
        if travel_frames < 2:
            raise ValueError("synthetic travel_frames must be at least 2")
        self._label = label
        self._confidence = confidence
        self._box_width_ratio = box_width_ratio
        self._box_height_ratio = box_height_ratio
        self._travel_frames = travel_frames

    def detect(self, frame: Frame) -> Sequence[Detection]:
        box_width = frame.width * self._box_width_ratio
        box_height = frame.height * self._box_height_ratio
        x_progress = self._triangle_wave(frame.sequence)
        y_progress = self._triangle_wave(frame.sequence + self._travel_frames // 2)
        left = x_progress * (frame.width - box_width)
        top = y_progress * (frame.height - box_height)
        return (
            Detection(
                label=self._label,
                confidence=self._confidence,
                left=left,
                top=top,
                right=left + box_width,
                bottom=top + box_height,
            ),
        )

    def _triangle_wave(self, sequence: int) -> float:
        span = self._travel_frames - 1
        position = sequence % (2 * span)
        if position > span:
            position = 2 * span - position
        return position / span
