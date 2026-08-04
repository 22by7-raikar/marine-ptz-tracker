"""Wall-clock-correct scheduling for a fixed-frame-rate video writer."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any, Protocol


class VideoWriter(Protocol):
    """Minimal writer surface required by the recorder."""

    def write(self, frame: Any) -> None:
        """Append one encoded frame."""

    def release(self) -> None:
        """Release the underlying encoder."""


class WallClockVideoRecorder:
    """Map processed frames onto a fixed-FPS timeline using frame holds."""

    def __init__(
        self,
        writer: VideoWriter,
        encoded_fps: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(encoded_fps, bool) or not isinstance(encoded_fps, (int, float)):
            raise ValueError("encoded output FPS must be a finite positive number")
        if not math.isfinite(float(encoded_fps)) or encoded_fps <= 0.0:
            raise ValueError("encoded output FPS must be a finite positive number")
        self._writer = writer
        self._encoded_fps = float(encoded_fps)
        self._monotonic = monotonic
        self._started_s: float | None = None
        self._last_timestamp_s: float | None = None
        self._held_frame: Any | None = None
        self._submitted_frames = 0
        self._encoded_frames = 0
        self._duplicated_frames = 0
        self._released = False

    @property
    def encoded_fps(self) -> float:
        return self._encoded_fps

    @property
    def submitted_frame_count(self) -> int:
        """Number of distinct processed frames offered to the recorder."""
        return self._submitted_frames

    @property
    def encoded_frame_count(self) -> int:
        """Number of frames written to the fixed-FPS output stream."""
        return self._encoded_frames

    @property
    def duplicated_frame_count(self) -> int:
        """Number of encoded slots filled by holding an earlier frame."""
        return self._duplicated_frames

    def write(self, frame: Any, *, timestamp_s: float | None = None) -> None:
        """Submit one processed frame at a finite nondecreasing timestamp."""
        if self._released:
            raise RuntimeError("cannot write after video recorder release")
        timestamp = self._monotonic() if timestamp_s is None else timestamp_s
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
        ):
            raise RuntimeError("video recorder monotonic timestamp must be finite")
        now = float(timestamp)
        if self._last_timestamp_s is not None and now < self._last_timestamp_s:
            raise RuntimeError("video recorder monotonic timestamp moved backwards")

        if self._started_s is None:
            self._writer.write(frame)
            self._started_s = now
            self._encoded_frames = 1
        else:
            target_count = math.floor((now - self._started_s) * self._encoded_fps + 1e-9) + 1
            missing = target_count - self._encoded_frames
            if missing > 0:
                for _ in range(missing - 1):
                    self._writer.write(self._held_frame)
                    self._encoded_frames += 1
                    self._duplicated_frames += 1
                self._writer.write(frame)
                self._encoded_frames += 1

        self._held_frame = frame
        self._last_timestamp_s = now
        self._submitted_frames += 1

    def release(self) -> None:
        """Release the wrapped writer exactly once without extending time."""
        if self._released:
            return
        self._released = True
        self._writer.release()
