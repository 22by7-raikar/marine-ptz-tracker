"""Pure post-detection digital-zoom geometry and smoothing state."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import Detection


@dataclass(frozen=True, slots=True)
class ZoomTransform:
    """One integer crop mapped back to the original output dimensions."""

    frame_width: int
    frame_height: int
    crop_left: int
    crop_top: int
    crop_right: int
    crop_bottom: int
    zoom_factor: float

    @property
    def crop_width(self) -> int:
        return self.crop_right - self.crop_left

    @property
    def crop_height(self) -> int:
        return self.crop_bottom - self.crop_top

    @property
    def is_identity(self) -> bool:
        return (
            self.crop_left == 0
            and self.crop_top == 0
            and self.crop_right == self.frame_width
            and self.crop_bottom == self.frame_height
        )

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        """Map a full-frame point into output pixels."""
        return (
            (x - self.crop_left) * self.frame_width / self.crop_width,
            (y - self.crop_top) * self.frame_height / self.crop_height,
        )

    def transform_box(
        self,
        detection: Detection,
    ) -> tuple[float, float, float, float] | None:
        """Map and clip a full-frame box, or return None when fully outside."""
        left = max(float(self.crop_left), detection.left)
        top = max(float(self.crop_top), detection.top)
        right = min(float(self.crop_right), detection.right)
        bottom = min(float(self.crop_bottom), detection.bottom)
        if right <= left or bottom <= top:
            return None
        transformed_left, transformed_top = self.transform_point(left, top)
        transformed_right, transformed_bottom = self.transform_point(right, bottom)
        return (
            max(0.0, min(float(self.frame_width), transformed_left)),
            max(0.0, min(float(self.frame_height), transformed_top)),
            max(0.0, min(float(self.frame_width), transformed_right)),
            max(0.0, min(float(self.frame_height), transformed_bottom)),
        )


class DigitalZoomController:
    """Smooth a target-following visualization crop without affecting control."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_zoom: float = 2.0,
        smoothing: float = 0.25,
        lost_smoothing: float = 0.15,
        target_fill: float = 0.5,
    ) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("digital zoom enabled must be bool")
        for name, value in (
            ("max_zoom", max_zoom),
            ("smoothing", smoothing),
            ("lost_smoothing", lost_smoothing),
            ("target_fill", target_fill),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"digital zoom {name} must be finite")
            if not math.isfinite(float(value)):
                raise ValueError(f"digital zoom {name} must be finite")
        if max_zoom < 1.0:
            raise ValueError("digital zoom max_zoom must be at least 1.0")
        if not 0.0 < smoothing <= 1.0 or not 0.0 < lost_smoothing <= 1.0:
            raise ValueError("digital zoom smoothing values must be in (0, 1]")
        if not 0.0 < target_fill <= 1.0:
            raise ValueError("digital zoom target_fill must be in (0, 1]")
        self._enabled = enabled
        self._max_zoom = float(max_zoom)
        self._smoothing = float(smoothing)
        self._lost_smoothing = float(lost_smoothing)
        self._target_fill = float(target_fill)
        self._zoom = 1.0
        self._center_x = 0.0
        self._center_y = 0.0
        self._frame_size: tuple[int, int] | None = None

    def update(
        self,
        frame_width: int,
        frame_height: int,
        selected: Detection | None,
    ) -> ZoomTransform:
        """Advance smoothing and return a bounded full-frame-to-display crop."""
        if (
            isinstance(frame_width, bool)
            or isinstance(frame_height, bool)
            or not isinstance(frame_width, int)
            or not isinstance(frame_height, int)
            or frame_width <= 0
            or frame_height <= 0
        ):
            raise ValueError("digital zoom frame dimensions must be positive integers")
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0
        if self._frame_size != (frame_width, frame_height):
            self._frame_size = (frame_width, frame_height)
            self._zoom = 1.0
            self._center_x = frame_center_x
            self._center_y = frame_center_y

        if not self._enabled:
            self._zoom = 1.0
            self._center_x = frame_center_x
            self._center_y = frame_center_y
            return _transform(frame_width, frame_height, self._center_x, self._center_y, 1.0)

        if selected is None:
            desired_zoom = 1.0
            desired_center_x = frame_center_x
            desired_center_y = frame_center_y
            alpha = self._lost_smoothing
        else:
            box_width = max(0.0, min(float(frame_width), selected.right) - max(0.0, selected.left))
            box_height = max(
                0.0,
                min(float(frame_height), selected.bottom) - max(0.0, selected.top),
            )
            fraction = max(box_width / frame_width, box_height / frame_height)
            desired_zoom = (
                self._max_zoom
                if fraction <= 0.0
                else min(self._max_zoom, max(1.0, self._target_fill / fraction))
            )
            desired_center_x, desired_center_y = selected.center
            desired_center_x = min(float(frame_width), max(0.0, desired_center_x))
            desired_center_y = min(float(frame_height), max(0.0, desired_center_y))
            alpha = self._smoothing

        self._zoom += alpha * (desired_zoom - self._zoom)
        self._zoom = min(self._max_zoom, max(1.0, self._zoom))
        self._center_x += alpha * (desired_center_x - self._center_x)
        self._center_y += alpha * (desired_center_y - self._center_y)
        return _transform(
            frame_width,
            frame_height,
            self._center_x,
            self._center_y,
            self._zoom,
        )


def identity_transform(frame_width: int, frame_height: int) -> ZoomTransform:
    """Return an explicit unzoomed transform."""
    return _transform(
        frame_width,
        frame_height,
        frame_width / 2.0,
        frame_height / 2.0,
        1.0,
    )


def _transform(
    frame_width: int,
    frame_height: int,
    center_x: float,
    center_y: float,
    zoom_factor: float,
) -> ZoomTransform:
    crop_width = max(1, min(frame_width, round(frame_width / zoom_factor)))
    crop_height = max(1, min(frame_height, round(frame_height / zoom_factor)))
    left = round(center_x - crop_width / 2.0)
    top = round(center_y - crop_height / 2.0)
    left = min(frame_width - crop_width, max(0, left))
    top = min(frame_height - crop_height, max(0, top))
    return ZoomTransform(
        frame_width=frame_width,
        frame_height=frame_height,
        crop_left=left,
        crop_top=top,
        crop_right=left + crop_width,
        crop_bottom=top + crop_height,
        zoom_factor=zoom_factor,
    )
