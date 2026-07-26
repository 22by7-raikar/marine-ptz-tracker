"""OpenCV-backed camera and finite-media frame source."""

from __future__ import annotations

import time
from collections.abc import Callable
from importlib import import_module
from math import isfinite
from pathlib import Path
from types import ModuleType
from typing import Any

from .types import Frame


class OpenCVSourceError(RuntimeError):
    """Base error for actionable OpenCV source failures."""


class OpenCVDependencyError(OpenCVSourceError):
    """Raised when the optional OpenCV dependency is unavailable."""


class SourceOpenError(OpenCVSourceError):
    """Raised when a camera or media source cannot be opened."""


class CameraReadError(OpenCVSourceError):
    """Raised when a live camera fails to return a frame."""


class MediaReadError(OpenCVSourceError):
    """Raised when finite media fails before a verifiable normal end."""


class ImageDimensionError(OpenCVSourceError):
    """Raised when OpenCV returns an invalid image shape."""


def parse_source_argument(value: str) -> int | str:
    """Parse a CLI source, using ``path:`` to disambiguate numeric paths.

    ``0`` and ``camera:0`` select camera index 0. ``path:0`` selects a
    filesystem entry literally named ``0``.
    """
    if value.startswith("path:"):
        path = value[5:]
        if not path:
            raise ValueError("path: source must include a filesystem path")
        return path
    if value.startswith("camera:"):
        value = value[7:]
        if not value.isdecimal():
            raise ValueError("camera source must use a non-negative integer index")
        return int(value)
    if value.isdecimal():
        return int(value)
    return value


class OpenCVSource:
    """Read frames from a camera index or finite media through OpenCV."""

    def __init__(
        self,
        source: int | str | Path,
        *,
        clock: Callable[[], float] = time.monotonic,
        cv2_module: ModuleType | Any | None = None,
        capture_factory: Callable[[int | str], Any] | None = None,
    ) -> None:
        if isinstance(source, bool):
            raise ValueError("camera index must be an integer, not bool")
        if isinstance(source, int) and source < 0:
            raise ValueError("camera index must be non-negative")
        self._is_live = isinstance(source, int)
        self._source = source if isinstance(source, int) else str(source)
        self._clock = clock
        self._cv2 = cv2_module if cv2_module is not None else _load_cv2()
        factory = capture_factory or self._cv2.VideoCapture
        try:
            self._capture = factory(self._source)
        except Exception as exc:
            raise SourceOpenError(
                f"unable to create OpenCV capture for source {self._source!r}: {exc}"
            ) from exc
        self._sequence = 0
        self._closed = False
        self._successful_reads = 0
        self._is_image = not self._is_live and _is_image_path(self._source)
        try:
            opened = bool(self._capture.isOpened())
        except Exception as exc:
            self._capture.release()
            self._closed = True
            raise SourceOpenError(
                f"unable to inspect OpenCV source {self._source!r}: {exc}"
            ) from exc
        if not opened:
            self.close()
            kind = "camera" if self._is_live else "media"
            raise SourceOpenError(f"unable to open {kind} source {self._source!r}")
        self._frame_count = (
            None if self._is_live else self._capture_metric("CAP_PROP_FRAME_COUNT")
        )
        if self._frame_count is not None and self._frame_count <= 0:
            self._frame_count = None

    @property
    def cv2(self) -> Any:
        """Return the lazily loaded OpenCV module for rendering helpers."""
        return self._cv2

    @property
    def is_live(self) -> bool:
        return self._is_live

    @property
    def is_image(self) -> bool:
        """Return whether the finite-media path has a recognized image suffix."""
        return self._is_image

    @property
    def total_frames(self) -> int | None:
        """Return finite-media frame count when OpenCV provides a usable value."""
        if self._frame_count is None or not self._frame_count.is_integer():
            return None
        return int(self._frame_count)

    def read(self) -> Frame | None:
        if self._closed:
            return None
        try:
            result = self._capture.read()
        except Exception as exc:
            self.close()
            error_type = CameraReadError if self._is_live else MediaReadError
            raise error_type(
                f"{self._source_kind} source {self._source!r} raised while reading: {exc}"
            ) from exc
        if not isinstance(result, tuple) or len(result) != 2:
            self.close()
            error_type = CameraReadError if self._is_live else MediaReadError
            raise error_type(
                f"{self._source_kind} source {self._source!r} returned an invalid read result"
            )
        ok, image = result
        if not isinstance(ok, bool):
            self.close()
            error_type = CameraReadError if self._is_live else MediaReadError
            raise error_type(
                f"{self._source_kind} source {self._source!r} returned an invalid read status"
            )
        if not ok:
            if self._is_live:
                self.close()
                raise CameraReadError(f"camera source {self._source!r} failed to read a frame")
            return self._finish_finite_read()
        if image is None:
            self.close()
            error_type = CameraReadError if self._is_live else MediaReadError
            raise error_type(
                f"{self._source_kind} source {self._source!r} returned no image for a successful read"
            )
        try:
            height, width = _image_dimensions(image)
        except ImageDimensionError:
            self.close()
            raise
        frame = Frame(
            image=image,
            timestamp_s=self._clock(),
            width=width,
            height=height,
            sequence=self._sequence,
        )
        self._sequence += 1
        self._successful_reads += 1
        return frame

    def close(self) -> None:
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> OpenCVSource:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def _source_kind(self) -> str:
        return "camera" if self._is_live else "media"

    def _capture_metric(self, constant_name: str) -> float | None:
        constant = getattr(self._cv2, constant_name, None)
        get = getattr(self._capture, "get", None)
        if constant is None or not callable(get):
            return None
        try:
            value = get(constant)
            if isinstance(value, bool):
                return None
            number = float(value)
        except Exception:
            return None
        return number if isfinite(number) else None

    def _finish_finite_read(self) -> None:
        position = self._capture_metric("CAP_PROP_POS_FRAMES")
        verified_eof = False
        if self._frame_count is not None:
            verified_eof = (
                self._successful_reads >= self._frame_count
                or (
                    position is not None
                    and position >= max(0.0, self._frame_count - 0.5)
                )
            )
        elif self._is_image and self._successful_reads == 1:
            verified_eof = True
        self.close()
        if verified_eof:
            return None
        raise MediaReadError(
            f"media source {self._source!r} failed before a verifiable end-of-file"
        )


def _load_cv2() -> ModuleType:
    try:
        return import_module("cv2")
    except (ImportError, ModuleNotFoundError) as exc:
        raise OpenCVDependencyError(
            "OpenCV is unavailable; install the constrained vision dependencies"
        ) from exc


def _image_dimensions(image: object) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if not isinstance(shape, tuple) or len(shape) not in {2, 3}:
        raise ImageDimensionError("OpenCV frame must have a 2D or 3D image shape")
    height, width = shape[:2]
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise ImageDimensionError("OpenCV frame dimensions must be positive integers")
    if len(shape) == 3:
        channels = shape[2]
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels not in {1, 3, 4}
        ):
            raise ImageDimensionError(
                "OpenCV frame channels must be one of 1, 3, or 4"
            )
    return height, width


def _is_image_path(source: int | str) -> bool:
    if isinstance(source, int):
        return False
    return Path(source).suffix.casefold() in {
        ".bmp",
        ".dib",
        ".jpeg",
        ".jpg",
        ".jpe",
        ".jp2",
        ".png",
        ".webp",
        ".avif",
        ".pbm",
        ".pgm",
        ".ppm",
        ".pxm",
        ".pnm",
        ".pfm",
        ".sr",
        ".ras",
        ".tiff",
        ".tif",
        ".exr",
        ".hdr",
        ".pic",
    }
