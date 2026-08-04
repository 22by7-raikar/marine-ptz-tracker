from __future__ import annotations

import pytest

import marine_ptz.opencv_source as opencv_source
from marine_ptz.interfaces import CameraSource
from marine_ptz.opencv_source import (
    CameraReadError,
    ImageDimensionError,
    MediaReadError,
    OpenCVDependencyError,
    OpenCVSource,
    SourceOpenError,
    parse_source_argument,
)


class FakeImage:
    def __init__(self, shape: tuple[object, ...]) -> None:
        self.shape = shape


class FakeCapture:
    def __init__(
        self,
        reads: list[tuple[bool, object | None] | Exception],
        *,
        opened: bool = True,
        frame_count: float | None = None,
        position: float | None = None,
        fps: float | None = 30.0,
    ) -> None:
        self.reads = list(reads)
        self.opened = opened
        self.frame_count = frame_count
        self.position = position
        self.fps = fps
        self.successful_reads = 0
        self.release_count = 0

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, object | None]:
        result = self.reads.pop(0)
        if isinstance(result, Exception):
            raise result
        if result[0]:
            self.successful_reads += 1
        return result

    def get(self, property_id: int) -> float:
        if property_id == FakeCV2.CAP_PROP_FRAME_COUNT:
            return self.frame_count if self.frame_count is not None else 0.0
        if property_id == FakeCV2.CAP_PROP_POS_FRAMES:
            return self.position if self.position is not None else float(self.successful_reads)
        if property_id == FakeCV2.CAP_PROP_FPS:
            return self.fps if self.fps is not None else 0.0
        return 0.0

    def release(self) -> None:
        self.release_count += 1


class FakeCV2:
    CAP_PROP_FRAME_COUNT = 1
    CAP_PROP_POS_FRAMES = 2
    CAP_PROP_FPS = 3


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0),
        ("camera:2", 2),
        ("video.mp4", "video.mp4"),
        ("path:0", "0"),
    ],
)
def test_parse_source_argument_distinguishes_camera_and_path(
    raw: str,
    expected: int | str,
) -> None:
    assert parse_source_argument(raw) == expected


def test_integer_source_is_camera_but_string_zero_is_path() -> None:
    received: list[int | str] = []

    def factory(source: int | str) -> FakeCapture:
        received.append(source)
        return FakeCapture([])

    camera = OpenCVSource(0, cv2_module=FakeCV2(), capture_factory=factory)
    path = OpenCVSource("0", cv2_module=FakeCV2(), capture_factory=factory)

    assert camera.is_live is True
    assert path.is_live is False
    assert received == [0, "0"]


def test_finite_video_exposes_validated_declared_fps() -> None:
    source = OpenCVSource(
        "video.mp4",
        cv2_module=FakeCV2(),
        capture_factory=lambda _value: FakeCapture([], fps=29.97),
    )

    assert source.fps == pytest.approx(29.97)


@pytest.mark.parametrize("fps", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_finite_video_rejects_missing_or_invalid_fps(fps: float | None) -> None:
    capture = FakeCapture([], fps=fps)

    with pytest.raises(SourceOpenError, match="valid positive FPS"):
        OpenCVSource(
            "video.mp4",
            cv2_module=FakeCV2(),
            capture_factory=lambda _value: capture,
        )

    assert capture.release_count == 1


def test_live_camera_does_not_require_declared_fps() -> None:
    source = OpenCVSource(
        0,
        cv2_module=FakeCV2(),
        capture_factory=lambda _value: FakeCapture([], fps=None),
    )

    assert source.fps is None


def test_open_failure_releases_capture() -> None:
    capture = FakeCapture([], opened=False)

    with pytest.raises(SourceOpenError, match="unable to open"):
        OpenCVSource(
            "missing.mp4",
            cv2_module=FakeCV2(),
            capture_factory=lambda source: capture,
        )

    assert capture.release_count == 1


def test_open_inspection_failure_releases_capture() -> None:
    capture = FakeCapture([])

    def fail_open_check() -> bool:
        raise RuntimeError("backend failure")

    capture.isOpened = fail_open_check  # type: ignore[method-assign]

    with pytest.raises(SourceOpenError, match="unable to inspect"):
        OpenCVSource(
            "broken.mp4",
            cv2_module=FakeCV2(),
            capture_factory=lambda source: capture,
        )

    assert capture.release_count == 1


def test_close_is_idempotent() -> None:
    capture = FakeCapture([])
    source = OpenCVSource(
        "video.mp4",
        cv2_module=FakeCV2(),
        capture_factory=lambda value: capture,
    )

    source.close()
    source.close()

    assert capture.release_count == 1


def test_successful_reads_sequence_dimensions_timestamps_and_cleanup() -> None:
    capture = FakeCapture(
        [
            (True, FakeImage((48, 64, 3))),
            (True, FakeImage((24, 32))),
        ]
    )
    timestamps = iter((10.0, 11.5))

    with OpenCVSource(
        "video.mp4",
        clock=lambda: next(timestamps),
        cv2_module=FakeCV2(),
        capture_factory=lambda source: capture,
    ) as source:
        first = source.read()
        second = source.read()
        assert isinstance(source, CameraSource)

    assert first is not None
    assert second is not None
    assert (first.sequence, first.timestamp_s, first.width, first.height) == (
        0,
        10.0,
        64,
        48,
    )
    assert (second.sequence, second.timestamp_s, second.width, second.height) == (
        1,
        11.5,
        32,
        24,
    )
    assert capture.release_count == 1


def test_verified_finite_media_eof_returns_none_and_releases() -> None:
    capture = FakeCapture(
        [(True, FakeImage((10, 20, 3))), (False, None)],
        frame_count=1.0,
    )
    source = OpenCVSource(
        "video.mp4",
        cv2_module=FakeCV2(),
        capture_factory=lambda value: capture,
    )

    assert source.read() is not None
    assert source.read() is None
    assert capture.release_count == 1


def test_live_camera_read_failure_raises_and_releases() -> None:
    capture = FakeCapture([(False, None)])
    source = OpenCVSource(
        0,
        cv2_module=FakeCV2(),
        capture_factory=lambda value: capture,
    )

    with pytest.raises(CameraReadError, match="failed to read"):
        source.read()

    assert capture.release_count == 1


def test_midstream_media_decode_failure_raises_and_releases() -> None:
    capture = FakeCapture(
        [(True, FakeImage((10, 20, 3))), (False, None)],
        frame_count=3.0,
    )
    source = OpenCVSource(
        "video.mp4",
        cv2_module=FakeCV2(),
        capture_factory=lambda value: capture,
    )

    assert source.read() is not None
    with pytest.raises(MediaReadError, match="verifiable end-of-file"):
        source.read()

    assert capture.release_count == 1


def test_unknown_media_length_uses_explicit_error_and_releases() -> None:
    capture = FakeCapture([(False, None)])
    source = OpenCVSource(
        "stream.mkv",
        cv2_module=FakeCV2(),
        capture_factory=lambda value: capture,
    )

    with pytest.raises(MediaReadError, match="verifiable end-of-file"):
        source.read()

    assert capture.release_count == 1


def test_image_source_completes_normally_after_one_frame_and_releases() -> None:
    capture = FakeCapture(
        [(True, FakeImage((10, 20, 4))), (False, None)],
    )
    source = OpenCVSource(
        "still.png",
        cv2_module=FakeCV2(),
        capture_factory=lambda value: capture,
    )

    assert source.read() is not None
    assert source.read() is None
    assert capture.release_count == 1


@pytest.mark.parametrize("source_value", [0, "video.mp4"])
def test_capture_read_exception_is_wrapped_and_released(source_value: int | str) -> None:
    capture = FakeCapture([RuntimeError("decoder exploded")], frame_count=2.0)
    source = OpenCVSource(
        source_value,
        cv2_module=FakeCV2(),
        capture_factory=lambda value: capture,
    )

    expected = CameraReadError if source_value == 0 else MediaReadError
    with pytest.raises(expected, match="raised while reading"):
        source.read()

    assert capture.release_count == 1


@pytest.mark.parametrize("source_value", [0, "video.mp4"])
def test_invalid_read_result_is_rejected_and_released(source_value: int | str) -> None:
    capture = FakeCapture([], frame_count=1.0)
    capture.read = lambda: object()  # type: ignore[method-assign]
    source = OpenCVSource(
        source_value,
        cv2_module=FakeCV2(),
        capture_factory=lambda value: capture,
    )

    expected = CameraReadError if source_value == 0 else MediaReadError
    with pytest.raises(expected, match="invalid read result"):
        source.read()

    assert capture.release_count == 1


@pytest.mark.parametrize("source_value", [0, "video.mp4"])
def test_missing_image_in_successful_read_is_rejected_and_released(
    source_value: int | str,
) -> None:
    capture = FakeCapture([(True, None)], frame_count=1.0)
    source = OpenCVSource(
        source_value,
        cv2_module=FakeCV2(),
        capture_factory=lambda value: capture,
    )

    expected = CameraReadError if source_value == 0 else MediaReadError
    with pytest.raises(expected):
        source.read()

    assert capture.release_count == 1


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (10,),
        (0, 10, 3),
        (10, 0, 3),
        (True, 10, 3),
        ("10", 10, 3),
        (10, 10, 0),
        (10, 10, 2),
        (10, 10, 5),
        (10, 10, "3"),
    ],
)
def test_invalid_image_dimensions_are_rejected(shape: tuple[object, ...]) -> None:
    capture = FakeCapture([(True, FakeImage(shape))])
    source = OpenCVSource(
        "bad.mp4",
        cv2_module=FakeCV2(),
        capture_factory=lambda value: capture,
    )

    with pytest.raises(ImageDimensionError, match="shape|dimensions|channels"):
        source.read()

    assert capture.release_count == 1


@pytest.mark.parametrize("shape", [(10, 20), (10, 20, 1), (10, 20, 3), (10, 20, 4)])
def test_supported_grayscale_and_color_shapes_are_accepted(
    shape: tuple[int, ...],
) -> None:
    source = OpenCVSource(
        "valid.mp4",
        cv2_module=FakeCV2(),
        capture_factory=lambda value: FakeCapture([(True, FakeImage(shape))]),
    )

    frame = source.read()

    assert frame is not None
    assert (frame.width, frame.height) == (20, 10)


def test_missing_opencv_has_clear_lazy_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(opencv_source, "import_module", missing)

    with pytest.raises(OpenCVDependencyError, match="OpenCV is unavailable"):
        OpenCVSource("video.mp4")
