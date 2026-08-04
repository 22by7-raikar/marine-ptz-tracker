from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tools.extract_frames import ExtractionOptions, extract_frames
from tools.record_dataset import RecordingOptions, record_clip


class FakeImage:
    shape = (1080, 1920, 3)

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def tobytes(self) -> bytes:
        return self.payload


class FakeCapture:
    def __init__(self, frames: list[FakeImage | BaseException], *, fps: float = 30.0) -> None:
        self.frames = list(frames)
        self.fps = fps
        self.index = 0
        self.released = False
        self.requests: list[tuple[int, float | int]] = []

    def isOpened(self) -> bool:
        return True

    def set(self, property_id: int, value: float | int) -> bool:
        self.requests.append((property_id, value))
        return True

    def read(self) -> tuple[bool, FakeImage | None]:
        if self.index >= len(self.frames):
            return False, None
        value = self.frames[self.index]
        self.index += 1
        if isinstance(value, BaseException):
            raise value
        return True, value

    def get(self, property_id: int) -> float:
        if property_id == FakeCV2.CAP_PROP_FPS:
            return self.fps
        if property_id == FakeCV2.CAP_PROP_POS_MSEC:
            return max(0, self.index - 1) * 1000.0 / self.fps
        return 0.0

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self) -> None:
        self.frames: list[FakeImage] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, image: FakeImage) -> None:
        self.frames.append(image)

    def release(self) -> None:
        self.released = True


class FakeCV2:
    CAP_V4L2 = 200
    CAP_PROP_FOURCC = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FPS = 4
    CAP_PROP_POS_MSEC = 5
    IMWRITE_JPEG_QUALITY = 6

    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture
        self.writer = FakeWriter()
        self.capture_arguments: tuple[Any, ...] | None = None

    def VideoCapture(self, *args: object) -> FakeCapture:
        self.capture_arguments = args
        return self.capture

    def VideoWriter_fourcc(self, *_values: str) -> int:
        return 1234

    def VideoWriter(self, *_args: object) -> FakeWriter:
        return self.writer


def test_recording_requests_mjpeg_profile_and_releases_resources(tmp_path: Path) -> None:
    capture = FakeCapture([FakeImage(b"frame")])
    cv2 = FakeCV2(capture)
    clock = iter((0.0, 0.0, 1.0))
    output: list[str] = []

    frames = record_clip(
        RecordingOptions(
            camera="/dev/v4l/by-id/test-camera",
            output=tmp_path / "session.avi",
            duration_s=0.5,
            width=1920,
            height=1080,
            fps=30.0,
        ),
        cv2_module=cv2,
        monotonic=lambda: next(clock),
        emit=output.append,
    )

    assert frames == 1
    assert cv2.capture_arguments == ("/dev/v4l/by-id/test-camera", cv2.CAP_V4L2)
    assert capture.requests == [
        (cv2.CAP_PROP_FOURCC, 1234),
        (cv2.CAP_PROP_FRAME_WIDTH, 1920.0),
        (cv2.CAP_PROP_FRAME_HEIGHT, 1080.0),
        (cv2.CAP_PROP_FPS, 30.0),
    ]
    assert capture.released is True
    assert cv2.writer.released is True
    assert "1920x1080@30 MJPEG" in output[0]


def test_recording_ctrl_c_releases_capture_without_serial_or_writer(tmp_path: Path) -> None:
    capture = FakeCapture([KeyboardInterrupt()])
    cv2 = FakeCV2(capture)

    with pytest.raises(KeyboardInterrupt):
        record_clip(
            RecordingOptions(
                camera="/dev/v4l/by-id/test-camera",
                output=tmp_path / "session.avi",
                duration_s=1.0,
                width=1920,
                height=1080,
                fps=30.0,
            ),
            cv2_module=cv2,
            monotonic=lambda: 0.0,
        )

    assert capture.released is True
    assert cv2.writer.frames == []


def test_extraction_dry_run_removes_duplicates_and_near_adjacent_frames(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.avi"
    source.write_bytes(b"fake clip")
    capture = FakeCapture(
        [
            FakeImage(b"same"),
            FakeImage(b"neighbor"),
            FakeImage(b"same"),
            FakeImage(b"new"),
        ]
    )
    cv2 = FakeCV2(capture)
    output_dir = tmp_path / "frames"
    manifest = tmp_path / "manifest.json"

    records = extract_frames(
        ExtractionOptions(
            source=source,
            output_dir=output_dir,
            manifest=manifest,
            session="session-01",
            sample_fps=30.0,
            minimum_frame_gap=2,
            dry_run=True,
        ),
        cv2_module=cv2,
    )

    assert [record.frame_index for record in records] == [0, 3]
    assert all(record.image_file.startswith("session-01__clip__") for record in records)
    assert output_dir.exists() is False
    assert manifest.exists() is False
    assert capture.released is True
