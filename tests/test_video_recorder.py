from __future__ import annotations

import pytest

from marine_ptz.video_recorder import WallClockVideoRecorder


class FakeWriter:
    def __init__(self, *, fail_on_write: int | None = None) -> None:
        self.frames: list[object] = []
        self.fail_on_write = fail_on_write
        self.release_count = 0

    def write(self, frame: object) -> None:
        if self.fail_on_write == len(self.frames) + 1:
            raise RuntimeError("encoder write failed")
        self.frames.append(frame)

    def release(self) -> None:
        self.release_count += 1


def test_slow_processed_frames_are_held_on_the_fixed_fps_timeline() -> None:
    writer = FakeWriter()
    recorder = WallClockVideoRecorder(writer, 30.0)
    frames = [object(), object(), object()]

    for frame, timestamp in zip(frames, (0.0, 0.106, 0.212), strict=True):
        recorder.write(frame, timestamp_s=timestamp)

    assert writer.frames == [
        frames[0],
        frames[0],
        frames[0],
        frames[1],
        frames[1],
        frames[1],
        frames[2],
    ]
    assert recorder.submitted_frame_count == 3
    assert recorder.encoded_frame_count == 7
    assert recorder.duplicated_frame_count == 4
    assert abs(recorder.encoded_frame_count / recorder.encoded_fps - 0.212) <= 1 / 30


def test_long_recording_duration_matches_wall_clock_within_one_frame() -> None:
    writer = FakeWriter()
    recorder = WallClockVideoRecorder(writer, 30.0)

    recorder.write("first", timestamp_s=100.0)
    recorder.write("last", timestamp_s=187.8)

    encoded_duration = recorder.encoded_frame_count / recorder.encoded_fps
    assert encoded_duration == pytest.approx(2635 / 30)
    assert abs(encoded_duration - 87.8) <= 1 / 30
    assert writer.frames[-1] == "last"


def test_processing_faster_than_encoded_fps_drops_intermediate_output_frames() -> None:
    writer = FakeWriter()
    recorder = WallClockVideoRecorder(writer, 30.0)

    for frame, timestamp in zip("ABCD", (0.0, 0.01, 0.02, 0.04), strict=True):
        recorder.write(frame, timestamp_s=timestamp)

    assert writer.frames == ["A", "D"]
    assert recorder.submitted_frame_count == 4
    assert recorder.encoded_frame_count == 2
    assert recorder.duplicated_frame_count == 0


def test_injected_monotonic_clock_drives_timestamps() -> None:
    writer = FakeWriter()
    timestamps = iter((10.0, 10.1))
    recorder = WallClockVideoRecorder(writer, 10.0, monotonic=lambda: next(timestamps))

    recorder.write("first")
    recorder.write("second")

    assert writer.frames == ["first", "second"]


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_timestamps_are_rejected(timestamp: float) -> None:
    recorder = WallClockVideoRecorder(FakeWriter(), 30.0)

    with pytest.raises(RuntimeError, match="finite"):
        recorder.write(object(), timestamp_s=timestamp)


def test_backwards_timestamp_is_rejected_without_writing() -> None:
    writer = FakeWriter()
    recorder = WallClockVideoRecorder(writer, 30.0)
    recorder.write("first", timestamp_s=2.0)

    with pytest.raises(RuntimeError, match="backwards"):
        recorder.write("second", timestamp_s=1.0)

    assert writer.frames == ["first"]


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_encoded_fps_is_rejected(fps: float) -> None:
    with pytest.raises(ValueError, match="FPS"):
        WallClockVideoRecorder(FakeWriter(), fps)


def test_writer_failure_propagates_and_release_remains_idempotent() -> None:
    writer = FakeWriter(fail_on_write=2)
    recorder = WallClockVideoRecorder(writer, 30.0)
    recorder.write("first", timestamp_s=0.0)

    with pytest.raises(RuntimeError, match="encoder write failed"):
        recorder.write("second", timestamp_s=0.1)

    recorder.release()
    recorder.release()
    assert writer.release_count == 1
    assert recorder.encoded_frame_count == 1


def test_write_after_release_is_rejected() -> None:
    recorder = WallClockVideoRecorder(FakeWriter(), 30.0)
    recorder.release()

    with pytest.raises(RuntimeError, match="after.*release"):
        recorder.write(object(), timestamp_s=0.0)
