from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import marine_ptz.vision_cli as vision_cli
from marine_ptz.config import load_config
from marine_ptz.types import Detection, Frame
from marine_ptz.vision_cli import VisionOptions, _target_classes, run_vision


class FakeImage:
    shape = (100, 100, 3)

    def __init__(self, *, copy_error: Exception | None = None) -> None:
        self.copy_error = copy_error

    def copy(self) -> FakeImage:
        if self.copy_error is not None:
            raise self.copy_error
        return FakeImage()


class FakeSource:
    def __init__(self, frames: list[Frame | None], cv2: Any | None = None) -> None:
        self.frames = list(frames)
        self.cv2 = cv2
        self.closed = False

    def read(self) -> Frame | None:
        return self.frames.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeDetector:
    active_device = "cpu"
    last_inference_ms = 2.5

    def __init__(
        self,
        detections: list[tuple[Detection, ...] | Exception],
    ) -> None:
        self.detections = list(detections)

    def detect(self, frame: Frame) -> tuple[Detection, ...]:
        value = self.detections.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class FakeWriter:
    def __init__(
        self,
        *,
        opened: bool = True,
        write_error: Exception | None = None,
    ) -> None:
        self.frames: list[object] = []
        self.opened = opened
        self.write_error = write_error
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def write(self, image: object) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.frames.append(image)

    def release(self) -> None:
        self.released = True


class FakeCV2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0
    MARKER_CROSS = 0

    def __init__(
        self,
        writer: FakeWriter,
        *,
        wait_key: int = -1,
        writer_error: Exception | None = None,
        annotation_error: Exception | None = None,
    ) -> None:
        self.writer = writer
        self.wait_key = wait_key
        self.writer_error = writer_error
        self.annotation_error = annotation_error
        self.windows_destroyed = False

    def VideoWriter_fourcc(self, *values: str) -> int:
        return 1

    def VideoWriter(self, *args: object) -> FakeWriter:
        if self.writer_error is not None:
            raise self.writer_error
        return self.writer

    def rectangle(self, *args: object) -> None:
        if self.annotation_error is not None:
            raise self.annotation_error

    def putText(self, *args: object) -> None:
        pass

    def drawMarker(self, *args: object) -> None:
        pass

    def line(self, *args: object) -> None:
        pass

    def imshow(self, *args: object) -> None:
        pass

    def waitKey(self, delay: int) -> int:
        return self.wait_key

    def destroyAllWindows(self) -> None:
        self.windows_destroyed = True


def options(
    *,
    output: Path | None = None,
    display: bool = False,
    max_frames: int | None = None,
) -> VisionOptions:
    return VisionOptions(
        source="video.mp4",
        model="fake.pt",
        device="cpu",
        target_classes=("boat",),
        confidence_threshold=0.5,
        iou_threshold=0.45,
        image_size=320,
        max_frames=max_frames,
        display=display,
        output=output,
    )


def frame(sequence: int) -> Frame:
    return Frame(FakeImage(), float(sequence), 100, 100, sequence)


def test_simulated_closed_loop_handles_target_and_lost_target() -> None:
    source = FakeSource([frame(0), frame(1), None])
    detector = FakeDetector(
        [
            (Detection("boat", 0.9, 70, 70, 90, 90),),
            (),
        ]
    )
    lines: list[str] = []
    timestamps = iter((1.0, 2.0))

    processed = run_vision(
        load_config("configs/development.yaml"),
        options(),
        source_factory=lambda value: source,
        detector_factory=lambda *args, **kwargs: detector,
        monotonic=lambda: next(timestamps),
        emit=lines.append,
    )

    assert processed == 2
    assert source.closed is True
    assert "target=boat" in lines[0]
    assert "target=none" in lines[1]
    assert "pan=93.00 tilt=93.00" in lines[0]
    assert "pan=93.00 tilt=93.00" in lines[1]


def test_cleanup_releases_source_writer_and_windows_after_normal_eof(
    tmp_path: Path,
) -> None:
    writer = FakeWriter()
    cv2 = FakeCV2(writer)
    source = FakeSource([frame(0), None], cv2)
    detector = FakeDetector([(Detection("boat", 0.9, 40, 40, 60, 60),)])

    run_vision(
        load_config("configs/development.yaml"),
        options(output=tmp_path / "annotated.mp4", display=True),
        source_factory=lambda value: source,
        detector_factory=lambda *args, **kwargs: detector,
    )

    assert source.closed is True
    assert writer.released is True
    assert len(writer.frames) == 1
    assert cv2.windows_destroyed is True


def test_cleanup_releases_resources_after_detection_error(tmp_path: Path) -> None:
    writer = FakeWriter()
    cv2 = FakeCV2(writer)
    source = FakeSource([frame(0), frame(1)], cv2)
    detector = FakeDetector(
        [
            (Detection("boat", 0.9, 40, 40, 60, 60),),
            RuntimeError("inference failed"),
        ]
    )

    with pytest.raises(RuntimeError, match="inference failed"):
        run_vision(
            load_config("configs/development.yaml"),
            options(output=tmp_path / "annotated.mp4", display=True),
            source_factory=lambda value: source,
            detector_factory=lambda *args, **kwargs: detector,
        )

    assert source.closed is True
    assert writer.released is True
    assert cv2.windows_destroyed is True


def test_q_terminates_display_and_releases_resources() -> None:
    writer = FakeWriter()
    cv2 = FakeCV2(writer, wait_key=ord("q"))
    source = FakeSource([frame(0), frame(1)], cv2)
    detector = FakeDetector([(Detection("boat", 0.9, 40, 40, 60, 60),)])

    processed = run_vision(
        load_config("configs/development.yaml"),
        options(display=True),
        source_factory=lambda value: source,
        detector_factory=lambda *args, **kwargs: detector,
    )

    assert processed == 1
    assert source.closed is True
    assert cv2.windows_destroyed is True


def test_annotation_failure_releases_source_and_windows() -> None:
    cv2 = FakeCV2(FakeWriter(), annotation_error=RuntimeError("annotation failed"))
    source = FakeSource([frame(0)], cv2)
    detector = FakeDetector([(Detection("boat", 0.9, 40, 40, 60, 60),)])

    with pytest.raises(RuntimeError, match="annotation failed"):
        run_vision(
            load_config("configs/development.yaml"),
            options(display=True),
            source_factory=lambda value: source,
            detector_factory=lambda *args, **kwargs: detector,
        )

    assert source.closed is True
    assert cv2.windows_destroyed is True


def test_writer_creation_failure_releases_source_and_windows(tmp_path: Path) -> None:
    cv2 = FakeCV2(FakeWriter(), writer_error=RuntimeError("codec unavailable"))
    source = FakeSource([frame(0)], cv2)
    detector = FakeDetector([(Detection("boat", 0.9, 40, 40, 60, 60),)])

    with pytest.raises(RuntimeError, match="unable to create annotated video"):
        run_vision(
            load_config("configs/development.yaml"),
            options(output=tmp_path / "annotated.mp4", display=True),
            source_factory=lambda value: source,
            detector_factory=lambda *args, **kwargs: detector,
        )

    assert source.closed is True
    assert cv2.windows_destroyed is True


def test_writer_not_open_failure_releases_every_resource(tmp_path: Path) -> None:
    writer = FakeWriter(opened=False)
    cv2 = FakeCV2(writer)
    source = FakeSource([frame(0)], cv2)
    detector = FakeDetector([(Detection("boat", 0.9, 40, 40, 60, 60),)])

    with pytest.raises(RuntimeError, match="unable to open annotated video"):
        run_vision(
            load_config("configs/development.yaml"),
            options(output=tmp_path / "annotated.mp4", display=True),
            source_factory=lambda value: source,
            detector_factory=lambda *args, **kwargs: detector,
        )

    assert writer.released is True
    assert source.closed is True
    assert cv2.windows_destroyed is True


def test_writer_write_failure_releases_writer_source_and_windows(tmp_path: Path) -> None:
    writer = FakeWriter(write_error=RuntimeError("write failed"))
    cv2 = FakeCV2(writer)
    source = FakeSource([frame(0)], cv2)
    detector = FakeDetector([(Detection("boat", 0.9, 40, 40, 60, 60),)])

    with pytest.raises(RuntimeError, match="write failed"):
        run_vision(
            load_config("configs/development.yaml"),
            options(output=tmp_path / "annotated.mp4", display=True),
            source_factory=lambda value: source,
            detector_factory=lambda *args, **kwargs: detector,
        )

    assert writer.released is True
    assert source.closed is True
    assert cv2.windows_destroyed is True


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["boat", "ship"], ("boat", "ship")),
        (["boat,ship"], ("boat", "ship")),
        (["boat, ship", "tugboat"], ("boat", "ship", "tugboat")),
    ],
)
def test_repeatable_and_comma_separated_target_classes(
    values: list[str],
    expected: tuple[str, ...],
) -> None:
    assert _target_classes(values, ("default",)) == expected


def test_cli_values_override_configuration_and_defaults_fill_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[VisionOptions] = []
    monkeypatch.setattr(
        vision_cli,
        "run_vision",
        lambda config, selected: captured.append(selected) or 0,
    )

    result = vision_cli.main(
        [
            "--config",
            "configs/development.yaml",
            "--source",
            "camera:2",
            "--model",
            "custom.pt",
            "--device",
            "cpu",
            "--target-class",
            "boat,ship",
            "--target-class",
            "tugboat",
            "--confidence",
            "0.7",
            "--iou",
            "0.3",
            "--image-size",
            "320",
            "--max-frames",
            "5",
        ]
    )

    assert result == 0
    assert captured == [
        VisionOptions(
            source=2,
            model="custom.pt",
            device="cpu",
            target_classes=("boat", "ship", "tugboat"),
            confidence_threshold=0.7,
            iou_threshold=0.3,
            image_size=320,
            max_frames=5,
            display=False,
            output=None,
        )
    ]


def test_cli_uses_config_detection_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[VisionOptions] = []
    monkeypatch.setattr(
        vision_cli,
        "run_vision",
        lambda config, selected: captured.append(selected) or 0,
    )

    assert (
        vision_cli.main(
            ["--config", "configs/development.yaml", "--source", "path:video.mp4"]
        )
        == 0
    )

    selected = captured[0]
    assert selected.source == "video.mp4"
    assert selected.model == "yolo11n.pt"
    assert selected.device == "auto"
    assert selected.target_classes == ("boat",)
    assert selected.confidence_threshold == 0.6
    assert selected.iou_threshold == 0.45
    assert selected.image_size == 640


@pytest.mark.parametrize("max_frames", [0, -1])
def test_max_frames_validation_prevents_resource_construction(max_frames: int) -> None:
    constructed = False

    def source_factory(value: object) -> FakeSource:
        nonlocal constructed
        constructed = True
        return FakeSource([])

    with pytest.raises(ValueError, match="max_frames"):
        run_vision(
            load_config("configs/development.yaml"),
            options(max_frames=max_frames),
            source_factory=source_factory,
        )

    assert constructed is False


def test_hardware_config_still_constructs_only_simulated_actuator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[object] = []
    real_actuator = vision_cli.SimulatedPTZActuator

    def simulated_factory(*args: object, **kwargs: object) -> object:
        actuator = real_actuator(*args, **kwargs)
        constructed.append(actuator)
        return actuator

    monkeypatch.setattr(vision_cli, "SimulatedPTZActuator", simulated_factory)
    source = FakeSource([frame(0), None])
    detector = FakeDetector([()])

    run_vision(
        load_config("configs/hardware.yaml"),
        options(),
        source_factory=lambda value: source,
        detector_factory=lambda *args, **kwargs: detector,
    )

    assert len(constructed) == 1
