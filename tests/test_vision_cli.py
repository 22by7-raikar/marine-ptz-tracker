from __future__ import annotations

import signal
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import marine_ptz.vision_cli as vision_cli
from marine_ptz.cancellation import OperationCancelled
from marine_ptz.config import load_config
from marine_ptz.evaluation import ClosedLoopMetricsCollector
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

    def __getitem__(self, key: object) -> FakeImage:
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
        fail_on_write_number: int | None = None,
    ) -> None:
        self.frames: list[object] = []
        self.opened = opened
        self.write_error = write_error
        self.fail_on_write_number = fail_on_write_number
        self.write_count = 0
        self.released = False
        self.release_count = 0

    def isOpened(self) -> bool:
        return self.opened

    def write(self, image: object) -> None:
        self.write_count += 1
        if self.write_error is not None and (
            self.fail_on_write_number is None or self.write_count == self.fail_on_write_number
        ):
            raise self.write_error
        self.frames.append(image)

    def release(self) -> None:
        self.release_count += 1
        self.released = True


class FakeCV2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0
    MARKER_CROSS = 0
    INTER_LINEAR = 1

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
        self.annotation_text: list[str] = []
        self.shown_frames: list[object] = []
        self.resize_calls: list[tuple[object, tuple[int, int]]] = []

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
        self.annotation_text.append(str(args[1]))

    def drawMarker(self, *args: object) -> None:
        pass

    def line(self, *args: object) -> None:
        pass

    def resize(
        self,
        image: object,
        size: tuple[int, int],
        *,
        interpolation: int,
    ) -> FakeImage:
        self.resize_calls.append((image, size))
        return FakeImage()

    def imshow(self, _name: str, image: object) -> None:
        self.shown_frames.append(image)

    def waitKey(self, delay: int) -> int:
        return self.wait_key

    def destroyAllWindows(self) -> None:
        self.windows_destroyed = True


def options(
    *,
    output: Path | None = None,
    display: bool = False,
    max_frames: int | None = None,
    digital_zoom: bool = False,
    max_digital_zoom: float = 2.0,
    output_fps: float | None = None,
    tracker_backend: str = "none",
    tracker_config: Path | None = None,
    target_classes: tuple[str, ...] = ("boat",),
) -> VisionOptions:
    return VisionOptions(
        source="video.mp4",
        model="fake.pt",
        device="cpu",
        target_classes=target_classes,
        confidence_threshold=0.5,
        iou_threshold=0.45,
        image_size=320,
        max_frames=max_frames,
        display=display,
        output=output,
        digital_zoom=digital_zoom,
        max_digital_zoom=max_digital_zoom,
        output_fps=output_fps,
        tracker_backend=tracker_backend,
        tracker_config=tracker_config,
        tracker_input_confidence=0.20,
        tracker_min_confirmation_hits=2,
        tracker_max_unsupported_age_s=0.30,
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
    assert "logical_pan=93.00 logical_tilt=93.00" in lines[0]
    assert "logical_pan=93.00 logical_tilt=93.00" in lines[1]
    assert "physical_pan=n/a physical_tilt=n/a" in lines[0]


def test_single_runtime_emits_passive_observation_and_control_metrics() -> None:
    source = FakeSource([frame(0), None])
    selected = Detection("boat", 0.9, 70, 40, 90, 60)
    collector = ClosedLoopMetricsCollector()

    processed = run_vision(
        load_config("configs/development.yaml"),
        options(),
        source_factory=lambda _value: source,
        detector_factory=lambda *_args, **_kwargs: FakeDetector([(selected,)]),
        evaluation_observer=collector,
        emit=lambda _line: None,
    )

    assert processed == 1
    assert collector.observations[0].target_present is True
    assert collector.controls[0].target_supplied is True
    assert collector.controls[0].command is not None
    assert collector.controls[0].result_age_ms is None


def test_botsort_miss_uses_lost_target_hold_then_same_track_resumes() -> None:
    source = FakeSource([frame(0), frame(1), frame(2), frame(3), None])
    observation = Detection("marine_target", 0.90, 70, 40, 90, 60, 7)
    detector = FakeDetector([(observation,), (observation,), (), (observation,)])
    lines: list[str] = []

    processed = run_vision(
        load_config("configs/development.yaml"),
        options(
            tracker_backend="botsort",
            tracker_config=Path("configs/trackers/general_botsort.yaml"),
            target_classes=("marine_target",),
        ),
        source_factory=lambda _value: source,
        detector_factory=lambda *_args, **_kwargs: detector,
        emit=lines.append,
    )

    assert processed == 4
    assert "state=acquiring" in lines[0]
    assert "state=locked" in lines[1]
    assert "logical_pan=93.00" in lines[1]
    assert "state=occluded" in lines[2]
    assert "target=none" in lines[2]
    assert "logical_pan=93.00" in lines[2]
    assert "state=locked" in lines[3]
    assert "logical_pan=96.00" in lines[3]


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


def test_zoomed_writer_and_display_receive_the_same_rendered_view(tmp_path: Path) -> None:
    writer = FakeWriter()
    cv2 = FakeCV2(writer)
    source = FakeSource([frame(0), None], cv2)
    detector = FakeDetector([(Detection("boat", 0.9, 70, 40, 90, 60),)])

    run_vision(
        load_config("configs/development.yaml"),
        options(
            output=tmp_path / "zoomed.mp4",
            display=True,
            digital_zoom=True,
            max_digital_zoom=2.0,
        ),
        source_factory=lambda value: source,
        detector_factory=lambda *args, **kwargs: detector,
    )

    assert len(cv2.resize_calls) == 1
    assert len(writer.frames) == 1
    assert cv2.shown_frames == writer.frames
    assert any(text.startswith("digital_zoom=1.25x") for text in cv2.annotation_text)


def test_tracker_none_passes_the_current_selected_detection_to_digital_zoom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[Detection | None] = []
    original_zoom_controller = vision_cli.DigitalZoomController

    class RecordingZoomController(original_zoom_controller):
        def update(
            self,
            frame_width: int,
            frame_height: int,
            selected: Detection | None,
        ) -> object:
            updates.append(selected)
            return super().update(frame_width, frame_height, selected)

    monkeypatch.setattr(vision_cli, "DigitalZoomController", RecordingZoomController)
    writer = FakeWriter()
    cv2 = FakeCV2(writer)
    source = FakeSource([frame(0), None], cv2)
    observed = Detection("boat", 0.9, 70, 40, 90, 60)

    run_vision(
        load_config("configs/development.yaml"),
        options(display=True, digital_zoom=True),
        source_factory=lambda _value: source,
        detector_factory=lambda *_args, **_kwargs: FakeDetector([(observed,)]),
    )

    assert updates == [observed]
    assert len(cv2.resize_calls) == 1


def test_botsort_digital_zoom_uses_only_current_locked_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[Detection | None] = []
    original_zoom_controller = vision_cli.DigitalZoomController

    class RecordingZoomController(original_zoom_controller):
        def update(
            self,
            frame_width: int,
            frame_height: int,
            selected: Detection | None,
        ) -> object:
            updates.append(selected)
            return super().update(frame_width, frame_height, selected)

    monkeypatch.setattr(vision_cli, "DigitalZoomController", RecordingZoomController)
    writer = FakeWriter()
    cv2 = FakeCV2(writer)
    source = FakeSource([frame(0), frame(1), frame(2), None], cv2)
    observed = Detection("marine_target", 0.9, 70, 40, 90, 60, 7)
    prediction_only = Detection("marine_target", 0.9, 0, 0, 20, 20)

    run_vision(
        load_config("configs/development.yaml"),
        options(
            display=True,
            digital_zoom=True,
            tracker_backend="botsort",
            tracker_config=Path("configs/trackers/general_botsort.yaml"),
            target_classes=("marine_target",),
        ),
        source_factory=lambda _value: source,
        detector_factory=lambda *_args, **_kwargs: FakeDetector(
            [(observed,), (observed,), (prediction_only,)]
        ),
    )

    # The first hit acquires, the second locks and can zoom, and a detection
    # without a current tracker identity is never treated as a zoom target.
    assert updates == [None, observed, None]
    assert len(cv2.resize_calls) == 2


def test_output_video_uses_wall_clock_timestamps_without_changing_display_rate(
    tmp_path: Path,
) -> None:
    writer = FakeWriter()
    cv2 = FakeCV2(writer)
    source = FakeSource([frame(0), frame(1), None], cv2)
    detector = FakeDetector([(), ()])
    timestamps = iter((0.0, 0.11))

    processed = run_vision(
        load_config("configs/development.yaml"),
        options(output=tmp_path / "annotated.mp4", display=True, output_fps=30.0),
        source_factory=lambda value: source,
        detector_factory=lambda *args, **kwargs: detector,
        monotonic=lambda: next(timestamps),
    )

    assert processed == 2
    assert len(cv2.shown_frames) == 2
    assert len(writer.frames) == 4
    assert writer.frames[0] is cv2.shown_frames[0]
    assert writer.frames[-1] is cv2.shown_frames[-1]
    assert writer.released is True


def test_digital_zoom_does_not_change_detection_or_controller_coordinates() -> None:
    selected = Detection("boat", 0.9, 70, 70, 90, 90)
    command_lines: list[str] = []

    for enabled in (False, True):
        source = FakeSource([frame(0), None])
        detector = FakeDetector([(selected,)])
        lines: list[str] = []
        run_vision(
            load_config("configs/development.yaml"),
            options(digital_zoom=enabled),
            source_factory=lambda value, source=source: source,
            detector_factory=lambda *args, detector=detector, **kwargs: detector,
            emit=lines.append,
        )
        command_lines.append(lines[0])

    for line in command_lines:
        assert "error_px=30.0,30.0" in line
        assert "logical_pan=93.00 logical_tilt=93.00" in line
    assert "digital_zoom=1.00x" in command_lines[0]
    assert "digital_zoom=1.25x" in command_lines[1]


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


def test_annotation_includes_tracking_state() -> None:
    writer = FakeWriter()
    cv2 = FakeCV2(writer)
    source = FakeSource([frame(0), None], cv2)
    detector = FakeDetector([(Detection("boat", 0.9, 40, 40, 60, 60),)])

    run_vision(
        load_config("configs/development.yaml"),
        options(display=True),
        source_factory=lambda value: source,
        detector_factory=lambda *args, **kwargs: detector,
    )

    assert "tracking=locked" in cv2.annotation_text


def test_annotation_labels_tracks_and_emphasizes_locked_selection_state() -> None:
    writer = FakeWriter()
    cv2 = FakeCV2(writer)
    source = FakeSource([frame(0), frame(1), None], cv2)
    observation = Detection("marine_target", 0.90, 40, 40, 60, 60, 7)
    detector = FakeDetector([(observation,), (observation,)])

    run_vision(
        load_config("configs/development.yaml"),
        options(
            display=True,
            tracker_backend="botsort",
            tracker_config=Path("configs/trackers/general_botsort.yaml"),
            target_classes=("marine_target",),
        ),
        source_factory=lambda _value: source,
        detector_factory=lambda *_args, **_kwargs: detector,
    )

    assert "marine_target id=7 0.90" in cv2.annotation_text
    assert any("tracker=botsort state=locked" in line for line in cv2.annotation_text)
    assert "confirmation_hits=2 unsupported_age=0.000s" in cv2.annotation_text


def test_terminal_and_display_diagnostics_distinguish_logical_physical_and_limits() -> None:
    config = load_config("configs/hardware.yaml")
    actuator = SimpleNamespace(
        pan_deg=105.0,
        tilt_deg=75.0,
        physical_pan_deg=75,
        physical_tilt_deg=105,
    )
    current_frame = frame(7)
    error = (12.5, -8.5)

    line = vision_cli._telemetry_line(
        current_frame,
        (),
        None,
        error,
        actuator,
        11.25,
        9.4,
        "cuda:0",
        "arduino_serial",
        config.actuator.pan_limits,
        config.actuator.tilt_limits,
    )

    assert "error_px=12.5,-8.5" in line
    assert "logical_pan=105.00 logical_tilt=75.00" in line
    assert "physical_pan=75 physical_tilt=105" in line
    assert "saturation_pan=max saturation_tilt=min" in line

    cv2 = FakeCV2(FakeWriter())
    vision_cli._annotate(
        cv2,
        current_frame,
        (),
        None,
        error,
        actuator,
        11.25,
        9.4,
        "cuda:0",
        config.actuator.pan_limits,
        config.actuator.tilt_limits,
    )

    assert "logical pan=105.00 tilt=75.00" in cv2.annotation_text
    assert "physical pan=75 tilt=105" in cv2.annotation_text
    assert "saturation pan=max tilt=min" in cv2.annotation_text
    assert "error=(12.5, -8.5) px" in cv2.annotation_text


def test_escape_terminates_display_and_releases_resources() -> None:
    writer = FakeWriter()
    cv2 = FakeCV2(writer, wait_key=27)
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


def test_duplicate_writer_failure_runs_bounded_cleanup_once(tmp_path: Path) -> None:
    writer = FakeWriter(
        write_error=RuntimeError("duplicate write failed"),
        fail_on_write_number=2,
    )
    cv2 = FakeCV2(writer)
    source = FakeSource([frame(0), frame(1), None], cv2)
    detector = FakeDetector([(), ()])
    timestamps = iter((0.0, 0.11))

    with pytest.raises(RuntimeError, match="duplicate write failed"):
        run_vision(
            load_config("configs/development.yaml"),
            options(output=tmp_path / "annotated.mp4", display=True, output_fps=30.0),
            source_factory=lambda value: source,
            detector_factory=lambda *args, **kwargs: detector,
            monotonic=lambda: next(timestamps),
        )

    assert writer.release_count == 1
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
        lambda config, selected, **kwargs: captured.append(selected) or 0,
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
            "--digital-zoom",
            "--max-digital-zoom",
            "1.75",
            "--output-fps",
            "24",
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
            digital_zoom=True,
            max_digital_zoom=1.75,
            output_fps=24.0,
        )
    ]


@pytest.mark.parametrize("value", ["nan", "inf", "0.9"])
def test_cli_rejects_invalid_maximum_digital_zoom_before_source_construction(
    value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = vision_cli.main(
        [
            "--config",
            "configs/development.yaml",
            "--source",
            "path:fixture.mp4",
            "--max-digital-zoom",
            value,
        ]
    )

    assert result == 2
    assert "maximum digital zoom" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_cli_rejects_invalid_output_fps_before_source_construction(
    value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = vision_cli.main(
        [
            "--config",
            "configs/development.yaml",
            "--source",
            "path:fixture.mp4",
            "--output-fps",
            value,
        ]
    )

    assert result == 2
    assert "encoded output FPS" in capsys.readouterr().err


def test_cli_uses_config_detection_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[VisionOptions] = []
    monkeypatch.setattr(
        vision_cli,
        "run_vision",
        lambda config, selected, **kwargs: captured.append(selected) or 0,
    )

    assert (
        vision_cli.main(["--config", "configs/development.yaml", "--source", "path:video.mp4"]) == 0
    )

    selected = captured[0]
    assert selected.source == "video.mp4"
    assert selected.model == "yolo11n.pt"
    assert selected.device == "auto"
    assert selected.target_classes == ("boat",)
    assert selected.confidence_threshold == 0.6
    assert selected.iou_threshold == 0.45
    assert selected.image_size == 640
    assert selected.runtime_mode == "single"
    assert selected.result_freshness_s is None
    assert selected.tracker_backend == "none"
    assert selected.tracker_config is None


def test_cli_tracker_overrides_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[VisionOptions] = []
    monkeypatch.setattr(
        vision_cli,
        "run_vision",
        lambda _config, selected, **_kwargs: captured.append(selected) or 0,
    )

    result = vision_cli.main(
        [
            "--config",
            "configs/development.yaml",
            "--source",
            "path:video.mp4",
            "--tracker",
            "botsort",
            "--tracker-config",
            "configs/trackers/general_botsort.yaml",
            "--tracker-input-confidence",
            "0.20",
            "--tracker-min-confirmation-hits",
            "3",
            "--tracker-max-unsupported-age",
            "0.40",
        ]
    )

    assert result == 0
    selected = captured[0]
    assert selected.tracker_backend == "botsort"
    assert selected.tracker_config == Path("configs/trackers/general_botsort.yaml")
    assert selected.tracker_input_confidence == 0.20
    assert selected.tracker_min_confirmation_hits == 3
    assert selected.tracker_max_unsupported_age_s == 0.40


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tracker_backend": "invalid"}, "tracker"),
        ({"tracker_backend": "botsort", "tracker_config": None}, "tracker-config"),
        (
            {
                "tracker_backend": "botsort",
                "tracker_config": Path("configs/trackers/general_botsort.yaml"),
                "tracker_input_confidence": 0.70,
            },
            "input confidence",
        ),
        ({"tracker_min_confirmation_hits": 0}, "confirmation hits"),
        ({"tracker_max_unsupported_age_s": 0.0}, "unsupported age"),
    ],
)
def test_tracker_option_validation_precedes_resource_construction(
    changes: dict[str, object],
    message: str,
) -> None:
    constructed = False

    def source_factory(_value: object) -> FakeSource:
        nonlocal constructed
        constructed = True
        return FakeSource([])

    with pytest.raises(ValueError, match=message):
        run_vision(
            load_config("configs/development.yaml"),
            replace(options(), **changes),
            source_factory=source_factory,
        )

    assert constructed is False


def test_cli_runtime_mode_and_freshness_override_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[VisionOptions] = []
    monkeypatch.setattr(
        vision_cli,
        "run_vision",
        lambda _config, selected, **_kwargs: captured.append(selected) or 0,
    )

    result = vision_cli.main(
        [
            "--config",
            "configs/development.yaml",
            "--source",
            "path:video.mp4",
            "--runtime-mode",
            "concurrent",
            "--result-freshness",
            "0.25",
        ]
    )

    assert result == 0
    assert captured[0].runtime_mode == "concurrent"
    assert captured[0].result_freshness_s == 0.25


@pytest.mark.parametrize(
    ("completion", "description"),
    [(0, "EOF"), (1, "q"), (5, "max frames")],
)
def test_main_returns_zero_for_normal_runtime_completion(
    monkeypatch: pytest.MonkeyPatch,
    completion: int,
    description: str,
) -> None:
    """The library count is not a process exit status for normal completion."""
    monkeypatch.setattr(
        vision_cli,
        "run_vision",
        lambda _config, _options, **_kwargs: completion,
    )

    assert (
        vision_cli.main(["--config", "configs/development.yaml", "--source", "path:fixture.mp4"])
        == 0
    ), description


@pytest.mark.parametrize(
    ("signal_number", "expected"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_main_returns_conventional_signal_status(
    monkeypatch: pytest.MonkeyPatch,
    signal_number: int,
    expected: int,
) -> None:
    request = vision_cli._TerminationRequest()
    request.request(signal_number)

    @contextmanager
    def termination_context() -> Any:
        yield request

    monkeypatch.setattr(vision_cli, "_termination_signals", termination_context)
    monkeypatch.setattr(vision_cli, "run_vision", lambda *_args, **_kwargs: 0)

    assert (
        vision_cli.main(["--config", "configs/development.yaml", "--source", "path:fixture.mp4"])
        == expected
    )


def test_main_returns_failure_for_unexpected_component_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def cancel(*_args: object, **_kwargs: object) -> int:
        raise OperationCancelled("detector cancelled")

    monkeypatch.setattr(vision_cli, "run_vision", cancel)

    assert (
        vision_cli.main(["--config", "configs/development.yaml", "--source", "path:fixture.mp4"])
        == 2
    )
    assert "detector cancelled" in capsys.readouterr().err


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


def test_explicit_simulated_options_override_hardware_configuration(
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


def test_cli_hardware_configuration_requires_explicit_arming(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = vision_cli.main(
        [
            "--config",
            "configs/hardware.yaml",
            "--source",
            "path:video.mp4",
            "--device",
            "cpu",
        ]
    )

    assert result == 2
    assert "--arm-hardware" in capsys.readouterr().err


def test_cli_simulated_override_avoids_hardware_arming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[VisionOptions] = []
    monkeypatch.setattr(
        vision_cli,
        "run_vision",
        lambda config, selected, **kwargs: captured.append(selected) or 0,
    )

    result = vision_cli.main(
        [
            "--config",
            "configs/hardware.yaml",
            "--source",
            "path:video.mp4",
            "--actuator-backend",
            "simulated",
        ]
    )

    assert result == 0
    assert captured[0].actuator_backend == "simulated"
    assert captured[0].arm_hardware is False
    assert captured[0].serial_port is None


def test_cli_values_override_hardware_backend_and_serial_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[VisionOptions] = []
    monkeypatch.setattr(
        vision_cli,
        "run_vision",
        lambda config, selected, **kwargs: captured.append(selected) or 0,
    )

    result = vision_cli.main(
        [
            "--config",
            "configs/hardware.yaml",
            "--source",
            "camera:1",
            "--actuator-backend",
            "arduino_serial",
            "--serial-port",
            "/dev/explicit-test-port",
            "--arm-hardware",
        ]
    )

    assert result == 0
    assert captured[0].source == 1
    assert captured[0].actuator_backend == "arduino_serial"
    assert captured[0].serial_port == "/dev/explicit-test-port"
    assert captured[0].arm_hardware is True


def test_cli_uses_validated_camera_and_serial_paths_when_not_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[VisionOptions] = []
    monkeypatch.setattr(
        vision_cli,
        "run_vision",
        lambda config, selected, **kwargs: captured.append(selected) or 0,
    )

    result = vision_cli.main(
        [
            "--config",
            "configs/hardware.yaml",
            "--arm-hardware",
        ]
    )

    assert result == 0
    assert captured[0].source == (
        "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"
    )
    assert captured[0].actuator_backend == "arduino_serial"
    assert captured[0].serial_port is None
