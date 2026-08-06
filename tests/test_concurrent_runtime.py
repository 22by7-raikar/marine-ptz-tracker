from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import marine_ptz.vision_cli as vision_cli
from marine_ptz.cancellation import OperationCancelled
from marine_ptz.concurrent_runtime import (
    LatestValueChannel,
    MonotonicControlSchedule,
    MonotonicPlaybackSchedule,
    RuntimeMetricsCollector,
    summarize_distribution,
)
from marine_ptz.config import AppConfig, load_config
from marine_ptz.types import Detection, Frame, PTZCommand
from marine_ptz.vision_cli import (
    StalePerceptionError,
    UnexpectedCancellationError,
    VisionOptions,
    run_vision,
)

pytestmark = pytest.mark.integration


class FakeImage:
    shape = (100, 100, 3)

    def copy(self) -> FakeImage:
        return FakeImage()


def frame(sequence: int) -> Frame:
    return Frame(FakeImage(), float(sequence), 100, 100, sequence)


def options(
    *,
    backend: str = "simulated",
    arm_hardware: bool = False,
    max_frames: int | None = None,
    freshness_s: float | None = None,
    tracker_backend: str = "none",
    tracker_config: Path | None = None,
    target_classes: tuple[str, ...] = ("boat",),
) -> VisionOptions:
    return VisionOptions(
        source="local.mp4",
        model="local.pt",
        device="cpu",
        target_classes=target_classes,
        confidence_threshold=0.5,
        iou_threshold=0.45,
        image_size=320,
        max_frames=max_frames,
        display=False,
        output=None,
        actuator_backend=backend,
        arm_hardware=arm_hardware,
        runtime_mode="concurrent",
        result_freshness_s=freshness_s,
        tracker_backend=tracker_backend,
        tracker_config=tracker_config,
        tracker_input_confidence=0.20,
        tracker_min_confirmation_hits=2,
        tracker_max_unsupported_age_s=0.30,
    )


class FiniteSource:
    def __init__(
        self,
        values: list[Frame | None | BaseException],
        events: list[str] | None = None,
        *,
        fps: float | None = None,
    ) -> None:
        self.values = list(values)
        self.events = events
        self.close_count = 0
        self.owner_threads: set[int] = set()
        self.owner_names: set[str] = set()
        self.owner_daemon_flags: set[bool] = set()
        self.fps = fps
        self.is_live = False
        self.is_image = False
        self.total_frames = sum(isinstance(value, Frame) for value in values)

    def read(self) -> Frame | None:
        self.owner_threads.add(threading.get_ident())
        self.owner_names.add(threading.current_thread().name)
        self.owner_daemon_flags.add(threading.current_thread().daemon)
        value = self.values.pop(0)
        if self.events is not None:
            self.events.append(
                "source:eof"
                if value is None
                else f"source:read:{getattr(value, 'sequence', 'error')}"
            )
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self) -> None:
        self.owner_threads.add(threading.get_ident())
        self.owner_names.add(threading.current_thread().name)
        self.owner_daemon_flags.add(threading.current_thread().daemon)
        if self.close_count == 0 and self.events is not None:
            self.events.append("source:close")
        self.close_count += 1


class FakeDetector:
    active_device = "cpu"
    last_inference_ms = 2.0

    def __init__(
        self,
        outputs: list[tuple[Detection, ...] | BaseException],
        events: list[str] | None = None,
    ) -> None:
        self.outputs = list(outputs)
        self.events = events
        self.owner_threads: set[int] = set()
        self.owner_names: set[str] = set()
        self.owner_daemon_flags: set[bool] = set()

    def detect(self, value: Frame) -> tuple[Detection, ...]:
        self.owner_threads.add(threading.get_ident())
        self.owner_names.add(threading.current_thread().name)
        self.owner_daemon_flags.add(threading.current_thread().daemon)
        if self.events is not None:
            self.events.append(f"detect:{value.sequence}")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


class FakeActuator:
    def __init__(
        self,
        events: list[str] | None = None,
        *,
        apply_error: BaseException | None = None,
        after_apply: Callable[[int], None] | None = None,
    ) -> None:
        self.events = events
        self.apply_error = apply_error
        self.after_apply = after_apply
        self.pan_deg = 90.0
        self.tilt_deg = 90.0
        self.commands: list[PTZCommand] = []
        self.owner_threads: set[int] = set()
        self.disable_count = 0
        self.close_count = 0

    def _record(self, name: str) -> None:
        self.owner_threads.add(threading.get_ident())
        if self.events is not None:
            self.events.append(name)

    def open(self) -> None:
        self._record("actuator:open")

    def enable(self) -> None:
        self._record("actuator:enable")

    def apply(self, command: PTZCommand) -> None:
        self._record(f"actuator:apply:{command.sequence}")
        if self.apply_error is not None:
            raise self.apply_error
        self.commands.append(command)
        self.pan_deg = command.pan_deg
        self.tilt_deg = command.tilt_deg
        if self.after_apply is not None:
            self.after_apply(len(self.commands))

    def disable(self) -> None:
        self._record("actuator:disable")
        self.disable_count += 1

    def close(self) -> None:
        self._record("actuator:close")
        self.close_count += 1


class _SharedClock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0.0

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> float:
        with self._lock:
            self._value += seconds
            return self._value


def run_fake(
    config: AppConfig,
    source: Any,
    detector: FakeDetector,
    actuator: FakeActuator,
    *,
    selected_options: VisionOptions | None = None,
    monotonic: Callable[[], float] | None = None,
    rate_monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    stop_requested: Callable[[], bool] = lambda: False,
    metrics: list[Any] | None = None,
    emit: Callable[[str], None] = lambda _line: None,
) -> int:
    kwargs: dict[str, Any] = {}
    if monotonic is not None:
        kwargs["monotonic"] = monotonic
    if rate_monotonic is not None:
        kwargs["rate_monotonic"] = rate_monotonic
    if sleep is not None:
        kwargs["sleep"] = sleep
    return run_vision(
        config,
        selected_options or options(),
        source_factory=lambda _value: source,
        detector_factory=lambda *_args, **_kwargs: detector,
        actuator_factory=lambda _config, _options: actuator,
        stop_requested=stop_requested,
        emit=emit,
        concurrent_metrics_sink=None if metrics is None else metrics.append,
        worker_join_timeout_s=0.2,
        **kwargs,
    )


def test_latest_value_channel_replaces_without_backlog() -> None:
    channel: LatestValueChannel[int] = LatestValueChannel()

    channel.publish(1)
    channel.publish(2)
    channel.publish(3)
    value = channel.latest()

    assert value is not None
    assert (value.version, value.value) == (3, 3)
    assert channel.overwritten_count == 2
    assert channel.latest(after_version=3, timeout_s=0.0) is None


def test_playback_schedule_is_30_fps_without_cumulative_drift() -> None:
    schedule = MonotonicPlaybackSchedule(30.0, started_s=10.0)
    now = 10.0

    for frame_index in range(1, 2636):
        now += schedule.wait_seconds(now)
        schedule.frame_started(now)
        assert now == pytest.approx(10.0 + frame_index / 30.0, abs=1e-9)

    assert schedule.next_deadline_s == pytest.approx(10.0 + 2636 / 30.0)


def test_playback_schedule_skips_missed_deadlines_without_a_burst() -> None:
    schedule = MonotonicPlaybackSchedule(30.0, started_s=0.0)

    schedule.frame_started(0.2)

    assert schedule.next_deadline_s == pytest.approx(7.0 / 30.0)
    assert schedule.wait_seconds(0.2) == pytest.approx(1.0 / 30.0)


def test_finite_video_capture_follows_declared_monotonic_fps() -> None:
    clock = _SharedClock()
    read_times: list[float] = []
    readiness_s = 7.192

    class TimedSource(FiniteSource):
        def read(self) -> Frame | None:
            read_times.append(clock())
            return super().read()

    source = TimedSource(
        [frame(0), frame(1), frame(2), frame(3), None],
        fps=30.0,
    )

    class DelayedFirstDetector(FakeDetector):
        def detect(self, value: Frame) -> tuple[Detection, ...]:
            if value.sequence == 0:
                clock.advance(readiness_s)
            return super().detect(value)

    reports: list[Any] = []
    actuator = FakeActuator()
    config = load_config("configs/development.yaml")

    run_fake(
        config,
        source,
        DelayedFirstDetector([(), (), (), ()]),
        actuator,
        monotonic=clock,
        rate_monotonic=clock,
        sleep=clock.advance,
        metrics=reports,
    )

    assert read_times == pytest.approx(
        [0.0, *(readiness_s + index / 30.0 for index in range(1, 5))]
    )
    assert reports[0].source_type == "video"
    assert reports[0].declared_source_fps == 30.0
    assert reports[0].pacing_applied is True
    assert reports[0].capture_outcome == "normal_eof"
    assert reports[0].capture.count == 4
    assert reports[0].capture.elapsed_s == pytest.approx(3.0 / 30.0)
    assert reports[0].capture.rate_hz == pytest.approx(30.0)
    assert reports[0].detector_readiness_ms == pytest.approx(7192.0)
    assert actuator.commands
    assert all(
        config.actuator.pan_limits.contains(command.pan_deg)
        and config.actuator.tilt_limits.contains(command.tilt_deg)
        for command in actuator.commands
    )


def test_detector_warmup_readiness_precedes_paced_finite_capture() -> None:
    events: list[str] = []
    clock = _SharedClock()
    source = FiniteSource([frame(0), frame(1), None], events, fps=30.0)

    class SynchronizingDetector(FakeDetector):
        def synchronize(self) -> None:
            events.append("detector:synchronize")

    processed = run_fake(
        load_config("configs/development.yaml"),
        source,
        SynchronizingDetector([(), ()], events),
        FakeActuator(),
        monotonic=clock,
        rate_monotonic=clock,
        sleep=clock.advance,
    )

    second_read = events.index("source:read:1")
    synchronizations = [
        index for index, event in enumerate(events) if event == "detector:synchronize"
    ]
    assert processed == 2
    assert len(synchronizations) == 2
    assert synchronizations[-1] < second_read


def test_normal_eof_drains_final_inflight_result_and_recorder(tmp_path: Path) -> None:
    eof_reached = threading.Event()
    clock = _SharedClock()

    class DrainSource(FiniteSource):
        cv2 = FakeCV2(FailingWriter())

        def read(self) -> Frame | None:
            value = super().read()
            if value is None:
                eof_reached.set()
            return value

    class DrainDetector(FakeDetector):
        def detect(self, value: Frame) -> tuple[Detection, ...]:
            if value.sequence == 1:
                assert eof_reached.wait(0.5)
            return super().detect(value)

    source = DrainSource([frame(0), frame(1), None], fps=30.0)
    writer = FailingWriter()
    source.cv2 = FakeCV2(writer)

    class Reports(list[Any]):
        def append(self, report: Any) -> None:
            assert writer.released is True
            super().append(report)

    reports: list[Any] = Reports()
    telemetry: list[str] = []
    processed = run_fake(
        load_config("configs/development.yaml"),
        source,
        DrainDetector([(), ()]),
        FakeActuator(),
        selected_options=replace(options(), output=tmp_path / "annotated.mp4"),
        monotonic=clock,
        rate_monotonic=clock,
        sleep=clock.advance,
        metrics=reports,
        emit=telemetry.append,
    )

    assert processed >= 1
    assert reports[0].capture_outcome == "normal_eof"
    assert reports[0].capture.count == 2
    assert reports[0].unique_inference.count == 2
    assert reports[0].display.count == processed
    assert reports[0].recorder_submitted_frames == processed
    assert reports[0].recorder_consumed_frames == processed
    assert reports[0].dropped_recorder_frames == 0
    frame_lines = [line for line in telemetry if line.startswith("seq=")]
    assert frame_lines[-1].startswith("seq=000001 ")


def test_concurrent_workers_have_single_component_owners() -> None:
    main_thread = threading.get_ident()
    construction_threads: dict[str, int] = {}
    source = FiniteSource([frame(0), None])
    detector = FakeDetector([()])
    actuator = FakeActuator()

    processed = run_vision(
        load_config("configs/development.yaml"),
        options(max_frames=1),
        source_factory=lambda _value: (
            construction_threads.setdefault("source", threading.get_ident()) and source
        ),
        detector_factory=lambda *_args, **_kwargs: (
            construction_threads.setdefault("detector", threading.get_ident()) and detector
        ),
        actuator_factory=lambda _config, _options: (
            construction_threads.setdefault("actuator", threading.get_ident()) and actuator
        ),
        emit=lambda _line: None,
    )

    assert processed == 1
    assert len(source.owner_threads) == len(detector.owner_threads) == 1
    assert construction_threads["source"] in source.owner_threads
    assert construction_threads["detector"] in detector.owner_threads
    assert source.owner_names == {"marine-ptz-capture"}
    assert detector.owner_names == {"marine-ptz-inference"}
    assert source.owner_daemon_flags == detector.owner_daemon_flags == {False}
    assert construction_threads["actuator"] == main_thread
    assert actuator.owner_threads == {main_thread}


def test_concurrent_tracker_model_and_selector_keep_distinct_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector_owner_names: set[str] = set()
    original_selector = vision_cli.MarineTargetSelector

    class RecordingSelector(original_selector):
        def select(self, frame_value: Frame, detections: Any) -> Any:
            selector_owner_names.add(threading.current_thread().name)
            return super().select(frame_value, detections)

    monkeypatch.setattr(vision_cli, "MarineTargetSelector", RecordingSelector)
    observation = Detection("marine_target", 0.90, 40, 40, 60, 60, 4)
    source = FiniteSource([frame(0), frame(1), None])
    detector = FakeDetector([(observation,), (observation,)])
    actuator = FakeActuator()

    processed = run_fake(
        load_config("configs/development.yaml"),
        source,
        detector,
        actuator,
        selected_options=options(
            max_frames=2,
            tracker_backend="botsort",
            tracker_config=Path("configs/trackers/general_botsort.yaml"),
            target_classes=("marine_target",),
        ),
    )

    assert processed == 2
    assert detector.owner_names == {"marine-ptz-inference"}
    assert selector_owner_names == {threading.current_thread().name}


def test_fresh_zero_detection_is_a_healthy_hold_command() -> None:
    source = FiniteSource([frame(0), None])
    detector = FakeDetector([()])
    actuator = FakeActuator()
    reports: list[Any] = []

    processed = run_fake(
        load_config("configs/development.yaml"),
        source,
        detector,
        actuator,
        selected_options=options(max_frames=1),
        metrics=reports,
    )

    assert processed == 1
    assert [(item.pan_deg, item.tilt_deg) for item in actuator.commands] == [(90.0, 90.0)]
    assert reports[0].stale_result_events == 0


def test_slow_model_warmup_finishes_before_hardware_enable() -> None:
    events: list[str] = []
    clock_lock = threading.Lock()
    clock_value = 0.0

    def clock() -> float:
        with clock_lock:
            return clock_value

    class SlowDetector(FakeDetector):
        def detect(self, value: Frame) -> tuple[Detection, ...]:
            nonlocal clock_value
            events.append("model:warmup:start")
            with clock_lock:
                clock_value = 1.25
            result = super().detect(value)
            events.append("model:warmup:complete")
            return result

    source = FiniteSource([frame(0), None], events)
    actuator = FakeActuator(events)

    processed = run_vision(
        load_config("configs/hardware.yaml"),
        options(backend="arduino_serial", arm_hardware=True, max_frames=1),
        source_factory=lambda _value: events.append("camera:open") or source,
        detector_factory=lambda *_args, **_kwargs: (
            events.append("model:load") or SlowDetector([()], events)
        ),
        actuator_factory=lambda *_args: events.append("actuator:construct") or actuator,
        monotonic=clock,
        emit=lambda _line: None,
    )

    assert processed == 1
    assert events.index("model:warmup:complete") < events.index("actuator:construct")
    assert events.index("actuator:construct") < events.index("actuator:enable")
    assert actuator.commands[0] == PTZCommand(90.0, 90.0, 0)
    assert actuator.disable_count == actuator.close_count == 1


class _ThreadClock:
    def __init__(self, main_values: list[float], worker_value: float = 0.0) -> None:
        self._main = threading.get_ident()
        self._values = iter(main_values)
        self._worker_value = worker_value

    def __call__(self) -> float:
        if threading.get_ident() == self._main:
            return next(self._values)
        return self._worker_value


def test_stale_perception_stops_set_and_disables() -> None:
    source = BlockingSource()
    detector = FakeDetector([()])
    actuator = FakeActuator()
    monotonic = _ThreadClock([0.0, 2.0, 2.0])
    rate_clock = iter((0.0, 0.0, 0.2))

    with pytest.raises(StalePerceptionError, match="stale"):
        run_fake(
            load_config("configs/hardware.yaml"),
            source,
            detector,
            actuator,
            selected_options=options(
                backend="arduino_serial",
                arm_hardware=True,
                freshness_s=0.5,
            ),
            monotonic=monotonic,
            rate_monotonic=lambda: next(rate_clock),
        )

    assert len(actuator.commands) == 1
    assert actuator.disable_count == actuator.close_count == 1


def test_camera_failure_never_constructs_detector_or_actuator() -> None:
    constructed: list[str] = []

    with pytest.raises(RuntimeError, match="capture worker failed: camera failed"):
        run_vision(
            load_config("configs/development.yaml"),
            options(),
            source_factory=lambda _value: (_ for _ in ()).throw(RuntimeError("camera failed")),
            detector_factory=lambda *_args, **_kwargs: constructed.append("detector"),
            actuator_factory=lambda *_args: constructed.append("actuator"),
            emit=lambda _line: None,
        )

    assert constructed == []


def test_detector_failure_never_constructs_actuator_and_closes_camera() -> None:
    source = FiniteSource([frame(0), None])
    detector = FakeDetector([RuntimeError("inference failed")])
    actuator_constructed = False

    def actuator_factory(_config: AppConfig, _options: VisionOptions) -> FakeActuator:
        nonlocal actuator_constructed
        actuator_constructed = True
        return FakeActuator()

    with pytest.raises(RuntimeError, match="inference worker failed: inference failed"):
        run_vision(
            load_config("configs/development.yaml"),
            options(),
            source_factory=lambda _value: source,
            detector_factory=lambda *_args, **_kwargs: detector,
            actuator_factory=actuator_factory,
            emit=lambda _line: None,
        )

    assert actuator_constructed is False
    assert source.close_count == 1


@pytest.mark.parametrize("stage", ["capture", "inference", "actuator"])
def test_uncorrelated_component_cancellation_is_actionable(stage: str) -> None:
    source = FiniteSource(
        [OperationCancelled("capture cancelled") if stage == "capture" else frame(0), None]
    )
    detector = FakeDetector(
        [OperationCancelled("inference cancelled") if stage == "inference" else ()]
    )
    actuator = FakeActuator(
        apply_error=(OperationCancelled("actuator cancelled") if stage == "actuator" else None)
    )

    with pytest.raises(UnexpectedCancellationError) as raised:
        run_fake(
            load_config("configs/development.yaml"),
            source,
            detector,
            actuator,
        )

    causes: list[BaseException] = []
    cause = raised.value.__cause__
    while cause is not None:
        causes.append(cause)
        cause = cause.__cause__
    assert any(isinstance(item, OperationCancelled) for item in causes)
    assert actuator.close_count <= 1


def test_serial_failure_disables_closes_and_sends_no_later_command() -> None:
    source = FiniteSource([frame(0), None])
    detector = FakeDetector([()])
    actuator = FakeActuator(apply_error=RuntimeError("serial failed"))

    with pytest.raises(RuntimeError, match="serial failed"):
        run_fake(
            load_config("configs/hardware.yaml"),
            source,
            detector,
            actuator,
            selected_options=options(backend="arduino_serial", arm_hardware=True),
        )

    assert actuator.commands == []
    assert actuator.disable_count == actuator.close_count == 1


class FailingWriter:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, _image: object) -> None:
        if self.error is not None:
            raise self.error

    def release(self) -> None:
        self.released = True


class FakeCV2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0
    MARKER_CROSS = 0

    def __init__(
        self,
        writer: FailingWriter,
        *,
        display_error: BaseException | None = None,
        wait_key: int = -1,
    ) -> None:
        self.writer = writer
        self.display_error = display_error
        self.wait_key = wait_key
        self.destroyed = False
        self.annotation_text: list[str] = []

    def VideoWriter_fourcc(self, *_values: str) -> int:
        return 1

    def VideoWriter(self, *_args: object) -> FailingWriter:
        return self.writer

    def namedWindow(self, _name: str) -> None:
        pass

    def rectangle(self, *_args: object) -> None:
        pass

    def putText(self, _image: object, text: str, *_args: object) -> None:
        self.annotation_text.append(text)

    def drawMarker(self, *_args: object) -> None:
        pass

    def line(self, *_args: object) -> None:
        pass

    def imshow(self, *_args: object) -> None:
        if self.display_error is not None:
            raise self.display_error

    def waitKey(self, _delay: int) -> int:
        return self.wait_key

    def destroyAllWindows(self) -> None:
        self.destroyed = True


def test_recorder_worker_failure_is_surfaced_after_bounded_cleanup(tmp_path: Path) -> None:
    writer = FailingWriter(RuntimeError("encode failed"))
    source = FiniteSource([frame(0), None])
    source.cv2 = FakeCV2(writer)
    actuator = FakeActuator()

    with pytest.raises(RuntimeError, match="recorder worker failed: encode failed"):
        run_fake(
            load_config("configs/development.yaml"),
            source,
            FakeDetector([()]),
            actuator,
            selected_options=replace(
                options(max_frames=1),
                output=tmp_path / "annotated.mp4",
            ),
        )

    assert writer.released is True
    assert actuator.close_count == source.close_count == 1


def test_display_failure_cancels_workers_and_releases_everything() -> None:
    writer = FailingWriter()
    cv2 = FakeCV2(writer, display_error=RuntimeError("display failed"))
    source = FiniteSource([frame(0), None])
    source.cv2 = cv2
    actuator = FakeActuator()

    with pytest.raises(RuntimeError, match="display failed"):
        run_fake(
            load_config("configs/development.yaml"),
            source,
            FakeDetector([()]),
            actuator,
            selected_options=replace(options(max_frames=1), display=True),
        )

    assert cv2.destroyed is True
    assert actuator.close_count == source.close_count == 1


@pytest.mark.parametrize("key", [ord("q"), 27])
def test_q_and_escape_stop_concurrent_display_with_bounded_cleanup(key: int) -> None:
    cv2 = FakeCV2(FailingWriter(), wait_key=key)
    source = FiniteSource([frame(0), None])
    source.cv2 = cv2
    actuator = FakeActuator()

    processed = run_fake(
        load_config("configs/development.yaml"),
        source,
        FakeDetector([()]),
        actuator,
        selected_options=replace(options(), display=True),
    )

    assert processed == 1
    assert cv2.destroyed is True
    assert actuator.close_count == source.close_count == 1


class BlockingSource:
    def __init__(self) -> None:
        self._first = True
        self._closed = threading.Event()
        self.close_count = 0

    def read(self) -> Frame | None:
        if self._first:
            self._first = False
            return frame(0)
        self._closed.wait()
        return None

    def close(self) -> None:
        if not self._closed.is_set():
            self.close_count += 1
            self._closed.set()


def test_live_camera_capture_is_not_artificially_paced() -> None:
    stopped = False
    sleeps: list[float] = []

    class LiveSource(BlockingSource):
        is_live = True
        is_image = False
        fps = 30.0
        total_frames = None

    def after_apply(_count: int) -> None:
        nonlocal stopped
        stopped = True

    reports: list[Any] = []
    run_fake(
        load_config("configs/development.yaml"),
        LiveSource(),
        FakeDetector([()]),
        FakeActuator(after_apply=after_apply),
        selected_options=replace(options(), source=0),
        stop_requested=lambda: stopped,
        sleep=sleeps.append,
        metrics=reports,
    )

    assert sleeps == []
    assert reports[0].source_type == "camera"
    assert reports[0].declared_source_fps == 30.0
    assert reports[0].pacing_applied is False
    assert reports[0].capture_outcome == "cancelled"
    assert reports[0].capture.count == 1
    assert reports[0].capture.rate_hz is None
    assert reports[0].detector_readiness_ms is None


def test_finite_capture_failure_remains_hard_and_is_reported() -> None:
    clock = _SharedClock()
    reports: list[Any] = []
    source = FiniteSource(
        [frame(0), RuntimeError("decoder failed")],
        fps=30.0,
    )

    with pytest.raises(RuntimeError, match="capture worker failed: decoder failed"):
        run_fake(
            load_config("configs/development.yaml"),
            source,
            FakeDetector([()]),
            FakeActuator(),
            monotonic=clock,
            rate_monotonic=clock,
            sleep=clock.advance,
            metrics=reports,
        )

    assert len(reports) == 1
    assert reports[0].capture_outcome == "capture_fault"
    assert reports[0].session_faults == 0


class AdvancingClock:
    def __init__(self, step: float) -> None:
        self._lock = threading.Lock()
        self._value = -step
        self._step = step

    def __call__(self) -> float:
        with self._lock:
            self._value += self._step
            return self._value


def test_no_target_refreshes_same_pose_and_cancellation_stops_later_commands() -> None:
    stopped = False

    def after_apply(count: int) -> None:
        nonlocal stopped
        if count == 2:
            stopped = True

    source = BlockingSource()
    detector = FakeDetector([()])
    actuator = FakeActuator(after_apply=after_apply)

    processed = run_fake(
        load_config("configs/development.yaml"),
        source,
        detector,
        actuator,
        rate_monotonic=AdvancingClock(0.1),
        stop_requested=lambda: stopped,
    )

    assert processed == 1
    assert [(item.pan_deg, item.tilt_deg) for item in actuator.commands] == [
        (90.0, 90.0),
        (90.0, 90.0),
    ]
    assert source.close_count == 1
    assert actuator.close_count == 1
    assert not any(thread.name.startswith("marine-ptz-") for thread in threading.enumerate())


def test_results_render_at_30_hz_while_control_remains_bounded_to_10_hz(
    tmp_path: Path,
) -> None:
    clock = _SharedClock()
    condition = threading.Condition()
    progress = {"detected": -1, "rendered": -1, "closed": False}
    telemetry: list[str] = []

    class PacedSource:
        cv2 = FakeCV2(FailingWriter())

        def __init__(self) -> None:
            self.sequence = 0
            self.close_count = 0

        def read(self) -> Frame | None:
            with condition:
                if self.sequence > 0:
                    condition.wait_for(
                        lambda: progress["detected"] >= self.sequence - 1 or progress["closed"]
                    )
                if progress["closed"] or self.sequence >= 30:
                    return None
                value = frame(self.sequence)
                self.sequence += 1
                return value

        def close(self) -> None:
            with condition:
                self.close_count += 1
                progress["closed"] = True
                condition.notify_all()

    class PacedDetector(FakeDetector):
        def __init__(self) -> None:
            super().__init__([])

        def detect(self, value: Frame) -> tuple[Detection, ...]:
            with condition:
                if value.sequence > 0:
                    condition.wait_for(
                        lambda: progress["rendered"] >= value.sequence - 1 or progress["closed"]
                    )
                progress["detected"] = value.sequence
                condition.notify_all()
            return (Detection("boat", 0.9, 80.0, 40.0, 90.0, 60.0),)

    class InternallyRateLimitedActuator(FakeActuator):
        def __init__(self) -> None:
            super().__init__()
            self.last_completed_s: float | None = None
            self.sleep_calls: list[float] = []
            self.started_s: list[float] = []

        def apply(self, command: PTZCommand) -> None:
            started_s = clock()
            if self.last_completed_s is not None:
                remaining = 0.1 - (started_s - self.last_completed_s)
                if remaining > 1e-9:
                    self.sleep_calls.append(remaining)
                    clock.advance(remaining)
            self.started_s.append(clock())
            super().apply(command)
            self.last_completed_s = clock()

    source = PacedSource()
    actuator = InternallyRateLimitedActuator()
    reports: list[Any] = []

    def emit(line: str) -> None:
        if not line.startswith("seq="):
            return
        telemetry.append(line)
        sequence = int(line.split()[0].split("=")[1])
        clock.advance(1.0 / 30.0)
        with condition:
            progress["rendered"] = sequence
            condition.notify_all()

    processed = run_vision(
        load_config("configs/development.yaml"),
        replace(
            options(max_frames=30),
            display=True,
            output=tmp_path / "annotated.mp4",
        ),
        source_factory=lambda _value: source,
        detector_factory=lambda *_args, **_kwargs: PacedDetector(),
        actuator_factory=lambda *_args: actuator,
        monotonic=clock,
        rate_monotonic=clock,
        emit=emit,
        concurrent_metrics_sink=reports.append,
        worker_join_timeout_s=0.5,
    )

    assert processed == 30
    assert len(telemetry) == 30
    assert 7 <= len(actuator.commands) <= 11
    assert len(actuator.commands) < processed
    assert actuator.sleep_calls == []
    assert [command.pan_deg for command in actuator.commands] == [
        90.0 + 3.0 * index for index in range(1, len(actuator.commands) + 1)
    ]
    assert [command.sequence for command in actuator.commands] == sorted(
        {command.sequence for command in actuator.commands}
    )
    assert all(
        later - earlier >= 0.1 - 1e-9
        for earlier, later in zip(actuator.started_s, actuator.started_s[1:])
    )
    assert reports[0].display.count == 30
    assert reports[0].display.rate_hz == pytest.approx(30.0)
    assert reports[0].recorder_submitted_frames == 30
    assert reports[0].recorder_consumed_frames + reports[0].dropped_recorder_frames == 30
    assert reports[0].recorder_consumed_frames >= 1
    assert reports[0].encoded_output.count >= 1
    assert reports[0].encoded_duplicate_frames <= reports[0].encoded_output.count
    assert all("unique_inference_fps=" in line for line in telemetry)
    assert all("display_fps=" in line for line in telemetry)
    assert all("control_hz=" in line for line in telemetry)
    assert all("encoded_output_fps=" in line for line in telemetry)
    assert all(" inference_ms=" in line and " fps=" not in line for line in telemetry)
    overlay = " ".join(source.cv2.annotation_text)
    assert "unique_inference_fps=" in overlay
    assert "display_fps=" in overlay
    assert "control_hz=" in overlay
    assert "encoded_output_fps=" in overlay
    assert "  fps=" not in overlay
    assert actuator.close_count == source.close_count == 1


def test_control_schedule_skips_delayed_slots_without_a_command_burst() -> None:
    schedule = MonotonicControlSchedule(10.0, first_deadline_s=0.0)

    assert schedule.is_due(0.0)
    schedule.command_completed(0.25)

    assert schedule.next_deadline_s == pytest.approx(0.35)
    assert schedule.is_due(0.25) is False
    assert schedule.is_due(0.349) is False
    assert schedule.bounded_wait(0.25, 0.05) == pytest.approx(0.05)
    assert schedule.is_due(0.35)

    schedule.command_completed(0.60)
    assert schedule.next_deadline_s == pytest.approx(0.70)


def test_metrics_separate_rates_and_cover_percentile_edges() -> None:
    assert summarize_distribution(()).to_dict() == {
        "count": 0,
        "mean": None,
        "p50": None,
        "p95": None,
        "p99": None,
    }
    singleton = summarize_distribution((7.0,))
    assert (singleton.mean, singleton.p50, singleton.p95, singleton.p99) == (
        7.0,
        7.0,
        7.0,
        7.0,
    )

    collector = RuntimeMetricsCollector()
    for timestamp in (0.0, 0.033, 0.066, 0.099):
        collector.record_event("capture", timestamp)
    for timestamp in (0.01, 0.043, 0.076):
        collector.record_event("inference", timestamp)
    for timestamp in (0.0, 0.1):
        collector.record_event("command", timestamp)
    collector.record_latency("capture_to_inference_start", 0.01)
    collector.record_latency("capture_to_inference_complete", 0.02)
    collector.record_latency("result_age_at_control", 0.03)
    collector.record_latency("capture_to_command", 0.04)

    report = collector.snapshot(
        dropped_capture_frames=4,
        dropped_render_frames=2,
        recorder_submitted_frames=3,
        recorder_consumed_frames=3,
        dropped_recorder_frames=0,
        encoded_frame_count=4,
        encoded_duplicate_frames=1,
        encoded_started_s=0.0,
        encoded_finished_s=0.1,
        shutdown_duration_s=0.02,
    )

    assert report.capture.rate_hz == pytest.approx(30.3030303)
    assert report.unique_inference.rate_hz == pytest.approx(30.3030303)
    assert report.actuator_commands.rate_hz == pytest.approx(10.0)
    assert report.encoded_output.count == 4
    assert report.recorder_submitted_frames == 3
    assert report.recorder_consumed_frames == 3
    assert report.encoded_duplicate_frames == 1
    assert report.encoded_duplicate_ratio == pytest.approx(0.25)
    assert report.dropped_capture_frames == 4
    assert report.capture_to_command_ms.mean == pytest.approx(40.0)
    assert report.to_dict()["captured_frames"] == 4
    assert report.to_dict()["inferred_frames"] == 3
    assert report.to_dict()["displayed_frames"] == 0

    sparse = RuntimeMetricsCollector()
    sparse.record_event("capture", 1.0)
    assert sparse.event_rate("capture").rate_hz is None


def test_finite_capture_metrics_exclude_readiness_for_reference_clip() -> None:
    collector = RuntimeMetricsCollector()
    readiness_s = 7.192
    frame_count = 2635
    collector.record_event("capture", 0.0)
    for frame_index in range(1, frame_count):
        collector.record_event(
            "capture",
            readiness_s + frame_index / 30.0,
        )

    report = collector.snapshot(
        dropped_capture_frames=0,
        dropped_render_frames=0,
        recorder_submitted_frames=frame_count,
        recorder_consumed_frames=frame_count,
        dropped_recorder_frames=0,
        encoded_frame_count=frame_count,
        encoded_duplicate_frames=0,
        encoded_started_s=readiness_s,
        encoded_finished_s=readiness_s + (frame_count - 1) / 30.0,
        shutdown_duration_s=0.00096,
        source_type="video",
        declared_source_fps=30.0,
        pacing_applied=True,
        capture_outcome="normal_eof",
        capture_playback_started_s=readiness_s,
        detector_readiness_s=readiness_s,
    )

    assert report.capture.count == 2635
    assert report.capture.elapsed_s == pytest.approx(87.8)
    assert report.capture.rate_hz == pytest.approx(30.0)
    assert report.detector_readiness_ms == pytest.approx(7192.0)
    assert report.encoded_output.count == 2635
    assert report.shutdown_duration_ms == pytest.approx(0.96)


def test_configuration_default_keeps_single_runtime_as_rollback() -> None:
    development = load_config(Path("configs/development.yaml"))
    hardware = load_config(Path("configs/hardware.yaml"))

    assert development.runtime.execution_mode == "single"
    assert hardware.runtime.execution_mode == "single"
    assert replace(options(), runtime_mode="single").runtime_mode == "single"


def test_hardware_freshness_override_must_remain_below_watchdog() -> None:
    source_constructed = False

    def source_factory(_value: object) -> FiniteSource:
        nonlocal source_constructed
        source_constructed = True
        return FiniteSource([])

    with pytest.raises(ValueError, match="watchdog"):
        run_vision(
            load_config("configs/hardware.yaml"),
            options(
                backend="arduino_serial",
                arm_hardware=True,
                freshness_s=1.0,
            ),
            source_factory=source_factory,
        )

    assert source_constructed is False
