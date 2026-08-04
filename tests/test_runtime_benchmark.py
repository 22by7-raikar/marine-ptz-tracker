from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from marine_ptz.concurrent_runtime import RuntimeMetricsCollector
from marine_ptz.config import load_config
from marine_ptz.runtime_benchmark import (
    RuntimeBenchmarkError,
    RuntimeBenchmarkOptions,
    _PacedVideoSource,
    run_runtime_benchmark,
)
from marine_ptz.types import AngleLimits, PTZCommand
from marine_ptz.vision_cli import (
    RuntimeCompleteEvent,
    RuntimeFrameEvent,
    RuntimeProcessingEndEvent,
    RuntimeStartEvent,
)


def _concurrent_metrics() -> Any:
    collector = RuntimeMetricsCollector()
    for timestamp in (0.0, 0.02, 0.04):
        collector.record_event("inference", timestamp)
    for timestamp in (0.0, 0.1):
        collector.record_event("command", timestamp)
    for value in (0.01, 0.02, 0.03):
        collector.record_latency("capture_to_command", value)
    return collector.snapshot(
        dropped_capture_frames=4,
        dropped_render_frames=1,
        recorder_submitted_frames=3,
        recorder_consumed_frames=2,
        dropped_recorder_frames=1,
        encoded_frame_count=0,
        encoded_duplicate_frames=0,
        encoded_started_s=None,
        encoded_finished_s=None,
        shutdown_duration_s=0.02,
    )


def test_runtime_benchmark_compares_both_modes_without_hardware(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(_config: Any, options: Any, **kwargs: Any) -> int:
        calls.append(options.runtime_mode)
        assert options.actuator_backend == "simulated"
        assert options.display is False
        assert options.output is None
        if options.runtime_mode == "concurrent":
            kwargs["concurrent_metrics_sink"](_concurrent_metrics())
            return 3
        observer = kwargs["observer"]
        limits = AngleLimits(20.0, 160.0)
        observer.on_start(
            RuntimeStartEvent(0.0, "video", 2, "cpu", "simulated", 90.0, 90.0, limits, limits)
        )
        for sequence, timestamp in enumerate((0.1, 0.2)):
            observer.on_frame(
                RuntimeFrameEvent(
                    processing_timestamp_s=timestamp,
                    frame_capture_timestamp_s=timestamp - 0.02,
                    frame_sequence=sequence,
                    frame_width=100,
                    frame_height=100,
                    detection_count=0,
                    selected_detection=None,
                    command=PTZCommand(90.0, 90.0, sequence),
                    actuator_pan_deg=90.0,
                    actuator_tilt_deg=90.0,
                    inference_ms=2.0,
                    resolved_device="cpu",
                )
            )
        observer.on_processing_end(RuntimeProcessingEndEvent(0.25, 2, "eof"))
        observer.on_complete(RuntimeCompleteEvent(2, "eof"))
        return 2

    report_path = tmp_path / "runtime-comparison.json"
    report = run_runtime_benchmark(
        load_config("configs/development.yaml"),
        RuntimeBenchmarkOptions(
            source=tmp_path / "private" / "session.mp4",
            model="local.pt",
            device="cpu",
            target_classes=("marine_target",),
            confidence_threshold=0.6,
            iou_threshold=0.45,
            image_size=640,
            max_frames=20,
            report_path=report_path,
        ),
        runner=runner,
        clock=lambda: 0.30,
    )

    assert calls == ["single", "concurrent"]
    assert [result.mode for result in report.results] == ["single", "concurrent"]
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["source"] == "<local-path:session.mp4>"
    assert payload["schema_version"] == 4
    assert payload["source_playback_fps"] == 30.0
    assert payload["actuator_backend"] == "simulated"
    assert payload["configuration"]["model"] == "local.pt"
    assert payload["results"][0]["unique_inference_fps"]["rate_hz"] == 10.0
    assert payload["results"][0]["display_fps"]["rate_hz"] == 10.0
    assert payload["results"][0]["source_kind"] == "video"
    assert payload["results"][0]["declared_source_fps"] == 30.0
    assert payload["results"][0]["pacing_applied"] is True
    assert payload["results"][0]["capture_outcome"] == "normal_eof"
    assert payload["results"][0]["detector_readiness_ms"] is None
    assert payload["results"][0]["captured_frames"] == 2
    assert payload["results"][0]["inferred_frames"] == 2
    assert payload["results"][0]["displayed_frames"] == 2
    assert payload["results"][1]["dropped_capture_frames"] == 4
    assert payload["results"][1]["detector_readiness_ms"] is None
    assert payload["results"][1]["dropped_render_results"] == 1
    assert payload["results"][1]["control_hz"]["rate_hz"] == 10.0
    assert payload["results"][1]["recorder_submitted_unique_frames"] == 3
    assert payload["results"][1]["recorder_consumed_unique_frames"] == 2
    assert payload["results"][1]["dropped_recorder_frames"] == 1
    assert payload["results"][1]["encoded_frames"] == 0
    assert payload["results"][1]["encoded_duplicate_ratio"] is None
    assert payload["results"][1]["capture_to_command_age_ms"]["p95"] == 29.0


def test_runtime_benchmark_rejects_source_report_collision(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"

    with pytest.raises(RuntimeBenchmarkError, match="must differ"):
        RuntimeBenchmarkOptions(
            source=source,
            model="local.pt",
            device="cpu",
            target_classes=("marine_target",),
            confidence_threshold=0.6,
            iou_threshold=0.45,
            image_size=640,
            max_frames=None,
            report_path=source,
        )


def test_local_video_source_is_paced_at_configured_camera_rate() -> None:
    now = 0.0
    sleeps: list[float] = []

    class Source:
        cv2 = object()
        is_image = False
        total_frames = 2

        def __init__(self) -> None:
            self.values = [object(), object(), None]
            self.closed = False

        def read(self) -> object | None:
            return self.values.pop(0)

        def close(self) -> None:
            self.closed = True

    source = Source()

    def advance(duration: float) -> None:
        nonlocal now
        sleeps.append(duration)
        now += duration

    paced = _PacedVideoSource(
        "fixture.mp4",
        20.0,
        source_factory=lambda _value: source,
        monotonic=lambda: now,
        sleep=advance,
    )

    assert paced.read() is not None
    assert paced.read() is not None
    assert sleeps == [0.05]
    paced.close()
    assert source.closed is True


def test_local_video_source_skips_missed_slots_without_catchup_burst() -> None:
    now = 0.0
    sleeps: list[float] = []

    class Source:
        cv2 = object()
        is_image = False
        total_frames = 3

        def __init__(self) -> None:
            self.values = [object(), object(), object()]

        def read(self) -> object:
            return self.values.pop(0)

        def close(self) -> None:
            pass

    def advance(duration: float) -> None:
        nonlocal now
        sleeps.append(duration)
        now += duration

    paced = _PacedVideoSource(
        "fixture.mp4",
        20.0,
        source_factory=lambda _value: Source(),
        monotonic=lambda: now,
        sleep=advance,
    )

    paced.read()
    now = 0.20
    paced.read()
    paced.read()

    assert sleeps == [pytest.approx(0.05)]
