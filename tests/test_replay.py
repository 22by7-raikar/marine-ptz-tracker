from __future__ import annotations

import json
import math
import os
import signal
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest

import marine_ptz.benchmark as benchmark
import marine_ptz.replay as replay
import marine_ptz.replay_cli as replay_cli
import marine_ptz.vision_cli as vision_cli
from marine_ptz.config import load_config
from marine_ptz.replay import (
    AcceptanceThresholds,
    ReplayCancelled,
    ReplayError,
    ReplayOptions,
    run_replay,
)
from marine_ptz.types import AngleLimits, Detection, Frame


class FakeImage:
    shape = (100, 100, 3)

    def copy(self) -> FakeImage:
        return FakeImage()


class FakeSource:
    def __init__(
        self,
        frames: list[Frame | None | BaseException],
        *,
        source: str = "fixture.mp4",
        total_frames: int | None = None,
    ) -> None:
        self._frames = list(frames)
        self._source = source
        self.is_image = Path(source).suffix.casefold() == ".png"
        self.total_frames = total_frames
        self.closed = 0

    def read(self) -> Frame | None:
        value = self._frames.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self) -> None:
        self.closed += 1


class FakeDetector:
    active_device = "cpu"

    def __init__(self, outputs: list[tuple[Detection, ...] | BaseException]) -> None:
        self._outputs = list(outputs)
        self.last_inference_ms = 4.0
        self.calls = 0

    def detect(self, _frame: Frame) -> tuple[Detection, ...]:
        self.calls += 1
        value = self._outputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        self.last_inference_ms = float(self.calls * 2)
        return value


class StepClock:
    def __init__(self, step: float = 0.1) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class SnapshotObserver:
    def __init__(self) -> None:
        self.start: object | None = None
        self.frames: list[object] = []
        self.processing_end: object | None = None
        self.complete: object | None = None

    def on_start(self, event: object) -> None:
        self.start = event

    def on_frame(self, event: object) -> None:
        self.frames.append(event)

    def on_processing_end(self, event: object) -> None:
        self.processing_end = event

    def on_complete(self, event: object) -> None:
        self.complete = event


def frame(sequence: int) -> Frame:
    return Frame(FakeImage(), float(sequence), 100, 100, sequence)


def boat(*, center_x: float = 75, center_y: float = 75) -> Detection:
    return Detection("boat", 0.9, center_x - 10, center_y - 10, center_x + 10, center_y + 10)


def options(tmp_path: Path, **changes: object) -> ReplayOptions:
    values: dict[str, object] = {
        "source": tmp_path / "fixture.mp4",
        "model": "local.pt",
        "device": "cpu",
        "target_classes": ("boat",),
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45,
        "image_size": 320,
        "max_frames": None,
        "report_path": tmp_path / "report.json",
    }
    values.update(changes)
    return ReplayOptions(**values)  # type: ignore[arg-type]


def run_fake(
    tmp_path: Path,
    frames: list[Frame | None | BaseException],
    detections: list[tuple[Detection, ...] | BaseException],
    **changes: object,
) -> replay.ReplayRunResult:
    source = FakeSource(frames, **changes.pop("source_kwargs", {}))
    detector = FakeDetector(detections)
    clock = changes.pop("clock", StepClock())
    return run_replay(
        load_config("configs/development.yaml"),
        options(tmp_path, **changes.pop("options", {})),
        source_factory=lambda _path: source,
        detector_factory=lambda *_args, **_kwargs: detector,
        monotonic=clock,
        rate_monotonic=lambda: 1.0,
        sleep=lambda _seconds: None,
        environment=lambda _device: {"torch_version": "fake", "secret": "never"},
        timestamp=lambda: datetime(2026, 7, 26, tzinfo=timezone.utc),
        command=("marine-ptz-replay", "--token", "secret"),
    )


def payload(result: replay.ReplayRunResult) -> dict[str, object]:
    return result.report.to_dict()


def test_single_image_uses_production_loop_and_writes_strict_report(tmp_path: Path) -> None:
    result = run_fake(
        tmp_path,
        [frame(0), None],
        [(boat(center_x=75, center_y=25),)],
        source_kwargs={"source": "fixture.png", "total_frames": 1},
        options={"source": tmp_path / "fixture.png"},
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    assert report == payload(result)
    assert report["schema_version"] == 1
    assert report["completion"] == {"status": "completed", "exit_reason": "eof"}
    assert report["input"]["source_type"] == "image"  # type: ignore[index]
    assert report["input"]["total_frames_available"] == 1  # type: ignore[index]
    assert report["metrics"]["processed_frames"] == 1  # type: ignore[index]
    assert report["metrics"]["simulated_actuator_command_count"] == 1  # type: ignore[index]
    assert report["environment"] == {"torch_version": "fake"}
    assert report["invocation"]["arguments"][-1] == "<redacted>"  # type: ignore[index]


def test_finite_video_metrics_cover_delayed_acquisition_and_lost_episodes(tmp_path: Path) -> None:
    result = run_fake(
        tmp_path,
        [frame(index) for index in range(7)] + [None],
        [(), (boat(),), (), (), (boat(center_x=50, center_y=50),), (), ()],
        source_kwargs={"total_frames": 7},
    )
    metrics = payload(result)["metrics"]

    assert metrics["processed_frames"] == 7  # type: ignore[index]
    assert metrics["frames_with_valid_detection"] == 2  # type: ignore[index]
    assert metrics["frames_with_selected_target"] == 2  # type: ignore[index]
    assert metrics["selected_target_ratio"] == pytest.approx(2 / 7)  # type: ignore[index]
    assert metrics["first_selected_target_frame"] == 1  # type: ignore[index]
    assert metrics["lost_target_episode_lengths_frames"] == [2, 2]  # type: ignore[index]
    assert metrics["lost_target_episode_count"] == 2  # type: ignore[index]
    assert metrics["approximate_processing_fps"] == pytest.approx(7 / 0.8)  # type: ignore[index]
    assert metrics["effective_command_rate_hz"] == pytest.approx(7 / 0.8)  # type: ignore[index]


def test_initial_pre_acquisition_absence_is_not_a_lost_target_episode(tmp_path: Path) -> None:
    result = run_fake(tmp_path, [frame(0), frame(1), None], [(), (boat(),)])
    metrics = payload(result)["metrics"]

    assert metrics["lost_target_episode_count"] == 0  # type: ignore[index]


def test_zero_frames_and_no_target_have_null_unavailable_metrics(tmp_path: Path) -> None:
    result = run_fake(tmp_path, [None], [])
    metrics = payload(result)["metrics"]

    assert metrics["processed_frames"] == 0  # type: ignore[index]
    assert metrics["selected_target_ratio"] is None  # type: ignore[index]
    assert metrics["inference_ms"] == {"mean": None, "median": None, "p95": None, "p99": None}  # type: ignore[index]
    assert metrics["normalized_center_error"]["absolute_euclidean"] == {"mean": None, "p95": None}  # type: ignore[index]


def test_detections_without_requested_target_are_not_selected(tmp_path: Path) -> None:
    ship = Detection("ship", 0.9, 40, 40, 60, 60)
    result = run_fake(tmp_path, [frame(0), None], [(ship,)])
    metrics = payload(result)["metrics"]

    assert metrics["frames_with_valid_detection"] == 1  # type: ignore[index]
    assert metrics["frames_with_selected_target"] == 0  # type: ignore[index]


def test_normalized_euclidean_error_and_single_sample_percentiles(tmp_path: Path) -> None:
    result = run_fake(tmp_path, [frame(0), None], [(boat(center_x=75, center_y=25),)])
    metrics = payload(result)["metrics"]

    assert metrics["inference_ms"] == {"mean": 2.0, "median": 2.0, "p95": 2.0, "p99": 2.0}  # type: ignore[index]
    error = metrics["normalized_center_error"]  # type: ignore[index]
    assert error["horizontal"]["mean"] == 0.5  # type: ignore[index]
    assert error["vertical"]["mean"] == -0.5  # type: ignore[index]
    assert error["absolute_euclidean"]["mean"] == pytest.approx(math.sqrt(0.5))  # type: ignore[index]


def test_saturation_uses_resulting_inclusive_simulated_angles(tmp_path: Path) -> None:
    config = load_config("configs/development.yaml")
    constrained = replace(
        config,
        actuator=replace(
            config.actuator,
            pan_limits=AngleLimits(90, 93),
            tilt_limits=AngleLimits(87, 90),
            initial_pan_deg=90,
            initial_tilt_deg=90,
        ),
    )
    source = FakeSource([frame(0), None])
    detector = FakeDetector([(boat(center_x=100, center_y=100),)])
    result = run_replay(
        constrained,
        options(tmp_path),
        source_factory=lambda _path: source,
        detector_factory=lambda *_args, **_kwargs: detector,
        monotonic=StepClock(),
        rate_monotonic=lambda: 1.0,
        sleep=lambda _seconds: None,
    )
    metrics = payload(result)["metrics"]

    assert metrics["resulting_pan_saturation_count"] == 1  # type: ignore[index]
    assert metrics["resulting_tilt_saturation_count"] == 1  # type: ignore[index]
    assert metrics["final_pan_deg"] == 93  # type: ignore[index]
    assert metrics["final_tilt_deg"] == 90  # type: ignore[index]


def test_hardware_configuration_still_composes_only_simulated_actuation(tmp_path: Path) -> None:
    result = run_replay(
        load_config("configs/hardware.yaml"),
        options(tmp_path),
        source_factory=lambda _path: FakeSource([frame(0), None]),
        detector_factory=lambda *_args, **_kwargs: FakeDetector([(boat(),)]),
        monotonic=StepClock(),
        rate_monotonic=lambda: 1.0,
        sleep=lambda _seconds: None,
    )

    assert payload(result)["metrics"]["simulated_actuator_command_count"] == 1  # type: ignore[index]


def test_max_frames_is_successful_and_does_not_read_extra_frame(tmp_path: Path) -> None:
    result = run_fake(
        tmp_path,
        [frame(0), frame(1)],
        [(), ()],
        options={"max_frames": 1},
    )
    assert payload(result)["completion"]["exit_reason"] == "max_frames"  # type: ignore[index]


def test_detector_or_decode_failure_preserves_existing_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"previous": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="detector failed"):
        run_fake(tmp_path, [frame(0)], [RuntimeError("detector failed")])
    assert report.read_text(encoding="utf-8") == '{"previous": true}\n'

    with pytest.raises(RuntimeError, match="decode failed"):
        run_fake(tmp_path, [RuntimeError("decode failed")], [])
    assert report.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_cancellation_does_not_write_or_claim_completed_report(tmp_path: Path) -> None:
    source = FakeSource([frame(0)])
    with pytest.raises(ReplayCancelled):
        run_replay(
            load_config("configs/development.yaml"),
            options(tmp_path),
            source_factory=lambda _path: source,
            detector_factory=lambda *_args, **_kwargs: FakeDetector([()]),
            stop_requested=lambda: True,
        )
    assert not (tmp_path / "report.json").exists()
    assert source.closed == 1


def test_threshold_pass_failure_and_unavailable_metric_are_reported(tmp_path: Path) -> None:
    passing = run_fake(
        tmp_path,
        [frame(0), None],
        [(boat(),)],
        options={"thresholds": AcceptanceThresholds(minimum_selected_target_ratio=1.0)},
    )
    assert passing.accepted is True

    failed = run_fake(
        tmp_path,
        [frame(0), None],
        [(boat(),)],
        options={"thresholds": AcceptanceThresholds(maximum_p95_inference_ms=1.0)},
    )
    assert failed.accepted is False
    assert json.loads((tmp_path / "report.json").read_text())["thresholds"]["acceptance"] is False

    unavailable = run_fake(
        tmp_path,
        [None],
        [],
        options={"thresholds": AcceptanceThresholds(maximum_p95_inference_ms=1.0)},
    )
    result = payload(unavailable)["thresholds"]["results"][0]  # type: ignore[index]
    assert result["passed"] is False
    assert result["reason"] == "metric unavailable"


def test_strict_json_rejects_non_finite_environment_and_preserves_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"previous": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Out of range float"):
        run_replay(
            load_config("configs/development.yaml"),
            options(tmp_path),
            source_factory=lambda _path: FakeSource([frame(0), None]),
            detector_factory=lambda *_args, **_kwargs: FakeDetector([()]),
            environment=lambda _device: {"torch_version": float("nan")},
        )
    assert report.read_text(encoding="utf-8") == '{"previous": true}\n'


@pytest.mark.parametrize(
    "changes",
    [
        {"confidence_threshold": float("nan")},
        {"iou_threshold": float("inf")},
        {"max_frames": 0},
        {"report_path": Path("same.mp4"), "source": Path("same.mp4")},
        {"annotated_output": Path("same.mp4"), "source": Path("same.mp4")},
    ],
)
def test_options_reject_invalid_numbers_and_path_collisions(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    if "report_path" in changes:
        changes = {
            **changes,
            "report_path": tmp_path / str(changes["report_path"]),
            "source": tmp_path / str(changes["source"]),
        }
    with pytest.raises(ReplayError):
        options(tmp_path, **changes)


def test_cli_rejects_live_hardware_and_colliding_paths_without_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fixture")
    invoked = False

    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal invoked
        invoked = True
        return SimpleNamespace(accepted=None)

    monkeypatch.setattr(replay_cli, "run_replay", fake_run)
    assert replay_cli.main(["--source", "camera:0", "--report", str(tmp_path / "r.json")]) == 2
    assert replay_cli.main(["--source", str(input_path), "--report", str(input_path)]) == 2
    assert (
        replay_cli.main(
            ["--source", str(input_path), "--report", str(tmp_path / "r.json"), "--arm-hardware"]
        )
        == 2
    )
    assert invoked is False


def test_cli_requires_explicit_replay_safe_for_hardware_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fixture")
    monkeypatch.setattr(
        replay_cli, "run_replay", lambda *_args, **_kwargs: SimpleNamespace(accepted=None)
    )

    command = [
        "--config",
        "configs/hardware.yaml",
        "--source",
        str(input_path),
        "--report",
        str(tmp_path / "r.json"),
    ]
    assert replay_cli.main(command) == 2
    assert replay_cli.main([*command, "--replay-safe"]) == 0


@pytest.mark.parametrize(
    ("signal_number", "expected"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_cli_cancellation_preserves_signal_exit_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_number: int,
    expected: int,
) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fixture")
    request = SimpleNamespace(signal_number=signal_number)

    class TerminationContext:
        def __enter__(self) -> SimpleNamespace:
            return request

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(replay_cli, "_termination_signals", TerminationContext)
    monkeypatch.setattr(
        replay_cli,
        "run_replay",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ReplayCancelled("cancelled")),
    )

    assert (
        replay_cli.main(["--source", str(input_path), "--report", str(tmp_path / "r.json")])
        == expected
    )


def test_complete_report_sanitizes_all_local_paths_and_preserves_safe_models(
    tmp_path: Path,
) -> None:
    report = replay.ReplayReport(
        timestamp_utc="2026-07-26T00:00:00+00:00",
        command=(
            "marine-ptz-replay",
            "--source",
            "/home/alice/private/boats/case1.mp4",
            "--model=/home/alice/models/yolo11n.pt",
            "--config",
            "/home/alice/private/config.yaml",
            "--report=/tmp/private-output/report.json",
            "--annotated-output",
            "path:/tmp/private-output/annotated.mp4",
            "path:/home/alice/private/positional.mp4",
            "/home/alice/private/another-positional.mp4",
            "https://user:password@example.com/model.pt?token=secret",
        ),
        options=options(
            tmp_path,
            source=Path("/home/alice/private/boats/case1.mp4"),
            model="/home/alice/models/yolo11n.pt",
            report_path=Path("/tmp/private-output/report.json"),
            annotated_output=Path("/tmp/private-output/annotated.mp4"),
        ),
        source_type="video",
        total_frames_available=1,
        frame_dimensions=(100, 100),
        resolved_device="cpu",
        environment={"torch_version": "/home/alice/private/torch"},
        processed_frames=1,
        frames_with_detection=0,
        frames_with_target=0,
        first_target_frame=None,
        acquisition_processing_s=None,
        lost_target_episodes=(),
        inference_ms=(1.0,),
        horizontal_errors=(),
        vertical_errors=(),
        absolute_errors=(),
        command_count=1,
        processing_elapsed_s=1.0,
        pan_saturation_count=0,
        tilt_saturation_count=0,
        final_pan_deg=90.0,
        final_tilt_deg=90.0,
        exit_reason="eof",
        threshold_results=(),
    )
    serialized = json.dumps(report.to_dict(), allow_nan=False, sort_keys=True)

    for private_fragment in (
        "/home/alice/private",
        "/home/alice/models",
        "/tmp/private-output",
    ):
        assert private_fragment not in serialized
    assert "<local-path:case1.mp4>" in serialized
    assert "<local-path:yolo11n.pt>" in serialized
    assert "<redacted>@example.com" in serialized
    assert "token=%3Credacted%3E" in serialized
    assert replay._sanitize_replay_model("yolo11n.pt") == "yolo11n.pt"


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        ("yolo11n.pt", "yolo11n.pt"),
        ("/home/alice/models/custom.pt", "<local-path:custom.pt>"),
        (
            "https://user:password@models.example/private/custom.pt?token=secret#fragment",
            "<remote-resource:custom.pt>",
        ),
        ("https://models.example", "<remote-resource:redacted>"),
        (
            "https://models.example/private/%253Ftoken%253Dsecret",
            "<remote-resource:redacted>",
        ),
    ),
)
def test_replay_structured_model_uses_model_field_policy(model: str, expected: str) -> None:
    assert replay._sanitize_replay_model(model) == expected


def test_replay_path_typed_urls_are_marked_by_field_type() -> None:
    source_url = "https://user:password@sources.example/private/frame.png?token=secret#fragment"
    report_url = "file:///home/alice/private/result.json?token=secret#fragment"
    config_url = "https://configs.example"
    annotated_url = "https://writer:pass@outputs.example/private/annotated.mp4?key=secret#fragment"

    assert replay.sanitize_replay_command(
        (
            "replay",
            f"--source={source_url}",
            "--report",
            report_url,
            f"--config={config_url}",
            "--annotated-output",
            annotated_url,
        )
    ) == (
        "replay",
        "--source=<local-path:frame.png>",
        "--report",
        "<local-path:result.json>",
        "--config=<local-path:redacted>",
        "--annotated-output",
        "<local-path:annotated.mp4>",
    )


def test_replay_path_forced_url_values_keep_only_safe_url_path_basenames(
    tmp_path: Path,
) -> None:
    remote_model = "path:https://user:pass@example.com/private/model.pt?token=secret#fragment"
    file_source = "path:file:///home/alice/private/frame.png?token=secret#fragment"
    report_path = "path:https://report-user:report-pass@reports.example/private/report.json?key=secret#fragment"
    command = (
        "marine-ptz-replay",
        "--model",
        remote_model,
        f"--source={file_source}",
        "--report",
        report_path,
    )

    assert replay.sanitize_replay_command(command) == (
        "marine-ptz-replay",
        "--model",
        "<local-path:model.pt>",
        "--source=<local-path:frame.png>",
        "--report",
        "<local-path:report.json>",
    )
    assert replay.sanitize_replay_command(
        (
            "marine-ptz-replay",
            "--model",
            "path:/home/alice/models/model.pt",
            "--source=path:../frames/frame.png",
        )
    ) == (
        "marine-ptz-replay",
        "--model",
        "<local-path:model.pt>",
        "--source=<local-path:frame.png>",
    )

    original = _report_for_path_privacy(tmp_path)
    report = replace(
        original,
        command=command,
        options=options(tmp_path, model=remote_model),
    )
    payload = report.to_dict()
    _assert_replay_private_fragments_absent(
        payload,
        (
            "user",
            "pass",
            "token",
            "secret",
            "fragment",
            "example.com",
            "reports.example",
            "private",
            "/home/alice",
            "alice",
        ),
    )
    serialized = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert "<local-path:model.pt>" in serialized
    assert "<local-path:frame.png>" in serialized
    assert "<local-path:report.json>" in serialized
    assert replay._sanitize_replay_model("yolo11n.pt") == "yolo11n.pt"


def test_replay_report_path_typed_urls_never_retain_url_structure(
    tmp_path: Path,
) -> None:
    model_url = (
        "https://model-user:model-pass@models.example/private/custom.pt?token=secret#fragment"
    )
    source_url = (
        "https://source-user:source-pass@sources.example/private/frame.png?token=secret#fragment"
    )
    report_url = "file:///home/alice/private/result.json?%74oken=%73ecret#fragment"
    config_url = "https://configs.example"
    annotated_url = (
        "https://output-user:output-pass@outputs.example/private/annotated.mp4?key=secret#fragment"
    )
    report = replace(
        _report_for_path_privacy(tmp_path),
        command=(
            "marine-ptz-replay",
            "--model",
            model_url,
            f"--source={source_url}",
            "--report",
            report_url,
            f"--config={config_url}",
            "--annotated-output",
            annotated_url,
        ),
        options=options(
            tmp_path,
            source=Path("/home/alice/private/source.mp4"),
            model=model_url,
        ),
    )

    payload = report.to_dict()
    _assert_replay_private_fragments_absent(
        payload,
        (
            "model-user",
            "model-pass",
            "source-user",
            "source-pass",
            "output-user",
            "output-pass",
            "models.example",
            "sources.example",
            "outputs.example",
            "configs.example",
            "private",
            "token",
            "secret",
            "fragment",
            "%74oken",
            "%73ecret",
            "/home/alice",
            "alice",
        ),
    )
    serialized = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert "<remote-resource:custom.pt>" in serialized
    assert "<local-path:source.mp4>" in serialized
    assert "<local-path:frame.png>" in serialized
    assert "<local-path:result.json>" in serialized
    assert "<local-path:annotated.mp4>" in serialized
    assert "<local-path:redacted>" in serialized


@pytest.mark.parametrize(
    "argument",
    (
        "path:https://example.com",
        "path:file://",
        "path:https://[",
        "path:https://example.com/%3Ftoken%3Dsecret",
        "path:https://example.com/%253Ftoken%253Dsecret",
    ),
)
def test_replay_path_forced_malformed_or_encoded_urls_use_generic_marker(argument: str) -> None:
    assert replay.sanitize_replay_command(("replay", "--model", argument)) == (
        "replay",
        "--model",
        "<local-path:redacted>",
    )


def _report_for_path_privacy(tmp_path: Path) -> replay.ReplayReport:
    return replay.ReplayReport(
        timestamp_utc="2026-07-26T00:00:00+00:00",
        command=(),
        options=options(tmp_path),
        source_type="video",
        total_frames_available=1,
        frame_dimensions=(100, 100),
        resolved_device="cpu",
        environment={},
        processed_frames=1,
        frames_with_detection=0,
        frames_with_target=0,
        first_target_frame=None,
        acquisition_processing_s=None,
        lost_target_episodes=(),
        inference_ms=(1.0,),
        horizontal_errors=(),
        vertical_errors=(),
        absolute_errors=(),
        command_count=1,
        processing_elapsed_s=1.0,
        pan_saturation_count=0,
        tilt_saturation_count=0,
        final_pan_deg=90.0,
        final_tilt_deg=90.0,
        exit_reason="eof",
        threshold_results=(),
    )


def _assert_replay_private_fragments_absent(
    value: object,
    fragments: tuple[str, ...],
) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _assert_replay_private_fragments_absent(nested, fragments)
        return
    if isinstance(value, list):
        for nested in value:
            _assert_replay_private_fragments_absent(nested, fragments)
        return
    if isinstance(value, str):
        for fragment in fragments:
            assert fragment not in value


def test_immutable_observer_events_contain_only_snapshots_and_cannot_mutate(
    tmp_path: Path,
) -> None:
    source = FakeSource([frame(0), None])
    detector = FakeDetector([(boat(),)])
    observer = SnapshotObserver()
    actuator = []

    processed = vision_cli.run_vision(
        load_config("configs/development.yaml"),
        vision_cli.VisionOptions(
            source="fixture.mp4",
            model="local.pt",
            device="cpu",
            target_classes=("boat",),
            confidence_threshold=0.5,
            iou_threshold=0.45,
            image_size=320,
            max_frames=None,
            display=False,
            output=None,
        ),
        source_factory=lambda _source: source,
        detector_factory=lambda *_args, **_kwargs: detector,
        actuator_factory=lambda config, _options: (
            actuator.append(
                vision_cli.SimulatedPTZActuator(
                    config.actuator.pan_limits,
                    config.actuator.tilt_limits,
                    initial_pan_deg=config.actuator.initial_pan_deg,
                    initial_tilt_deg=config.actuator.initial_tilt_deg,
                )
            )
            or actuator[0]
        ),
        observer=observer,
        emit=lambda _line: None,
    )

    assert processed == 1
    assert isinstance(observer.start, vision_cli.RuntimeStartEvent)
    assert isinstance(observer.frames[0], vision_cli.RuntimeFrameEvent)
    event = observer.frames[0]
    assert not hasattr(event, "image")
    assert not hasattr(event, "actuator")
    with pytest.raises(FrozenInstanceError):
        event.frame_width = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        event.selected_detection.label = "mutated"  # type: ignore[union-attr,misc]
    assert actuator[0].pan_deg == 93.0


@pytest.mark.parametrize("phase", ["start", "frame", "processing_end"])
def test_observer_failure_still_cleans_and_does_not_overwrite_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text('{"previous": true}\n', encoding="utf-8")
    source = FakeSource([frame(0), None])

    class FailingObserver:
        def on_start(self, _event: object) -> None:
            if phase == "start":
                raise RuntimeError("observer start failed")

        def on_frame(self, _event: object) -> None:
            if phase == "frame":
                raise RuntimeError("observer frame failed")

        def on_processing_end(self, _event: object) -> None:
            if phase == "processing_end":
                raise RuntimeError("observer end failed")

        def on_complete(self, _event: object) -> None:
            pass

    monkeypatch.setattr(replay, "ReplayMetrics", FailingObserver)
    with pytest.raises(RuntimeError, match="observer"):
        run_replay(
            load_config("configs/development.yaml"),
            options(tmp_path),
            source_factory=lambda _path: source,
            detector_factory=lambda *_args, **_kwargs: FakeDetector([(boat(),)]),
        )
    assert source.closed == 1
    assert report_path.read_text(encoding="utf-8") == '{"previous": true}\n'


class ProcessingEndFailingObserver(SnapshotObserver):
    """Record the lifecycle boundary, then fail deterministically."""

    def __init__(self, message: str = "observer end failed") -> None:
        super().__init__()
        self.message = message
        self.processing_end_calls = 0

    def on_processing_end(self, event: object) -> None:
        super().on_processing_end(event)
        self.processing_end_calls += 1
        raise RuntimeError(self.message)


def _vision_options() -> vision_cli.VisionOptions:
    return vision_cli.VisionOptions(
        source="fixture.mp4",
        model="local.pt",
        device="cpu",
        target_classes=("boat",),
        confidence_threshold=0.5,
        iou_threshold=0.45,
        image_size=320,
        max_frames=None,
        display=False,
        output=None,
    )


@pytest.mark.parametrize("failure_kind", ["detector", "source"])
def test_processing_failure_remains_primary_when_processing_end_observer_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """The production loop keeps control-path failures ahead of observers."""
    report_path = tmp_path / "report.json"
    report_path.write_text('{"previous": true}\n', encoding="utf-8")
    primary = RuntimeError(f"{failure_kind} failed")
    source = FakeSource(
        [primary] if failure_kind == "source" else [frame(0)],
    )
    detector = FakeDetector(
        [] if failure_kind == "source" else [primary],
    )
    base_metrics = replay.ReplayMetrics

    class EndFailingMetrics(base_metrics):
        def on_processing_end(self, event: object) -> None:
            super().on_processing_end(event)
            raise RuntimeError("observer end failed")

    monkeypatch.setattr(replay, "ReplayMetrics", EndFailingMetrics)

    with pytest.raises(RuntimeError) as captured:
        run_replay(
            load_config("configs/development.yaml"),
            options(tmp_path),
            source_factory=lambda _path: source,
            detector_factory=lambda *_args, **_kwargs: detector,
        )

    assert captured.value is primary
    diagnostic = captured.value.__cause__
    assert isinstance(diagnostic, vision_cli.RuntimeSecondaryFailures)
    assert str(diagnostic.observer_error) == "observer end failed"
    assert diagnostic.cleanup_errors == ()
    assert source.closed == 1
    assert report_path.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_actuator_failure_remains_primary_when_processing_end_observer_fails() -> None:
    primary = RuntimeError("actuator failed")
    source = FakeSource([frame(0)])
    observer = ProcessingEndFailingObserver()
    config = load_config("configs/development.yaml")

    class FailingActuator(vision_cli.SimulatedPTZActuator):
        def __init__(self) -> None:
            super().__init__(
                config.actuator.pan_limits,
                config.actuator.tilt_limits,
                initial_pan_deg=config.actuator.initial_pan_deg,
                initial_tilt_deg=config.actuator.initial_tilt_deg,
            )
            self.close_calls = 0

        def apply(self, _command: object) -> None:
            raise primary

        def close(self) -> None:
            self.close_calls += 1
            super().close()

    actuator = FailingActuator()
    with pytest.raises(RuntimeError) as captured:
        vision_cli.run_vision(
            config,
            _vision_options(),
            source_factory=lambda _source: source,
            detector_factory=lambda *_args, **_kwargs: FakeDetector([(boat(),)]),
            actuator_factory=lambda _config, _options: actuator,
            observer=observer,
            emit=lambda _line: None,
        )

    assert captured.value is primary
    diagnostic = captured.value.__cause__
    assert isinstance(diagnostic, vision_cli.RuntimeSecondaryFailures)
    assert str(diagnostic.observer_error) == "observer end failed"
    assert diagnostic.cleanup_errors == ()
    assert observer.processing_end_calls == 1
    assert source.closed == actuator.close_calls == 1


def test_processing_failure_retains_observer_and_cleanup_failures_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = RuntimeError("detector failed")
    cleanup_error = RuntimeError("source cleanup failed")
    report_path = tmp_path / "report.json"
    report_path.write_text('{"previous": true}\n', encoding="utf-8")

    class FailingCloseSource(FakeSource):
        def close(self) -> None:
            super().close()
            raise cleanup_error

    source = FailingCloseSource([frame(0)])
    base_metrics = replay.ReplayMetrics

    class EndFailingMetrics(base_metrics):
        def on_processing_end(self, event: object) -> None:
            super().on_processing_end(event)
            raise RuntimeError("observer end failed")

    monkeypatch.setattr(replay, "ReplayMetrics", EndFailingMetrics)
    with pytest.raises(RuntimeError) as captured:
        run_replay(
            load_config("configs/development.yaml"),
            options(tmp_path),
            source_factory=lambda _path: source,
            detector_factory=lambda *_args, **_kwargs: FakeDetector([primary]),
        )

    assert captured.value is primary
    diagnostic = captured.value.__cause__
    assert isinstance(diagnostic, vision_cli.RuntimeSecondaryFailures)
    assert str(diagnostic.observer_error) == "observer end failed"
    assert diagnostic.cleanup_errors == (cleanup_error,)
    assert source.closed == 1
    assert report_path.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_eof_processing_end_observer_failure_is_primary_and_cli_returns_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_error = RuntimeError("source cleanup failed")

    class FailingCloseSource(FakeSource):
        def close(self) -> None:
            super().close()
            raise cleanup_error

    source = FailingCloseSource([None])
    observer = ProcessingEndFailingObserver()
    with pytest.raises(RuntimeError, match="observer end failed") as captured:
        vision_cli.run_vision(
            load_config("configs/development.yaml"),
            _vision_options(),
            source_factory=lambda _source: source,
            detector_factory=lambda *_args, **_kwargs: FakeDetector([]),
            observer=observer,
            emit=lambda _line: None,
        )
    diagnostic = captured.value.__cause__
    assert isinstance(diagnostic, vision_cli.RuntimeSecondaryFailures)
    assert diagnostic.observer_error is None
    assert diagnostic.cleanup_errors == (cleanup_error,)
    assert observer.processing_end_calls == source.closed == 1

    input_path = tmp_path / "input.mp4"
    report_path = tmp_path / "report.json"
    input_path.write_bytes(b"fixture")
    report_path.write_text('{"previous": true}\n', encoding="utf-8")
    monkeypatch.setattr(
        replay_cli,
        "run_replay",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("observer end failed")),
    )
    assert replay_cli.main(["--source", str(input_path), "--report", str(report_path)]) == 2
    assert report_path.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_expected_cancellation_processing_end_observer_failure_is_runtime_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"stop": False}
    source = FakeSource([frame(0)])
    observer = ProcessingEndFailingObserver()

    class CancellingDetector(FakeDetector):
        def detect(self, value: Frame) -> tuple[Detection, ...]:
            state["stop"] = True
            return super().detect(value)

    with pytest.raises(RuntimeError, match="observer end failed"):
        vision_cli.run_vision(
            load_config("configs/development.yaml"),
            _vision_options(),
            source_factory=lambda _source: source,
            detector_factory=lambda *_args, **_kwargs: CancellingDetector([()]),
            stop_requested=lambda: state["stop"],
            observer=observer,
            emit=lambda _line: None,
        )
    assert observer.processing_end_calls == source.closed == 1

    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fixture")
    request = SimpleNamespace(signal_number=signal.SIGINT)

    class TerminationContext:
        def __enter__(self) -> SimpleNamespace:
            return request

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(replay_cli, "_termination_signals", TerminationContext)
    monkeypatch.setattr(
        replay_cli,
        "run_replay",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("observer end failed")),
    )
    assert replay_cli.main(["--source", str(input_path), "--report", str(tmp_path / "r.json")]) == 2


def test_processing_metrics_end_before_slow_cleanup(tmp_path: Path) -> None:
    clock = ManualClock()

    class TimedSource(FakeSource):
        def read(self) -> Frame | None:
            value = super().read()
            if value is not None:
                clock.advance(1.0)
            return value

        def close(self) -> None:
            clock.advance(100.0)
            super().close()

    source = TimedSource([frame(0), None])
    result = run_replay(
        load_config("configs/development.yaml"),
        options(tmp_path),
        source_factory=lambda _path: source,
        detector_factory=lambda *_args, **_kwargs: FakeDetector([()]),
        monotonic=clock,
        rate_monotonic=lambda: 1.0,
        sleep=lambda _seconds: None,
    )
    metrics = payload(result)["metrics"]

    assert result.report.processing_elapsed_s == 1.0
    assert metrics["approximate_processing_fps"] == 1.0  # type: ignore[index]
    assert metrics["effective_command_rate_hz"] == 1.0  # type: ignore[index]
    assert source.closed == 1


@pytest.mark.parametrize("outcome", ["max_frames", "cancelled", "failure"])
def test_processing_end_precedes_cleanup_for_non_eof_outcomes(
    outcome: str,
) -> None:
    clock = ManualClock()

    class SlowCloseSource(FakeSource):
        def close(self) -> None:
            clock.advance(100.0)
            super().close()

    source = SlowCloseSource([frame(0), None])
    observer = SnapshotObserver()
    detector: FakeDetector

    def stop_requested() -> bool:
        return False

    options_value = vision_cli.VisionOptions(
        source="fixture.mp4",
        model="local.pt",
        device="cpu",
        target_classes=("boat",),
        confidence_threshold=0.5,
        iou_threshold=0.45,
        image_size=320,
        max_frames=1 if outcome == "max_frames" else None,
        display=False,
        output=None,
    )
    if outcome == "failure":
        detector = FakeDetector([RuntimeError("detector failure")])
    else:
        detector = FakeDetector([()])
    if outcome == "cancelled":
        state = {"stop": False}

        class CancellingDetector(FakeDetector):
            def detect(self, value: Frame) -> tuple[Detection, ...]:
                state["stop"] = True
                return super().detect(value)

        detector = CancellingDetector([()])

        def stop_requested() -> bool:
            return state["stop"]

    with (
        pytest.raises(RuntimeError, match="detector failure")
        if outcome == "failure"
        else _does_not_raise()
    ):
        vision_cli.run_vision(
            load_config("configs/development.yaml"),
            options_value,
            source_factory=lambda _source: source,
            detector_factory=lambda *_args, **_kwargs: detector,
            monotonic=clock,
            rate_monotonic=lambda: 1.0,
            sleep=lambda _seconds: None,
            stop_requested=stop_requested,
            observer=observer,
            emit=lambda _line: None,
        )
    assert isinstance(observer.processing_end, vision_cli.RuntimeProcessingEndEvent)
    assert observer.processing_end.exit_reason == ("failure" if outcome == "failure" else outcome)
    assert observer.processing_end.processing_ended_s == 0.0
    assert source.closed == 1
    assert clock.value == 100.0


class _does_not_raise:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> bool:
        return False


@pytest.mark.parametrize("failure", ["write", "fsync", "replace"])
def test_atomic_replay_write_failures_preserve_existing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text('{"previous": true}\n', encoding="utf-8")
    if failure == "write":
        monkeypatch.setattr(
            benchmark,
            "_write_serialized_report",
            lambda *_args: (_ for _ in ()).throw(OSError("write failed")),
        )
    elif failure == "fsync":
        monkeypatch.setattr(
            benchmark.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("fsync failed"))
        )
    else:
        monkeypatch.setattr(
            benchmark.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace failed"))
        )

    with pytest.raises(OSError):
        run_fake(tmp_path, [frame(0), None], [()])
    assert report_path.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_symlink_equivalent_output_collision_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    alias = tmp_path / "source-alias.mp4"
    try:
        os.symlink(source, alias)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ReplayError, match="must differ"):
        options(tmp_path, source=source, report_path=alias)


def test_passive_observer_does_not_change_production_loop_result() -> None:
    def run(observer: object | None) -> tuple[int, float, float]:
        source = FakeSource([frame(0), None])
        actuator: list[object] = []
        processed = vision_cli.run_vision(
            load_config("configs/development.yaml"),
            vision_cli.VisionOptions(
                source="fixture.mp4",
                model="local.pt",
                device="cpu",
                target_classes=("boat",),
                confidence_threshold=0.5,
                iou_threshold=0.45,
                image_size=320,
                max_frames=None,
                display=False,
                output=None,
            ),
            source_factory=lambda _source: source,
            detector_factory=lambda *_args, **_kwargs: FakeDetector([(boat(),)]),
            actuator_factory=lambda config, _options: (
                actuator.append(
                    vision_cli.SimulatedPTZActuator(
                        config.actuator.pan_limits,
                        config.actuator.tilt_limits,
                        initial_pan_deg=config.actuator.initial_pan_deg,
                        initial_tilt_deg=config.actuator.initial_tilt_deg,
                    )
                )
                or actuator[0]
            ),
            observer=observer,
            emit=lambda _line: None,
        )
        selected = actuator[0]
        return processed, selected.pan_deg, selected.tilt_deg  # type: ignore[union-attr]

    assert run(None) == run(SnapshotObserver())


@pytest.mark.parametrize("writer_state", ["not_open", "write_error"])
def test_annotated_output_failure_releases_writer_and_preserves_report(
    tmp_path: Path,
    writer_state: str,
) -> None:
    class Writer:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return writer_state != "not_open"

        def write(self, _image: object) -> None:
            if writer_state == "write_error":
                raise RuntimeError("writer failed")

        def release(self) -> None:
            self.released = True

    class CV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 0
        MARKER_CROSS = 0

        def __init__(self, writer: Writer) -> None:
            self.writer = writer

        def VideoWriter_fourcc(self, *_values: str) -> int:
            return 1

        def VideoWriter(self, *_args: object) -> Writer:
            return self.writer

        def rectangle(self, *_args: object) -> None:
            pass

        def putText(self, *_args: object) -> None:
            pass

        def drawMarker(self, *_args: object) -> None:
            pass

        def line(self, *_args: object) -> None:
            pass

    writer = Writer()
    source = FakeSource([frame(0), None])
    source.cv2 = CV2(writer)  # type: ignore[attr-defined]
    report_path = tmp_path / "report.json"
    report_path.write_text('{"previous": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="unable to open|writer failed"):
        run_replay(
            load_config("configs/development.yaml"),
            options(tmp_path, annotated_output=tmp_path / "annotated.mp4"),
            source_factory=lambda _path: source,
            detector_factory=lambda *_args, **_kwargs: FakeDetector([(boat(),)]),
        )
    assert writer.released is True
    assert source.closed == 1
    assert report_path.read_text(encoding="utf-8") == '{"previous": true}\n'
