from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import marine_ptz.benchmark as benchmark
from marine_ptz.benchmark import (
    BenchmarkConfig,
    BenchmarkError,
    BenchmarkReport,
    percentile,
    run_benchmark,
    sanitize_command,
    summarize_samples,
    write_report,
)


@dataclass
class FakeDevice:
    resolved_device: str = "cpu"
    requires_synchronization: bool = False
    sync_calls: int = 0

    def synchronize(self) -> None:
        self.sync_calls += 1

    def environment(self) -> Mapping[str, object]:
        return {"torch_version": "fake", "cuda_version": "none"}


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[object]:
        self.calls.append(kwargs)
        return [SimpleNamespace(speed={"preprocess": 1.0, "inference": 5.0, "postprocess": 2.0})]


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class FakeImage:
    shape = (480, 640, 3)


def _completed_report() -> BenchmarkReport:
    return run_benchmark(
        BenchmarkConfig(warmup_iterations=0, measured_iterations=1),
        model_loader=lambda name: FakeModel(),
        input_loader=lambda settings: FakeImage(),
        device=FakeDevice(),
        clock=SequenceClock([0.0, 0.010, 0.020, 0.030, 0.040, 0.050]),
        timestamp=lambda: datetime(2026, 7, 24, tzinfo=timezone.utc),
        command=("marine-ptz-benchmark", "--device", "cpu"),
    )


def _temporary_reports(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*.tmp"))


def test_percentiles_and_summary_are_reproducible() -> None:
    summary = summarize_samples([1.0, 2.0, 3.0, 4.0])

    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert summary.mean_ms == 2.5
    assert summary.median_ms == 2.5
    assert summary.p95_ms == pytest.approx(3.85)
    assert summary.p99_ms == pytest.approx(3.97)
    assert summary.approximate_fps == 400.0


def test_single_sample_percentiles_and_zero_duration_are_defined() -> None:
    single = summarize_samples([7.0])
    zero = summarize_samples([0.0])

    assert single.p95_ms == 7.0
    assert single.p99_ms == 7.0
    assert single.standard_deviation_ms == 0.0
    assert zero.approximate_fps is None


@pytest.mark.parametrize(
    "samples",
    [[], [float("nan")], [float("inf")], [-0.1], [True]],
)
def test_invalid_timing_samples_are_rejected(samples: list[float]) -> None:
    with pytest.raises(BenchmarkError, match="timing samples|at least one"):
        summarize_samples(samples)


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("nan")])
def test_invalid_percentile_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(BenchmarkError, match="percentile"):
        percentile([1.0], fraction)


def test_cpu_measurement_never_calls_synchronize() -> None:
    model = FakeModel()
    device = FakeDevice()
    config = BenchmarkConfig(warmup_iterations=0, measured_iterations=1)
    clock = SequenceClock([0.0, 0.010, 0.020, 0.026, 0.030, 0.036])

    report = run_benchmark(
        config,
        model_loader=lambda name: model,
        input_loader=lambda settings: FakeImage(),
        device=device,
        clock=clock,
    )

    assert device.sync_calls == 0
    assert report.model_load_ms == 10.0
    assert report.first_inference_ms == pytest.approx(6.0)
    assert report.warm.mean_ms == pytest.approx(6.0)


def test_cuda_measurement_synchronizes_each_timed_section() -> None:
    model = FakeModel()
    device = FakeDevice(resolved_device="cuda:0", requires_synchronization=True)
    config = BenchmarkConfig(warmup_iterations=1, measured_iterations=2)
    clock = SequenceClock(
        [
            0.0,
            0.010,
            0.020,
            0.025,
            0.030,
            0.035,
            0.040,
            0.046,
            0.050,
            0.058,
        ]
    )

    report = run_benchmark(
        config,
        model_loader=lambda name: model,
        input_loader=lambda settings: FakeImage(),
        device=device,
        clock=clock,
    )

    assert device.sync_calls == 10
    assert report.first_inference_ms == pytest.approx(5.0)
    assert report.warm.mean_ms == pytest.approx(7.0)
    assert report.ultralytics_speed_ms == {
        "preprocess": 1.0,
        "inference": 5.0,
        "postprocess": 2.0,
    }


def test_cuda_synchronization_order_surrounds_warmup_and_measurements() -> None:
    events: list[str] = []

    class TraceDevice:
        resolved_device = "cuda:0"
        requires_synchronization = True

        def synchronize(self) -> None:
            events.append("sync")

        def environment(self) -> Mapping[str, object]:
            events.append("environment")
            return {}

    class TraceModel:
        def predict(self, **kwargs: object) -> list[object]:
            events.append("infer")
            return []

    clock_values = iter([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])

    def clock() -> float:
        events.append("clock")
        return next(clock_values)

    def load_model(name: str) -> TraceModel:
        events.append("load")
        return TraceModel()

    def load_input(config: BenchmarkConfig) -> FakeImage:
        events.append("input")
        return FakeImage()

    run_benchmark(
        BenchmarkConfig(warmup_iterations=1, measured_iterations=1),
        model_loader=load_model,
        input_loader=load_input,
        device=TraceDevice(),
        clock=clock,
    )

    assert events == [
        "sync",
        "clock",
        "load",
        "sync",
        "clock",
        "input",
        "sync",
        "clock",
        "infer",
        "sync",
        "clock",
        "sync",
        "clock",
        "infer",
        "sync",
        "clock",
        "sync",
        "clock",
        "infer",
        "sync",
        "clock",
        "environment",
    ]


def test_cold_first_and_warm_measurements_are_separate() -> None:
    model = FakeModel()
    config = BenchmarkConfig(warmup_iterations=1, measured_iterations=3)
    clock = SequenceClock(
        [
            0.0,
            0.100,
            0.200,
            0.220,
            0.300,
            0.330,
            0.400,
            0.440,
            0.500,
            0.550,
            0.600,
            0.660,
        ]
    )

    report = run_benchmark(
        config,
        model_loader=lambda name: model,
        input_loader=lambda settings: FakeImage(),
        device=FakeDevice(),
        clock=clock,
    )

    assert report.model_load_ms == 100.0
    assert report.first_inference_ms == pytest.approx(20.0)
    assert report.warm.mean_ms == pytest.approx(50.0)
    assert len(model.calls) == 5
    assert model.calls[0]["device"] == "cpu"


def test_json_output_schema_and_parent_directory_creation(tmp_path: Path) -> None:
    output = tmp_path / "generated" / "nested" / "benchmark.json"
    config = BenchmarkConfig(warmup_iterations=0, measured_iterations=1, output_path=output)
    clock = SequenceClock([0.0, 0.010, 0.020, 0.030, 0.040, 0.050])

    report = run_benchmark(
        config,
        model_loader=lambda name: FakeModel(),
        input_loader=lambda settings: FakeImage(),
        device=FakeDevice(),
        clock=clock,
        timestamp=lambda: datetime(2026, 7, 24, tzinfo=timezone.utc),
        command=("marine-ptz-benchmark", "--device", "cpu"),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["timestamp_utc"] == "2026-07-24T00:00:00+00:00"
    assert payload["configuration"]["resolved_device"] == "cpu"
    assert payload["input_resolution"] == {"width": 640, "height": 480}
    assert payload["metrics_ms"]["first_inference"] == pytest.approx(10.0)
    assert payload["environment"]["torch_version"] == "fake"
    assert payload["invocation"] == {
        "program": "marine-ptz-benchmark",
        "arguments": ["--device", "cpu"],
    }
    assert report.config.output_path == output


def test_atomic_report_replaces_existing_file_and_removes_temporary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "benchmark.json"
    output.write_bytes(b"previous report\n")

    write_report(_completed_report(), output)

    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
    assert _temporary_reports(output) == []


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        {"torch_version": {"nested": math.nan}},
        {"torch_version": {"nested": math.inf}},
        {"torch_version": {"nested": -math.inf}},
        {"torch_version": object()},
    ],
)
def test_invalid_json_metadata_preserves_existing_report(
    tmp_path: Path,
    invalid_metadata: Mapping[str, object],
) -> None:
    output = tmp_path / "benchmark.json"
    original = b"previous report\n"
    output.write_bytes(original)
    report = replace(_completed_report(), environment=invalid_metadata)

    with pytest.raises(BenchmarkError, match="finite JSON|unsupported JSON"):
        write_report(report, output)

    assert output.read_bytes() == original
    assert _temporary_reports(output) == []


def test_write_failure_preserves_existing_report_and_cleans_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "benchmark.json"
    original = b"previous report\n"
    output.write_bytes(original)

    def fail_write(stream: Any, serialized: str) -> None:
        stream.write("partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(benchmark, "_write_serialized_report", fail_write)

    with pytest.raises(OSError, match="simulated write failure"):
        write_report(_completed_report(), output)

    assert output.read_bytes() == original
    assert _temporary_reports(output) == []


def test_fsync_failure_preserves_existing_report_and_cleans_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "benchmark.json"
    original = b"previous report\n"
    output.write_bytes(original)

    def fail_fsync(descriptor: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(benchmark.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="simulated fsync failure"):
        write_report(_completed_report(), output)

    assert output.read_bytes() == original
    assert _temporary_reports(output) == []


def test_replace_failure_preserves_existing_report_and_cleans_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "benchmark.json"
    original = b"previous report\n"
    output.write_bytes(original)

    def fail_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(benchmark.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        write_report(_completed_report(), output)

    assert output.read_bytes() == original
    assert _temporary_reports(output) == []


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            ("benchmark", "https://user:password@example.com/model.pt"),
            ("benchmark", "https://<redacted>@example.com/model.pt"),
        ),
        (
            ("benchmark", "https://example.com/model.pt?token=secret&size=640"),
            (
                "benchmark",
                "https://example.com/model.pt?token=%3Credacted%3E&size=640",
            ),
        ),
        (
            ("benchmark", "--API-KEY=secret"),
            ("benchmark", "--API-KEY=<redacted>"),
        ),
        (
            ("benchmark", "--password", "secret"),
            ("benchmark", "--password", "<redacted>"),
        ),
        (
            ("benchmark", "--model", "/models/local.pt", "--device=cpu"),
            ("benchmark", "--model", "/models/local.pt", "--device=cpu"),
        ),
    ],
)
def test_command_sanitization(
    command: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    assert sanitize_command(command) == expected


def test_structured_model_configuration_is_sanitized() -> None:
    report = replace(
        _completed_report(),
        config=BenchmarkConfig(
            model="https://user:password@example.com/model.pt?token=secret"
        ),
    )

    assert report.to_dict()["configuration"]["model"] == (
        "https://<redacted>@example.com/model.pt?token=%3Credacted%3E"
    )


@pytest.mark.parametrize(
    "name",
    [
        "token",
        "access_token",
        "api_key",
        "key",
        "password",
        "secret",
        "signature",
        "credential",
    ],
)
def test_sensitive_query_parameter_names_are_redacted(name: str) -> None:
    sanitized = sanitize_command(
        ("benchmark", f"https://example.com/model.pt?{name.upper()}=secret")
    )

    assert "secret" not in sanitized[1]
    assert "%3Credacted%3E" in sanitized[1]


def test_environment_metadata_uses_an_explicit_allowlist() -> None:
    report = replace(
        _completed_report(),
        environment={
            "torch_version": "2.11.0+cu128",
            "AWS_SECRET_ACCESS_KEY": "must-not-be-recorded",
        },
    )

    assert report.to_dict()["environment"] == {
        "torch_version": "2.11.0+cu128"
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup_iterations": -1},
        {"measured_iterations": 0},
        {"measured_iterations": -1},
        {"measured_iterations": True},
        {"image_size": 0},
        {"confidence_threshold": math.nan},
        {"iou_threshold": 1.1},
    ],
)
def test_benchmark_configuration_rejects_nonsensical_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(BenchmarkError):
        BenchmarkConfig(**kwargs)


def test_cli_validation_fails_before_optional_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "_default_components",
        lambda config: pytest.fail("optional components should not be constructed"),
    )

    with pytest.raises(SystemExit) as error:
        benchmark.main(["--iterations", "0"])

    assert error.value.code == 2


def test_cli_uses_fake_model_without_model_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    loaded_models: list[str] = []

    def fake_components(config: BenchmarkConfig) -> tuple[Any, Any, FakeDevice]:
        return (
            lambda name: loaded_models.append(name) or model,
            lambda settings: FakeImage(),
            FakeDevice(),
        )

    monkeypatch.setattr(benchmark, "_default_components", fake_components)
    output = tmp_path / "benchmark.json"

    assert (
        benchmark.main(
            [
                "--model",
                "local-model.pt",
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert loaded_models == ["local-model.pt"]
    assert output.exists()
    assert len(model.calls) == 2


def test_model_load_failure_does_not_create_or_replace_report(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    original = b"previous report\n"
    output.write_bytes(original)
    config = BenchmarkConfig(
        warmup_iterations=0,
        measured_iterations=1,
        output_path=output,
    )

    def fail_model_load(name: str) -> object:
        raise BenchmarkError("model load failed")

    with pytest.raises(BenchmarkError, match="model load failed"):
        run_benchmark(
            config,
            model_loader=fail_model_load,
            input_loader=lambda settings: FakeImage(),
            device=FakeDevice(),
            clock=SequenceClock([0.0]),
        )

    assert output.read_bytes() == original
    assert _temporary_reports(output) == []


def test_first_inference_failure_does_not_create_or_replace_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "benchmark.json"
    original = b"previous report\n"
    output.write_bytes(original)
    config = BenchmarkConfig(
        warmup_iterations=0,
        measured_iterations=1,
        output_path=output,
    )

    class FailingModel:
        def predict(self, **kwargs: object) -> list[object]:
            raise BenchmarkError("first inference failed")

    with pytest.raises(BenchmarkError, match="first inference failed"):
        run_benchmark(
            config,
            model_loader=lambda name: FailingModel(),
            input_loader=lambda settings: FakeImage(),
            device=FakeDevice(),
            clock=SequenceClock([0.0, 0.01, 0.02]),
        )

    assert output.read_bytes() == original
    assert _temporary_reports(output) == []


@pytest.mark.parametrize("image_name", ["missing.jpg", "corrupt.jpg"])
def test_missing_or_corrupt_local_image_fails_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_name: str,
) -> None:
    image_path = tmp_path / image_name
    if image_name == "corrupt.jpg":
        image_path.write_bytes(b"not an image")
    fake_modules = {
        "torch": SimpleNamespace(),
        "ultralytics": SimpleNamespace(YOLO=lambda model: FakeModel()),
        "cv2": SimpleNamespace(imread=lambda path: None),
        "numpy": SimpleNamespace(),
    }
    monkeypatch.setattr(
        benchmark,
        "_optional_module",
        lambda name: fake_modules[name],
    )
    monkeypatch.setattr(
        benchmark,
        "resolve_inference_device",
        lambda device, torch_module: "cpu",
    )
    config = BenchmarkConfig(image_path=image_path)
    _, input_loader, _ = benchmark._default_components(config)

    with pytest.raises(BenchmarkError, match="unable to read benchmark image"):
        input_loader(config)


def test_default_output_path_is_ignored_generated_artifact() -> None:
    assert benchmark._default_output_path().parent == Path("artifacts/benchmarks")
