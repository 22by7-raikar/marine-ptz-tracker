"""Compare single and concurrent production runtimes on one local video."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .benchmark import local_path_marker, sanitize_command, write_report
from .concurrent_runtime import (
    ConcurrentRuntimeMetrics,
    MonotonicPlaybackSchedule,
    RuntimeMetricsCollector,
)
from .config import AppConfig, ConfigError, load_config
from .opencv_source import OpenCVSource
from .vision_cli import (
    RuntimeCompleteEvent,
    RuntimeFrameEvent,
    RuntimeObserver,
    RuntimeProcessingEndEvent,
    RuntimeStartEvent,
    VisionOptions,
    _target_classes,
    run_vision,
)

RUNTIME_BENCHMARK_SCHEMA_VERSION = 4


class RuntimeBenchmarkError(ValueError):
    """Raised for invalid local runtime-comparison inputs."""


@dataclass(frozen=True, slots=True)
class RuntimeBenchmarkOptions:
    """Validated local-video settings shared by both runtime modes."""

    source: Path
    model: str
    device: str
    target_classes: tuple[str, ...]
    confidence_threshold: float
    iou_threshold: float
    image_size: int
    max_frames: int | None
    report_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.source, Path) or not str(self.source):
            raise RuntimeBenchmarkError("source must be a local filesystem path")
        if not isinstance(self.report_path, Path) or not str(self.report_path):
            raise RuntimeBenchmarkError("report must be a filesystem path")
        if self.source.resolve(strict=False) == self.report_path.resolve(strict=False):
            raise RuntimeBenchmarkError("source and report paths must differ")
        if not isinstance(self.model, str) or not self.model.strip():
            raise RuntimeBenchmarkError("model must be a non-empty name or local path")
        if not isinstance(self.device, str) or not self.device.strip():
            raise RuntimeBenchmarkError("device must be a non-empty string")
        if not self.target_classes or any(
            not isinstance(value, str) or not value.strip() for value in self.target_classes
        ):
            raise RuntimeBenchmarkError("at least one non-empty target class is required")
        for name, value in (
            ("confidence", self.confidence_threshold),
            ("IoU", self.iou_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise RuntimeBenchmarkError(f"{name} must be a finite number between 0 and 1")
        if (
            isinstance(self.image_size, bool)
            or not isinstance(self.image_size, int)
            or self.image_size <= 0
        ):
            raise RuntimeBenchmarkError("image size must be a positive integer")
        if self.max_frames is not None and (
            isinstance(self.max_frames, bool)
            or not isinstance(self.max_frames, int)
            or self.max_frames <= 0
        ):
            raise RuntimeBenchmarkError("max frames must be a positive integer when provided")


@dataclass(frozen=True, slots=True)
class RuntimeModeResult:
    """Comparable measurements for one execution topology."""

    mode: str
    processed_frames: int
    metrics: ConcurrentRuntimeMetrics

    def to_dict(self) -> dict[str, object]:
        frame_age = self.metrics.capture_to_command_ms
        return {
            "mode": self.mode,
            "processed_frames": self.processed_frames,
            "source_kind": self.metrics.source_type,
            "declared_source_fps": self.metrics.declared_source_fps,
            "pacing_applied": self.metrics.pacing_applied,
            "capture_outcome": self.metrics.capture_outcome,
            "detector_readiness_ms": self.metrics.detector_readiness_ms,
            "captured_frames": self.metrics.capture.count,
            "inferred_frames": self.metrics.unique_inference.count,
            "displayed_frames": self.metrics.display.count,
            "unique_inference_fps": self.metrics.unique_inference.to_dict(),
            "display_fps": self.metrics.display.to_dict(),
            "capture_to_command_age_ms": frame_age.to_dict(),
            "dropped_capture_frames": self.metrics.dropped_capture_frames,
            "dropped_render_results": self.metrics.dropped_render_frames,
            "control_hz": self.metrics.actuator_commands.to_dict(),
            "recorder_submitted_unique_frames": self.metrics.recorder_submitted_frames,
            "recorder_consumed_unique_frames": self.metrics.recorder_consumed_frames,
            "dropped_recorder_frames": self.metrics.dropped_recorder_frames,
            "encoded_frames": self.metrics.encoded_output.count,
            "encoded_output_fps": self.metrics.encoded_output.to_dict(),
            "encoded_duplicate_frames": self.metrics.encoded_duplicate_frames,
            "encoded_duplicate_ratio": self.metrics.encoded_duplicate_ratio,
            "shutdown_duration_ms": self.metrics.shutdown_duration_ms,
        }


@dataclass(frozen=True, slots=True)
class RuntimeBenchmarkReport:
    """Strict JSON report comparing both modes on the same local video."""

    source: Path
    source_playback_fps: float
    options: RuntimeBenchmarkOptions
    results: tuple[RuntimeModeResult, RuntimeModeResult]

    def to_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": RUNTIME_BENCHMARK_SCHEMA_VERSION,
            "source": local_path_marker(str(self.source)),
            "source_playback_fps": self.source_playback_fps,
            "actuator_backend": "simulated",
            "configuration": {
                "model": sanitize_command(("runtime-benchmark", "--model", self.options.model))[2],
                "device": self.options.device,
                "target_classes": list(self.options.target_classes),
                "confidence_threshold": self.options.confidence_threshold,
                "iou_threshold": self.options.iou_threshold,
                "image_size": self.options.image_size,
                "max_frames": self.options.max_frames,
            },
            "results": [result.to_dict() for result in self.results],
        }


class _PacedVideoSource:
    """Pace finite-media reads so a local file models a live camera rate."""

    def __init__(
        self,
        source: int | str,
        fps: float,
        *,
        source_factory: Callable[..., Any] = OpenCVSource,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if isinstance(source, int):
            raise RuntimeBenchmarkError("runtime benchmark does not accept camera indexes")
        self._source = source_factory(source)
        self._fps = fps
        self._monotonic = monotonic
        self._sleep = sleep
        self._schedule: MonotonicPlaybackSchedule | None = None

    @property
    def cv2(self) -> Any:
        return self._source.cv2

    @property
    def is_live(self) -> bool:
        return False

    @property
    def is_image(self) -> bool:
        return bool(getattr(self._source, "is_image", False))

    @property
    def total_frames(self) -> int | None:
        return getattr(self._source, "total_frames", None)

    def read(self) -> Any:
        if self._schedule is not None:
            while True:
                now = self._monotonic()
                remaining = self._schedule.wait_seconds(now)
                if remaining <= 0.0:
                    break
                self._sleep(remaining)
            self._schedule.frame_started(now)
        frame = self._source.read()
        if frame is not None and self._schedule is None:
            self._schedule = MonotonicPlaybackSchedule(
                self._fps,
                started_s=self._monotonic(),
            )
        return frame

    def close(self) -> None:
        self._source.close()


class _SingleRuntimeMetrics(RuntimeObserver):
    def __init__(self, source_fps: float) -> None:
        self.collector = RuntimeMetricsCollector()
        self.processing_ended_s: float | None = None
        self.source_type = "unknown"
        self.source_fps = source_fps
        self.capture_outcome = "unknown"

    def on_start(self, event: RuntimeStartEvent) -> None:
        if event.actuator_backend != "simulated":
            raise RuntimeError("runtime benchmark requires simulated actuation")
        self.source_type = event.source_type

    def on_frame(self, event: RuntimeFrameEvent) -> None:
        age_s = event.processing_timestamp_s - event.frame_capture_timestamp_s
        if age_s < 0.0:
            raise RuntimeError("runtime benchmark clocks are not comparable")
        self.collector.record_event("capture", event.frame_capture_timestamp_s)
        self.collector.record_event("inference", event.processing_timestamp_s)
        self.collector.record_event("display", event.processing_timestamp_s)
        self.collector.record_event("command", event.processing_timestamp_s)
        self.collector.record_latency("capture_to_command", age_s)

    def on_processing_end(self, event: RuntimeProcessingEndEvent) -> None:
        self.processing_ended_s = event.processing_ended_s
        self.capture_outcome = "normal_eof" if event.exit_reason == "eof" else "stopped"

    def on_complete(self, event: RuntimeCompleteEvent) -> None:
        if self.processing_ended_s is None:
            raise RuntimeError("single runtime completed without a processing-end event")

    def snapshot(self, completed_s: float) -> ConcurrentRuntimeMetrics:
        if self.processing_ended_s is None:
            raise RuntimeError("single runtime did not report its processing-end boundary")
        return self.collector.snapshot(
            dropped_capture_frames=0,
            dropped_render_frames=0,
            recorder_submitted_frames=0,
            recorder_consumed_frames=0,
            dropped_recorder_frames=0,
            encoded_frame_count=0,
            encoded_duplicate_frames=0,
            encoded_started_s=None,
            encoded_finished_s=None,
            shutdown_duration_s=max(0.0, completed_s - self.processing_ended_s),
            source_type=self.source_type,
            declared_source_fps=self.source_fps,
            pacing_applied=True,
            capture_outcome=self.capture_outcome,
        )


def run_runtime_benchmark(
    config: AppConfig,
    options: RuntimeBenchmarkOptions,
    *,
    runner: Callable[..., int] = run_vision,
    clock: Callable[[], float] = time.monotonic,
    source_factory: Callable[..., Any] | None = None,
) -> RuntimeBenchmarkReport:
    """Run the same finite source once per mode with simulated actuation."""
    base = VisionOptions(
        source=str(options.source),
        model=options.model,
        device=options.device,
        target_classes=options.target_classes,
        confidence_threshold=options.confidence_threshold,
        iou_threshold=options.iou_threshold,
        image_size=options.image_size,
        max_frames=options.max_frames,
        display=False,
        output=None,
        actuator_backend="simulated",
        runtime_mode="single",
    )

    single_metrics = _SingleRuntimeMetrics(config.camera.fps)
    supplied_source_factory = source_factory
    if supplied_source_factory is None:

        def single_source_factory(source: int | str) -> _PacedVideoSource:
            return _PacedVideoSource(source, config.camera.fps)

        concurrent_source_factory: Callable[..., Any] = OpenCVSource
    else:
        single_source_factory = supplied_source_factory
        concurrent_source_factory = supplied_source_factory

    single_count = runner(
        config,
        base,
        observer=single_metrics,
        emit=lambda _line: None,
        source_factory=single_source_factory,
    )
    single_completed_s = clock()
    concurrent_reports: list[ConcurrentRuntimeMetrics] = []
    concurrent_count = runner(
        config,
        replace(base, runtime_mode="concurrent"),
        concurrent_metrics_sink=concurrent_reports.append,
        emit=lambda _line: None,
        source_factory=concurrent_source_factory,
    )
    if len(concurrent_reports) != 1:
        raise RuntimeError("concurrent runtime did not produce exactly one metrics report")

    report = RuntimeBenchmarkReport(
        source=options.source,
        source_playback_fps=config.camera.fps,
        options=options,
        results=(
            RuntimeModeResult(
                "single",
                single_count,
                single_metrics.snapshot(single_completed_s),
            ),
            RuntimeModeResult("concurrent", concurrent_count, concurrent_reports[0]),
        ),
    )
    write_report(report, options.report_path)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/development.yaml"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--device")
    parser.add_argument("--target-class", action="append")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if not args.source.is_file():
            raise RuntimeBenchmarkError(
                f"runtime benchmark source is not a local file: {args.source}"
            )
        if args.report.exists() and args.report.is_dir():
            raise RuntimeBenchmarkError("runtime benchmark report path must name a file")
        selected = RuntimeBenchmarkOptions(
            source=args.source,
            model=args.model or config.detection.model,
            device=args.device or config.detection.device,
            target_classes=_target_classes(args.target_class, config.detection.target_labels),
            confidence_threshold=(
                config.detection.confidence_threshold
                if args.confidence is None
                else args.confidence
            ),
            iou_threshold=(config.detection.iou_threshold if args.iou is None else args.iou),
            image_size=(
                config.detection.image_size if args.image_size is None else args.image_size
            ),
            max_frames=args.max_frames,
            report_path=args.report,
        )
        if not Path(selected.model).is_file():
            print(
                "warning: use a verified local model path to avoid an Ultralytics download",
                file=sys.stderr,
            )
        run_runtime_benchmark(config, selected)
    except (ConfigError, RuntimeBenchmarkError, OSError, RuntimeError, ValueError) as error:
        print(f"runtime benchmark error: {error}", file=sys.stderr)
        return 2
    print(f"runtime benchmark report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
