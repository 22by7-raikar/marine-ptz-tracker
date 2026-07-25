"""Reproducible, camera-free Ultralytics inference benchmarking."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, TextIO
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .yolo_detector import InferenceDeviceError, resolve_inference_device


_ENVIRONMENT_METADATA_KEYS = frozenset(
    {
        "torch_version",
        "cuda_version",
        "ultralytics_version",
        "opencv_version",
        "gpu_name",
        "gpu_compute_capability",
    }
)
_SENSITIVE_ARGUMENT_NAMES = frozenset(
    {
        "token",
        "access_token",
        "api_key",
        "key",
        "password",
        "secret",
        "signature",
        "credential",
    }
)
_REDACTED = "<redacted>"


class BenchmarkError(ValueError):
    """Raised when benchmark configuration or samples are invalid."""


class BenchmarkDependencyError(RuntimeError):
    """Raised when optional vision dependencies are unavailable."""


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Validated settings for one model-inference benchmark."""

    model: str = "yolo11n.pt"
    device: str = "auto"
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    warmup_iterations: int = 5
    measured_iterations: int = 30
    image_path: Path | None = None
    output_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise BenchmarkError("model must be a non-empty name or path")
        if not isinstance(self.device, str) or not self.device.strip():
            raise BenchmarkError("device must be a non-empty string")
        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int):
            raise BenchmarkError("image_size must be a positive integer")
        if self.image_size <= 0:
            raise BenchmarkError("image_size must be a positive integer")
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("iou_threshold", self.iou_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BenchmarkError(f"{name} must be a finite number between 0 and 1")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise BenchmarkError(f"{name} must be a finite number between 0 and 1")
        if isinstance(self.warmup_iterations, bool) or not isinstance(
            self.warmup_iterations, int
        ):
            raise BenchmarkError("warmup_iterations must be a non-negative integer")
        if self.warmup_iterations < 0:
            raise BenchmarkError("warmup_iterations must be a non-negative integer")
        if isinstance(self.measured_iterations, bool) or not isinstance(
            self.measured_iterations, int
        ):
            raise BenchmarkError("measured_iterations must be a positive integer")
        if self.measured_iterations <= 0:
            raise BenchmarkError("measured_iterations must be a positive integer")


@dataclass(frozen=True, slots=True)
class SampleStatistics:
    """Summary statistics for millisecond timings."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    standard_deviation_ms: float
    approximate_fps: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "standard_deviation_ms": self.standard_deviation_ms,
            "approximate_fps": self.approximate_fps,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """JSON-serializable result for one benchmark execution."""

    timestamp_utc: str
    command: tuple[str, ...]
    config: BenchmarkConfig
    resolved_device: str
    input_resolution: tuple[int, int]
    model_load_ms: float
    first_inference_ms: float
    warm: SampleStatistics
    ultralytics_speed_ms: Mapping[str, float]
    environment: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        sanitized_command = sanitize_command(self.command)
        payload: dict[str, object] = {
            "schema_version": 1,
            "timestamp_utc": self.timestamp_utc,
            "invocation": {
                "program": sanitized_command[0] if sanitized_command else None,
                "arguments": list(sanitized_command[1:]),
            },
            "configuration": {
                "model": _sanitize_url(self.config.model),
                "requested_device": self.config.device,
                "resolved_device": self.resolved_device,
                "image_size": self.config.image_size,
                "confidence_threshold": self.config.confidence_threshold,
                "iou_threshold": self.config.iou_threshold,
                "warmup_iterations": self.config.warmup_iterations,
                "measured_iterations": self.config.measured_iterations,
                "image_path": None if self.config.image_path is None else str(self.config.image_path),
            },
            "input_resolution": {
                "width": self.input_resolution[0],
                "height": self.input_resolution[1],
            },
            "metrics_ms": {
                "model_load": self.model_load_ms,
                "first_inference": self.first_inference_ms,
                "warm": self.warm.to_dict(),
                "ultralytics_speed": dict(self.ultralytics_speed_ms),
            },
            "environment": {
                name: self.environment[name]
                for name in sorted(_ENVIRONMENT_METADATA_KEYS)
                if name in self.environment
            },
        }
        return _json_compatible(payload)


class DeviceAdapter(Protocol):
    """Minimal runtime boundary for deterministic CPU/CUDA timing tests."""

    resolved_device: str
    requires_synchronization: bool

    def synchronize(self) -> None:
        """Synchronize only when the selected device requires it."""

    def environment(self) -> Mapping[str, object]:
        """Describe the active benchmark runtime."""


@dataclass(slots=True)
class TorchDeviceAdapter:
    """Device timing and metadata adapter backed by an imported Torch module."""

    torch: Any
    ultralytics: Any
    cv2: Any
    resolved_device: str

    @property
    def requires_synchronization(self) -> bool:
        return self.resolved_device.startswith("cuda")

    def synchronize(self) -> None:
        if self.resolved_device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def environment(self) -> Mapping[str, object]:
        details: dict[str, object] = {
            "torch_version": str(getattr(self.torch, "__version__", "unknown")),
            "cuda_version": str(getattr(getattr(self.torch, "version", None), "cuda", "unknown")),
            "ultralytics_version": str(getattr(self.ultralytics, "__version__", "unknown")),
            "opencv_version": str(getattr(self.cv2, "__version__", "unknown")),
        }
        if self.resolved_device.startswith("cuda"):
            index = _cuda_index(self.resolved_device)
            details["gpu_name"] = str(self.torch.cuda.get_device_name(index))
            capability = self.torch.cuda.get_device_capability(index)
            details["gpu_compute_capability"] = f"{capability[0]}.{capability[1]}"
        return details


def percentile(samples: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile from finite timing samples."""
    values = _validated_samples(samples)
    if not isinstance(fraction, (int, float)) or not math.isfinite(float(fraction)):
        raise BenchmarkError("percentile fraction must be finite and between 0 and 1")
    if not 0.0 <= float(fraction) <= 1.0:
        raise BenchmarkError("percentile fraction must be finite and between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(fraction)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(samples: Sequence[float]) -> SampleStatistics:
    """Summarize non-negative millisecond samples."""
    values = _validated_samples(samples)
    mean = statistics.fmean(values)
    return SampleStatistics(
        mean_ms=mean,
        median_ms=statistics.median(values),
        p95_ms=percentile(values, 0.95),
        p99_ms=percentile(values, 0.99),
        standard_deviation_ms=statistics.pstdev(values),
        approximate_fps=None if mean == 0.0 else 1000.0 / mean,
    )


def run_benchmark(
    config: BenchmarkConfig,
    *,
    model_loader: Callable[[str], Any],
    input_loader: Callable[[BenchmarkConfig], object],
    device: DeviceAdapter,
    clock: Callable[[], float] = time.perf_counter,
    timestamp: Callable[[], datetime] | None = None,
    command: Sequence[str] = (),
) -> BenchmarkReport:
    """Measure cold, first, warm-up, and steady-state model inference timings."""
    model, model_load_ms = _measure(lambda: model_loader(config.model), clock, device)
    image = input_loader(config)
    input_resolution = _image_resolution(image, config.image_size)

    def infer() -> tuple[Any, ...]:
        results = model.predict(
            source=image,
            conf=config.confidence_threshold,
            iou=config.iou_threshold,
            imgsz=config.image_size,
            device=device.resolved_device,
            verbose=False,
        )
        return tuple(results)

    first_results, first_inference_ms = _measure(infer, clock, device)
    for _ in range(config.warmup_iterations):
        _measure(infer, clock, device)

    samples: list[float] = []
    speed_samples: list[Mapping[str, float]] = []
    for _ in range(config.measured_iterations):
        results, elapsed_ms = _measure(infer, clock, device)
        samples.append(elapsed_ms)
        speed_samples.extend(_result_speeds(results))

    report = BenchmarkReport(
        timestamp_utc=(timestamp or _utc_now)().isoformat(),
        command=tuple(command),
        config=config,
        resolved_device=device.resolved_device,
        input_resolution=input_resolution,
        model_load_ms=model_load_ms,
        first_inference_ms=first_inference_ms,
        warm=summarize_samples(samples),
        ultralytics_speed_ms=_mean_speeds(speed_samples),
        environment=device.environment(),
    )
    if config.output_path is not None:
        write_report(report, config.output_path)
    return report


def write_report(report: BenchmarkReport, path: Path) -> None:
    """Atomically replace a report with strict JSON written on the same filesystem."""
    serialized = json.dumps(
        report.to_dict(),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            _write_serialized_report(stream, serialized)
        os.replace(temporary_path, path)
        temporary_path = None
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def sanitize_command(command: Sequence[str]) -> tuple[str, ...]:
    """Return a shareable invocation with sensitive argument values redacted."""
    sanitized: list[str] = []
    redact_next = False
    for raw_argument in command:
        argument = str(raw_argument)
        if redact_next:
            sanitized.append(_REDACTED)
            redact_next = False
            continue

        sanitized_url = _sanitize_url(argument)
        if sanitized_url != argument:
            sanitized.append(sanitized_url)
            continue

        if "=" in argument:
            name, value = argument.split("=", 1)
            if _is_sensitive_name(name):
                sanitized.append(f"{name}={_REDACTED}")
            else:
                sanitized.append(f"{name}={_sanitize_url(value)}")
            continue

        sanitized.append(_sanitize_url(argument))
        if _is_sensitive_name(argument):
            redact_next = True
    return tuple(sanitized)


def format_report(report: BenchmarkReport) -> str:
    """Format concise human-readable benchmark telemetry."""
    warm = report.warm
    return "\n".join(
        (
            f"model={report.config.model} device={report.resolved_device} "
            f"input={report.input_resolution[0]}x{report.input_resolution[1]}",
            f"model_load_ms={report.model_load_ms:.2f} first_inference_ms={report.first_inference_ms:.2f}",
            f"warm_mean_ms={warm.mean_ms:.2f} median_ms={warm.median_ms:.2f} "
            f"p95_ms={warm.p95_ms:.2f} p99_ms={warm.p99_ms:.2f} "
            f"stddev_ms={warm.standard_deviation_ms:.2f} "
            f"fps={'n/a' if warm.approximate_fps is None else f'{warm.approximate_fps:.2f}'}",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optional-dependency benchmark CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = args.output or _default_output_path()
        config = BenchmarkConfig(
            model=args.model,
            device=args.device,
            image_size=args.image_size,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            warmup_iterations=args.warmup,
            measured_iterations=args.iterations,
            image_path=args.image,
            output_path=output,
        )
        model_loader, input_loader, adapter = _default_components(config)
        report = run_benchmark(
            config,
            model_loader=model_loader,
            input_loader=input_loader,
            device=adapter,
            command=("marine-ptz-benchmark", *(argv if argv is not None else sys.argv[1:])),
        )
    except (BenchmarkError, BenchmarkDependencyError, InferenceDeviceError, OSError) as exc:
        parser.error(str(exc))
    print(format_report(report))
    print(f"json_output={config.output_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model name or local path")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=5, help="unmeasured inference iterations")
    parser.add_argument("--iterations", type=int, default=30, help="measured inference iterations")
    parser.add_argument("--image", type=Path, help="optional local image instead of a blank frame")
    parser.add_argument("--output", type=Path, help="JSON path; defaults under artifacts/benchmarks")
    return parser


def _default_components(
    config: BenchmarkConfig,
) -> tuple[Callable[[str], Any], Callable[[BenchmarkConfig], object], TorchDeviceAdapter]:
    torch = _optional_module("torch")
    ultralytics = _optional_module("ultralytics")
    cv2 = _optional_module("cv2")
    numpy = _optional_module("numpy")
    resolved_device = resolve_inference_device(config.device, torch_module=torch)
    adapter = TorchDeviceAdapter(torch, ultralytics, cv2, resolved_device)

    def load_model(model: str) -> Any:
        try:
            return ultralytics.YOLO(model)
        except Exception as exc:
            raise BenchmarkError(f"unable to load model {model!r}: {exc}") from exc

    def load_input(settings: BenchmarkConfig) -> object:
        if settings.image_path is not None:
            image = cv2.imread(str(settings.image_path))
            if image is None:
                raise BenchmarkError(f"unable to read benchmark image {settings.image_path}")
            return image
        return numpy.zeros((settings.image_size, settings.image_size, 3), dtype=numpy.uint8)

    return load_model, load_input, adapter


def _write_serialized_report(stream: TextIO, serialized: str) -> None:
    stream.write(serialized)
    stream.flush()
    os.fsync(stream.fileno())


def _json_compatible(
    value: object,
    *,
    path: str = "$",
    active_containers: set[int] | None = None,
) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BenchmarkError(f"{path} must contain only finite JSON numbers")
        return value

    active = active_containers if active_containers is not None else set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise BenchmarkError(f"{path} contains a recursive JSON container")
        active.add(identity)
        try:
            normalized: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise BenchmarkError(f"{path} contains a non-string JSON object key")
                normalized[key] = _json_compatible(
                    item,
                    path=f"{path}.{key}",
                    active_containers=active,
                )
            return normalized
        finally:
            active.remove(identity)

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise BenchmarkError(f"{path} contains a recursive JSON container")
        active.add(identity)
        try:
            return [
                _json_compatible(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)

    raise BenchmarkError(
        f"{path} contains unsupported JSON value of type {type(value).__name__}"
    )


def _is_sensitive_name(name: str) -> bool:
    normalized = name.lstrip("-").replace("-", "_").casefold()
    return normalized in _SENSITIVE_ARGUMENT_NAMES


def _sanitize_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or not parsed.netloc:
        return value

    netloc = parsed.netloc
    if "@" in netloc:
        netloc = f"{_REDACTED}@{netloc.rsplit('@', 1)[1]}"

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    sanitized_query = urlencode(
        [
            (name, _REDACTED if _is_sensitive_name(name) else item)
            for name, item in query_pairs
        ]
    )
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, sanitized_query, parsed.fragment)
    )


def _optional_module(name: str) -> Any:
    try:
        return import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise BenchmarkDependencyError(
            f"{name} is unavailable; install the constrained vision dependencies"
        ) from exc


def _measure(
    callback: Callable[[], Any],
    clock: Callable[[], float],
    device: DeviceAdapter,
) -> tuple[Any, float]:
    if device.requires_synchronization:
        device.synchronize()
    started = clock()
    value = callback()
    if device.requires_synchronization:
        device.synchronize()
    elapsed_ms = (clock() - started) * 1000.0
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
        raise BenchmarkError("timing clock produced an invalid elapsed duration")
    return value, elapsed_ms


def _validated_samples(samples: Sequence[float]) -> list[float]:
    if not samples:
        raise BenchmarkError("at least one timing sample is required")
    values: list[float] = []
    for sample in samples:
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            raise BenchmarkError("timing samples must be finite non-negative numbers")
        value = float(sample)
        if not math.isfinite(value) or value < 0.0:
            raise BenchmarkError("timing samples must be finite non-negative numbers")
        values.append(value)
    return values


def _result_speeds(results: Sequence[Any]) -> list[Mapping[str, float]]:
    speeds: list[Mapping[str, float]] = []
    for result in results:
        raw = getattr(result, "speed", None)
        if not isinstance(raw, Mapping):
            continue
        parsed: dict[str, float] = {}
        for name in ("preprocess", "inference", "postprocess"):
            value = raw.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                parsed[name] = float(value)
        if parsed:
            speeds.append(parsed)
    return speeds


def _mean_speeds(samples: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        name: statistics.fmean(sample[name] for sample in samples if name in sample)
        for name in ("preprocess", "inference", "postprocess")
        if any(name in sample for sample in samples)
    }


def _image_resolution(image: object, fallback_size: int) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 2 and all(
        isinstance(value, int) and value > 0 for value in shape[:2]
    ):
        return shape[1], shape[0]
    return fallback_size, fallback_size


def _cuda_index(device: str) -> int:
    return int(device.split(":", 1)[1]) if ":" in device else 0


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts/benchmarks") / f"yolo-benchmark-{timestamp}.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())
