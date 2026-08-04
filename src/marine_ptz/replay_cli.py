"""Run finite local media through the production loop and write acceptance evidence."""

from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

from .benchmark import environment_metadata
from .config import ConfigError, load_config
from .opencv_source import OpenCVSourceError
from .replay import AcceptanceThresholds, ReplayCancelled, ReplayError, ReplayOptions, run_replay
from .vision_cli import _target_classes, _termination_signals
from .yolo_detector import InferenceDeviceError, UltralyticsDetectorError

ACCEPTANCE_FAILURE_STATUS = 3


def main(argv: Sequence[str] | None = None) -> int:
    """Run one finite-media replay; never construct physical actuation."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        _validate_hardware_rejection(args, config.actuator.backend)
        source = _local_source(args.source)
        _validate_output_path(args.report, "report")
        if args.annotated_output is not None:
            _validate_output_path(args.annotated_output, "annotated output")
        options = ReplayOptions(
            source=source,
            model=args.model or config.detection.model,
            device=args.device or config.detection.device,
            target_classes=_target_classes(args.target_class, config.detection.target_labels),
            confidence_threshold=(
                args.confidence
                if args.confidence is not None
                else config.detection.confidence_threshold
            ),
            iou_threshold=args.iou if args.iou is not None else config.detection.iou_threshold,
            image_size=(
                args.image_size if args.image_size is not None else config.detection.image_size
            ),
            max_frames=args.max_frames,
            report_path=args.report,
            annotated_output=args.annotated_output,
            thresholds=AcceptanceThresholds(
                minimum_selected_target_ratio=args.min_selected_target_ratio,
                maximum_p95_inference_ms=args.max_p95_inference_ms,
                maximum_mean_normalized_error=args.max_mean_normalized_error,
                maximum_p95_normalized_error=args.max_p95_normalized_error,
                maximum_pan_saturation_count=args.max_pan_saturation_count,
                maximum_tilt_saturation_count=args.max_tilt_saturation_count,
            ),
        )
        if not Path(options.model).is_file():
            print(
                "warning: an uncached named model can trigger an Ultralytics download; "
                "use a local model path for network-free replay",
                file=sys.stderr,
            )
        with _termination_signals() as termination:
            try:
                result = run_replay(
                    config,
                    options,
                    environment=environment_metadata,
                    stop_requested=termination,
                    command=(
                        "marine-ptz-replay",
                        *(argv if argv is not None else sys.argv[1:]),
                    ),
                )
            except ReplayCancelled:
                if termination.signal_number == signal.SIGTERM:
                    return 143
                return 130
        if termination.signal_number == signal.SIGINT:
            return 130
        if termination.signal_number == signal.SIGTERM:
            return 143
        if result.accepted is False:
            print(f"replay acceptance failed; report={options.report_path}", file=sys.stderr)
            return ACCEPTANCE_FAILURE_STATUS
        print(f"replay report={options.report_path}")
        return 0
    except KeyboardInterrupt:
        print("replay interrupted", file=sys.stderr)
        return 130
    except (
        ConfigError,
        ReplayError,
        OpenCVSourceError,
        UltralyticsDetectorError,
        InferenceDeviceError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"replay error: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/development.yaml"))
    parser.add_argument("--source", required=True, help="existing local image or video path")
    parser.add_argument("--model", help="local model path or named Ultralytics model")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--target-class",
        action="append",
        help="repeat or use comma-separated classes",
    )
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--annotated-output", type=Path)
    parser.add_argument("--min-selected-target-ratio", type=float)
    parser.add_argument("--max-p95-inference-ms", type=float)
    parser.add_argument("--max-mean-normalized-error", type=float)
    parser.add_argument("--max-p95-normalized-error", type=float)
    parser.add_argument("--max-pan-saturation-count", type=int)
    parser.add_argument("--max-tilt-saturation-count", type=int)
    parser.add_argument(
        "--replay-safe",
        action="store_true",
        help="explicitly override a hardware config with simulated-only replay",
    )
    parser.add_argument("--actuator-backend", help=argparse.SUPPRESS)
    parser.add_argument("--serial-port", help=argparse.SUPPRESS)
    parser.add_argument("--arm-hardware", action="store_true", help=argparse.SUPPRESS)
    return parser


def _validate_hardware_rejection(args: argparse.Namespace, configured_backend: str) -> None:
    if args.arm_hardware or args.serial_port is not None:
        raise ReplayError("offline replay does not accept hardware arming or serial ports")
    if args.actuator_backend not in {None, "simulated"}:
        raise ReplayError("offline replay always uses simulated actuation")
    if configured_backend == "arduino_serial" and not args.replay_safe:
        raise ReplayError(
            "hardware configuration requires explicit --replay-safe for simulated-only replay"
        )


def _local_source(value: str) -> Path:
    if value.startswith("camera:") or value.startswith("path:") or value.isdecimal():
        raise ReplayError(
            "offline replay requires an explicit local image or video path, not a camera source"
        )
    path = Path(value)
    if not path.is_file():
        raise ReplayError(f"offline replay source does not exist or is not a file: {path}")
    return path


def _validate_output_path(path: Path, name: str) -> None:
    if path.exists() and path.is_dir():
        raise ReplayError(f"{name} path must name a file, not a directory")


if __name__ == "__main__":
    raise SystemExit(main())
