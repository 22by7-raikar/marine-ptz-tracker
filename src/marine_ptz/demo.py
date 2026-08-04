"""Command-line synthetic tracking demonstration."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from .actuators import SimulatedPTZActuator
from .config import AppConfig, ConfigError, load_config
from .synthetic import SyntheticCamera, SyntheticTargetDetector
from .tracking import HighestConfidenceTargetSelector, ProportionalPTZController


def run(config: AppConfig, *, steps: int, sleep: bool = True) -> None:
    """Run a finite synthetic tracking loop."""
    if steps <= 0:
        raise ValueError("steps must be greater than zero")
    if config.camera.source != "synthetic":
        raise ConfigError("synthetic demo requires camera.source: synthetic")
    if config.actuator.backend != "simulated":
        raise ConfigError("synthetic demo requires actuator.backend: simulated")

    camera = SyntheticCamera(config.camera.width, config.camera.height)
    detector = SyntheticTargetDetector(label=config.detection.target_labels[0])
    selector = HighestConfidenceTargetSelector()
    actuator = SimulatedPTZActuator(
        config.actuator.pan_limits,
        config.actuator.tilt_limits,
        initial_pan_deg=config.actuator.initial_pan_deg,
        initial_tilt_deg=config.actuator.initial_tilt_deg,
    )
    controller = ProportionalPTZController(
        config.tracking,
        config.actuator.pan_limits,
        config.actuator.tilt_limits,
        initial_pan_deg=config.actuator.initial_pan_deg,
        initial_tilt_deg=config.actuator.initial_tilt_deg,
    )
    interval_s = 1 / config.tracking.update_rate_hz
    deadline = time.monotonic()

    try:
        for _ in range(steps):
            if sleep:
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            frame = camera.read()
            if frame is None:
                break
            detections = detector.detect(frame)
            target = selector.select(frame, detections)
            command = controller.command_for_target(frame, target)
            actuator.apply(command)
            if config.runtime.telemetry_enabled:
                center = target.detection.center if target is not None else None
                center_text = "none" if center is None else f"{center[0]:.1f},{center[1]:.1f}"
                print(
                    f"seq={frame.sequence:03d} target={center_text} "
                    f"pan={actuator.pan_deg:.2f} tilt={actuator.tilt_deg:.2f}"
                )
            deadline += interval_s
    finally:
        camera.close()
        actuator.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/development.yaml"),
        help="YAML configuration path",
    )
    parser.add_argument("--steps", type=int, default=20, help="number of frames to run")
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="run without real-time pacing",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        run(config, steps=args.steps, sleep=not args.no_sleep)
    except (ConfigError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
