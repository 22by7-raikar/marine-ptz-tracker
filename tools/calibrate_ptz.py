#!/usr/bin/env python3
"""Guided, explicitly armed, single-axis PTZ range calibration."""

# ruff: noqa: E402, I001 -- add the src layout before importing the local package.

from __future__ import annotations

import argparse
import select
import sys
import termios
import tty
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_ptz.config import AppConfig, ConfigError, load_config
from marine_ptz.serial_actuator import ArduinoSerialActuator, SerialActuatorError
from marine_ptz.serial_transport import SerialTransportError
from marine_ptz.types import PTZCommand


KeyReader = Callable[[float], str | None]


def run_guided_calibration(
    config: AppConfig,
    actuator: ArduinoSerialActuator,
    *,
    axis: str,
    read_key: KeyReader,
    emit: Callable[[str], None] = print,
) -> None:
    """Run one explicitly selected axis while refreshing the watchdog."""
    if axis not in {"pan", "tilt"}:
        raise ValueError("axis must be pan or tilt")
    if config.serial is None:
        raise ConfigError("guided calibration requires serial configuration")
    if (
        config.actuator.initial_pan_deg != 90.0
        or config.actuator.initial_tilt_deg != 90.0
        or config.serial.neutral_pan_deg != 90.0
        or config.serial.neutral_tilt_deg != 90.0
    ):
        raise ConfigError("guided calibration requires logical and physical neutral 90/90")

    pan = 90.0
    tilt = 90.0
    sequence = 1
    interval_s = 1.0 / config.serial.command_rate_hz
    selected_limits = config.actuator.pan_limits if axis == "pan" else config.actuator.tilt_limits

    try:
        actuator.open()
        actuator.enable()
        actuator.apply(PTZCommand(pan, tilt, sequence))
        sequence += 1
        emit(
            f"calibration armed axis={axis} limits="
            f"{selected_limits.minimum_deg:.0f}..{selected_limits.maximum_deg:.0f}"
        )
        emit("keys: +=logical +1  -=logical -1  c=CENTER  t=STATUS  q/Esc/x=DISABLE")
        emit(_position_line("start", actuator))

        while True:
            key = read_key(interval_s)
            action = "refresh"
            if key in {"q", "Q", "x", "X", "\x1b"}:
                return
            if key in {"+", "="}:
                action = f"{axis}+1"
                candidate = (pan if axis == "pan" else tilt) + 1.0
                if selected_limits.contains(candidate):
                    if axis == "pan":
                        pan = candidate
                    else:
                        tilt = candidate
                else:
                    emit(
                        f"{axis} maximum {selected_limits.maximum_deg:.0f} reached; "
                        "limits were not expanded"
                    )
            elif key in {"-", "_"}:
                action = f"{axis}-1"
                candidate = (pan if axis == "pan" else tilt) - 1.0
                if selected_limits.contains(candidate):
                    if axis == "pan":
                        pan = candidate
                    else:
                        tilt = candidate
                else:
                    emit(
                        f"{axis} minimum {selected_limits.minimum_deg:.0f} reached; "
                        "limits were not expanded"
                    )
            elif key in {"c", "C"}:
                actuator.center()
                pan = tilt = 90.0
                action = "CENTER"
            elif key in {"t", "T"}:
                actuator.status()
                pan = float(actuator.pan_deg)
                tilt = float(actuator.tilt_deg)
                action = "STATUS"
            elif key in {"h", "H", "?"}:
                emit("keys: +=logical +1  -=logical -1  c=CENTER  t=STATUS  q/Esc/x=DISABLE")
                action = "help"
            elif key is not None:
                emit(f"ignored key {key!r}; press h for controls")
                action = "ignored"

            # SET is the only accepted motion command that refreshes the
            # firmware watchdog. Re-send the same bounded pose while observing.
            actuator.apply(PTZCommand(pan, tilt, sequence))
            sequence += 1
            if action != "refresh":
                emit(_position_line(action, actuator))
    finally:
        try:
            if actuator.connected:
                actuator.disable()
        finally:
            actuator.close()


def _position_line(action: str, actuator: Any) -> str:
    return (
        f"action={action} logical={actuator.pan_deg:.0f},{actuator.tilt_deg:.0f} "
        f"physical={actuator.physical_pan_deg},{actuator.physical_tilt_deg}"
    )


def _terminal_key_reader(stream: TextIO) -> KeyReader:
    def read_key(timeout_s: float) -> str | None:
        readable, _, _ = select.select([stream], [], [], timeout_s)
        if not readable:
            return None
        value = stream.read(1)
        return value or None

    return read_key


@contextmanager
def _cbreak_terminal(stream: TextIO) -> Iterator[None]:
    if not stream.isatty():
        yield
        return
    descriptor = stream.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hardware.yaml"))
    parser.add_argument("--port", required=True, help="explicit verified Arduino serial port")
    parser.add_argument("--axis", required=True, choices=("pan", "tilt"))
    parser.add_argument(
        "--arm-hardware",
        action="store_true",
        help="explicitly permit servo ENABLE for this guided session",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.arm_hardware:
        print("calibration error: --arm-hardware is required", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
        if config.serial is None:
            raise ConfigError("selected configuration does not define serial settings")
        settings = replace(config.serial, device=args.port)
        actuator = ArduinoSerialActuator(
            settings,
            config.actuator.pan_limits,
            config.actuator.tilt_limits,
            initial_pan_deg=config.actuator.initial_pan_deg,
            initial_tilt_deg=config.actuator.initial_tilt_deg,
        )
        with _cbreak_terminal(sys.stdin):
            run_guided_calibration(
                config,
                actuator,
                axis=args.axis,
                read_key=_terminal_key_reader(sys.stdin),
            )
    except KeyboardInterrupt:
        print("\ncalibration stopped by operator", file=sys.stderr)
        return 130
    except (
        ConfigError,
        SerialActuatorError,
        SerialTransportError,
        ValueError,
    ) as exc:
        print(f"calibration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
