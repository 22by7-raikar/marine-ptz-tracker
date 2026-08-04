#!/usr/bin/env python3
"""Exercise the PTZ serial protocol against a fake unless --port is explicit."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_ptz.config import ConfigError, load_config  # noqa: E402
from marine_ptz.firmware_model import (  # noqa: E402
    FirmwareSafetyConfig,
    FirmwareStateMachine,
)
from marine_ptz.serial_actuator import (  # noqa: E402
    ArduinoSerialActuator,
    SerialActuatorError,
)
from marine_ptz.serial_transport import (  # noqa: E402
    InMemorySerialTransport,
    SerialTransportError,
)
from marine_ptz.types import PTZCommand  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hardware.yaml"),
    )
    parser.add_argument(
        "--port",
        help="explicit physical serial port; omit to use the in-memory firmware model",
    )
    parser.add_argument("--pan", type=float, default=100.0)
    parser.add_argument("--tilt", type=float, default=80.0)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.5,
        help="keep the physical actuator enabled briefly before closing",
    )
    args = parser.parse_args(argv)

    actuator: ArduinoSerialActuator | None = None
    try:
        config = load_config(args.config)
        if config.serial is None:
            raise ConfigError("selected configuration does not define serial settings")
        settings = (
            replace(config.serial, device=args.port) if args.port is not None else config.serial
        )
        if args.port is None:
            firmware = FirmwareStateMachine(
                FirmwareSafetyConfig(
                    pan_min_deg=int(settings.physical_pan_limits.minimum_deg),
                    pan_max_deg=int(settings.physical_pan_limits.maximum_deg),
                    tilt_min_deg=int(settings.physical_tilt_limits.minimum_deg),
                    tilt_max_deg=int(settings.physical_tilt_limits.maximum_deg),
                    neutral_pan_deg=int(settings.neutral_pan_deg),
                    neutral_tilt_deg=int(settings.neutral_tilt_deg),
                    watchdog_timeout_s=settings.watchdog_timeout_s,
                )
            )
            transport = InMemorySerialTransport(firmware.handle_frame)
            mode = "fake"
        else:
            transport = None
            mode = f"physical:{args.port}"
        actuator = ArduinoSerialActuator(
            settings,
            config.actuator.pan_limits,
            config.actuator.tilt_limits,
            initial_pan_deg=config.actuator.initial_pan_deg,
            initial_tilt_deg=config.actuator.initial_tilt_deg,
            transport=transport,
        )
        actuator.open()
        actuator.enable()
        actuator.apply(PTZCommand(args.pan, args.tilt, sequence=1))
        status = actuator.status()
        print(
            f"mode={mode} connected={actuator.connected} enabled={status.enabled} "
            f"logical={actuator.pan_deg:.1f},{actuator.tilt_deg:.1f} "
            f"physical={status.pan_deg},{status.tilt_deg} "
            f"watchdog={status.watchdog_state.value}"
        )
        time.sleep(args.hold_seconds)

    except (
        ConfigError,
        SerialActuatorError,
        SerialTransportError,
        ValueError,
    ) as exc:
        print(f"serial demo error: {exc}", file=sys.stderr)
        return 2
    finally:
        if actuator is not None:
            actuator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
