from __future__ import annotations

from collections.abc import Callable

import pytest

from marine_ptz.config import AppConfig, load_config
from marine_ptz.firmware_model import FirmwareSafetyConfig, FirmwareStateMachine
from marine_ptz.serial_actuator import ArduinoSerialActuator
from marine_ptz.serial_protocol import (
    CenterCommand,
    DisableCommand,
    SetCommand,
    StatusCommand,
    parse_command,
)
from marine_ptz.serial_transport import InMemorySerialTransport
from tools.calibrate_ptz import main, run_guided_calibration


def _actuator(config: AppConfig) -> tuple[ArduinoSerialActuator, InMemorySerialTransport]:
    assert config.serial is not None
    settings = config.serial
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
    now = [0.0]
    actuator = ArduinoSerialActuator(
        settings,
        config.actuator.pan_limits,
        config.actuator.tilt_limits,
        initial_pan_deg=config.actuator.initial_pan_deg,
        initial_tilt_deg=config.actuator.initial_tilt_deg,
        transport=transport,
        monotonic=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )
    return actuator, transport


def _keys(values: list[str | None]) -> Callable[[float], str | None]:
    iterator = iter(values)

    def read_key(_timeout_s: float) -> str | None:
        return next(iterator)

    return read_key


def _commands(transport: InMemorySerialTransport) -> list[object]:
    return [parse_command(frame) for frame in transport.writes]


def test_guided_calibration_steps_one_axis_refreshes_and_disables() -> None:
    config = load_config("configs/hardware.yaml")
    actuator, transport = _actuator(config)
    read_key = _keys([None, "+", "t", "c", "q"])
    lines: list[str] = []

    run_guided_calibration(
        config,
        actuator,
        axis="pan",
        read_key=read_key,
        emit=lines.append,
    )

    commands = _commands(transport)
    set_commands = [command for command in commands if isinstance(command, SetCommand)]
    assert [(command.pan_deg, command.tilt_deg) for command in set_commands] == [
        (90, 90),
        (90, 90),
        (89, 90),
        (89, 90),
        (90, 90),
    ]
    assert any(isinstance(command, StatusCommand) for command in commands)
    assert any(isinstance(command, CenterCommand) for command in commands)
    assert isinstance(commands[-1], DisableCommand)
    assert actuator.closed is True
    assert "action=pan+1 logical=91,90 physical=89,90" in lines


def test_guided_calibration_never_expands_configured_limit() -> None:
    config = load_config("configs/hardware.yaml")
    actuator, transport = _actuator(config)
    read_key = _keys([*(["+"] * 17), "q"])
    lines: list[str] = []

    run_guided_calibration(
        config,
        actuator,
        axis="pan",
        read_key=read_key,
        emit=lines.append,
    )

    set_commands = [command for command in _commands(transport) if isinstance(command, SetCommand)]
    assert all(75 <= command.pan_deg <= 105 for command in set_commands)
    assert min(command.pan_deg for command in set_commands) == 75
    assert any("limits were not expanded" in line for line in lines)
    assert actuator.closed is True


def test_guided_calibration_exception_disables_and_closes() -> None:
    config = load_config("configs/hardware.yaml")
    actuator, transport = _actuator(config)

    def fail_after_enable(_timeout_s: float) -> str | None:
        raise RuntimeError("operator input failed")

    with pytest.raises(RuntimeError, match="operator input failed"):
        run_guided_calibration(
            config,
            actuator,
            axis="tilt",
            read_key=fail_after_enable,
            emit=lambda _line: None,
        )

    commands = _commands(transport)
    assert isinstance(commands[-1], DisableCommand)
    assert actuator.enabled is False
    assert actuator.closed is True


def test_calibration_cli_requires_explicit_hardware_arming() -> None:
    result = main(
        [
            "--config",
            "configs/hardware.yaml",
            "--port",
            "/dev/never-opened",
            "--axis",
            "pan",
        ]
    )

    assert result == 2
