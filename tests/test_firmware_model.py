from __future__ import annotations

import re
from pathlib import Path

from marine_ptz.config import load_config
from marine_ptz.firmware_model import (
    COMMANDS_PER_LOOP,
    RX_BYTES_PER_LOOP,
    TX_BYTES_PER_LOOP,
    FirmwareRuntimeModel,
    FirmwareSafetyConfig,
    FirmwareStateMachine,
    millis_add,
    millis_deadline_reached,
)
from marine_ptz.serial_protocol import (
    MAX_SEQUENCE,
    AckResponse,
    EnableCommand,
    ErrorCode,
    ErrorResponse,
    HelloCommand,
    ReadyResponse,
    SetCommand,
    StatusCommand,
    StatusResponse,
    WatchdogState,
)


def test_firmware_starts_disabled_and_handshake_establishes_sequence() -> None:
    machine = FirmwareStateMachine()

    assert machine.enabled is False
    assert machine.watchdog_state is WatchdogState.IDLE
    assert machine.handle(SetCommand(1, 90, 90)) == ErrorResponse(
        1,
        ErrorCode.BAD_SEQUENCE,
    )
    assert machine.handle(HelloCommand(1)) == ReadyResponse(1, "1", "0.1.0")
    assert machine.handle(EnableCommand(2)) == AckResponse(2, "ENABLE", 90, 90)


def test_firmware_accepts_defined_sequence_wraparound() -> None:
    machine = FirmwareStateMachine()

    machine.handle(HelloCommand(MAX_SEQUENCE))

    assert machine.handle(EnableCommand(1)) == AckResponse(1, "ENABLE", 90, 90)


def test_duplicate_set_is_idempotent_and_conflicting_duplicate_is_rejected() -> None:
    machine = FirmwareStateMachine()
    machine.handle(HelloCommand(1))
    machine.handle(EnableCommand(2))

    first = machine.handle(SetCommand(3, 100, 80))
    duplicate = machine.handle(SetCommand(3, 100, 80))
    conflicting = machine.handle(SetCommand(3, 101, 80))

    assert first == duplicate == AckResponse(3, "SET", 100, 80)
    assert (machine.pan_deg, machine.tilt_deg) == (100, 80)
    assert conflicting == ErrorResponse(3, ErrorCode.STALE_SEQUENCE)


def test_accepted_duplicate_set_refreshes_watchdog_after_lost_ack() -> None:
    now = [0.0]
    machine = FirmwareStateMachine(clock=lambda: now[0])
    machine.handle(HelloCommand(1))
    machine.handle(EnableCommand(2))
    now[0] = 0.1
    accepted = machine.handle(SetCommand(3, 100, 80))

    now[0] = 1.0
    retry = machine.handle(SetCommand(3, 100, 80))
    now[0] = 1.5
    machine.tick()

    assert accepted == retry == AckResponse(3, "SET", 100, 80)
    assert machine.enabled is True

    now[0] = 2.0
    machine.tick()
    assert machine.enabled is False
    assert machine.watchdog_state is WatchdogState.EXPIRED


def test_rejected_and_conflicting_duplicate_sets_do_not_refresh_watchdog() -> None:
    now = [0.0]
    machine = FirmwareStateMachine(clock=lambda: now[0])
    machine.handle(HelloCommand(1))
    machine.handle(EnableCommand(2))
    rejected = machine.handle(SetCommand(3, 10, 90))

    now[0] = 0.5
    repeated = machine.handle(SetCommand(3, 10, 90))
    now[0] = 0.9
    conflict = machine.handle(SetCommand(3, 11, 90))
    now[0] = 1.0
    machine.tick()

    assert rejected == repeated == ErrorResponse(3, ErrorCode.OUT_OF_RANGE)
    assert conflict == ErrorResponse(3, ErrorCode.STALE_SEQUENCE)
    assert machine.enabled is False
    assert machine.watchdog_state is WatchdogState.EXPIRED


def test_rejected_set_replays_cached_error_across_state_change_without_refresh() -> None:
    now = [0.0]
    machine = FirmwareStateMachine(clock=lambda: now[0])
    machine.handle(HelloCommand(1))
    rejected = machine.handle(SetCommand(2, 100, 80))
    machine.handle(EnableCommand(3))

    now[0] = 0.9
    repeated = machine.handle(SetCommand(2, 100, 80))
    now[0] = 1.0
    machine.tick()

    assert rejected == repeated == ErrorResponse(2, ErrorCode.NOT_ENABLED)
    assert machine.enabled is False
    assert machine.watchdog_state is WatchdogState.EXPIRED


def test_accepted_set_duplicate_after_expiry_is_not_enabled_and_does_not_move() -> None:
    now = [0.0]
    machine = FirmwareStateMachine(clock=lambda: now[0])
    machine.handle(HelloCommand(1))
    machine.handle(EnableCommand(2))
    machine.handle(SetCommand(3, 100, 80))

    now[0] = 1.0
    repeated = machine.handle(SetCommand(3, 100, 80))

    assert repeated == ErrorResponse(3, ErrorCode.NOT_ENABLED)
    assert machine.enabled is False
    assert machine.watchdog_state is WatchdogState.EXPIRED
    assert (machine.pan_deg, machine.tilt_deg) == (100, 80)


def test_firmware_rejects_safe_limit_violation_without_moving() -> None:
    machine = FirmwareStateMachine()
    machine.handle(HelloCommand(1))
    machine.handle(EnableCommand(2))

    response = machine.handle(SetCommand(3, 10, 90))

    assert response == ErrorResponse(3, ErrorCode.OUT_OF_RANGE)
    assert (machine.pan_deg, machine.tilt_deg) == (90, 90)


def test_watchdog_expires_to_disabled_and_malformed_input_does_not_refresh() -> None:
    now = [0.0]
    machine = FirmwareStateMachine(clock=lambda: now[0])
    machine.handle(HelloCommand(1))
    machine.handle(EnableCommand(2))
    machine.handle(SetCommand(3, 100, 80))

    now[0] = 0.9
    malformed = machine.handle_frame(b"SET 4 nope 90")
    assert malformed == (b"ERR 4 BAD_INTEGER\r\n",)
    assert machine.enabled is True

    now[0] = 1.0
    machine.tick()

    assert machine.enabled is False
    assert machine.watchdog_state is WatchdogState.EXPIRED
    status = machine.handle(StatusCommand(4))
    assert status == StatusResponse(4, False, 100, 80, WatchdogState.EXPIRED)


def test_enable_starts_watchdog_window_without_blocking() -> None:
    now = [10.0]
    machine = FirmwareStateMachine(
        FirmwareSafetyConfig(watchdog_timeout_s=0.5),
        clock=lambda: now[0],
    )
    machine.handle(HelloCommand(1))
    machine.handle(EnableCommand(2))

    now[0] = 10.5
    machine.tick()

    assert machine.enabled is False
    assert machine.watchdog_state is WatchdogState.EXPIRED
    assert machine.handle(EnableCommand(2)) == ErrorResponse(
        2,
        ErrorCode.NOT_ENABLED,
    )


def test_continuous_rx_cannot_prevent_watchdog_expiry() -> None:
    now = [0.0]
    machine = FirmwareStateMachine(clock=lambda: now[0])
    machine.handle(HelloCommand(1))
    machine.handle(EnableCommand(2))
    runtime = FirmwareRuntimeModel(machine, announce_startup=False)
    runtime.feed_rx(b"\n" * 1_000)

    for step in range(11):
        now[0] = step / 10
        runtime.step(tx_capacity=16)

    assert machine.enabled is False
    assert machine.watchdog_state is WatchdogState.EXPIRED
    assert runtime.queued_rx_bytes > 0


def test_slow_tx_cannot_prevent_expiry_or_allow_pending_overwrite() -> None:
    now = [0.0]
    machine = FirmwareStateMachine(clock=lambda: now[0])
    machine.handle(HelloCommand(1))
    machine.handle(EnableCommand(2))
    runtime = FirmwareRuntimeModel(machine, announce_startup=False)
    runtime.feed_rx(b"STATUS 3\nSTATUS 4\n")

    assert runtime.step(tx_capacity=0) == b""
    first_pending = runtime.pending_output
    queued_after_first = runtime.queued_rx_bytes
    assert first_pending.startswith(b"STATUS 3 ")

    now[0] = 1.0
    assert runtime.step(tx_capacity=0) == b""

    assert machine.enabled is False
    assert machine.watchdog_state is WatchdogState.EXPIRED
    assert runtime.pending_output == first_pending
    assert runtime.queued_rx_bytes == queued_after_first


def test_pending_output_is_emitted_incrementally_before_next_command() -> None:
    machine = FirmwareStateMachine()
    runtime = FirmwareRuntimeModel(
        machine,
        announce_startup=False,
        tx_byte_budget=5,
    )
    runtime.feed_rx(b"HELLO 1\nHELLO 2\n")

    pieces: list[bytes] = []
    queued_while_pending: list[int] = []
    for _ in range(20):
        pieces.append(runtime.step(tx_capacity=5))
        if runtime.pending_output:
            queued_while_pending.append(runtime.queued_rx_bytes)
        if not runtime.pending_output and runtime.queued_rx_bytes == 0:
            break

    assert b"".join(pieces) == (b"READY 1 1 0.1.0\r\nREADY 2 1 0.1.0\r\n")
    assert queued_while_pending
    assert max(queued_while_pending) > 0


def test_runtime_enforces_one_aggregate_tx_budget_after_rx_processing() -> None:
    machine = FirmwareStateMachine()
    runtime = FirmwareRuntimeModel(machine, announce_startup=False)
    runtime.feed_rx(b"STATUS 1\n")

    emitted = runtime.step(tx_capacity=64)

    assert len(emitted) == TX_BYTES_PER_LOOP
    assert emitted.startswith(b"ERR 1 BAD_SEQUEN")
    assert runtime.pending_output == b"CE\r\n"


def test_unsigned_millis_deadline_comparison_handles_wraparound() -> None:
    armed_at = 0xFFFF_FF00
    deadline = millis_add(armed_at, 1_000)

    assert deadline == 744
    assert millis_deadline_reached(743, deadline) is False
    assert millis_deadline_reached(744, deadline) is True
    assert millis_deadline_reached(1_000, deadline) is True


def test_firmware_constants_agree_with_checked_in_hardware_config() -> None:
    config = load_config("configs/hardware.yaml")
    assert config.serial is not None
    header = Path("firmware/ptz_controller/protocol_config.h").read_text(encoding="utf-8")

    expected = {
        "SERIAL_BAUD_RATE": config.serial.baudrate,
        "PAN_MIN_DEG": int(config.serial.physical_pan_limits.minimum_deg),
        "PAN_MAX_DEG": int(config.serial.physical_pan_limits.maximum_deg),
        "TILT_MIN_DEG": int(config.serial.physical_tilt_limits.minimum_deg),
        "TILT_MAX_DEG": int(config.serial.physical_tilt_limits.maximum_deg),
        "PAN_NEUTRAL_DEG": int(config.serial.neutral_pan_deg),
        "TILT_NEUTRAL_DEG": int(config.serial.neutral_tilt_deg),
        "WATCHDOG_TIMEOUT_MS": int(config.serial.watchdog_timeout_s * 1000),
    }
    for name, value in expected.items():
        match = re.search(
            rf"constexpr\s+[^=]+\s+{name}\s*=\s*([0-9]+)",
            header,
        )
        assert match is not None, name
        assert int(match.group(1)) == value

    assert "PAN_SERVO_PIN = 9" in header
    assert "TILT_SERVO_PIN = 10" in header
    assert f"MAX_RX_BYTES_PER_LOOP = {RX_BYTES_PER_LOOP}" in header
    assert f"MAX_COMMANDS_PER_LOOP = {COMMANDS_PER_LOOP}" in header
    assert f"MAX_TX_BYTES_PER_LOOP = {TX_BYTES_PER_LOOP}" in header


def test_firmware_source_uses_fixed_buffers_and_nonblocking_loop() -> None:
    files = [
        Path("firmware/ptz_controller/ptz_controller.ino"),
        Path("firmware/ptz_controller/protocol.cpp"),
        Path("firmware/ptz_controller/protocol.h"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)

    assert "char payload[MAX_LINE_LENGTH + 1]" in source
    assert "char pendingTx[MAX_LINE_LENGTH + 2U]" in source
    assert "Serial.availableForWrite()" in source
    assert "MAX_RX_BYTES_PER_LOOP" in source
    assert "MAX_COMMANDS_PER_LOOP" in source
    assert "MAX_TX_BYTES_PER_LOOP" in source
    assert "parseInt32Strict(" in source
    assert "millis()" in source
    assert "Serial.println" not in source
    assert "delay(" not in source
    assert "String" not in source
    assert "malloc(" not in source
    assert "new " not in source

    sketch = files[0].read_text(encoding="utf-8")
    loop_body = sketch.split("void loop() {", 1)[1].split("}", 1)[0]
    assert loop_body.count("flushPendingResponseBounded();") == 1
