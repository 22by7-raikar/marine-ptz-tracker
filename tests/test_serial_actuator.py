from __future__ import annotations

from dataclasses import replace

import pytest

from marine_ptz.cancellation import OperationCancelled
from marine_ptz.config import AppConfig, load_config
from marine_ptz.firmware_model import FirmwareSafetyConfig, FirmwareStateMachine
from marine_ptz.interfaces import Actuator
from marine_ptz.serial_actuator import (
    ArduinoSerialActuator,
    SerialActuatorConnectionError,
    SerialActuatorError,
    SerialActuatorProtocolError,
    SerialActuatorTimeoutError,
)
from marine_ptz.serial_protocol import (
    MAX_SEQUENCE,
    AckResponse,
    CenterCommand,
    DisableCommand,
    EnableCommand,
    ErrorCode,
    ErrorResponse,
    HelloCommand,
    ReadyResponse,
    SetCommand,
    StatusCommand,
    StatusResponse,
    WatchdogState,
    encode_response,
    parse_command,
)
from marine_ptz.serial_transport import (
    InMemorySerialTransport,
    SerialTransportError,
)
from marine_ptz.tracking import ProportionalPTZController
from marine_ptz.types import Detection, Frame, PTZCommand, Target


def _firmware(config: AppConfig) -> FirmwareStateMachine:
    assert config.serial is not None
    settings = config.serial
    return FirmwareStateMachine(
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


def _actuator(
    config: AppConfig,
    transport: InMemorySerialTransport,
    **kwargs: object,
) -> ArduinoSerialActuator:
    assert config.serial is not None
    return ArduinoSerialActuator(
        config.serial,
        config.actuator.pan_limits,
        config.actuator.tilt_limits,
        initial_pan_deg=config.actuator.initial_pan_deg,
        initial_tilt_deg=config.actuator.initial_tilt_deg,
        transport=transport,
        **kwargs,  # type: ignore[arg-type]
    )


def test_handshake_enable_set_status_and_protocol_conformance() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    transport = InMemorySerialTransport(firmware.handle_frame)
    actuator = _actuator(config, transport)

    actuator.open()
    actuator.enable()
    actuator.apply(PTZCommand(100.0, 80.0, sequence=99))
    status = actuator.status()

    assert isinstance(actuator, Actuator)
    assert actuator.connected is True
    assert actuator.enabled is True
    assert (actuator.pan_deg, actuator.tilt_deg) == (100.0, 80.0)
    assert (status.pan_deg, status.tilt_deg) == (80, 100)
    assert (actuator.physical_pan_deg, actuator.physical_tilt_deg) == (80, 100)
    assert transport.reset_count == 1
    actuator.close()


def test_construction_and_disabled_apply_do_not_write() -> None:
    config = load_config("configs/hardware.yaml")
    transport = InMemorySerialTransport(_firmware(config).handle_frame)
    actuator = _actuator(config, transport)

    assert transport.writes == []
    with pytest.raises(SerialActuatorConnectionError):
        actuator.apply(PTZCommand(90.0, 90.0, sequence=1))
    assert transport.writes == []

    actuator.open()
    writes_after_open = list(transport.writes)
    with pytest.raises(SerialActuatorError, match="disabled"):
        actuator.apply(PTZCommand(90.0, 90.0, sequence=2))
    assert transport.writes == writes_after_open


def test_direction_and_offset_mapping_remain_separate_from_logical_state() -> None:
    config = load_config("configs/hardware.yaml")
    assert config.serial is not None
    reversed_settings = replace(
        config.serial,
        pan_direction=-1,
        tilt_direction=-1,
        pan_offset_deg=180.0,
        tilt_offset_deg=180.0,
    )
    firmware = _firmware(config)
    transport = InMemorySerialTransport(firmware.handle_frame)
    actuator = ArduinoSerialActuator(
        reversed_settings,
        config.actuator.pan_limits,
        config.actuator.tilt_limits,
        initial_pan_deg=90.0,
        initial_tilt_deg=90.0,
        transport=transport,
    )

    actuator.open()
    actuator.enable()
    actuator.apply(PTZCommand(100.0, 80.0, sequence=1))

    assert (actuator.pan_deg, actuator.tilt_deg) == (100.0, 80.0)
    assert (actuator.physical_pan_deg, actuator.physical_tilt_deg) == (80, 100)


@pytest.mark.parametrize(
    ("logical_pan", "expected_physical_pan"),
    [(75.0, 105), (90.0, 90), (105.0, 75)],
)
def test_calibrated_pan_mapping_and_status_conversion(
    logical_pan: float,
    expected_physical_pan: int,
) -> None:
    config = load_config("configs/hardware.yaml")
    actuator = _actuator(
        config,
        InMemorySerialTransport(_firmware(config).handle_frame),
    )

    actuator.open()
    actuator.enable()
    actuator.apply(PTZCommand(logical_pan, 90.0, sequence=1))
    status = actuator.status()

    assert actuator.pan_deg == logical_pan
    assert actuator.physical_pan_deg == expected_physical_pan
    assert status.pan_deg == expected_physical_pan
    assert actuator.tilt_deg == 90.0
    assert actuator.physical_tilt_deg == status.tilt_deg == 90


@pytest.mark.parametrize(
    ("target_y", "expected_logical_tilt", "expected_physical_tilt"),
    [
        (75.0, 93.0, 87),
        (25.0, 87.0, 93),
        (50.0, 90.0, 90),
    ],
)
def test_calibrated_vertical_mapping_follows_image_direction(
    target_y: float,
    expected_logical_tilt: float,
    expected_physical_tilt: int,
) -> None:
    config = load_config("configs/hardware.yaml")
    frame = Frame(image=None, timestamp_s=0.0, width=100, height=100, sequence=1)
    target = Target(
        Detection("person", 0.9, 40.0, target_y - 5.0, 60.0, target_y + 5.0),
        frame.sequence,
    )
    controller = ProportionalPTZController(
        config.tracking,
        config.actuator.pan_limits,
        config.actuator.tilt_limits,
        initial_pan_deg=config.actuator.initial_pan_deg,
        initial_tilt_deg=config.actuator.initial_tilt_deg,
    )
    command = controller.command_for(frame, target)
    firmware = _firmware(config)
    actuator = _actuator(config, InMemorySerialTransport(firmware.handle_frame))

    actuator.open()
    actuator.enable()
    actuator.apply(command)
    status = actuator.status()

    assert command.pan_deg == actuator.pan_deg == 90.0
    assert command.tilt_deg == actuator.tilt_deg == expected_logical_tilt
    assert actuator.physical_pan_deg == 90
    assert actuator.physical_tilt_deg == expected_physical_tilt
    assert status.tilt_deg == expected_physical_tilt


def test_calibrated_mapping_preserves_pan_and_physical_endpoints() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    actuator = _actuator(config, InMemorySerialTransport(firmware.handle_frame))

    actuator.open()
    actuator.enable()
    actuator.apply(PTZCommand(75.0, 105.0, sequence=1))
    assert (actuator.physical_pan_deg, actuator.physical_tilt_deg) == (105, 75)
    actuator.apply(PTZCommand(105.0, 75.0, sequence=2))
    assert (actuator.physical_pan_deg, actuator.physical_tilt_deg) == (75, 105)
    status = actuator.status()

    assert (actuator.pan_deg, actuator.tilt_deg) == (105.0, 75.0)
    assert (status.pan_deg, status.tilt_deg) == (75, 105)


def test_host_rejects_logical_limit_violation_without_writing() -> None:
    config = load_config("configs/hardware.yaml")
    transport = InMemorySerialTransport(_firmware(config).handle_frame)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()
    writes = len(transport.writes)

    with pytest.raises(ValueError, match="logical pan"):
        actuator.apply(PTZCommand(10.0, 90.0, sequence=1))

    assert len(transport.writes) == writes


def test_lost_ack_after_applied_set_retries_same_idempotent_sequence() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    set_attempts: list[SetCommand] = []

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        if isinstance(command, SetCommand):
            set_attempts.append(command)
            responses = firmware.handle_frame(frame)
            if len(set_attempts) == 1:
                return ()
            return responses
        return firmware.handle_frame(frame)

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()

    actuator.apply(PTZCommand(100.0, 80.0, sequence=1))

    assert len(set_attempts) == 2
    assert set_attempts[0] == set_attempts[1]


def test_stale_acknowledgement_is_ignored_until_correlated_response() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        responses = firmware.handle_frame(frame)
        if isinstance(command, SetCommand):
            stale_sequence = command.sequence - 1 or 65_535
            stale = encode_response(
                AckResponse(stale_sequence, "SET", command.pan_deg, command.tilt_deg)
            )
            return (stale, *responses)
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()

    actuator.apply(PTZCommand(100.0, 80.0, sequence=1))

    assert actuator.physical_pan_deg == 80


def test_malformed_response_retries_then_reports_clear_error() -> None:
    config = load_config("configs/hardware.yaml")
    transport = InMemorySerialTransport(lambda frame: (b"MALFORMED\n",))
    actuator = _actuator(config, transport)

    with pytest.raises(SerialActuatorProtocolError, match="malformed"):
        actuator.open()

    assert actuator.connected is False
    assert transport.close_count == 1


def test_handshake_protocol_version_mismatch_fails_closed() -> None:
    config = load_config("configs/hardware.yaml")

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        assert isinstance(command, HelloCommand)
        return (encode_response(ReadyResponse(command.sequence, "2", "0.1.0")),)

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)

    with pytest.raises(SerialActuatorProtocolError, match="version mismatch"):
        actuator.open()

    assert actuator.connected is False
    assert transport.close_count == 1


def test_handshake_ignores_unsolicited_startup_ready_zero() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        responses = firmware.handle_frame(frame)
        if isinstance(command, HelloCommand):
            return (firmware.startup_frame(), *responses)
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)

    actuator.open()

    assert actuator.connected is True
    assert actuator.startup_ready_seen is True


def test_handshake_retries_identical_hello_after_first_is_lost() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    hellos: list[HelloCommand] = []
    now = [0.0]

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        if isinstance(command, HelloCommand):
            hellos.append(command)
            if len(hellos) == 1:
                return ()
        return firmware.handle_frame(frame)

    def sleep(duration: float) -> None:
        now[0] += duration

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=sleep,
    )

    actuator.open()

    assert len(hellos) == 2
    assert hellos[0] == hellos[1]
    assert now[0] == pytest.approx(0.25)


def test_resetting_transport_honors_boot_grace_and_delayed_firmware() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    now = [0.0]
    hello_writes: list[bytes] = []

    class ResettingTransport(InMemorySerialTransport):
        @property
        def may_reset_on_open(self) -> bool:
            return True

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        if isinstance(command, HelloCommand):
            hello_writes.append(frame)
            if now[0] < 2.5:
                return ()
        return firmware.handle_frame(frame)

    def sleep(duration: float) -> None:
        now[0] += duration

    transport = ResettingTransport(responder)
    actuator = _actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=sleep,
    )

    actuator.open()

    assert now[0] == pytest.approx(2.5)
    assert len(hello_writes) == 3
    assert len(set(hello_writes)) == 1
    assert actuator.connected is True


def test_mismatched_correlated_ready_is_rejected_and_partial_open_closes() -> None:
    config = load_config("configs/hardware.yaml")

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        assert isinstance(command, HelloCommand)
        wrong = command.sequence + 1
        return (encode_response(ReadyResponse(wrong, "1", "0.1.0")),)

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)

    with pytest.raises(SerialActuatorProtocolError, match="READY sequence"):
        actuator.open()

    assert actuator.connected is False
    assert actuator.enabled is False
    assert transport.close_count == 1


def test_correlated_ready_at_handshake_timeout_boundary_is_accepted() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    now = [0.0]

    class BoundaryTransport(InMemorySerialTransport):
        delayed: bytes | None = None

        def write(self, data: bytes) -> None:
            self._require_open()
            self.writes.append(bytes(data))
            command = parse_command(data)
            responses = firmware.handle_frame(data)
            if isinstance(command, HelloCommand):
                self.delayed = responses[0]
            else:
                for response in responses:
                    self.queue_encoded(response)

        def read_frame(
            self,
            maximum_length: int = 63,
            *,
            timeout_s: float | None = None,
        ) -> bytes | None:
            if self.delayed is not None:
                assert timeout_s is not None
                boundary = 4.0
                if now[0] + timeout_s >= boundary:
                    now[0] = boundary
                    delayed = self.delayed
                    self.delayed = None
                    self.queue_encoded(delayed)
                else:
                    now[0] += timeout_s
            return super().read_frame(maximum_length, timeout_s=timeout_s)

    transport = BoundaryTransport()
    actuator = _actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )

    actuator.open()

    assert now[0] == pytest.approx(4.0)
    assert actuator.connected is True


def test_handshake_timeout_is_bounded_and_fails_closed() -> None:
    config = load_config("configs/hardware.yaml")
    transport = InMemorySerialTransport(lambda frame: ())
    now = [0.0]

    def sleep(duration: float) -> None:
        now[0] += duration

    actuator = _actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=sleep,
    )

    with pytest.raises(SerialActuatorTimeoutError, match="startup deadline"):
        actuator.open()

    assert now[0] == pytest.approx(4.0)
    assert len(transport.writes) == 16
    assert transport.close_count == 1


def test_partial_open_failure_resets_state_and_closes_transport() -> None:
    config = load_config("configs/hardware.yaml")

    class ResetFailureTransport(InMemorySerialTransport):
        def reset_input_buffer(self) -> None:
            raise SerialTransportError("reset failed")

    transport = ResetFailureTransport()
    actuator = _actuator(config, transport)

    with pytest.raises(SerialTransportError, match="reset failed"):
        actuator.open()

    assert actuator.connected is False
    assert actuator.enabled is False
    assert actuator.position_uncertain is False
    assert transport.close_count == 1


def test_failed_reopen_preserves_existing_position_uncertainty() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    mode = ["normal"]
    now = [0.0]

    def responder(frame: bytes) -> tuple[bytes, ...]:
        if mode[0] == "silent":
            return ()
        return firmware.handle_frame(frame)

    class FailingTransport(InMemorySerialTransport):
        def write(self, data: bytes) -> None:
            if mode[0] == "disconnect" and isinstance(
                parse_command(data),
                SetCommand,
            ):
                raise SerialTransportError("device disconnected")
            super().write(data)

    transport = FailingTransport(responder)
    actuator = _actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )
    actuator.open()
    actuator.enable()
    mode[0] = "disconnect"

    with pytest.raises(SerialActuatorConnectionError, match="disconnected"):
        actuator.apply(PTZCommand(100.0, 80.0, sequence=1))
    assert actuator.position_uncertain is True

    mode[0] = "silent"
    with pytest.raises(SerialActuatorTimeoutError):
        actuator.open()

    assert actuator.connected is False
    assert actuator.enabled is False
    assert actuator.position_uncertain is True
    assert transport.close_count == 1


def test_disconnect_during_command_clears_connection_state() -> None:
    config = load_config("configs/hardware.yaml")

    class DisconnectingTransport(InMemorySerialTransport):
        disconnect = False

        def write(self, data: bytes) -> None:
            if self.disconnect:
                raise SerialTransportError("device disconnected")
            super().write(data)

    transport = DisconnectingTransport(_firmware(config).handle_frame)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()
    transport.disconnect = True

    with pytest.raises(SerialActuatorConnectionError, match="disconnected"):
        actuator.apply(PTZCommand(100.0, 80.0, sequence=1))

    assert actuator.connected is False
    assert actuator.enabled is False


def test_command_rate_uses_one_bounded_sleep_without_busy_loop() -> None:
    config = load_config("configs/hardware.yaml")
    now = [10.0]
    sleeps: list[float] = []

    def sleep(duration: float) -> None:
        sleeps.append(duration)
        now[0] += duration

    transport = InMemorySerialTransport(_firmware(config).handle_frame)
    actuator = _actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=sleep,
    )
    actuator.open()
    actuator.enable()
    actuator.apply(PTZCommand(90.0, 90.0, sequence=1))
    actuator.apply(PTZCommand(91.0, 91.0, sequence=2))

    assert sleeps == [pytest.approx(0.1)]


def test_cancellation_during_command_rate_wait_prevents_set_write() -> None:
    config = load_config("configs/hardware.yaml")
    now = [10.0]
    cancelled = [False]

    def sleep(duration: float) -> None:
        now[0] += duration
        cancelled[0] = True

    transport = InMemorySerialTransport(_firmware(config).handle_frame)
    actuator = _actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=sleep,
        cancellation_requested=lambda: cancelled[0],
    )
    actuator.open()
    actuator.enable()
    actuator.apply(PTZCommand(90.0, 90.0, sequence=1))
    writes_before_cancelled_apply = list(transport.writes)

    with pytest.raises(OperationCancelled, match="cancelled"):
        actuator.apply(PTZCommand(91.0, 91.0, sequence=2))

    assert transport.writes == writes_before_cancelled_apply
    assert (actuator.pan_deg, actuator.tilt_deg) == (90.0, 90.0)
    assert actuator.enabled is True


def test_cancellation_before_enable_prevents_enable_write() -> None:
    config = load_config("configs/hardware.yaml")
    cancelled = [False]
    transport = InMemorySerialTransport(_firmware(config).handle_frame)
    actuator = _actuator(
        config,
        transport,
        cancellation_requested=lambda: cancelled[0],
    )
    actuator.open()
    writes_before_enable = list(transport.writes)
    cancelled[0] = True

    with pytest.raises(OperationCancelled, match="cancelled"):
        actuator.enable()

    assert transport.writes == writes_before_enable
    assert actuator.enabled is False


def test_enable_requires_exact_neutral_acknowledgement() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    corrupt_command = [True]

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        responses = firmware.handle_frame(frame)
        if isinstance(command, EnableCommand) and corrupt_command[0]:
            corrupt_command[0] = False
            return (encode_response(AckResponse(command.sequence, "ENABLE", 91, 90)),)
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()

    with pytest.raises(SerialActuatorProtocolError, match="physical neutral"):
        actuator.enable()

    assert actuator.enabled is False
    assert actuator.connected is False
    assert actuator.session_faulted is True
    assert actuator.position_uncertain is True
    assert (actuator.physical_pan_deg, actuator.physical_tilt_deg) == (90, 90)


def test_center_requires_exact_neutral_acknowledgement() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        responses = firmware.handle_frame(frame)
        if isinstance(command, CenterCommand):
            return (encode_response(AckResponse(command.sequence, "CENTER", 90, 91)),)
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()
    actuator.apply(PTZCommand(100.0, 80.0, sequence=1))

    with pytest.raises(SerialActuatorProtocolError, match="physical neutral"):
        actuator.center()

    assert (actuator.pan_deg, actuator.tilt_deg) == (100.0, 80.0)
    assert actuator.enabled is False
    assert actuator.connected is False
    assert actuator.session_faulted is True
    assert actuator.position_uncertain is True


def test_wrong_ack_command_and_set_angles_fail_without_advancing_state() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    corrupt = ["command"]

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        responses = firmware.handle_frame(frame)
        if isinstance(command, SetCommand) and corrupt:
            mode = corrupt.pop(0)
            if mode == "command":
                return (
                    encode_response(
                        AckResponse(
                            command.sequence,
                            "CENTER",
                            command.pan_deg,
                            command.tilt_deg,
                        )
                    ),
                )
            return (
                encode_response(
                    AckResponse(
                        command.sequence,
                        "SET",
                        command.pan_deg + 1,
                        command.tilt_deg,
                    )
                ),
            )
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()

    with pytest.raises(SerialActuatorProtocolError, match="expected ACK"):
        actuator.apply(PTZCommand(100.0, 80.0, sequence=1))

    assert (actuator.pan_deg, actuator.tilt_deg) == (90.0, 90.0)
    assert actuator.enabled is False
    assert actuator.connected is False
    assert actuator.session_faulted is True
    assert actuator.position_uncertain is True


def test_set_ack_with_wrong_angles_fails_without_advancing_state() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        responses = firmware.handle_frame(frame)
        if isinstance(command, SetCommand):
            return (
                encode_response(
                    AckResponse(
                        command.sequence,
                        "SET",
                        command.pan_deg + 1,
                        command.tilt_deg,
                    )
                ),
            )
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()

    with pytest.raises(SerialActuatorProtocolError, match="requested physical"):
        actuator.apply(PTZCommand(100.0, 80.0, sequence=1))

    assert (actuator.pan_deg, actuator.tilt_deg) == (90.0, 90.0)
    assert actuator.connected is False
    assert actuator.session_faulted is True
    assert actuator.position_uncertain is True


@pytest.mark.parametrize(
    "response",
    [
        StatusResponse(4, True, 0, 90, WatchdogState.ARMED),
        StatusResponse(4, True, 90, 90, WatchdogState.IDLE),
        StatusResponse(4, False, 90, 90, WatchdogState.ARMED),
    ],
)
def test_invalid_status_faults_session_without_mutating_confirmed_coordinates(
    response: StatusResponse,
) -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        responses = firmware.handle_frame(frame)
        if isinstance(command, StatusCommand):
            return (
                encode_response(
                    StatusResponse(
                        command.sequence,
                        response.enabled,
                        response.pan_deg,
                        response.tilt_deg,
                        response.watchdog_state,
                    )
                ),
            )
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()
    confirmed = (
        actuator.pan_deg,
        actuator.tilt_deg,
        actuator.physical_pan_deg,
        actuator.physical_tilt_deg,
    )

    with pytest.raises((SerialActuatorProtocolError, ValueError)):
        actuator.status()

    writes_after_status = len(transport.writes)
    with pytest.raises(SerialActuatorConnectionError, match="faulted"):
        actuator.apply(PTZCommand(100.0, 80.0, sequence=1))

    assert (
        actuator.pan_deg,
        actuator.tilt_deg,
        actuator.physical_pan_deg,
        actuator.physical_tilt_deg,
    ) == confirmed
    assert len(transport.writes) == writes_after_status
    assert actuator.connected is False
    assert actuator.enabled is False
    assert actuator.session_faulted is True
    assert actuator.position_uncertain is True


def test_successful_reopen_clears_protocol_fault_before_motion() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    corrupt_status = [True]

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        responses = firmware.handle_frame(frame)
        if isinstance(command, StatusCommand) and corrupt_status[0]:
            corrupt_status[0] = False
            return (
                encode_response(
                    StatusResponse(
                        command.sequence,
                        True,
                        0,
                        90,
                        WatchdogState.ARMED,
                    )
                ),
            )
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()

    with pytest.raises(SerialActuatorProtocolError):
        actuator.status()
    assert actuator.session_faulted is True

    actuator.open()
    actuator.enable()
    actuator.apply(PTZCommand(100.0, 80.0, sequence=1))

    assert actuator.connected is True
    assert actuator.enabled is True
    assert actuator.session_faulted is False
    assert actuator.position_uncertain is False
    assert (actuator.pan_deg, actuator.tilt_deg) == (100.0, 80.0)


def test_invalid_disable_ack_faults_session_and_blocks_motion() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    corrupt_disable = [False]

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        responses = firmware.handle_frame(frame)
        if isinstance(command, DisableCommand) and corrupt_disable[0]:
            corrupt_disable[0] = False
            return (encode_response(AckResponse(command.sequence, "DISABLE", 0, 90)),)
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()
    actuator.apply(PTZCommand(100.0, 80.0, sequence=1))
    confirmed = (
        actuator.pan_deg,
        actuator.tilt_deg,
        actuator.physical_pan_deg,
        actuator.physical_tilt_deg,
    )
    corrupt_disable[0] = True

    with pytest.raises(SerialActuatorProtocolError, match="physical pan"):
        actuator.disable()

    writes_after_disable = len(transport.writes)
    with pytest.raises(SerialActuatorConnectionError, match="faulted"):
        actuator.apply(PTZCommand(101.0, 81.0, sequence=2))

    assert len(transport.writes) == writes_after_disable
    assert (
        actuator.pan_deg,
        actuator.tilt_deg,
        actuator.physical_pan_deg,
        actuator.physical_tilt_deg,
    ) == confirmed
    assert actuator.connected is False
    assert actuator.enabled is False
    assert actuator.session_faulted is True
    assert actuator.position_uncertain is True


def test_faulted_session_cleanup_is_bounded_when_disable_and_recovery_are_silent() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    mode = ["normal"]
    now = [0.0]

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        if mode[0] == "silent":
            return ()
        responses = firmware.handle_frame(frame)
        if mode[0] == "corrupt" and isinstance(command, StatusCommand):
            return (
                encode_response(
                    StatusResponse(
                        command.sequence,
                        True,
                        0,
                        90,
                        WatchdogState.ARMED,
                    )
                ),
            )
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )
    actuator.open()
    actuator.enable()
    mode[0] = "corrupt"
    with pytest.raises(SerialActuatorProtocolError):
        actuator.status()

    mode[0] = "silent"
    actuator.close()

    assert now[0] == pytest.approx(config.serial.handshake_timeout_seconds)
    assert actuator.closed is True
    assert transport.close_count == 1


def test_not_enabled_error_and_expired_status_reconcile_enabled_state() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    transport = InMemorySerialTransport(firmware.handle_frame)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()
    firmware.enabled = False
    firmware.watchdog_state = WatchdogState.EXPIRED

    with pytest.raises(
        SerialActuatorProtocolError,
        match="NOT_ENABLED.*watchdog may have expired",
    ):
        actuator.apply(PTZCommand(100.0, 80.0, sequence=1))
    assert actuator.enabled is False

    actuator.enable()
    firmware.enabled = False
    firmware.watchdog_state = WatchdogState.EXPIRED
    status = actuator.status()

    assert status.watchdog_state is WatchdogState.EXPIRED
    assert actuator.enabled is False


def test_non_state_firmware_error_does_not_mutate_confirmed_state() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        responses = firmware.handle_frame(frame)
        if isinstance(command, SetCommand):
            return (encode_response(ErrorResponse(command.sequence, ErrorCode.OUT_OF_RANGE)),)
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()
    before = (
        actuator.enabled,
        actuator.pan_deg,
        actuator.tilt_deg,
        actuator.position_uncertain,
    )

    with pytest.raises(SerialActuatorProtocolError, match="OUT_OF_RANGE"):
        actuator.apply(PTZCommand(100.0, 80.0, sequence=1))

    assert (
        actuator.enabled,
        actuator.pan_deg,
        actuator.tilt_deg,
        actuator.position_uncertain,
    ) == before


def test_only_wrong_sequence_acknowledgements_end_in_conservative_timeout() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        responses = firmware.handle_frame(frame)
        if isinstance(command, SetCommand):
            stale = command.sequence - 1 or MAX_SEQUENCE
            return (encode_response(AckResponse(stale, "SET", command.pan_deg, command.tilt_deg)),)
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()

    with pytest.raises(SerialActuatorTimeoutError):
        actuator.apply(PTZCommand(100.0, 80.0, sequence=1))

    assert actuator.enabled is False
    assert actuator.position_uncertain is True
    assert (actuator.pan_deg, actuator.tilt_deg) == (90.0, 90.0)


def test_half_degree_rounding_and_physical_endpoints_are_explicit() -> None:
    config = load_config("configs/hardware.yaml")
    assert config.serial is not None
    serial = config.serial
    firmware = _firmware(config)
    now = [0.0]

    def sleep(duration: float) -> None:
        now[0] += duration

    transport = InMemorySerialTransport(firmware.handle_frame)
    actuator = _actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=sleep,
    )
    actuator.open()
    actuator.enable()

    actuator.apply(PTZCommand(90.5, 89.5, sequence=1))
    assert (actuator.physical_pan_deg, actuator.physical_tilt_deg) == (90, 91)

    minimum = (
        int(serial.physical_pan_limits.minimum_deg),
        int(serial.physical_tilt_limits.minimum_deg),
    )
    maximum = (
        int(serial.physical_pan_limits.maximum_deg),
        int(serial.physical_tilt_limits.maximum_deg),
    )
    actuator.apply(PTZCommand(float(maximum[0]), float(maximum[1]), sequence=2))
    assert (actuator.physical_pan_deg, actuator.physical_tilt_deg) == minimum

    actuator.apply(PTZCommand(float(minimum[0]), float(minimum[1]), sequence=3))
    assert (actuator.physical_pan_deg, actuator.physical_tilt_deg) == maximum


def test_handshake_and_disable_sequence_wraparound() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    transport = InMemorySerialTransport(firmware.handle_frame)
    actuator = _actuator(config, transport)
    actuator._next_sequence = MAX_SEQUENCE

    actuator.open()

    commands = [parse_command(frame) for frame in transport.writes]
    assert commands[0] == HelloCommand(MAX_SEQUENCE)
    assert commands[1] == DisableCommand(1)
    assert actuator.connected is True


def test_close_sends_disable_and_is_idempotent() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    transport = InMemorySerialTransport(firmware.handle_frame)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()

    actuator.close()
    actuator.close()

    assert isinstance(parse_command(transport.writes[-1]), DisableCommand)
    assert firmware.enabled is False
    assert transport.close_count == 1
    assert actuator.closed is True


def test_actuator_context_manager_opens_and_closes_transport() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    transport = InMemorySerialTransport(firmware.handle_frame)
    actuator = _actuator(config, transport)

    with actuator as active:
        assert active.connected is True
        active.enable()

    assert firmware.enabled is False
    assert transport.close_count == 1
    assert actuator.closed is True


def test_close_still_closes_when_firmware_does_not_answer_disable() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)
    disable_count = 0

    def responder(frame: bytes) -> tuple[bytes, ...]:
        nonlocal disable_count
        command = parse_command(frame)
        if isinstance(command, DisableCommand):
            disable_count += 1
            if disable_count > 1:
                return ()
        return firmware.handle_frame(frame)

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()

    actuator.close()

    assert transport.close_count == 1
    assert actuator.closed is True


def test_close_rehandshakes_before_disable_after_sequence_desynchronization() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = _firmware(config)

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        if isinstance(command, SetCommand):
            return ()
        return firmware.handle_frame(frame)

    transport = InMemorySerialTransport(responder)
    actuator = _actuator(config, transport)
    actuator.open()
    actuator.enable()

    with pytest.raises(SerialActuatorTimeoutError):
        actuator.apply(PTZCommand(100.0, 80.0, sequence=1))
    assert actuator.enabled is False

    actuator.close()

    assert firmware.enabled is False
    assert transport.close_count == 1
