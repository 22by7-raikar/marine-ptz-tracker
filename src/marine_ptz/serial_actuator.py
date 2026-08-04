"""Actuator-compatible Arduino serial backend with bounded injected transport."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from numbers import Real

from .cancellation import OperationCancelled
from .config import SerialConfig
from .serial_protocol import (
    MAX_FRAME_LENGTH,
    MIN_SEQUENCE,
    PROTOCOL_VERSION,
    AckResponse,
    CenterCommand,
    DisableCommand,
    EnableCommand,
    ErrorCode,
    ErrorResponse,
    HelloCommand,
    HostCommand,
    ProtocolError,
    ReadyResponse,
    SetCommand,
    StatusCommand,
    StatusResponse,
    WatchdogState,
    encode_command,
    next_sequence,
    parse_response,
)
from .serial_transport import (
    PySerialTransport,
    SerialTransport,
    SerialTransportError,
)
from .types import AngleLimits, PTZCommand

MAX_UNRELATED_RESPONSES = 8


def _never_cancel() -> bool:
    return False


class SerialActuatorError(RuntimeError):
    """Base failure raised by the host Arduino actuator."""


class SerialActuatorConnectionError(SerialActuatorError):
    """Raised when the injected serial transport is unavailable."""


class SerialActuatorTimeoutError(SerialActuatorError):
    """Raised after a bounded exchange or startup deadline expires."""


class SerialActuatorProtocolError(SerialActuatorError):
    """Raised for malformed, mismatched, or firmware-error responses."""


class ArduinoSerialActuator:
    """Map logical PTZ angles to bounded, acknowledged physical commands."""

    def __init__(
        self,
        settings: SerialConfig,
        pan_limits: AngleLimits,
        tilt_limits: AngleLimits,
        *,
        initial_pan_deg: float,
        initial_tilt_deg: float,
        transport: SerialTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        cancellation_requested: Callable[[], bool] = _never_cancel,
    ) -> None:
        if not pan_limits.contains(initial_pan_deg):
            raise ValueError("initial pan angle must be within logical pan limits")
        if not tilt_limits.contains(initial_tilt_deg):
            raise ValueError("initial tilt angle must be within logical tilt limits")
        self._settings = settings
        self._pan_limits = pan_limits
        self._tilt_limits = tilt_limits
        self._neutral_pan_deg = initial_pan_deg
        self._neutral_tilt_deg = initial_tilt_deg
        self._transport = transport
        self._monotonic = monotonic
        self._sleep = sleep
        if not callable(cancellation_requested):
            raise ValueError("cancellation_requested must be callable")
        self._cancellation_requested = cancellation_requested
        self._next_sequence = MIN_SEQUENCE
        self._connected = False
        self._enabled = False
        self._session_faulted = False
        self._closed = False
        self._position_uncertain = False
        self._startup_ready_seen = False
        self._last_set_time: float | None = None
        self._pan_deg = initial_pan_deg
        self._tilt_deg = initial_tilt_deg
        self._physical_pan_deg = _map_degree(
            initial_pan_deg,
            settings.pan_direction,
            settings.pan_offset_deg,
        )
        self._physical_tilt_deg = _map_degree(
            initial_tilt_deg,
            settings.tilt_direction,
            settings.tilt_offset_deg,
        )
        self._neutral_physical_pan_deg = int(settings.neutral_pan_deg)
        self._neutral_physical_tilt_deg = int(settings.neutral_tilt_deg)
        self._validate_physical_target(
            self._physical_pan_deg,
            self._physical_tilt_deg,
        )

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def session_faulted(self) -> bool:
        """Return whether protocol integrity requires an explicit re-handshake."""

        return self._session_faulted

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def position_uncertain(self) -> bool:
        """Return whether a motion may have been applied without confirmation."""

        return self._position_uncertain

    @property
    def startup_ready_seen(self) -> bool:
        """Return whether an unsolicited sequence-zero startup READY was observed."""

        return self._startup_ready_seen

    @property
    def pan_deg(self) -> float:
        return self._pan_deg

    @property
    def tilt_deg(self) -> float:
        return self._tilt_deg

    @property
    def physical_pan_deg(self) -> int:
        return self._physical_pan_deg

    @property
    def physical_tilt_deg(self) -> int:
        return self._physical_tilt_deg

    def set_cancellation_predicate(
        self,
        cancellation_requested: Callable[[], bool],
    ) -> None:
        """Bind a runtime-owned cancellation predicate without changing Actuator."""
        if not callable(cancellation_requested):
            raise ValueError("cancellation_requested must be callable")
        self._cancellation_requested = cancellation_requested

    def open(self) -> None:
        if self._closed:
            raise SerialActuatorConnectionError("serial actuator is closed")
        if self._connected:
            return
        self._startup_ready_seen = False
        transport = self._transport
        if transport is None:
            transport = PySerialTransport(
                self._settings.device,
                baudrate=self._settings.baudrate,
                read_timeout_s=self._settings.read_timeout_s,
                write_timeout_s=self._settings.write_timeout_s,
            )
            self._transport = transport
        try:
            open_transport = getattr(transport, "open", None)
            if callable(open_transport):
                open_transport()
            if not transport.is_open:
                raise SerialTransportError("serial transport did not open")

            # Clear bytes from an earlier process immediately after opening. A
            # fresh READY 0 emitted during the bounded boot grace remains valid.
            transport.reset_input_buffer()
            opened_at = self._monotonic()
            startup_deadline = opened_at + self._settings.handshake_timeout_seconds
            if bool(getattr(transport, "may_reset_on_open", False)):
                self._sleep_until(
                    min(
                        startup_deadline,
                        opened_at + self._settings.boot_grace_seconds,
                    )
                )
            sequence = self._take_sequence()
            ready = self._handshake(sequence, startup_deadline)
            if ready.protocol_version != PROTOCOL_VERSION:
                raise SerialActuatorProtocolError(
                    f"protocol version mismatch: host={PROTOCOL_VERSION}, "
                    f"firmware={ready.protocol_version}"
                )

            disable = self._acknowledged(
                DisableCommand(self._take_sequence()),
                "DISABLE",
            )
            try:
                logical_pan, logical_tilt = self._logical_ack_position(disable)
            except SerialActuatorProtocolError:
                self._session_integrity_fault(position_uncertain=True)
                raise
            self._update_physical(disable)
            self._pan_deg = logical_pan
            self._tilt_deg = logical_tilt
            self._enabled = False
            self._position_uncertain = False
            self._session_faulted = False
            self._last_set_time = None
            self._connected = True
        except Exception:
            self._connected = False
            self._enabled = False
            try:
                transport.close()
            except Exception:
                pass
            raise

    def enable(self) -> None:
        self._require_connected()
        self._raise_if_cancelled()
        response = self._acknowledged(EnableCommand(self._take_sequence()), "ENABLE")
        try:
            self._validate_neutral_ack(response, "ENABLE")
        except SerialActuatorProtocolError:
            self._session_integrity_fault(position_uncertain=True)
            raise
        self._update_physical(response)
        self._pan_deg = self._neutral_pan_deg
        self._tilt_deg = self._neutral_tilt_deg
        self._enabled = True
        self._position_uncertain = False
        self._last_set_time = None

    def disable(self) -> None:
        self._require_connected()
        self._disable_internal()

    def center(self) -> None:
        self._require_enabled()
        self._raise_if_cancelled()
        response = self._acknowledged(CenterCommand(self._take_sequence()), "CENTER")
        try:
            self._validate_neutral_ack(response, "CENTER")
        except SerialActuatorProtocolError:
            self._session_integrity_fault(position_uncertain=True)
            raise
        self._update_physical(response)
        self._pan_deg = self._neutral_pan_deg
        self._tilt_deg = self._neutral_tilt_deg
        self._position_uncertain = False

    def status(self) -> StatusResponse:
        self._require_connected()
        sequence = self._take_sequence()
        response = self._exchange(StatusCommand(sequence), StatusResponse)
        try:
            logical_pan, logical_tilt = self._validated_status(response)
        except SerialActuatorProtocolError:
            self._session_integrity_fault(position_uncertain=True)
            raise
        self._physical_pan_deg = response.pan_deg
        self._physical_tilt_deg = response.tilt_deg
        self._pan_deg = logical_pan
        self._tilt_deg = logical_tilt
        self._enabled = response.enabled
        self._position_uncertain = False
        if response.watchdog_state is WatchdogState.EXPIRED:
            self._enabled = False
        return response

    def apply(self, command: PTZCommand) -> None:
        self._require_enabled()
        self._raise_if_cancelled()
        pan = _finite_logical_degree(command.pan_deg, "command.pan_deg")
        tilt = _finite_logical_degree(command.tilt_deg, "command.tilt_deg")
        if not self._pan_limits.contains(pan):
            raise ValueError("command.pan_deg is outside logical pan limits")
        if not self._tilt_limits.contains(tilt):
            raise ValueError("command.tilt_deg is outside logical tilt limits")
        physical_pan = _map_degree(
            pan,
            self._settings.pan_direction,
            self._settings.pan_offset_deg,
        )
        physical_tilt = _map_degree(
            tilt,
            self._settings.tilt_direction,
            self._settings.tilt_offset_deg,
        )
        self._validate_physical_target(physical_pan, physical_tilt)
        self._wait_for_command_slot()
        self._raise_if_cancelled()
        response = self._acknowledged(
            SetCommand(self._take_sequence(), physical_pan, physical_tilt),
            "SET",
        )
        if (response.pan_deg, response.tilt_deg) != (physical_pan, physical_tilt):
            self._session_integrity_fault(position_uncertain=True)
            raise SerialActuatorProtocolError(
                "SET acknowledgement does not match the requested physical angles"
            )
        self._pan_deg = pan
        self._tilt_deg = tilt
        self._update_physical(response)
        self._position_uncertain = False
        self._last_set_time = self._monotonic()

    def close(self) -> None:
        if self._closed:
            return
        transport = self._transport
        try:
            if (
                (self._connected or self._session_faulted)
                and transport is not None
                and transport.is_open
            ):
                self._attempt_safe_disable_on_close()
        finally:
            self._connected = False
            self._enabled = False
            self._closed = True
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass

    def __enter__(self) -> ArduinoSerialActuator:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.close()

    def _handshake(self, sequence: int, deadline: float) -> ReadyResponse:
        transport = self._require_transport()
        command = HelloCommand(sequence)
        encoded = encode_command(command)
        while self._monotonic() < deadline:
            attempt_start = self._monotonic()
            attempt_deadline = min(
                deadline,
                attempt_start + self._settings.handshake_retry_interval_seconds,
            )
            try:
                transport.write(encoded)
                for _ in range(MAX_UNRELATED_RESPONSES):
                    remaining = max(0.0, attempt_deadline - self._monotonic())
                    frame = transport.read_frame(
                        MAX_FRAME_LENGTH,
                        timeout_s=remaining,
                    )
                    if frame is None:
                        break
                    try:
                        response = parse_response(frame)
                    except ProtocolError as exc:
                        raise SerialActuatorProtocolError(
                            f"malformed handshake response: {exc}"
                        ) from exc
                    if isinstance(response, ReadyResponse):
                        if response.sequence == 0:
                            self._startup_ready_seen = True
                            continue
                        if response.sequence != sequence:
                            raise SerialActuatorProtocolError(
                                "READY sequence does not match HELLO sequence: "
                                f"expected {sequence}, received {response.sequence}"
                            )
                        return response
                    if isinstance(response, ErrorResponse) and response.sequence in {
                        0,
                        sequence,
                    }:
                        raise SerialActuatorProtocolError(
                            f"firmware rejected HELLO sequence {sequence}: "
                            f"{response.error_code.value}"
                        )
            except SerialTransportError as exc:
                raise SerialActuatorConnectionError(str(exc)) from exc
            self._sleep_until(attempt_deadline)

        self._enabled = False
        raise SerialActuatorTimeoutError(
            f"no correlated READY for HELLO sequence {sequence} before "
            f"the {self._settings.handshake_timeout_seconds:g}-second "
            "startup deadline"
        )

    def _disable_internal(self) -> None:
        response = self._acknowledged(
            DisableCommand(self._take_sequence()),
            "DISABLE",
        )
        try:
            logical_pan, logical_tilt = self._logical_ack_position(response)
        except SerialActuatorProtocolError:
            self._session_integrity_fault(position_uncertain=True)
            raise
        self._update_physical(response)
        self._pan_deg = logical_pan
        self._tilt_deg = logical_tilt
        self._enabled = False
        self._last_set_time = None

    def _acknowledged(self, command: HostCommand, expected: str) -> AckResponse:
        response = self._exchange(command, AckResponse)
        if response.command != expected:
            self._session_integrity_fault(position_uncertain=True)
            raise SerialActuatorProtocolError(
                f"expected ACK for {expected}, received ACK for {response.command}"
            )
        return response

    def _exchange(
        self,
        command: HostCommand,
        expected_type: type[AckResponse] | type[StatusResponse],
    ) -> AckResponse | StatusResponse:
        transport = self._require_transport()
        encoded = encode_command(command)
        last_malformed: ProtocolError | None = None
        motion_command = isinstance(command, (SetCommand, CenterCommand, EnableCommand))
        motion_write_started = False
        for _attempt in range(self._settings.retries + 1):
            if motion_command:
                try:
                    self._raise_if_cancelled()
                except OperationCancelled:
                    if motion_write_started:
                        self._mark_uncertain_for(command)
                    raise
            try:
                transport.write(encoded)
                if motion_command:
                    motion_write_started = True
                for _ in range(MAX_UNRELATED_RESPONSES):
                    frame = transport.read_frame(MAX_FRAME_LENGTH)
                    if frame is None:
                        break
                    try:
                        response = parse_response(frame)
                    except ProtocolError as exc:
                        last_malformed = exc
                        break
                    if isinstance(response, ErrorResponse):
                        if response.sequence in {0, command.sequence}:
                            detail = ""
                            if response.error_code is ErrorCode.NOT_ENABLED:
                                self._enabled = False
                                detail = (
                                    " (firmware watchdog may have expired before "
                                    "the command was refreshed)"
                                )
                            raise SerialActuatorProtocolError(
                                f"firmware rejected sequence {command.sequence}: "
                                f"{response.error_code.value}{detail}"
                            )
                        continue
                    if isinstance(response, ReadyResponse):
                        if response.sequence == 0:
                            self._startup_ready_seen = True
                        continue
                    if response.sequence != command.sequence:
                        continue
                    if not isinstance(response, expected_type):
                        self._session_integrity_fault(position_uncertain=True)
                        raise SerialActuatorProtocolError(
                            f"unexpected response type {type(response).__name__} "
                            f"for sequence {command.sequence}"
                        )
                    return response
            except SerialTransportError as exc:
                self._connected = False
                self._enabled = False
                self._mark_uncertain_for(command)
                raise SerialActuatorConnectionError(str(exc)) from exc
        self._mark_uncertain_for(command)
        if last_malformed is not None:
            self._session_integrity_fault(position_uncertain=True)
            raise SerialActuatorProtocolError(
                f"malformed firmware response: {last_malformed}"
            ) from last_malformed
        raise SerialActuatorTimeoutError(
            f"no matching response for sequence {command.sequence} "
            f"after {self._settings.retries + 1} attempt(s); "
            "a motion command may have been applied"
        )

    def _take_sequence(self) -> int:
        sequence = self._next_sequence
        self._next_sequence = next_sequence(sequence)
        return sequence

    def _attempt_safe_disable_on_close(self) -> None:
        try:
            self._disable_internal()
            return
        except Exception:
            pass
        transport = self._transport
        if transport is None or not transport.is_open:
            return
        try:
            transport.reset_input_buffer()
            sequence = self._take_sequence()
            ready = self._handshake(
                sequence,
                self._monotonic() + self._settings.handshake_timeout_seconds,
            )
            if ready.protocol_version != PROTOCOL_VERSION:
                return
            self._disable_internal()
        except Exception:
            pass

    def _validated_status(
        self,
        response: StatusResponse,
    ) -> tuple[float, float]:
        try:
            self._validate_physical_target(response.pan_deg, response.tilt_deg)
        except ValueError as exc:
            raise SerialActuatorProtocolError(str(exc)) from exc
        if response.enabled and response.watchdog_state is not WatchdogState.ARMED:
            raise SerialActuatorProtocolError("enabled STATUS must report an ARMED watchdog")
        if not response.enabled and response.watchdog_state is WatchdogState.ARMED:
            raise SerialActuatorProtocolError("disabled STATUS cannot report an ARMED watchdog")
        logical_pan = _unmap_degree(
            response.pan_deg,
            self._settings.pan_direction,
            self._settings.pan_offset_deg,
        )
        logical_tilt = _unmap_degree(
            response.tilt_deg,
            self._settings.tilt_direction,
            self._settings.tilt_offset_deg,
        )
        if not self._pan_limits.contains(logical_pan):
            raise SerialActuatorProtocolError("STATUS pan angle maps outside logical pan limits")
        if not self._tilt_limits.contains(logical_tilt):
            raise SerialActuatorProtocolError("STATUS tilt angle maps outside logical tilt limits")
        return logical_pan, logical_tilt

    def _validate_neutral_ack(self, response: AckResponse, command: str) -> None:
        self._validate_ack_within_limits(response)
        if (response.pan_deg, response.tilt_deg) != (
            self._neutral_physical_pan_deg,
            self._neutral_physical_tilt_deg,
        ):
            raise SerialActuatorProtocolError(
                f"{command} acknowledgement must report configured physical neutral"
            )

    def _validate_ack_within_limits(self, response: AckResponse) -> None:
        try:
            self._validate_physical_target(response.pan_deg, response.tilt_deg)
        except ValueError as exc:
            raise SerialActuatorProtocolError(str(exc)) from exc

    def _logical_ack_position(self, response: AckResponse) -> tuple[float, float]:
        self._validate_ack_within_limits(response)
        logical_pan = _unmap_degree(
            response.pan_deg,
            self._settings.pan_direction,
            self._settings.pan_offset_deg,
        )
        logical_tilt = _unmap_degree(
            response.tilt_deg,
            self._settings.tilt_direction,
            self._settings.tilt_offset_deg,
        )
        if not self._pan_limits.contains(logical_pan):
            raise SerialActuatorProtocolError(
                "acknowledged pan angle maps outside logical pan limits"
            )
        if not self._tilt_limits.contains(logical_tilt):
            raise SerialActuatorProtocolError(
                "acknowledged tilt angle maps outside logical tilt limits"
            )
        return logical_pan, logical_tilt

    def _require_transport(self) -> SerialTransport:
        transport = self._transport
        if transport is None or not transport.is_open:
            raise SerialActuatorConnectionError("serial transport is not connected")
        return transport

    def _require_connected(self) -> None:
        if self._closed:
            raise SerialActuatorConnectionError("serial actuator is closed")
        if not self._connected:
            if self._session_faulted:
                raise SerialActuatorConnectionError(
                    "serial actuator session is faulted; call open() to re-handshake"
                )
            raise SerialActuatorConnectionError("serial actuator is not connected")

    def _require_enabled(self) -> None:
        self._require_connected()
        if not self._enabled:
            raise SerialActuatorError("serial actuator is disabled")

    def _wait_for_command_slot(self) -> None:
        self._raise_if_cancelled()
        if self._last_set_time is None:
            return
        interval = 1.0 / self._settings.command_rate_hz
        remaining = interval - (self._monotonic() - self._last_set_time)
        if remaining > 0.0:
            self._sleep(remaining)
            self._raise_if_cancelled()

    def _raise_if_cancelled(self) -> None:
        if self._cancellation_requested():
            raise OperationCancelled("serial motion dispatch cancelled before transport submission")

    def _sleep_until(self, deadline: float) -> None:
        remaining = deadline - self._monotonic()
        if remaining > 0.0:
            self._sleep(remaining)

    def _validate_physical_target(self, pan_deg: int, tilt_deg: int) -> None:
        if not self._settings.physical_pan_limits.contains(pan_deg):
            raise ValueError("mapped pan angle is outside physical pan limits")
        if not self._settings.physical_tilt_limits.contains(tilt_deg):
            raise ValueError("mapped tilt angle is outside physical tilt limits")

    def _update_physical(self, response: AckResponse) -> None:
        self._validate_ack_within_limits(response)
        self._physical_pan_deg = response.pan_deg
        self._physical_tilt_deg = response.tilt_deg

    def _session_integrity_fault(self, *, position_uncertain: bool) -> None:
        """Fail the local session closed without claiming firmware is disabled."""

        self._connected = False
        self._enabled = False
        self._session_faulted = True
        self._last_set_time = None
        if position_uncertain:
            self._position_uncertain = True

    def _mark_uncertain_for(self, command: HostCommand) -> None:
        self._enabled = False
        if isinstance(command, (SetCommand, CenterCommand, EnableCommand)):
            self._position_uncertain = True


def _finite_logical_degree(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    return number


def _map_degree(logical: float, direction: int, offset: float) -> int:
    mapped = direction * logical + offset
    if not math.isfinite(mapped):
        raise ValueError("mapped physical angle must be finite")
    return _round_half_away_from_zero(mapped)


def _unmap_degree(physical: int, direction: int, offset: float) -> float:
    logical = (physical - offset) / direction
    if not math.isfinite(logical):
        raise SerialActuatorProtocolError("unmapped logical angle must be finite")
    return logical


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))
