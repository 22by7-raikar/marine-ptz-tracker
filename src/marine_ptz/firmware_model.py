"""Pure models of the Arduino PTZ protocol, watchdog, framing, and loop policy."""

from __future__ import annotations

import math
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .serial_protocol import (
    MAX_SEQUENCE,
    MAX_WIRE_INTEGER,
    MIN_WIRE_INTEGER,
    PROTOCOL_VERSION,
    AckResponse,
    BadCommandError,
    CenterCommand,
    DegreeValueError,
    DisableCommand,
    EnableCommand,
    ErrorCode,
    ErrorResponse,
    FirmwareResponse,
    FrameEncodingError,
    FrameTooLongError,
    FramingIssue,
    HelloCommand,
    HostCommand,
    IntegerValueError,
    LineFramer,
    MalformedFrameError,
    ReadyResponse,
    SequenceValueError,
    SetCommand,
    StatusCommand,
    StatusResponse,
    TokenCountError,
    WatchdogState,
    encode_response,
    next_sequence,
    parse_command,
    validate_protocol_degree,
)

FIRMWARE_VERSION = "0.1.0"
RX_BYTES_PER_LOOP = 16
COMMANDS_PER_LOOP = 1
TX_BYTES_PER_LOOP = 16
_UINT32_MASK = 0xFFFF_FFFF
_UINT32_HALF_RANGE = 0x8000_0000
_CANONICAL_INTEGER = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")


@dataclass(frozen=True, slots=True)
class FirmwareSafetyConfig:
    pan_min_deg: int = 20
    pan_max_deg: int = 160
    tilt_min_deg: int = 35
    tilt_max_deg: int = 145
    neutral_pan_deg: int = 90
    neutral_tilt_deg: int = 90
    watchdog_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "pan_min_deg",
            "pan_max_deg",
            "tilt_min_deg",
            "tilt_max_deg",
            "neutral_pan_deg",
            "neutral_tilt_deg",
        ):
            validate_protocol_degree(getattr(self, name), name)
        if self.pan_min_deg >= self.pan_max_deg:
            raise ValueError("pan minimum must be less than pan maximum")
        if self.tilt_min_deg >= self.tilt_max_deg:
            raise ValueError("tilt minimum must be less than tilt maximum")
        if not self.pan_min_deg <= self.neutral_pan_deg <= self.pan_max_deg:
            raise ValueError("neutral pan must be within pan limits")
        if not self.tilt_min_deg <= self.neutral_tilt_deg <= self.tilt_max_deg:
            raise ValueError("neutral tilt must be within tilt limits")
        if (
            isinstance(self.watchdog_timeout_s, bool)
            or not isinstance(self.watchdog_timeout_s, (int, float))
            or not math.isfinite(float(self.watchdog_timeout_s))
            or not 0.0 < float(self.watchdog_timeout_s) <= 60.0
        ):
            raise ValueError("watchdog timeout must be greater than zero and at most 60 seconds")


class FirmwareStateMachine:
    """Deterministic command and watchdog model with no hardware dependencies."""

    def __init__(
        self,
        config: FirmwareSafetyConfig = FirmwareSafetyConfig(),
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self.enabled = False
        self.pan_deg = config.neutral_pan_deg
        self.tilt_deg = config.neutral_tilt_deg
        self.watchdog_state = WatchdogState.IDLE
        self._watchdog_deadline: float | None = None
        self._last_sequence: int | None = None
        self._last_command: HostCommand | None = None
        self._last_response: FirmwareResponse | None = None
        self._rejected_set_cache: tuple[SetCommand, ErrorResponse] | None = None

    @property
    def startup_response(self) -> ReadyResponse:
        return ReadyResponse(0, PROTOCOL_VERSION, FIRMWARE_VERSION)

    def startup_frame(self) -> bytes:
        return encode_response(self.startup_response)

    def tick(self) -> None:
        if (
            self.enabled
            and self._watchdog_deadline is not None
            and self._clock() >= self._watchdog_deadline
        ):
            self.enabled = False
            self.watchdog_state = WatchdogState.EXPIRED
            self._watchdog_deadline = None

    def handle_frame(self, frame: bytes) -> tuple[bytes, ...]:
        self.tick()
        try:
            command = parse_command(frame)
        except (
            BadCommandError,
            DegreeValueError,
            FrameEncodingError,
            FrameTooLongError,
            IntegerValueError,
            MalformedFrameError,
            SequenceValueError,
            TokenCountError,
        ) as exc:
            response = ErrorResponse(
                _best_effort_sequence(frame),
                _error_code_for_exception(exc),
            )
            return (encode_response(response),)
        return (encode_response(self.handle(command)),)

    def handle(self, command: HostCommand) -> FirmwareResponse:
        self.tick()
        if isinstance(command, HelloCommand):
            self._rejected_set_cache = None
            response = ReadyResponse(
                command.sequence,
                PROTOCOL_VERSION,
                FIRMWARE_VERSION,
            )
            self._remember(command, response)
            return response

        if self._last_sequence is None:
            return ErrorResponse(command.sequence, ErrorCode.BAD_SEQUENCE)
        if command.sequence == self._last_sequence:
            if command != self._last_command:
                return ErrorResponse(command.sequence, ErrorCode.STALE_SEQUENCE)
            return self._duplicate_response(command)
        expected_sequence = next_sequence(self._last_sequence)
        if (
            self._rejected_set_cache is not None
            and command.sequence == self._rejected_set_cache[0].sequence
            and command.sequence != expected_sequence
        ):
            cached_command, cached_response = self._rejected_set_cache
            if command != cached_command:
                return ErrorResponse(command.sequence, ErrorCode.STALE_SEQUENCE)
            return cached_response
        if command.sequence != expected_sequence:
            return ErrorResponse(command.sequence, ErrorCode.BAD_SEQUENCE)

        response = self._apply(command)
        self._remember(command, response)
        return response

    def _apply(self, command: HostCommand) -> FirmwareResponse:
        if isinstance(command, SetCommand):
            if not (
                self.config.pan_min_deg <= command.pan_deg <= self.config.pan_max_deg
                and self.config.tilt_min_deg <= command.tilt_deg <= self.config.tilt_max_deg
            ):
                return ErrorResponse(command.sequence, ErrorCode.OUT_OF_RANGE)
            if not self.enabled:
                return ErrorResponse(command.sequence, ErrorCode.NOT_ENABLED)
            self.pan_deg = command.pan_deg
            self.tilt_deg = command.tilt_deg
            self._arm_watchdog()
            return self._ack(command.sequence, "SET")

        if isinstance(command, EnableCommand):
            self.enabled = True
            self.pan_deg = self.config.neutral_pan_deg
            self.tilt_deg = self.config.neutral_tilt_deg
            self._arm_watchdog()
            return self._ack(command.sequence, "ENABLE")

        if isinstance(command, DisableCommand):
            self.enabled = False
            self.watchdog_state = WatchdogState.IDLE
            self._watchdog_deadline = None
            return self._ack(command.sequence, "DISABLE")

        if isinstance(command, CenterCommand):
            if not self.enabled:
                return ErrorResponse(command.sequence, ErrorCode.NOT_ENABLED)
            self.pan_deg = self.config.neutral_pan_deg
            self.tilt_deg = self.config.neutral_tilt_deg
            return self._ack(command.sequence, "CENTER")

        if isinstance(command, StatusCommand):
            return StatusResponse(
                command.sequence,
                self.enabled,
                self.pan_deg,
                self.tilt_deg,
                self.watchdog_state,
            )

        return ErrorResponse(command.sequence, ErrorCode.BAD_COMMAND)

    def _duplicate_response(self, command: HostCommand) -> FirmwareResponse:
        if isinstance(command, SetCommand):
            if isinstance(self._last_response, ErrorResponse):
                return self._last_response
            if not self.enabled:
                return ErrorResponse(command.sequence, ErrorCode.NOT_ENABLED)
            if isinstance(self._last_response, AckResponse):
                self._arm_watchdog()
                return self._last_response
        if isinstance(command, (CenterCommand, EnableCommand)):
            if not self.enabled:
                return ErrorResponse(command.sequence, ErrorCode.NOT_ENABLED)
        if isinstance(command, StatusCommand):
            return StatusResponse(
                command.sequence,
                self.enabled,
                self.pan_deg,
                self.tilt_deg,
                self.watchdog_state,
            )
        if self._last_response is None:
            return ErrorResponse(command.sequence, ErrorCode.INTERNAL)
        return self._last_response

    def _remember(
        self,
        command: HostCommand,
        response: FirmwareResponse,
    ) -> None:
        if (
            self._rejected_set_cache is not None
            and command.sequence == self._rejected_set_cache[0].sequence
        ):
            self._rejected_set_cache = None
        self._last_sequence = command.sequence
        self._last_command = command
        self._last_response = response
        if (
            isinstance(command, SetCommand)
            and isinstance(response, ErrorResponse)
            and response.error_code in {ErrorCode.OUT_OF_RANGE, ErrorCode.NOT_ENABLED}
        ):
            self._rejected_set_cache = (command, response)

    def _arm_watchdog(self) -> None:
        self.watchdog_state = WatchdogState.ARMED
        self._watchdog_deadline = self._clock() + self.config.watchdog_timeout_s

    def _ack(self, sequence: int, command: str) -> AckResponse:
        return AckResponse(sequence, command, self.pan_deg, self.tilt_deg)


class FirmwareStreamModel:
    """Apply the exact incremental line framer to a firmware state machine."""

    def __init__(self, machine: FirmwareStateMachine) -> None:
        self.machine = machine
        self._framer = LineFramer()

    def feed(self, data: bytes) -> tuple[bytes, ...]:
        responses: list[bytes] = []
        for event in self._framer.feed(data):
            if isinstance(event, FramingIssue):
                responses.append(
                    encode_response(ErrorResponse(0, _error_code_for_exception(event.error)))
                )
            else:
                responses.extend(self.machine.handle_frame(event))
        return tuple(responses)


class FirmwareRuntimeModel:
    """Model bounded RX work and resumable nonblocking TX in the Arduino loop."""

    def __init__(
        self,
        machine: FirmwareStateMachine,
        *,
        announce_startup: bool = True,
        rx_byte_budget: int = RX_BYTES_PER_LOOP,
        command_budget: int = COMMANDS_PER_LOOP,
        tx_byte_budget: int = TX_BYTES_PER_LOOP,
    ) -> None:
        if rx_byte_budget <= 0 or command_budget <= 0 or tx_byte_budget <= 0:
            raise ValueError("firmware loop budgets must be positive")
        self.machine = machine
        self._stream = FirmwareStreamModel(machine)
        self._rx: deque[int] = deque()
        self._pending = bytearray(machine.startup_frame() if announce_startup else b"")
        self._rx_byte_budget = rx_byte_budget
        self._command_budget = command_budget
        self._tx_byte_budget = tx_byte_budget

    @property
    def pending_output(self) -> bytes:
        return bytes(self._pending)

    @property
    def queued_rx_bytes(self) -> int:
        return len(self._rx)

    def feed_rx(self, data: bytes) -> None:
        self._rx.extend(data)

    def step(self, *, tx_capacity: int) -> bytes:
        if tx_capacity < 0:
            raise ValueError("tx_capacity must be zero or greater")
        self.machine.tick()

        if not self._pending:
            bytes_processed = 0
            commands_processed = 0
            while (
                self._rx
                and bytes_processed < self._rx_byte_budget
                and commands_processed < self._command_budget
                and not self._pending
            ):
                value = self._rx.popleft()
                bytes_processed += 1
                responses = self._stream.feed(bytes((value,)))
                if responses:
                    commands_processed += 1
                    self._pending.extend(responses[0])

        available = min(tx_capacity, self._tx_byte_budget)
        emitted = self._emit_pending(available)
        self.machine.tick()
        return emitted

    def _emit_pending(self, capacity: int) -> bytes:
        count = min(capacity, len(self._pending))
        if count == 0:
            return b""
        emitted = bytes(self._pending[:count])
        del self._pending[:count]
        return emitted


def millis_deadline_reached(now_ms: int, deadline_ms: int) -> bool:
    """Apply the firmware's unsigned-32-bit deadline comparison."""

    for value, name in ((now_ms, "now_ms"), (deadline_ms, "deadline_ms")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not 0 <= value <= _UINT32_MASK:
            raise ValueError(f"{name} must fit uint32")
    return ((now_ms - deadline_ms) & _UINT32_MASK) < _UINT32_HALF_RANGE


def millis_add(now_ms: int, interval_ms: int) -> int:
    if isinstance(interval_ms, bool) or not isinstance(interval_ms, int):
        raise ValueError("interval_ms must be an integer")
    if not 0 <= interval_ms < _UINT32_HALF_RANGE:
        raise ValueError("interval_ms must be between zero and 2^31-1")
    if isinstance(now_ms, bool) or not isinstance(now_ms, int):
        raise ValueError("now_ms must be an integer")
    if not 0 <= now_ms <= _UINT32_MASK:
        raise ValueError("now_ms must fit uint32")
    return (now_ms + interval_ms) & _UINT32_MASK


def _best_effort_sequence(frame: bytes) -> int:
    try:
        text = frame.decode("ascii")
    except UnicodeDecodeError:
        return 0
    tokens = text.split(" ")
    if len(tokens) < 2 or _CANONICAL_INTEGER.fullmatch(tokens[1]) is None:
        return 0
    value = int(tokens[1])
    if not MIN_WIRE_INTEGER <= value <= MAX_WIRE_INTEGER:
        return 0
    return value if 1 <= value <= MAX_SEQUENCE else 0


def _error_code_for_exception(error: Exception) -> ErrorCode:
    if isinstance(error, FrameTooLongError):
        return ErrorCode.LINE_TOO_LONG
    if isinstance(error, FrameEncodingError):
        return ErrorCode.NON_ASCII
    if isinstance(error, BadCommandError):
        return ErrorCode.BAD_COMMAND
    if isinstance(error, TokenCountError):
        return ErrorCode.BAD_TOKEN_COUNT
    if isinstance(error, IntegerValueError):
        return ErrorCode.BAD_INTEGER
    if isinstance(error, SequenceValueError):
        return ErrorCode.BAD_SEQUENCE
    if isinstance(error, DegreeValueError):
        return ErrorCode.OUT_OF_RANGE
    return ErrorCode.BAD_COMMAND
