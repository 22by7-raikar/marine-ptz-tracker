"""Hardware-independent codec for the bounded Arduino PTZ serial protocol."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import TypeAlias


PROTOCOL_VERSION = "1"
MAX_FRAME_LENGTH = 63
MIN_SEQUENCE = 1
MAX_SEQUENCE = 65_535
MIN_PROTOCOL_DEGREES = 0
MAX_PROTOCOL_DEGREES = 180
MIN_WIRE_INTEGER = -(2**31)
MAX_WIRE_INTEGER = 2**31 - 1

_INTEGER_PATTERN = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class ProtocolError(ValueError):
    """Base error for protocol framing, encoding, and parsing failures."""


class FrameTooLongError(ProtocolError):
    """Raised when a frame exceeds the fixed protocol payload limit."""


class FrameEncodingError(ProtocolError):
    """Raised when a frame contains non-ASCII or invalid control bytes."""


class MalformedFrameError(ProtocolError):
    """Raised when a frame has invalid grammar."""


class BadCommandError(MalformedFrameError):
    """Raised for an empty or unknown command/response name."""


class TokenCountError(MalformedFrameError):
    """Raised when a known message has the wrong token layout."""


class IntegerValueError(MalformedFrameError):
    """Raised for a noncanonical or non-int32 wire integer."""


class SequenceValueError(ProtocolError):
    """Raised when a canonical sequence is outside the protocol sequence range."""


class DegreeValueError(ProtocolError):
    """Raised when an angle is outside the wire protocol range."""


class ErrorCode(str, Enum):
    BAD_COMMAND = "BAD_COMMAND"
    BAD_TOKEN_COUNT = "BAD_TOKEN_COUNT"
    BAD_INTEGER = "BAD_INTEGER"
    BAD_SEQUENCE = "BAD_SEQUENCE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    LINE_TOO_LONG = "LINE_TOO_LONG"
    NON_ASCII = "NON_ASCII"
    NOT_ENABLED = "NOT_ENABLED"
    STALE_SEQUENCE = "STALE_SEQUENCE"
    INTERNAL = "INTERNAL"


class WatchdogState(str, Enum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class HelloCommand:
    sequence: int

    def __post_init__(self) -> None:
        validate_sequence(self.sequence)


@dataclass(frozen=True, slots=True)
class SetCommand:
    sequence: int
    pan_deg: int
    tilt_deg: int

    def __post_init__(self) -> None:
        validate_sequence(self.sequence)
        validate_protocol_degree(self.pan_deg, "pan_deg")
        validate_protocol_degree(self.tilt_deg, "tilt_deg")


@dataclass(frozen=True, slots=True)
class CenterCommand:
    sequence: int

    def __post_init__(self) -> None:
        validate_sequence(self.sequence)


@dataclass(frozen=True, slots=True)
class EnableCommand:
    sequence: int

    def __post_init__(self) -> None:
        validate_sequence(self.sequence)


@dataclass(frozen=True, slots=True)
class DisableCommand:
    sequence: int

    def __post_init__(self) -> None:
        validate_sequence(self.sequence)


@dataclass(frozen=True, slots=True)
class StatusCommand:
    sequence: int

    def __post_init__(self) -> None:
        validate_sequence(self.sequence)


HostCommand: TypeAlias = (
    HelloCommand
    | SetCommand
    | CenterCommand
    | EnableCommand
    | DisableCommand
    | StatusCommand
)


@dataclass(frozen=True, slots=True)
class ReadyResponse:
    sequence: int
    protocol_version: str
    firmware_version: str

    def __post_init__(self) -> None:
        validate_error_sequence(self.sequence)
        _validate_version(self.protocol_version, "protocol_version")
        _validate_version(self.firmware_version, "firmware_version")


@dataclass(frozen=True, slots=True)
class AckResponse:
    sequence: int
    command: str
    pan_deg: int
    tilt_deg: int

    def __post_init__(self) -> None:
        validate_sequence(self.sequence)
        if self.command not in {"SET", "CENTER", "ENABLE", "DISABLE"}:
            raise MalformedFrameError(f"unsupported ACK command {self.command!r}")
        validate_protocol_degree(self.pan_deg, "pan_deg")
        validate_protocol_degree(self.tilt_deg, "tilt_deg")


@dataclass(frozen=True, slots=True)
class StatusResponse:
    sequence: int
    enabled: bool
    pan_deg: int
    tilt_deg: int
    watchdog_state: WatchdogState

    def __post_init__(self) -> None:
        validate_sequence(self.sequence)
        if not isinstance(self.enabled, bool):
            raise MalformedFrameError("enabled must be a boolean")
        validate_protocol_degree(self.pan_deg, "pan_deg")
        validate_protocol_degree(self.tilt_deg, "tilt_deg")
        if not isinstance(self.watchdog_state, WatchdogState):
            raise MalformedFrameError("watchdog_state must be a WatchdogState")


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    sequence: int
    error_code: ErrorCode

    def __post_init__(self) -> None:
        validate_error_sequence(self.sequence)
        if not isinstance(self.error_code, ErrorCode):
            raise MalformedFrameError("error_code must be a supported ErrorCode")


FirmwareResponse: TypeAlias = ReadyResponse | AckResponse | StatusResponse | ErrorResponse


@dataclass(frozen=True, slots=True)
class FramingIssue:
    """One discarded malformed input line and its stable framing error."""

    error: ProtocolError


FramingEvent: TypeAlias = bytes | FramingIssue


class LineFramer:
    """Incrementally split ASCII LF/CRLF frames with bounded recovery."""

    def __init__(self, maximum_length: int = MAX_FRAME_LENGTH) -> None:
        if (
            isinstance(maximum_length, bool)
            or not isinstance(maximum_length, int)
            or maximum_length <= 0
        ):
            raise ValueError("maximum_length must be a positive integer")
        self._maximum_length = maximum_length
        self._buffer = bytearray()
        self._discard_error: ProtocolError | None = None
        self._pending_cr = False

    def feed(self, data: bytes) -> tuple[FramingEvent, ...]:
        if not isinstance(data, bytes):
            raise TypeError("framer input must be bytes")
        events: list[FramingEvent] = []
        for value in data:
            if value == 0x0A:
                events.append(self._finish_line())
                continue
            if self._discard_error is not None:
                continue
            if self._pending_cr:
                self._discard_error = FrameEncodingError(
                    "carriage return must be followed immediately by newline"
                )
                self._pending_cr = False
                self._buffer.clear()
                continue
            if value == 0x0D:
                self._pending_cr = True
                continue
            if value < 0x20 or value > 0x7E:
                self._discard_error = FrameEncodingError(
                    "frame contains non-ASCII or control bytes"
                )
                self._buffer.clear()
                continue
            if len(self._buffer) >= self._maximum_length:
                self._discard_error = FrameTooLongError(
                    f"frame exceeds {self._maximum_length} bytes"
                )
                self._buffer.clear()
                continue
            self._buffer.append(value)
        return tuple(events)

    def reset(self) -> None:
        self._buffer.clear()
        self._discard_error = None
        self._pending_cr = False

    def _finish_line(self) -> FramingEvent:
        if self._discard_error is not None:
            event: FramingEvent = FramingIssue(self._discard_error)
        else:
            event = bytes(self._buffer)
        self.reset()
        return event


def validate_sequence(sequence: object) -> int:
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise SequenceValueError("sequence must be an integer")
    if not MIN_SEQUENCE <= sequence <= MAX_SEQUENCE:
        raise SequenceValueError(
            f"sequence must be between {MIN_SEQUENCE} and {MAX_SEQUENCE}"
        )
    return sequence


def validate_error_sequence(sequence: object) -> int:
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise SequenceValueError("error sequence must be an integer")
    if not 0 <= sequence <= MAX_SEQUENCE:
        raise SequenceValueError(
            f"error sequence must be between 0 and {MAX_SEQUENCE}"
        )
    return sequence


def next_sequence(sequence: int) -> int:
    validate_sequence(sequence)
    return MIN_SEQUENCE if sequence == MAX_SEQUENCE else sequence + 1


def validate_protocol_degree(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DegreeValueError(f"{field_name} must be an integer number of degrees")
    if not MIN_PROTOCOL_DEGREES <= value <= MAX_PROTOCOL_DEGREES:
        raise DegreeValueError(
            f"{field_name} must be between {MIN_PROTOCOL_DEGREES} "
            f"and {MAX_PROTOCOL_DEGREES}"
        )
    return value


def encode_command(command: HostCommand) -> bytes:
    if isinstance(command, HelloCommand):
        text = f"HELLO {command.sequence}"
    elif isinstance(command, SetCommand):
        text = f"SET {command.sequence} {command.pan_deg} {command.tilt_deg}"
    elif isinstance(command, CenterCommand):
        text = f"CENTER {command.sequence}"
    elif isinstance(command, EnableCommand):
        text = f"ENABLE {command.sequence}"
    elif isinstance(command, DisableCommand):
        text = f"DISABLE {command.sequence}"
    elif isinstance(command, StatusCommand):
        text = f"STATUS {command.sequence}"
    else:
        raise TypeError(f"unsupported host command type: {type(command).__name__}")
    return _encode_line(text, b"\n")


def encode_response(response: FirmwareResponse) -> bytes:
    if isinstance(response, ReadyResponse):
        text = (
            f"READY {response.sequence} "
            f"{response.protocol_version} {response.firmware_version}"
        )
    elif isinstance(response, AckResponse):
        text = (
            f"ACK {response.sequence} {response.command} "
            f"{response.pan_deg} {response.tilt_deg}"
        )
    elif isinstance(response, StatusResponse):
        text = (
            f"STATUS {response.sequence} {int(response.enabled)} "
            f"{response.pan_deg} {response.tilt_deg} {response.watchdog_state.value}"
        )
    elif isinstance(response, ErrorResponse):
        text = f"ERR {response.sequence} {response.error_code.value}"
    else:
        raise TypeError(
            f"unsupported firmware response type: {type(response).__name__}"
        )
    return _encode_line(text, b"\r\n")


def parse_command(frame: bytes | str) -> HostCommand:
    tokens = _tokens(frame)
    command_name = tokens[0]
    expected = {
        "HELLO": 2,
        "SET": 4,
        "CENTER": 2,
        "ENABLE": 2,
        "DISABLE": 2,
        "STATUS": 2,
    }.get(command_name)
    if expected is None:
        raise BadCommandError(f"unknown or empty host command {command_name!r}")
    _require_token_count(tokens, expected)
    sequence = _parsed_sequence(tokens[1])
    if command_name == "HELLO":
        return HelloCommand(sequence)
    if command_name == "SET":
        return SetCommand(
            sequence,
            _parsed_degree(tokens[2], "pan_deg"),
            _parsed_degree(tokens[3], "tilt_deg"),
        )
    if command_name == "CENTER":
        return CenterCommand(sequence)
    if command_name == "ENABLE":
        return EnableCommand(sequence)
    if command_name == "DISABLE":
        return DisableCommand(sequence)
    return StatusCommand(sequence)


def parse_response(frame: bytes | str) -> FirmwareResponse:
    tokens = _tokens(frame)
    response_name = tokens[0]
    if response_name == "READY":
        _require_token_count(tokens, 4)
        sequence = _parse_integer(tokens[1], "sequence")
        validate_error_sequence(sequence)
        return ReadyResponse(sequence, tokens[2], tokens[3])
    if response_name == "ACK":
        _require_token_count(tokens, 5)
        return AckResponse(
            _parsed_sequence(tokens[1]),
            tokens[2],
            _parsed_degree(tokens[3], "pan_deg"),
            _parsed_degree(tokens[4], "tilt_deg"),
        )
    if response_name == "STATUS":
        _require_token_count(tokens, 6)
        enabled_value = _parse_integer(tokens[2], "enabled")
        if enabled_value not in {0, 1}:
            raise MalformedFrameError("STATUS enabled must be 0 or 1")
        try:
            watchdog = WatchdogState(tokens[5])
        except ValueError as exc:
            raise MalformedFrameError(
                f"unsupported watchdog state {tokens[5]!r}"
            ) from exc
        return StatusResponse(
            _parsed_sequence(tokens[1]),
            bool(enabled_value),
            _parsed_degree(tokens[3], "pan_deg"),
            _parsed_degree(tokens[4], "tilt_deg"),
            watchdog,
        )
    if response_name == "ERR":
        _require_token_count(tokens, 3)
        sequence = _parse_integer(tokens[1], "sequence")
        validate_error_sequence(sequence)
        try:
            code = ErrorCode(tokens[2])
        except ValueError as exc:
            raise MalformedFrameError(f"unsupported error code {tokens[2]!r}") from exc
        return ErrorResponse(sequence, code)
    raise BadCommandError(f"unknown or empty firmware response {response_name!r}")


def _tokens(frame: bytes | str) -> list[str]:
    return _decode_frame(frame).split(" ")


def _decode_frame(frame: bytes | str) -> str:
    if isinstance(frame, bytes):
        raw = frame
    elif isinstance(frame, str):
        try:
            raw = frame.encode("ascii")
        except UnicodeEncodeError as exc:
            raise FrameEncodingError("frame must contain ASCII only") from exc
    else:
        raise TypeError("frame must be bytes or str")
    if raw.endswith(b"\n"):
        raw = raw[:-1]
        if raw.endswith(b"\r"):
            raw = raw[:-1]
    if b"\n" in raw or b"\r" in raw:
        raise FrameEncodingError("frame contains an embedded or incomplete line ending")
    if len(raw) > MAX_FRAME_LENGTH:
        raise FrameTooLongError(f"frame exceeds {MAX_FRAME_LENGTH} bytes")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise FrameEncodingError("frame must contain ASCII only") from exc
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in text):
        raise FrameEncodingError("frame contains non-printable ASCII")
    return text


def _encode_line(text: str, terminator: bytes) -> bytes:
    encoded = text.encode("ascii")
    if len(encoded) > MAX_FRAME_LENGTH:
        raise FrameTooLongError(f"frame exceeds {MAX_FRAME_LENGTH} bytes")
    return encoded + terminator


def _parse_integer(token: str, field_name: str) -> int:
    if _INTEGER_PATTERN.fullmatch(token) is None:
        raise IntegerValueError(f"{field_name} must be a canonical integer")
    value = int(token)
    if not MIN_WIRE_INTEGER <= value <= MAX_WIRE_INTEGER:
        raise IntegerValueError(
            f"{field_name} must fit the signed 32-bit wire integer domain"
        )
    return value


def _parsed_sequence(token: str) -> int:
    sequence = _parse_integer(token, "sequence")
    validate_sequence(sequence)
    return sequence


def _parsed_degree(token: str, field_name: str) -> int:
    value = _parse_integer(token, field_name)
    validate_protocol_degree(value, field_name)
    return value


def _require_token_count(tokens: list[str], expected: int) -> None:
    if len(tokens) != expected or any(not token for token in tokens):
        name = tokens[0] if tokens and tokens[0] else "message"
        raise TokenCountError(
            f"{name} requires exactly {expected} non-empty tokens"
        )


def _validate_version(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _VERSION_PATTERN.fullmatch(value) is None:
        raise MalformedFrameError(
            f"{field_name} must contain only letters, digits, dot, underscore, or dash"
        )
