from __future__ import annotations

import pytest

from marine_ptz.serial_protocol import (
    MAX_FRAME_LENGTH,
    MAX_SEQUENCE,
    AckResponse,
    CenterCommand,
    DegreeValueError,
    DisableCommand,
    EnableCommand,
    ErrorCode,
    ErrorResponse,
    FrameEncodingError,
    FrameTooLongError,
    FramingIssue,
    HelloCommand,
    LineFramer,
    MalformedFrameError,
    ReadyResponse,
    SequenceValueError,
    SetCommand,
    StatusCommand,
    StatusResponse,
    WatchdogState,
    encode_command,
    encode_response,
    next_sequence,
    parse_command,
    parse_response,
)


@pytest.mark.parametrize(
    ("command", "encoded"),
    [
        (HelloCommand(1), b"HELLO 1\n"),
        (SetCommand(2, 90, 80), b"SET 2 90 80\n"),
        (CenterCommand(3), b"CENTER 3\n"),
        (EnableCommand(4), b"ENABLE 4\n"),
        (DisableCommand(5), b"DISABLE 5\n"),
        (StatusCommand(6), b"STATUS 6\n"),
    ],
)
def test_every_host_command_round_trips(command: object, encoded: bytes) -> None:
    assert encode_command(command) == encoded  # type: ignore[arg-type]
    assert parse_command(encoded) == command


@pytest.mark.parametrize(
    ("response", "encoded"),
    [
        (ReadyResponse(1, "1", "0.1.0"), b"READY 1 1 0.1.0\r\n"),
        (AckResponse(2, "SET", 90, 80), b"ACK 2 SET 90 80\r\n"),
        (AckResponse(3, "CENTER", 90, 90), b"ACK 3 CENTER 90 90\r\n"),
        (AckResponse(4, "ENABLE", 90, 90), b"ACK 4 ENABLE 90 90\r\n"),
        (AckResponse(5, "DISABLE", 90, 90), b"ACK 5 DISABLE 90 90\r\n"),
        (
            StatusResponse(6, True, 90, 80, WatchdogState.ARMED),
            b"STATUS 6 1 90 80 ARMED\r\n",
        ),
        (ErrorResponse(0, ErrorCode.NON_ASCII), b"ERR 0 NON_ASCII\r\n"),
    ],
)
def test_every_firmware_response_round_trips(
    response: object,
    encoded: bytes,
) -> None:
    assert encode_response(response) == encoded  # type: ignore[arg-type]
    assert parse_response(encoded) == response


@pytest.mark.parametrize("code", list(ErrorCode))
def test_every_stable_error_code_round_trips(code: ErrorCode) -> None:
    response = ErrorResponse(1, code)

    assert parse_response(encode_response(response)) == response


def test_parser_accepts_crlf_but_rejects_embedded_line_endings() -> None:
    assert parse_response(b"READY 1 1 0.1.0\r\n") == ReadyResponse(
        1,
        "1",
        "0.1.0",
    )

    with pytest.raises(FrameEncodingError, match="embedded"):
        parse_response(b"READY 1\r 1 0.1.0")


def test_incremental_framer_handles_partial_multiple_and_crlf_frames() -> None:
    framer = LineFramer()

    assert framer.feed(b"ACK 1 SET 90") == ()
    events = framer.feed(b" 80\r\nSTATUS 2 1 90 80 ARMED\nREADY 1 1 0.1.0")

    assert events == (
        b"ACK 1 SET 90 80",
        b"STATUS 2 1 90 80 ARMED",
    )
    assert framer.feed(b"\n") == (b"READY 1 1 0.1.0",)


def test_framer_discards_non_ascii_through_newline_then_recovers() -> None:
    framer = LineFramer()

    events = framer.feed(b"ACK 1 SET 90 \xffignored\nREADY 1 1 0.1.0\n")

    assert len(events) == 2
    assert isinstance(events[0], FramingIssue)
    assert isinstance(events[0].error, FrameEncodingError)
    assert events[1] == b"READY 1 1 0.1.0"


def test_framer_discards_overlong_line_then_recovers() -> None:
    framer = LineFramer()

    events = framer.feed(b"X" * (MAX_FRAME_LENGTH + 1) + b"\nREADY 1 1 0.1.0\n")

    assert len(events) == 2
    assert isinstance(events[0], FramingIssue)
    assert isinstance(events[0].error, FrameTooLongError)
    assert events[1] == b"READY 1 1 0.1.0"


@pytest.mark.parametrize(
    "frame",
    [
        b"",
        b"ACK 1 SET 90",
        b"STATUS 1 1 90 90",
        b"ERR 1",
        b"READY 1 1 0.1.0 extra",
        b"ACK  1 SET 90 90",
    ],
)
def test_invalid_response_token_counts_and_spacing_are_rejected(
    frame: bytes,
) -> None:
    with pytest.raises(MalformedFrameError):
        parse_response(frame)


@pytest.mark.parametrize(
    "frame",
    [
        b"ACK one SET 90 90",
        b"ACK 1 SET ninety 90",
        b"STATUS 1 yes 90 90 ARMED",
        b"ERR -1 BAD_COMMAND",
        b"ACK 01 SET 90 90",
    ],
)
def test_invalid_integer_tokens_are_rejected(frame: bytes) -> None:
    with pytest.raises((MalformedFrameError, SequenceValueError)):
        parse_response(frame)


@pytest.mark.parametrize(
    "frame",
    [
        b"SET 1 -1 90",
        b"SET 1 90 -1",
        b"SET 1 181 90",
        b"SET 1 90 181",
    ],
)
def test_negative_and_protocol_out_of_range_angles_are_rejected(frame: bytes) -> None:
    with pytest.raises(DegreeValueError):
        parse_command(frame)


def test_sequence_boundaries_and_wraparound_are_defined() -> None:
    assert next_sequence(1) == 2
    assert next_sequence(MAX_SEQUENCE) == 1
    assert HelloCommand(MAX_SEQUENCE).sequence == MAX_SEQUENCE

    with pytest.raises(SequenceValueError):
        HelloCommand(0)
    with pytest.raises(SequenceValueError):
        HelloCommand(MAX_SEQUENCE + 1)


def test_direct_parser_enforces_maximum_frame_length_and_ascii() -> None:
    with pytest.raises(FrameTooLongError):
        parse_response(b"X" * (MAX_FRAME_LENGTH + 1))
    with pytest.raises(FrameEncodingError):
        parse_response(b"READY 1 1 \xff")
