# Arduino PTZ controller protocol

This directory contains a compile-verified firmware foundation for an Arduino
Uno R3-compatible controller and two hobby positional servos. The
platform-neutral protocol core is compiled and executed by host CI, and the
sketch compiles for the Uno, but it has not been uploaded or bench-tested.
Candidate pins, directions, servo models, current draw, limits, and neutral
positions require confirmation with the delivered hardware.

## Wire format

- 115200 baud, 8 data bits, no parity, one stop bit.
- Printable ASCII payloads only.
- One command or response per line.
- Host commands may end in LF or CRLF. Firmware responses always end in CRLF.
- Payload length is at most 63 ASCII bytes, excluding LF or CRLF.
- CR is a terminator prefix, never payload, and must be followed immediately by
  LF. Split CR/LF reads are valid.
- Angles are absolute integer degrees. Pulse widths are not exposed.
- Sequences are 1 through 65535 and wrap from 65535 to 1. Sequence zero is
  reserved for unsolicited startup and errors without a recoverable sequence.

Grammar:

```text
HELLO   <sequence>
SET     <sequence> <pan_degrees> <tilt_degrees>
CENTER  <sequence>
ENABLE  <sequence>
DISABLE <sequence>
STATUS  <sequence>

READY  <sequence_or_zero> <protocol_version> <firmware_version>
ACK    <sequence> <command> <pan_degrees> <tilt_degrees>
STATUS <sequence> <enabled> <pan_degrees> <tilt_degrees> <watchdog_state>
ERR    <sequence_or_zero> <error_code>
```

`enabled` is `0` or `1`. Watchdog state is `IDLE`, `ARMED`, or `EXPIRED`.
Protocol version is `1`; the current firmware-foundation version is `0.1.0`.

Startup and correlated handshake examples:

```text
READY 0 1 0.1.0
HELLO 1
READY 1 1 0.1.0
DISABLE 2
ACK 2 DISABLE 90 90
```

Other command and response examples:

```text
ENABLE 3
ACK 3 ENABLE 90 90
SET 4 100 80
ACK 4 SET 100 80
CENTER 5
ACK 5 CENTER 90 90
STATUS 6
STATUS 6 1 90 90 ARMED
ERR 4 OUT_OF_RANGE
ERR 0 LINE_TOO_LONG
```

## Integer domain and error precedence

Wire integers are canonical signed 32-bit decimal values
`-2147483648..2147483647`. A leading plus, leading zero on a multi-digit value,
and negative zero are rejected. `0` is the only canonical zero. Values outside
the signed 32-bit domain are `BAD_INTEGER`; representable angles outside the
wire or calibrated safety range are `OUT_OF_RANGE`.

Errors are selected in this exact order:

1. Framing, encoding, line length, or invalid CR use: `NON_ASCII` or
   `LINE_TOO_LONG`, sequence zero.
2. Empty or unknown command: `BAD_COMMAND`.
3. Known command with wrong token count, empty token, or extra separator:
   `BAD_TOKEN_COUNT`.
4. Noncanonical or non-int32 sequence token: `BAD_INTEGER`.
5. Canonical sequence outside 1–65535: `BAD_SEQUENCE`.
6. Noncanonical or non-int32 angle token: `BAD_INTEGER`.
7. Representable angle outside the wire range 0–180: `OUT_OF_RANGE`.
8. Same-sequence duplicate rules, then expected-successor validation.
9. Wire-valid angle outside calibrated limits: `OUT_OF_RANGE`.
10. Motion requiring detached outputs: `NOT_ENABLED`.

A valid sequence is included in an error when it can be recovered without
changing the selected error. Syntax/framing errors do not advance firmware
sequence state. Valid, sequenced application rejections such as calibrated
`OUT_OF_RANGE` and `NOT_ENABLED` do advance it.

Stable error codes:

| Code | Meaning |
| --- | --- |
| `BAD_COMMAND` | Empty or unknown command. |
| `BAD_TOKEN_COUNT` | Wrong token count, empty token, or extra separator. |
| `BAD_INTEGER` | Noncanonical or non-int32 integer. |
| `BAD_SEQUENCE` | Canonical sequence outside the range or expected session order. |
| `OUT_OF_RANGE` | Representable angle outside wire or calibrated limits. |
| `LINE_TOO_LONG` | Payload exceeded 63 bytes. |
| `NON_ASCII` | Non-ASCII/control input or invalid CR framing. |
| `NOT_ENABLED` | A motion command requires detached outputs. |
| `STALE_SEQUENCE` | A duplicate sequence carried different valid content. |
| `INTERNAL` | Reserved for an internal invariant failure. |

## Reset-tolerant handshake

Opening an Uno USB serial device may toggle its reset circuitry. The host does
not suppress or manually pulse DTR/RTS. Hardware flow control is disabled, and
the startup policy deliberately tolerates the normal reset:

1. Open the port and immediately clear bytes left by an earlier process.
2. Allow the configured bounded boot grace.
3. Re-send the identical `HELLO` sequence at the configured retry interval
   until the total handshake deadline.
4. Record or ignore unsolicited `READY 0`, but require
   `READY <hello_sequence> 1 <firmware_version>`.
5. Require a correlated `DISABLE` acknowledgement.
6. Mark the host connected only after both correlated steps succeed.

Prototype defaults are a 2-second boot grace, 4-second total handshake
deadline, and 0.25-second retry interval. These startup values are separate
from ordinary 0.25-second command reads and two command retries.

## Duplicate and watchdog policy

Retries retransmit the exact encoded command and sequence.

- An identical duplicate of a previously acknowledged `SET` does not move
  again and, while outputs remain enabled, does refresh the watchdog. After
  expiry or another disabled state it returns `NOT_ENABLED` and never
  re-enables or moves the servos. This supports the applied-command/lost-ACK
  case without bypassing the enable boundary.
- An identical duplicate of a rejected `SET` returns the cached rejection and
  does not refresh the watchdog, even after a later state change or command.
- Different content under the same sequence returns `STALE_SEQUENCE` and does
  not refresh the watchdog.
- Malformed, out-of-range, stale, and disabled commands never refresh it.
- Duplicate `STATUS` recomputes current state; `STATUS` itself never refreshes.

Firmware starts with both Servo outputs detached. `ENABLE` attaches at
configured neutral and arms the watchdog. An accepted `SET` refreshes it.
`DISABLE` detaches immediately. At the 1000 ms compile-time deadline, firmware
enters `EXPIRED`, disables output, and detaches exactly once. A new sequenced
`ENABLE` is required to recover.

A host timeout may occur after a motion was physically applied. The host keeps
its last confirmed coordinates, marks position uncertain, blocks further
`apply()` calls by clearing local enabled state, and relies on bounded
re-handshake/disable or watchdog expiry for recovery.

The watchdog is a software fallback, not a certified emergency stop.

## Bounded firmware scheduling

The sketch evaluates the watchdog before and after serial work. Each loop
processes at most 16 RX bytes and one complete command. It does not process
another command while a response is pending.

Responses are formatted into a fixed buffer with CRLF. Serial RX is serviced
first when no response is pending, allowing a response generated in the
current iteration to be queued. One subsequent TX service sends an aggregate
maximum of 16 bytes per loop and never more than
`Serial.availableForWrite()` reports. Partial output resumes in later
iterations, so a slow/full TX ring or continuous RX cannot keep the watchdog
service from running. There is no `delay()`, unbounded RX drain, Arduino
`String`, or dynamic allocation.

Overflow or encoding failure discards input through the next LF before a
following frame is accepted. Exactly 63 payload bytes plus LF or CRLF is valid;
64 payload bytes is rejected.

## Compatibility and verification

Protocol version `1` defines the grammar and semantics above. Incompatible
grammar or semantic changes require another protocol version; firmware version
changes independently.

`protocol.cpp` and `protocol.h` are platform-neutral C++11 with fixed-width
types and fixed buffers. Hardware-free tests compile them with strong host
compiler warnings and compare their exact output against independently
maintained golden vectors also used by the Python model.

The compile-only Arduino baseline uses Arduino CLI `1.5.1`, Arduino AVR Boards
`arduino:avr@1.8.8`, the official `Servo@1.3.0` library, and FQBN
`arduino:avr:uno`:

```bash
ARDUINO_BUILD_DIR="$(mktemp -d)"
arduino-cli compile \
  --fqbn arduino:avr:uno \
  --warnings all \
  --build-path "${ARDUINO_BUILD_DIR}" \
  firmware/ptz_controller
```

The pre-remediation result was 7,856 bytes of flash (24%) and 705 bytes of
static RAM (34%). The current result, including the rejected-`SET` retry cache,
is 8,120 bytes of flash (25%) and 715 bytes of static RAM (34%). Current
warnings come only from the official AVR core's `new.cpp`; project sources
produce no warnings. Firmware upload and physical verification remain pending.

Servo power must come from the external regulated 5 V supply with a common
Arduino ground. See [wiring](../../docs/wiring.md) before applying power.
