/**
 * @file protocol.h
 * @brief Platform-neutral PTZ serial protocol: line framing, command parsing,
 *        sequencing, and the output watchdog.
 *
 * This module has no Arduino dependencies. The sketch feeds it bytes and
 * applies the returned ProtocolAction to the servos; host tests compile the
 * same source and compare its output against shared golden vectors.
 *
 * All buffers are fixed-size and no function allocates memory.
 */
#ifndef MARINE_PTZ_PROTOCOL_H
#define MARINE_PTZ_PROTOCOL_H

#include <stdint.h>

#include "protocol_config.h"

/// Servo-output side effect the caller must apply after a protocol step.
enum ProtocolAction : uint8_t {
  ACTION_NONE = 0,       ///< No output change.
  ACTION_ATTACH_CENTER,  ///< Write neutral, then attach both servos.
  ACTION_WRITE,          ///< Write the current state angles to attached servos.
  ACTION_DETACH,         ///< Detach both servos (outputs go limp).
};

/// Output watchdog state, reported by `STATUS`.
enum WatchdogMode : uint8_t {
  WATCHDOG_IDLE = 0,  ///< Outputs disabled by command or never enabled.
  WATCHDOG_ARMED,     ///< Outputs enabled; an accepted `SET` must arrive before the deadline.
  WATCHDOG_EXPIRED,   ///< Deadline passed; outputs were detached until the next `ENABLE`.
};

/// Recognized sequenced commands. `HELLO` is handled separately.
enum CommandKind : uint8_t {
  COMMAND_NONE = 0,
  COMMAND_SET,      ///< `SET <seq> <pan> <tilt>`: move to absolute angles.
  COMMAND_CENTER,   ///< `CENTER <seq>`: move to neutral.
  COMMAND_ENABLE,   ///< `ENABLE <seq>`: attach at neutral and arm the watchdog.
  COMMAND_DISABLE,  ///< `DISABLE <seq>`: detach immediately.
  COMMAND_STATUS,   ///< `STATUS <seq>`: report state without side effects.
};

/// Why the framer is discarding input until the next LF.
enum FrameDiscardReason : uint8_t {
  FRAME_DISCARD_NONE = 0,
  FRAME_DISCARD_TOO_LONG,  ///< Payload exceeded #MAX_LINE_LENGTH.
  FRAME_DISCARD_ENCODING,  ///< Non-printable byte or CR not followed by LF.
};

/// Result of feeding one byte to the framer.
enum FrameEvent : uint8_t {
  FRAME_EVENT_NONE = 0,  ///< Byte consumed; no complete line yet.
  FRAME_EVENT_LINE,      ///< ProtocolFramer::payload holds a complete line.
  FRAME_EVENT_TOO_LONG,  ///< A discarded over-length line ended.
  FRAME_EVENT_ENCODING,  ///< A discarded malformed line ended.
};

/// Incremental LF / CRLF line framer with a fixed payload buffer.
struct ProtocolFramer {
  char payload[MAX_LINE_LENGTH + 1];  ///< NUL-terminated line on FRAME_EVENT_LINE.
  uint8_t length;                     ///< Bytes accumulated so far.
  bool pendingCr;                     ///< A CR was seen and must be followed by LF.
  FrameDiscardReason discardReason;   ///< Non-NONE while skipping to the next LF.
};

/**
 * @brief Complete protocol and output state.
 *
 * Besides the commanded angles and watchdog, it caches the last sequenced
 * command so that an exact retransmission (same sequence, same arguments) is
 * answered idempotently instead of being applied twice.
 */
struct ProtocolState {
  bool enabled;                 ///< Servos attached and accepting `SET`.
  int16_t panDeg;               ///< Current commanded pan angle (degrees).
  int16_t tiltDeg;              ///< Current commanded tilt angle (degrees).
  WatchdogMode watchdog;        ///< Output watchdog state.
  uint32_t watchdogDeadlineMs;  ///< millis() value at which an armed watchdog expires.
  bool hasSequence;             ///< A `HELLO` has started a sequenced session.
  uint16_t lastSequence;        ///< Sequence of the last accepted command.
  CommandKind lastCommand;      ///< Command kind cached for duplicate detection.
  int16_t lastPanArgument;      ///< Pan argument cached for duplicate detection.
  int16_t lastTiltArgument;     ///< Tilt argument cached for duplicate detection.
  bool lastSetAccepted;         ///< Whether the cached command was an accepted `SET`.
  char lastResponse[MAX_LINE_LENGTH + 1];  ///< Response replayed for duplicates.
  bool hasRejectedSet;                     ///< A rejected `SET` is cached for retries.
  uint16_t rejectedSetSequence;            ///< Sequence of the cached rejected `SET`.
  int16_t rejectedSetPanArgument;          ///< Pan argument of the cached rejected `SET`.
  int16_t rejectedSetTiltArgument;         ///< Tilt argument of the cached rejected `SET`.
  bool rejectedSetOutOfRange;  ///< Cached rejection was OUT_OF_RANGE (else NOT_ENABLED).
};

/// Output of one protocol step: a side effect and an optional response line.
struct ProtocolResult {
  ProtocolAction action;              ///< Servo action for the caller to apply.
  char response[MAX_LINE_LENGTH + 1];  ///< Response payload without CRLF; empty for none.
};

/// Reset the framer to an empty buffer.
void protocolFramerInit(ProtocolFramer &framer);

/**
 * @brief Feed one received byte to the framer.
 * @return FRAME_EVENT_LINE when ProtocolFramer::payload holds a complete line,
 *         a discard event when a malformed line ends, otherwise FRAME_EVENT_NONE.
 */
FrameEvent protocolFeedByte(ProtocolFramer &framer, uint8_t value);

/// Reset protocol state: outputs disabled, neutral angles, no session.
void protocolInit(ProtocolState &state);

/// Format the unsolicited `READY 0 <protocol> <firmware>` startup line.
void protocolFormatStartup(ProtocolResult &result);

/// Format `ERR 0 LINE_TOO_LONG` or `ERR 0 NON_ASCII` for a framing event.
void protocolFormatFramingError(FrameEvent event, ProtocolResult &result);

/**
 * @brief Parse and execute one complete command line.
 *
 * Validates grammar, canonical integers, sequence order, duplicates, and the
 * compile-time angle limits, in the precedence documented in the firmware
 * README. Only an accepted `SET` (or its exact duplicate) refreshes the
 * watchdog.
 *
 * @param state   Protocol state, updated in place.
 * @param line    NUL-terminated payload; tokenized in place.
 * @param nowMs   Current millis() value.
 * @param result  Receives the servo action and response line.
 */
void protocolHandleLine(
    ProtocolState &state,
    char *line,
    uint32_t nowMs,
    ProtocolResult &result);

/**
 * @brief Expire the watchdog if its deadline has passed.
 *
 * Rollover-safe. On expiry, disables outputs, enters WATCHDOG_EXPIRED, and
 * returns ACTION_DETACH exactly once.
 */
void protocolTickWatchdog(
    ProtocolState &state,
    uint32_t nowMs,
    ProtocolResult &result);

#endif
