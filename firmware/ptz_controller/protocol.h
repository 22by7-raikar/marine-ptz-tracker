#ifndef MARINE_PTZ_PROTOCOL_H
#define MARINE_PTZ_PROTOCOL_H

#include <stdint.h>

#include "protocol_config.h"

enum ProtocolAction : uint8_t {
  ACTION_NONE = 0,
  ACTION_ATTACH_CENTER,
  ACTION_WRITE,
  ACTION_DETACH,
};

enum WatchdogMode : uint8_t {
  WATCHDOG_IDLE = 0,
  WATCHDOG_ARMED,
  WATCHDOG_EXPIRED,
};

enum CommandKind : uint8_t {
  COMMAND_NONE = 0,
  COMMAND_SET,
  COMMAND_CENTER,
  COMMAND_ENABLE,
  COMMAND_DISABLE,
  COMMAND_STATUS,
};

enum FrameDiscardReason : uint8_t {
  FRAME_DISCARD_NONE = 0,
  FRAME_DISCARD_TOO_LONG,
  FRAME_DISCARD_ENCODING,
};

enum FrameEvent : uint8_t {
  FRAME_EVENT_NONE = 0,
  FRAME_EVENT_LINE,
  FRAME_EVENT_TOO_LONG,
  FRAME_EVENT_ENCODING,
};

struct ProtocolFramer {
  char payload[MAX_LINE_LENGTH + 1];
  uint8_t length;
  bool pendingCr;
  FrameDiscardReason discardReason;
};

struct ProtocolState {
  bool enabled;
  int16_t panDeg;
  int16_t tiltDeg;
  WatchdogMode watchdog;
  uint32_t watchdogDeadlineMs;
  bool hasSequence;
  uint16_t lastSequence;
  CommandKind lastCommand;
  int16_t lastPanArgument;
  int16_t lastTiltArgument;
  bool lastSetAccepted;
  char lastResponse[MAX_LINE_LENGTH + 1];
  bool hasRejectedSet;
  uint16_t rejectedSetSequence;
  int16_t rejectedSetPanArgument;
  int16_t rejectedSetTiltArgument;
  bool rejectedSetOutOfRange;
};

struct ProtocolResult {
  ProtocolAction action;
  char response[MAX_LINE_LENGTH + 1];
};

void protocolFramerInit(ProtocolFramer &framer);
FrameEvent protocolFeedByte(ProtocolFramer &framer, uint8_t value);

void protocolInit(ProtocolState &state);
void protocolFormatStartup(ProtocolResult &result);
void protocolFormatFramingError(FrameEvent event, ProtocolResult &result);
void protocolHandleLine(
    ProtocolState &state,
    char *line,
    uint32_t nowMs,
    ProtocolResult &result);
void protocolTickWatchdog(
    ProtocolState &state,
    uint32_t nowMs,
    ProtocolResult &result);

#endif
