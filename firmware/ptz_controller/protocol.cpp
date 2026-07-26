#include "protocol.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

namespace {

constexpr uint16_t MIN_SEQUENCE = 1U;
constexpr uint16_t MAX_SEQUENCE = 65535U;
constexpr uint8_t MAX_TOKENS = 5U;

void clearResult(ProtocolResult &result) {
  result.action = ACTION_NONE;
  result.response[0] = '\0';
}

void resetFramerState(ProtocolFramer &framer) {
  framer.length = 0U;
  framer.pendingCr = false;
  framer.discardReason = FRAME_DISCARD_NONE;
}

void formatError(ProtocolResult &result, uint16_t sequence, const char *code) {
  snprintf(
      result.response,
      sizeof(result.response),
      "ERR %u %s",
      static_cast<unsigned int>(sequence),
      code);
}

void formatAck(
    ProtocolResult &result,
    uint16_t sequence,
    const char *command,
    const ProtocolState &state) {
  snprintf(
      result.response,
      sizeof(result.response),
      "ACK %u %s %d %d",
      static_cast<unsigned int>(sequence),
      command,
      static_cast<int>(state.panDeg),
      static_cast<int>(state.tiltDeg));
}

const char *watchdogText(WatchdogMode watchdog) {
  switch (watchdog) {
    case WATCHDOG_ARMED:
      return "ARMED";
    case WATCHDOG_EXPIRED:
      return "EXPIRED";
    default:
      return "IDLE";
  }
}

void formatStatus(
    ProtocolResult &result,
    uint16_t sequence,
    const ProtocolState &state) {
  snprintf(
      result.response,
      sizeof(result.response),
      "STATUS %u %u %d %d %s",
      static_cast<unsigned int>(sequence),
      state.enabled ? 1U : 0U,
      static_cast<int>(state.panDeg),
      static_cast<int>(state.tiltDeg),
      watchdogText(state.watchdog));
}

bool parseInt32Strict(const char *token, int32_t &value) {
  if (token == nullptr || token[0] == '\0' || token[0] == '+') {
    return false;
  }
  const bool negative = token[0] == '-';
  const char *digits = negative ? token + 1 : token;
  if (digits[0] == '\0') {
    return false;
  }
  if (digits[0] == '0' && digits[1] != '\0') {
    return false;
  }
  if (negative && digits[0] == '0') {
    return false;
  }

  const uint32_t limit = negative ? 2147483648UL : 2147483647UL;
  uint32_t magnitude = 0UL;
  for (const char *cursor = digits; *cursor != '\0'; ++cursor) {
    if (*cursor < '0' || *cursor > '9') {
      return false;
    }
    const uint32_t digit = static_cast<uint32_t>(*cursor - '0');
    if (magnitude > (limit - digit) / 10UL) {
      return false;
    }
    magnitude = magnitude * 10UL + digit;
  }

  if (!negative) {
    value = static_cast<int32_t>(magnitude);
  } else if (magnitude == 2147483648UL) {
    value = static_cast<int32_t>(-2147483647L - 1L);
  } else {
    value = -static_cast<int32_t>(magnitude);
  }
  return true;
}

bool valueIsSequence(int32_t value) {
  return value >= static_cast<int32_t>(MIN_SEQUENCE) &&
         value <= static_cast<int32_t>(MAX_SEQUENCE);
}

uint16_t expectedSequence(uint16_t previous) {
  return previous == MAX_SEQUENCE ? MIN_SEQUENCE
                                  : static_cast<uint16_t>(previous + 1U);
}

uint8_t tokenizePreservingEmpty(
    char *line,
    char *tokens[MAX_TOKENS]) {
  uint8_t count = 1U;
  tokens[0] = line;
  for (char *cursor = line; *cursor != '\0'; ++cursor) {
    if (*cursor != ' ') {
      continue;
    }
    *cursor = '\0';
    if (count >= MAX_TOKENS) {
      return MAX_TOKENS;
    }
    tokens[count++] = cursor + 1;
  }
  return count;
}

bool hasEmptyToken(char *const tokens[MAX_TOKENS], uint8_t count) {
  for (uint8_t index = 0U; index < count; ++index) {
    if (tokens[index] == nullptr || tokens[index][0] == '\0') {
      return true;
    }
  }
  return false;
}

uint16_t bestEffortSequence(
    char *const tokens[MAX_TOKENS],
    uint8_t count) {
  if (count < 2U || tokens[1] == nullptr) {
    return 0U;
  }
  int32_t parsed = 0;
  if (!parseInt32Strict(tokens[1], parsed) || !valueIsSequence(parsed)) {
    return 0U;
  }
  return static_cast<uint16_t>(parsed);
}

CommandKind commandKind(const char *name) {
  if (strcmp(name, "SET") == 0) {
    return COMMAND_SET;
  }
  if (strcmp(name, "CENTER") == 0) {
    return COMMAND_CENTER;
  }
  if (strcmp(name, "ENABLE") == 0) {
    return COMMAND_ENABLE;
  }
  if (strcmp(name, "DISABLE") == 0) {
    return COMMAND_DISABLE;
  }
  if (strcmp(name, "STATUS") == 0) {
    return COMMAND_STATUS;
  }
  return COMMAND_NONE;
}

uint8_t expectedTokenCount(CommandKind command) {
  return command == COMMAND_SET ? 4U : 2U;
}

void armWatchdog(ProtocolState &state, uint32_t nowMs) {
  state.watchdog = WATCHDOG_ARMED;
  state.watchdogDeadlineMs =
      static_cast<uint32_t>(nowMs + WATCHDOG_TIMEOUT_MS);
}

bool deadlineReached(uint32_t nowMs, uint32_t deadlineMs) {
  return static_cast<uint32_t>(nowMs - deadlineMs) < 0x80000000UL;
}

void remember(
    ProtocolState &state,
    uint16_t sequence,
    CommandKind command,
    int16_t panArgument,
    int16_t tiltArgument,
    const ProtocolResult &result) {
  if (state.hasRejectedSet && sequence == state.rejectedSetSequence) {
    state.hasRejectedSet = false;
  }
  state.hasSequence = true;
  state.lastSequence = sequence;
  state.lastCommand = command;
  state.lastPanArgument = panArgument;
  state.lastTiltArgument = tiltArgument;
  state.lastSetAccepted =
      command == COMMAND_SET && result.action == ACTION_WRITE;
  strncpy(
      state.lastResponse,
      result.response,
      sizeof(state.lastResponse) - 1U);
  state.lastResponse[sizeof(state.lastResponse) - 1U] = '\0';
  if (command == COMMAND_SET && !state.lastSetAccepted) {
    state.hasRejectedSet = true;
    state.rejectedSetSequence = sequence;
    state.rejectedSetPanArgument = panArgument;
    state.rejectedSetTiltArgument = tiltArgument;
    state.rejectedSetOutOfRange =
        strstr(result.response, " OUT_OF_RANGE") != nullptr;
  }
}

bool handleDuplicate(
    ProtocolState &state,
    uint16_t sequence,
    CommandKind command,
    int16_t panArgument,
    int16_t tiltArgument,
    uint32_t nowMs,
    ProtocolResult &result) {
  if (!state.hasSequence || sequence != state.lastSequence) {
    return false;
  }
  if (command != state.lastCommand ||
      panArgument != state.lastPanArgument ||
      tiltArgument != state.lastTiltArgument) {
    formatError(result, sequence, "STALE_SEQUENCE");
    return true;
  }
  if (command == COMMAND_SET) {
    if (!state.lastSetAccepted) {
      strncpy(result.response, state.lastResponse, sizeof(result.response) - 1U);
      result.response[sizeof(result.response) - 1U] = '\0';
      return true;
    }
    if (!state.enabled) {
      formatError(result, sequence, "NOT_ENABLED");
      return true;
    }
    armWatchdog(state, nowMs);
    strncpy(result.response, state.lastResponse, sizeof(result.response) - 1U);
    result.response[sizeof(result.response) - 1U] = '\0';
    return true;
  }
  if ((command == COMMAND_CENTER || command == COMMAND_ENABLE) &&
      !state.enabled) {
    formatError(result, sequence, "NOT_ENABLED");
    return true;
  }
  if (command == COMMAND_STATUS) {
    formatStatus(result, sequence, state);
    return true;
  }
  strncpy(result.response, state.lastResponse, sizeof(result.response) - 1U);
  result.response[sizeof(result.response) - 1U] = '\0';
  return true;
}

bool handleCachedRejectedSet(
    const ProtocolState &state,
    uint16_t sequence,
    CommandKind command,
    int16_t panArgument,
    int16_t tiltArgument,
    ProtocolResult &result) {
  if (!state.hasRejectedSet ||
      sequence != state.rejectedSetSequence ||
      (state.hasSequence &&
       sequence == expectedSequence(state.lastSequence))) {
    return false;
  }
  if (command != COMMAND_SET ||
      panArgument != state.rejectedSetPanArgument ||
      tiltArgument != state.rejectedSetTiltArgument) {
    formatError(result, sequence, "STALE_SEQUENCE");
    return true;
  }
  formatError(
      result,
      sequence,
      state.rejectedSetOutOfRange ? "OUT_OF_RANGE" : "NOT_ENABLED");
  return true;
}

bool acceptNewSequence(
    const ProtocolState &state,
    uint16_t sequence,
    ProtocolResult &result) {
  if (!state.hasSequence || sequence != expectedSequence(state.lastSequence)) {
    formatError(result, sequence, "BAD_SEQUENCE");
    return false;
  }
  return true;
}

}  // namespace

void protocolFramerInit(ProtocolFramer &framer) {
  framer.payload[0] = '\0';
  resetFramerState(framer);
}

FrameEvent protocolFeedByte(ProtocolFramer &framer, uint8_t value) {
  if (value == static_cast<uint8_t>('\n')) {
    FrameEvent event = FRAME_EVENT_LINE;
    if (framer.discardReason == FRAME_DISCARD_TOO_LONG) {
      event = FRAME_EVENT_TOO_LONG;
    } else if (framer.discardReason == FRAME_DISCARD_ENCODING) {
      event = FRAME_EVENT_ENCODING;
    } else {
      framer.payload[framer.length] = '\0';
    }
    resetFramerState(framer);
    return event;
  }
  if (framer.discardReason != FRAME_DISCARD_NONE) {
    return FRAME_EVENT_NONE;
  }
  if (framer.pendingCr) {
    framer.length = 0U;
    framer.pendingCr = false;
    framer.discardReason = FRAME_DISCARD_ENCODING;
    return FRAME_EVENT_NONE;
  }
  if (value == static_cast<uint8_t>('\r')) {
    framer.pendingCr = true;
    return FRAME_EVENT_NONE;
  }
  if (value < 0x20U || value > 0x7EU) {
    framer.length = 0U;
    framer.discardReason = FRAME_DISCARD_ENCODING;
    return FRAME_EVENT_NONE;
  }
  if (framer.length >= MAX_LINE_LENGTH) {
    framer.length = 0U;
    framer.discardReason = FRAME_DISCARD_TOO_LONG;
    return FRAME_EVENT_NONE;
  }
  framer.payload[framer.length++] = static_cast<char>(value);
  return FRAME_EVENT_NONE;
}

void protocolInit(ProtocolState &state) {
  state.enabled = false;
  state.panDeg = static_cast<int16_t>(PAN_NEUTRAL_DEG);
  state.tiltDeg = static_cast<int16_t>(TILT_NEUTRAL_DEG);
  state.watchdog = WATCHDOG_IDLE;
  state.watchdogDeadlineMs = 0UL;
  state.hasSequence = false;
  state.lastSequence = 0U;
  state.lastCommand = COMMAND_NONE;
  state.lastPanArgument = 0;
  state.lastTiltArgument = 0;
  state.lastSetAccepted = false;
  state.lastResponse[0] = '\0';
  state.hasRejectedSet = false;
  state.rejectedSetSequence = 0U;
  state.rejectedSetPanArgument = 0;
  state.rejectedSetTiltArgument = 0;
  state.rejectedSetOutOfRange = false;
}

void protocolFormatStartup(ProtocolResult &result) {
  clearResult(result);
  snprintf(
      result.response,
      sizeof(result.response),
      "READY 0 %s %s",
      PROTOCOL_VERSION,
      FIRMWARE_VERSION);
}

void protocolFormatFramingError(FrameEvent event, ProtocolResult &result) {
  clearResult(result);
  if (event == FRAME_EVENT_TOO_LONG) {
    formatError(result, 0U, "LINE_TOO_LONG");
  } else {
    formatError(result, 0U, "NON_ASCII");
  }
}

void protocolTickWatchdog(
    ProtocolState &state,
    uint32_t nowMs,
    ProtocolResult &result) {
  clearResult(result);
  if (state.enabled &&
      state.watchdog == WATCHDOG_ARMED &&
      deadlineReached(nowMs, state.watchdogDeadlineMs)) {
    state.enabled = false;
    state.watchdog = WATCHDOG_EXPIRED;
    result.action = ACTION_DETACH;
  }
}

void protocolHandleLine(
    ProtocolState &state,
    char *line,
    uint32_t nowMs,
    ProtocolResult &result) {
  clearResult(result);
  char *tokens[MAX_TOKENS] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  const uint8_t count = tokenizePreservingEmpty(line, tokens);
  const char *name = tokens[0];

  if (name[0] == '\0') {
    formatError(result, 0U, "BAD_COMMAND");
    return;
  }

  const bool isHello = strcmp(name, "HELLO") == 0;
  const CommandKind command = commandKind(name);
  if (!isHello && command == COMMAND_NONE) {
    formatError(result, bestEffortSequence(tokens, count), "BAD_COMMAND");
    return;
  }

  const uint8_t expected = isHello ? 2U : expectedTokenCount(command);
  if (count != expected || hasEmptyToken(tokens, count)) {
    formatError(
        result,
        bestEffortSequence(tokens, count),
        "BAD_TOKEN_COUNT");
    return;
  }

  int32_t parsedSequence = 0;
  if (!parseInt32Strict(tokens[1], parsedSequence)) {
    formatError(result, 0U, "BAD_INTEGER");
    return;
  }
  if (!valueIsSequence(parsedSequence)) {
    formatError(result, 0U, "BAD_SEQUENCE");
    return;
  }
  const uint16_t sequence = static_cast<uint16_t>(parsedSequence);

  if (isHello) {
    snprintf(
        result.response,
        sizeof(result.response),
        "READY %u %s %s",
        static_cast<unsigned int>(sequence),
        PROTOCOL_VERSION,
        FIRMWARE_VERSION);
    state.hasRejectedSet = false;
    remember(state, sequence, COMMAND_NONE, 0, 0, result);
    return;
  }

  int16_t panArgument = 0;
  int16_t tiltArgument = 0;
  if (command == COMMAND_SET) {
    int32_t parsedPan = 0;
    int32_t parsedTilt = 0;
    if (!parseInt32Strict(tokens[2], parsedPan) ||
        !parseInt32Strict(tokens[3], parsedTilt)) {
      formatError(result, sequence, "BAD_INTEGER");
      return;
    }
    if (parsedPan < 0 || parsedPan > 180 ||
        parsedTilt < 0 || parsedTilt > 180) {
      formatError(result, sequence, "OUT_OF_RANGE");
      return;
    }
    panArgument = static_cast<int16_t>(parsedPan);
    tiltArgument = static_cast<int16_t>(parsedTilt);
  }

  if (handleDuplicate(
          state,
          sequence,
          command,
          panArgument,
          tiltArgument,
          nowMs,
          result)) {
    return;
  }
  if (handleCachedRejectedSet(
          state,
          sequence,
          command,
          panArgument,
          tiltArgument,
          result)) {
    return;
  }
  if (!acceptNewSequence(state, sequence, result)) {
    return;
  }

  switch (command) {
    case COMMAND_SET:
      if (panArgument < PAN_MIN_DEG ||
          panArgument > PAN_MAX_DEG ||
          tiltArgument < TILT_MIN_DEG ||
          tiltArgument > TILT_MAX_DEG) {
        formatError(result, sequence, "OUT_OF_RANGE");
      } else if (!state.enabled) {
        formatError(result, sequence, "NOT_ENABLED");
      } else {
        state.panDeg = panArgument;
        state.tiltDeg = tiltArgument;
        armWatchdog(state, nowMs);
        result.action = ACTION_WRITE;
        formatAck(result, sequence, "SET", state);
      }
      break;
    case COMMAND_CENTER:
      if (!state.enabled) {
        formatError(result, sequence, "NOT_ENABLED");
      } else {
        state.panDeg = static_cast<int16_t>(PAN_NEUTRAL_DEG);
        state.tiltDeg = static_cast<int16_t>(TILT_NEUTRAL_DEG);
        result.action = ACTION_WRITE;
        formatAck(result, sequence, "CENTER", state);
      }
      break;
    case COMMAND_ENABLE:
      state.enabled = true;
      state.panDeg = static_cast<int16_t>(PAN_NEUTRAL_DEG);
      state.tiltDeg = static_cast<int16_t>(TILT_NEUTRAL_DEG);
      armWatchdog(state, nowMs);
      result.action = ACTION_ATTACH_CENTER;
      formatAck(result, sequence, "ENABLE", state);
      break;
    case COMMAND_DISABLE:
      state.enabled = false;
      state.watchdog = WATCHDOG_IDLE;
      state.watchdogDeadlineMs = 0UL;
      result.action = ACTION_DETACH;
      formatAck(result, sequence, "DISABLE", state);
      break;
    case COMMAND_STATUS:
      formatStatus(result, sequence, state);
      break;
    default:
      formatError(result, sequence, "BAD_COMMAND");
      break;
  }
  remember(
      state,
      sequence,
      command,
      panArgument,
      tiltArgument,
      result);
}
