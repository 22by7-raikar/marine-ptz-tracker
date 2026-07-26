#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "protocol.h"

namespace {

int hexNibble(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  return -1;
}

bool parseEvent(
    const char *argument,
    uint32_t &nowMs,
    const char *&hex) {
  const char *separator = strchr(argument, ':');
  if (separator == nullptr) {
    hex = argument;
    return true;
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long parsed = strtoul(argument, &end, 10);
  if (errno != 0 || end != separator || parsed > 0xFFFFFFFFUL) {
    return false;
  }
  nowMs = static_cast<uint32_t>(parsed);
  hex = separator + 1;
  return true;
}

void emitResponse(const ProtocolResult &result) {
  if (result.response[0] == '\0') {
    return;
  }
  fwrite(result.response, 1U, strlen(result.response), stdout);
  fwrite("\r\n", 1U, 2U, stdout);
}

void tick(ProtocolState &state, uint32_t nowMs) {
  ProtocolResult result;
  protocolTickWatchdog(state, nowMs, result);
}

}  // namespace

int main(int argc, char *argv[]) {
  if (argc < 2 || (strcmp(argv[1], "0") != 0 && strcmp(argv[1], "1") != 0)) {
    fprintf(stderr, "usage: protocol_harness STARTUP [MILLIS:HEX ...]\n");
    return 2;
  }

  ProtocolState state;
  ProtocolFramer framer;
  protocolInit(state);
  protocolFramerInit(framer);
  if (strcmp(argv[1], "1") == 0) {
    ProtocolResult startup;
    protocolFormatStartup(startup);
    emitResponse(startup);
  }

  uint32_t nowMs = 0UL;
  for (int index = 2; index < argc; ++index) {
    const char *hex = nullptr;
    if (!parseEvent(argv[index], nowMs, hex)) {
      fprintf(stderr, "invalid event: %s\n", argv[index]);
      return 2;
    }
    tick(state, nowMs);
    const size_t length = strlen(hex);
    if (length % 2U != 0U) {
      fprintf(stderr, "hex input has odd length\n");
      return 2;
    }
    for (size_t offset = 0U; offset < length; offset += 2U) {
      const int high = hexNibble(hex[offset]);
      const int low = hexNibble(hex[offset + 1U]);
      if (high < 0 || low < 0) {
        fprintf(stderr, "hex input contains an invalid digit\n");
        return 2;
      }
      const uint8_t value =
          static_cast<uint8_t>((static_cast<unsigned int>(high) << 4U) |
                               static_cast<unsigned int>(low));
      const FrameEvent event = protocolFeedByte(framer, value);
      if (event == FRAME_EVENT_NONE) {
        continue;
      }
      ProtocolResult result;
      if (event == FRAME_EVENT_LINE) {
        protocolHandleLine(state, framer.payload, nowMs, result);
      } else {
        protocolFormatFramingError(event, result);
      }
      emitResponse(result);
    }
    tick(state, nowMs);
  }
  return 0;
}
