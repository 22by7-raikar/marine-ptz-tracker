/**
 * @file ptz_controller.ino
 * @brief Arduino Uno sketch: binds the platform-neutral protocol to Serial
 *        and the two Servo outputs.
 *
 * Each loop() services the watchdog, processes a bounded slice of received
 * bytes, sends a bounded slice of the pending response, and services the
 * watchdog again, so slow or continuous serial traffic cannot delay expiry.
 */
#include <Servo.h>
#include <string.h>

#include "protocol.h"
#include "protocol_config.h"

// Servo V+ must come from the external regulated 5 V supply, never the Uno
// 5 V pin. The external supply ground and Uno ground must be common.
Servo panServo;
Servo tiltServo;
ProtocolState protocolState;
ProtocolFramer protocolFramer;

/// Response line plus CRLF, transmitted incrementally.
char pendingTx[MAX_LINE_LENGTH + 2U];
uint8_t pendingTxLength = 0U;
uint8_t pendingTxOffset = 0U;

/// True while part of the queued response is still unsent.
bool hasPendingResponse() {
  return pendingTxOffset < pendingTxLength;
}

/**
 * @brief Queue a response line (with CRLF) for transmission.
 * @return false if another response is still pending; true if queued or empty.
 */
bool queueResponse(const ProtocolResult &result) {
  if (result.response[0] == '\0') {
    return true;
  }
  if (hasPendingResponse()) {
    return false;
  }
  const size_t payloadLength = strlen(result.response);
  if (payloadLength > MAX_LINE_LENGTH) {
    return false;
  }
  memcpy(pendingTx, result.response, payloadLength);
  pendingTx[payloadLength] = '\r';
  pendingTx[payloadLength + 1U] = '\n';
  pendingTxLength = static_cast<uint8_t>(payloadLength + 2U);
  pendingTxOffset = 0U;
  return true;
}

/// Send at most #MAX_TX_BYTES_PER_LOOP bytes, never more than the TX buffer accepts.
void flushPendingResponseBounded() {
  if (!hasPendingResponse()) {
    return;
  }
  const int available = Serial.availableForWrite();
  if (available <= 0) {
    return;
  }
  uint8_t count =
      available > MAX_TX_BYTES_PER_LOOP
          ? MAX_TX_BYTES_PER_LOOP
          : static_cast<uint8_t>(available);
  const uint8_t remaining =
      static_cast<uint8_t>(pendingTxLength - pendingTxOffset);
  if (count > remaining) {
    count = remaining;
  }
  const size_t written = Serial.write(
      reinterpret_cast<const uint8_t *>(pendingTx + pendingTxOffset),
      count);
  const uint8_t accepted =
      written > count ? count : static_cast<uint8_t>(written);
  pendingTxOffset = static_cast<uint8_t>(pendingTxOffset + accepted);
  if (pendingTxOffset >= pendingTxLength) {
    pendingTxOffset = 0U;
    pendingTxLength = 0U;
  }
}

/// Apply a protocol side effect to the servos. Neutral is written before attach.
void applyAction(const ProtocolResult &result) {
  switch (result.action) {
    case ACTION_ATTACH_CENTER:
      panServo.write(protocolState.panDeg);
      tiltServo.write(protocolState.tiltDeg);
      if (!panServo.attached()) {
        panServo.attach(PAN_SERVO_PIN);
      }
      if (!tiltServo.attached()) {
        tiltServo.attach(TILT_SERVO_PIN);
      }
      panServo.write(protocolState.panDeg);
      tiltServo.write(protocolState.tiltDeg);
      break;
    case ACTION_WRITE:
      if (protocolState.enabled && panServo.attached() && tiltServo.attached()) {
        panServo.write(protocolState.panDeg);
        tiltServo.write(protocolState.tiltDeg);
      }
      break;
    case ACTION_DETACH:
      panServo.detach();
      tiltServo.detach();
      break;
    default:
      break;
  }
}

/// Detach the outputs if the protocol watchdog has expired.
void serviceWatchdog() {
  ProtocolResult result;
  protocolTickWatchdog(
      protocolState,
      static_cast<uint32_t>(millis()),
      result);
  applyAction(result);
}

/// Consume up to #MAX_RX_BYTES_PER_LOOP bytes and at most one command; skipped while a response is pending.
void processSerialBounded() {
  if (hasPendingResponse()) {
    return;
  }
  uint8_t bytesProcessed = 0U;
  uint8_t commandsProcessed = 0U;
  while (bytesProcessed < MAX_RX_BYTES_PER_LOOP &&
         commandsProcessed < MAX_COMMANDS_PER_LOOP &&
         !hasPendingResponse() &&
         Serial.available() > 0) {
    const int incoming = Serial.read();
    if (incoming < 0) {
      return;
    }
    ++bytesProcessed;
    const FrameEvent event =
        protocolFeedByte(protocolFramer, static_cast<uint8_t>(incoming));
    if (event == FRAME_EVENT_NONE) {
      continue;
    }

    ProtocolResult result;
    if (event == FRAME_EVENT_LINE) {
      protocolHandleLine(
          protocolState,
          protocolFramer.payload,
          static_cast<uint32_t>(millis()),
          result);
    } else {
      protocolFormatFramingError(event, result);
    }
    applyAction(result);
    if (queueResponse(result)) {
      ++commandsProcessed;
    }
  }
}

/// Start detached, open Serial, and queue the unsolicited READY line.
void setup() {
  protocolInit(protocolState);
  protocolFramerInit(protocolFramer);
  panServo.detach();
  tiltServo.detach();
  Serial.begin(SERIAL_BAUD_RATE);
  ProtocolResult startup;
  protocolFormatStartup(startup);
  queueResponse(startup);
}

/// One bounded scheduling pass; never blocks.
void loop() {
  serviceWatchdog();
  processSerialBounded();
  flushPendingResponseBounded();
  serviceWatchdog();
}
