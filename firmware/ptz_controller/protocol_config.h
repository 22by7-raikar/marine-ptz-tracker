#ifndef MARINE_PTZ_PROTOCOL_CONFIG_H
#define MARINE_PTZ_PROTOCOL_CONFIG_H

#include <stdint.h>

// Controlled physical operation verified the 75..105 envelope. Revalidate it
// after mechanical changes; any expansion requires separate guided evidence.
constexpr uint8_t PAN_SERVO_PIN = 9;
constexpr uint8_t TILT_SERVO_PIN = 10;
constexpr uint32_t SERIAL_BAUD_RATE = 115200UL;

constexpr int PAN_MIN_DEG = 75;
constexpr int PAN_MAX_DEG = 105;
constexpr int TILT_MIN_DEG = 75;
constexpr int TILT_MAX_DEG = 105;
constexpr int PAN_NEUTRAL_DEG = 90;
constexpr int TILT_NEUTRAL_DEG = 90;

constexpr uint32_t WATCHDOG_TIMEOUT_MS = 1000UL;
constexpr uint8_t MAX_LINE_LENGTH = 63;
constexpr uint8_t MAX_RX_BYTES_PER_LOOP = 16;
constexpr uint8_t MAX_COMMANDS_PER_LOOP = 1;
constexpr uint8_t MAX_TX_BYTES_PER_LOOP = 16;

constexpr char PROTOCOL_VERSION[] = "1";
constexpr char FIRMWARE_VERSION[] = "0.1.0";

#endif
