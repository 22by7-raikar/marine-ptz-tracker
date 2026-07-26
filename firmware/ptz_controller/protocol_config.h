#ifndef MARINE_PTZ_PROTOCOL_CONFIG_H
#define MARINE_PTZ_PROTOCOL_CONFIG_H

#include <stdint.h>

// Candidate bench values. Confirm pins, directions, limits, and neutral angles
// against the delivered mechanism before attaching powered servos.
constexpr uint8_t PAN_SERVO_PIN = 9;
constexpr uint8_t TILT_SERVO_PIN = 10;
constexpr uint32_t SERIAL_BAUD_RATE = 115200UL;

constexpr int PAN_MIN_DEG = 20;
constexpr int PAN_MAX_DEG = 160;
constexpr int TILT_MIN_DEG = 35;
constexpr int TILT_MAX_DEG = 145;
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
