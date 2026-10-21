/**
 * @file protocol_config.h
 * @brief Compile-time hardware, safety, and scheduling constants.
 *
 * These values are the firmware's final safety boundary: the host cannot
 * override them. `tests/test_firmware_model.py` checks that they agree with
 * `configs/hardware.yaml`.
 */
#ifndef MARINE_PTZ_PROTOCOL_CONFIG_H
#define MARINE_PTZ_PROTOCOL_CONFIG_H

#include <stdint.h>

// The 75..105 envelope was verified on the rig. Recalibrate after mechanical
// changes; widen it only through the staged procedure in docs/hardware.md.
constexpr uint8_t PAN_SERVO_PIN = 9;             ///< Pan servo signal pin.
constexpr uint8_t TILT_SERVO_PIN = 10;           ///< Tilt servo signal pin.
constexpr uint32_t SERIAL_BAUD_RATE = 115200UL;  ///< 8N1 serial rate.

constexpr int PAN_MIN_DEG = 75;       ///< Lowest accepted pan angle (degrees).
constexpr int PAN_MAX_DEG = 105;      ///< Highest accepted pan angle (degrees).
constexpr int TILT_MIN_DEG = 75;      ///< Lowest accepted tilt angle (degrees).
constexpr int TILT_MAX_DEG = 105;     ///< Highest accepted tilt angle (degrees).
constexpr int PAN_NEUTRAL_DEG = 90;   ///< Pan angle for `ENABLE` and `CENTER`.
constexpr int TILT_NEUTRAL_DEG = 90;  ///< Tilt angle for `ENABLE` and `CENTER`.

/// Outputs detach if no accepted `SET` arrives within this interval.
constexpr uint32_t WATCHDOG_TIMEOUT_MS = 1000UL;
/// Maximum payload bytes per line, excluding LF or CRLF.
constexpr uint8_t MAX_LINE_LENGTH = 63;
/// Receive bytes processed per loop() iteration.
constexpr uint8_t MAX_RX_BYTES_PER_LOOP = 16;
/// Complete commands processed per loop() iteration.
constexpr uint8_t MAX_COMMANDS_PER_LOOP = 1;
/// Transmit bytes queued per loop() iteration.
constexpr uint8_t MAX_TX_BYTES_PER_LOOP = 16;

constexpr char PROTOCOL_VERSION[] = "1";      ///< Wire grammar version.
constexpr char FIRMWARE_VERSION[] = "0.1.0";  ///< Firmware release version.

#endif
