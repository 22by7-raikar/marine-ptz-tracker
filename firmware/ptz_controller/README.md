# Arduino PTZ controller

This directory is reserved for the Arduino Uno R3 firmware. It is intentionally not implemented during repository bootstrap.

Firmware responsibilities:

- accept a versioned serial command with sequence number, pan angle, and tilt angle;
- reject malformed or out-of-range values using the calibrated limits in the deployed configuration;
- drive only the two SG90 signal pins; and
- return concise acknowledgements or errors for laptop telemetry.

The laptop remains responsible for vision and control policy. Establish the command framing, servo pins, neutral positions, angle limits, and watchdog behavior during bench testing before writing the firmware. See [wiring](../../docs/wiring.md) and [decisions](../../docs/decisions.md).
