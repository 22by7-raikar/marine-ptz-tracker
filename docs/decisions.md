# Architecture decisions

## 2026-07-23: Keep hardware behind protocols

Camera, detector, target-selection, control, and actuator dependencies are represented by Python `Protocol` interfaces. This supports deterministic fakes and prevents test or import paths from requiring physical equipment.

## 2026-07-23: Use absolute bounded PTZ commands

`PTZCommand` carries pan and tilt degrees rather than servo pulses or relative motion. The controller can be tested independently, while the actuator and Arduino translate and enforce calibrated physical limits.

## 2026-07-23: Keep the Arduino narrow

The laptop performs vision and policy. The Arduino's future firmware will validate its serial protocol, reject unsafe angles, and drive servo signals. This keeps the time-sensitive actuator boundary simple and inspectable.

## 2026-07-23: Defer model and transport choices

No detection model, camera backend library, serial package, or firmware framework is chosen until the camera is available and bench measurements inform the choice. This avoids unnecessary downloads and premature dependencies before the August 4 deadline.

## 2026-07-24: Build the first slice with deterministic synthetic components

The initial runnable loop uses a metadata-only camera, repeatable moving bounding boxes, proportional image-error control, and an in-memory actuator. Detection generation, target selection, control policy, and actuation remain separate so each can be replaced and tested independently.

## 2026-07-24: Parse configuration with safe YAML and typed validation

PyYAML is the sole runtime dependency. Configuration is converted immediately into frozen dataclasses, rejects unknown keys and non-finite numeric values, enforces backend-required device settings, and reports field-qualified validation errors. This prevents untyped configuration mappings from spreading into tracking code.
