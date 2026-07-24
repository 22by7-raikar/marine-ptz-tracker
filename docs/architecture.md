# Architecture

## Responsibility split

```text
USB camera -> CameraSource -> Detector -> TargetSelector -> PTZController -> Actuator -> Arduino -> servos
```

The Ubuntu laptop owns all image processing, target policy, control calculations, telemetry, and serial transport. The Arduino Uno is a narrow safety boundary: it receives validated pan/tilt commands, applies local limits, and drives two SG90 servos.

`src/marine_ptz/types.py` contains the stable values shared across layers. `interfaces.py` contains dependency-inversion boundaries. Concrete V4L2, GPU detector, serial Arduino, and simulated implementations belong outside these base modules and must be constructed explicitly by an application entry point.

## Data flow

1. `CameraSource.read()` produces a timestamped `Frame`.
2. `Detector.detect()` produces zero or more `Detection` values.
3. `TargetSelector.select()` chooses one `Target`, or no target.
4. `PTZController.command_for()` produces a bounded absolute `PTZCommand`.
5. `Actuator.apply()` sends that command to a simulated or physical mechanism.

The controller and Arduino should both enforce calibrated limits. A loss of target should result in an explicit policy (hold, neutral, or stop) selected after bench testing; it is not assumed by this bootstrap.
