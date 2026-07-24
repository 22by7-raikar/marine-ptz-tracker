# Architecture

## Responsibility split

```text
CameraSource -> Detector -> TargetSelector -> PTZController -> Actuator
     |                                                        |
synthetic now                                      simulated now
USB later                                         Arduino later
```

The Ubuntu laptop owns all image processing, target policy, control calculations, telemetry, and serial transport. The Arduino Uno is a narrow safety boundary: it receives validated pan/tilt commands, applies local limits, and drives two SG90 servos.

`src/marine_ptz/types.py` contains the stable values shared across layers. `interfaces.py` contains dependency-inversion boundaries. The synthetic camera/detector, tracking policy, and simulated actuator are separate implementations composed by `demo.py`. Future V4L2, model, and serial implementations remain outside the base modules and must be constructed explicitly by an application entry point.

## Data flow

1. `CameraSource.read()` produces a timestamped `Frame`.
2. `Detector.detect()` produces zero or more `Detection` values.
3. `TargetSelector.select()` chooses one `Target`, or no target.
4. `PTZController.command_for()` produces a bounded absolute `PTZCommand`.
5. `Actuator.apply()` sends that command to a simulated or physical mechanism.

The proportional controller applies per-axis pixel deadbands, normalized proportional gain, maximum step limits, and servo-angle limits. Positive image x/y error increases logical pan/tilt; a future hardware actuator may map those logical directions to its wiring. Lost-target policy is explicit and supports holding position or returning toward neutral. The simulated actuator independently clamps commands as a defensive boundary; the future Arduino must do the same.
