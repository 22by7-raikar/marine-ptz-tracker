# Architecture

## Responsibility split

```text
CameraSource -> Detector -> TargetSelector -> PTZController -> Actuator
     |                                                        |
synthetic/OpenCV                                  simulated now
Synthetic/YOLO detector                           Arduino later
```

The Ubuntu laptop owns all image processing, target policy, control calculations, telemetry, and serial transport. The Arduino Uno is a narrow safety boundary: it receives validated pan/tilt commands, applies local limits, and drives two SG90 servos.

`src/marine_ptz/types.py` contains the stable values shared across layers.
`Frame.image` is opaque and optional: synthetic frames remain lightweight,
while OpenCV frames carry NumPy arrays without importing NumPy in the base
package. Image payloads are excluded from frame equality, hashing, and repr;
the capture metadata defines frame value equality. `interfaces.py` contains
dependency-inversion boundaries. Synthetic components are composed by
`demo.py`; OpenCV capture, Ultralytics detection, marine selection, control,
and simulated actuation are composed by `vision_cli.py`.

## Data flow

1. `CameraSource.read()` produces a timestamped `Frame`.
2. `Detector.detect()` produces zero or more `Detection` values.
3. `TargetSelector.select()` chooses one `Target`, or no target.
4. `PTZController.command_for()` produces a bounded absolute `PTZCommand`.
5. `Actuator.apply()` sends that command to a simulated or physical mechanism.

The proportional controller applies per-axis pixel deadbands, normalized proportional gain, maximum step limits, and servo-angle limits. Positive image x/y error increases logical pan/tilt; a future hardware actuator may map those logical directions to its wiring. Lost-target policy is explicit and supports holding position or returning toward neutral. The simulated actuator independently clamps commands as a defensive boundary; the future Arduino must do the same.

For `CameraSource.read()`, `None` means evidence indicates normal end-of-file
for finite media. The OpenCV source compares usable frame-count and position
metadata when available. Unknown-length media failures raise `MediaReadError`;
recognized image files finish normally after their single successful frame.
Live-camera read failure raises `CameraReadError`. Every terminal read path
releases the capture, and OpenCV validates grayscale or supported channel
shapes at the capture boundary. Optional vision libraries and model weights
are loaded only when a real vision component is constructed.
