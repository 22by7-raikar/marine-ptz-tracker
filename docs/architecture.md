# Architecture

## Responsibility split

```text
CameraSource -> Detector -> TargetSelector -> PTZController -> Actuator
     |                                                        |
synthetic/OpenCV                                  simulated now
Synthetic/YOLO detector                           Arduino serial foundation
```

The Ubuntu laptop owns all image processing, target policy, control calculations, telemetry, and serial transport. The Arduino Uno is a narrow safety boundary: it receives validated pan/tilt commands, applies local limits, and drives two SG90 servos.

`src/marine_ptz/types.py` contains the stable values shared across layers.
`Frame.image` is opaque and optional: synthetic frames remain lightweight,
while OpenCV frames carry NumPy arrays without importing NumPy in the base
package. Image payloads are excluded from frame equality, hashing, and repr;
the capture metadata defines frame value equality. `interfaces.py` contains
dependency-inversion boundaries. Synthetic components are composed by
`demo.py`; OpenCV capture, Ultralytics detection, marine selection, control,
and simulated actuation are composed by `vision_cli.py`. The host serial
actuator is intentionally not composed into either existing pipeline yet; it
is exercised only through an injected transport or the explicit serial tool.

## Data flow

1. `CameraSource.read()` produces a timestamped `Frame`.
2. `Detector.detect()` produces zero or more `Detection` values.
3. `TargetSelector.select()` chooses one `Target`, or no target.
4. `PTZController.command_for()` produces a bounded absolute `PTZCommand`.
5. `Actuator.apply()` sends that command to a simulated or physical mechanism.

The proportional controller applies per-axis pixel deadbands, normalized proportional gain, maximum step limits, and servo-angle limits. Positive image x/y error increases logical pan/tilt. `ArduinoSerialActuator` maps those logical values through per-axis directions and offsets to integer physical degrees. The host checks logical and physical limits; firmware independently rejects anything outside compile-time limits. Lost-target policy remains in the controller.

The serial boundary is split into a hardware-free codec/framer, a small
`SerialTransport` protocol, a deterministic firmware state model, and the
actuator. Pyserial is imported only when `PySerialTransport.open()` is called.
Opening an Uno port may trigger its normal auto-reset, so startup uses a
separate bounded boot grace and handshake deadline. Sequence-zero startup
`READY` is informational; a correlated `READY` and then correlated `DISABLE`
are required before the actuator reports connected.

Ordinary transport reads, ignored responses, writes, and retries are bounded;
there is no background thread or blocking output drain. Local write completion
means a frame reached the OS/driver; its correlated response is the end-to-end
confirmation. After a possibly applied command loses its acknowledgement, the
actuator keeps its last confirmed coordinates, reports position uncertainty,
and blocks further motion until explicit recovery. A semantically impossible
ACK or STATUS faults and disconnects the trusted local session without claiming
the physical firmware is disabled; a successful re-handshake plus correlated
`DISABLE` is required before motion can resume. The Arduino owns final range
enforcement, servo attachment, and the communication watchdog. Its loop bounds
RX bytes and commands, evaluates the watchdog before and after serial work, and
uses one aggregate fixed-buffer CRLF TX budget after RX processing.

For `CameraSource.read()`, `None` means evidence indicates normal end-of-file
for finite media. The OpenCV source compares usable frame-count and position
metadata when available. Unknown-length media failures raise `MediaReadError`;
recognized image files finish normally after their single successful frame.
Live-camera read failure raises `CameraReadError`. Every terminal read path
releases the capture, and OpenCV validates grayscale or supported channel
shapes at the capture boundary. Optional vision libraries and model weights
are loaded only when a real vision component is constructed.
