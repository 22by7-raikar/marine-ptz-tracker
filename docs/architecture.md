# Architecture

## Responsibility split

```text
CameraSource -> Detector -> TargetSelector -> PTZController -> Actuator
     |                                                        |
synthetic/OpenCV                                  simulated (safe default)
Synthetic/YOLO detector                           explicitly armed Arduino serial
```

The Ubuntu laptop owns all image processing, target policy, control calculations, telemetry, and serial transport. The Arduino Uno is a narrow safety boundary: it receives validated pan/tilt commands, applies local limits, and drives two SG90 servos.

`src/marine_ptz/types.py` contains the stable values shared across layers.
`Frame.image` is opaque and optional: synthetic frames remain lightweight,
while OpenCV frames carry NumPy arrays without importing NumPy in the base
package. Image payloads are excluded from frame equality, hashing, and repr;
the capture metadata defines frame value equality. `interfaces.py` contains
dependency-inversion boundaries. Synthetic components are composed by
`demo.py`. `vision_cli.py` is the single real-vision composition root for
OpenCV capture, Ultralytics detection, marine selection, the existing
controller, telemetry, and either simulated or explicitly armed serial
actuation. Backend construction is centralized rather than spread through the
tracking loop.

## Data flow

1. `CameraSource.read()` produces a timestamped `Frame`.
2. `Detector.detect()` produces zero or more `Detection` values.
3. `TargetSelector.select()` chooses one `Target`, or no target.
4. `PTZController.command_for()` produces a bounded absolute `PTZCommand`.
5. `Actuator.apply()` sends that command to a simulated or physical mechanism.

For physical actuation the startup order is fixed:

1. parse and validate configuration and CLI interlocks;
2. construct the OpenCV source and Ultralytics model;
3. construct selection and control policy;
4. open serial and complete correlated `HELLO`/`READY`/`DISABLE`;
5. call `ENABLE` only when the backend is `arduino_serial` and
   `--arm-hardware` was supplied; and
6. enter the loop.

Camera or model construction failure therefore occurs before serial opening or
servo enablement. A serial handshake failure closes resources without calling
`ENABLE`.

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

Runtime shutdown is coordinated in ordinary control flow. A local signal
handler records only a termination request; the loop performs cleanup. Motion
shutdown is attempted first, then the actuator transport, writer, source, and
display resources are closed idempotently. Finite-media EOF, frame limits, and
display `q` return CLI status 0. SIGINT and SIGTERM are operationally expected
shutdown requests after bounded cleanup, but `main()` returns their conventional
Unix statuses 130 and 143. A component-raised `OperationCancelled` is normal
only when the shared termination predicate is true; otherwise it is an
actionable failure chained to the original exception and returns status 2.
Detection, malformed frame, camera, serial, actuator, annotation, writer, and
other unexpected failures also return status 2. `run_vision()` returns a
processed-frame count only; `main()` owns the process-status contract. Cleanup
failures are reported alongside the primary failure rather than silently
replacing it. The same precedence applies to runtime observers: a control-path
failure remains authoritative, while a processing-end observer failure and any
cleanup failures are retained in a Python-3.10-compatible
`RuntimeSecondaryFailures` chained diagnostic. If normal completion or an
otherwise expected signal cancellation reaches a failing processing-end
observer, that observer failure is the lifecycle failure and returns status 2;
no replay completion report is written.

Cancellation is checked after source/model construction, before serial open,
after the serial `HELLO`/`READY`/`DISABLE` handshake, immediately before
`ENABLE`, and before/after every capture, inference, selection, control, and
dispatch boundary. `ArduinoSerialActuator` receives the runtime's predicate
without changing the base `Actuator` protocol. It observes cancellation before
and after its one bounded rate-limit sleep and immediately before an `ENABLE`,
`CENTER`, or `SET` transport write. Once a boundary observes cancellation, no
new `ENABLE` or `SET` is initiated. A write already handed to the serial/OS
layer may still complete; ordinary control flow then attempts bounded
`DISABLE` and resource cleanup.

Offline replay does not duplicate this loop. `run_vision()` has an optional
observer that receives frozen start, applied-frame, processing-end, and normal
completion snapshots. No snapshot exposes a source, detector, actuator, serial
transport, image, mutable detection container, or callback. The processing-end
snapshot is emitted immediately when the frame loop ends and before bounded
cleanup, so replay FPS, command rate, and acquisition timing exclude cleanup
duration. `replay.py` consumes those snapshots for finite-media metrics and
feeds the existing strict atomic report writer from `benchmark.py`. The replay CLI fixes actuation to
`SimulatedPTZActuator`, rejects live/serial options, and uses the regular
source/detector factories only when invoked by an operator with local media and
a model. Replay report serialization replaces local CLI/configuration paths
with basename-only markers while retaining generic URL/secret redaction. The
observer does not own vision, selection, controller, or actuator policy.
