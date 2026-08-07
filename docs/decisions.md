# Architecture decisions

## 2026-07-23: Keep hardware behind protocols

Camera, detector, target-selection, control, and actuator dependencies are represented by Python `Protocol` interfaces. This supports deterministic fakes and prevents test or import paths from requiring physical equipment.

## 2026-07-23: Use absolute bounded PTZ commands

`PTZCommand` carries pan and tilt degrees rather than servo pulses or relative motion. The controller can be tested independently, while the actuator and Arduino translate and enforce calibrated physical limits.

## 2026-07-23: Keep the Arduino narrow

The laptop performs vision and policy. The Arduino firmware validates its serial protocol, rejects unsafe angles, and drives servo signals. This keeps the time-sensitive actuator boundary simple and inspectable.

## 2026-07-23: Defer model and transport choices (superseded)

Superseded for the prototype vision milestone on 2026-07-24. At the time of
this decision, no detection model, camera backend library, serial package, or
firmware framework had been chosen. Vision choices were superseded on
2026-07-24, and serial transport/protocol choices were superseded on 2026-07-25.

## 2026-07-24: Build the first slice with deterministic synthetic components

The initial runnable loop uses a metadata-only camera, repeatable moving bounding boxes, proportional image-error control, and an in-memory actuator. Detection generation, target selection, control policy, and actuation remain separate so each can be replaced and tested independently.

## 2026-07-24: Parse configuration with safe YAML and typed validation

PyYAML is the sole runtime dependency. Configuration is converted immediately into frozen dataclasses, rejects unknown keys and non-finite numeric values, enforces backend-required device settings, and reports field-qualified validation errors. This prevents untyped configuration mappings from spreading into tracking code.

## 2026-07-24: Keep real vision optional and simulated at the actuator boundary (superseded)

OpenCV capture and Ultralytics YOLO are optional implementations loaded lazily.
The base and synthetic packages do not import NumPy, OpenCV, Torch, or
Ultralytics. The real vision CLI may access a camera and GPU but always sends
control output to `SimulatedPTZActuator`; serial and Arduino actuation remain
out of scope. The simulated-only composition choice was superseded on
2026-07-26; optional dependency isolation remains current.

The default model is lightweight `yolo11n.pt`. Classes are resolved from model
metadata rather than fixed COCO IDs. The verified CUDA installation remains a
separate first stage, and the project vision extra is installed under the
cu128 constraints. Ultralytics licensing must be reviewed before distribution
or commercial deployment.

The selected prototype choices are OpenCV capture, Ultralytics inference, and
`yolo11n.pt` as the configurable default model; callers may supply another
model name or local path. This milestone deliberately retains simulated
actuation even when the camera and detector are real.

## 2026-07-25: Use a bounded ASCII serial protocol and detached watchdog state

The Uno protocol uses printable newline-delimited ASCII with fixed token
counts, canonical signed-32-bit integers, 16-bit nonzero command sequence IDs,
and a 63-byte payload bound for LF or CRLF. Startup uses sequence-zero
announcements and sequence-correlated `HELLO` responses. JSON and Arduino
`String` are excluded to keep RAM use and parsing behavior predictable.
Same-sequence retries are idempotent; only a duplicate of an acknowledged
`SET` refreshes the watchdog. Host and firmware both enforce limits without
silently clamping malformed wire values.

Pyserial is isolated in an optional hardware extra and imported only by an
explicit transport open. The actuator retains logical controller coordinates
while direction/offset mapping produces bounded physical degrees. The firmware
starts detached, attaches at neutral only after `ENABLE`, and detaches on
`DISABLE` or watchdog expiry. This is a prototype communication safety policy,
not an emergency-stop certification. Existing synthetic and vision composition
continued to use simulated actuation until the explicitly armed runtime
composition on 2026-07-26.

Opening an Uno serial device may cause its normal auto-reset. The host uses a
bounded boot grace, total handshake deadline, repeated identical `HELLO`, and a
correlated `DISABLE`; it does not suppress or manually pulse DTR/RTS. Firmware
services its watchdog before and after bounded serial work and emits pending
CRLF responses incrementally according to available TX capacity.

## 2026-07-26: Compile the production Uno sketch in isolated CI

The production sketch is compiled in a dedicated, read-only GitHub Actions job
using Arduino CLI `1.5.1`, `arduino:avr@1.8.8`, and `Servo@1.3.0` for
`arduino:avr:uno` with all warnings. A checked-in helper shares that exact
compile contract with local development while refusing to install, upload, or
enumerate hardware. This adds reproducible toolchain coverage without implying
an upload or physical validation result.

## 2026-07-26: Use one fail-closed real-vision composition root

`marine_ptz.vision_cli` remains the sole OpenCV/Ultralytics tracking loop and
now selects either simulated or Arduino serial actuation through one factory.
The existing selector, controller, actuator, source, and detector
implementations are reused; no parallel hardware pipeline is introduced.

Simulation remains the development default. Physical motion requires a
validated configuration whose actuator backend is `arduino_serial`, explicit
`--arm-hardware`, and a serial device supplied by CLI or validated
configuration. Serial ports are never discovered automatically. Source, model,
selector, and controller initialization precede the serial
`HELLO`/`READY`/`DISABLE` handshake, and `ENABLE` is the final startup action.

The loop enforces the configured tracking update rate and uses one idempotent
cleanup coordinator. It requests bounded disablement before closing the
actuator and video resources. Signal handlers only set a local termination
request. The runtime and production serial actuator share that request at
motion safety boundaries, including the serial rate-limit wait, so an observed
cancellation starts no new `ENABLE` or `SET`; an already submitted write may
still complete. Primary runtime and cleanup failures are both reported. These
interlocks reduce accidental motion; they do not create a safety-rated
emergency-stop system or replace pending physical validation.

## 2026-07-26: Measure offline replay through a runtime observer

Offline acceptance reporting observes successful iterations of the existing
unified runtime rather than creating another detector/controller loop. The
replay path fixes actuation to the existing simulated actuator and reuses the
benchmark module's sanitized strict JSON and atomic writer. This keeps report
metrics (target acquisition/loss, normalized error, command rate, and limit
saturation) separate from the generic runtime while ensuring they describe the
same selection and control behavior that production uses. No default acceptance
limits are claimed; future toy-boat and physical evidence must set and record
their own reviewed thresholds.

## 2026-08-02: Split the one-class tugboat dataset by recording session

The physical Green Toys tugboat was not detected by the pretrained YOLO11n
COCO `boat` baseline. Custom training therefore uses exactly one class,
`marine_target`. Capture and annotation stay local, and generated media,
labels, weights, predictions, and runs remain outside Git.

Frames are grouped by recording session before deterministic 70/20/10
allocation. Neighboring frames from one clip can never cross train,
validation, and test boundaries. The preparation stage validates normalized
class-0 boxes and copies rather than mutates staging data. OpenCV and
Ultralytics remain lazy operator-only dependencies; the core validation/split
policy is hardware-free and testable without CUDA or model weights.

## 2026-08-06: Add optional BoT-SORT identity without permitting prediction-only control

Frame-by-frame detection remains available as `tracker_backend: none`, but it
cannot protect a selected identity when multiple detections compete or frame
confidence varies. The optional BoT-SORT path therefore uses Ultralytics
`model.track(..., persist=True)` inside the existing detector. The detector
instance is the sole owner of the model and persistent tracker state; in
concurrent mode that ownership stays entirely within the inference worker.

Track IDs are evidence for a separate `MarineTargetSelector`, not authority to
actuate. New acquisition requires two current high-confidence observations by
default. A locked ID may remain protected for a short unsupported interval, but
the selector returns no cached or prediction-only coordinates during that
interval. A different ID cannot take over until the old lock expires, after
which ordinary confirmation is required again. This accepts limited identity
continuity while retaining the existing lost-target, freshness, controller,
limit, and actuator safety boundaries.

BoT-SORT remains optional because association can switch IDs, lose targets
under occlusion, and depends on detector quality and tuning. The checked-in
configuration keeps ReID disabled, avoiding a second model and any claim of
long-term re-identification.

## 2026-08-06: Rejected experiment: observed-only alpha-beta forward projection

This was not a controller-replacement experiment. The proportional controller
remained unchanged. An optional predictor was implemented and tested locally:
it estimated target-centroid position and velocity, then projected the current
observed point with a 40 ms horizon before passing it to the existing controller.
The baseline horizon was 0 ms. Projection required a current, fresh,
detector-supported locked observation, so prediction alone could not cause
motion during occlusion or target loss. It did not reduce the underlying
capture, inference, serial, or actuator latency.

The supervised physical A/B comparison did not demonstrate meaningful improved
tracking. The experiment was not pushed, merged, or adopted; the public runtime
continues to use the simpler bounded proportional controller. This decision
does not claim that prediction is universally ineffective—only that this
prototype produced insufficient evidence to justify it.

> Physical A/B testing did not demonstrate a tracking benefit sufficient to justify the additional estimator state and tuning complexity, so the simpler proportional-control baseline was retained.
