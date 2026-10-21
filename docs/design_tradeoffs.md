# Design trade-offs

Strengths and weaknesses of each major algorithm and design choice, what each
costs, and when to change it. Numbers come from `configs/hardware.yaml` and
the code unless marked as an estimate.

## Summary

| Choice | Main strength | Main weakness |
| --- | --- | --- |
| Custom one-class YOLO11n | Detects the real target at ~7 ms/frame | Narrow dataset; false-positive rate unmeasured |
| BoT-SORT identity (ReID off) | Stable IDs with no second model | ID switches; motion model disturbed by the moving camera |
| Confirm-then-lock selector | Rejects one-frame false positives; no lock stealing | Adds acquisition delay; holds still during occlusion |
| Incremental proportional control | Simple, bounded, no windup, zero error on a still target | Lags moving targets; 30°/s slew cap; no angle feedback |
| Laptop vision + Uno output boundary | Fast iteration; independent hardware limits | Two devices, USB serial reset and latency |
| ASCII, sequenced, acknowledged protocol | Debuggable, idempotent retries, parity-tested | One round trip per command; no checksum |
| Detach-on-timeout watchdog | Fails safe when the host disappears | Servos go limp; firmware hang not covered |
| Capacity-one latest-value threads | Bounded latency, no backlog | Frames dropped by design; ownership enforced by convention |
| Single composition root + replay observer | Replay exercises the production loop | `vision_cli.py` is a 2,500-line module |

## Perception

### Custom one-class YOLO11n

The pretrained COCO `boat` class never detected the toy, so the detector is a
fine-tuned YOLO11n with the single class `marine_target`.

**Pros**
- Detects the actual target: V3 held-out precision 1.000, recall 0.936,
  mAP50–95 0.803.
- Small and fast: 7.4 ms GPU inference per image on an RTX 3060 Laptop GPU,
  leaving headroom at 30 FPS capture.
- Ultralytics provides training, evaluation, and tracking behind one API, which
  kept the wrappers thin.

**Cons**
- The dataset is one toy in 19 indoor sessions (359 images). Generalization to
  real vessels, water glare, or outdoor light is unknown.
- The held-out split has no negative images, so the false-positive rate is
  unmeasured. Live use showed a wallet in poor light detected at 0.50–0.77.
- A new target type needs a new dataset and retraining.
- Ultralytics is AGPL-3.0; redistribution or commercial use needs a license
  review.

**Change when** targets diversify: add held-out reviewed negatives first, then
scale data across sessions before considering a larger model.

### BoT-SORT identity

`model.track(..., persist=True)` with sparse-optical-flow camera-motion
compensation and ReID disabled.

**Pros**
- Persistent track IDs make confirmation and lock protection possible.
- No second (ReID) model: no extra GPU memory or latency.
- Optional: `tracker_backend: none` falls back to per-frame detection.

**Cons**
- IDs can switch under occlusion or between similar objects; there is no
  long-term re-identification.
- The camera is the thing being moved. Pan/tilt motion shifts the whole scene,
  so the tracker's Kalman motion model relies on optical-flow compensation;
  fast slews degrade association.
- Tracker state lives inside the Ultralytics model object, which couples the
  tracker to the pinned Ultralytics version and to one owning thread.
- Two thresholds (0.20 support, 0.60 acquisition) and `track_buffer` interact;
  tuning is empirical.

**Change when** ID switches dominate failures: feed the commanded pan/tilt
deltas into motion compensation, or enable ReID if appearance is distinctive.

### Target selector

A new ID must reach 0.60 confidence on two distinct frames to lock. While
locked, only a current detection with the locked ID drives control. If the ID
disappears, the lock is protected for 0.30 s, but no coordinates are produced.

**Pros**
- A single-frame false positive cannot move the camera.
- Another object cannot steal the lock during a brief occlusion.
- Control never acts on cached or predicted boxes, so the camera does not
  chase a target that is not actually visible.
- Deterministic ranking (confidence, area, center distance, coordinates, ID)
  makes behavior reproducible in tests and replay.

**Cons**
- Acquisition costs at least one extra frame, and the first correction waits
  for the next 10 Hz control slot.
- During occlusion the camera holds still while the target keeps moving, which
  can push it out of frame before it reappears.
- After a genuine tracker ID switch, the new ID must wait for the 0.30 s
  protection to expire and then reconfirm: a visible stall.
- One target only; multiple boats are resolved by ranking, not by intent.

## Control

### Incremental proportional controller

Each 10 Hz update adds a bounded correction to the current angle:

```text
e      = (target_px - center_px) / (frame_extent / 2)    # normalized, -1..1
step   = 0                         if |error_px| <= 24 px
       = clamp(8.0 * e, -3°, +3°)  otherwise
angle  = clamp(angle + step, 75°, 105°)
```

Because the command accumulates, this behaves as an integrator on image error.

**Pros**
- Zero steady-state error (within the 24 px deadband) on a stationary target.
- Every update is bounded to 3°, which limits mechanical shock and current
  spikes on hobby servos.
- No integral windup, no derivative noise, nothing to tune beyond gain,
  deadband, and step.
- The deadband stops hunting caused by servo backlash and detector jitter.

**Cons**
- A moving target is followed with a lag that grows with its angular speed.
- Hard slew cap of 3° × 10 Hz = **30°/s**; anything faster is lost.
- The step saturates once `|e| ≥ 3/8`, so large errors close at constant speed
  rather than proportionally.
- Effective loop gain depends on the camera's field of view, which is not in
  the configuration. Each update corrects about `8 / (FOV/2)` of the angular
  error. *Estimate:* with roughly 60° horizontal and 35° vertical FOV, that is
  ~27% per update on pan and ~46% on tilt, so tilt responds noticeably faster.
- Open loop on the mechanism: there is no encoder, so backlash and load sag go
  uncorrected except through the image.
- Inference runs at ~30 Hz, but control uses one result in three.

**Rejected alternatives.** An alpha-beta forward projection (40 ms horizon)
showed no measurable benefit on the rig; see [decisions](decisions.md).
PI/PID or model-based control would need measured actuator angle to be
meaningful.

**Change when** the mechanism gets position feedback: a feedforward term from
estimated target angular velocity removes the moving-target lag, and a faster
control rate becomes useful once serial round trips are no longer the limit.

### Digital zoom

Display and recording crop only.

**Pros:** cannot affect detection or control; smooth framing for operators.
**Cons:** adds no optical detail; small distant targets still detect poorly.

## System design

### Laptop vision, Uno output boundary

**Pros**
- Python and CUDA on the laptop make iteration fast.
- The firmware is small enough to review completely and is tested bit-for-bit
  on the host against golden vectors.
- Angle limits are enforced in firmware, independent of any host bug or
  configuration.

**Cons**
- Two devices and a USB link. Opening the port resets the Uno, which costs
  about 2 s of handshake at startup.
- A laptop is not a deployable outdoor controller.
- Serial round-trip time adds to every command.

### Serial protocol

ASCII lines, 16-bit sequence numbers, one acknowledgement per command,
idempotent duplicates.

**Pros**
- Readable in a plain serial terminal, which makes bring-up and fault
  diagnosis easy.
- Exact retransmission is safe: a duplicate `SET` never moves twice.
- One grammar, tested in both Python and C++ against shared vectors.
- Strict canonical parsing rejects ambiguous input instead of guessing.

**Cons**
- Stop-and-wait: the control thread blocks on each `ACK`, so serial latency is
  in series with control.
- No checksum or CRC; integrity relies on USB CDC framing.
- If a command is lost entirely, host and firmware sequence numbers diverge,
  and recovery needs a full re-handshake.

### Watchdog and fail-safe behavior

**Pros**
- Host crash, USB loss, or a stalled pipeline detaches the servos within 1 s.
- Only accepted `SET` commands feed the watchdog, so a host that is alive but
  not tracking cannot keep the outputs alive with `STATUS` or `CENTER`.
- Startup cannot move hardware until camera, model, and handshake have all
  succeeded and `--arm-hardware` is given.

**Cons**
- The safe state is limp: a camera-loaded tilt axis can sag onto its stop.
- A firmware hang is not covered, because servo pulses come from a timer
  interrupt (see [hardware gaps](hardware.md#known-hardware-gaps)).
- 1 s is long relative to the 30°/s slew rate.
- It is not an emergency stop; external power removal is still required.

### Absolute commands

**Pros:** retries are idempotent and firmware stays nearly stateless.
**Cons:** the host must track position; after a lost acknowledgement the
position is uncertain until a `STATUS` or re-handshake.

### Concurrency model

Capture, inference, and recording run in threads joined by capacity-one
latest-value channels. Only the main thread selects targets, controls, and
talks to the actuator.

**Pros**
- OpenCV decoding and CUDA inference release the GIL, so threads overlap them
  without multiprocessing or serialization.
- Latency is bounded: an unread frame is replaced instead of queued.
- A stale result is treated as a fault, not as "no target", so a stalled
  pipeline stops the servos rather than holding them.
- `single` mode remains as a simpler rollback.

**Cons**
- Frames are dropped under load by design; offline metrics must account for
  it.
- Single ownership of the model, actuator, and source is enforced by
  convention and tests, not by the type system.
- Two runtimes double the surface to test and maintain.
- The decoupled render/control schedule has not been re-run on the rig.

### Code organization

**Pros**
- `Protocol` interfaces with fakes give about 800 hardware-free tests that run
  in seconds.
- Configuration is parsed once into frozen, validated dataclasses, so invalid
  configuration fails at startup.
- Replay observes the production loop instead of reimplementing it.

**Cons**
- `vision_cli.py` (about 2,500 lines) mixes argument parsing, composition, both
  runtimes, and cleanup; it is the hardest file to change safely.
- The test suite is about as large as the source, and some tests are coupled to
  implementation details.
- `serial.watchdog_timeout_s` is not negotiated with the firmware, so the two
  values can drift silently.

## Highest-value improvements

1. **Encoder-equipped, stiffer pan/tilt mechanism.** It enables measured
   settling and overshoot, velocity feedforward, and meaningful PI/PID.
2. **Hardware-in-the-loop fixture.** A repeatable target trajectory with ground
   truth turns qualitative tracking into measured control performance.
3. **Broader dataset with held-out negatives.** Add glare, reflections,
   distance changes, and similar objects.
4. **Split `vision_cli.py`** into CLI parsing, composition, and one module per
   runtime.
5. **Firmware hardening:** add the AVR hardware watchdog and report the compiled
   watchdog timeout in `READY`.
6. **Profile before optimizing compute.** Reduce camera buffering first; move
   to TensorRT or an embedded GPU only if inference becomes the bottleneck.
