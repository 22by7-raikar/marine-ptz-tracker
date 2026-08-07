# Marine PTZ Tracker — Technical Submission Report

**A safety-bounded laptop/Arduino prototype for tracking a small marine target**

This Markdown file is the editable source for the five-page submission brief.
It describes the public implementation at the documented release commit. It
does not include private media, datasets, model weights, generated reports, or
the unmerged latency-predictor experiment.

## Evidence language

Statements use one of five evidence categories:

| Category | Interpretation |
| --- | --- |
| **Unit/integration tested** | Deterministic behavior exercised without physical devices, including fakes, in-memory firmware, and the production C++ protocol harness. |
| **Measured recorded-run evidence** | Numeric result from one controlled recorded run or identified finite replay. It is not generalized accuracy or reliability. |
| **Qualitative physical observation** | Behavior observed on the supervised desk rig without independent metrology or repeated statistical trials. |
| **Design property** | Property established by code, configuration, or architecture; not a measured outcome. |
| **Proposed future improvement** | Work not included in the public system. |

## 1. System overview

### Problem and requirements

The challenge was to build a practical pan/tilt tracker around inexpensive,
available hardware. A USB camera observes one Green Toys tugboat. An Ubuntu
laptop must detect the target, retain a useful target identity, calculate
bounded camera corrections, and communicate with an Arduino Uno that drives two
SG90-class servos. The prototype must run locally, stay testable without
hardware, avoid stale perception backlogs, and stop motion safely when the
perception or serial path becomes untrustworthy.

### Implemented prototype

The implemented path is:

```text
UVC camera or finite media
  -> OpenCV capture
  -> custom one-class YOLO11n detector
  -> optional persistent Ultralytics BoT-SORT
  -> confirmation/current-observation target selector
  -> bounded proportional pan/tilt controller
  -> simulated actuator OR explicitly armed serial actuator
  -> Arduino sequence/limit/watchdog boundary
  -> pan and tilt servos
```

Python performs capture, CUDA inference, target selection, control, telemetry,
display/recording, and serial communication. The Uno parses a narrow bounded
protocol, enforces its own angle range, attaches/detaches outputs, and times out
after one second without an accepted command. Digital zoom is an optional
software crop for display and recording only; the detector and controller use
original full-frame coordinates.

### Role and responsibilities

I designed and implemented the repository architecture, typed configuration,
synthetic vertical slice, custom-data workflow, YOLO/BoT-SORT integration,
selector state machine, proportional controller, single and concurrent runtime
composition, serial protocol and actuator, Arduino firmware, replay/evaluation
and benchmark evidence, container and CI validation, and operator/safety
documentation. I also performed supervised hardware bring-up and reported the
limits of that evidence rather than treating bench observation as certification.

### Architecture and evidence snapshot

| Item | Result | Evidence class |
| --- | ---: | --- |
| Unique inference cadence | approximately 29.5 FPS | Measured recorded-run evidence |
| p95 result age at control | approximately 40.6 ms | Measured recorded-run evidence |
| p95 capture-to-command | approximately 57.6 ms | Measured recorded-run evidence |
| Relevant detection observations | 1,241 / 1,278 (97.10%) | Measured recorded-run evidence |
| Selected/controller-supported observations | 1,222 / 1,278 (95.62%) | Measured recorded-run evidence |
| Stale-result / watchdog faults | 0 / 0 | Measured recorded-run evidence |
| Track acquisition | two current qualifying hits by default | Design property; unit/integration tested |
| Servo envelope | logical/physical 75–105°, neutral 90° | Design property; limits tested and qualitatively calibrated |

These measurements come from one controlled physical concurrent run. Software
timing is not physical actuator response. The mechanism has no encoder, so no
measured physical angle or settling-time claim is made.

## 2. Hardware selection and connections

### Selected hardware and trade-offs

| Component | Role and selection rationale | Limitation |
| --- | --- | --- |
| InnoMaker U20CAM 2 MP low-light UVC camera | Standard UVC interface provides straightforward OpenCV capture; 1920×1080 MJPEG at 30 FPS was observed on the development host. | Fixed field of view, no motorized optical zoom, USB buffering/exposure effects. |
| Arduino Uno R3 | Small, inspectable actuator boundary with mature Servo support and simple USB serial. | No operating-system device-identity guarantee; limited RAM/flash; USB reset behavior. |
| Two SG90-class servos and low-cost bracket | Commodity, low-cost mechanism supports rapid prototype iteration. | Backlash, flex, deadband, variable response, no position feedback. |
| Regulated 5 V / 3 A supply | Separates servo-current demand from USB and the Uno regulator. | Capacity and wiring must be verified under actual load; not a certified power system. |
| NVIDIA development laptop | Flexible Python development and low-latency CUDA inference without an embedded deployment constraint. | Higher power/size than an edge deployment and not weatherized. |
| Green Toys tugboat | Repeatable one-class target for the challenge demonstration. | Does not represent broad marine-object diversity. |

### Wiring boundary

| Connection | Required wiring | Safety reason |
| --- | --- | --- |
| Pan signal | Arduino digital pin 9 to pan-servo signal | Firmware owns bounded pan output. |
| Tilt signal | Arduino digital pin 10 to tilt-servo signal | Firmware owns bounded tilt output. |
| Servo positive | External regulated 5 V supply | Do not draw two-servo current from USB or Uno 5 V. |
| Servo ground | External supply negative | Provides servo current return. |
| Common reference | Arduino ground to external supply ground | Required reference for pan/tilt PWM signals. |
| Camera | Separate laptop USB connection | Camera power/data remains independent of servo power. |
| Arduino | Laptop USB connection | Provides Uno power and bounded serial data only. |

The operator must check voltage and polarity with a meter, remove external servo
power before rewiring, mechanically center horns, keep a direct power-removal
path, and stop on buzzing, collision, flex, binding, or cable tension.

### What the hardware does not establish

The frozen 75–105° envelope and inverse mappings were observed on the desk rig,
but commanded servo angle is not measured camera angle. There is no encoder,
torque/current measurement, independent target-position reference, rigid
metrology fixture, weatherproof enclosure, ingress protection, safety rating,
or long-duration thermal/mechanical qualification.

## 3. Software and algorithm trade-offs

### Detection and persistent identity

The pretrained COCO `boat` class produced zero detections of the physical toy in
the observed baseline. A local one-class dataset and custom YOLO11n model were
therefore used for class `marine_target`. YOLO11n was appropriate for rapid
fine-tuning and the available laptop GPU.

Detector-only mode calls `model.predict()`. Optional BoT-SORT mode calls
`model.track(..., persist=True, tracker=<yaml>)` with a 0.20 tracker input
threshold. The detector instance owns the model and tracker state. In concurrent
mode, only the inference worker constructs, warms, synchronizes, and invokes
YOLO/CUDA/BoT-SORT.

The selector is deliberately stricter than the tracker. A new ID must exceed
the acquisition threshold and appear on two distinct current frames by default.
Once locked, only a valid current detector-supported box with the locked ID can
drive zoom or control. During a short unsupported interval, the identity is
protected against immediate takeover, but no cached or prediction-only
coordinates are used. After 0.30 seconds by default, the lock expires and a new
ID must confirm normally. Stale concurrent results fault before target/control
advancement.

### Decision table

| Decision | Alternative | Why selected | Remaining limitation |
| --- | --- | --- | --- |
| Custom one-class YOLO11n | Generic pretrained COCO detector | The toy was not detected in the observed generic baseline; custom training targets the actual object. | Small, narrow dataset does not prove open-world generalization. |
| Optional BoT-SORT | Independent frame-by-frame ranking | Persistent IDs support confirmation and protected lock while retaining detector-only rollback. | ID switches, occlusion loss, detector dependence, tuning sensitivity, no guaranteed ReID. |
| Current-observation-only actuation | Kalman/prediction-only motion | Prevents invented coordinates from driving servos when evidence is absent. | Brief occlusion uses lost-target hold rather than anticipatory motion. |
| Bounded proportional control | PI/PID or model-based control | Simple, explainable, stable under limited calibration; no integral windup or model assumption. | No encoder feedback; residual error and physical dynamics are not measured. |
| Latest-value capacity-one channels | Lossless FIFO processing | Bounds latency and prevents stale work from accumulating. | Obsolete frames/results are intentionally dropped. |
| Python/OpenCV/PyTorch threads | Immediate C++/TensorRT rewrite | Native OpenCV/CUDA work overlaps while interfaces stay replaceable and testable. | Deployment efficiency may improve if future profiling identifies a bottleneck. |

### Control and safety boundary

The proportional controller applies a 24 px per-axis deadband, gain 8.0,
three-degree maximum step, 10 Hz update schedule, and inclusive 75–105° logical
limits. Both installed axes use `physical = -logical + 180`, preserving neutral
90° and the same physical bounds. Lost target behavior is `hold`: the last safe
position is resent at scheduled updates without creating movement.

Physical movement is impossible through the CLI unless the operator selects
`arduino_serial`, supplies an explicit serial path, and adds `--arm-hardware`.
Camera/model initialization and first inference occur before the serial
handshake; `ENABLE` is the final startup action and an immediate bounded hold
follows. The Uno enforces compile-time limits and detaches after a one-second
watchdog timeout. These are design/tested safety properties, not certification.

## 4. Experiments: useful, successful, and rejected

### Evidence-producing work

| Experiment | Result | Evidence class |
| --- | --- | --- |
| UVC capture validation | 1920×1080 MJPEG at 30 FPS observed on the configured camera/host. | Qualitative physical observation / camera capability observation |
| External servo power and common ground | Controlled pan/tilt motion observed without powering servos from the Uno. | Qualitative physical observation |
| Protocol and watchdog | Codec, sequence, duplicate, state, limit, and watchdog behavior exercised in Python and production C++; firmware compiled for Uno. | Unit/integration tested |
| Generic COCO `boat` baseline | Zero detections of the physical toy in the observed baseline. | Qualitative physical observation |
| Custom marine-target model | One-class training/evaluation workflow and positive held-out metrics recorded; weights remain external. | Measured model-run evidence |
| BoT-SORT integration | Persistent calls, IDs, two-hit confirmation, short unsupported lock, expiry, and current-only gating exercised. | Unit/integration tested |
| Concurrent latest-value runtime | One capture owner, one inference owner, one serial owner, freshness rejection, capacity-one replacement, and bounded shutdown exercised. | Unit/integration tested; one recorded physical run |
| Replay/evaluation | Strict atomic reports, path/credential redaction, finite metrics, and simulated-actuator replay implemented. | Unit/integration tested; measured finite replay evidence where identified |
| Controlled tracking | Pan, tilt, target loss/reacquisition, and digital display zoom observed on the desk rig. | Qualitative physical observation |

### Rejected α-β latency compensation

An optional current-observation-gated α-β predictor was implemented locally as
an experiment. It estimated observed target-centroid velocity and projected the
control point over a 40 ms horizon. It never permitted prediction-only actuation
during loss or occlusion. It also could not reduce the actual capture,
inference, serial, or actuator latency; it only changed the point presented to
the existing controller.

Physical A/B observation did not show better tracking. No numeric error
improvement or regression is claimed because the trial did not establish one.
The experiment was not pushed, merged, or retained in the public implementation.

> Physical A/B testing did not demonstrate a tracking benefit sufficient to
> justify the additional estimator state and tuning complexity, so the simpler
> proportional-control baseline was retained.

Rejecting an unsupported feature is an engineering outcome: it preserved the
smaller state space and easier safety argument until repeatable evidence can
justify additional estimator behavior.

### Recorded-run boundary

The reported 29.5 FPS inference, 40.6 ms p95 result age, 57.6 ms p95
capture-to-command, 97.10% relevant-detection ratio, 95.62%
selected/controller-supported ratio, and zero stale/watchdog faults all belong
to one controlled recorded run. They do not measure open-water accuracy,
physical position, servo response, or settling. Selector-state seconds are not
quoted because the report has a known time-base discrepancy for those
durations. A reviewed movement window was already within tolerance at its end,
so no 0 ms settling claim is made.

## 5. Limitations and high-impact improvements

### Current limitations

- One toy, limited environments, and a small one-class dataset constrain
  detector evidence.
- BoT-SORT does not guarantee identity: switches, missed association, short
  occlusions, similar targets, and configuration sensitivity remain.
- SG90-class servos and the low-cost mechanism provide no position feedback and
  exhibit backlash, flex, deadband, vibration, and load-dependent response.
- Digital zoom is a display crop, not optical zoom; it adds no captured detail.
- USB device identity, camera buffering/exposure, serial reset timing, cable
  loading, and power integrity can vary between setups.
- No independent physical-angle/target-position ground truth, long soak,
  disconnect trial, outdoor/glare campaign, weatherproofing, or safety
  certification exists.

### Prioritized improvements

1. **Encoder-equipped actuators.** Actual pan/tilt feedback would permit
   defensible physical-angle, settling, overshoot, and disturbance measurements
   and make PI/PID or model-based control meaningful.
2. **Rigid, low-backlash mechanism.** A stiffer mount and better drivetrain
   would improve repeatability and reduce flex, vibration, and camera-orientation
   uncertainty.
3. **Repeatable hardware-in-the-loop fixture.** A controlled target trajectory,
   target-position ground truth, actuator-angle measurement, synchronized logs,
   and repeated A/B trials would turn qualitative tracking into defensible
   control evidence.
4. **Larger, more varied marine dataset.** Add independent sessions containing
   water glare, reflections, partial occlusion, distance/scale change, multiple
   targets, similar objects, and outdoor lighting, with reviewed negatives in
   the held-out split.
5. **Camera improvements only if field evidence justifies them.** Optical zoom,
   global shutter, improved optics, manual exposure, or lower-buffer capture
   should follow measured field limitations rather than assumptions.
6. **Deployment compute optimization only after profiling.** First reduce
   camera/buffer latency; then consider Jetson-class deployment, TensorRT FP16
   if inference is the bottleneck, and C++ only where profiles show Python
   overhead materially constrains the system.

Better mechanics, feedback, and repeatable testing would change what the system
can actually measure and control. They are likely higher-value next steps than
adding unvalidated estimator or controller complexity to the current hobby-servo
prototype.

## Responsible-use conclusion

The repository demonstrates a well-bounded prototype architecture, not a
production marine system. Operate under direct supervision with a physical
power-removal path. Do not use it for navigation, collision avoidance, life
safety, autonomous vessel control, identification of people, or unattended
monitoring. Review all reports and media for privacy before publication.
