# Marine PTZ Tracker — Submission Summary

**A safety-bounded, laptop-driven prototype for tracking a small marine target**

## Problem and requirements

The challenge was to build a practical pan/tilt tracker around inexpensive,
available hardware: an InnoMaker USB camera, Arduino Uno, two SG90-class servos,
an externally powered pan/tilt bracket, and an NVIDIA-capable Ubuntu laptop. The
system needed to detect a Green Toys tugboat, keep it near the image center, run
without cloud services, and fail safely when perception or serial communication
became unreliable.

## Implementation and role

I designed and implemented the complete prototype workflow: typed configuration,
camera and detector boundaries, synthetic tests, custom-data preparation, YOLO11n
training/evaluation wrappers, BoT-SORT integration, target selection, proportional
control, serial protocol, Arduino firmware, replay/benchmark evidence, concurrent
runtime scheduling, container verification, and operator documentation. Hardware,
GPU, serial, and firmware concerns remain separated behind explicit interfaces so
the full software behavior can be tested with deterministic fakes.

## Design choices

- **Custom YOLO11n detector:** the pretrained COCO `boat` class produced no
  detections of the physical toy, so a one-class `marine_target` dataset and model
  were used. YOLO11n fits the prototype's GPU and latency budget.
- **BoT-SORT identity with current-observation control:** two observations are
  required before lock. A short occlusion can preserve identity, but Kalman-only
  predictions never provide coordinates to the controller.
- **Bounded proportional control:** normalized pixel error drives pan/tilt through
  a deadband, proportional gain, three-degree maximum step, and 75–105 degree
  logical/physical limits. This is understandable, tunable, and appropriate for
  low-cost hobby servos with no position feedback.
- **Safety split:** the laptop owns perception and control; the Uno validates
  sequence-correlated commands, enforces limits, and detaches outputs after a
  one-second watchdog timeout. Physical use requires an explicit device identity
  and `--arm-hardware`; startup finishes camera/model initialization before
  `ENABLE`, and shutdown attempts bounded `DISABLE`.
- **Replaceable runtime topology:** single mode is the rollback path. Opt-in
  concurrent mode uses capacity-one latest-value handoffs, one inference owner,
  one serial owner, stale-result rejection, and an independent 10 Hz control
  deadline without accumulating work.

## Defensible validation evidence

Deterministic tests cover configuration, selection, controller signs and clamps,
digital zoom, cancellation, serial faults, Python/C++ protocol parity, firmware
state/watchdog behavior, replay privacy, and bounded cleanup. CI validates Python
3.10/3.11 and compiles the Uno firmware without uploading it.

In one recorded physical concurrent run, the system measured approximately **29.5
unique inferences/s**, **40.6 ms p95 result age at control**, and **57.6 ms p95
capture-to-command latency**, with **zero stale-result or watchdog faults**. Of
**1,278 observations**, **1,241 contained a relevant detection (97.10%)** and
**1,222 supplied a selected/controller-supported target (95.62%)**. These are
run-specific engineering measurements, not generalized accuracy or safety claims.

## Limitations

The dataset represents one toy and limited environments. Lighting, reflections,
occlusion, similar objects, USB identity changes, hobby-servo backlash, cable
loading, and long-duration thermal behavior require further evaluation. The
system has no physical position feedback and is not safety-rated. No physical
settling-time claim is made: the reviewed event had already returned within
tolerance at the end of its declared window. Replay selector-state durations are
also omitted until their time-base discrepancy is corrected.

## Improvements with more time or resources

Priority improvements are a rigid camera/mechanism mount, more varied marine and
reflective-background recordings, additional reviewed negatives, and servos or an
actuator with position feedback. A global-shutter or optical-zoom camera would be
considered only if motion blur or field-of-view measurements justified it.
TensorRT FP16 would likewise be optional for a demonstrated constrained
deployment.

PD/PID control, TensorRT, a C++ rewrite, and custom CUDA were intentionally not
implemented. Current evidence did not show a control deficiency or compute
bottleneck that justified their complexity, integration risk, calibration burden,
or reduced portability. The chosen architecture delivers measurable closed-loop
behavior while keeping the $60–80 prototype understandable, testable, and safe to
bring up under direct supervision.
