# Marine PTZ Tracker

[![CI](https://github.com/22by7-raikar/marine-ptz-tracker/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/22by7-raikar/marine-ptz-tracker/actions/workflows/ci.yml)

A laptop-driven prototype that detects and follows a small marine target with a
USB camera and a two-servo pan/tilt mechanism. The laptop performs capture,
custom YOLO inference, BoT-SORT identity tracking, target selection,
proportional control, telemetry, and serial communication. An Arduino Uno
validates bounded commands and owns the final servo-output safety boundary.

This repository is an engineering prototype for a controlled tabletop
demonstration. It is not a marine navigation system, an unattended surveillance
system, or safety-certified equipment. The editable technical brief is
[docs/submission-report.md](docs/submission-report.md).

## Demonstration summary

https://github.com/user-attachments/assets/c84c46f1-63f1-40c4-8cee-3291b3216307

The demonstration shows the physical desk prototype tracking one Green Toys
tugboat. The complete camera-to-servo path was observed operating under direct
supervision. Generated videos, private captures, datasets, reports, and model
weights are deliberately excluded from Git.

Evidence in this repository is classified explicitly:

| Category | Meaning |
| --- | --- |
| Unit/integration tested | Deterministic software or firmware behavior exercised with fakes, an in-memory transport, or the C++ protocol harness. |
| Measured recorded-run evidence | Numeric measurements from one identified controlled recording or replay; not a general accuracy claim. |
| Qualitative physical observation | Supervised bench behavior that was observed but not established by a repeatable metrology setup. |
| Design property | A property established by implementation, configuration, or architecture—not a measured outcome. |
| Proposed future improvement | Work not present in the public implementation. |

## Hardware stack

- Ubuntu laptop with an NVIDIA RTX 3060 Laptop GPU for capture, CUDA inference,
  control, and serial communication;
- InnoMaker U20CAM 1080p UVC camera, configured for 1920×1080 MJPEG at 30 FPS;
- Arduino Uno R3 as the narrow actuator/protocol boundary;
- two SG90-class servos on a low-cost pan/tilt assembly;
- separate regulated 5 V / 3 A servo supply with common ground; and
- Green Toys tugboat as the one-class physical target.

The servos must not be powered from the Uno 5 V pin or laptop USB. See the
[wiring guide](docs/wiring.md) and [bill of materials](docs/bom.md).

## Software stack

- Python 3.10+ and a `src/` package layout;
- OpenCV capture and annotation;
- a custom one-class Ultralytics YOLO11n model for `marine_target`;
- optional Ultralytics BoT-SORT with persistent track IDs;
- deterministic target-selection and bounded proportional-control policy;
- single-threaded rollback and opt-in latest-value concurrent runtimes;
- simulated or explicitly armed Arduino serial actuation;
- strict, path-sanitized replay, evaluation, benchmark, and evidence reports;
- Python/C++ protocol parity tests and reproducible Arduino Uno compilation.

The base and synthetic package remain lightweight. NumPy, OpenCV, Torch, and
Ultralytics are imported only by optional vision implementations when used.

## System architecture

```text
 UVC camera / finite video
            |
       OpenCV source
            |
  custom YOLO11n + optional persistent BoT-SORT   [inference owner]
            |
 confirmation/current-observation target selector
            |
 bounded proportional PTZ controller             [main/control owner]
            |
 simulated actuator OR explicitly armed serial actuator
            |
  Arduino limits + sequence protocol + 1 s watchdog
            |
       pan / tilt servos
```

Camera, detector, target selector, controller, and actuator remain behind typed
interfaces. The laptop owns perception and control policy; the Uno owns final
range enforcement, servo attachment, and watchdog detachment. Digital zoom is
a downstream display/recording crop only: detection, target selection, and PTZ
control always use original full-frame coordinates.

Concurrent mode has one capture owner, one model/tracker inference owner, one
main control/serial/GUI owner, and an optional recorder owner. Capacity-one
channels replace unread frames and results with the newest value, preventing a
growing backlog of stale perception. Only the main context selects a target,
advances controller state, or accesses the actuator. See
[docs/architecture.md](docs/architecture.md) for lifecycle and failure details.

## Detection and tracking

The generic pretrained YOLO11n COCO `boat` class produced no detections of the
physical toy in the observed baseline. The implemented detector therefore uses
a custom one-class YOLO11n model whose class name is exactly `marine_target`.
This is a qualitative baseline observation, not a COCO precision/recall result.

Frame-by-frame detection can choose the best box in each frame but cannot
protect the identity of an already selected object when candidates change.
BoT-SORT was selected to add persistent per-frame track IDs using an integration
already supported by the pinned Ultralytics version. With `--tracker botsort`,
the detector calls `model.track(..., persist=True, tracker=<yaml>)`; detector-only
mode continues to call `model.predict(...)` unchanged.

The detector instance owns the Ultralytics model and persistent tracker state.
In concurrent mode it is constructed, warmed, and called only by the inference
worker, so CUDA and BoT-SORT state never have multiple runtime owners. The
checked-in tracker configuration uses motion/IoU association and sparse optical
flow with ReID disabled; no second ReID model is loaded.

Two confidence roles are intentional:

- the default tracker input/support threshold is `0.20`, allowing a previously
  observed ID to remain detector-supported at lower confidence; and
- the default acquisition threshold is the configured detector threshold,
  `0.60` in `configs/hardware.yaml`, preventing a low-confidence box from
  starting a new lock.

BoT-SORT improves identity continuity but does not guarantee it. ID switches,
short occlusions, similar-looking targets, detector errors, and configuration
sensitivity remain possible; long-term re-identification is not claimed.

## Target-selection state machine

The selector is a separate safety and policy layer; tracker output never
bypasses it.

```text
 none -- qualifying ID --> acquiring -- 2 current hits --> locked
  ^                            |                              |
  |      changed/missing ID ---+          unsupported current observation
  |                                                           |
  +----- unsupported age > 0.30 s <---- occluded <-------------+
                                      |
                              same current ID resumes lock
```

- **None:** no qualifying current detection exists.
- **Acquiring:** the same current tracked ID must meet the acquisition threshold
  on two distinct frames by default. A changed ID resets confirmation.
- **Locked:** only a current, valid detection with the locked ID supplies a
  target center to zoom or control.
- **Occluded/unsupported:** for up to `0.30 s` by default, the locked identity is
  protected against an immediate switch, but no cached or prediction-only box
  is supplied to the controller. Lost-target policy applies instead.
- **Expired:** after the unsupported interval, the lock is released and a new
  ID must pass normal confirmation.
- **Stale concurrent result:** a result older than the configured freshness
  threshold is rejected before selection/control and triggers the existing
  fault/disable path; it is not treated as current tracker support.

Malformed boxes, missing track IDs in BoT-SORT mode, and detections outside the
requested class are rejected. Stable ranking resolves remaining ties by
confidence, area, frame-center distance, coordinates, and track ID.

## Control and hardware-safety boundaries

The controller converts full-frame target-center error into absolute logical
pan/tilt commands. Current hardware settings are:

| Property | Value | Evidence class |
| --- | ---: | --- |
| Pixel deadband | 24 px per axis | Design property |
| Proportional gain | 8.0 | Design property |
| Maximum change | 3° per scheduled update | Design property; unit tested |
| Command schedule | 10 Hz | Design property; measured near 10 Hz in the recorded run |
| Logical limits | 75–105° per axis | Configured and unit/integration tested |
| Physical mapping | `physical = -logical + 180` on both axes | Configured; qualitatively calibrated |
| Physical limits / neutral | 75–105° / 90° | Host and firmware enforcement tested |
| Lost target | hold the last safe command | Design property; unit/integration tested |
| Firmware watchdog | detach after 1 s without an accepted command | Firmware-model/C++ tested; physical disconnect test remains pending |

For physical startup, camera/model validation, first inference, optional
display/output setup, and the serial `HELLO`/`READY`/detached `DISABLE`
handshake all precede `ENABLE`. The first bounded hold command follows enable.
Normal no-target/hold updates resend the same safe position at the scheduled
rate rather than inventing movement. Serial or perception failure prevents
subsequent motion and bounded cleanup attempts `DISABLE`.

Hardware motion requires all three: the `arduino_serial` backend, an explicit
serial path, and `--arm-hardware`. Serial devices are never auto-discovered.
These controls reduce prototype risk but are not an emergency stop; an operator
must retain a direct way to remove external servo power.

## Installation

The base development installation does not include the vision stack:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --editable ".[dev]"
```

For the verified CUDA environment, install the PyTorch pair separately from
the official CUDA 12.8 index, then constrain the project vision installation so
it cannot replace that pair:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0

python -m pip install \
  --constraint constraints/vision-cu128.txt \
  --editable ".[vision]"
```

The validated host versions were `torch 2.11.0+cu128`, `torchvision
0.26.0+cu128`, `ultralytics 8.4.104`, and `opencv-python 5.0.0.93`, with NVIDIA
driver 570.211.01. Review Ultralytics licensing before redistribution or
commercial deployment.

## Model-weight placement

Model weights are local generated artifacts and must remain outside Git. Put a
reviewed weight file in an ignored local path such as:

```text
runs/detect/runs/tugboat/<run-name>/weights/best.pt
```

Then pass that path with `--model`. Do not add the file to the repository or
publish a fabricated download URL. The hardware configuration contains a local
V4 development path as a deployment reference; operators must verify that the
file exists and contains the `marine_target` class before running.

## Camera and Arduino setup

1. Wire pan signal to Uno pin 9 and tilt signal to pin 10.
2. Power both servos from the external regulated 5 V supply and connect supply
   ground to Arduino ground.
3. Compile/upload the reviewed firmware following
   [firmware/ptz_controller/README.md](firmware/ptz_controller/README.md).
4. Discover and verify the camera and serial identities locally. Do not guess a
   `/dev/tty*` path or add automatic discovery.
5. Complete the [calibration procedure](docs/calibration.md) and keep the frozen
   75–105° envelope unless new physical evidence justifies a staged change.
6. Run camera/model validation with simulated actuation before arming hardware.

## Simulation and replay-safe usage

The deterministic smoke loop opens no camera, serial device, model, or GPU:

```bash
python -m marine_ptz.demo \
  --config configs/development.yaml --steps 20 --no-sleep
```

Offline replay uses the production source/detector/selector/controller path with
simulated actuation. Supply reviewed local media and weights:

```bash
python -m marine_ptz.replay_cli \
  --config configs/development.yaml \
  --source /path/to/local-input.mp4 \
  --model /path/to/local-best.pt \
  --device cuda:0 \
  --target-class marine_target \
  --tracker botsort \
  --tracker-config configs/trackers/general_botsort.yaml \
  --tracker-input-confidence 0.20 \
  --tracker-min-confirmation-hits 2 \
  --tracker-max-unsupported-age 0.30 \
  --report artifacts/replay/report.json
```

Replay reports use strict finite JSON, sanitized paths/arguments, allowlisted
metadata, and atomic replacement. The current schema reports tracker identity
and selector-state counts, but selector-state durations in seconds are excluded
from evidence pending correction of a known report time-base discrepancy. See
[docs/replay.md](docs/replay.md).

## Supervised physical-hardware usage

First use the same camera/model command with `--actuator-backend simulated`.
Only after wiring, firmware, neutral, directions, limits, model class, and
device identities are checked should an operator run:

```bash
python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --source /dev/v4l/by-id/<verified-camera> \
  --model /path/to/local-best.pt \
  --device cuda:0 \
  --target-class marine_target \
  --tracker botsort \
  --tracker-config configs/trackers/general_botsort.yaml \
  --tracker-input-confidence 0.20 \
  --tracker-min-confirmation-hits 2 \
  --tracker-max-unsupported-age 0.30 \
  --actuator-backend arduino_serial \
  --serial-port /dev/serial/by-id/<verified-uno> \
  --arm-hardware \
  --runtime-mode concurrent \
  --digital-zoom --max-digital-zoom 2.0 \
  --display
```

If this Uno exposes no stable by-id link, verify the current `/dev/ttyACM*`
identity immediately before use and pass that exact value. Press `q`, Escape,
or Ctrl+C for bounded shutdown; cleanup attempts `DISABLE` before closing
serial, camera, writer, and display resources. Single runtime mode remains the
rollback path (`--runtime-mode single`).

## Testing and validation

```bash
python -m compileall -q src tests tools
python tools/smoke_check.py
python -m pytest
python -m pip check
ruff check .
tools/compile_arduino.sh
```

Tests require no camera, GPU, Arduino, or serial port. Coverage includes tracker
configuration, `model.track(..., persist=True)`, GPU-tensor conversion, track-ID
validation, acquisition/occlusion/expiry transitions, protected lock, digital
zoom gating, single/concurrent ownership, stale-result rejection, cancellation,
serial faults, protocol parity, firmware watchdog behavior, replay privacy, and
atomic reports. CI exercises Python 3.10/3.11 and compiles the Uno sketch with
pinned Arduino CLI 1.5.1, Arduino AVR core 1.8.8, and Servo 1.3.0. The reviewed
Uno build uses 8,120 bytes flash (25%) and 715 bytes static RAM (34%).

## Recorded-run evidence

The following figures are measured recorded-run evidence from **one controlled
physical concurrent run**, not generalized accuracy, reliability, settling, or
field-performance claims:

| Measurement | Result |
| --- | ---: |
| Unique inference rate | approximately 29.5 FPS |
| p95 result age at control | approximately 40.6 ms |
| p95 capture-to-command latency | approximately 57.6 ms |
| Frames with a relevant detection | 1,241 / 1,278 (97.10%) |
| Selected/controller-supported observations | 1,222 / 1,278 (95.62%) |
| Stale-result faults | 0 |
| Firmware-watchdog faults | 0 |

Software latency is not physical actuator response. The rig has no encoder, so
commanded angle is not measured camera angle. No physical settling-time claim is
made: the reviewed event was already within tolerance at the end of its declared
movement window. The recorded-run figures do not establish open-water or
open-world performance.

## Known limitations

- The custom dataset covers one toy and limited controlled environments.
- BoT-SORT can switch IDs and depends on detector quality and tuning; it does
  not guarantee long-term re-identification.
- Occlusion, glare, shadows, low light, scale change, similar objects, and
  multiple targets need broader recorded evaluation.
- Hobby servos have backlash, flex, and no position/velocity feedback.
- Commanded angle, physical camera angle, and target settling are not measured
  by an independent reference.
- The fixed-focus camera and digital crop provide no optical zoom or added
  spatial detail.
- USB identities can change; the current Uno may lack a stable by-id link.
- Long-duration thermal, mechanical, disconnect, and outdoor/weather behavior
  remain unqualified.

## Evaluated alternatives

| Alternative | Decision |
| --- | --- |
| Generic COCO `boat` detector | Rejected after the physical toy produced zero observed detections; a custom one-class model was trained. |
| Frame-by-frame selection only | Retained as `tracker=none`, but optional BoT-SORT was added for protected persistent identity. |
| Prediction-only tracker motion | Rejected; only current detector-supported boxes may drive control. |
| PI/PID or model-based control | Deferred because the hobby mechanism has no measured position feedback and current evidence did not justify more controller state. |
| Queued/lossless vision processing | Rejected for real-time control; capacity-one latest-value channels bound stale work. |
| TensorRT, C++, or custom CUDA | Deferred until profiling shows the current Python/CUDA path is the limiting factor. |

### Rejected experiment: observed-only alpha-beta forward projection

This was not a P-to-PI, P-to-PID, or controller-replacement experiment. The
proportional controller remained unchanged. A local experimental branch
estimated target-centroid position and velocity, then projected the observed
target point forward before passing that point to the existing controller. The
tested 40 ms projection horizon was compared with the 0 ms baseline.

| Item                | Finding                                                                                |
| ------------------- | -------------------------------------------------------------------------------------- |
| Hypothesis          | Forward projection of the observed target center might reduce apparent trailing error. |
| Baseline            | Existing proportional controller with a 0 ms projection horizon.                       |
| Experimental change | Alpha-beta forward projection with a 40 ms horizon before the unchanged P controller.  |
| Safety boundary     | No prediction-only actuation during target loss or occlusion.                          |
| Pipeline latency    | Unchanged, as expected.                                                                |
| Physical tracking   | No meaningful improvement observed in the controlled A/B comparison.                   |
| Decision            | Retain the simpler proportional-control baseline.                                      |

Projection was eligible only for a current, fresh, detector-supported locked
observation. Prediction alone could not move the camera during occlusion or
target loss, and it could not make capture, inference, serial communication, or
the servos faster. The supervised physical A/B comparison did not justify the
extra estimator state and tuning complexity. The experimental branch was not
pushed, merged, or adopted into the public baseline.

> Physical A/B testing did not demonstrate a tracking benefit sufficient to justify the additional estimator state and tuning complexity, so the simpler proportional-control baseline was retained.

## Responsible use

Operate only under direct supervision in a controlled area with a physical
power-removal path. Do not use this prototype for navigation,
collision-avoidance, life safety, autonomous-vessel control, identification of
people, or unattended monitoring. Obtain permission before recording and review
all evidence for private paths, credentials, faces, or sensitive surroundings
before publication.

## Documentation

- [Technical submission report](docs/submission-report.md)
- [Architecture and lifecycle](docs/architecture.md)
- [Final evidence and results](docs/final_results.md)
- [Replay methodology](docs/replay.md)
- [Test plan](docs/test_plan.md)
- [Failure modes](docs/failure_modes.md)
- [Deployment and rollback](docs/deployment.md)
- [Dataset and training workflow](docs/dataset_training.md)
- [Firmware protocol](firmware/ptz_controller/README.md)
- [Submission summary](docs/submission-summary.md)

No license is included pending repository ownership/publication clarification.
