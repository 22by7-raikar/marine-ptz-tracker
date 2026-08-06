# Marine PTZ Tracker

[![CI](https://github.com/22by7-raikar/marine-ptz-tracker/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/22by7-raikar/marine-ptz-tracker/actions/workflows/ci.yml)

A laptop-driven prototype that detects and follows a small marine target with a
USB camera and a two-servo pan/tilt mechanism. The laptop performs capture,
YOLO inference, target selection, proportional control, telemetry, and serial
communication. An Arduino Uno validates bounded commands and owns the final
servo-output safety boundary.

## Demo

https://github.com/user-attachments/assets/c84c46f1-63f1-40c4-8cee-3291b3216307

The demonstration shows the physical desk prototype, not a marine deployment or
a safety-certified product. Generated videos, private captures, datasets, and
model weights are deliberately excluded from this repository.

## System at a glance

Hardware:

- Ubuntu laptop with an NVIDIA RTX 3060 Laptop GPU;
- InnoMaker U20CAM 1080p UVC camera;
- Arduino Uno R3;
- two-SG90 pan/tilt assembly; and
- separate regulated 5 V / 3 A servo supply with common ground.

Software:

- Python 3.10+ with OpenCV, PyTorch, Ultralytics YOLO11n, and optional BoT-SORT;
- typed YAML configuration and hardware-neutral component interfaces;
- a deterministic synthetic camera and simulated actuator;
- an explicitly armed serial actuator and bounded Arduino protocol; and
- hardware-free replay, benchmark, acceptance-report, and protocol-parity tools.

```text
camera/video -> YOLO -> target selector -> proportional PTZ controller
                                                |
                         simulated actuator or explicitly armed serial actuator
                                                |
                                      Arduino -> pan/tilt servos
```

The vision pipeline and controller always use original full-frame coordinates.
Digital zoom changes only display and recorded output.

## Algorithm and safety choices

- A custom one-class YOLO11n detector recognizes `marine_target`; the pretrained
  COCO `boat` class did not detect the physical toy in the observed baseline.
- Optional BoT-SORT assigns track identities. A track must be observed twice
  before lock. Short unsupported intervals preserve the locked identity, while
  prediction-only state never supplies coordinates to control.
- The selector provides only a current detector-supported locked observation.
  Missing, malformed, occluded, or stale results cannot invent a target center.
- A proportional controller uses a pixel deadband, bounded gain, a maximum
  three-degree update, and inclusive logical limits. Both installed servo axes
  map `physical = -logical + 180` around neutral 90 degrees.
- The current verified logical and physical envelope is 75–105 degrees on both
  axes. The firmware watchdog detaches outputs after one second without an
  accepted command.
- Physical startup validates camera/model/display resources and completes a
  serial handshake before `ENABLE`. Hardware requires `--arm-hardware` and an
  explicit serial identity; serial devices are never auto-selected.

These controls reduce prototype risk but are not an emergency stop. Keep a
direct way to remove external servo power during bench work and follow the
[wiring guide](docs/wiring.md), [calibration procedure](docs/calibration.md),
and [demo runbook](docs/demo_runbook.md) before physical operation.

## Repository layout

- `src/marine_ptz/` — domain types, interfaces, vision, selection, control,
  replay, telemetry, and simulated/serial actuator implementations.
- `firmware/ptz_controller/` — Arduino Uno state machine and bounded protocol.
- `configs/` — safe development defaults and reviewed hardware configuration.
- `tests/` — deterministic hardware/GPU-independent unit and integration tests.
- `tools/` — smoke, benchmark, replay support, dataset, and firmware helpers.
- `docs/` — architecture, deployment, evidence, safety, and operator procedures.

## Installation

The base project remains lightweight:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --editable ".[dev]"
```

For the verified CUDA environment, install PyTorch separately from the official
CUDA 12.8 index, then constrain the project vision installation so it cannot
replace that pair:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0

python -m pip install \
  --constraint constraints/vision-cu128.txt \
  --editable ".[vision]"
```

The validated host versions were `torch 2.11.0+cu128`, `torchvision
0.26.0+cu128`, `ultralytics 8.4.104`, and `opencv-python 5.0.0.93` with NVIDIA
driver 570.211.01. Review Ultralytics licensing before redistribution or
commercial deployment.

## Safe demonstrations

The deterministic synthetic loop opens no camera, serial device, model, or GPU:

```bash
python -m marine_ptz.demo \
  --config configs/development.yaml --steps 20 --no-sleep
```

Offline replay uses the production vision/selection/control path with simulated
actuation. Supply your own reviewed local media and weights; this repository
does not provide them and does not fabricate download links:

```bash
python -m marine_ptz.replay_cli \
  --config configs/development.yaml \
  --source /path/to/local-input.mp4 \
  --model /path/to/local-best.pt \
  --device cuda:0 \
  --target-class marine_target \
  --tracker botsort \
  --tracker-config configs/trackers/general_botsort.yaml \
  --report artifacts/replay/report.json
```

Reports use strict finite JSON, sanitized paths/arguments, allowlisted metadata,
and atomic replacement. Schema v3 adds passive current-observation and control
metrics; explicit target-motion windows are required for settling analysis. See
[replay methodology](docs/replay.md).

## Physical run boundary

First validate the model and camera with `--actuator-backend simulated`. Only
after wiring, firmware, neutral position, directions, limits, and device
identity have been checked should an operator replace both placeholders below
with exact discovered paths:

```bash
python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --source /dev/v4l/by-id/<verified-camera> \
  --model /path/to/local-best.pt \
  --device cuda:0 \
  --target-class marine_target \
  --tracker botsort \
  --tracker-config configs/trackers/general_botsort.yaml \
  --actuator-backend arduino_serial \
  --serial-port /dev/serial/by-id/<verified-uno> \
  --arm-hardware \
  --runtime-mode concurrent \
  --display
```

If the Uno exposes no stable by-id link, verify the current `/dev/ttyACM*`
identity immediately before use and pass that exact value. Do not guess or add
automatic device discovery. Press `q`, Escape, or Ctrl+C for bounded shutdown;
cleanup attempts `DISABLE` before closing serial and video resources.

## Validation

```bash
python -m compileall -q src tests tools
python tools/smoke_check.py
python -m pytest
python -m pip check
ruff check .
tools/compile_arduino.sh
```

Tests use fakes and in-memory transports; they require no camera, GPU, Arduino,
or serial port. CI covers Python 3.10/3.11 and compiles the Uno firmware with
Arduino CLI 1.5.1, Arduino AVR core 1.8.8, and Servo 1.3.0. The reviewed Uno
build uses 8,120 bytes of flash (25%) and 715 bytes of static RAM (34%).

## Evidence and limitations

One recorded physical concurrent run measured approximately 29.5 unique
inferences/s, 40.6 ms p95 result age at control, and 57.6 ms p95
capture-to-command latency. It recorded zero stale-result and watchdog faults.
Of 1,278 observations, 1,241 contained a relevant detection (97.10%) and 1,222
provided a selected/controller-supported target (95.62%). These are observations
from that run, not generalized accuracy, reliability, or safety guarantees.

No physical settling-time claim is made: the reviewed event was already within
tolerance at the end of its declared movement window. Selector-state durations
from the new replay report are also excluded pending correction of their
time-base discrepancy. Count-based ratios remain suitable evidence.

Known limitations include a small, single-object dataset; sensitivity to
lighting, occlusion, reflections, and look-alike objects; hobby-servo backlash
and no position feedback; fixed-focus/fixed-field-of-view imaging; USB identity
changes; and unmeasured long-duration thermal/mechanical behavior. More varied
real marine media and controlled physical trials are required before any
open-world or unattended-use claim.

This prototype should be used only under direct supervision in a controlled
area. It is not navigation, collision-avoidance, life-safety, surveillance, or
autonomous-vessel equipment.

## Further documentation

- [Architecture](docs/architecture.md)
- [Final evidence and results](docs/final_results.md)
- [Test plan](docs/test_plan.md)
- [Failure modes](docs/failure_modes.md)
- [Deployment and rollback](docs/deployment.md)
- [Dataset and training workflow](docs/dataset_training.md)
- [Firmware protocol](firmware/ptz_controller/README.md)
- [Submission summary](docs/submission-summary.md)

No license is included pending repository ownership/publication clarification.
