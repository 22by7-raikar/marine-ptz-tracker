# Marine PTZ Tracker

[![CI](https://github.com/22by7-raikar/marine-ptz-tracker/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/22by7-raikar/marine-ptz-tracker/actions/workflows/ci.yml)

Foundation for a laptop-driven system that detects and follows a small marine target with a USB camera, a two-servo pan/tilt assembly, and an Arduino Uno.

The laptop owns capture, detection, target selection, control, telemetry, and serial communication. The Arduino only validates bounded pan/tilt commands and drives the servos. Hardware is intentionally not accessed at import time.

## Status

The hardware-independent synthetic slice and a real OpenCV + Ultralytics vision
pipeline are implemented. The vision pipeline reads a camera or media file,
detects configured marine classes, selects a target, and drives only the
    simulated PTZ actuator. A bounded Arduino protocol, lazy host serial actuator,
    in-memory firmware model, and Uno firmware foundation are implemented and
    compile successfully for an Arduino Uno. They have not been uploaded or
    bench-verified. The synthetic and vision
tracking demos never select physical actuation; only the serial protocol tool
can do so when an operator explicitly supplies `--port`.

## Layout

- `src/marine_ptz`: typed configuration, domain values, interfaces, synthetic and OpenCV sources, YOLO detection, tracking policy, simulated actuation, and CLI composition.
- `configs`: development defaults and hardware deployment values.
- `firmware/ptz_controller`: Arduino firmware contract and implementation area.
- `docs`: architecture, wiring, bill of materials, tests, and decisions.
- `tests`: hardware-free unit and smoke tests.
- `tools`: local verification helpers.

## Synthetic demo

Python 3.10+ is required. From an environment with the project dependencies installed, run:

```bash
PYTHONPATH=src python -m marine_ptz.demo --config configs/development.yaml --steps 20
```

Use `--no-sleep` for a deterministic run without real-time pacing:

```bash
PYTHONPATH=src python -m marine_ptz.demo --config configs/development.yaml --steps 20 --no-sleep
```

Run the hardware-independent checks with:

```bash
python -m compileall -q src tests tools
python -m pytest -m "not hardware and not gpu"
python tools/smoke_check.py
```

The CI-equivalent suite deliberately excludes opt-in hardware and GPU tests:

```bash
python -m pytest -m "not hardware and not gpu"
```

Pytest uses strict marker validation: only registered markers are accepted, so
a typo fails collection instead of silently changing test selection. CI
installs the base and development extras through `constraints/ci.txt`:

```bash
python -m pip install \
  --constraint constraints/ci.txt \
  --editable ".[dev]"
```

This is a reviewed lightweight CI constraint set, not a hash-locked
supply-chain lockfile, and it does not promise identical resolution across
operating systems or future package-index changes. The workflow pins the
official checkout and Python setup actions to verified immutable v6 revisions.

Run explicitly connected hardware or CUDA tests only when the required device
is available:

```bash
python -m pytest -m hardware
python -m pytest -m gpu
```

For full local validation, run compileall, the smoke check, the CI-equivalent
suite, and dependency consistency checks:

```bash
python -m compileall -q src tests tools
python tools/smoke_check.py
python -m pytest -m "not hardware and not gpu"
python -m pip check
```

## Verified vision environment

The validated host stack uses an NVIDIA 570.211.01 driver and an RTX 3060
Laptop GPU:

- `torch==2.11.0+cu128`
- `torchvision==0.26.0+cu128`
- `ultralytics==8.4.104`
- `opencv-python==5.0.0.93`

Install this stack in two stages. First, install the CUDA-specific PyTorch pair
from the official cu128 index:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0
```

Then install the vision packages with the verified PyTorch pair constrained:

```bash
python -m pip install \
  --constraint constraints/vision-cu128.txt \
  --editable ".[vision]"
```

Run the second command only after the first succeeds. The constraint prevents
dependency resolution from replacing the cu128 pair with a newer build such as
`torch 2.13+cu130`. Do not treat missing CUDA access inside a coding-agent
sandbox as evidence that the validated host installation must be changed.

Ultralytics is offered under AGPL-3.0 and Enterprise licensing. Review the
applicable Ultralytics license before distributing or commercially deploying a
system that uses it.

## Real vision pipeline

The exact validated host command for camera index 0 with CUDA inference is:

```bash
conda run -n marine_ptz python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --source camera:0 \
  --model yolo11n.pt \
  --device cuda:0 \
  --target-class boat \
  --confidence 0.60 \
  --iou 0.45 \
  --image-size 640 \
  --display
```

`--source 0` is also camera index 0. Use `--source path:0` for a file literally
named `0`; other media paths can be passed directly. Target classes may be
repeated or comma-separated. Use `--output annotated.mp4` for annotated video
and `--max-frames N` for a finite run. Press `q` to leave display mode.

`yolo11n.pt` is the lightweight default. Ultralytics may download named
pretrained weights when they are not already cached; pass a local model path
when network-free startup is required. Model weights and generated recordings
are ignored by Git.

## Benchmarking

The camera-free benchmark measures model loading and inference against either a
local image or a deterministic blank frame. It does not command a servo or open
a camera. Run it with a local model path when network-free startup is required:

```bash
python tools/benchmark_yolo.py \
  --model /path/to/yolo11n.pt \
  --device cuda:0 \
  --image-size 640 \
  --warmup 5 \
  --iterations 30
```

JSON reports default to ignored `artifacts/benchmarks/`. Reports use strict,
finite JSON, retain only allowlisted runtime metadata and a sanitized
invocation, and replace existing reports atomically after a complete
same-directory write is flushed to disk. See
[benchmark methodology and preliminary results](docs/benchmarks.md).

## Arduino serial foundation

The compile-only firmware baseline is verified with:

- Arduino CLI `1.5.1`;
- Arduino AVR Boards `arduino:avr@1.8.8`;
- official Servo library `Servo@1.3.0`; and
- Arduino Uno FQBN `arduino:avr:uno`.

With those prerequisites installed in the Arduino CLI user data directory, the
reproducible compile command is:

```bash
ARDUINO_BUILD_DIR="$(mktemp -d)"
arduino-cli compile \
  --fqbn arduino:avr:uno \
  --warnings all \
  --build-path "${ARDUINO_BUILD_DIR}" \
  firmware/ptz_controller
```

The pre-remediation compile used 7,856 bytes of flash (24%) and 705 bytes of
static RAM (34%). After adding the rejected-command retry cache, the current
compile uses 8,120 bytes of flash (25%) and 715 bytes of static RAM (34%).
Warnings emitted with `--warnings all` originate only from the official
`arduino:avr@1.8.8` core's `new.cpp`; project firmware compiles without
warnings. Upload and physical validation remain pending.

The default protocol demonstration is hardware-free:

```bash
python tools/serial_protocol_demo.py
```

It uses the in-memory transport and firmware state model. It does not import
pyserial or open a device. Explicit serial probing requires the optional
hardware extra and an explicit port:

```bash
python -m pip install --editable ".[hardware]"
python tools/serial_protocol_demo.py --port /dev/ttyACM0
```

Do not run the physical form until wiring, limits, directions, neutral angles,
servo current, and device identity have been verified. The serial actuator is
not yet composed into `marine_ptz.demo` or `marine_ptz.vision_cli`.

Opening an Uno serial port may reset the board. The host deliberately allows a
bounded 2-second boot grace and requires a sequence-correlated `READY` plus
`DISABLE` before reporting connected. Startup announces
`READY 0 1 0.1.0`; `HELLO N` must receive `READY N 1 0.1.0`.

The protocol uses 115200-baud newline-delimited ASCII, a 63-byte payload limit
for either LF or CRLF, canonical signed-32-bit integer tokens, correlated
1–65535 sequences, bounded idempotent retries, strict range checks, and an
Arduino watchdog that detaches outputs after communication loss. While outputs
remain enabled, an identical retry of an acknowledged `SET` refreshes the
watchdog; after expiry it returns `NOT_ENABLED`. Rejected or conflicting
duplicates never refresh the watchdog. See the
[firmware protocol](firmware/ptz_controller/README.md).

Host writes use one elapsed PySerial write deadline and do not call a blocking
output drain. Successful local write completion means the complete frame was
accepted by the OS/driver; only its correlated firmware response confirms the
command end to end. A semantically impossible ACK or STATUS faults the local
session, blocks motion, and requires an explicit successful `open()` handshake
before actuation can resume.

## Configuration

`configs/development.yaml` selects the synthetic camera and simulated actuator.
`configs/hardware.yaml` records the OpenCV/Ultralytics vision defaults and
candidate Arduino serial, mapping, physical-limit, neutral, and watchdog
settings, plus the boot grace and total handshake deadline. The vision CLI
always uses simulated actuation and does not open a serial device.
Configuration loading and serial-actuator construction alone never start
hardware.

See [architecture](docs/architecture.md), [wiring](docs/wiring.md), and the [test plan](docs/test_plan.md) before connecting equipment.
