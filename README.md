# Marine PTZ Tracker

[![CI](https://github.com/22by7-raikar/marine-ptz-tracker/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/22by7-raikar/marine-ptz-tracker/actions/workflows/ci.yml)

Foundation for a laptop-driven system that detects and follows a small marine target with a USB camera, a two-servo pan/tilt assembly, and an Arduino Uno.

The laptop owns capture, detection, target selection, control, telemetry, and serial communication. The Arduino only validates bounded pan/tilt commands and drives the servos. Hardware is intentionally not accessed at import time.

## Status

The hardware-independent synthetic slice and the unified OpenCV + Ultralytics
runtime are implemented. The runtime reads a camera or finite media source,
detects configured marine classes, selects a target, applies the existing
bounded controller, and drives either the simulated actuator or the
explicitly armed Arduino serial actuator.

The bounded Arduino protocol, lazy host serial actuator, in-memory firmware
model, and Uno firmware compile successfully. They have not been uploaded or
bench-verified. Physical actuation is fail-closed: it requires the hardware
configuration, explicit `--arm-hardware`, and an explicit serial path from the
CLI or validated configuration. The runtime never discovers a serial port.

## Layout

- `src/marine_ptz`: typed configuration, domain values, interfaces, synthetic and OpenCV sources, YOLO detection, tracking policy, simulated/serial actuation, and the unified CLI composition.
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

## End-to-end runtime

The default configuration and programmatic `VisionOptions` select simulated
actuation. Use a local model path in network-free operation; a missing named
Ultralytics model may otherwise cause Ultralytics to attempt a download.

Offline video with simulated actuation:

```bash
conda run -n marine_ptz python -m marine_ptz.vision_cli \
  --config configs/development.yaml \
  --source path:/path/to/input.mp4 \
  --model /path/to/local-yolo11n.pt \
  --device cpu \
  --target-class boat \
  --max-frames 300
```

Live camera with simulated actuation:

```bash
conda run -n marine_ptz python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --source camera:0 \
  --model /path/to/local-yolo11n.pt \
  --device cuda:0 \
  --target-class boat \
  --confidence 0.60 \
  --iou 0.45 \
  --image-size 640 \
  --actuator-backend simulated \
  --display
```

Live camera with armed Arduino actuation is intentionally verbose:

```bash
conda run -n marine_ptz python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --source camera:0 \
  --model /path/to/local-yolo11n.pt \
  --device cuda:0 \
  --target-class boat \
  --actuator-backend arduino_serial \
  --serial-port /dev/ttyACM0 \
  --arm-hardware \
  --display
```

Do not run the armed form until camera identity, serial-device identity,
logical/physical limits, servo directions, neutral angles, external servo
power, and common ground have been bench-verified. Omitting
`--actuator-backend` uses the validated configuration: development selects
simulation, while the hardware configuration selects `arduino_serial` and
therefore still refuses to start without `--arm-hardware`.

`--source 0` is also camera index 0. Use `--source path:0` for a file literally
named `0`; other media paths can be passed directly. Target classes may be
repeated or comma-separated. Use `--output annotated.mp4` for annotated video
and `--max-frames N` for a finite run. Press `q` to leave display mode.

Startup initializes capture, model, selection, and control before opening the
serial transport. `ArduinoSerialActuator.open()` completes correlated
`HELLO`/`READY`/`DISABLE` while firmware remains disabled; only then can the
explicitly armed runtime call `ENABLE`. Shutdown first attempts bounded
`DISABLE`, closes the actuator transport, and then releases video resources.
Finite-media EOF and `--max-frames` are successful exits. Detector, frame,
camera, actuator, or protocol failures stop the loop, run bounded cleanup, and
produce a nonzero CLI status.

### Process exit status

| Runtime outcome | CLI status |
| --- | ---: |
| Finite-media EOF, `q`, `--max-frames`, or another normal programmatic completion | 0 |
| SIGINT-requested shutdown | 130 |
| SIGTERM-requested shutdown | 143 |
| Unexpected `OperationCancelled` without a recorded runtime termination request | 2 |
| Detector, source, protocol, actuator, configuration, observer, or cleanup failure | 2 |

Signal-requested shutdown is operationally expected: the runtime records the
request, completes bounded cleanup, and stops issuing new motion commands.
The nonzero 130/143 statuses follow normal Unix signal conventions rather than
indicating that cleanup was skipped or that an operator cancellation was a
runtime defect. `run_vision()` returns a processed-frame count for library
callers; `main()` alone maps runtime outcomes to process statuses.

For offline replay, a control-path failure remains authoritative if the
processing-end observer or cleanup also fails; those later failures remain
inspectable diagnostics and no completed report is written. A processing-end
observer failure after otherwise normal EOF/frame-limit completion is itself a
failure. The same rule applies during an expected SIGINT/SIGTERM cancellation:
the observer failure takes precedence and the replay CLI returns 2 after its
single bounded cleanup pass.

`yolo11n.pt` remains the lightweight configured default. Model weights,
recordings, and annotated outputs are ignored by Git.

## Safety limitations

The arming flag, bounded host exchanges, firmware limits, and watchdog reduce
risk; they do not make this an emergency-stop or safety-rated motion system.
The runtime checks its shared termination request after source and model
startup, before/after serial handshake, immediately before `ENABLE`, and at
each tracking dispatch boundary. The serial actuator checks the same request
before and after its bounded command-rate sleep and immediately before a
motion write. Once cancellation is observed at one of those boundaries, no new
`ENABLE` or `SET` is initiated; cleanup still attempts bounded `DISABLE`.
An operation already handed to the serial/OS layer may still complete.
USB disconnects, process termination, or serial faults can leave the last
confirmed physical position uncertain until the firmware watchdog detaches
the servos. Cleanup attempts `DISABLE`, but software cannot guarantee delivery
after a transport or power failure. Keep a direct way to remove external servo
power during all bench work. Camera, serial, mechanics, power, thermal
behavior, watchdog timing, and full-system tracking remain pending physical
validation.

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

## Offline replay acceptance evidence

Offline replay runs one existing local image or finite video through the same
production OpenCV source, YOLO detector, marine selector, PTZ controller, and
`SimulatedPTZActuator` used by the unified runtime. It never opens a live
camera or serial port, and it does not arm an Arduino. It is evidence of
software behavior on finite media, not evidence of toy-boat accuracy, camera
performance, or physical servo tracking.

Use a local model path to avoid a possible named-model download:

```bash
python -m marine_ptz.replay_cli \
  --config configs/development.yaml \
  --source /path/to/local-input.mp4 \
  --model /path/to/local-yolo11n.pt \
  --device cpu \
  --target-class boat \
  --max-frames 300 \
  --report artifacts/replay/acceptance.json
```

`--report` is required. The report is strict finite JSON and atomically
replaces its destination only after a complete successful replay. Optional
acceptance limits (`--min-selected-target-ratio`, `--max-p95-inference-ms`,
`--max-mean-normalized-error`, `--max-p95-normalized-error`,
`--max-pan-saturation-count`, and `--max-tilt-saturation-count`) are all
reported with their actual value and pass/fail result. A failed supplied limit
writes the report and exits with status 3; normal EOF or `--max-frames` exits
0; invalid input, decode/inference failure, report-write failure, or unexpected
runtime failure exits 2; SIGINT/SIGTERM use 130/143. A hardware configuration
requires explicit `--replay-safe` before it can be used for simulated-only
replay. See [offline replay methodology](docs/replay.md).

## Arduino serial foundation

The compile-only firmware baseline is configured as a separate GitHub Actions
job. It compiles only—never uploads or enumerates serial devices—using:

- Arduino CLI `1.5.1`;
- Arduino AVR Boards `arduino:avr@1.8.8`;
- official Servo library `Servo@1.3.0`; and
- Arduino Uno FQBN `arduino:avr:uno`.

With those prerequisites installed in the Arduino CLI user data directory, the
same reproducible local and CI command is:

```bash
tools/compile_arduino.sh
```

Pass an optional temporary build directory when it must be retained for local
inspection: `tools/compile_arduino.sh "$(mktemp -d)"`. The helper checks the
pinned CLI, core, and library but does not install anything; CI performs the
pinned installation first and writes builds only under the runner temporary
directory. Set `ARDUINO_CLI` or `ARDUINO_CLI_CONFIG_DIR` when either location
is outside your shell's defaults.

The pre-remediation compile used 7,856 bytes of flash (24%) and 705 bytes of
static RAM (34%). After adding the rejected-command retry cache, the current
compile uses 8,120 bytes of flash (25%) and 715 bytes of static RAM (34%).
Warnings emitted with `--warnings all` originate only from the official
`arduino:avr@1.8.8` core's `new.cpp`; project firmware compiles without
warnings. Remote CI verification, upload, and physical validation remain
pending.

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
servo current, and device identity have been verified. The protocol tool
remains useful for isolated bench diagnosis; the end-to-end runtime composes
the same serial actuator only behind its additional hardware-arming gates.

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
settings, plus the boot grace and total handshake deadline. Configuration
loading and serial-actuator construction alone never start hardware. The
vision CLI opens serial only when the resolved backend is `arduino_serial`,
and calls `ENABLE` only after `--arm-hardware` and successful dependency and
handshake initialization.

See [architecture](docs/architecture.md), [wiring](docs/wiring.md), and the [test plan](docs/test_plan.md) before connecting equipment.
