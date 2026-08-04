# Marine PTZ Tracker

[![CI](https://github.com/22by7-raikar/marine-ptz-tracker/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/22by7-raikar/marine-ptz-tracker/actions/workflows/ci.yml)

Laptop-driven system for detecting and following a small marine target with a
USB camera, a two-servo pan/tilt assembly, and an Arduino Uno. The laptop owns
perception and control; the Uno is the bounded servo-command safety boundary.

Containerized CPU verification, headless GPU replay, the explicitly armed
hardware template, release manifests, deployment verification, incident
response, and exact rollback procedures are documented in
[container deployment and rollback](docs/deployment.md). Container use remains
optional; native single-mode execution is the rollback path.

The CPU test service is deliberately isolated from replay and hardware
interpolation, so it needs no `MARINE_PTZ_*` variables:

```bash
docker compose -f compose.yaml build test
docker compose -f compose.yaml run --rm test
```

GPU replay uses `-f compose.replay.yaml`; explicitly armed hardware uses
`-f compose.hardware.yaml --profile hardware`. Supply every required absolute
mount/device path and group ID before invoking those files; exact commands are
in the deployment guide.

```bash
docker compose -f compose.replay.yaml build replay
docker compose -f compose.replay.yaml run --rm replay
docker compose -f compose.hardware.yaml --profile hardware build hardware
docker compose -f compose.hardware.yaml --profile hardware run --rm hardware
```

The laptop owns capture, detection, target selection, control, telemetry, and serial communication. The Arduino only validates bounded pan/tilt commands and drives the servos. Hardware is intentionally not accessed at import time.

## Current validation status

The hardware-independent synthetic slice and the unified OpenCV + Ultralytics
runtime are implemented. The runtime reads a camera or finite media source,
detects configured marine classes, selects a target, applies the existing
bounded controller, and drives either the simulated actuator or the
explicitly armed Arduino serial actuator.

The bounded Arduino protocol, lazy host serial actuator, in-memory firmware
model, and Uno firmware compile successfully in local and remote CI. The Uno
firmware is uploaded and the loaded pan/tilt assembly has passed controlled
movement checks with the camera mounted. Physical actuation remains fail-closed:
it requires the hardware configuration, explicit `--arm-hardware`, and an
explicit serial path from the CLI or validated configuration. The runtime never
discovers a serial port.

Software-only evidence includes the hardware/GPU-independent test suite,
Python 3.10 and 3.11 CI, and an Arduino Uno compile-only CI job. The current
Uno build uses 8,120 bytes of flash (25%) and 715 bytes of static RAM (34%).
The complete physical system has tracked the V3 `marine_target`, including
smooth pan, tilt, target loss, and reacquisition within the verified 75–105
degree limits. Final artifact review and submission packaging remain operator
work; this is not a safety certification or an open-world robustness claim.

> **Hardware warning:** Do not connect or power servos until the staged
> [demo runbook](docs/demo_runbook.md) safety checks and calibration evidence
> have been completed. The watchdog is a communication fallback, not an
> emergency stop.

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

This is the safest complete demonstration: it uses deterministic synthetic
frames and simulated actuation, so it opens no camera, serial port, or GPU.
The complete staged procedure—including replay, compile-only firmware,
controlled physical operation, and artifact review—is in
[docs/demo_runbook.md](docs/demo_runbook.md).
Final V1/V3 metrics, verified observations, software-tested behavior, and
limitations are consolidated in [docs/final_results.md](docs/final_results.md).

Final simulated digital-zoom check (no servo movement):

```bash
conda run --no-capture-output -n marine_ptz \
  python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --model runs/detect/runs/tugboat/yolo11n-marine-target-v3/weights/best.pt \
  --device cuda:0 --target-class marine_target --confidence 0.60 \
  --actuator-backend simulated \
  --digital-zoom --max-digital-zoom 2.0 --display
```

`--digital-zoom` affects only the annotated display/video crop. YOLO and the
PTZ controller always consume original full-frame pixels and coordinates. It is
software magnification, not optical zoom; omitting the flag preserves the
unzoomed runtime.

The exact held-out V3 evaluation and explicitly armed recorded-demonstration
commands are in [final results](docs/final_results.md) and the
[demo runbook](docs/demo_runbook.md). Generated reports and videos are ignored,
but must still be reviewed before publication.
Use the [submission checklist](docs/submission_checklist.md) for the final
hardware, artifact, claim, and safe-shutdown review.

Final held-out positive results at confidence 0.60 are 36 images, precision
1.000000, recall 0.936044, mAP50 0.962213, and mAP50–95 0.802989. The run
reported 1.7 ms/image preprocessing, 7.4 ms/image GPU inference, and
1.0 ms/image postprocessing. Its test split contained zero reviewed negatives,
so held-out negative false-positive rate and maximum negative confidence are
**not measured**, not 0%. Live hand/forearm-only rejection is qualitative
evidence; the intermittent 0.50–0.77 wallet false positive remains a documented
out-of-distribution limitation.

The camera supports 1920×1080 MJPEG at 30 FPS. The rollback `single` loop
processed about 9.4 unique frames/s because it couples each frame to a
rate-limited actuator command. A pre-correction physical `concurrent` run
measured 29.53 unique inferences/s but only 10.17 displays/s; its control branch
ran before rendering and could enter the actuator's sleep-based limiter. The
concurrent scheduler now renders new results independently and issues commands
only at its monotonic control deadline. These rates are distinct from GPU
inference latency and from the fixed encoded-video FPS. `--output` recordings
use monotonic time and hold/duplicate annotated frames to preserve wall-clock
duration; encoded duplicates are not reported as uniquely processed frames.
Use `--output-fps N` to override the configured 30 FPS output rate.

The final headless CUDA container replay processed all 2,635 source frames at
about 30 unique inferences/displays per second while keeping control at 9.976
Hz. It recorded zero capture/render/recorder drops, one held encoded frame,
zero stale/watchdog/session faults, capture-to-command p95 44.96 ms, and
bounded shutdown in about 0.95 ms. The output contains 2,635 frames at 30 FPS
and 87.834 seconds. These are finite-media simulated-actuator measurements,
not physical-servo performance; full provenance and limitations are in
[final results](docs/final_results.md).

Run the hardware-independent checks with:

```bash
python -m compileall -q src tests tools
python -m pytest
python tools/smoke_check.py
```

All current tests are deterministic and hardware/GPU-free. The explicit
filtered form remains available as a safe subset if opt-in tests are added:

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

For full local validation, run compileall, the smoke check, the complete suite,
and dependency consistency checks:

```bash
python -m compileall -q src tests tools
python tools/smoke_check.py
python -m pytest
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

## Tugboat dataset and custom model

The pretrained YOLO11n COCO baseline produced zero detections of the physical
Green Toys tugboat when run with `--target-class boat`. This is an observed
baseline outcome, not a precision/recall/mAP result. The custom dataset has one
class named exactly `marine_target`; generated data, recordings, labels,
weights, manifests, predictions, and runs remain ignored by Git.

Record the first of ten independent sessions from the configured InnoMaker
camera without opening Arduino or serial hardware:

```bash
conda run --no-capture-output -n marine_ptz \
  python tools/record_dataset.py \
  --config configs/hardware.yaml \
  --output captures/tugboat/session-01.avi \
  --duration 60 --display
```

Preview extraction, then rerun without `--dry-run` to write frames and a
manifest:

```bash
conda run -n marine_ptz python tools/extract_frames.py \
  --source captures/tugboat/session-01.avi \
  --output-dir data/tugboat/staging/session-01/images \
  --session session-01 \
  --manifest data/tugboat/staging/session-01/session-01-manifest.json \
  --sample-fps 2 --minimum-frame-gap 5 --dry-run
```

An already-installed Label Studio Community instance can remain local and keep
its state under the ignored dataset directory:

```bash
label-studio start --data-dir "$PWD/data/tugboat/label-studio-state"
```

Create one rectangle label named `marine_target`, export YOLO detection labels
locally, and place matching `.txt` files under each session's `labels/`
directory. No hosted account, cloud connector, or API is part of the workflow.

For an already downloaded private Ultralytics Platform Detect NDJSON export,
create a separate staging tree containing only annotated rows. First add
`--dry-run`, then remove it after every basename and annotation validates:

```bash
conda run -n marine_ptz python tools/import_ultralytics_ndjson.py \
  --ndjson data/tugboat/platform-export/tugboat.ndjson \
  --source-staging-root data/tugboat/staging \
  --output-staging-root data/tugboat/platform-imported
```

The importer is local-only, requires exactly class `0: marine_target`, ignores
unannotated rows, and never modifies the export or original staging images.
It rejects an existing output root. Do not use the partially annotated original
staging root as input to the splitter.

For the cumulative v2 export, zero-box rows may be included only through the
explicit reviewed-negative flag. **Use this flag only when every unboxed
exported image has been manually reviewed and confirmed not to contain the
tugboat.** It creates zero-byte labels and never invents boxes:

```bash
conda run -n marine_ptz python tools/import_ultralytics_ndjson.py \
  --ndjson data/tugboat/platform_export/tugboat-marine-target_v2.ndjson \
  --source-staging-root data/tugboat/staging \
  --output-staging-root data/tugboat/platform-imported-v2 \
  --include-unannotated-as-negatives \
  --dry-run
```

Version 1 directories remain untouched. The exact v2 preparation, training
(`--name yolo11n-marine-target-v2`), evaluation, simulated-runtime, and
explicitly armed physical-runtime commands are in
[local dataset and training](docs/dataset_training.md).

Validate and inspect the whole-session 70/20/10 split, then remove `--dry-run`
to copy the prepared dataset:

```bash
conda run -n marine_ptz python tools/prepare_tugboat_dataset.py \
  --staging-root data/tugboat/platform-imported \
  --output-root data/tugboat \
  --data data/tugboat/data.yaml \
  --seed 20260802 --dry-run
```

Fine-tune from a verified local YOLO11n checkpoint:

```bash
conda run --no-capture-output -n marine_ptz \
  python tools/train_tugboat.py \
  --data data/tugboat/data.yaml \
  --model /path/to/local/yolo11n.pt \
  --epochs 50 --image-size 640 --device cuda:0 --batch auto \
  --project runs/tugboat --name yolo11n-marine-target
```

Evaluate the custom model on held-out test sessions and write predictions:

```bash
conda run --no-capture-output -n marine_ptz \
  python tools/evaluate_tugboat.py \
  --model runs/tugboat/yolo11n-marine-target/weights/best.pt \
  --data data/tugboat/data.yaml \
  --image-size 640 --device cuda:0 --batch 8 \
  --project runs/tugboat-evaluation --name heldout-test \
  --report artifacts/evaluation/tugboat-test.json
```

Use the resulting model with simulated actuation first:

```bash
conda run --no-capture-output -n marine_ptz \
  python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --model runs/tugboat/yolo11n-marine-target/weights/best.pt \
  --device cuda:0 \
  --target-class marine_target \
  --actuator-backend simulated --display
```

The complete ten-session capture plan, YOLO label contract, annotation review,
split policy, metrics, and interpretation boundaries are in
[local dataset and training](docs/dataset_training.md). Never split random
neighboring frames across train, validation, and test; recording session is the
split unit.

## End-to-end runtime

The default configuration and programmatic `VisionOptions` select simulated
actuation and the verified `single` rollback runtime. `concurrent` remains an
explicit opt-in while the corrected render/control schedule awaits controlled
physical revalidation. Use a local model path in network-free operation; a
missing named Ultralytics model may otherwise cause Ultralytics to attempt a
download.

Mock/non-hardware run:

```bash
conda run -n marine_ptz python -m marine_ptz.demo \
  --config configs/development.yaml --steps 20 --no-sleep
```

Camera and person-detection verification without physical movement:

```bash
conda run -n marine_ptz python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --model /path/to/local-yolo11n.pt \
  --device cuda:0 \
  --target-class person \
  --actuator-backend simulated \
  --display
```

The camera path is supplied by `configs/hardware.yaml`; this command opens it
continuously but uses only the simulated actuator. A local pretrained model is
required to avoid an automatic model download.

Canonical physical person-tracking smoke test:

```bash
conda run -n marine_ptz python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --model /path/to/local-yolo11n.pt \
  --device cuda:0 \
  --target-class person \
  --actuator-backend arduino_serial \
  --serial-port /dev/ttyACM0 \
  --arm-hardware \
  --display
```

Canonical eventual boat-tracking run:

```bash
conda run -n marine_ptz python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --model /path/to/local-yolo11n.pt \
  --device cuda:0 \
  --target-class boat \
  --actuator-backend arduino_serial \
  --serial-port /dev/ttyACM0 \
  --arm-hardware \
  --display
```

The hardware configuration uses the exact InnoMaker camera by-id path. At the
last host check, no `/dev/serial/by-id` symlink existed for the Uno, so the
verified `/dev/ttyACM0` path is configured and passed explicitly above. Recheck
that identity after any USB change; the runtime never auto-discovers a port.

The verified servo envelope is **pan 75–105 degrees and tilt 75–105 degrees**;
this stage must not be widened without new guided physical evidence. Before
opening serial, hardware-mode startup opens
and validates the camera, loads the model, runs the cold first inference on one
real frame, initializes display/output resources, and reports that initialization
is complete. It then completes serial `HELLO`/`READY`/`DISABLE`, reports the
`ENABLE` boundary, enables once, and promptly sends a bounded neutral/hold
`SET`. The prepared frame's target correction begins on the next configured
10 Hz command slot.

In `single` mode, every successfully processed frame produces a bounded command.
With no valid target, including startup and temporary loss, `hold` resends the
last safe position without creating movement and refreshes the firmware
watchdog. This verified rollback path deliberately does not keep servos enabled
through a stalled camera or inference call: a stall approaching the 1-second
watchdog deadline causes firmware to detach. Held-out positive evaluation measured
7.4 ms/image GPU inference, while the integrated physical loop operated around
9.4 processed frames/s; neither number is the camera's 30 FPS acquisition
capability. A `NOT_ENABLED` response is still a
hard protocol failure and now identifies watchdog expiry as a likely cause; the
runtime does not automatically re-enable.

`--runtime-mode concurrent` separates one capture worker, one inference worker,
main-context control/serial/GUI work, and an optional encoder worker. Both
capture and completed-result handoffs have capacity one: a newer value replaces
an unread older value instead of building latency. Commands remain scheduled at
10 Hz and use only a result younger than `runtime.result_freshness_s` (0.5 s by
default). A fresh zero-detection result invokes the normal lost-target policy;
a stale or failed pipeline stops `SET`, attempts `DISABLE`, and fails the run.
There is no independent keepalive that can disguise stalled perception.
The main context consumes and renders each available latest result before
checking the next control deadline. The deadline is reset from completion of
the acknowledged command, so a delayed slot is skipped instead of producing a
catch-up burst or a second rate-limit sleep. Concurrent terminal/overlay output
labels `unique_inference_fps`, `display_fps`, `control_hz`, and
`encoded_output_fps` explicitly; the live encoded label is the configured fixed
timeline rate, while final metrics report the observed encoded rate. Final
metrics separately count intentional
capture/result/recorder replacements, recorder submissions and consumption,
encoded duplicates, stale results, and session faults. The corrected scheduler
is hardware-free tested and requires a repeat controlled physical run; use
`--runtime-mode single` as the safe fallback until that revalidation is recorded.

The display shows detections, selected-target and frame centers, tracking state,
pan/tilt command, inference latency, and explicit concurrent rates. Single mode
retains its backward-compatible rolling `fps` label. Press `q`, Escape, or
Ctrl+C to stop: the runtime stops sending motion, attempts bounded `DISABLE`,
closes serial, and releases the camera/display resources.

`--source 0` is also camera index 0. Use `--source path:0` for a file literally
named `0`; other media paths can be passed directly. Target classes may be
repeated or comma-separated. Use `--output annotated.mp4` for wall-clock-correct
annotated video, optionally with `--output-fps N`, and `--max-frames N` for a
finite run. Press `q` to leave display mode.

Startup initializes capture, model, selection, control, first-frame inference,
and requested display/output resources before opening the serial transport.
`ArduinoSerialActuator.open()` completes correlated `HELLO`/`READY`/`DISABLE`
while firmware remains disabled; only then can the explicitly armed runtime
call `ENABLE` and immediately send its first neutral/hold `SET`. Shutdown first
attempts bounded `DISABLE`, closes the actuator transport, and then releases
video resources. Finite-media EOF and `--max-frames` are successful exits.
Detector, frame, camera, actuator, or protocol failures stop the loop, run
bounded cleanup, and produce a nonzero CLI status.

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
power during all bench work. The complete system has operated physically, but
software still cannot prove emergency-stop behavior, long-duration mechanical
or thermal margins, or guaranteed disable delivery after a fault.

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

To compare both runtime topologies on the same existing local video with a
simulated actuator, run:

```bash
python tools/benchmark_runtime.py \
  --config configs/development.yaml \
  --source /path/to/local-input.mp4 \
  --model /path/to/local-model.pt \
  --device cuda:0 --target-class marine_target \
  --max-frames 600 \
  --report artifacts/benchmarks/runtime-comparison.json
```

The command paces finite-media reads at the configured camera FPS, runs
`single` and `concurrent` sequentially, and reports unique inference FPS,
display/result-consumption FPS, capture-to-command age p50/p95/p99, intentional
latest-value drops, control Hz, recorder-submitted and recorder-consumed unique
frames, encoded frames, duplicate ratio, and shutdown duration. It opens no
live camera or serial device. With no `--output`, recorder and encoded counts
are zero by design. Treat the output as operator measurement, not a checked-in
performance claim.

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
warnings. Remote CI is passing; firmware upload and physical validation remain
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
python tools/serial_protocol_demo.py --port /dev/ttyACM0 --hold-seconds 0.5
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
`configs/hardware.yaml` records the verified InnoMaker camera path, the current
Uno `/dev/ttyACM0` fallback because no serial by-id symlink exists, mapping,
physical-limit, neutral, and watchdog settings, plus the frozen V3
`marine_target` model/class at confidence 0.60, boot grace, and total handshake
deadline. Configuration
loading and serial-actuator construction alone never start hardware. The
vision CLI opens serial only when the resolved backend is `arduino_serial`,
and calls `ENABLE` only after `--arm-hardware` and successful dependency and
handshake initialization. Both calibrated axes use `physical = -logical +
180`, preserving neutral 90 while correcting the installed servo directions.
Both logical and physical axes are bounded to the verified 75–105 degree
range. Use the
[guided calibration procedure](docs/calibration.md) before considering any
range or neutral-pitch change.

Both checked-in configurations explicitly select
`runtime.execution_mode: single`. The CLI may opt in with
`--runtime-mode concurrent`;
`--result-freshness` overrides the validated `runtime.result_freshness_s` only
for that invocation. Arduino operation rejects a freshness limit greater than
or equal to its configured watchdog timeout.

The explicitly armed, single-axis calibration command is:

```bash
conda run --no-capture-output -n marine_ptz python tools/calibrate_ptz.py \
  --config configs/hardware.yaml --port /dev/ttyACM0 \
  --axis pan --arm-hardware
```

Run it with `--axis tilt` only after completing and disabling the pan session.
The tool holds the unselected axis at neutral, steps one logical degree per key,
refreshes the watchdog without inventing motion, and cannot expand configured
limits.

See [architecture](docs/architecture.md), [wiring](docs/wiring.md), the
[test plan](docs/test_plan.md), [acceptance criteria](docs/acceptance_criteria.md),
and the [physical evidence checklist](docs/evidence_checklist.md) before
connecting equipment.
