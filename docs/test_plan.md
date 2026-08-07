# Test plan

## Deployment checks

Hardware-free tests validate strict atomic release manifests, basename-only
evidence records, clean/dirty Git metadata, preflight mode separation, lack of
device discovery, optional-import isolation, Compose trust boundaries, and
entrypoint signal propagation. CI runs Ruff formatting and lint on the
deployment files, the complete deterministic suite, compileall, the smoke
check, patch hygiene, and a CPU-only container build/run.
The GPU and hardware images are not exercised by CI; their runtime/toolkit,
device permissions, and physical behavior remain operator verification steps.

## Current repository checks

- Run `python -m compileall -q src tests tools` to compile the package and checks.
- Run `python -m pytest` for the complete current suite: configuration,
  synthetic and OpenCV sources, YOLO conversion, device policy, target
  selection, tracking, unified runtime composition, simulated/serial
  actuation, protocol framing, firmware state policy, cleanup, CI structure,
  and benchmark logic. Every current test is deterministic and uses fakes; it
  requires no hardware, serial device, model weights, display, network, or
  CUDA. The explicit `-m "not hardware and not gpu"` form remains a safe
  subset command if opt-in tests are added later.
- Run `PYTHONPATH=src python -m marine_ptz.demo --config configs/development.yaml --steps 20 --no-sleep` for the deterministic vertical-slice demonstration.
- The ordinary pytest run compiles the platform-neutral C++11 firmware core
  with `g++ -Wall -Wextra -Werror -pedantic` and compares it with the same
  independent golden vectors used by the Python firmware model. Arduino AVR
  compilation is a separate compile-only check described below.

## Arduino Uno compile-only CI

GitHub Actions is configured with a separate Ubuntu firmware job that checks
out with read-only credentials, installs Arduino CLI `1.5.1`,
`arduino:avr@1.8.8`, and official `Servo@1.3.0`, then compiles
`firmware/ptz_controller` for `arduino:avr:uno` with `--warnings all`. It uses
`tools/compile_arduino.sh` with a build directory under the runner temporary
directory. The job never calls `upload`, probes serial devices, accesses
hardware, or uploads build artifacts.

The shared helper checks its pinned prerequisites and may be run locally as
`tools/compile_arduino.sh`; it creates and removes only a temporary directory
that it owns unless a caller supplies a build directory. The locally observed
Uno footprint is 8,120 bytes of flash (25%) and 715 bytes of static RAM (34%).
Warnings from the official AVR core's `new.cpp` may appear with all warnings
enabled; project firmware sources have no observed warnings. Prior remote CI
revisions passed, and the final committed revision must use CI as the
authoritative result. Upload, handshake, and bounded integrated movement have
been observed; watchdog-disconnect timing, electrical measurements,
mechanical soak, and broader operating evidence remain pending.

## Pytest categories

- `unit` is the default for unmarked deterministic tests.
- `integration` is reserved for hardware-free component integration through fakes.
- `hardware` requires explicitly connected physical equipment and is opt-in with `python -m pytest -m hardware`.
- `gpu` requires an explicitly available CUDA device and is opt-in with `python -m pytest -m gpu`.

Unknown markers fail collection under `--strict-markers`. Unmarked tests receive
the `unit` category automatically only when they do not already have a
registered category marker.

Ordinary CI runs `python -m pytest`; all current tests remain hardware/GPU-free
and it does not install the vision extra, access CUDA, or open a camera. It installs `.[dev]`
through the reviewed lightweight `constraints/ci.txt` set. That file is not a
hash-locked cross-platform supply-chain lockfile.

## Controlled hardware revalidation sequence

1. Camera: confirm USB enumeration, supported resolutions, frame rate, and stable image capture without PTZ motion.
2. Power: meter-check the regulated 5 V rail; verify common ground and test each servo with conservative limits.
3. Arduino: test serial framing, acknowledgement/error responses, limit rejection, and startup safe position.
4. Mechanics: identify safe pan and tilt ranges that never hit physical stops.
5. Integration: use a simulated detector/selector to command known positions, then observe bounded physical motion.
6. Vision: introduce the tugboat target; measure detection latency, target-selection stability, and control response.
7. Soak: run the intended session duration while recording telemetry and checking servo supply temperature and stability.

## Acceptance gates to confirm before integration

- target label/model and detection confidence threshold;
- maximum allowed command rate and serial timeout behavior;
- hardware lost-target behavior (the synthetic implementation supports hold and return-to-neutral);
- calibrated servo limits and neutral position; and
- acceptable tracking latency and target reacquisition behavior.

Before camera bench testing, run the vision CLI against a local video with
`--device cpu --max-frames N`, a local model path, and
`--actuator-backend simulated`. Host CUDA validation is a separate manual test
because the coding-agent sandbox may not expose the GPU.

## Unified runtime checks

Hardware-free integration tests inject source, detector, and actuator fakes
through the production `run_vision()` composition. They cover detected and
lost-target commands, the simulated default, hardware interlock rejection,
explicit serial-port requirements, camera/model/first-inference work before
enablement, pre-enable EOF, immediate first hold after enable, no-target and
lost-target hold refresh at the configured rate, configuration-derived 75–105
degree bounds, startup
work exceeding the watchdog duration while firmware is still detached,
handshake failure, detector and protocol faults after enablement,
KeyboardInterrupt, finite EOF, frame validation, no commands after a fault, and
ordered idempotent cleanup.

Cancellation tests deterministically request termination during source/model
startup, serial open, detector inference, selection/control, immediately
before dispatch, and Arduino command-rate waiting. They prove that no new
`ENABLE` or `SET` follows an observed request and that cleanup executes once.
The signal-request object is tested as a state-only mechanism; it performs no
serial or cleanup I/O.

Runtime-level serial integration uses the production `ArduinoSerialActuator`,
`InMemorySerialTransport`, and `FirmwareStateMachine` with fake frames and
detector output. It covers startup `READY`/`HELLO`/`DISABLE`, `ENABLE`, real
controller `SET`, shutdown disablement, lost ENABLE/SET acknowledgements,
protocol faults, handshake failure, watchdog fallback, confirmed-coordinate
conservatism, and cancellation during the actuator's real rate-limit wait.

CLI tests cover configuration defaults, CLI precedence, repeatable target
classes, explicit simulated override of hardware configuration, and rejection
of unarmed Arduino selection. Import-isolation tests load the unified CLI in a
clean subprocess and prove that NumPy, OpenCV, Torch, torchvision,
Ultralytics, and pyserial remain unloaded. Existing rendering tests cover
display `q`, annotation failure, writer construction/open/write failures, and
resource release without opening a real display or file.

BoT-SORT tests keep tracker behavior distinct from detector-only behavior. They
prove the pinned tracker YAML (including disabled ReID and required
`model: auto` compatibility field), persistent `model.track()` arguments,
per-frame GPU-tensor-to-CPU ID conversion, rejection of missing/malformed IDs,
separate model/tracker ownership per detector, two-hit acquisition, changed-ID
confirmation reset, protected lock during a short unsupported interval, no
prediction-only target/zoom coordinates, expiry followed by ordinary
reacquisition, and stable ranking. Replay, single-runtime, and concurrent tests
also prove that tracker configuration reaches the production detector/selector
without giving the tracker direct controller or actuator access.

Concurrent-runtime tests exercise the capacity-one replacement channel,
single-owner capture/detector/actuator contexts, fresh zero-detection hold,
stale-result shutdown, startup warm-up before enable, source/detector/serial/
display/recorder failures, no commands after cancellation, bounded worker joins,
30 Hz result/display processing with independently bounded 10 Hz control,
post-acknowledgement deadlines, no actuator-rate sleep on non-control renders,
no delayed-deadline bursts, controller advancement only on scheduled updates,
recorder progress between controls, drop accounting, explicit
inference/display/control/encoded labels, and empty/singleton/interpolated metric
percentiles. They use fakes, events, and
injected clocks; they open no camera, display, serial device, model, media,
network, or CUDA context. Existing single-runtime tests remain the compatibility
and rollback contract.

Guided-calibration tests use the production serial actuator with the in-memory
firmware model. They prove explicit arming, initial 90/90, one-axis one-degree
steps, logical/physical reporting, periodic bounded hold refresh, CENTER,
STATUS, configured-limit enforcement, DISABLE, and close behavior without
opening a serial device.

## Dataset preparation checks

Dataset tests are stdlib/PyYAML-only and create temporary text/image fixtures;
they never import OpenCV, Torch, or Ultralytics. They prove that the checked-in
data file defines exactly class `marine_target`, valid class-0 normalized YOLO
boxes are accepted, non-finite/out-of-frame/wrong-class labels are rejected,
missing and orphan labels are reported, ten recording sessions produce a
deterministic 7/2/1 split without session leakage, source staging files remain
unchanged, existing destinations are not overwritten, and dry-run writes
nothing. Injected OpenCV fakes also prove the recorder requests MJPEG at
1920×1080/30 FPS, releases on interruption, and extraction removes exact
duplicates and temporally adjacent candidates without writing during dry-run.
Real camera recording, codec behavior, manual annotation, and CUDA
training/evaluation remain operator-run stages. The frozen V3 metrics and
observations are recorded in `docs/final_results.md`; tests do not reproduce
those GPU/media measurements.

The Platform-import tests use local NDJSON and image-byte fixtures only. They
cover the exact `0: marine_target` metadata mapping, strict normalized boxes,
unannotated-row skipping, exact-basename resolution, missing and ambiguous
images, duplicate filenames, existing-output rejection, dry-run behavior,
relative manifest mappings, count verification, atomic no-partial publication,
source preservation, and compatibility with the existing session splitter.
They do not access Platform URLs or APIs.

Reviewed-negative tests additionally prove that default mode continues to skip
zero-box rows, the explicit flag resolves and copies them under the same strict
policy, their labels are exactly zero bytes, manifest kinds and positive/negative
counts are accurate, metadata and annotation validation are unchanged, and a
negative-row failure cannot publish a partial positive import. Versioned
preparation tests prove a missing v2 YAML is dry-run-only or canonically created,
an existing YAML is validated without rewrite, all paths remain beneath v2, and
v1 files are untouched.

## Offline replay checks

Replay tests inject finite fake image/video sources, detector outputs, and
monotonic clocks into the production runtime. They cover zero/one/multiple
frames, no requested target, immediate and delayed acquisition, post-acquisition
lost-target episodes, latency/error percentiles, command count/rate, inclusive
simulated saturation, EOF, max-frame completion, decode/detector failure,
cancellation, threshold pass/failure/unavailable metrics, strict JSON, atomic
report preservation, path collisions, and hardware-option rejection. They do
not decode a real file, load YOLO weights, access CUDA, display a window, or
open a serial device. The acceptance report is therefore software evidence;
real media, toy-boat evaluation, and physical-servo acceptance remain manual
future work.

These tests validate software ordering and failure policy only. Physical
camera enumeration and EOF behavior for specific codecs, CUDA inference,
serial device identity and permissions, Uno reset timing, firmware upload,
servo direction/limits, watchdog timing, power integrity, and mechanical
motion remain manual hardware checks.

## Serial and firmware checks

Ordinary tests use `InMemorySerialTransport` and `FirmwareStateMachine`; they
must never enumerate or open `/dev/tty*`. They cover command/response grammar,
partial and malformed frames, overflow recovery, sequence wrap, retries,
response correlation, mapping, double-sided limits, watchdog expiry, and
disable-on-close behavior. Golden vectors cover exact-max LF/CRLF, one-byte
overflow, fragmented CRLF, control bytes, signed-32-bit boundaries, error
precedence, accepted/rejected/conflicting duplicates, watchdog refresh, and
unsigned-millis wraparound in both Python and executable host C++.

Firmware runtime-model tests also prove that continuous RX and a full/slow TX
path cannot prevent watchdog expiry, pending output is incremental, and a
pending response is not overwritten. Transport tests use injected pyserial
connections for partial-write completion, zero-progress and elapsed-deadline
failures, proof that no output drain is called, fragmented reads, timeout
override, and mid-frame disconnects. Startup tests cover boot grace, lost
`HELLO`, unsolicited `READY 0`, mismatched correlated `READY`, the handshake
deadline boundary, partial-open cleanup, and sequence wrap.

Run `python tools/serial_protocol_demo.py` for a fake-only handshake and command
cycle. A bench operator may explicitly provide the currently verified
`--port /dev/ttyACM0` only after rechecking that device identity, wiring, and
installation of `.[hardware]`; that physical probe is outside ordinary CI. The
unified runtime adds independent
`arduino_serial`/`--arm-hardware` gates and never enumerates a port.

## Final zoom and evaluation checks

Digital-zoom tests exercise disabled 1.0x identity, maximum clamping, every
edge/corner crop bound, target-coordinate transformation, target-size fill,
smoothing, gradual loss return, headless execution, unchanged full-frame
detector/controller errors, and identical writer/display frames. No test opens
OpenCV; injected image/CV2 fakes implement only the exercised operations.

Evaluation tests build temporary one-class test splits and inject a fake
Ultralytics model. They prove positive versus zero-byte reviewed-negative
classification, thresholded negative-image false-positive count/rate, maximum
negative confidence, fake GPU tensor `detach().cpu()` conversion, conditional
annotated-example calls, missing-label rejection before model load, strict
finite values, and the shared atomic JSON report path. Actual V3 evaluation is
an operator-run CUDA/media command.

Fixed-FPS recorder tests inject monotonic timestamps and a minimal writer fake.
They prove slow processed frames are held to match wall-clock duration within
one encoded-frame interval, fast processed frames are downsampled, submitted
and duplicated counts remain distinct, invalid/backwards clocks fail, writer
errors propagate, and release is idempotent. Runtime integration tests prove
display updates remain one per processed frame while the encoded stream may
contain held frames, and all resources still close after failures.

Evaluation schema-version-2 tests cover both branches: numeric reviewed-negative
metrics when negatives exist, and explicit `null` plus
`not_measured_no_reviewed_negative_images` when none exist. Evaluation batch
defaults to integer 8 and rejects training-only `auto` without loading a model.

The compile-only baseline uses Arduino CLI `1.5.1`,
`arduino:avr@1.8.8`, the official `Servo@1.3.0` library, and Uno FQBN
`arduino:avr:uno`:

```bash
ARDUINO_BUILD_DIR="$(mktemp -d)"
arduino-cli compile \
  --fqbn arduino:avr:uno \
  --warnings all \
  --build-path "${ARDUINO_BUILD_DIR}" \
  firmware/ptz_controller
```

The pre-remediation compile used 7,856 bytes of flash (24%) and 705 bytes of
static RAM (34%). The current retry-cache build uses 8,120 bytes of flash (25%)
and 715 bytes of static RAM (34%). `--warnings all` currently reports warnings
only from the official AVR core's `new.cpp`, not project firmware. Compilation
is verified; upload and controlled physical operation have been observed.
Watchdog-disconnect, power, mechanical-soak, and final operator evidence remain
pending. Host C++ parity must not be reported as an AVR build.
