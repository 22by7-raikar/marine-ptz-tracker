# Test plan

## Current repository checks

- Run `python -m compileall -q src tests tools` to compile the package and checks.
- Run `python -m pytest -m "not hardware and not gpu"` for the CI-equivalent suite: configuration, synthetic and OpenCV sources, YOLO conversion, device policy, target selection, tracking, simulated/serial actuation, protocol framing, firmware state policy, cleanup, CI structure, and benchmark logic. Tests require no hardware, serial device, model weights, display, network, or CUDA.
- Run `PYTHONPATH=src python -m marine_ptz.demo --config configs/development.yaml --steps 20 --no-sleep` for the deterministic vertical-slice demonstration.
- The ordinary pytest run compiles the platform-neutral C++11 firmware core
  with `g++ -Wall -Wextra -Werror -pedantic` and compares it with the same
  independent golden vectors used by the Python firmware model. Arduino AVR
  compilation is a separate compile-only check described below.

## Pytest categories

- `unit` is the default for unmarked deterministic tests.
- `integration` is reserved for hardware-free component integration through fakes.
- `hardware` requires explicitly connected physical equipment and is opt-in with `python -m pytest -m hardware`.
- `gpu` requires an explicitly available CUDA device and is opt-in with `python -m pytest -m gpu`.

Unknown markers fail collection under `--strict-markers`. Unmarked tests receive
the `unit` category automatically only when they do not already have a
registered category marker.

Ordinary CI runs `python -m pytest -m "not hardware and not gpu"`; it does not
install the vision extra, access CUDA, or open a camera. It installs `.[dev]`
through the reviewed lightweight `constraints/ci.txt` set. That file is not a
hash-locked cross-platform supply-chain lockfile.

## Bench sequence after hardware arrives

1. Camera: confirm USB enumeration, supported resolutions, frame rate, and stable image capture without PTZ motion.
2. Power: meter-check the regulated 5 V rail; verify common ground and test each servo with conservative limits.
3. Arduino: test serial framing, acknowledgement/error responses, limit rejection, and startup safe position.
4. Mechanics: identify safe pan and tilt ranges that never hit physical stops.
5. Integration: use a simulated detector/selector to command known positions, then observe bounded physical motion.
6. Vision: introduce the tugboat target; measure detection latency, target-selection stability, and control response.
7. Soak: run the intended session duration while recording telemetry and checking servo supply temperature and stability.

## Acceptance criteria to define before integration

- target label/model and detection confidence threshold;
- maximum allowed command rate and serial timeout behavior;
- hardware lost-target behavior (the synthetic implementation supports hold and return-to-neutral);
- calibrated servo limits and neutral position; and
- acceptable tracking latency and target reacquisition behavior.

Before camera bench testing, run the vision CLI against a local video with
`--device cpu --max-frames N` and a local model path. Host CUDA validation is a
separate manual test because the coding-agent sandbox may not expose the GPU.

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
cycle. A future bench operator may explicitly provide `--port /dev/ttyACM0`
after wiring review and installation of `.[hardware]`; that physical probe is
outside ordinary CI.

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
is verified; upload and all physical validation remain pending. Host C++ parity
must not be reported as an AVR build.
