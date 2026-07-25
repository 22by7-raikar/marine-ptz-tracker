# Test plan

## Current repository checks

- Run `python -m compileall -q src tests tools` to compile the package and checks.
- Run `python -m pytest -m "not hardware and not gpu"` for the CI-equivalent suite: configuration, synthetic and OpenCV sources, YOLO conversion, device policy, target selection, tracking, actuation, cleanup, CI structure, and benchmark logic. Tests require no hardware, model weights, display, network, or CUDA.
- Run `PYTHONPATH=src python -m marine_ptz.demo --config configs/development.yaml --steps 20 --no-sleep` for the deterministic vertical-slice demonstration.

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
