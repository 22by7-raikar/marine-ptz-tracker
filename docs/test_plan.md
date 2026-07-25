# Test plan

## Current repository checks

- Run `python -m compileall -q src tests tools` to compile the package and checks.
- Run `python -m pytest` for configuration, synthetic and OpenCV sources, YOLO result conversion, device policy, marine target selection, tracking, actuation, and cleanup tests. Tests require no hardware, model weights, display, network, or CUDA.
- Run `PYTHONPATH=src python -m marine_ptz.demo --config configs/development.yaml --steps 20 --no-sleep` for the deterministic vertical-slice demonstration.

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
