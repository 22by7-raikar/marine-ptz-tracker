# Test plan

## Current repository checks

- Run `python3.10 tools/smoke_check.py` to compile and import the package with no hardware.
- Run `python3.10 -m pytest` after installing the optional test dependency. Tests must rely on fakes only.

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
- lost-target behavior;
- calibrated servo limits and neutral position; and
- acceptable tracking latency and target reacquisition behavior.
