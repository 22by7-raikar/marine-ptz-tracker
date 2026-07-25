# Repository guidance for Codex

## Scope and architecture

- This project is a laptop-driven marine PTZ tracker; preserve the boundary between laptop software and Arduino servo control.
- Keep all camera, GPU, serial, and GPIO access behind `CameraSource`, `Detector`, `TargetSelector`, `PTZController`, or `Actuator` implementations. Do not access hardware during import or through module globals.
- Keep base domain values small, typed, and immutable where practical. Preserve the separation between vision, target selection, control, and simulated versus physical actuation.
- Do not introduce ROS, Docker, a web UI, cloud services, a database, model downloads, or package installation unless explicitly requested.

## Build and test

- Target Python 3.10+ and use the `src/` layout.
- Run `python3.10 tools/smoke_check.py` after Python changes.
- Run `python3.10 -m pytest` when pytest is available. Tests must use fakes and must not require a camera, GPU, Arduino, or serial device.
- Keep configuration changes synchronized with the relevant documentation.
- Preserve the verified host vision stack: torch 2.11.0+cu128, torchvision 0.26.0+cu128, ultralytics 8.4.104, and opencv-python 5.0.0.93. Follow the README's two-stage constrained installation; never use an unconstrained command that may replace the cu128 pair.
- A coding-agent sandbox may not expose CUDA even when the host does. Do not change implementation or dependencies in response to sandbox-only CUDA unavailability.

## Preservation rules

- Do not commit, push, add a license, or rewrite unrelated user changes.
- Keep datasets, model weights, videos, logs, virtual environments, build output, and generated training runs out of Git.
- Treat servo power wiring as safety-critical: preserve command limits and the separate external servo supply requirement in code and docs.
