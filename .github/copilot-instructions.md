# Marine PTZ repository instructions

- Use Python 3.10+, the `src/marine_ptz` package, type hints, and dataclasses for domain data.
- Keep camera, GPU, serial, and Arduino access behind the project interfaces; no import-time hardware access or hidden global state.
- Tests must use fakes and run with no camera, GPU, Arduino, or serial device.
- Validate Python changes with `python3.10 tools/smoke_check.py`; run `python3.10 -m pytest` when available.
- Do not add ROS, Docker, web/cloud/database features, model downloads, package installs, or a license unless asked.
- Preserve safe servo limits and external 5 V / 3 A supply wiring. Do not commit or push unless explicitly asked.
- Never add datasets, weights, videos, logs, virtual environments, build output, or training runs to Git.
