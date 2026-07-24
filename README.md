# Marine PTZ Tracker

Foundation for a laptop-driven system that detects and follows a small marine target with a USB camera, a two-servo pan/tilt assembly, and an Arduino Uno.

The laptop owns capture, detection, target selection, control, telemetry, and serial communication. The Arduino only validates bounded pan/tilt commands and drives the servos. Hardware is intentionally not accessed at import time.

## Status

Repository bootstrap only. The camera, Arduino, and external 5 V / 3 A servo supply are not yet connected. Detection models and the tracking loop are deliberately out of scope for this initial structure.

## Layout

- `src/marine_ptz`: typed domain values and hardware-independent interfaces.
- `configs`: development defaults and hardware deployment values.
- `firmware/ptz_controller`: Arduino firmware contract and implementation area.
- `docs`: architecture, wiring, bill of materials, tests, and decisions.
- `tests`: hardware-free unit and smoke tests.
- `tools`: local verification helpers.

## Quick check

Python 3.10+ is required. No package installation is needed for the built-in smoke check:

```bash
python3.10 tools/smoke_check.py
```

For unit tests, install the optional development dependency in an environment outside the repository (or an ignored `.venv`) and run:

```bash
python3.10 -m pytest
```

## Configuration

`configs/development.yaml` is safe for development: it declares the intended synthetic-camera and simulated-actuator backend selections. Those backend implementations are not included yet. `configs/hardware.yaml` records intended hardware values and must be reviewed against actual camera and serial-device discovery before use. Neither configuration starts hardware on its own.

See [architecture](docs/architecture.md), [wiring](docs/wiring.md), and the [test plan](docs/test_plan.md) before connecting equipment.
