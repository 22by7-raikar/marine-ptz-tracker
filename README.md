# Marine PTZ Tracker

Foundation for a laptop-driven system that detects and follows a small marine target with a USB camera, a two-servo pan/tilt assembly, and an Arduino Uno.

The laptop owns capture, detection, target selection, control, telemetry, and serial communication. The Arduino only validates bounded pan/tilt commands and drives the servos. Hardware is intentionally not accessed at import time.

## Status

The first hardware-independent vertical slice is implemented: a synthetic camera and target detector feed target selection, proportional tracking control, and a simulated PTZ actuator. The camera, Arduino, and external 5 V / 3 A servo supply are not yet connected, and no real camera, detector, serial, or firmware backend exists.

## Layout

- `src/marine_ptz`: typed configuration, domain values, interfaces, synthetic sources, tracking policy, simulated actuation, and demo composition.
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
python -m pytest
python tools/smoke_check.py
```

## Configuration

`configs/development.yaml` selects the implemented synthetic camera and simulated actuator. `configs/hardware.yaml` records intended hardware values but has no corresponding real backends yet; it must be reviewed against actual camera and serial-device discovery before future use. Neither configuration starts hardware on its own.

See [architecture](docs/architecture.md), [wiring](docs/wiring.md), and the [test plan](docs/test_plan.md) before connecting equipment.
