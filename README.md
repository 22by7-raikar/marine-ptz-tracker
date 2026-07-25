# Marine PTZ Tracker

Foundation for a laptop-driven system that detects and follows a small marine target with a USB camera, a two-servo pan/tilt assembly, and an Arduino Uno.

The laptop owns capture, detection, target selection, control, telemetry, and serial communication. The Arduino only validates bounded pan/tilt commands and drives the servos. Hardware is intentionally not accessed at import time.

## Status

The hardware-independent synthetic slice and a real OpenCV + Ultralytics vision
pipeline are implemented. The vision pipeline reads a camera or media file,
detects configured marine classes, selects a target, and drives only the
simulated PTZ actuator. Arduino, serial, and physical-actuator backends remain
unimplemented.

## Layout

- `src/marine_ptz`: typed configuration, domain values, interfaces, synthetic and OpenCV sources, YOLO detection, tracking policy, simulated actuation, and CLI composition.
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

## Real vision pipeline

The exact validated host command for camera index 0 with CUDA inference is:

```bash
conda run -n marine_ptz python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --source camera:0 \
  --model yolo11n.pt \
  --device cuda:0 \
  --target-class boat \
  --confidence 0.60 \
  --iou 0.45 \
  --image-size 640 \
  --display
```

`--source 0` is also camera index 0. Use `--source path:0` for a file literally
named `0`; other media paths can be passed directly. Target classes may be
repeated or comma-separated. Use `--output annotated.mp4` for annotated video
and `--max-frames N` for a finite run. Press `q` to leave display mode.

`yolo11n.pt` is the lightweight default. Ultralytics may download named
pretrained weights when they are not already cached; pass a local model path
when network-free startup is required. Model weights and generated recordings
are ignored by Git.

## Configuration

`configs/development.yaml` selects the synthetic camera and simulated actuator.
`configs/hardware.yaml` records the OpenCV/Ultralytics vision defaults and
future Arduino settings. The vision CLI always uses simulated actuation and
does not open a serial device. Configuration loading alone never starts
hardware.

See [architecture](docs/architecture.md), [wiring](docs/wiring.md), and the [test plan](docs/test_plan.md) before connecting equipment.
