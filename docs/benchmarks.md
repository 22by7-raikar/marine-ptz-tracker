# YOLO benchmark methodology

`tools/benchmark_yolo.py` measures an Ultralytics model without camera, serial,
servo, or GUI dependencies. It loads a selected model, performs one separately
reported first inference, executes unreported warm-up iterations, then reports
steady-state samples. With no `--image`, it creates a deterministic zero-valued
three-channel frame at `--image-size`; `--image` accepts a local file.

## What latency means

`model_load_ms` is cold model construction time. `first_inference_ms` is the
first prediction after construction. Warm statistics are only the measured
post-warm-up `model.predict` calls, and include the wrapper call, model work,
and CUDA synchronization when CUDA is selected. They do not include camera
capture, image transport, target selection, PTZ control, serial transport,
servo movement, or end-to-end display/output work.

The report records mean, median, p95, p99, population standard deviation,
approximate FPS, and the averaged Ultralytics preprocess/inference/postprocess
timings when results provide them. CUDA timing synchronizes before and after
each timed section; CPU timing deliberately does not synchronize.

## Running reproducibly

Install the verified vision environment as documented in the README, then use a
local model path to avoid any named-model download:

```bash
python tools/benchmark_yolo.py \
  --model /path/to/yolo11n.pt \
  --device cpu \
  --image-size 640 \
  --warmup 5 \
  --iterations 30
```

For CUDA, change `--device cuda:0`. Keep the same model, image, image size,
warm-up count, measured count, power mode, driver, and background load when
comparing runs. Blank-frame results are suitable for model-inference baselines,
not scene realism. Local image decoding is performed before timing and is not a
camera benchmark. JSON reports default to ignored `artifacts/benchmarks/` and
record structured configuration, the resolved device, allowlisted versions and
GPU metadata, and a sanitized invocation. Report serialization rejects
non-finite or unsupported JSON values. A complete same-directory temporary file
is flushed and synchronized before it atomically replaces the destination, so a
failed write does not overwrite a prior report.

## Preliminary host result — July 24, 2026

This is a preliminary blank-frame model benchmark, **not** measured
camera-to-servo latency.

| Item | Value |
| --- | --- |
| Hardware | NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6 |
| Torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| Ultralytics / OpenCV | 8.4.104 / 5.0.0 |
| Model and input | `yolo11n.pt`, 640×640 blank frame |
| Warm-up / measured | 5 / 30 iterations |
| Mean / median | 6.65 ms / 6.56 ms |
| p95 | 6.96 ms |
| Approximate FPS | 150.36 |
| Reported Ultralytics inference | 5.4567 ms |

A separate first CLI inference observation of **1249.5 ms** is cold
initialization, not steady-state latency, and must not be compared to the warm
sample statistics above.

## Future result template

| Scenario | Camera | Target | Model/input | Warm-up / measured | Mean | p95 | End-to-end notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Real camera inference | TBD | Green Toys tugboat | TBD | TBD | TBD | TBD | Capture-to-detection only |
| Toy-boat tracking | TBD | Green Toys tugboat | TBD | TBD | TBD | TBD | Detection through simulated PTZ |
| End-to-end PTZ | TBD | Green Toys tugboat | TBD | TBD | TBD | TBD | Include camera, control, serial, and servo behavior |
