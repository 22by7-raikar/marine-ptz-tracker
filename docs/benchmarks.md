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
GPU metadata, and a sanitized invocation. The field type takes precedence over
the value syntax: path-typed `--image`, `--output`, `--report`, and `--config`
values always become basename-only `<local-path:basename>` markers, even when
they look like URLs. Model fields preserve safe named identifiers such as
`yolo11n.pt`, use `<local-path:basename>` for filesystem paths, and use
`<remote-resource:basename>` for remote URLs. Marker derivation discards URL
authority, credentials, query, and fragment. Empty, malformed, or encoded
ambiguous basenames use the matching `...:redacted` marker. Credentials and
sensitive argument values are replaced with `<redacted>`. Reports are designed
to be shareable, but must still be reviewed before publication. Report serialization
rejects non-finite or unsupported JSON values. A complete same-directory
temporary file is flushed and synchronized before it atomically replaces the
destination, so a failed write does not overwrite a prior report.

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

Offline replay is a separate tool and evidence type. It runs finite local media
through the full source/detector/selector/controller/simulated-actuator path
and records acceptance metrics, but it is not a blank-frame inference benchmark
and does not establish physical tracking. See [offline replay](replay.md).

## Single versus concurrent local-video comparison

`tools/benchmark_runtime.py` runs the production composition twice against the
same existing local video and local model, always with simulated actuation and
no display. OpenCV validates finite-video FPS. Concurrent mode uses that source
FPS with the production monotonic playback schedule; the single comparison uses
the same deadline/skip policy through its finite-source wrapper. Capture cannot
race to EOF during cold detector initialization because the first validation
frame is held behind the detector-readiness barrier. It first selects `single`,
then `concurrent`, and writes one strict atomic JSON report:

```bash
python tools/benchmark_runtime.py \
  --config configs/development.yaml \
  --source /path/to/local-input.mp4 \
  --model /path/to/local-model.pt \
  --device cuda:0 --target-class marine_target \
  --max-frames 600 \
  --report artifacts/benchmarks/runtime-comparison.json
```

Compare unique inference FPS, display/result-consumption FPS,
capture-to-command age p50/p95/p99, intentional capture/result/recorder
replacements, control Hz, recorder-submitted and recorder-consumed unique
frames, encoded frames, encoded duplicate ratio, and shutdown duration. The
terminal and report names are explicit: `unique_inference_fps`, `display_fps`,
`control_hz`, and `encoded_output_fps`. In the default headless comparison,
`display_fps` is the main-loop rate at which results are made display-ready;
recorder and encoded counts are zero because `--output` is not selected.
The live overlay's encoded rate is the configured fixed timeline; the completed
runtime metrics calculate the observed encoded rate from encoded count and
submission span.
Encoded held/duplicate frames are never counted as unique inference or recorder
submissions. Keep the local video, model,
image size, frame limit, device, power mode, and background load fixed. The
source and model must already exist locally; this command does not make a
camera, serial, servo, or physical-performance claim. Basename-only path
markers make the report easier to share, but it still requires review before
publication.

For an isolated terminal run of either topology on local media, keep every
argument identical except the explicit mode:

```bash
python -m marine_ptz.vision_cli \
  --config configs/development.yaml \
  --source path:/path/to/local-input.mp4 \
  --model /path/to/local-model.pt --device cuda:0 \
  --target-class marine_target --actuator-backend simulated \
  --runtime-mode single --max-frames 600

python -m marine_ptz.vision_cli \
  --config configs/development.yaml \
  --source path:/path/to/local-input.mp4 \
  --model /path/to/local-model.pt --device cuda:0 \
  --target-class marine_target --actuator-backend simulated \
  --runtime-mode concurrent --max-frames 600
```

Use the comparison tool—not ad hoc terminal timing—for the shareable paired
report.

The schema reports source kind, declared FPS, whether pacing was applied,
capture outcome (`normal_eof`, `capture_fault`, cancellation, or an explicit
stop), captured/inferred/displayed counts, capacity-one replacement counts,
recorder submissions, encoded/held frames, fault counters, latency summaries,
and bounded shutdown. Event rates require at least two timestamps; a singleton
is reported with a null rate instead of a fabricated FPS. For a complete replay,
`normal_eof` is expected. A capture fault is never normalized into EOF.

For paced finite video, `captured_frames` includes every successfully published
frame, including the initial validation frame used for representative detector
warm-up. `capture_fps.count` is therefore identical to that total. Its elapsed
window is active playback only: it begins when the detector-readiness barrier
releases and ends at the last captured frame. The validation frame anchors the
first of the `(count - 1)` playback intervals; readiness time is not silently
discarded, but appears separately as `source.detector_readiness_ms` and as
`detector_readiness_ms` in each paired benchmark result. Thus 2635 frames at
30 FPS span approximately `2634 / 30 = 87.8` active seconds and report roughly
30 capture FPS even if cold readiness takes several additional seconds. Live
camera capture retains its backward-compatible first-capture-to-last-capture
window and reports null detector-readiness duration because it has no finite
playback barrier.

Latest-value drops are intentional overload behavior, not faults. The report
keeps them separate from stale-result, watchdog, and session-fault counters.
`recorder_submitted_unique_frames` counts main-loop offers to the capacity-one
recorder handoff; `recorder_consumed_unique_frames` counts values actually
accepted by the wall-clock recorder. `encoded_frames` includes held slots, and
`encoded_duplicate_ratio` is held duplicate slots divided by encoded slots.

The final measured 2,635-frame CUDA/container replay and its exact evidence
boundaries are recorded in [final results](final_results.md). Do not merge that
simulated-actuator result with the separately observed physical reference or
with the blank-frame model benchmark.

## Remaining evidence template

| Scenario | Camera | Target | Model/input | Warm-up / measured | Mean | p95 | Evidence boundary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Corrected concurrent physical revalidation | InnoMaker U20CAM | Green Toys tugboat | Frozen V3 model | TBD | TBD | TBD | Validate the decoupled render/control schedule; retain single mode as rollback. |
| Controlled watchdog disconnect | InnoMaker U20CAM | Green Toys tugboat | Frozen V3 model | TBD | TBD | TBD | Measure detach behavior only; the watchdog is not an emergency stop. |
| Mechanical/electrical soak | InnoMaker U20CAM | Green Toys tugboat | Frozen V3 model | TBD | TBD | TBD | Record supply, heat, binding, and shutdown evidence; do not infer safety certification. |
