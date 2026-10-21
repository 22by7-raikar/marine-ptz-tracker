# Evaluation

Four tools measure different things. Do not compare their numbers across
types.

| Tool | Measures | Excludes |
| --- | --- | --- |
| `tools/evaluate_tugboat.py` | Held-out detector precision/recall/mAP | Tracking, control, latency |
| `tools/benchmark_yolo.py` | Warm model inference latency on one image | Capture, selection, control, serial |
| `python -m marine_ptz.replay_cli` | Full production pipeline on finite media, simulated actuation | Physical actuation |
| `tools/benchmark_runtime.py` | `single` vs `concurrent` topology on the same video | Physical actuation |

All reports are strict finite JSON (`null`, never `NaN`), written atomically
via a synced same-directory temporary file, with local paths reduced to
`<local-path:basename>` markers and credentials redacted. Review before sharing
anyway. Named models such as `yolo11n.pt` may download if uncached; pass a local
path for offline work.

## Model results

The pretrained YOLO11n COCO `boat` class produced zero detections of the
physical tugboat, so a custom
one-class `marine_target` model was trained. See
[dataset and training](dataset_training.md).

| Model / split | Confidence | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 validation | — | 0.999 | 1.000 | 0.995 | 0.923 |
| V3 validation | — | 0.998 | 1.000 | 0.995 | 0.939 |
| V3 held-out test (36 positive images) | 0.60 | 1.000 | 0.936 | 0.962 | 0.803 |

The V3 dataset has 359 images (316 positive, 43 reviewed negative) from 19
recording sessions. The held-out split contains **no** reviewed negatives, so
its false-positive rate is not measured; the report marks it
`not_measured_no_reviewed_negative_images` rather than reporting 0. Held-out
timing was 1.7 ms preprocess, 7.4 ms GPU inference, and 1.0 ms postprocess per
image. `configs/hardware.yaml` references a later V4 checkpoint that has no
recorded held-out metrics.

## Offline replay

Replay runs the production `run_vision()` loop — OpenCV source, Ultralytics
detector, `MarineTargetSelector`, proportional controller — with
`SimulatedPTZActuator` fixed. It rejects camera indexes, serial paths, and
`--arm-hardware`; a hardware configuration additionally requires
`--replay-safe`. It uses the `single` runtime so that each processed frame
produces one applied command.

```bash
python -m marine_ptz.replay_cli \
  --config configs/development.yaml \
  --source /path/to/local-input.mp4 \
  --model /path/to/best.pt --device cuda:0 \
  --target-class marine_target \
  --tracker botsort --tracker-config configs/trackers/general_botsort.yaml \
  --report artifacts/replay/report.json \
  --annotated-output artifacts/replay/annotated.mp4
```

Report schema version 3 (version 2 remains readable through
`normalize_replay_report()`; other versions fail closed). Definitions:

- **Selected-target ratio** — frames with a selected target ÷ processed frames.
- **Center error** — `ex = (x - cx) / (width/2)`, `ey = (y - cy) / (height/2)`,
  radial `er = sqrt(ex² + ey²)`, in original full-frame coordinates (never
  zoomed display coordinates). Means, p50/p95/p99, and maximum.
- **Lost-target episode** — consecutive frames without a selected target
  *after* first acquisition.
- **Command rate / processing FPS** — counts ÷ processing-loop time. The loop
  timer stops before cleanup, so shutdown never depresses these numbers.
- **Saturation** — resulting angles equal to a configured limit.
- **Closed loop** — *observations* (each new inference result) and *control
  decisions* (each scheduled controller update) are separate populations.
  Centering statistics use only decisions backed by a current,
  detector-supported locked observation; occlusion, stale, and lost states
  carry identity metadata but no invented error.
- **Identity** — unique IDs, reacquisitions (same-ID recovery vs after
  expiry), and protected-lock ID changes describe run-local tracker behavior,
  not ground-truth identity accuracy.
- **Settling / overshoot** — only for events declared with
  `--settling-event NAME:START_FRAME:END_FRAME:TOLERANCE:DWELL_SECONDS`.

`selector_state_seconds` has a known time-base discrepancy; use selector-state
frame counts and ratios instead.

No acceptance thresholds are built in. Supplied thresholds (for example
`--min-selected-target-ratio`, `--max-p95-inference-ms`) produce exit status 3
on failure after the report is written; an unavailable metric fails with
`metric unavailable`.

An observer failure is a runtime failure: cleanup still runs, no report is
written, and a prior report is preserved. The original runtime error stays
primary; observer and cleanup errors are attached as a
`RuntimeSecondaryFailures` cause.

## Model inference benchmark

```bash
python tools/benchmark_yolo.py --model /path/to/yolo11n.pt \
  --device cuda:0 --image-size 640 --warmup 5 --iterations 30
```

Without `--image` it uses a deterministic blank frame. Model load and first
inference are reported separately from warm statistics (mean, median,
p95/p99, standard deviation, FPS, and Ultralytics component timings). CUDA
timing synchronizes around each sample. Reference on the RTX 3060 Laptop GPU:
`yolo11n.pt` at 640, 6.65 ms mean, 6.96 ms p95; the cold first inference was
about 1.25 s.

## Runtime topology comparison

```bash
python tools/benchmark_runtime.py \
  --config configs/development.yaml \
  --source /path/to/local-input.mp4 \
  --model /path/to/best.pt --device cuda:0 \
  --target-class marine_target --max-frames 600 \
  --report artifacts/benchmarks/runtime-comparison.json
```

Runs `single` then `concurrent` on the same video with simulated actuation and
no display. Finite video is paced at its declared FPS after a detector-readiness
barrier, so capture cannot race ahead during cold model start. Key fields:
`unique_inference_fps`, `display_fps`, `control_hz`, `encoded_output_fps`,
capture-to-command age p50/p95/p99, capacity-one replacement counts, recorder
submitted/consumed/encoded/duplicate counts, fault counters, and shutdown
duration.

Latest-value replacements are intentional overload behavior, not faults.
Encoded duplicate frames are held recorder slots, never counted as inference.
Rates with fewer than two timestamps are `null`.

## Recorded physical run

One physical run in concurrent mode measured roughly 29.5 FPS unique
inference, 40.6 ms p95 result age at control, 57.6 ms p95 capture-to-command
latency, 97.1% of frames with a relevant detection, and zero stale-result or
watchdog faults. Software latency is not actuator response: the rig has no
encoder, so commanded angle is not measured camera angle.
