# Final checkpoint results

The last fully documented dataset/evaluation checkpoint uses the one-class
target `marine_target`, frozen dataset `data/tugboat-v3`, and frozen evaluated
model `runs/detect/runs/tugboat/yolo11n-marine-target-v3/weights/best.pt`.
The current hardware configuration references a later local V4 weight path for
the BoT-SORT runtime. This document contains no V4 precision/recall/mAP table,
so the V3 model metrics below must not be attributed to V4. Generated data,
weights, reports, and recordings remain outside Git.

## Measured model evidence

The pretrained YOLO11n COCO `boat` baseline detected the physical Green Toys
tugboat zero times. That physical observation is not a precision/recall/mAP
measurement.

| Evidence | Confidence | Images | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 custom validation | training validation | — | 0.999 | 1.000 | 0.995 | 0.923 |
| V3 custom validation | training validation | — | 0.998 | 1.000 | 0.995 | 0.939 |
| V3 held-out positive test | 0.60 | 36 positive | 1.000000 | 0.936044 | 0.962213 | 0.802989 |

The complete V3 dataset contains 316 positive and 43 manually reviewed
negative images: 359 images from 19 recording sessions. However, the final
held-out test split contains **zero reviewed-negative images**. Consequently,
its held-out negative false-positive rate and maximum negative confidence are
**not measured** and must never be reported as 0% or 0.0.

The held-out positive run reported 1.7 ms/image preprocessing, 7.4 ms/image GPU
inference, and 1.0 ms/image postprocessing. These component timings are not the
integrated camera-to-actuation loop rate.

V3 was selected because its validation improved mAP50–95 over V1 and the
dataset added reviewed hard negatives. Expansion stopped after the controlled
demonstration goal was met; more adjacent samples would not establish
open-world robustness.

## Evaluation command and report semantics

Evaluation uses a positive integer batch size; `auto` is intentionally rejected
because Ultralytics validation does not accept the wrapper's training-time
automatic-batch sentinel.

```bash
conda run --no-capture-output -n marine_ptz \
  python tools/evaluate_tugboat.py \
  --model runs/detect/runs/tugboat/yolo11n-marine-target-v3/weights/best.pt \
  --data data/tugboat-v3/data.yaml \
  --image-size 640 --device cuda:0 --batch 8 --confidence 0.60 \
  --project runs/tugboat-evaluation --name heldout-test-v3 \
  --report artifacts/evaluation/tugboat-v3-test-schema-v2.json
```

Schema version 2 preserves precision, recall, mAP50, and mAP50–95. With the
actual zero-negative held-out split, reviewed-negative fields are explicit:

```json
{
  "schema_version": 2,
  "held_out_composition": {
    "positive_images": 36,
    "reviewed_negative_images": 0
  },
  "reviewed_negative_metrics": {
    "evaluation_performed": false,
    "status": "not_measured_no_reviewed_negative_images",
    "false_positive_images": null,
    "false_positive_image_rate": null,
    "maximum_confidence": null,
    "annotated_false_positive_directory": null
  }
}
```

When a held-out split contains one or more reviewed negatives, the count, rate,
and maximum confidence remain numeric, and annotated false-positive examples
are saved when any meet the configured threshold. Reports use strict finite
JSON, atomic replacement, and basename-only path markers, but still require
human review before publication.

## Timing and recording evidence

| Layer | Defensible statement |
| --- | --- |
| Camera acquisition capability | The InnoMaker U20CAM supports 1920×1080 MJPEG at 30 FPS. |
| Held-out model inference | 7.4 ms/image GPU inference, separate from 1.7 ms preprocessing and 1.0 ms postprocessing. |
| Integrated unique-frame loop | Approximately 9.4 processed frames/s during the physical demonstration. |
| Actuator command scheduling | Intentionally bounded to approximately 10 Hz. |
| Encoded output | Configured fixed FPS, 30 FPS by default; encoded slots may hold duplicated annotated frames. |

The original raw demonstration encoded each uniquely processed frame once at
30 FPS. Its 27.534-second playback represented approximately 87.8 seconds of
wall-clock operation. The presentation copy was corrected externally by
stretching timestamps with FFmpeg.

Future application recordings use monotonic processed-frame timestamps. A
fixed-FPS recorder duplicates/holds the most recent annotated frame as needed,
so encoded duration follows elapsed wall-clock time within one encoded-frame
interval. Duplicated frames are encoded timeline slots, not newly processed
camera frames; inference, tracking telemetry, display updates, and the 10 Hz
control path remain unchanged.

### Final headless container replay

The measured final candidate was image
`marine-ptz:rc-final-20260804T023059Z`, ID
`sha256:cd62530190fb58df36016bf99f9ce84cf80014cfabece18c03b8572373cd149c`.
It ran finite local media with CUDA inference, concurrent scheduling, and the
simulated actuator; it did not access a camera, serial port, or servos.

| Measurement | Result |
| --- | ---: |
| Captured / inferred / displayed / recorder-submitted frames | 2,635 / 2,635 / 2,635 / 2,635 |
| Source / unique inference / display / control | 30 FPS / 29.995 FPS / 29.996 FPS / 9.976 Hz |
| Active playback / detector readiness | approximately 87.8 s / 7.054 s |
| Capture, render, recorder drops / encoded duplicates | 0 / 0 / 0 / 1 |
| Capture-to-inference-complete p50 / p95 / p99 | 12.70 / 17.75 / 19.81 ms |
| Capture-to-command p50 / p95 / p99 | 29.22 / 44.96 / 47.84 ms |
| Stale-result / watchdog / session faults | 0 / 0 / 0 |
| Shutdown duration | approximately 0.95 ms |

The annotated MPEG-4 output is 640×480, 30/1 FPS, 2,635 frames,
87.834 seconds, and 19,714,734 bytes. The container remained primary
UID/GID 10001:10001; the measured artifact group was GID 1000, and the
container-created permission check was mode 0640, UID 10001, GID 1000.
Ultralytics configuration directories were mode 0700. These identity and GID
values describe the measured host only, not portable deployment defaults.

The ignored `artifacts/container-replay/SHA256SUMS` file contains portable
basenames for the video, final log, and image-identity record. All three hashes
must verify before publication. This successful replay is CUDA/container
performance and lifecycle evidence; it is not a detector generalization,
reviewed-negative, physical actuation, watchdog-disconnect, or field result.

## Observed physical behavior and limitations

- The complete physical system operates with logical and physical limits of
  75–105 degrees, neutral 90/90, both mappings `physical = -logical + 180`, a
  1000 ms firmware watchdog, and a 10 Hz command limit.
- Pan, tilt, target loss, reacquisition, and digital zoom were observed working
  smoothly. Digital zoom changes only display/output coordinates; detection
  and control always use the original full frame. The rig has no optical zoom.
- Live trials correctly rejected the previously problematic hand- and
  forearm-only scenes. This is qualitative live evidence, not a held-out
  negative false-positive rate.
- A wallet on a box in poor lighting produced an intermittent
  out-of-distribution false positive at approximately 0.50–0.77 confidence.
- The intended claim is a controlled, reasonably lit tabletop demonstration
  with one selected Green Toys tugboat—not arbitrary-scene, open-world, marine,
  weatherized, or safety-rated production robustness.

## Final artifacts

- Raw demonstration with the known timebase defect:
  `artifacts/demo/marine-ptz-v3-digital-zoom.mp4`
- Externally time-corrected presentation copy:
  `artifacts/demo/marine-ptz-v3-digital-zoom-realtime.mp4`
- Final evaluation report:
  `artifacts/evaluation/tugboat-v3-test-schema-v2.json`
- Frozen model:
  `runs/detect/runs/tugboat/yolo11n-marine-target-v3/weights/best.pt`

These generated artifacts are ignored by Git. Verify their identity, duration,
contents, and absence of private path or credential data before submission.
The preserved `artifacts/evaluation/tugboat-v3-test.json` is a superseded
schema-v1 report whose zero-valued reviewed-negative fields are misleading for
a zero-negative test split. Keep it only for provenance; do not publish it or
use it in a release manifest.
