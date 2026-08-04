# Local tugboat dataset and custom training

This workflow creates a one-class, local-only Ultralytics dataset for the Green
Toys tugboat. The class name is exactly `marine_target` and its numeric class ID
is exactly `0`.

The pretrained YOLO11n COCO baseline was run against the physical toy with
`--target-class boat` and produced zero detections. That observation motivates
custom training; it is not a precision, recall, or mAP measurement and must not
be compared numerically with held-out custom-model metrics.

Recordings, extracted images, labels, manifests, weights, predictions, and
training runs are ignored by Git. Keep a separate backup outside the repository.
None of these tools opens an Arduino or serial device.

## Dataset layout

The checked-in [data file](../data/tugboat/data.yaml) declares this generated
layout:

```text
data/tugboat/
├── data.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Before splitting, use this ignored staging layout:

```text
data/tugboat/staging/
└── session-01/
    ├── images/*.jpg
    ├── labels/*.txt
    └── session-01-manifest.json
```

Each label line is:

```text
0 center_x center_y width height
```

Coordinates are normalized by image width/height. Centers must be in `[0, 1]`,
width and height in `(0, 1]`, and the complete box must remain inside the image.
Use class `0` only. An image with no target is a valid negative example but must
have a matching empty `.txt` label. The preparation tool validates every line,
requires every image/label pair, rejects orphan labels, and reports failures
before copying anything.

## Ten-session capture plan

Record ten distinct sessions rather than one long clip. Keep the tugboat,
camera, and background arrangement fixed within a session, then deliberately
change conditions between sessions:

| Session | Primary variation |
| --- | --- |
| 01 | Diffuse indoor light, front/three-quarter views. |
| 02 | Indoor side views from the toy's left. |
| 03 | Indoor side views from the toy's right. |
| 04 | Higher and lower camera elevation. |
| 05 | Near distances while keeping the full toy visible. |
| 06 | Far distances where the toy occupies fewer pixels. |
| 07 | Cluttered marine-colored background and clean background. |
| 08 | Dimmer but usable light; avoid motion blur. |
| 09 | Bright natural light or outdoor shade without glare. |
| 10 | Controlled toy/camera motion, partial occlusion, and negative frames. |

Aim for 30–60 seconds per session. Include different orientations, scales, and
frame locations. Avoid long static holds and accidental neighboring-session
duplicates. Do not change every factor simultaneously; session identity is the
unit used to prevent leakage across train, validation, and test.

## 1. Capture

With Arduino software stopped and servo power disabled, recheck the configured
camera identity. Record session 01 as 1920×1080, 30 FPS MJPEG:

```bash
conda run --no-capture-output -n marine_ptz \
  python tools/record_dataset.py \
  --config configs/hardware.yaml \
  --output captures/tugboat/session-01.avi \
  --duration 60 \
  --display
```

The default camera is `camera.device` from `configs/hardware.yaml`. `--camera`
may override it with another explicit path. Output must be a new `.avi`; the
tool refuses overwrite, requests MJPEG and the configured 1920×1080 at 30 FPS,
and releases capture/writer/windows on `q`, Escape, Ctrl+C, or failure. Repeat
with `session-02.avi` through `session-10.avi`.

## 2. Extract temporally separated frames

Preview the deterministic selection without writing:

```bash
conda run -n marine_ptz python tools/extract_frames.py \
  --source captures/tugboat/session-01.avi \
  --output-dir data/tugboat/staging/session-01/images \
  --session session-01 \
  --manifest data/tugboat/staging/session-01/session-01-manifest.json \
  --sample-fps 2 \
  --minimum-frame-gap 5 \
  --dry-run
```

Remove `--dry-run` to write the selected JPEGs and JSON manifest. Use a `.csv`
manifest path for CSV instead. Filenames include the session, source frame
index, and timestamp. Exact image-content duplicates are discarded, and two
selected source frames must be separated by at least the configured frame gap.
This is a temporal/exact-duplicate filter, not perceptual near-duplicate image
analysis; manually remove visually redundant frames before annotation.

Repeat extraction independently for all ten clips. Never combine neighboring
frames from different recording sessions under one session name.

## 3. Annotate locally and offline

Use an already-installed local CVAT or Label Studio Community instance bound to
localhost. Do not use hosted projects, cloud storage connectors, Roboflow, API
tokens, or a cloud account. If an installation offers cloud login, remain in
its local/offline mode; this repository does not integrate with either tool.

For an already-installed Label Studio Community executable, keep all state in
the ignored dataset directory and launch it locally with:

```bash
label-studio start --data-dir "$PWD/data/tugboat/label-studio-state"
```

Open only the localhost URL it prints. No hosted Label Studio account is part
of this workflow. Import and export files manually through the local UI.

1. Create one local project with one rectangle label named exactly
   `marine_target`.
2. Import one session's `images/` directory at a time and retain the session ID
   in the task/project name.
3. Draw a tight box around the complete visible toy. Use the same label for
   partial occlusion; do not label reflections, pictures, or unrelated boats.
4. Mark genuine no-tugboat frames as negative examples.
5. Review every box at full resolution, then export Ultralytics/YOLO detection
   labels locally.
6. Place matching `.txt` files under that session's `labels/` directory. Create
   an empty matching label file for each retained negative image if the export
   omitted it.

Do not use automated cloud annotation or mix sessions during export. A final
manual reviewer should check class name, missing boxes, overly loose boxes, and
edge-clipped boxes before splitting.

## 4. Import a private Ultralytics Platform NDJSON export

Platform access and export remain manual. Download the private Detect dataset's
NDJSON export through the Platform UI, then place that local file somewhere
ignored, for example `data/tugboat/platform-export/tugboat.ndjson`. The importer
does not call a Platform API or use URLs embedded in the export.

Validate the export and its exact-basename matches without writing:

```bash
conda run -n marine_ptz python tools/import_ultralytics_ndjson.py \
  --ndjson data/tugboat/platform-export/tugboat.ndjson \
  --source-staging-root data/tugboat/staging \
  --output-staging-root data/tugboat/platform-imported \
  --dry-run
```

The metadata must describe task `detect` with exactly
`{"0": "marine_target"}`. Only rows containing one or more valid normalized
`annotations.boxes` entries are imported; rows without boxes are counted and
skipped. Each exported filename is reduced to its exact basename and must match
exactly one image below `data/tugboat/staging/<session>/images/`. Duplicate,
missing, or ambiguous basenames fail the complete import.

After resolving every failure, remove `--dry-run`:

```bash
conda run -n marine_ptz python tools/import_ultralytics_ndjson.py \
  --ndjson data/tugboat/platform-export/tugboat.ndjson \
  --source-staging-root data/tugboat/staging \
  --output-staging-root data/tugboat/platform-imported
```

The output root must not already exist. The importer copies only annotated
images into `platform-imported/<session>/images/`, writes matching YOLO labels
under `platform-imported/<session>/labels/`, and writes
`platform-imported/ultralytics-ndjson-import.json`. Manifest paths are relative
to the two staging roots. The original images, captures, and staging labels are
never changed. The Platform split field is retained for traceability but is not
used for allocation; the existing preparation tool re-splits complete original
recording sessions to prevent neighboring-frame leakage. Keep the private export
and generated imported tree out of Git.

Do **not** point `prepare_tugboat_dataset.py` at the original
`data/tugboat/staging` when Platform contains annotations for only a subset of
those frames. The original tree would correctly fail its image/label pairing
contract.

If annotations came from the local CVAT/Label Studio workflow instead, skip the
NDJSON import and keep using the session staging tree with manually exported
matching labels.

## 5. Validate and split by recording session

Use `data/tugboat/platform-imported` below for the Platform path. For the fully
local annotation path, substitute the complete labeled
`data/tugboat/staging` tree instead.

First perform a non-writing validation and show the deterministic 70/20/10
whole-session allocation:

```bash
conda run -n marine_ptz python tools/prepare_tugboat_dataset.py \
  --staging-root data/tugboat/platform-imported \
  --output-root data/tugboat \
  --data data/tugboat/data.yaml \
  --seed 20260802 \
  --dry-run
```

After fixing every reported failure, remove `--dry-run`:

```bash
conda run -n marine_ptz python tools/prepare_tugboat_dataset.py \
  --staging-root data/tugboat/platform-imported \
  --output-root data/tugboat \
  --data data/tugboat/data.yaml \
  --seed 20260802
```

With ten sessions the default allocation is seven train, two validation, and
one test session. A session is never divided between splits. The tool copies
validated image/label pairs, never changes staging images or raw recordings,
and refuses to overwrite an existing prepared sample. Delete or archive an old
generated split deliberately before rerunning; the tool never cleans it.

## 6. Fine-tune YOLO11n

Use a verified local YOLO11n starting checkpoint to avoid a download:

```bash
conda run --no-capture-output -n marine_ptz \
  python tools/train_tugboat.py \
  --data data/tugboat/data.yaml \
  --model /path/to/local/yolo11n.pt \
  --epochs 50 \
  --image-size 640 \
  --device cuda:0 \
  --batch auto \
  --project runs/tugboat \
  --name yolo11n-marine-target
```

Defaults are 50 epochs, image size 640, `cuda:0`, and Ultralytics automatic
batch sizing. The wrapper validates the one-class data file and refuses an
existing run directory. It does not install packages or download a checkpoint
itself, but a bare named model such as `yolo11n.pt` may cause Ultralytics to
download when it is not cached; provide the local path above for offline work.

## 7. Evaluate only on the held-out test sessions

```bash
conda run --no-capture-output -n marine_ptz \
  python tools/evaluate_tugboat.py \
  --model runs/tugboat/yolo11n-marine-target/weights/best.pt \
  --data data/tugboat/data.yaml \
  --image-size 640 \
  --device cuda:0 \
  --batch 8 \
  --project runs/tugboat-evaluation \
  --name heldout-test \
  --report artifacts/evaluation/tugboat-test.json
```

The wrapper evaluates `split=test`, prints precision, recall, mAP50, and
mAP50–95, atomically writes a strict JSON report, and writes annotated
predictions plus YOLO prediction text under
`runs/tugboat-evaluation/heldout-test-predictions/`. Review false positives,
false negatives, tightness, lighting, distance, and occlusion rather than
selecting a checkpoint solely by one aggregate metric.

## 8. Run the custom model without physical actuation first

```bash
conda run --no-capture-output -n marine_ptz \
  python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --model runs/tugboat/yolo11n-marine-target/weights/best.pt \
  --device cuda:0 \
  --target-class marine_target \
  --actuator-backend simulated \
  --display
```

Only after representative live-camera and offline replay evidence is reviewed
should the separately gated Arduino backend be considered. Custom-model metrics
remain model/media evidence; they do not prove physical tracking safety.

## Version 2 reviewed-negative workflow

Version 1 artifacts are immutable inputs to this workflow. The cumulative v2
export is `data/tugboat/platform_export/tugboat-marine-target_v2.ndjson`; import
it only to the new `data/tugboat/platform-imported-v2` tree and prepare it only
under `data/tugboat-v2`.

By default, the importer continues to skip every zero-box row. The option
`--include-unannotated-as-negatives` is deliberately explicit: **use it only
when every unboxed exported image has been manually reviewed and confirmed not
to contain the tugboat**. It is appropriate for the reviewed forearm, hand,
face, person, shadow, and low-light background images described for v2. Never
use it to bulk-convert arbitrary unlabeled images into negatives.

Validate the cumulative export without writing:

```bash
conda run -n marine_ptz python tools/import_ultralytics_ndjson.py \
  --ndjson data/tugboat/platform_export/tugboat-marine-target_v2.ndjson \
  --source-staging-root data/tugboat/staging \
  --output-staging-root data/tugboat/platform-imported-v2 \
  --include-unannotated-as-negatives \
  --dry-run
```

Review the positive, reviewed-negative, skipped, label, and session counts.
Then publish the new staging tree:

```bash
conda run -n marine_ptz python tools/import_ultralytics_ndjson.py \
  --ndjson data/tugboat/platform_export/tugboat-marine-target_v2.ndjson \
  --source-staging-root data/tugboat/staging \
  --output-staging-root data/tugboat/platform-imported-v2 \
  --include-unannotated-as-negatives
```

Positive manifest entries are marked `positive`. Reviewed zero-box entries are
marked `reviewed_negative` and receive a zero-byte YOLO label. Exact-basename
resolution and every existing metadata, annotation, preflight, and atomic
publication check apply to both kinds.

Plan the separate v2 dataset and its new YAML:

```bash
conda run -n marine_ptz python tools/prepare_tugboat_dataset.py \
  --staging-root data/tugboat/platform-imported-v2 \
  --output-root data/tugboat-v2 \
  --data data/tugboat-v2/data.yaml \
  --seed 20260802 \
  --dry-run
```

When the YAML is absent, dry-run writes nothing. The real command creates a
canonical one-class YAML using only relative v2 paths. If it already exists, it
must validate and is never overwritten. Prepare after reviewing the session
allocation:

```bash
conda run -n marine_ptz python tools/prepare_tugboat_dataset.py \
  --staging-root data/tugboat/platform-imported-v2 \
  --output-root data/tugboat-v2 \
  --data data/tugboat-v2/data.yaml \
  --seed 20260802
```

Train with the required v2 run name and a verified local starting checkpoint:

```bash
conda run --no-capture-output -n marine_ptz \
  python tools/train_tugboat.py \
  --data data/tugboat-v2/data.yaml \
  --model /path/to/local/yolo11n.pt \
  --epochs 50 \
  --image-size 640 \
  --device cuda:0 \
  --batch auto \
  --project runs/tugboat \
  --name yolo11n-marine-target-v2
```

Evaluate without reusing v1 report or prediction directories:

```bash
conda run --no-capture-output -n marine_ptz \
  python tools/evaluate_tugboat.py \
  --model runs/tugboat/yolo11n-marine-target-v2/weights/best.pt \
  --data data/tugboat-v2/data.yaml \
  --image-size 640 \
  --device cuda:0 \
  --batch 8 \
  --project runs/tugboat-evaluation \
  --name heldout-test-v2 \
  --report artifacts/evaluation/tugboat-v2-test.json
```

Run the v2 weights with simulated actuation first:

```bash
conda run --no-capture-output -n marine_ptz \
  python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --model runs/tugboat/yolo11n-marine-target-v2/weights/best.pt \
  --device cuda:0 \
  --target-class marine_target \
  --actuator-backend simulated \
  --display
```

Only after v2 held-out review, simulated live tracking, wiring, calibration,
limits, and the physical runbook are approved may an operator use the explicitly
armed physical form:

```bash
conda run --no-capture-output -n marine_ptz \
  python -m marine_ptz.vision_cli \
  --config configs/hardware.yaml \
  --model runs/tugboat/yolo11n-marine-target-v2/weights/best.pt \
  --device cuda:0 \
  --target-class marine_target \
  --actuator-backend arduino_serial \
  --serial-port /dev/ttyACM0 \
  --arm-hardware \
  --display
```

The physical command remains pending operator validation. Training and software
tests do not prove physical tracking or servo safety.
