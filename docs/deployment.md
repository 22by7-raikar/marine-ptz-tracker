# Container deployment and rollback

The container deployment packages the existing runtime; it does not change its
tracking or safety behavior. `single` remains the default and known rollback
topology. Concurrent execution is selected explicitly. Images are never
published by CI, and the operator supplies all model, configuration, media,
device, and artifact paths.

Compose interpolation applies to the complete selected file before service
selection. To keep CPU verification independent from GPU and hardware inputs,
the services are intentionally split: `compose.yaml` contains only `test`,
`compose.replay.yaml` contains only `replay`, and `compose.hardware.yaml`
contains only the profile-gated `hardware` service. Do not combine these files
with multiple `-f` arguments; select exactly the file for the intended trust
boundary.

## Trust boundaries and prerequisites

The `test` image is CPU-only and contains development dependencies. The
`runtime-headless` image contains the verified CUDA 12.8 PyTorch pair and a
headless vision stack. The `replay` service has one GPU reservation, no camera
or serial device, no network, read-only inputs, and simulated actuation. The
`hardware` service is an operator template behind the `hardware` profile; it
maps only one explicitly named camera and one explicitly named serial device.
The disposable test stage also contains Git and a C++ compiler solely for the
temporary-repository manifest tests and the production C++ protocol harness;
neither tool is present in the runtime stage.

All services run as UID/GID 10001 with every Linux capability dropped,
`no-new-privileges`, no host networking, no Docker socket, a read-only root
filesystem, and an explicit writable `/tmp`. Replay and hardware may write only
under `/artifacts`. The minimal entrypoint performs a read-only preflight and
then uses `exec`, so the Python process receives SIGINT/SIGTERM directly.
Compose `init: true` provides reaping and forwards termination.

Replay and hardware never execute generated files, so their writable `/tmp`
filesystems remain `noexec`, `nosuid`, and `nodev`. CPU verification is the one
deliberate exception: its trusted repository tests compile and execute the C++
protocol parity harness. The test service keeps ordinary `/tmp` as `noexec` and
sets `TMPDIR=/run/marine-ptz-test-tmp`, an isolated 128 MiB tmpfs mounted
`rw,exec,nosuid,nodev`, owned by UID/GID 10001, and restricted to mode 0700.
This executable storage is neither a host bind mount nor present in either
runtime service.

GPU services require a compatible NVIDIA driver, Docker Engine with Compose,
and NVIDIA Container Toolkit configured on the host. The image does not mount
host CUDA directories. Confirm the toolkit independently before deployment;
CUDA hidden in a coding sandbox is not evidence that the deployed host lacks it.

## Dependency and image policy

Use an immutable release tag containing a Git commit, for example
`marine-ptz:git-0123456789ab`; never deploy `latest`. The Dockerfile pins Python
3.11.11 and reviewed direct dependencies. It installs `torch==2.11.0` and
`torchvision==0.26.0` from the official CUDA 12.8 PyTorch index first. It then
installs `opencv-python-headless==5.0.0.93` and the explicit Ultralytics runtime
dependencies, followed by `ultralytics==8.4.104 --no-deps`. The GUI
`opencv-python` wheel is not installed in the headless image. Native development
continues to use the README's verified two-stage installation.

The constraint file is a reviewed direct-dependency snapshot, not a hash-locked
supply-chain lockfile. Record the built image digest and promote that exact
digest between machines. Rebuild deliberately when the base image or a
transitive dependency changes. Image installs pin the build backend and disable
isolated local-project builds so an unrecorded setuptools version is not fetched
during package installation.

## Build and CPU verification

No model, GPU, device, media, display, or network is available to the running
test container:

```bash
docker compose -f compose.yaml build test
docker compose -f compose.yaml run --rm test
```

The target runs repository-wide Ruff formatting and lint, the complete
deterministic pytest suite, compileall, and the import smoke check. Native
development remains:

```bash
python -m pip install --constraint constraints/ci.txt --editable ".[dev]"
ruff format --check src tests tools
ruff check src tests tools
python -m pytest
python -m compileall -q src tests tools
python tools/smoke_check.py
```

## Headless GPU replay

Create a group-writable artifact directory first and supply absolute paths.
The replay process keeps primary UID/GID 10001:10001 and receives only the
directory's numeric GID as a supplementary group. Compose binds the three
inputs read-only and never exposes a camera or serial port:

```bash
install -d -m 2770 "$PWD/artifacts/container-replay"
export MARINE_PTZ_IMAGE=marine-ptz:git-$(git rev-parse --short=12 HEAD)
export MARINE_PTZ_CONFIG="$PWD/configs/development.yaml"
export MARINE_PTZ_MODEL="$PWD/runs/detect/runs/tugboat/yolo11n-marine-target-v3/weights/best.pt"
export MARINE_PTZ_INPUT_VIDEO="/absolute/path/to/input.mp4"
export MARINE_PTZ_ARTIFACTS="$PWD/artifacts/container-replay"
export MARINE_PTZ_ARTIFACT_GID="$(stat -c '%g' "$MARINE_PTZ_ARTIFACTS")"
export MARINE_PTZ_GPU_DEVICE=0
docker compose -f compose.replay.yaml build replay
set -o pipefail
docker compose -f compose.replay.yaml run --rm replay 2>&1 \
  | tee "$MARINE_PTZ_ARTIFACTS/container.log"
```

The command explicitly selects `marine_target`, confidence 0.60, concurrent
mode, and simulated actuation. The annotated video is written beneath the
artifact mount. Docker stdout/stderr is the telemetry/log stream; redirect or
collect it only into the same artifact directory. A named model is not used,
so Ultralytics cannot download weights. `YOLO_CONFIG_DIR=/tmp/ultralytics`
keeps Ultralytics settings beneath the existing writable, `noexec` temporary
filesystem instead of attempting to write under the read-only image. Before
preflight imports Ultralytics, the non-root entrypoint creates both that
directory and its `Ultralytics` child explicitly at mode `0700`. It then sets
runtime umask `0027`; the private-directory setup therefore does not leak an
owner-only umask into OpenCV. The host artifact directory is mode `2770`, so
its setgid bit assigns new outputs to `MARINE_PTZ_ARTIFACT_GID`. Ordinary
regular files such as `annotated.mp4` are consequently created as `0640`:
owner read/write, artifact-group read, and no access for other users. The
container remains primary UID/GID `10001:10001`; only the artifact GID is
supplementary. Runtime `/tmp` remains `noexec,nosuid,nodev`. If the artifact
directory ownership changes, recompute `MARINE_PTZ_ARTIFACT_GID`; never use an
unrelated permissive group or make the directory or output world-accessible.

Finite replay is paced from the input's validated declared FPS only after the
inference worker completes representative detector/CUDA warm-up. Normal EOF
drains the final published frame, in-flight result, render, and recorder
submission. For the 2635-frame, 30 FPS, 87.834-second reference clip, require:

- `source.kind=video`, `declared_fps=30`, `pacing_applied=true`, and
  `capture_outcome=normal_eof`;
- exactly 2635 captured frames and no capture fault;
- inferred, displayed, and recorder-submitted unique counts that are at least
  95% of captured frames under the already-measured warm inference throughput;
- encoded duration within one 30 FPS frame period of the source duration;
- zero stale-result, watchdog, and session faults; and
- bounded shutdown within the configured two-second worker join deadline.

Capacity-one replacement remains intentional, so this check does not require
zero drops or claim that held encoded frames are unique inference results. A
result resembling the defective run (2633 capture drops, two inferences, or one
display/encoded frame) fails acceptance.

## Optional hardware template

Hardware container execution is Linux-host-specific and is not a universal
deployment claim. Verify stable device identity immediately before use. Supply
the exact camera path and exact serial path; never substitute a guessed
`/dev/video*` or `/dev/tty*`. Determine numeric group IDs with `getent group
video` and `getent group dialout`. The aliases inside the container are fixed,
but each maps only the operator-supplied host device.

```bash
install -d -m 2770 "$PWD/artifacts/container-hardware"
export MARINE_PTZ_IMAGE=marine-ptz:git-$(git rev-parse --short=12 HEAD)
export MARINE_PTZ_CONFIG="$PWD/configs/hardware.yaml"
export MARINE_PTZ_MODEL="$PWD/runs/detect/runs/tugboat/yolo11n-marine-target-v3/weights/best.pt"
export MARINE_PTZ_ARTIFACTS="$PWD/artifacts/container-hardware"
export MARINE_PTZ_ARTIFACT_GID="$(stat -c '%g' "$MARINE_PTZ_ARTIFACTS")"
export MARINE_PTZ_CAMERA_DEVICE="/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"
export MARINE_PTZ_SERIAL_DEVICE="/dev/ttyACM0"
export MARINE_PTZ_VIDEO_GID="<numeric-video-gid>"
export MARINE_PTZ_DIALOUT_GID="<numeric-dialout-gid>"
export MARINE_PTZ_GPU_DEVICE=0
docker compose -f compose.hardware.yaml --profile hardware build hardware
set -o pipefail
docker compose -f compose.hardware.yaml --profile hardware run --rm hardware 2>&1 \
  | tee "$MARINE_PTZ_ARTIFACTS/container.log"
```

The paths above are the recorded machine-specific identities; recheck both
immediately before use and stop if either is ambiguous. This host exposed no
stable serial by-id link. The profile does not start by default. The service
preserves `--arm-hardware`, uses the known rollback `single` topology, and does
no device discovery. Its setgid artifact directory, supplementary artifact
GID, and runtime umask give new regular outputs the same group-readable
contract as replay. Before
running, validate the external regulated 5 V servo supply, common ground,
polarity, clearance, 75–105 degree bounds, neutral 90/90, and immediate access
to remove servo power. The firmware watchdog is not an emergency stop.

## Release manifest

Create manifests only from already-generated local evidence. Filenames and
SHA-256 hashes are recorded; parent paths, changed-file names, environment
variables, credentials, usernames, and device identities are not. JSON inputs
are strict and finite. Publication is same-directory, synchronized, atomic,
and no-overwrite unless explicitly authorized:

```bash
python tools/create_release_manifest.py \
  --config configs/hardware.yaml \
  --model runs/detect/runs/tugboat/yolo11n-marine-target-v3/weights/best.pt \
  --evaluation-report artifacts/evaluation/tugboat-v3-test-schema-v2.json \
  --benchmark-report artifacts/benchmarks/runtime-v3.json \
  --demonstration-video artifacts/demo/marine-ptz-v3.mp4 \
  --image-tag marine-ptz:git-$(git rev-parse --short=12 HEAD) \
  --physical-validation-claimed \
  --output artifacts/releases/marine-ptz-v3.json
```

Omit `--physical-validation-claimed` unless the evidence package actually
contains reviewed physical results. Use `--overwrite` only after preserving the
old manifest and obtaining explicit operator approval.

## Deployment verification

1. Verify the Git commit is clean and matches the immutable image label/tag.
2. Review the manifest hashes against the exact model, config, evaluation,
   benchmark, and optional demo files. Never mix a new model with an old config
   or firmware evidence.
3. Run the CPU test target and archive its transcript.
4. Run preflight directly or let the entrypoint run it. A JSON `ok: false`
   result blocks startup before model/device initialization.
5. Run replay, confirm `marine_target`, simulated actuation, expected output
   duration, no stale/watchdog/session faults, and bounded shutdown.
6. For hardware only, follow the demo runbook's staged checks with a safety
   observer. Record exact device identities outside the shareable manifest.
7. Store the container digest, manifest, logs, reports, and checksums together
   beneath the ignored artifact directory; review all evidence before sharing.

The established physical reference is approximately 29.52 unique inference
FPS, 29.49 display FPS, 9.38 Hz control, 55.01 ms capture-to-command p95, 6.88%
encoded duplicates, zero stale/watchdog/session faults, and 122 ms shutdown.
Treat these as comparison evidence, not universal acceptance limits.

## Safe shutdown and incident response

Use `q`/Escape when a display is present, Ctrl+C, or
`docker compose -f compose.hardware.yaml --profile hardware stop -t 5 hardware`.
Do not use `docker kill` during ordinary operation. Python performs
bounded disable/cleanup; if state is uncertain, remove external servo power.
Do not automatically re-enable after a watchdog or protocol failure.

| Incident | Stop/deployment action |
| --- | --- |
| GPU missing | Do not fall back silently; verify NVIDIA Container Toolkit and requested GPU, then rerun replay preflight. |
| Camera unavailable | Do not arm; re-verify the exact by-id path, permissions, and cable. |
| Serial unavailable | Do not guess a tty path; re-verify board identity and dialout group access. |
| Watchdog expiry / `NOT_ENABLED` | Stop, preserve logs, remove power if uncertain, and diagnose timing/transport; never auto-enable. |
| Stale perception | Stop commands and preserve concurrent metrics; use single mode until understood. |
| Corrupted model/artifact | Reject the release when SHA-256 differs; restore the matching known-good evidence set. |
| Shutdown exceeds bound | Remove external power if needed, preserve logs, and do not redeploy that candidate. |

## Exact rollback

Keep each known-good image digest and its matching manifest/evidence set.

1. **Previous image:** stop the service, select the previously recorded
   immutable tag/digest with `MARINE_PTZ_IMAGE`, verify its manifest hashes, run
   preflight and replay without `--build`, then restart. Compose uses
   `pull_policy: never`, so the exact image must already be present and is never
   silently replaced. Never rebuild the old tag.
2. **Previous model/config:** stop first; restore the model and configuration as
   one pair from the prior manifest, verify both hashes plus compatible firmware
   protocol/safety values, run replay, then arm only through the staged runbook.
3. **Native single mode:** stop the container, activate the verified Conda
   environment, verify the same manifest inputs, and run the documented native
   command with explicit `--runtime-mode single`. Do not run container and native
   processes against the same device.

A known-good release has one reviewed Git commit, immutable image digest,
matching hashes for model/config/evidence, the expected `marine_target` class,
the preserved safety record, passing CPU/replay verification, and separately
stored physical evidence when physical validation is claimed.

## Field operator checklist

- Match commit, immutable image digest, manifest, model, config, and protocol.
- Confirm exact devices; no auto-discovery or guessed paths.
- Confirm external 5 V supply, common ground, polarity, clearance, and observer.
- Run preflight and replay before exposing hardware.
- Keep single mode as rollback; arm only with the explicit profile and gate.
- Keep power removal accessible; stop on faults, uncertainty, binding, or heat.
- Preserve logs/artifacts, shut down safely, then remove external servo power.
