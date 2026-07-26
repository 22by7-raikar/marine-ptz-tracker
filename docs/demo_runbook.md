# Demonstration runbook

This staged runbook separates verified software evidence from pending physical
evidence. Stop at any failed safety gate. No stage below changes the fact that
the watchdog is a communication fallback, not an emergency stop.

`/dev/serial/by-id/<verified-uno>` is deliberately unresolved. Replace it with
the exact discovered Uno identity before any armed run; never substitute a
guessed `/dev/tty*` path.

## Stage 0 — Safety inspection (PENDING — servo)

| Item | Detail |
| --- | --- |
| Prerequisites | Hardware is disconnected; multimeter available; [wiring plan](wiring.md) reviewed. |
| Command | None. Inspect wiring and measure the external supply before connecting it. |
| Expected result | Confirm external servo supply polarity and voltage, common Arduino/supply ground, conservative configured limits, camera removed for first motion, and one-servo-only first test. Confirm servos are **not** powered from Arduino 5 V or USB. |
| Stop conditions | Unknown polarity/voltage, missing common ground, mechanical binding, or any uncertainty about the supply. |
| Evidence | Wiring photos, supply-voltage reading, and signed inspection checklist. |

Emergency action for every powered stage: **remove external servo power**. Do
not substitute USB unplugging for this action.

## Stage 1 — Software-only synthetic demo (PASS — software validated)

| Item | Detail |
| --- | --- |
| Prerequisites | Project checkout and `marine_ptz` Conda environment. No camera, Arduino, serial device, GPU, or model is needed. |
| Command | `conda run -n marine_ptz python -m marine_ptz.demo --config configs/development.yaml --steps 20 --no-sleep` |
| Expected result | Twenty concise deterministic telemetry lines with synthetic target motion and simulated actuator state; no physical-device access. |
| Stop conditions | Configuration/import error or output that indicates a non-synthetic backend. |
| Evidence | Terminal transcript and the command/version used. |

## Stage 2 — CUDA verification (PASS — host environment previously validated)

The verified host is an RTX 3060 Laptop GPU with Torch `2.11.0+cu128`,
torchvision `0.26.0+cu128`, Ultralytics `8.4.104`, and OpenCV `5.0.0.93`.
This check does not install or change packages.

| Item | Detail |
| --- | --- |
| Prerequisites | Verified host environment; do not treat sandbox CUDA unavailability as host failure. |
| Command | `conda run -n marine_ptz python -c 'import cv2, torch, torchvision, ultralytics; print("torch", torch.__version__); print("torchvision", torchvision.__version__); print("ultralytics", ultralytics.__version__); print("opencv", cv2.__version__); print("cuda_available", torch.cuda.is_available()); print("device_count", torch.cuda.device_count()); print("device_0", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")'` |
| Expected result | Recorded versions match the verified stack and the host reports the RTX 3060 when CUDA is available. |
| Stop conditions | Version mismatch or CUDA unavailable on the physical host; record evidence before changing anything. |
| Evidence | Full terminal transcript and `nvidia-smi` output captured by the bench operator. |

## Stage 3 — Offline image/video replay (PENDING — real media)

| Item | Detail |
| --- | --- |
| Prerequisites | Existing local finite image/video and local model file. Replay is simulated-only and opens neither a live camera nor serial port. |
| Command | `conda run -n marine_ptz python -m marine_ptz.replay_cli --config configs/development.yaml --source /path/to/local-input.mp4 --model /path/to/local-yolo11n.pt --device cpu --target-class boat --max-frames 300 --report artifacts/replay/acceptance.json --annotated-output artifacts/replay/annotated.mp4` |
| Expected result | Strict finite JSON report and, when requested, annotated output. Both destinations are ignored by Git. |
| Stop conditions | Missing/invalid local media or model, decode/inference error, unexpected device access, or a report write error. A named model such as `yolo11n.pt` may download if not cached; use a local path for network-free work. |
| Evidence | Sanitized replay JSON, annotated output if used, command line, media identity, and model checksum/version stored outside Git. |

Exit status `0` means normal completion, `3` means supplied acceptance limits
failed after a report was written, `2` means runtime/input/report failure, and
`130`/`143` mean SIGINT/SIGTERM. For example,
`--min-selected-target-ratio 0.50 --max-p95-inference-ms 50` is an
**illustrative experiment only**, not accepted project criteria. Thresholds
must be reviewed after representative media is measured.

## Stage 4 — Arduino compile, upload, and no-servo handshake

### Compile-only (PASS — AVR compile validated)

| Item | Detail |
| --- | --- |
| Prerequisites | User-local Arduino CLI `1.5.1`, `arduino:avr@1.8.8`, and `Servo@1.3.0`; no board connection required. |
| Command | `tools/compile_arduino.sh` (set `ARDUINO_CLI` and `ARDUINO_CLI_CONFIG_DIR` only when using non-default user-local locations). |
| Expected result | Uno compile succeeds with 8,120 bytes flash (25%) and 715 bytes static RAM (34%); official AVR core `new.cpp` warnings may appear, but project-source warnings are not expected. |
| Stop conditions | Missing pinned prerequisite or compile failure. Diagnose before changing firmware. |
| Evidence | Complete compile transcript and tool/core/library versions. |

### Board discovery and explicit upload (PENDING — serial)

| Item | Detail |
| --- | --- |
| Prerequisites | Uno connected by known USB data cable; servos and external servo supply disconnected. Identify the board without guessing a port: `ls -l /dev/serial/by-id/` and `arduino-cli board list`. |
| Command | After recording the identity, use `arduino-cli upload --fqbn arduino:avr:uno --port /dev/serial/by-id/<verified-uno> firmware/ptz_controller`. |
| Expected result | Upload acknowledgement for the selected, recorded board only. |
| Stop conditions | Ambiguous board identity, wrong FQBN, upload error, or any unexpected reset behavior. |
| Evidence | Board-list output, stable `/dev/serial/by-id/...` path, and upload transcript. |

### No-servo handshake (PENDING — serial)

| Item | Detail |
| --- | --- |
| Prerequisites | Firmware uploaded, servos still disconnected, verified serial identity, 115200 8N1 terminal. |
| Command | Open a terminal on `/dev/serial/by-id/<verified-uno>` and send `HELLO 1`, then `DISABLE 2`. |
| Expected result | `READY 1 1 0.1.0` followed by `ACK 2 DISABLE 90 90`; no servo power or motion is involved. |
| Stop conditions | Missing/malformed/corrupt response, unexpected enablement, or device identity change. |
| Evidence | Timestamped terminal transcript and board identity. |

## Stage 5 — One-servo calibration (PENDING — servo)

| Item | Detail |
| --- | --- |
| Prerequisites | Stages 0 and 4 complete; one servo connected; camera removed; external supply available; direct external-power removal available. |
| Command | No generic motion command is approved before calibration. Use the explicitly armed runtime only after recording the verified port and temporary conservative limits; do not begin with camera-driven motion. |
| Expected result | Controlled movement within a manually observed safe range. |
| Stop conditions | Binding, unexpected direction, rail sag, heat, noise, end-stop approach, or uncertain command origin. Remove external servo power. |
| Evidence | Complete this record: servo identity `<fill>`; pin `<fill>`; direction `<fill>`; neutral `<fill>`; minimum/maximum `<fill>`; binding observations `<fill>`; current/power observations `<fill>`. |

## Stage 6 — Two-axis calibration (PENDING — servo)

| Axis | Servo identity | Pin | Direction | Neutral | Minimum | Maximum | Binding/clearance | Supply/current observation | Reviewer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pan | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| Tilt | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

| Item | Detail |
| --- | --- |
| Prerequisites | One-servo calibration approved, both axes mechanically clear, limits updated only after measured confirmation. |
| Command | No generic motion command is approved before the table is completed. Repeat bounded manual command checks one axis at a time, then both axes at low rate. |
| Expected result | Observed direction, neutral, and safe travel match the recorded table. |
| Stop conditions | Any binding, unexpected interaction, supply instability, or endpoint contact. |
| Evidence | Completed table, photos/video, and supply measurements. |

## Stage 7 — Live camera with simulated actuator (PENDING — camera)

| Item | Detail |
| --- | --- |
| Prerequisites | Verified camera index/format, local model path, no Arduino port, and simulated actuator selected. |
| Command | `conda run -n marine_ptz python -m marine_ptz.vision_cli --config configs/hardware.yaml --source camera:<verified-index> --model /path/to/local-yolo11n.pt --device cuda:0 --target-class boat --actuator-backend simulated --display --max-frames 300` |
| Expected result | Camera frames, detections, selected target, and bounded simulated PTZ telemetry without physical movement. |
| Stop conditions | Wrong camera identity, decode/model failure, unstable telemetry, or a request to arm hardware. |
| Evidence | Camera format/FPS record, terminal telemetry, optional screen recording, and model/version details. |

## Stage 8 — Explicitly armed physical tracking (PENDING — end-to-end)

| Item | Detail |
| --- | --- |
| Prerequisites | Stages 0–7 approved; verified camera and `/dev/serial/by-id/<verified-uno>` identity; limits/directions/neutral recorded; external power emergency action staffed. |
| Command | `conda run -n marine_ptz python -m marine_ptz.vision_cli --config configs/hardware.yaml --source camera:<verified-index> --model /path/to/local-yolo11n.pt --device cuda:0 --target-class boat --actuator-backend arduino_serial --serial-port /dev/serial/by-id/<verified-uno> --arm-hardware --display` |
| Expected result | Startup constructs camera/model/policy, completes `HELLO`/`READY`/`DISABLE`, then enables only after the explicit arming gate. Telemetry shows detection/selection/commands; shutdown attempts bounded `DISABLE` before resources close. |
| Stop conditions | Any malformed serial response, lost acknowledgement, unexpected direction/range, oscillation, power issue, camera/model fault, or uncertain state. Remove external servo power. |
| Evidence | Startup/handshake transcript, telemetry, calibrated limits, video, and shutdown result. Include a controlled watchdog disconnect test with safe clearance: record detach behavior and recovery, never treat it as an emergency-stop test. |

## Stage 9 — Recorded final demonstration (PENDING — end-to-end)

| Item | Detail |
| --- | --- |
| Prerequisites | Approved physical tracking and a prepared safety observer. |
| Command | Record the approved Stage 8 command/run; no additional repository command is required. |
| Expected result | One video shows the terminal startup/arming state, camera view with target annotation, PTZ response, telemetry, safe lost-target behavior, and bounded shutdown. |
| Stop conditions | Any Stage 8 stop condition or missing safety observer/evidence. |
| Evidence | Video, terminal transcript, camera view, telemetry export, wiring/supply photos, and final evidence checklist. Store media outside Git. |
