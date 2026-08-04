# Demonstration runbook

This staged runbook is the repeatable operator procedure. Historical measured
and observed results are summarized in [final results](final_results.md);
`REVALIDATE` labels below are gates for the next run, not claims that the
historical stage never occurred. Stop at any failed safety gate. No stage below
changes the fact that the watchdog is a communication fallback, not an
emergency stop.

The current host exposes no `/dev/serial/by-id` Uno symlink. The verified
`/dev/ttyACM0` fallback is recorded in `configs/hardware.yaml`; recheck that
identity after any USB change. The runtime never auto-discovers a serial port.

## Stage 0 — Safety inspection (REQUIRED before every powered run)

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

## Stage 3 — Offline image/video replay (PASS — finite-media evidence measured)

| Item | Detail |
| --- | --- |
| Prerequisites | Existing local finite image/video and local model file. Replay is simulated-only and opens neither a live camera nor serial port. |
| Command | `conda run -n marine_ptz python -m marine_ptz.replay_cli --config configs/development.yaml --source /path/to/local-input.mp4 --model /path/to/yolo11n-marine-target-v3-best.pt --device cpu --target-class marine_target --confidence 0.60 --max-frames 300 --report artifacts/replay/acceptance.json --annotated-output artifacts/replay/annotated.mp4` |
| Expected result | Strict finite JSON report and, when requested, annotated output. Both destinations are ignored by Git. |
| Stop conditions | Missing/invalid local media or model, decode/inference error, unexpected device access, or a report write error. A named model such as `yolo11n.pt` may download if not cached; use a local path for network-free work. |
| Evidence | Sanitized replay JSON, annotated output if used, command line, media identity, and model checksum/version stored outside Git. |

Exit status `0` means normal completion, `3` means supplied acceptance limits
failed after a report was written, `2` means runtime/input/report failure, and
`130`/`143` mean SIGINT/SIGTERM. For example,
`--min-selected-target-ratio 0.50 --max-p95-inference-ms 50` is an
**illustrative experiment only**, not accepted project criteria. Thresholds
must be reviewed after representative media is measured.

### Runtime-topology comparison (PASS — final concurrent container replay measured)

Before any concurrent hardware attempt, compare both modes on the same local
video with simulated actuation:

```bash
conda run --no-capture-output -n marine_ptz \
  python tools/benchmark_runtime.py \
  --config configs/development.yaml \
  --source /path/to/local-input.mp4 \
  --model /path/to/local-model.pt \
  --device cuda:0 --target-class marine_target \
  --max-frames 600 \
  --report artifacts/benchmarks/runtime-comparison.json
```

Record unique inference FPS, display/result-consumption FPS, frame-age
p50/p95/p99, intentional latest-value drops, control Hz, recorder submission /
consumption, encoded-frame and duplicate-ratio fields, and shutdown duration.
Do not infer physical safety or servo performance from this test.

## Stage 4 — Arduino compile, upload, and no-servo handshake

### Compile-only (PASS — AVR compile validated)

| Item | Detail |
| --- | --- |
| Prerequisites | User-local Arduino CLI `1.5.1`, `arduino:avr@1.8.8`, and `Servo@1.3.0`; no board connection required. |
| Command | `tools/compile_arduino.sh` (set `ARDUINO_CLI` and `ARDUINO_CLI_CONFIG_DIR` only when using non-default user-local locations). |
| Expected result | Uno compile succeeds with 8,120 bytes flash (25%) and 715 bytes static RAM (34%); official AVR core `new.cpp` warnings may appear, but project-source warnings are not expected. |
| Stop conditions | Missing pinned prerequisite or compile failure. Diagnose before changing firmware. |
| Evidence | Complete compile transcript and tool/core/library versions. |

### Board discovery and explicit upload (REVALIDATE before any upload)

| Item | Detail |
| --- | --- |
| Prerequisites | Uno connected by known USB data cable; servos and external servo supply disconnected. Confirm the verified `/dev/ttyACM0` fallback still names the Uno with `arduino-cli board list`. |
| Command | If a re-upload is required, use `arduino-cli upload --fqbn arduino:avr:uno --port /dev/ttyACM0 firmware/ptz_controller`. |
| Expected result | Upload acknowledgement for the selected, recorded board only. |
| Stop conditions | Ambiguous board identity, wrong FQBN, upload error, or any unexpected reset behavior. |
| Evidence | Board-list output, recorded `/dev/ttyACM0` fallback, and upload transcript. |

### No-servo handshake (REVALIDATE before the next armed run)

| Item | Detail |
| --- | --- |
| Prerequisites | Firmware uploaded, servos still disconnected, verified serial identity, 115200 8N1 terminal. |
| Command | Open a terminal on `/dev/ttyACM0` and send `HELLO 1`, then `DISABLE 2`. |
| Expected result | `READY 1 1 0.1.0` followed by `ACK 2 DISABLE 90 90`; no servo power or motion is involved. |
| Stop conditions | Missing/malformed/corrupt response, unexpected enablement, or device identity change. |
| Evidence | Timestamped terminal transcript and board identity. |

## Stage 5 — One-servo calibration (REVALIDATE before the next armed run)

| Item | Detail |
| --- | --- |
| Prerequisites | Stages 0 and 4 complete; one servo connected; camera removed; external supply available; direct external-power removal available. |
| Command | Follow [guided calibration](calibration.md) with one connected axis: `conda run --no-capture-output -n marine_ptz python tools/calibrate_ptz.py --config configs/hardware.yaml --port /dev/ttyACM0 --axis pan --arm-hardware`. It starts at 90/90, steps only one logical degree, refreshes the watchdog while holding, and cannot exceed the staged 75–105 range. Do not begin with camera-driven motion. |
| Expected result | Controlled movement within a manually observed safe range. |
| Stop conditions | Binding, unexpected direction, rail sag, heat, noise, end-stop approach, or uncertain command origin. Remove external servo power. |
| Evidence | Complete this record: servo identity `<fill>`; pin `<fill>`; direction `<fill>`; neutral `<fill>`; minimum/maximum `<fill>`; binding observations `<fill>`; current/power observations `<fill>`. |

## Stage 6 — Two-axis calibration (REVALIDATE before the next armed run)

| Axis | Servo identity | Pin | Direction | Neutral | Minimum | Maximum | Binding/clearance | Supply/current observation | Reviewer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pan | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| Tilt | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

| Item | Detail |
| --- | --- |
| Prerequisites | One-servo calibration approved, both axes mechanically clear, limits updated only after measured confirmation. |
| Command | Complete separate `--axis pan` and `--axis tilt` guided sessions and record logical/physical results. Do not widen limits through the tool. Any later boundary trial requires a separately reviewed configuration/firmware update, one boundary at a time and no more than five degrees per trial. |
| Expected result | Observed direction, neutral, and safe travel match the recorded table. |
| Stop conditions | Any binding, unexpected interaction, supply instability, or endpoint contact. |
| Evidence | Completed table, photos/video, and supply measurements. |

## Stage 7 — Live camera with simulated actuator (PASS — rerun before arming)

| Item | Detail |
| --- | --- |
| Prerequisites | Verified InnoMaker camera path/format, local model path, no Arduino port, and simulated actuator selected. |
| Command | `conda run --no-capture-output -n marine_ptz python -m marine_ptz.vision_cli --config configs/hardware.yaml --model runs/detect/runs/tugboat/yolo11n-marine-target-v3/weights/best.pt --device cuda:0 --target-class marine_target --confidence 0.60 --actuator-backend simulated --runtime-mode single --digital-zoom --max-digital-zoom 2.0 --display --max-frames 300` |
| Expected result | Full-frame detection/control coordinates, bounded simulated PTZ telemetry, and a smoothed software crop in the display without physical movement. |
| Stop conditions | Wrong camera identity, decode/model failure, unstable telemetry, or a request to arm hardware. |
| Evidence | Camera format/FPS record, terminal telemetry, optional screen recording, and model/version details. |

## Stage 8 — Explicitly armed physical tracking (PASS — single mode observed)

| Item | Detail |
| --- | --- |
| Prerequisites | Stages 0–7 approved; verified camera and `/dev/ttyACM0` identity; the staged 75–105 limits, directions, and neutral have been physically observed and recorded; external power emergency action staffed. |
| Command | `conda run --no-capture-output -n marine_ptz python -m marine_ptz.vision_cli --config configs/hardware.yaml --model runs/detect/runs/tugboat/yolo11n-marine-target-v3/weights/best.pt --device cuda:0 --target-class marine_target --confidence 0.60 --actuator-backend arduino_serial --serial-port /dev/ttyACM0 --arm-hardware --runtime-mode single --digital-zoom --max-digital-zoom 2.0 --display` |
| Expected result | Startup validates one camera frame, completes cold first inference and display initialization while detached, then performs `HELLO`/`READY`/`DISABLE`, explicit `ENABLE`, and an immediate 90/90 hold `SET`. Enabling may move a detached mechanism once to neutral; the hold refresh itself must not add motion. `marine_target` corrections begin on the next 10 Hz slot, remain within 75–105, and no-target `hold` resends the last pose. Shutdown attempts bounded `DISABLE` before resources close. |
| Stop conditions | Any malformed serial response, lost acknowledgement, unexpected direction/range, oscillation, power issue, camera/model fault, or uncertain state. Remove external servo power. |
| Evidence | Startup/handshake transcript, telemetry, calibrated limits, video, and shutdown result. Include a controlled watchdog disconnect test with safe clearance: record detach behavior and recovery, never treat it as an emergency-stop test. |

The table deliberately selects the verified `single` fallback. Concurrent mode
remains **PENDING physical validation**. Only after the local-video comparison
and a repeated simulated live-camera run pass may the same controlled command
be repeated with `--runtime-mode concurrent`, a safety observer, immediate
external-power removal access, and the same stop conditions. The concurrent
terminal summary must show a command rate near 10 Hz, fresh-result ages below
0.5 seconds, no session/watchdog faults, and bounded shutdown. It must not be
made the checked-in default based on software evidence alone.

## Stage 9 — Recorded final demonstration (PASS — publication review pending)

Create the ignored local output directory first with
`mkdir -p artifacts/demo`; the runtime does not create a missing video parent.

| Item | Detail |
| --- | --- |
| Prerequisites | Approved physical tracking, a prepared safety observer, measured external 5 V supply polarity/voltage, shared ground, clear mechanism, and immediate access to remove external servo power. Confirm the configured camera and `/dev/ttyACM0` identities before arming. |
| Command | `conda run --no-capture-output -n marine_ptz python -m marine_ptz.vision_cli --config configs/hardware.yaml --model runs/detect/runs/tugboat/yolo11n-marine-target-v3/weights/best.pt --device cuda:0 --target-class marine_target --confidence 0.60 --actuator-backend arduino_serial --serial-port /dev/ttyACM0 --arm-hardware --runtime-mode single --digital-zoom --max-digital-zoom 2.0 --display --output-fps 30 --output artifacts/demo/marine-ptz-v3-digital-zoom-wall-clock-rerun.mp4` |
| Expected result | One annotated video shows tracking and software zoom, while terminal output records startup/arming, full-frame error, logical/physical angles, saturation, inference latency, unique processed FPS, and safe bounded shutdown. The recorder holds annotated frames so 30 FPS playback follows monotonic wall-clock duration; held frames are not new inference/control frames. |
| Stop conditions | Any Stage 8 stop condition or missing safety observer/evidence. |
| Evidence | Video, terminal transcript, camera view, telemetry export, wiring/supply photos, and final evidence checklist. Store media outside Git. |

The original raw artifact at
`artifacts/demo/marine-ptz-v3-digital-zoom.mp4` predates wall-clock scheduling:
its 27.534-second playback represents about 87.8 seconds of operation. Preserve
it for provenance; the rerun command deliberately uses another output path.
The corrected presentation copy is
`artifacts/demo/marine-ptz-v3-digital-zoom-realtime.mp4`. Future recordings
made by the command above are wall-clock-correct in the application and do not
need external PTS stretching.

### Final recorded sequence

Keep the scene controlled and reasonably lit. Use one selected Green Toys
tugboat and do not represent this as open-world detection evidence.

1. Start with no target and show `tracking=lost` at an unzoomed view.
2. Introduce the tugboat and show a `marine_target` lock.
3. Move it left and right to demonstrate bounded pan.
4. Move it up and down to demonstrate bounded tilt.
5. Move it farther away and show automatic digital zoom increasing without
   changing the full-frame tracking error.
6. Partially occlude the tugboat and observe selection behavior.
7. Remove it and show target loss plus the gradual return toward 1.0x.
8. Reintroduce it and show reacquisition.
9. Pass a hand or forearm through without the tugboat and record the expected
   negative behavior.
10. Press `q` or Escape, verify the bounded `DISABLE`/close result in the
    terminal, then remove external servo power.

Stop immediately for buzzing, collision, binding, bracket flex, cable tension,
unexpected direction, serial uncertainty, or supply instability. `Ctrl+C` is
also a supported bounded shutdown path, but external-power removal remains the
physical fallback.
