# Operations

Staged bring-up, normal operation, shutdown, and failure response for the
physical rig. Read [hardware](hardware.md) first. Stop at any failed step.

The examples use these local paths; substitute your own:

```bash
UNO=/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_9503331303135180C052-if00
MODEL=/path/to/marine-target/weights/best.pt
```

## Bring-up

Run the stages in order. Stages 1–3 touch no hardware.

1. **Synthetic loop** — no camera, model, GPU, or serial:

   ```bash
   python -m marine_ptz.demo --config configs/development.yaml --steps 20 --no-sleep
   ```

2. **CUDA stack** — confirm the pinned versions and the GPU:

   ```bash
   python -c 'import torch, torchvision, ultralytics, cv2; print(torch.__version__, torchvision.__version__, ultralytics.__version__, cv2.__version__, torch.cuda.is_available())'
   ```

   Expect `2.11.0+cu128 0.26.0+cu128 8.4.104 5.0.0 True`. A coding sandbox
   may hide CUDA even when the host has it.

3. **Offline replay** on local media with simulated actuation — see
   [evaluation](evaluation.md).

4. **Firmware** — compile, then upload with servo power disconnected:

   ```bash
   tools/compile_arduino.sh
   arduino-cli upload --fqbn arduino:avr:uno --port "$UNO" firmware/ptz_controller
   ```

5. **No-servo handshake** — servos disconnected, 115200 8N1 terminal on
   `$UNO`. Send `HELLO 1` then `DISABLE 2`; expect `READY 1 1 0.1.0` then
   `ACK 2 DISABLE 90 90`.

6. **Calibration** — one axis at a time, camera removed for the first run. See
   [hardware](hardware.md#limits-and-calibration).

7. **Live camera, simulated actuator:**

   ```bash
   python -m marine_ptz.vision_cli --config configs/hardware.yaml \
     --model "$MODEL" --device cuda:0 --target-class marine_target \
     --actuator-backend simulated --runtime-mode single \
     --digital-zoom --max-digital-zoom 2.0 --display --max-frames 300
   ```

8. **Armed tracking** — external power removal staffed:

   ```bash
   python -m marine_ptz.vision_cli --config configs/hardware.yaml \
     --model "$MODEL" --device cuda:0 --target-class marine_target \
     --actuator-backend arduino_serial --serial-port "$UNO" --arm-hardware \
     --runtime-mode single --digital-zoom --max-digital-zoom 2.0 --display
   ```

   Startup validates a camera frame, completes the cold first inference and
   display setup while detached, then performs `HELLO`/`READY`/`DISABLE`,
   `ENABLE` (which may move a detached mechanism once to neutral), and an
   immediate 90/90 hold `SET`. Corrections start on the next 10 Hz slot.

`single` is the default and rollback runtime. `--runtime-mode concurrent` was
exercised in one physical run; its decoupled render/control
schedule is hardware-free tested but not yet physically revalidated, so keep
`single` as the checked-in default.

To record, create the output directory first (`mkdir -p artifacts/demo`) and
add `--output artifacts/demo/run.mp4 --output-fps 30`. The recorder holds the
latest annotated frame so playback follows wall-clock time; held frames are not
new inference or control frames.

## Shutdown

Press `q` or Escape (with a display) or Ctrl+C. Cleanup attempts a bounded
`DISABLE` before closing serial, camera, writer, and display, then exits. Remove
external servo power afterwards.

| Exit status | Meaning |
| --- | --- |
| 0 | Normal completion (EOF, frame limit, `q`). |
| 2 | Runtime, input, or report failure. |
| 3 | Replay acceptance thresholds failed (report still written). |
| 130 / 143 | SIGINT / SIGTERM after bounded cleanup. |

## Failure response

The software never re-enables automatically after a fault. When physical state
is uncertain, **remove external servo power**; unplugging USB is not a
substitute.

| Failure | Software and firmware behavior | Operator action |
| --- | --- | --- |
| No target | `hold` resends the last safe pose at 10 Hz, which keeps the watchdog fed. | None if behavior is as expected. |
| Stale concurrent perception | Result older than `runtime.result_freshness_s` stops new `SET`s, cancels workers, attempts `DISABLE`. | Preserve logs; use `single` until understood. |
| Camera open/read failure or malformed frame | Fails before serial open at startup; otherwise stops and cleans up. Watchdog detaches if traffic stops. | Re-verify camera path and cable. Do not arm. |
| Model or CUDA failure | Fails before `ENABLE` at startup; otherwise stops and cleans up. No silent CPU fallback. | Diagnose with simulated actuation. |
| Serial port absent | Fails closed; no auto-discovery, no `ENABLE`. | Re-verify board identity and `dialout` access. |
| `HELLO` timeout | Closes without `ENABLE`. | Check cable and board with servos disconnected. |
| Lost `ENABLE`/`SET` acknowledgement | Keeps last confirmed coordinates, marks position uncertain, blocks further motion, runs cleanup. | Treat position as uncertain. |
| Malformed firmware response | Faults the session; recovery requires a new handshake. | Stop. Board may stay enabled until the watchdog fires. |
| Watchdog expiry (`NOT_ENABLED`) | Firmware detaches once and enters `EXPIRED`; host reports a hard failure. | Stop, preserve logs, diagnose timing or transport. |
| USB disconnect | Source or transport failure stops the loop; watchdog detaches. | Inspect before reconnecting. |
| Servo power loss, ground loss, reversed polarity | Not detectable by software. | Remove power; meter-check before restoring. |
| Mechanical binding or saturation | Angles are bounded, but torque and binding are not sensed. | Remove power; fix mechanics. Do not widen limits. |
| Oscillation | Gain, deadband, max step, and rate are bounded and configurable. | Stop; tune from recorded telemetry. |
| False or missed detection | Selector filters by class and confirmation; motion stays bounded. A wallet in poor light produced intermittent 0.50–0.77 confidence false positives. | Evaluate on representative media before changing thresholds. |
