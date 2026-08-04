# Submission checklist

Complete this checklist against the actual checkout and ignored artifacts. It
does not replace physical judgment or make the rig safety-rated.

## Hardware and configuration

- [ ] Confirm camera, Uno, two SG90 servos, pan/tilt assembly, and fasteners are
  secure and unobstructed.
- [ ] Confirm servos use the regulated external 5 V supply, never Arduino/USB
  5 V, and confirm the supply and Arduino share ground.
- [ ] Verify the exact camera by-id path and confirm `/dev/ttyACM0` still names
  the intended Uno; stop on ambiguity. No serial auto-discovery exists.
- [ ] Confirm logical/physical limits are 75–105, both mappings are
  `physical = -logical + 180`, neutral is 90/90, watchdog is 1000 ms, command
  rate is 10 Hz, and confidence remains 0.60.
- [ ] Keep immediate external-servo-power removal available.

## Software and demonstration

- [ ] Verify the frozen model exists at
  `runs/detect/runs/tugboat/yolo11n-marine-target-v3/weights/best.pt` and the
  class is exactly `marine_target`.
- [ ] Run the hardware-independent validation commands and preserve their
  terminal output.
- [ ] Run the simulated digital-zoom command before arming hardware.
- [ ] For physical operation, provide `arduino_serial`, the explicit serial
  port, and `--arm-hardware`; verify initialization and first hold precede
  tracking motion.
- [ ] Record the ten-step sequence in `docs/demo_runbook.md`: lost, lock, pan,
  tilt, distance/zoom, occlusion, removal/zoom-out, reacquisition, hand/forearm
  negative, and `q`/Escape shutdown.
- [ ] Confirm bounded `DISABLE`/close output, then remove external servo power.

## Artifact and claim review

- [ ] Preserve the original raw video
  `artifacts/demo/marine-ptz-v3-digital-zoom.mp4` and note its known timebase
  defect: 27.534 seconds of playback represents about 87.8 seconds wall time.
- [ ] Verify the presentation copy
  `artifacts/demo/marine-ptz-v3-digital-zoom-realtime.mp4` has the intended
  real-time duration and playable media metadata.
- [ ] Verify `artifacts/evaluation/tugboat-v3-test-schema-v2.json` is strict JSON, refers
  to the frozen model/data, contains 36 positive and zero reviewed-negative
  test images, and represents unmeasured negative metrics as `null`.
- [ ] Keep the superseded schema-v1 `tugboat-v3-test.json` out of the release
  manifest and publication package; its zero-valued negative fields do not
  represent measured negative performance.
- [ ] Review every report, video, screenshot, and transcript for credentials,
  usernames, private paths, unrelated content, and correct artifact identity.
- [ ] Verify all three final container evidence hashes from within
  `artifacts/container-replay`, confirm the 2,635-frame/87.834-second video
  metadata, and keep those ignored binaries/logs outside Git.
- [ ] State positive held-out metrics exactly: precision 1.000000, recall
  0.936044, mAP50 0.962213, and mAP50–95 0.802989 at confidence 0.60.
- [ ] Keep camera capability (1920×1080 MJPEG 30 FPS), held-out GPU inference
  (7.4 ms/image), integrated unique-frame loop (~9.4 FPS), command rate
  (~10 Hz), and encoded output FPS separate.
- [ ] State that held-out negative false-positive rate was not measured. Keep
  live hand/forearm rejection qualitative and disclose the wallet OOD false
  positive (~0.50–0.77 confidence).
- [ ] Describe zoom as post-control digital zoom, never optical zoom or added
  detector detail.
- [ ] Do not claim open-world robustness, marine deployment readiness,
  weatherization, emergency-stop certification, or safety-rated operation.
