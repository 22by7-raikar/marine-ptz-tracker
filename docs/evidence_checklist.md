# Physical evidence checklist

Do not create placeholder evidence. Capture and store this material outside
Git as each physical stage is actually performed.

- [ ] Wiring photos showing the separate servo supply and common ground.
- [ ] Measured external-supply voltage and polarity record.
- [ ] Arduino board identity and current `/dev/ttyACM0` fallback rechecked after
  the latest USB change; record that no stable by-id link is exposed.
- [ ] Timestamped `HELLO`/`READY`/`DISABLE` serial-handshake transcript.
- [ ] Completed one-servo and two-axis calibration tables.
- [ ] Camera formats, chosen resolution/FPS, and sample capture evidence.
- [ ] Representative toy-boat detection images or recorded media.
- [ ] Simulated-actuator live-camera tracking transcript/video.
- [ ] Explicitly armed physical-tracking video with visible safety observer.
- [ ] Lost-target behavior record.
- [ ] Controlled watchdog disconnect/reattachment test record.
- [ ] Sanitized offline replay JSON report and annotated output, if produced.
- [ ] Final benchmark table, with model-only and camera/end-to-end timing kept separate.
- [ ] Frozen V3 schema-v2 evaluation JSON
  (`tugboat-v3-test-schema-v2.json`) showing 36 positives, zero reviewed
  negatives, and `null` unmeasured negative metrics; exclude the superseded
  schema-v1 report from publication.
- [ ] Original raw and corrected real-time presentation videos identified and
  reviewed, including the original 27.534-second/87.8-second timebase defect.
- [ ] Final container-replay video, log, and image-identity record reviewed;
  run `cd artifacts/container-replay && sha256sum -c SHA256SUMS` and confirm all
  three portable-basename entries pass.
- [ ] Completed [submission checklist](submission_checklist.md).
