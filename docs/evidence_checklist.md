# Physical evidence checklist

Do not create placeholder evidence. Capture and store this material outside
Git as each physical stage is actually performed.

- [ ] Wiring photos showing the separate servo supply and common ground.
- [ ] Measured external-supply voltage and polarity record.
- [ ] Arduino board identity and stable `/dev/serial/by-id/...` path.
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
