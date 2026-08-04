# Presentation outline

This 12-slide outline keeps measured, observed, software-tested, and still
unmeasured claims visibly separate. Cross-check every artifact against the
submission checklist before presenting it.

## 1. Challenge and requirements

- **Key points:** Track a small marine target; laptop performs perception/control; Uno is the narrow servo boundary; safety and evidence gates precede physical claims.
- **Visual:** One-sentence system goal beside a staged validation ladder.
- **Evidence/source:** [README](../README.md), [acceptance criteria](acceptance_criteria.md).
- **Speaker note:** Lead with the working controlled demonstration, then state the non-production limitations.

## 2. Hardware, BOM, and budget

- **Key points:** Fixed-FOV USB camera, two-axis servo rig, Uno, external 5 V supply; starter-kit parts and tugboat are separately scoped; priced rig planning subtotal is about $79.58.
- **Visual:** BOM categories rather than an unverified wiring photo.
- **Evidence/source:** [BOM](bom.md), [wiring](wiring.md).
- **Speaker note:** State that tax/shipping and unpriced reusable/test items are excluded; the tugboat is a test fixture.

## 3. System architecture

- **Key points:** Camera/media to OpenCV to YOLO to selection to bounded control; simulated actuator is default; Arduino requires explicit arm/port; failure path is bounded cleanup.
- **Visual:** Mermaid architecture diagram.
- **Evidence/source:** [architecture](architecture.md).
- **Speaker note:** Point out that the diagram is a software design, not a physical validation result.

## 4. Perception pipeline

- **Key points:** OpenCV and Ultralytics are optional/lazy; the frozen class is `marine_target`; full-frame pixels drive inference/control; automatic digital zoom is display/output only.
- **Visual:** One reviewed annotated frame from the final presentation copy.
- **Evidence/source:** [README](../README.md), [replay methodology](replay.md), [BOM zoom scope](bom.md).
- **Speaker note:** Do not call boxes/crops optical zoom.

## 5. Target selection and controller

- **Key points:** Deterministic selection ordering; pixel-center error to bounded absolute pan/tilt; deadband/gain/max-step/limits/lost-target behavior are configured; simulation validates policy.
- **Visual:** Synthetic telemetry excerpt or controller diagram.
- **Evidence/source:** [architecture](architecture.md), `tests/test_tracking.py`.
- **Speaker note:** Avoid presenting synthetic behavior as measured boat tracking.

## 6. Serial protocol and firmware safety

- **Key points:** Correlated handshake before enable; bounded ASCII protocol; host and firmware limits; watchdog detaches after communication loss; watchdog is not an emergency stop.
- **Visual:** `HELLO`/`READY`/`DISABLE` sequence diagram or transcript placeholder.
- **Evidence/source:** [firmware protocol](../firmware/ptz_controller/README.md), [failure modes](failure_modes.md).
- **Speaker note:** Make lost acknowledgement uncertainty and direct servo-power removal explicit.

## 7. Testing strategy

- **Key points:** Hardware-free fakes cover runtime/serial/firmware parity; CI excludes hardware/GPU; import isolation prevents accidental camera/model/serial access; physical stages have their own evidence checklist.
- **Visual:** Test pyramid with software, AVR compile, and physical-evidence layers.
- **Evidence/source:** [test plan](test_plan.md), [acceptance criteria](acceptance_criteria.md).
- **Speaker note:** Quote the final validation transcript rather than a stale hard-coded test count.

## 8. GPU and firmware benchmarks

- **Key points:** Held-out positive run: 1.7 ms preprocess, 7.4 ms GPU inference, 1.0 ms postprocess; integrated loop about 9.4 unique FPS; actuator about 10 Hz; camera capability 1920×1080 MJPEG 30 FPS; Uno: 8,120 flash bytes (25%), 715 static RAM bytes (34%).
- **Visual:** Layered timing table with a bold “component timings are not integrated latency” label.
- **Evidence/source:** [benchmarks](benchmarks.md), [test plan](test_plan.md), firmware CI.
- **Speaker note:** Do not infer end-to-end latency from the blank-frame benchmark.

## 9. Hardware calibration results

- **Key points:** Both mappings are `physical = -logical + 180`; neutral is 90/90; verified logical/physical limits are 75–105; external 5 V servo supply and common ground remain mandatory.
- **Visual:** Completed pan/tilt mapping and limit table beside the wiring evidence.
- **Evidence/source:** [demo runbook](demo_runbook.md), [evidence checklist](evidence_checklist.md).
- **Speaker note:** Do not turn working calibration into a safety-rating claim.

## 10. End-to-end demonstration results

- **Key points:** V3 tracks the tugboat; pan/tilt/loss/reacquisition and digital zoom work; hand/forearm-only scenes were rejected live; original raw video has a known timebase defect and the presentation copy is corrected.
- **Visual:** Four-panel evidence: terminal, annotated presentation copy, rig, and telemetry.
- **Evidence/source:** [demo runbook](demo_runbook.md).
- **Speaker note:** Explain startup ordering, explicit `/dev/ttyACM0` re-verification, and `--arm-hardware`.

## 11. Failure handling and limitations

- **Key points:** Fail closed on serial setup; malformed responses fault host session; cleanup is bounded; supply polarity/ground/binding remain physical hazards; watchdog is fallback only; wallet OOD false positive reached about 0.50–0.77.
- **Visual:** Highest-risk rows from the failure-mode table.
- **Evidence/source:** [failure modes](failure_modes.md), [wiring](wiring.md).
- **Speaker note:** Direct external servo-power removal is the operational emergency action.

## 12. Submission and future production work

- **Key points:** Verify ignored artifacts and defensible claims for submission; held-out negative FP rate remains unmeasured; longer soak, safety engineering, weatherization, broader data, and licensing are outside the challenge.
- **Visual:** Final submission checklist with measured/observed/tested/unmeasured labels.
- **Evidence/source:** [demo runbook](demo_runbook.md), [acceptance criteria](acceptance_criteria.md).
- **Speaker note:** Close with the exact evidence needed to move each pending category forward.
