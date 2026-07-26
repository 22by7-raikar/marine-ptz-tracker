# Presentation outline

This 12-slide outline keeps software evidence, toolchain evidence, and pending
physical evidence visibly separate. Replace every `<fill>` item only with
captured evidence from the checklist.

## 1. Challenge and requirements

- **Key points:** Track a small marine target; laptop performs perception/control; Uno is the narrow servo boundary; safety and evidence gates precede physical claims.
- **Visual:** One-sentence system goal beside a staged validation ladder.
- **Evidence/source:** [README](../README.md), [acceptance criteria](acceptance_criteria.md).
- **Speaker note:** Lead with the distinction between implemented software and pending physical proof.

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

- **Key points:** OpenCV and Ultralytics are optional/lazy; configured target classes are normalized; local model path avoids downloads; full fixed-FOV image is used.
- **Visual:** One annotated replay frame placeholder labeled `<fill after replay>`.
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
- **Speaker note:** Current count is **540** hardware/GPU-independent tests; update only after a new validation run.

## 8. GPU and firmware benchmarks

- **Key points:** Blank 640×640 model-only warm inference: mean 6.65 ms, median 6.56 ms, p95 6.96 ms, about 150 FPS; Uno: 8,120 flash bytes (25%), 715 static RAM bytes (34%); Python 3.10/3.11 and remote Uno compile passed.
- **Visual:** Two-column benchmark table with a bold “not camera-to-servo” label.
- **Evidence/source:** [benchmarks](benchmarks.md), [test plan](test_plan.md), firmware CI.
- **Speaker note:** Do not infer end-to-end latency from the blank-frame benchmark.

## 9. Hardware calibration results (placeholder)

- **Key points:** Servo identity, pin, direction, neutral, safe min/max, clearance, current, and supply observations remain `<fill>`; camera formats/FPS remain `<fill>`.
- **Visual:** Empty-but-labeled pan/tilt calibration table.
- **Evidence/source:** [demo runbook](demo_runbook.md), [evidence checklist](evidence_checklist.md).
- **Speaker note:** Say “pending,” never “passed,” until captured and reviewed.

## 10. End-to-end demonstration results (placeholder)

- **Key points:** Requires reviewed camera/simulated run before arming; final video must show terminal, camera/annotation, telemetry, safety observer, lost target, and shutdown; results are `<fill>`.
- **Visual:** Four-panel layout placeholder: terminal, camera, rig, telemetry.
- **Evidence/source:** [demo runbook](demo_runbook.md).
- **Speaker note:** Explain startup ordering and explicit `/dev/serial/by-id/...` identity.

## 11. Failure handling and limitations

- **Key points:** Fail closed on serial setup; malformed responses fault host session; cleanup is bounded; supply polarity/ground/binding remain physical hazards; watchdog is fallback only.
- **Visual:** Highest-risk rows from the failure-mode table.
- **Evidence/source:** [failure modes](failure_modes.md), [wiring](wiring.md).
- **Speaker note:** Direct external servo-power removal is the operational emergency action.

## 12. Next production steps

- **Key points:** Complete staged evidence; set reviewed media/hardware thresholds; validate camera-to-motion timing; complete soak tests; decide publication/licensing only after evidence review.
- **Visual:** Stage 0–9 progress checklist with only software/AVR stages marked complete.
- **Evidence/source:** [demo runbook](demo_runbook.md), [acceptance criteria](acceptance_criteria.md).
- **Speaker note:** Close with the exact evidence needed to move each pending category forward.
