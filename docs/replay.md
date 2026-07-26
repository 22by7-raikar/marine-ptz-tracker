# Offline replay acceptance methodology

`python -m marine_ptz.replay_cli` is a hardware-free acceptance-evidence path.
It composes the production `run_vision()` loop with the finite OpenCV source,
Ultralytics detector, `MarineTargetSelector`, existing proportional controller,
and `SimulatedPTZActuator`. It rejects camera indexes, serial paths, Arduino
arming, and any non-simulated backend. A hardware configuration needs explicit
`--replay-safe`; this makes the safe override visible rather than silently
changing an Arduino deployment configuration.

Replay only accepts an existing local image or finite video. Named Ultralytics
models may download if uncached, so use a local model path for network-free
work. This repository does not include media or model weights.

## Report contract

Schema version 1 reports a replay-sanitized invocation, basename-only input path,
allowlisted package/GPU metadata, resolved inference device, finite-media frame
count when OpenCV provides it, dimensions once a frame is processed, metrics,
and explicit threshold results. The shared benchmark writer serializes with
`allow_nan=False`, flushes and fsyncs a same-directory temporary file, then
atomically replaces the report path. Failures before that replacement preserve
an existing report. Unavailable values are JSON `null`, never `NaN` or
infinity.

Replay sanitization builds on generic secret and URL redaction. It replaces
local source, model, config, report, and annotated-output paths in either
`--flag value` or `--flag=value` form with basename-only markers such as
`<local-path:case1.mp4>`; `path:/...` inputs retain the `path:` marker without
the directory. Named models such as `yolo11n.pt` remain identifiable. No path
resolution is performed while serializing the report, and report metadata or
diagnostics do not persist raw local paths.

Definitions are deterministic:

- `selected_target_ratio = frames_with_selected_target / processed_frames`;
  it is `null` for zero processed frames.
- Normalized horizontal and vertical center errors are `(target_x - width/2) /
  (width/2)` and `(target_y - height/2) / (height/2)`, respectively.
- Absolute normalized error is the Euclidean magnitude of those two normalized
  errors. Its mean and linearly interpolated p95 are reported separately from
  per-axis means and absolute p95 values.
- Acquisition is the first selected target's source frame sequence, with
  processing time measured from replay-loop start using the injected monotonic
  timestamp carried by an immutable production-loop event.
- A lost-target episode is consecutive selected-target-free processed frames
  **after first acquisition**. Initial pre-acquisition absence is not an
  episode. A trailing episode ends at EOF or `--max-frames`.
- Command count is each successful `SimulatedPTZActuator.apply()` represented
  by a production-loop applied-command event, including hold/lost-target
  commands. Effective command rate and processing FPS are counts divided by
  elapsed processing-loop time; they are `null` when elapsed time is zero.
  Processing ends immediately on EOF, frame limit, `q`, cancellation, or a
  runtime failure—before actuator/source/writer/display cleanup—so cleanup time
  never depresses these performance metrics.
- Saturation counts each resulting simulated pan or tilt angle that equals its
  configured inclusive lower or upper limit.
- Empty timing/error samples report `null` summaries; a single sample has equal
  mean, median, p95, and p99.

No threshold is invented. With no supplied threshold, acceptance is `null` and
the tool is report-only. A threshold with an unavailable actual metric fails
with reason `metric unavailable`. Threshold failure still writes a complete
report and returns process status 3.

## Interpretation boundaries

Synthetic validation proves deterministic control policy. The blank-frame CUDA
benchmark measures model inference only. Offline replay measures the production
software path on local finite media with simulated actuation. A future
toy-boat result requires representative recorded media and reviewed thresholds.
A future physical-servo result additionally requires camera, USB, Uno, power,
mechanical-limit, direction, neutral-position, and end-to-end latency testing.

## Observer boundary and failure behavior

The production loop supplies replay only frozen start, applied-frame,
processing-end, and normal-completion snapshots. They contain scalar metadata,
immutable domain values, and tuples only—never an image, source, detector,
actuator, serial transport, callback, or mutable detection collection. An
observer cannot alter control decisions or actuator state through an event. An
observer failure is a runtime failure: bounded cleanup still runs, no completed
replay report is written, and any prior valid report remains protected by the
shared atomic writer. Failure precedence is deterministic: an original
capture, detection, selection, control, actuator, or other runtime failure is
authoritative; a processing-end observer failure and then any cleanup failure
remain inspectable secondary diagnostics. On Python 3.10 the original error is
raised with a `RuntimeSecondaryFailures` direct cause whose `observer_error`
and `cleanup_errors` attributes retain those later failures. If processing
ends normally but its observer fails, that observer failure is primary.

Signal cancellation is normally operationally expected and maps to the usual
SIGINT/SIGTERM status. An observer failure at the processing-end boundary is
stronger than an otherwise expected cancellation: cleanup still runs once, no
report is written, and the replay CLI returns runtime-failure status 2.
