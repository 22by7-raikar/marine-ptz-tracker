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

Acceptance replay deliberately keeps the verified `single` runtime because its
observer contract records one applied command per processed frame. Use
`tools/benchmark_runtime.py` for a hardware-free, same-video comparison with
the opt-in concurrent topology; do not reinterpret that performance report as
an acceptance or physical-actuation report.

## Report contract

Schema version 3 reports a replay-sanitized invocation, basename-only input path,
allowlisted package/GPU metadata, resolved inference device, finite-media frame
count when OpenCV provides it, dimensions once a frame is processed, replay
metrics, closed-loop observation/control metrics, and explicit threshold results.
Schema-version-2 reports remain readable through `normalize_replay_report()` and
are explicitly marked as lacking closed-loop data; unsupported versions fail
closed. The shared benchmark writer serializes with
`allow_nan=False`, flushes and fsyncs a same-directory temporary file, then
atomically replaces the report path. Failures before that replacement preserve
an existing report. Unavailable values are JSON `null`, never `NaN` or
infinity.

Replay sanitization builds on generic secret and URL redaction. The field type
takes precedence over value syntax: path-typed `--source`, `--report`,
`--config`, and `--annotated-output` values in either `--flag value` or
`--flag=value` form always become basename-only `<local-path:basename>`
markers, even when they look like URLs. Model fields preserve safe named
identifiers such as `yolo11n.pt`, use `<local-path:basename>` for filesystem
paths, and use `<remote-resource:basename>` for remote URLs. Marker derivation
discards URL authority, credentials, query, and fragment. Empty, malformed, or
encoded ambiguous basenames use the matching `...:redacted` marker. Reports are
designed to be shareable, but must still be reviewed before publication. No
path resolution is performed while serializing the report, and report metadata
or diagnostics do not persist raw local paths.

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

The schema-v3 `closed_loop` section uses two explicit populations. An
observation is one newly consumed inference result; a control decision is one
scheduled controller opportunity. Concurrent rendering can therefore produce
more observations than the 10 Hz control schedule. Centering statistics include
only control decisions supplied with a current detector-supported locked
observation. Acquisition, occlusion, prediction-only results, missing/malformed
track IDs, stale results, and lost targets retain their state and identity
metadata but have no invented center or controller-target measurement.

Selector-state frame counts and ratios are the current evidence-safe summaries.
Do not quote `selector_state_seconds` as evaluation evidence until the known
replay time-base discrepancy for those durations is corrected.

For a current box center `(x, y)` and full-frame center `(cx, cy)`, the closed-loop
errors are `ex = (x - cx) / (width / 2)`, `ey = (y - cy) / (height / 2)`, and
`er = sqrt(ex^2 + ey^2)`. Signed means are reported per axis; absolute axis and
radial distributions report p50, p95, p99, and maximum. The coordinates are the
original full-frame coordinates used by PTZ control, never annotated or digitally
zoomed display coordinates.

Initial acquisition runs from the first relevant current observation to the
first locked current observation. Reacquisition events separate a short
same-locked-ID recovery from a post-expiration reacquisition. A direct locked-ID
change without an intervening unlocked/expired boundary is counted separately as
a protected-lock ID change; a different ID after expiration is not mislabeled as
that error. Missing IDs and malformed detection values have separate counters.

Control metrics describe existing commands: update rate, absolute command-delta
distributions, deadband ratio over controller-supported updates, pre-limit
maximum-step and logical-limit saturation, resulting logical endpoint hits,
direction reversals, and lost-target actions. Concurrent result age is sampled at
the actual control decision. Single-mode result age is `null` because that runtime
does not retain a separate decision timestamp; capture-to-command latency is not
substituted. Stale decisions have no controller-supported target or command.

Settling and overshoot are unavailable unless target-motion events are declared
explicitly. Add repeatable annotations as:

```text
--settling-event NAME:START_FRAME:END_FRAME:TOLERANCE:DWELL_SECONDS
```

Within each declared inclusive frame window, settling time starts at the first
observed window timestamp and ends at the first current-observation sample whose
radial normalized error begins a continuous in-tolerance dwell. Missing or
prediction-only observations break the dwell. Overshoot is the positive increase
from the first observed radial error to the largest observed radial error in the
window. These are evaluation annotations, not inferred target-step events.

No threshold is invented. With no supplied threshold, acceptance is `null` and
the tool is report-only. A threshold with an unavailable actual metric fails
with reason `metric unavailable`. Threshold failure still writes a complete
report and returns process status 3.

## Interpretation boundaries

Synthetic validation proves deterministic control policy. The blank-frame CUDA
benchmark measures model inference only. Offline replay measures the production
software path on local finite media with simulated actuation. A replay
acceptance claim requires representative recorded media and reviewed
thresholds; the measured final container replay is performance evidence, not a
thresholded acceptance result. A physical-servo claim additionally requires
camera, USB, Uno, power, mechanical-limit, direction, neutral-position, and
end-to-end evidence.

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
