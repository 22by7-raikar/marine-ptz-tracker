from __future__ import annotations

from dataclasses import replace

import pytest

from marine_ptz.config import load_config
from marine_ptz.evaluation import (
    ClosedLoopMetricsCollector,
    ObservationSample,
    SettlingEvent,
    make_control_sample,
    make_observation_sample,
)
from marine_ptz.tracking import TargetSelectionStatus
from marine_ptz.types import Detection, Frame, PTZCommand, Target


def frame(sequence: int, timestamp_s: float | None = None) -> Frame:
    return Frame(None, float(sequence) if timestamp_s is None else timestamp_s, 100, 100, sequence)


def status(
    state: str,
    *,
    locked_id: int | None = None,
    hits: int = 0,
    confidence: float | None = None,
    unsupported_age_s: float | None = None,
) -> TargetSelectionStatus:
    return TargetSelectionStatus(
        tracker="botsort",
        active_track_count=0 if locked_id is None else 1,
        locked_track_id=locked_id,
        state=state,
        confirmation_hits=hits,
        selected_confidence=confidence,
        unsupported_age_s=unsupported_age_s,
    )


def detection(
    center_x: float,
    center_y: float,
    *,
    track_id: int | None = 7,
) -> Detection:
    return Detection(
        "marine_target",
        0.9,
        center_x - 5,
        center_y - 5,
        center_x + 5,
        center_y + 5,
        track_id,
    )


def observation(
    sequence: int,
    center_x: float | None,
    center_y: float | None,
    selection_status: TargetSelectionStatus,
    *,
    timestamp_s: float | None = None,
    track_id: int | None = 7,
) -> ObservationSample:
    current_frame = frame(sequence, timestamp_s)
    boxes = (
        ()
        if center_x is None or center_y is None
        else (detection(center_x, center_y, track_id=track_id),)
    )
    target = (
        None
        if not boxes or selection_status.state != "locked"
        else Target(boxes[0], current_frame.sequence)
    )
    return make_observation_sample(
        current_frame,
        boxes,
        target,
        selection_status,
        ("marine_target",),
    )


def test_exact_full_frame_centering_error_formulas() -> None:
    sample = observation(
        4,
        75,
        25,
        status("locked", locked_id=7, hits=2, confidence=0.9),
    )

    assert sample.center_x_px == 75
    assert sample.center_y_px == 25
    assert sample.error_x_px == 25
    assert sample.error_y_px == -25
    assert sample.normalized_error_x == pytest.approx(0.5)
    assert sample.normalized_error_y == pytest.approx(-0.5)
    assert sample.radial_normalized_error == pytest.approx(2**0.5 / 2)


def test_empty_aggregation_has_null_percentiles_and_unavailable_settling() -> None:
    report = ClosedLoopMetricsCollector().to_dict()

    assert report["target_availability"]["detected_frame_ratio"] is None  # type: ignore[index]
    assert report["centering"]["radial"] == {  # type: ignore[index]
        "count": 0,
        "p50": None,
        "p95": None,
        "p99": None,
        "max": None,
    }
    assert report["settling"] == {  # type: ignore[index]
        "available": False,
        "events": [],
        "reason": "no settling events declared",
    }


def test_two_frame_acquisition_and_current_observation_only_centering() -> None:
    collector = ClosedLoopMetricsCollector()
    acquiring = observation(10, 70, 50, status("acquiring", hits=1), timestamp_s=1.0)
    locked = observation(
        11,
        70,
        50,
        status("locked", locked_id=7, hits=2, confidence=0.9),
        timestamp_s=1.1,
    )
    collector.on_observation(acquiring)
    collector.on_observation(locked)

    report = collector.to_dict()

    assert report["acquisition"] == {  # type: ignore[index]
        "available": True,
        "first_observation_frame": 10,
        "initial_lock_frame": 11,
        "frames_to_lock": 1,
        "milliseconds_to_lock": pytest.approx(100.0),
        "confirmation_observations": 2,
    }
    assert report["target_availability"]["selected_target_frames"] == 1  # type: ignore[index]


def test_same_id_recovery_and_expired_reacquisition_are_separate() -> None:
    collector = ClosedLoopMetricsCollector()
    samples = (
        observation(0, 60, 50, status("locked", locked_id=7, hits=2), timestamp_s=0.0),
        observation(
            1,
            None,
            None,
            status("occluded", locked_id=7, hits=2, unsupported_age_s=0.1),
            timestamp_s=0.1,
        ),
        observation(2, 60, 50, status("locked", locked_id=7, hits=2), timestamp_s=0.2),
        observation(
            3,
            None,
            None,
            status("occluded", locked_id=7, hits=2, unsupported_age_s=0.2),
            timestamp_s=0.3,
        ),
        observation(4, 70, 50, status("acquiring", hits=1), timestamp_s=0.7, track_id=8),
        observation(5, 70, 50, status("locked", locked_id=8, hits=2), timestamp_s=0.8, track_id=8),
    )
    for sample in samples:
        collector.on_observation(sample)

    report = collector.to_dict()

    assert report["reacquisition"]["same_id_recovery_count"] == 1  # type: ignore[index]
    assert report["reacquisition"]["post_expiration_reacquisition_count"] == 1  # type: ignore[index]
    assert report["identity"]["post_expiration_id_change_count"] == 1  # type: ignore[index]
    assert report["identity"]["protected_lock_id_change_count"] == 0  # type: ignore[index]


def test_direct_locked_id_change_is_a_protected_lock_change() -> None:
    collector = ClosedLoopMetricsCollector()
    collector.on_observation(
        observation(0, 50, 50, status("locked", locked_id=7, hits=2), track_id=7)
    )
    collector.on_observation(
        observation(1, 50, 50, status("locked", locked_id=8, hits=2), track_id=8)
    )

    identity = collector.to_dict()["identity"]

    assert identity["locked_id_change_count"] == 1  # type: ignore[index]
    assert identity["protected_lock_id_change_count"] == 1  # type: ignore[index]
    assert identity["post_expiration_id_change_count"] == 0  # type: ignore[index]


def test_missing_id_and_prediction_only_box_produce_no_center_metrics() -> None:
    sample = observation(0, 90, 90, status("acquiring", hits=0), track_id=None)
    collector = ClosedLoopMetricsCollector()
    collector.on_observation(sample)

    report = collector.to_dict()

    assert sample.target_present is False
    assert sample.radial_normalized_error is None
    assert report["identity"]["missing_id_event_count"] == 1  # type: ignore[index]
    assert report["centering"]["sample_count"] == 0  # type: ignore[index]


def test_malformed_detection_is_counted_without_inventing_an_observation() -> None:
    sample = make_observation_sample(
        frame(0),
        (object(),),
        None,
        status("none"),
        ("marine_target",),
    )

    assert sample.malformed_detection_count == 1
    assert sample.relevant_detection_count == 0
    assert sample.target_present is False
    assert sample.center_x_px is None


def test_control_flags_deadband_maximum_step_limit_hit_reversal_and_stale() -> None:
    config = load_config("configs/development.yaml")
    current_frame = frame(0)
    far_target = Target(detection(95, 50, track_id=None), 0)
    limited = make_control_sample(
        frame=current_frame,
        target=far_target,
        settings=config.tracking,
        pan_limits=config.actuator.pan_limits,
        tilt_limits=config.actuator.tilt_limits,
        neutral_pan_deg=config.actuator.initial_pan_deg,
        neutral_tilt_deg=config.actuator.initial_tilt_deg,
        previous_pan_deg=90,
        previous_tilt_deg=90,
        command=PTZCommand(93, 90, 0),
        timestamp_s=0.0,
        fresh=True,
        result_age_s=0.02,
    )
    at_limit = make_control_sample(
        frame=frame(1),
        target=Target(detection(95, 50, track_id=None), 1),
        settings=config.tracking,
        pan_limits=config.actuator.pan_limits,
        tilt_limits=config.actuator.tilt_limits,
        neutral_pan_deg=config.actuator.initial_pan_deg,
        neutral_tilt_deg=config.actuator.initial_tilt_deg,
        previous_pan_deg=config.actuator.pan_limits.maximum_deg,
        previous_tilt_deg=90,
        command=PTZCommand(config.actuator.pan_limits.maximum_deg, 90, 1),
        timestamp_s=0.1,
        fresh=True,
        result_age_s=0.03,
    )
    reverse = replace(
        at_limit,
        frame_sequence=2,
        timestamp_s=0.2,
        command=PTZCommand(90, 90, 2),
        pan_delta_deg=-3.0,
        pan_limit_hit=False,
    )
    stale = make_control_sample(
        frame=frame(3),
        target=None,
        settings=config.tracking,
        pan_limits=config.actuator.pan_limits,
        tilt_limits=config.actuator.tilt_limits,
        neutral_pan_deg=config.actuator.initial_pan_deg,
        neutral_tilt_deg=config.actuator.initial_tilt_deg,
        previous_pan_deg=90,
        previous_tilt_deg=90,
        command=None,
        timestamp_s=0.3,
        fresh=False,
        result_age_s=0.6,
    )
    centered = make_control_sample(
        frame=frame(4),
        target=Target(detection(50, 50, track_id=None), 4),
        settings=config.tracking,
        pan_limits=config.actuator.pan_limits,
        tilt_limits=config.actuator.tilt_limits,
        neutral_pan_deg=config.actuator.initial_pan_deg,
        neutral_tilt_deg=config.actuator.initial_tilt_deg,
        previous_pan_deg=90,
        previous_tilt_deg=90,
        command=PTZCommand(90, 90, 4),
        timestamp_s=0.4,
        fresh=True,
        result_age_s=0.01,
    )
    collector = ClosedLoopMetricsCollector()
    for sample in (limited, at_limit, reverse, stale, centered):
        collector.on_control(sample)

    report = collector.to_dict()
    control = report["control"]

    assert limited.pan_max_step_limited is True
    assert at_limit.pan_limit_saturated is True
    assert centered.within_deadband is True
    assert control["deadband_update_count"] == 1  # type: ignore[index]
    assert control["logical_limit_hit_count"]["pan"] == 1  # type: ignore[index]
    assert control["direction_reversal_count"]["pan"] == 1  # type: ignore[index]
    assert report["freshness"]["stale_result_count"] == 1  # type: ignore[index]
    assert report["freshness"]["result_age_at_control_ms"]["p50"] == pytest.approx(30)  # type: ignore[index]


def test_centering_aggregation_uses_only_controller_supported_observations() -> None:
    config = load_config("configs/development.yaml")
    collector = ClosedLoopMetricsCollector()
    for sequence, center_x in enumerate((50.0, 60.0, 70.0)):
        current_frame = frame(sequence, sequence / 10)
        current_detection = detection(center_x, 50, track_id=7)
        current_target = Target(current_detection, sequence)
        collector.on_observation(
            make_observation_sample(
                current_frame,
                (current_detection,),
                current_target,
                status("locked", locked_id=7, hits=2, confidence=0.9),
                ("marine_target",),
            )
        )
        collector.on_control(
            make_control_sample(
                frame=current_frame,
                target=current_target,
                settings=config.tracking,
                pan_limits=config.actuator.pan_limits,
                tilt_limits=config.actuator.tilt_limits,
                neutral_pan_deg=config.actuator.initial_pan_deg,
                neutral_tilt_deg=config.actuator.initial_tilt_deg,
                previous_pan_deg=90,
                previous_tilt_deg=90,
                command=PTZCommand(90, 90, sequence),
                timestamp_s=sequence / 10,
                fresh=sequence != 1,
                result_age_s=0.01,
            )
        )

    centering = collector.to_dict()["centering"]

    assert centering["sample_count"] == 2  # type: ignore[index]
    assert centering["horizontal"]["signed_mean"] == pytest.approx(0.2)  # type: ignore[index]
    assert centering["horizontal"]["absolute"]["p50"] == pytest.approx(0.2)  # type: ignore[index]
    assert centering["horizontal"]["absolute"]["max"] == pytest.approx(0.4)  # type: ignore[index]


def test_return_to_neutral_maximum_step_is_measured_without_changing_command() -> None:
    config = load_config("configs/development.yaml")
    settings = replace(config.tracking, lost_target_behavior="return_to_neutral")

    sample = make_control_sample(
        frame=frame(0),
        target=None,
        settings=settings,
        pan_limits=config.actuator.pan_limits,
        tilt_limits=config.actuator.tilt_limits,
        neutral_pan_deg=90,
        neutral_tilt_deg=90,
        previous_pan_deg=100,
        previous_tilt_deg=90,
        command=PTZCommand(97, 90, 0),
        timestamp_s=0.0,
        fresh=True,
        result_age_s=None,
    )

    assert sample.lost_target_action == "return_to_neutral"
    assert sample.pan_max_step_limited is True
    assert sample.command == PTZCommand(97, 90, 0)


def test_declared_settling_event_requires_tolerance_dwell() -> None:
    collector = ClosedLoopMetricsCollector((SettlingEvent("move-left", 0, 3, 0.1, 0.1),))
    samples = (
        observation(0, 75, 50, status("locked", locked_id=7), timestamp_s=0.0),
        observation(1, 54, 50, status("locked", locked_id=7), timestamp_s=0.1),
        observation(2, 53, 50, status("locked", locked_id=7), timestamp_s=0.2),
        observation(3, 52, 50, status("locked", locked_id=7), timestamp_s=0.3),
    )
    for sample in samples:
        collector.on_observation(sample)

    event = collector.to_dict()["settling"]["events"][0]  # type: ignore[index]

    assert event["available"] is True
    assert event["settled"] is True
    assert event["settling_time_ms"] == pytest.approx(100.0)
    assert event["overshoot_radial_normalized"] == 0.0
