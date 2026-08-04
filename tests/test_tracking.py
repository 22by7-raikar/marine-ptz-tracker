from types import SimpleNamespace

import pytest

from marine_ptz.config import TrackingConfig
from marine_ptz.tracking import ProportionalPTZController
from marine_ptz.types import AngleLimits, Detection, Frame, Target

PAN_LIMITS = AngleLimits(20.0, 160.0)
TILT_LIMITS = AngleLimits(35.0, 145.0)


def settings(
    *,
    deadband_px: float = 0.0,
    proportional_gain: float = 10.0,
    max_step_deg: float = 5.0,
    lost_target_behavior: str = "hold",
) -> TrackingConfig:
    return TrackingConfig(
        deadband_px=deadband_px,
        proportional_gain=proportional_gain,
        max_step_deg=max_step_deg,
        update_rate_hz=10.0,
        lost_target_behavior=lost_target_behavior,
    )


def frame(sequence: int = 1) -> Frame:
    return Frame(image=None, timestamp_s=1.0, width=1000, height=600, sequence=sequence)


def target_at(x: float, y: float, sequence: int = 1) -> Target:
    detection = Detection("boat", 1.0, x - 10, y - 10, x + 10, y + 10)
    return Target(detection, sequence)


def controller(
    tracking: TrackingConfig | None = None,
    *,
    pan: float = 90.0,
    tilt: float = 90.0,
    pan_limits: AngleLimits = PAN_LIMITS,
    tilt_limits: AngleLimits = TILT_LIMITS,
) -> ProportionalPTZController:
    return ProportionalPTZController(
        tracking or settings(),
        pan_limits,
        tilt_limits,
        initial_pan_deg=pan,
        initial_tilt_deg=tilt,
    )


def test_centered_target_causes_no_movement() -> None:
    command = controller().command_for(frame(), target_at(500.0, 300.0))

    assert command.pan_deg == 90.0
    assert command.tilt_deg == 90.0


def test_horizontal_and_vertical_correction_directions() -> None:
    positive = controller().command_for(frame(), target_at(750.0, 450.0))
    negative = controller().command_for(frame(), target_at(250.0, 150.0))

    assert positive.pan_deg > 90.0
    assert positive.tilt_deg > 90.0
    assert negative.pan_deg < 90.0
    assert negative.tilt_deg < 90.0


def test_errors_inside_deadband_cause_no_movement() -> None:
    tracking = settings(deadband_px=25.0)

    command = controller(tracking).command_for(frame(), target_at(520.0, 280.0))

    assert command.pan_deg == 90.0
    assert command.tilt_deg == 90.0


def test_maximum_step_and_servo_limits_are_clamped() -> None:
    tracking = settings(proportional_gain=100.0, max_step_deg=4.0)
    step_limited = controller(tracking).command_for(frame(), target_at(1000.0, 600.0))
    servo_limited = controller(
        tracking,
        pan=99.0,
        tilt=99.0,
        pan_limits=AngleLimits(80.0, 100.0),
        tilt_limits=AngleLimits(80.0, 100.0),
    ).command_for(frame(), target_at(1000.0, 600.0))

    assert step_limited.pan_deg == 94.0
    assert step_limited.tilt_deg == 94.0
    assert servo_limited.pan_deg == 100.0
    assert servo_limited.tilt_deg == 100.0


@pytest.mark.parametrize(
    ("behavior", "expected"),
    [("hold", 95.0), ("return_to_neutral", 90.0)],
)
def test_lost_target_behavior(behavior: str, expected: float) -> None:
    ptz = controller(settings(lost_target_behavior=behavior))
    ptz.command_for(frame(), target_at(1000.0, 600.0))

    command = ptz.command_for_target(frame(sequence=2), None)

    assert command.pan_deg == expected
    assert command.tilt_deg == expected


@pytest.mark.parametrize("behavior", ["hold", "return_to_neutral"])
@pytest.mark.parametrize(
    ("target_x", "target_y", "expected"),
    [(0.0, 0.0, 80.0), (1000.0, 600.0, 100.0)],
)
def test_lost_target_at_servo_limits_remains_bounded(
    behavior: str,
    target_x: float,
    target_y: float,
    expected: float,
) -> None:
    ptz = controller(
        settings(lost_target_behavior=behavior),
        pan_limits=AngleLimits(80.0, 100.0),
        tilt_limits=AngleLimits(80.0, 100.0),
    )
    for sequence in range(1, 5):
        ptz.command_for(frame(sequence), target_at(target_x, target_y, sequence))

    command = ptz.command_for_target(frame(sequence=5), None)

    assert 80.0 <= command.pan_deg <= 100.0
    assert 80.0 <= command.tilt_deg <= 100.0
    if behavior == "hold":
        assert command.pan_deg == expected
        assert command.tilt_deg == expected
    elif expected == 80.0:
        assert command.pan_deg == command.tilt_deg == 85.0
    else:
        assert command.pan_deg == command.tilt_deg == 95.0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_malformed_duck_typed_target_cannot_move_hold_controller(invalid: float) -> None:
    ptz = controller(settings(lost_target_behavior="hold"), pan=95.0, tilt=85.0)
    malformed = SimpleNamespace(
        detection=SimpleNamespace(center=(invalid, 300.0)),
        frame_sequence=1,
    )

    command = ptz.command_for(frame(), malformed)  # type: ignore[arg-type]

    assert command.pan_deg == 95.0
    assert command.tilt_deg == 85.0


def test_malformed_target_uses_return_to_neutral_policy() -> None:
    ptz = controller(
        settings(max_step_deg=2.0, lost_target_behavior="return_to_neutral"),
        pan=95.0,
        tilt=85.0,
    )
    ptz.command_for(frame(), target_at(1000.0, 0.0))
    malformed = SimpleNamespace(
        detection=SimpleNamespace(center=("not-a-number", None)),
        frame_sequence=2,
    )

    command = ptz.command_for(frame(sequence=2), malformed)  # type: ignore[arg-type]

    assert command.pan_deg == 95.0
    assert command.tilt_deg == 85.0
