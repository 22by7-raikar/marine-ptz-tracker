from marine_ptz.actuators import SimulatedPTZActuator
from marine_ptz.types import AngleLimits, PTZCommand


def test_simulated_actuator_updates_and_clamps_state() -> None:
    actuator = SimulatedPTZActuator(
        AngleLimits(20.0, 160.0),
        AngleLimits(35.0, 145.0),
        initial_pan_deg=90.0,
        initial_tilt_deg=90.0,
    )

    actuator.apply(PTZCommand(pan_deg=180.0, tilt_deg=30.0, sequence=7))

    assert actuator.pan_deg == 160.0
    assert actuator.tilt_deg == 35.0
    assert actuator.last_sequence == 7
