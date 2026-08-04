from marine_ptz.actuators import SimulatedPTZActuator
from marine_ptz.config import TrackingConfig
from marine_ptz.interfaces import Actuator, CameraSource, Detector, PTZController, TargetSelector
from marine_ptz.synthetic import SyntheticCamera, SyntheticTargetDetector
from marine_ptz.tracking import HighestConfidenceTargetSelector, ProportionalPTZController
from marine_ptz.types import AngleLimits, Frame


def test_synthetic_components_implement_existing_protocols() -> None:
    camera = SyntheticCamera(100, 60, clock=lambda: 12.5)
    detector = SyntheticTargetDetector(travel_frames=4)
    selector = HighestConfidenceTargetSelector()
    controller = ProportionalPTZController(
        TrackingConfig(0.0, 1.0, 1.0, 10.0, "hold"),
        AngleLimits(0.0, 180.0),
        AngleLimits(0.0, 180.0),
        initial_pan_deg=90.0,
        initial_tilt_deg=90.0,
    )
    actuator = SimulatedPTZActuator(
        AngleLimits(0.0, 180.0),
        AngleLimits(0.0, 180.0),
        initial_pan_deg=90.0,
        initial_tilt_deg=90.0,
    )

    assert isinstance(camera, CameraSource)
    assert isinstance(detector, Detector)
    assert isinstance(selector, TargetSelector)
    assert isinstance(controller, PTZController)
    assert isinstance(actuator, Actuator)


def test_synthetic_frames_and_detections_are_deterministic() -> None:
    timestamps = iter((12.5, 13.0))
    camera = SyntheticCamera(100, 60, clock=lambda: next(timestamps))
    detector = SyntheticTargetDetector(
        box_width_ratio=0.2,
        box_height_ratio=0.2,
        travel_frames=4,
    )

    first = camera.read()
    second = camera.read()

    assert first is not None
    assert second is not None
    assert first.timestamp_s == 12.5
    assert second.timestamp_s == 13.0
    assert first.sequence == 0
    assert second.sequence == 1
    assert detector.detect(first) != detector.detect(second)


def test_synthetic_bounding_boxes_remain_inside_frame_over_full_cycle() -> None:
    detector = SyntheticTargetDetector(
        box_width_ratio=0.2,
        box_height_ratio=0.3,
        travel_frames=7,
    )

    for sequence in range(12):
        frame = Frame(image=None, timestamp_s=0.0, width=100, height=60, sequence=sequence)
        detection = detector.detect(frame)[0]
        assert 0.0 <= detection.left <= detection.right <= frame.width
        assert 0.0 <= detection.top <= detection.bottom <= frame.height
