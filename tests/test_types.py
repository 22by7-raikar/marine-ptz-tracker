from marine_ptz import Detection, Frame, PTZCommand


def test_detection_center() -> None:
    detection = Detection("boat", 0.9, 10.0, 20.0, 30.0, 60.0)

    assert detection.center == (20.0, 40.0)


def test_domain_values_are_constructible_without_hardware() -> None:
    frame = Frame(image=None, timestamp_s=1.0, width=1920, height=1080, sequence=4)
    command = PTZCommand(pan_deg=90.0, tilt_deg=90.0, sequence=4)

    assert frame.sequence == command.sequence
