import math

import pytest

from marine_ptz import Detection, Frame, PTZCommand


def test_detection_center() -> None:
    detection = Detection("boat", 0.9, 10.0, 20.0, 30.0, 60.0)

    assert detection.center == (20.0, 40.0)


def test_domain_values_are_constructible_without_hardware() -> None:
    frame = Frame(image=None, timestamp_s=1.0, width=1920, height=1080, sequence=4)
    command = PTZCommand(pan_deg=90.0, tilt_deg=90.0, sequence=4)

    assert frame.sequence == command.sequence


@pytest.mark.parametrize("field_name", ["confidence", "left", "top", "right", "bottom"])
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_detection_rejects_nonfinite_values(field_name: str, invalid: float) -> None:
    values: dict[str, object] = {
        "label": "boat",
        "confidence": 0.9,
        "left": 10.0,
        "top": 20.0,
        "right": 30.0,
        "bottom": 60.0,
    }
    values[field_name] = invalid

    with pytest.raises(ValueError, match=field_name):
        Detection(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("coordinates", "message"),
    [
        ((10.0, 20.0, 10.0, 60.0), "width"),
        ((30.0, 20.0, 10.0, 60.0), "width"),
        ((10.0, 20.0, 30.0, 20.0), "height"),
        ((10.0, 60.0, 30.0, 20.0), "height"),
    ],
)
def test_detection_rejects_nonpositive_area(
    coordinates: tuple[float, float, float, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Detection("boat", 0.9, *coordinates)


@pytest.mark.parametrize("label", ["", " ", "\t"])
def test_detection_rejects_empty_normalized_label(label: str) -> None:
    with pytest.raises(ValueError, match="label"):
        Detection(label, 0.9, 0.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_detection_rejects_confidence_outside_valid_range(confidence: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        Detection("boat", confidence, 0.0, 0.0, 1.0, 1.0)


def test_numpy_backed_frame_equality_hash_and_repr_ignore_image_payload() -> None:
    numpy = pytest.importorskip("numpy")
    first = Frame(numpy.array([[1, 2]]), 1.0, 2, 1, 4)
    second = Frame(numpy.array([[9, 8]]), 1.0, 2, 1, 4)

    assert first == second
    assert hash(first) == hash(second)
    assert "image=" not in repr(first)
    assert "array" not in repr(first)


def test_synthetic_image_payload_remains_accessible_but_not_value_identity() -> None:
    from marine_ptz.synthetic import SyntheticImage

    first = Frame(SyntheticImage(1), 2.0, 100, 60, 7)
    second = Frame(SyntheticImage(2), 2.0, 100, 60, 7)

    assert first.image == SyntheticImage(1)
    assert first == second
