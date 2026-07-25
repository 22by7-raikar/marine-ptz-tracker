from types import SimpleNamespace

from marine_ptz.tracking import MarineTargetSelector
from marine_ptz.types import Detection, Frame


def frame() -> Frame:
    return Frame(None, 0.0, 100, 100, 7)


def test_selector_uses_confidence_before_other_factors() -> None:
    selector = MarineTargetSelector(("boat",))
    low = Detection("boat", 0.8, 0, 0, 80, 80)
    high = Detection("boat", 0.9, 0, 0, 10, 10)

    assert selector.select(frame(), (low, high)).detection == high  # type: ignore[union-attr]


def test_selector_uses_area_then_center_distance_then_coordinates() -> None:
    selector = MarineTargetSelector(("boat",))
    small = Detection("boat", 0.9, 40, 40, 50, 50)
    large_far = Detection("boat", 0.9, 0, 0, 20, 20)
    large_center = Detection("boat", 0.9, 40, 40, 60, 60)
    coordinate_later = Detection("boat", 0.9, 50, 40, 70, 60)

    selected = selector.select(
        frame(),
        (small, large_far, coordinate_later, large_center),
    )

    assert selected is not None
    assert selected.detection == large_center


def test_selector_coordinate_tie_break_is_stable() -> None:
    selector = MarineTargetSelector(("boat",))
    left = Detection("boat", 0.9, 30, 40, 50, 60)
    right = Detection("boat", 0.9, 50, 40, 70, 60)

    selected = selector.select(frame(), (right, left))

    assert selected is not None
    assert selected.detection == left


def test_selector_returns_none_when_requested_class_is_absent() -> None:
    selector = MarineTargetSelector(("boat",))

    assert selector.select(frame(), (Detection("person", 0.99, 0, 0, 90, 90),)) is None


def test_selector_excludes_malformed_duck_typed_detector_output() -> None:
    selector = MarineTargetSelector(("boat",))
    malformed = SimpleNamespace(
        label="boat",
        confidence=float("nan"),
        left=0.0,
        top=0.0,
        right=10.0,
        bottom=10.0,
    )

    assert selector.select(frame(), (malformed,)) is None  # type: ignore[arg-type]


def test_selector_handles_case_and_duplicate_ties_deterministically() -> None:
    selector = MarineTargetSelector(("BoAt",))
    first = Detection("boat", 0.9, 30, 40, 50, 60)
    duplicate = Detection("BOAT", 0.9, 30, 40, 50, 60)

    selected = selector.select(frame(), (duplicate, first))

    assert selected is not None
    assert selected.detection is duplicate
