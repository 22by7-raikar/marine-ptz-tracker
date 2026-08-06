from types import SimpleNamespace

import pytest

from marine_ptz.tracking import MarineTargetSelector
from marine_ptz.types import Detection, Frame


def frame(sequence: int = 7) -> Frame:
    return Frame(None, float(sequence), 100, 100, sequence)


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def tracked(track_id: int, confidence: float = 0.90, *, left: float = 40.0) -> Detection:
    return Detection("marine_target", confidence, left, 40.0, left + 20.0, 60.0, track_id)


def tracked_selector(clock: Clock) -> MarineTargetSelector:
    return MarineTargetSelector(
        ("marine_target",),
        tracker_backend="botsort",
        acquisition_confidence=0.60,
        support_confidence=0.20,
        minimum_confirmation_hits=2,
        maximum_unsupported_age_s=0.30,
        monotonic=clock,
    )


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


def test_tracked_selector_requires_two_consecutive_observations_and_locks() -> None:
    clock = Clock()
    selector = tracked_selector(clock)

    assert selector.select(frame(0), (tracked(7),)) is None
    assert selector.status.state == "acquiring"
    assert selector.status.confirmation_hits == 1

    clock.value = 0.03
    selected = selector.select(frame(1), (tracked(7),))

    assert selected is not None
    assert selected.detection.track_id == 7
    assert selector.status.locked_track_id == 7
    assert selector.status.state == "locked"


def test_changed_track_id_resets_acquisition_and_tie_breaks_by_id() -> None:
    clock = Clock()
    selector = tracked_selector(clock)

    assert selector.select(frame(0), (tracked(9),)) is None
    clock.value = 0.03
    assert selector.select(frame(1), (tracked(8),)) is None
    assert selector.status.confirmation_hits == 1
    clock.value = 0.06
    selected = selector.select(frame(2), (tracked(9), tracked(8)))

    assert selected is not None
    assert selected.detection.track_id == 8


def test_lower_confidence_unassociated_track_cannot_initialize_but_supports_lock() -> None:
    clock = Clock()
    selector = tracked_selector(clock)

    assert selector.select(frame(0), (tracked(4, 0.30),)) is None
    assert selector.status.state == "none"
    clock.value = 0.03
    assert selector.select(frame(1), (tracked(4, 0.90),)) is None
    clock.value = 0.06
    assert selector.select(frame(2), (tracked(4, 0.90),)) is not None
    clock.value = 0.09
    selected = selector.select(frame(3), (tracked(4, 0.30),))

    assert selected is not None
    assert selected.detection.confidence == 0.30


def test_other_track_cannot_steal_lock_and_all_active_tracks_are_counted() -> None:
    clock = Clock()
    selector = tracked_selector(clock)
    selector.select(frame(0), (tracked(1),))
    clock.value = 0.03
    selector.select(frame(1), (tracked(1),))
    clock.value = 0.06

    selected = selector.select(
        frame(2),
        (tracked(2, 0.99, left=10.0), tracked(1, 0.61, left=70.0)),
    )

    assert selected is not None
    assert selected.detection.track_id == 1
    assert selector.status.active_track_count == 2


def test_short_occlusion_returns_no_prediction_then_same_id_resumes() -> None:
    clock = Clock()
    selector = tracked_selector(clock)
    selector.select(frame(0), (tracked(5),))
    clock.value = 0.03
    selector.select(frame(1), (tracked(5),))
    clock.value = 0.20

    assert selector.select(frame(2), ()) is None
    assert selector.status.state == "occluded"
    assert selector.status.locked_track_id == 5
    assert selector.status.unsupported_age_s == pytest.approx(0.17)

    clock.value = 0.25
    resumed = selector.select(frame(3), (tracked(5, 0.25),))

    assert resumed is not None
    assert resumed.detection.track_id == 5
    assert selector.status.state == "locked"


def test_expired_lock_requires_normal_reacquisition() -> None:
    clock = Clock()
    selector = tracked_selector(clock)
    selector.select(frame(0), (tracked(5),))
    clock.value = 0.03
    selector.select(frame(1), (tracked(5),))
    clock.value = 0.34

    assert selector.select(frame(2), (tracked(8),)) is None
    assert selector.status.locked_track_id is None
    assert selector.status.state == "acquiring"
    assert selector.status.confirmation_hits == 1
    clock.value = 0.37
    reacquired = selector.select(frame(3), (tracked(8),))

    assert reacquired is not None
    assert reacquired.detection.track_id == 8


def test_repeated_inspection_of_same_frame_does_not_double_count_confirmation() -> None:
    clock = Clock()
    selector = tracked_selector(clock)
    observation = (tracked(12),)

    assert selector.select(frame(0), observation) is None
    assert selector.select(frame(0), observation) is None
    assert selector.status.confirmation_hits == 1
    clock.value = 0.03
    assert selector.select(frame(1), observation) is not None


def test_repeated_stale_result_cannot_keep_locked_observation_alive() -> None:
    clock = Clock()
    selector = tracked_selector(clock)
    selector.select(frame(0), (tracked(12),))
    clock.value = 0.03
    selector.select(frame(1), (tracked(12),))
    clock.value = 0.34

    assert selector.select(frame(1), (tracked(12),)) is None
    assert selector.status.locked_track_id is None
    assert selector.status.state == "none"


def test_selector_reset_discards_lock_and_requires_confirmation_again() -> None:
    clock = Clock()
    selector = tracked_selector(clock)
    selector.select(frame(0), (tracked(3),))
    clock.value = 0.03
    assert selector.select(frame(1), (tracked(3),)) is not None

    selector.reset()
    clock.value = 0.06

    assert selector.select(frame(0), (tracked(3),)) is None
    assert selector.status.state == "acquiring"
