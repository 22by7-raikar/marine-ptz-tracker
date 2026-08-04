from __future__ import annotations

import pytest

from marine_ptz.digital_zoom import DigitalZoomController
from marine_ptz.types import Detection


def detection(
    left: float = 40.0,
    top: float = 40.0,
    right: float = 60.0,
    bottom: float = 60.0,
) -> Detection:
    return Detection("marine_target", 0.9, left, top, right, bottom)


def test_disabled_zoom_is_an_identity_transform() -> None:
    transform = DigitalZoomController(enabled=False).update(
        100,
        80,
        detection(0, 0, 10, 10),
    )

    assert transform.is_identity
    assert transform.zoom_factor == 1.0
    assert transform.transform_point(23, 17) == pytest.approx((23, 17))


@pytest.mark.parametrize(
    "target",
    (
        detection(0, 0, 10, 10),
        detection(45, 0, 55, 10),
        detection(90, 0, 100, 10),
        detection(0, 35, 10, 45),
        detection(90, 35, 100, 45),
        detection(0, 70, 10, 80),
        detection(45, 70, 55, 80),
        detection(90, 70, 100, 80),
    ),
)
def test_zoom_factor_and_crop_remain_bounded_at_every_frame_edge(
    target: Detection,
) -> None:
    controller = DigitalZoomController(enabled=True, max_zoom=2.0, smoothing=1.0)

    transform = controller.update(100, 80, target)

    assert transform.zoom_factor == 2.0
    assert 0 <= transform.crop_left < transform.crop_right <= 100
    assert 0 <= transform.crop_top < transform.crop_bottom <= 80
    transformed = transform.transform_box(target)
    assert transformed is not None
    assert 0 <= transformed[0] < transformed[2] <= 100
    assert 0 <= transformed[1] < transformed[3] <= 80


def test_target_fill_drives_desired_zoom_without_exceeding_maximum() -> None:
    controller = DigitalZoomController(
        enabled=True,
        max_zoom=3.0,
        smoothing=1.0,
        target_fill=0.5,
    )

    medium = controller.update(100, 100, detection(30, 30, 70, 70))
    small = controller.update(100, 100, detection(45, 45, 55, 55))

    assert medium.zoom_factor == pytest.approx(1.25)
    assert small.zoom_factor == 3.0


def test_center_and_zoom_are_smoothed_across_visible_targets() -> None:
    controller = DigitalZoomController(enabled=True, max_zoom=2.0, smoothing=0.25)

    first = controller.update(100, 100, detection(80, 40, 100, 60))
    second = controller.update(100, 100, detection(80, 40, 100, 60))

    assert first.zoom_factor == pytest.approx(1.25)
    assert second.zoom_factor == pytest.approx(1.4375)
    assert first.crop_left < second.crop_left
    assert 0 <= second.crop_left < second.crop_right <= 100


def test_lost_target_smoothly_returns_toward_unzoomed_center() -> None:
    controller = DigitalZoomController(
        enabled=True,
        max_zoom=2.0,
        smoothing=1.0,
        lost_smoothing=0.5,
    )
    locked = controller.update(100, 100, detection(80, 40, 100, 60))
    first_lost = controller.update(100, 100, None)
    second_lost = controller.update(100, 100, None)

    assert locked.zoom_factor == 2.0
    assert first_lost.zoom_factor == pytest.approx(1.5)
    assert second_lost.zoom_factor == pytest.approx(1.25)
    assert locked.crop_left > first_lost.crop_left > second_lost.crop_left


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("max_zoom", float("nan")), ("max_zoom", 0.9), ("smoothing", 0.0)),
)
def test_invalid_zoom_configuration_is_rejected(keyword: str, value: float) -> None:
    with pytest.raises(ValueError):
        DigitalZoomController(**{keyword: value})
