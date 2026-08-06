from __future__ import annotations

from types import SimpleNamespace

import pytest

import marine_ptz.yolo_detector as yolo_detector
from marine_ptz.interfaces import Detector
from marine_ptz.types import Frame
from marine_ptz.yolo_detector import (
    InferenceDeviceError,
    UltralyticsDependencyError,
    UltralyticsDetector,
    resolve_inference_device,
)


class FakeCuda:
    def __init__(self, available: bool, count: int = 1) -> None:
        self._available = available
        self._count = count
        self.synchronized_devices: list[int] = []

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def synchronize(self, device: int) -> None:
        self.synchronized_devices.append(device)


class FakeBoxes:
    def __init__(
        self,
        xyxy: object,
        confidence: object,
        classes: object,
        track_ids: object | None = None,
    ) -> None:
        self.xyxy = xyxy
        self.conf = confidence
        self.cls = classes
        self.id = track_ids


class FakeResult:
    def __init__(self, boxes: object, names: object) -> None:
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, results: list[FakeResult], names: object) -> None:
        self.results = results
        self.names = names
        self.calls: list[dict[str, object]] = []
        self.track_calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[FakeResult]:
        self.calls.append(kwargs)
        return self.results

    def track(self, **kwargs: object) -> list[FakeResult]:
        self.track_calls.append(kwargs)
        return self.results


class FakeTensor:
    def __init__(self, converted: object) -> None:
        self.converted = converted
        self.calls: list[str] = []

    def detach(self) -> FakeTensor:
        self.calls.append("detach")
        return self

    def cpu(self) -> FakeTensor:
        self.calls.append("cpu")
        return self

    def tolist(self) -> object:
        self.calls.append("tolist")
        return self.converted


def image_frame(width: int = 100, height: int = 80) -> Frame:
    return Frame(object(), 1.0, width, height, 0)


@pytest.mark.parametrize(
    ("available", "requested", "expected"),
    [
        (False, "auto", "cpu"),
        (True, "auto", "cuda:0"),
        (False, "cpu", "cpu"),
        (True, "cuda", "cuda:0"),
        (True, "cuda:1", "cuda:1"),
    ],
)
def test_device_resolution(
    available: bool,
    requested: str,
    expected: str,
) -> None:
    torch = SimpleNamespace(cuda=FakeCuda(available, count=2))

    assert resolve_inference_device(requested, torch_module=torch) == expected


def test_explicit_unavailable_cuda_fails_clearly() -> None:
    torch = SimpleNamespace(cuda=FakeCuda(False))

    with pytest.raises(InferenceDeviceError, match="CUDA is unavailable"):
        resolve_inference_device("cuda", torch_module=torch)


def test_out_of_range_cuda_index_fails_clearly() -> None:
    torch = SimpleNamespace(cuda=FakeCuda(True, count=1))

    with pytest.raises(InferenceDeviceError, match="only 1 CUDA"):
        resolve_inference_device("cuda:2", torch_module=torch)


def test_detector_exposes_cuda_synchronization_for_runtime_warmup() -> None:
    cuda = FakeCuda(True, count=2)
    torch = SimpleNamespace(cuda=cuda)
    model = FakeModel([], {})
    detector = UltralyticsDetector(
        "fake.pt",
        device="cuda:1",
        model_factory=lambda _name: model,
        torch_module=torch,
    )

    detector.synchronize()

    assert cuda.synchronized_devices == [1]


def test_result_conversion_uses_model_names_filters_and_clamps() -> None:
    names = {0: "person", 8: "boat"}
    model = FakeModel(
        [
            FakeResult(
                FakeBoxes(
                    [
                        [-10.0, 5.0, 120.0, 90.0],
                        [10.0, 10.0, 30.0, 30.0],
                        [20.0, 20.0, 40.0, 40.0],
                    ],
                    [0.90, 0.99, 0.40],
                    [8.0, 0.0, 8.0],
                ),
                names,
            )
        ],
        names,
    )
    timestamps = iter((4.0, 4.012))
    detector = UltralyticsDetector(
        "fake.pt",
        device="cpu",
        confidence_threshold=0.50,
        iou_threshold=0.30,
        image_size=320,
        class_names=("boat",),
        model_factory=lambda name: model,
        clock=lambda: next(timestamps),
    )

    detections = detector.detect(image_frame())

    assert isinstance(detector, Detector)
    assert len(detections) == 1
    assert detections[0].label == "boat"
    assert detections[0].confidence == 0.90
    assert (
        detections[0].left,
        detections[0].top,
        detections[0].right,
        detections[0].bottom,
    ) == (0.0, 5.0, 100.0, 80.0)
    assert detector.last_inference_ms == pytest.approx(12.0)
    assert model.calls[0]["classes"] == [8]
    assert model.calls[0]["device"] == "cpu"
    assert model.calls[0]["imgsz"] == 320
    assert model.track_calls == []


def test_botsort_calls_track_once_per_frame_with_persistent_low_threshold() -> None:
    names = {0: "marine_target"}
    model = FakeModel(
        [
            FakeResult(
                FakeBoxes(
                    [[1.0, 2.0, 11.0, 12.0]],
                    [0.30],
                    [0.0],
                    [17.0],
                ),
                names,
            )
        ],
        names,
    )
    tracker_path = "configs/trackers/general_botsort.yaml"
    detector = UltralyticsDetector(
        "fake.pt",
        device="cpu",
        confidence_threshold=0.60,
        class_names=("marine_target",),
        tracker_backend="botsort",
        tracker_config=tracker_path,
        tracker_input_confidence=0.20,
        model_factory=lambda _name: model,
    )

    first = detector.detect(image_frame())
    second = detector.detect(image_frame())

    assert [detection.track_id for detection in first] == [17]
    assert second == first
    assert model.calls == []
    assert len(model.track_calls) == 2
    assert all(call["persist"] is True for call in model.track_calls)
    assert all(call["tracker"] == tracker_path for call in model.track_calls)
    assert all(call["conf"] == 0.20 for call in model.track_calls)


@pytest.mark.parametrize(
    "track_ids",
    [None, [None], [float("nan")], [float("inf")], [-1.0], [1.5], [], [1.0, 2.0]],
)
def test_botsort_discards_missing_invalid_or_mismatched_track_ids(
    track_ids: object,
) -> None:
    names = {0: "marine_target"}
    model = FakeModel(
        [
            FakeResult(
                FakeBoxes(
                    [[1.0, 2.0, 11.0, 12.0]],
                    [0.90],
                    [0.0],
                    track_ids,
                ),
                names,
            )
        ],
        names,
    )
    detector = UltralyticsDetector(
        "fake.pt",
        device="cpu",
        confidence_threshold=0.60,
        class_names=("marine_target",),
        tracker_backend="botsort",
        tracker_config="configs/trackers/general_botsort.yaml",
        tracker_input_confidence=0.20,
        model_factory=lambda _name: model,
    )

    assert detector.detect(image_frame()) == ()


def test_separate_botsort_detector_instances_own_separate_models() -> None:
    names = {0: "marine_target"}

    def model() -> FakeModel:
        return FakeModel(
            [
                FakeResult(
                    FakeBoxes([[1.0, 2.0, 11.0, 12.0]], [0.90], [0.0], [3.0]),
                    names,
                )
            ],
            names,
        )

    first_model = model()
    second_model = model()
    common = {
        "device": "cpu",
        "confidence_threshold": 0.60,
        "class_names": ("marine_target",),
        "tracker_backend": "botsort",
        "tracker_config": "configs/trackers/general_botsort.yaml",
        "tracker_input_confidence": 0.20,
    }
    first = UltralyticsDetector(
        "fake.pt",
        model_factory=lambda _name: first_model,
        **common,
    )
    second = UltralyticsDetector(
        "fake.pt",
        model_factory=lambda _name: second_model,
        **common,
    )

    first.detect(image_frame())
    second.detect(image_frame())

    assert len(first_model.track_calls) == 1
    assert len(second_model.track_calls) == 1


def test_detector_only_mode_allows_confidence_below_unused_tracker_default() -> None:
    model = FakeModel([], {0: "boat"})
    detector = UltralyticsDetector(
        "fake.pt",
        device="cpu",
        confidence_threshold=0.10,
        class_names=("boat",),
        model_factory=lambda _name: model,
    )

    assert detector.detect(image_frame()) == ()
    assert model.calls[0]["conf"] == 0.10
    assert model.track_calls == []


@pytest.mark.parametrize(
    ("box", "confidence", "class_id"),
    [
        ([0.0, 0.0, 10.0, 10.0], float("nan"), 8.0),
        ([0.0, 0.0, float("inf"), 10.0], 0.9, 8.0),
        ([0.0, 0.0, 10.0, 10.0], 0.9, float("-inf")),
        ([10.0, 10.0, 10.0, 20.0], 0.9, 8.0),
        ([20.0, 20.0, 10.0, 30.0], 0.9, 8.0),
        ([-10.0, -10.0, -1.0, -1.0], 0.9, 8.0),
    ],
)
def test_malformed_nonfinite_and_zero_area_boxes_are_discarded(
    box: list[float],
    confidence: float,
    class_id: float,
) -> None:
    names = {8: "boat"}
    model = FakeModel(
        [FakeResult(FakeBoxes([box], [confidence], [class_id]), names)],
        names,
    )
    detector = UltralyticsDetector(
        "fake.pt",
        device="cpu",
        class_names=("boat",),
        model_factory=lambda name: model,
    )

    assert detector.detect(image_frame()) == ()


def test_absent_requested_class_skips_inference() -> None:
    model = FakeModel([], {0: "person"})
    detector = UltralyticsDetector(
        "fake.pt",
        device="cpu",
        class_names=("boat",),
        model_factory=lambda name: model,
    )

    assert detector.detect(image_frame()) == ()
    assert model.calls == []


def test_sequence_model_names_and_case_insensitive_matching() -> None:
    names = ["person", "BoAt"]
    model = FakeModel(
        [
            FakeResult(
                FakeBoxes([[1.0, 2.0, 11.0, 12.0]], [0.9], [1.0]),
                names,
            )
        ],
        names,
    )
    detector = UltralyticsDetector(
        "fake.pt",
        device="cpu",
        class_names=("bOaT",),
        model_factory=lambda name: model,
    )

    detections = detector.detect(image_frame())

    assert len(detections) == 1
    assert detections[0].label == "BoAt"
    assert model.calls[0]["classes"] == [1]


def test_gpu_like_tensors_are_detached_moved_to_cpu_and_converted() -> None:
    coordinates = FakeTensor([[1.0, 2.0, 11.0, 12.0]])
    confidences = FakeTensor([0.9])
    classes = FakeTensor([8.0])
    model = FakeModel(
        [
            FakeResult(
                FakeBoxes(coordinates, confidences, classes),
                {8: "boat"},
            )
        ],
        {8: "boat"},
    )
    detector = UltralyticsDetector(
        "fake.pt",
        device="cpu",
        class_names=("boat",),
        model_factory=lambda name: model,
    )

    detections = detector.detect(image_frame())

    assert len(detections) == 1
    assert detections[0].left == 1.0
    assert all(
        tensor.calls == ["detach", "cpu", "tolist"]
        for tensor in (coordinates, confidences, classes)
    )
    assert not any(
        isinstance(value, FakeTensor)
        for detection in detections
        for value in (
            detection.confidence,
            detection.left,
            detection.top,
            detection.right,
            detection.bottom,
        )
    )


def test_gpu_like_track_ids_are_moved_to_cpu_before_conversion() -> None:
    track_ids = FakeTensor([21.0])
    model = FakeModel(
        [
            FakeResult(
                FakeBoxes(
                    [[1.0, 2.0, 11.0, 12.0]],
                    [0.90],
                    [0.0],
                    track_ids,
                ),
                {0: "marine_target"},
            )
        ],
        {0: "marine_target"},
    )
    detector = UltralyticsDetector(
        "fake.pt",
        device="cpu",
        confidence_threshold=0.60,
        class_names=("marine_target",),
        tracker_backend="botsort",
        tracker_config="configs/trackers/general_botsort.yaml",
        tracker_input_confidence=0.20,
        model_factory=lambda _name: model,
    )

    detections = detector.detect(image_frame())

    assert detections[0].track_id == 21
    assert track_ids.calls == ["detach", "cpu", "tolist"]


def test_duplicate_tied_results_are_preserved_in_source_order() -> None:
    boxes = FakeBoxes(
        [[1.0, 2.0, 11.0, 12.0], [1.0, 2.0, 11.0, 12.0]],
        [0.9, 0.9],
        [8.0, 8.0],
    )
    model = FakeModel([FakeResult(boxes, {8: "boat"})], {8: "boat"})
    detector = UltralyticsDetector(
        "fake.pt",
        device="cpu",
        class_names=("boat",),
        model_factory=lambda name: model,
    )

    detections = detector.detect(image_frame())

    assert len(detections) == 2
    assert detections[0] == detections[1]


@pytest.mark.parametrize(
    "boxes",
    [
        None,
        SimpleNamespace(xyxy=object(), conf=[0.9], cls=[8.0]),
        SimpleNamespace(xyxy=[[0.0, 0.0, 10.0]], conf=[0.9], cls=[8.0]),
        SimpleNamespace(xyxy=[[0.0, 0.0, 10.0, 10.0]], conf=object(), cls=[8.0]),
    ],
)
def test_malformed_result_structures_are_discarded(boxes: object) -> None:
    model = FakeModel([FakeResult(boxes, {8: "boat"})], {8: "boat"})
    detector = UltralyticsDetector(
        "fake.pt",
        device="cpu",
        class_names=("boat",),
        model_factory=lambda name: model,
    )

    assert detector.detect(image_frame()) == ()


def test_missing_ultralytics_has_clear_lazy_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        raise UltralyticsDependencyError(f"{name} missing")

    monkeypatch.setattr(yolo_detector, "_load_module", missing)

    with pytest.raises(UltralyticsDependencyError, match="ultralytics missing"):
        UltralyticsDetector("fake.pt", device="cpu")
