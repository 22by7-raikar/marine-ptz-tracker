"""Ultralytics YOLO detector with lazy optional dependencies."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from numbers import Real
from pathlib import Path
from typing import Any

from .types import Detection, Frame


class UltralyticsDetectorError(RuntimeError):
    """Base error for actionable YOLO detector failures."""


class UltralyticsDependencyError(UltralyticsDetectorError):
    """Raised when Torch or Ultralytics is unavailable."""


class InferenceDeviceError(UltralyticsDetectorError):
    """Raised when a requested inference device cannot be used."""


def resolve_inference_device(requested: str, *, torch_module: Any | None = None) -> str:
    """Resolve auto/cpu/cuda/cuda:N without importing Torch for CPU."""
    value = requested.strip().lower()
    if value == "cpu":
        return "cpu"
    if not re.fullmatch(r"(?:auto|cuda(?::\d+)?)", value):
        raise InferenceDeviceError("device must be auto, cpu, cuda, or cuda:N")
    torch = torch_module if torch_module is not None else _load_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    if value == "auto":
        return "cuda:0" if cuda_available else "cpu"
    if not cuda_available:
        raise InferenceDeviceError(
            f"device {value!r} was requested, but CUDA is unavailable to this process"
        )
    if ":" in value:
        index = int(value.split(":", 1)[1])
        count = int(torch.cuda.device_count())
        if index >= count:
            raise InferenceDeviceError(
                f"device {value!r} was requested, but only {count} CUDA device(s) are visible"
            )
        return value
    return "cuda:0"


class UltralyticsDetector:
    """Convert Ultralytics results into hardware-neutral detections."""

    def __init__(
        self,
        model: str | Path = "yolo11n.pt",
        *,
        device: str = "auto",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        class_names: Sequence[str] | None = None,
        tracker_backend: str = "none",
        tracker_config: str | Path | None = None,
        tracker_input_confidence: float = 0.20,
        model_factory: Callable[[str], Any] | None = None,
        torch_module: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be finite and between 0 and 1")
        if not math.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be finite and between 0 and 1")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if not isinstance(tracker_backend, str):
            raise ValueError("tracker_backend must be 'none' or 'botsort'")
        normalized_tracker = tracker_backend.strip().casefold()
        if normalized_tracker not in {"none", "botsort"}:
            raise ValueError("tracker_backend must be 'none' or 'botsort'")
        if (
            isinstance(tracker_input_confidence, bool)
            or not isinstance(tracker_input_confidence, Real)
            or not math.isfinite(float(tracker_input_confidence))
            or not 0 <= float(tracker_input_confidence) <= 1
            or (
                normalized_tracker == "botsort"
                and float(tracker_input_confidence) > confidence_threshold
            )
        ):
            raise ValueError(
                "tracker_input_confidence must be finite, between 0 and 1, and not "
                "exceed confidence_threshold"
            )
        tracker_path: Path | None = None
        if tracker_config is not None:
            tracker_path = Path(tracker_config)
        if tracker_path is not None and not tracker_path.is_file():
            raise ValueError("tracker_config must name an existing file when provided")
        if normalized_tracker == "botsort" and tracker_path is None:
            raise ValueError(
                "tracker_config must name an existing file when tracker_backend is 'botsort'"
            )
        self._device = resolve_inference_device(device, torch_module=torch_module)
        self._torch = (
            torch_module
            if self._device.startswith("cuda") and torch_module is not None
            else (_load_module("torch") if self._device.startswith("cuda") else None)
        )
        if model_factory is None:
            ultralytics = _load_module("ultralytics")
            model_factory = ultralytics.YOLO
        try:
            self._model = model_factory(str(model))
        except Exception as exc:
            raise UltralyticsDetectorError(
                f"unable to load Ultralytics model {str(model)!r}: {exc}"
            ) from exc
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._image_size = image_size
        self._tracker_backend = normalized_tracker
        self._tracker_config = tracker_path
        self._tracker_input_confidence = float(tracker_input_confidence)
        self._class_names = (
            frozenset(name.strip().casefold() for name in class_names if name.strip())
            if class_names
            else None
        )
        self._clock = clock
        self._last_inference_ms = 0.0

    @property
    def active_device(self) -> str:
        return self._device

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def tracker_backend(self) -> str:
        return self._tracker_backend

    def synchronize(self) -> None:
        """Synchronize the selected CUDA device when warm-up timing requires it."""
        if self._torch is None:
            return
        index = int(self._device.split(":", 1)[1])
        self._torch.cuda.synchronize(index)

    def detect(self, frame: Frame) -> Sequence[Detection]:
        if frame.image is None:
            raise UltralyticsDetectorError("YOLO detection requires an image-bearing frame")
        model_names = _normalize_names(getattr(self._model, "names", None))
        class_ids = _resolve_class_ids(model_names, self._class_names)
        if self._class_names is not None and model_names and not class_ids:
            self._last_inference_ms = 0.0
            return ()

        started = self._clock()
        try:
            common_arguments = {
                "source": frame.image,
                "conf": (
                    self._tracker_input_confidence
                    if self._tracker_backend == "botsort"
                    else self._confidence_threshold
                ),
                "iou": self._iou_threshold,
                "imgsz": self._image_size,
                "device": self._device,
                "classes": class_ids if class_ids else None,
                "verbose": False,
            }
            if self._tracker_backend == "botsort":
                results = list(
                    self._model.track(
                        **common_arguments,
                        persist=True,
                        tracker=str(self._tracker_config),
                    )
                )
            else:
                results = list(self._model.predict(**common_arguments))
        except Exception as exc:
            raise UltralyticsDetectorError(
                f"Ultralytics inference failed on {self._device}: {exc}"
            ) from exc
        self._last_inference_ms = max(0.0, (self._clock() - started) * 1000)
        detections: list[Detection] = []
        for result in results:
            names = _normalize_names(getattr(result, "names", None)) or model_names
            detections.extend(self._convert_result(result, names, frame))
        return tuple(detections)

    def _convert_result(
        self,
        result: Any,
        names: Mapping[int, str],
        frame: Frame,
    ) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        coordinates = _to_list(getattr(boxes, "xyxy", ()))
        confidences = _to_list(getattr(boxes, "conf", ()))
        classes = _to_list(getattr(boxes, "cls", ()))
        track_ids = (
            _to_list(getattr(boxes, "id", ()))
            if self._tracker_backend == "botsort"
            else [None] * len(coordinates)
        )
        if not (len(coordinates) == len(confidences) == len(classes) == len(track_ids)):
            return []
        converted: list[Detection] = []
        for raw_box, raw_confidence, raw_class, raw_track_id in zip(
            coordinates,
            confidences,
            classes,
            track_ids,
        ):
            parsed = _validated_box(
                raw_box,
                raw_confidence,
                raw_class,
                names,
                frame.width,
                frame.height,
                (
                    self._tracker_input_confidence
                    if self._tracker_backend == "botsort"
                    else self._confidence_threshold
                ),
                self._class_names,
                raw_track_id,
                require_track_id=self._tracker_backend == "botsort",
            )
            if parsed is not None:
                converted.append(parsed)
        return converted


def _load_module(name: str) -> Any:
    try:
        return import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise UltralyticsDependencyError(
            f"{name} is unavailable; install the constrained vision dependencies"
        ) from exc


def _normalize_names(raw: object) -> dict[int, str]:
    if isinstance(raw, Mapping):
        return {
            int(key): str(value)
            for key, value in raw.items()
            if isinstance(key, (int, str)) and str(key).isdigit()
        }
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return {index: str(value) for index, value in enumerate(raw)}
    return {}


def _resolve_class_ids(
    names: Mapping[int, str],
    requested: frozenset[str] | None,
) -> list[int]:
    if requested is None:
        return []
    return sorted(class_id for class_id, name in names.items() if name.casefold() in requested)


def _to_list(value: Any) -> list[Any]:
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _validated_box(
    raw_box: Any,
    raw_confidence: Any,
    raw_class: Any,
    names: Mapping[int, str],
    width: int,
    height: int,
    confidence_threshold: float,
    requested_names: frozenset[str] | None,
    raw_track_id: Any = None,
    *,
    require_track_id: bool = False,
) -> Detection | None:
    if not isinstance(raw_box, Sequence) or len(raw_box) != 4:
        return None
    values = (*raw_box, raw_confidence, raw_class)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        return None
    confidence = float(raw_confidence)
    if confidence < confidence_threshold or not 0 <= confidence <= 1:
        return None
    class_value = float(raw_class)
    class_id = int(class_value)
    if class_value != class_id or class_id not in names:
        return None
    label = names[class_id]
    if requested_names is not None and label.casefold() not in requested_names:
        return None
    left, top, right, bottom = (float(value) for value in raw_box)
    left = min(float(width), max(0.0, left))
    top = min(float(height), max(0.0, top))
    right = min(float(width), max(0.0, right))
    bottom = min(float(height), max(0.0, bottom))
    if right <= left or bottom <= top:
        return None
    track_id = _validated_track_id(raw_track_id)
    if require_track_id and track_id is None:
        return None
    if raw_track_id is not None and track_id is None:
        return None
    return Detection(label, confidence, left, top, right, bottom, track_id)


def _validated_track_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    track_id = int(number)
    if number != track_id or track_id < 0:
        return None
    return track_id
