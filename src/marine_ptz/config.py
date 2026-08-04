"""Typed application configuration loaded from YAML."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import yaml

from .types import AngleLimits


class ConfigError(ValueError):
    """Raised when configuration cannot be parsed or validated."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    mode: str
    telemetry_enabled: bool
    execution_mode: str
    result_freshness_s: float


@dataclass(frozen=True, slots=True)
class CameraConfig:
    source: str
    width: int
    height: int
    fps: float
    device: str | None = None


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    backend: str
    confidence_threshold: float
    target_labels: tuple[str, ...]
    model: str
    device: str
    iou_threshold: float
    image_size: int


@dataclass(frozen=True, slots=True)
class ActuatorConfig:
    backend: str
    pan_limits: AngleLimits
    tilt_limits: AngleLimits
    initial_pan_deg: float
    initial_tilt_deg: float


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    deadband_px: float
    proportional_gain: float
    max_step_deg: float
    update_rate_hz: float
    lost_target_behavior: str


@dataclass(frozen=True, slots=True)
class SerialConfig:
    device: str
    baudrate: int
    read_timeout_s: float
    write_timeout_s: float
    retries: int
    command_rate_hz: float
    watchdog_timeout_s: float
    boot_grace_seconds: float
    handshake_timeout_seconds: float
    handshake_retry_interval_seconds: float
    pan_direction: int
    tilt_direction: int
    pan_offset_deg: float
    tilt_offset_deg: float
    physical_pan_limits: AngleLimits
    physical_tilt_limits: AngleLimits
    neutral_pan_deg: float
    neutral_tilt_deg: float


@dataclass(frozen=True, slots=True)
class AppConfig:
    runtime: RuntimeConfig
    camera: CameraConfig
    detection: DetectionConfig
    actuator: ActuatorConfig
    tracking: TrackingConfig
    serial: SerialConfig | None = None


def load_config(path: str | Path) -> AppConfig:
    """Load and validate one YAML configuration file."""
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {config_path}: {exc}") from exc

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    root = _mapping(document, "configuration")
    _only_keys(
        root,
        {"runtime", "camera", "detection", "actuator", "tracking", "serial"},
        "configuration",
    )

    runtime_data = _required_mapping(root, "runtime")
    camera_data = _required_mapping(root, "camera")
    detection_data = _required_mapping(root, "detection")
    actuator_data = _required_mapping(root, "actuator")
    tracking_data = _required_mapping(root, "tracking")
    serial_data = _optional_mapping(root, "serial")

    runtime = _runtime_config(runtime_data)
    camera = _camera_config(camera_data)
    detection = _detection_config(detection_data)
    actuator = _actuator_config(actuator_data)
    tracking = _tracking_config(tracking_data)
    serial = _serial_config(serial_data) if serial_data is not None else None

    _validate_backend_requirements(camera, actuator, serial)

    if (
        actuator.backend == "arduino_serial"
        and serial is not None
        and runtime.result_freshness_s >= serial.watchdog_timeout_s
    ):
        raise ConfigError(
            "runtime.result_freshness_s must be less than "
            "serial.watchdog_timeout_s for Arduino actuation"
        )

    if tracking.deadband_px >= min(camera.width, camera.height) / 2:
        raise ConfigError("tracking.deadband_px must be smaller than half the frame")

    return AppConfig(runtime, camera, detection, actuator, tracking, serial)


def _validate_backend_requirements(
    camera: CameraConfig,
    actuator: ActuatorConfig,
    serial: SerialConfig | None,
) -> None:
    if camera.source == "v4l2" and camera.device is None:
        raise ConfigError("camera.device is required when camera.source is 'v4l2'")
    if actuator.backend == "arduino_serial" and serial is None:
        raise ConfigError("serial section is required when actuator.backend is 'arduino_serial'")
    if actuator.backend == "arduino_serial" and serial is not None:
        _validate_serial_mapping(actuator, serial)


def _runtime_config(data: Mapping[str, Any]) -> RuntimeConfig:
    path = "runtime"
    _only_keys(
        data,
        {"mode", "telemetry_enabled", "execution_mode", "result_freshness_s"},
        path,
    )
    mode = _string(data, "mode", path)
    if mode not in {"development", "hardware"}:
        raise ConfigError("runtime.mode must be 'development' or 'hardware'")
    execution_mode = _optional_string(data, "execution_mode", path, "single")
    if execution_mode not in {"single", "concurrent"}:
        raise ConfigError("runtime.execution_mode must be 'single' or 'concurrent'")
    result_freshness = _optional_number(data, "result_freshness_s", path, 0.5)
    if result_freshness <= 0.0:
        raise ConfigError("runtime.result_freshness_s must be greater than zero")
    return RuntimeConfig(
        mode,
        _boolean(data, "telemetry_enabled", path),
        execution_mode,
        result_freshness,
    )


def _camera_config(data: Mapping[str, Any]) -> CameraConfig:
    path = "camera"
    _only_keys(data, {"source", "device", "width", "height", "fps"}, path)
    width = _integer(data, "width", path)
    height = _integer(data, "height", path)
    fps = _number(data, "fps", path)
    if width <= 0:
        raise ConfigError("camera.width must be greater than zero")
    if height <= 0:
        raise ConfigError("camera.height must be greater than zero")
    if fps <= 0:
        raise ConfigError("camera.fps must be greater than zero")
    device = data.get("device")
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise ConfigError("camera.device must be a non-empty string when provided")
    return CameraConfig(_string(data, "source", path), width, height, fps, device)


def _detection_config(data: Mapping[str, Any]) -> DetectionConfig:
    path = "detection"
    _only_keys(
        data,
        {
            "backend",
            "confidence_threshold",
            "target_labels",
            "model",
            "device",
            "iou_threshold",
            "image_size",
        },
        path,
    )
    threshold = _number(data, "confidence_threshold", path)
    if not 0 <= threshold <= 1:
        raise ConfigError("detection.confidence_threshold must be between 0 and 1")
    iou_threshold = _optional_number(data, "iou_threshold", path, 0.45)
    if not 0 <= iou_threshold <= 1:
        raise ConfigError("detection.iou_threshold must be between 0 and 1")
    image_size = _optional_integer(data, "image_size", path, 640)
    if image_size <= 0:
        raise ConfigError("detection.image_size must be greater than zero")
    model = _optional_string(data, "model", path, "yolo11n.pt")
    device = _optional_string(data, "device", path, "auto")
    if not re.fullmatch(r"(?:auto|cpu|cuda(?::\d+)?)", device):
        raise ConfigError("detection.device must be auto, cpu, cuda, or cuda:N")
    raw_labels = data.get("target_labels")
    if not isinstance(raw_labels, list) or not raw_labels:
        raise ConfigError("detection.target_labels must be a non-empty list")
    if any(not isinstance(label, str) or not label.strip() for label in raw_labels):
        raise ConfigError("detection.target_labels entries must be non-empty strings")
    return DetectionConfig(
        _string(data, "backend", path),
        threshold,
        tuple(raw_labels),
        model,
        device,
        iou_threshold,
        image_size,
    )


def _actuator_config(data: Mapping[str, Any]) -> ActuatorConfig:
    path = "actuator"
    _only_keys(
        data,
        {
            "backend",
            "pan_limits_deg",
            "tilt_limits_deg",
            "initial_pan_deg",
            "initial_tilt_deg",
        },
        path,
    )
    pan_limits = _limits(data, "pan_limits_deg", path)
    tilt_limits = _limits(data, "tilt_limits_deg", path)
    initial_pan = _number(data, "initial_pan_deg", path)
    initial_tilt = _number(data, "initial_tilt_deg", path)
    if not pan_limits.contains(initial_pan):
        raise ConfigError("actuator.initial_pan_deg must be within actuator.pan_limits_deg")
    if not tilt_limits.contains(initial_tilt):
        raise ConfigError("actuator.initial_tilt_deg must be within actuator.tilt_limits_deg")
    return ActuatorConfig(
        _string(data, "backend", path),
        pan_limits,
        tilt_limits,
        initial_pan,
        initial_tilt,
    )


def _tracking_config(data: Mapping[str, Any]) -> TrackingConfig:
    path = "tracking"
    _only_keys(
        data,
        {
            "deadband_px",
            "proportional_gain",
            "max_step_deg",
            "update_rate_hz",
            "lost_target_behavior",
        },
        path,
    )
    deadband = _number(data, "deadband_px", path)
    gain = _number(data, "proportional_gain", path)
    max_step = _number(data, "max_step_deg", path)
    update_rate = _number(data, "update_rate_hz", path)
    behavior = _string(data, "lost_target_behavior", path)
    if deadband < 0:
        raise ConfigError("tracking.deadband_px must be zero or greater")
    if gain < 0:
        raise ConfigError("tracking.proportional_gain must be zero or greater")
    if max_step <= 0:
        raise ConfigError("tracking.max_step_deg must be greater than zero")
    if update_rate <= 0:
        raise ConfigError("tracking.update_rate_hz must be greater than zero")
    if behavior not in {"hold", "return_to_neutral"}:
        raise ConfigError("tracking.lost_target_behavior must be 'hold' or 'return_to_neutral'")
    return TrackingConfig(deadband, gain, max_step, update_rate, behavior)


def _serial_config(data: Mapping[str, Any]) -> SerialConfig:
    path = "serial"
    _only_keys(
        data,
        {
            "device",
            "baudrate",
            "read_timeout_s",
            "write_timeout_s",
            "retries",
            "command_rate_hz",
            "watchdog_timeout_s",
            "boot_grace_seconds",
            "handshake_timeout_seconds",
            "handshake_retry_interval_seconds",
            "pan_direction",
            "tilt_direction",
            "pan_offset_deg",
            "tilt_offset_deg",
            "physical_pan_limits_deg",
            "physical_tilt_limits_deg",
            "neutral_pan_deg",
            "neutral_tilt_deg",
        },
        path,
    )
    baudrate = _integer(data, "baudrate", path)
    if baudrate not in {9_600, 19_200, 38_400, 57_600, 115_200}:
        raise ConfigError("serial.baudrate must be a supported standard baud rate")
    read_timeout = _bounded_positive_number(data, "read_timeout_s", path, 10.0)
    write_timeout = _bounded_positive_number(data, "write_timeout_s", path, 10.0)
    retries = _integer(data, "retries", path)
    if not 0 <= retries <= 5:
        raise ConfigError("serial.retries must be between 0 and 5")
    command_rate = _bounded_positive_number(data, "command_rate_hz", path, 100.0)
    watchdog_timeout = _bounded_positive_number(
        data,
        "watchdog_timeout_s",
        path,
        60.0,
    )
    if watchdog_timeout <= 2.0 / command_rate:
        raise ConfigError("serial.watchdog_timeout_s must exceed two serial command intervals")
    boot_grace = _bounded_nonnegative_number(
        data,
        "boot_grace_seconds",
        path,
        10.0,
    )
    handshake_timeout = _bounded_positive_number(
        data,
        "handshake_timeout_seconds",
        path,
        30.0,
    )
    handshake_retry_interval = _bounded_positive_number(
        data,
        "handshake_retry_interval_seconds",
        path,
        5.0,
    )
    if boot_grace >= handshake_timeout:
        raise ConfigError(
            "serial.boot_grace_seconds must be less than serial.handshake_timeout_seconds"
        )
    if handshake_retry_interval >= handshake_timeout:
        raise ConfigError(
            "serial.handshake_retry_interval_seconds must be less than "
            "serial.handshake_timeout_seconds"
        )
    pan_direction = _direction(data, "pan_direction", path)
    tilt_direction = _direction(data, "tilt_direction", path)
    pan_offset = _number(data, "pan_offset_deg", path)
    tilt_offset = _number(data, "tilt_offset_deg", path)
    physical_pan_limits = _limits(data, "physical_pan_limits_deg", path)
    physical_tilt_limits = _limits(data, "physical_tilt_limits_deg", path)
    neutral_pan = _number(data, "neutral_pan_deg", path)
    neutral_tilt = _number(data, "neutral_tilt_deg", path)
    _validate_protocol_limits(physical_pan_limits, "serial.physical_pan_limits_deg")
    _validate_protocol_limits(physical_tilt_limits, "serial.physical_tilt_limits_deg")
    _validate_protocol_angle(neutral_pan, "serial.neutral_pan_deg")
    _validate_protocol_angle(neutral_tilt, "serial.neutral_tilt_deg")
    if not physical_pan_limits.contains(neutral_pan):
        raise ConfigError("serial.neutral_pan_deg must be within serial.physical_pan_limits_deg")
    if not physical_tilt_limits.contains(neutral_tilt):
        raise ConfigError("serial.neutral_tilt_deg must be within serial.physical_tilt_limits_deg")
    return SerialConfig(
        _string(data, "device", path),
        baudrate,
        read_timeout,
        write_timeout,
        retries,
        command_rate,
        watchdog_timeout,
        boot_grace,
        handshake_timeout,
        handshake_retry_interval,
        pan_direction,
        tilt_direction,
        pan_offset,
        tilt_offset,
        physical_pan_limits,
        physical_tilt_limits,
        neutral_pan,
        neutral_tilt,
    )


def _validate_serial_mapping(
    actuator: ActuatorConfig,
    serial: SerialConfig,
) -> None:
    for axis, logical, physical, direction, offset, initial, neutral in (
        (
            "pan",
            actuator.pan_limits,
            serial.physical_pan_limits,
            serial.pan_direction,
            serial.pan_offset_deg,
            actuator.initial_pan_deg,
            serial.neutral_pan_deg,
        ),
        (
            "tilt",
            actuator.tilt_limits,
            serial.physical_tilt_limits,
            serial.tilt_direction,
            serial.tilt_offset_deg,
            actuator.initial_tilt_deg,
            serial.neutral_tilt_deg,
        ),
    ):
        mapped = (
            direction * logical.minimum_deg + offset,
            direction * logical.maximum_deg + offset,
        )
        if any(not physical.contains(value) for value in mapped):
            raise ConfigError(
                f"actuator.{axis}_limits_deg maps outside serial.physical_{axis}_limits_deg"
            )
        mapped_initial = direction * initial + offset
        if abs(mapped_initial - neutral) > 1e-9:
            raise ConfigError(
                f"mapped actuator.initial_{axis}_deg must equal serial.neutral_{axis}_deg"
            )


def _bounded_positive_number(
    data: Mapping[str, Any],
    key: str,
    path: str,
    maximum: float,
) -> float:
    value = _number(data, key, path)
    if not 0.0 < value <= maximum:
        raise ConfigError(f"{path}.{key} must be greater than zero and at most {maximum:g}")
    return value


def _bounded_nonnegative_number(
    data: Mapping[str, Any],
    key: str,
    path: str,
    maximum: float,
) -> float:
    value = _number(data, key, path)
    if not 0.0 <= value <= maximum:
        raise ConfigError(f"{path}.{key} must be zero or greater and at most {maximum:g}")
    return value


def _direction(data: Mapping[str, Any], key: str, path: str) -> int:
    value = _integer(data, key, path)
    if value not in {-1, 1}:
        raise ConfigError(f"{path}.{key} must be exactly -1 or 1")
    return value


def _validate_protocol_limits(limits: AngleLimits, field: str) -> None:
    _validate_protocol_angle(limits.minimum_deg, f"{field}[0]")
    _validate_protocol_angle(limits.maximum_deg, f"{field}[1]")


def _validate_protocol_angle(value: float, field: str) -> None:
    if not 0.0 <= value <= 180.0 or not value.is_integer():
        raise ConfigError(f"{field} must be an integer degree value between 0 and 180")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{path} keys must be strings")
    return value


def _required_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in data:
        raise ConfigError(f"missing required section: {key}")
    return _mapping(data[key], key)


def _optional_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    if key not in data:
        return None
    return _mapping(data[key], key)


def _only_keys(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{path} contains unknown key(s): {', '.join(unknown)}")


def _string(data: Mapping[str, Any], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}.{key} must be a non-empty string")
    return value


def _optional_string(
    data: Mapping[str, Any],
    key: str,
    path: str,
    default: str,
) -> str:
    if key not in data:
        return default
    return _string(data, key, path)


def _boolean(data: Mapping[str, Any], key: str, path: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"{path}.{key} must be a boolean")
    return value


def _number(data: Mapping[str, Any], key: str, path: str) -> float:
    return _finite_number(data.get(key), f"{path}.{key}")


def _optional_number(
    data: Mapping[str, Any],
    key: str,
    path: str,
    default: float,
) -> float:
    if key not in data:
        return default
    return _number(data, key, path)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ConfigError(f"{field} must be a finite number")
    return number


def _integer(data: Mapping[str, Any], key: str, path: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}.{key} must be an integer")
    return value


def _optional_integer(
    data: Mapping[str, Any],
    key: str,
    path: str,
    default: int,
) -> int:
    if key not in data:
        return default
    return _integer(data, key, path)


def _limits(data: Mapping[str, Any], key: str, path: str) -> AngleLimits:
    raw = data.get(key)
    if not isinstance(raw, list) or len(raw) != 2:
        raise ConfigError(f"{path}.{key} must contain exactly two numbers")
    minimum = _finite_number(raw[0], f"{path}.{key}[0]")
    maximum = _finite_number(raw[1], f"{path}.{key}[1]")
    if minimum >= maximum:
        raise ConfigError(f"{path}.{key} minimum must be less than maximum")
    return AngleLimits(minimum, maximum)
