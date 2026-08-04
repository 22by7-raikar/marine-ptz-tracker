import re
from pathlib import Path

import pytest

from marine_ptz.config import ConfigError, load_config


def test_development_config_loads() -> None:
    config = load_config(Path("configs/development.yaml"))

    assert config.camera.source == "synthetic"
    assert config.camera.device is None
    assert config.actuator.backend == "simulated"
    assert config.serial is None
    assert config.tracking.lost_target_behavior == "hold"
    assert config.runtime.execution_mode == "single"
    assert config.runtime.result_freshness_s == 0.5


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("deadband_px: 24.0", "deadband_px: -1.0"),
        ("tracking:", "tracking_typo:"),
    ],
)
def test_configuration_validation_reports_clear_path(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    source = Path("configs/development.yaml").read_text(encoding="utf-8")
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(source.replace(replacement, message), encoding="utf-8")

    with pytest.raises(ConfigError, match="tracking"):
        load_config(invalid_path)


@pytest.mark.parametrize(
    ("config_name", "replacement", "field"),
    [
        ("development.yaml", "result_freshness_s: 0.5", "runtime.result_freshness_s"),
        ("development.yaml", "fps: 30", "camera.fps"),
        ("development.yaml", "confidence_threshold: 0.60", "detection.confidence_threshold"),
        ("development.yaml", "initial_pan_deg: 90.0", "actuator.initial_pan_deg"),
        ("development.yaml", "max_step_deg: 3.0", "tracking.max_step_deg"),
        ("hardware.yaml", "read_timeout_s: 0.25", "serial.read_timeout_s"),
        ("hardware.yaml", "pan_offset_deg: 180.0", "serial.pan_offset_deg"),
        ("hardware.yaml", "boot_grace_seconds: 2.0", "serial.boot_grace_seconds"),
        (
            "hardware.yaml",
            "handshake_timeout_seconds: 4.0",
            "serial.handshake_timeout_seconds",
        ),
        (
            "hardware.yaml",
            "handshake_retry_interval_seconds: 0.25",
            "serial.handshake_retry_interval_seconds",
        ),
    ],
)
@pytest.mark.parametrize("invalid", [".nan", ".inf", "-.inf"])
def test_configuration_rejects_non_finite_scalar_values(
    tmp_path: Path,
    config_name: str,
    replacement: str,
    field: str,
    invalid: str,
) -> None:
    source = Path("configs", config_name).read_text(encoding="utf-8")
    invalid_path = tmp_path / config_name
    invalid_path.write_text(
        source.replace(replacement, f"{replacement.split(':')[0]}: {invalid}"), encoding="utf-8"
    )

    with pytest.raises(ConfigError, match=field):
        load_config(invalid_path)


@pytest.mark.parametrize(
    ("key", "first", "second"),
    [("pan_limits_deg", "20.0", "160.0"), ("tilt_limits_deg", "35.0", "145.0")],
)
@pytest.mark.parametrize("invalid", [".nan", ".inf", "-.inf"])
@pytest.mark.parametrize("endpoint", [0, 1])
def test_configuration_rejects_non_finite_limit_endpoints(
    tmp_path: Path,
    key: str,
    first: str,
    second: str,
    invalid: str,
    endpoint: int,
) -> None:
    source = Path("configs/development.yaml").read_text(encoding="utf-8")
    replacement = f"{key}: [{first}, {second}]"
    invalid_range = (
        f"{key}: [{invalid}, {second}]" if endpoint == 0 else f"{key}: [{first}, {invalid}]"
    )
    field = f"actuator.{key}[{endpoint}]"
    invalid_path = tmp_path / "invalid-limits.yaml"
    invalid_path.write_text(source.replace(replacement, invalid_range), encoding="utf-8")

    with pytest.raises(ConfigError, match=re.escape(field)):
        load_config(invalid_path)


def test_hardware_config_loads_with_required_physical_fields() -> None:
    config = load_config(Path("configs/hardware.yaml"))

    assert config.detection.model == (
        "runs/detect/runs/tugboat/yolo11n-marine-target-v3/weights/best.pt"
    )
    assert config.detection.target_labels == ("marine_target",)
    assert config.detection.confidence_threshold == 0.60
    assert config.camera.device == (
        "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"
    )
    assert config.serial is not None
    assert config.serial.device == "/dev/ttyACM0"
    assert config.serial.baudrate == 115200
    assert config.serial.read_timeout_s == 0.25
    assert config.serial.write_timeout_s == 0.25
    assert config.serial.retries == 2
    assert config.serial.watchdog_timeout_s == 1.0
    assert config.serial.boot_grace_seconds == 2.0
    assert config.serial.handshake_timeout_seconds == 4.0
    assert config.serial.handshake_retry_interval_seconds == 0.25
    assert config.serial.pan_direction == -1
    assert config.serial.pan_offset_deg == 180.0
    assert config.serial.tilt_direction == -1
    assert config.serial.tilt_offset_deg == 180.0
    assert config.serial.physical_pan_limits.minimum_deg == 75.0
    assert config.serial.physical_pan_limits.maximum_deg == 105.0
    assert config.serial.physical_tilt_limits.minimum_deg == 75.0
    assert config.serial.physical_tilt_limits.maximum_deg == 105.0
    assert config.actuator.pan_limits.minimum_deg == 75.0
    assert config.actuator.pan_limits.maximum_deg == 105.0
    assert config.actuator.tilt_limits.minimum_deg == 75.0
    assert config.actuator.tilt_limits.maximum_deg == 105.0
    assert config.actuator.initial_pan_deg == 90.0
    assert config.actuator.initial_tilt_deg == 90.0
    assert config.runtime.execution_mode == "single"
    assert config.runtime.result_freshness_s < config.serial.watchdog_timeout_s


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("execution_mode: single", "execution_mode: parallel", "runtime.execution_mode"),
        ("result_freshness_s: 0.5", "result_freshness_s: 0", "runtime.result_freshness_s"),
        (
            "result_freshness_s: 0.5",
            "result_freshness_s: 1.0",
            "serial.watchdog_timeout_s",
        ),
    ],
)
def test_runtime_concurrency_configuration_is_fail_closed(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    source = Path("configs/hardware.yaml").read_text(encoding="utf-8")
    invalid = tmp_path / "invalid-runtime.yaml"
    invalid.write_text(source.replace(old, new), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(invalid)


def test_v4l2_camera_requires_device(tmp_path: Path) -> None:
    source = Path("configs/hardware.yaml").read_text(encoding="utf-8")
    invalid_path = tmp_path / "missing-camera-device.yaml"
    invalid_path.write_text(
        source.replace(
            "  device: /dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0\n",
            "",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="camera.device"):
        load_config(invalid_path)


@pytest.mark.parametrize(
    "missing",
    ["section", "baudrate"],
)
def test_arduino_serial_requires_serial_settings(
    tmp_path: Path,
    missing: str,
) -> None:
    source = Path("configs/hardware.yaml").read_text(encoding="utf-8")
    if missing == "section":
        source = re.sub(r"serial:\n(?:  .+\n)+\n(?=actuator:)", "", source)
    else:
        source = source.replace("  baudrate: 115200\n", "")
    invalid_path = tmp_path / "missing-serial.yaml"
    invalid_path.write_text(source, encoding="utf-8")

    with pytest.raises(ConfigError, match="serial"):
        load_config(invalid_path)


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ("device: /dev/ttyACM0", 'device: ""', "serial.device"),
        ("baudrate: 115200", "baudrate: 12345", "serial.baudrate"),
        ("read_timeout_s: 0.25", "read_timeout_s: 0", "serial.read_timeout_s"),
        ("write_timeout_s: 0.25", "write_timeout_s: 11", "serial.write_timeout_s"),
        ("retries: 2", "retries: 6", "serial.retries"),
        ("command_rate_hz: 10.0", "command_rate_hz: 0", "serial.command_rate_hz"),
        ("watchdog_timeout_s: 1.0", "watchdog_timeout_s: 61", "serial.watchdog_timeout_s"),
        ("watchdog_timeout_s: 1.0", "watchdog_timeout_s: 0.1", "serial.watchdog_timeout_s"),
        ("boot_grace_seconds: 2.0", "boot_grace_seconds: -1", "serial.boot_grace_seconds"),
        (
            "handshake_timeout_seconds: 4.0",
            "handshake_timeout_seconds: 31",
            "serial.handshake_timeout_seconds",
        ),
        (
            "handshake_retry_interval_seconds: 0.25",
            "handshake_retry_interval_seconds: 0",
            "serial.handshake_retry_interval_seconds",
        ),
        ("pan_direction: -1", "pan_direction: 0", "serial.pan_direction"),
        ("tilt_direction: -1", "tilt_direction: 2", "serial.tilt_direction"),
        ("neutral_pan_deg: 90.0", "neutral_pan_deg: 180.0", "serial.neutral_pan_deg"),
    ],
)
def test_serial_configuration_rejects_unsafe_values(
    tmp_path: Path,
    old: str,
    new: str,
    field: str,
) -> None:
    source = Path("configs/hardware.yaml").read_text(encoding="utf-8")
    invalid = tmp_path / "unsafe-serial.yaml"
    invalid.write_text(source.replace(old, new), encoding="utf-8")

    with pytest.raises(ConfigError, match=field):
        load_config(invalid)


def test_serial_mapping_must_fit_physical_limits(tmp_path: Path) -> None:
    source = Path("configs/hardware.yaml").read_text(encoding="utf-8")
    invalid = tmp_path / "unsafe-mapping.yaml"
    invalid.write_text(
        source.replace("pan_offset_deg: 180.0", "pan_offset_deg: 0.0"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="maps outside"):
        load_config(invalid)


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        (
            "boot_grace_seconds: 2.0",
            "boot_grace_seconds: 4.0",
            "serial.boot_grace_seconds",
        ),
        (
            "handshake_retry_interval_seconds: 0.25",
            "handshake_retry_interval_seconds: 4.0",
            "serial.handshake_retry_interval_seconds",
        ),
    ],
)
def test_serial_startup_intervals_must_fit_handshake_deadline(
    tmp_path: Path,
    old: str,
    new: str,
    field: str,
) -> None:
    source = Path("configs/hardware.yaml").read_text(encoding="utf-8")
    invalid = tmp_path / "unsafe-startup.yaml"
    invalid.write_text(source.replace(old, new), encoding="utf-8")

    with pytest.raises(ConfigError, match=field):
        load_config(invalid)
