from pathlib import Path
import re

import pytest

from marine_ptz.config import ConfigError, load_config


def test_development_config_loads() -> None:
    config = load_config(Path("configs/development.yaml"))

    assert config.camera.source == "synthetic"
    assert config.camera.device is None
    assert config.actuator.backend == "simulated"
    assert config.serial is None
    assert config.tracking.lost_target_behavior == "hold"


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
        ("development.yaml", "fps: 30", "camera.fps"),
        ("development.yaml", "confidence_threshold: 0.60", "detection.confidence_threshold"),
        ("development.yaml", "initial_pan_deg: 90.0", "actuator.initial_pan_deg"),
        ("development.yaml", "max_step_deg: 3.0", "tracking.max_step_deg"),
        ("hardware.yaml", "timeout_s: 0.25", "serial.timeout_s"),
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
    invalid_path.write_text(source.replace(replacement, f"{replacement.split(':')[0]}: {invalid}"), encoding="utf-8")

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
        f"{key}: [{invalid}, {second}]"
        if endpoint == 0
        else f"{key}: [{first}, {invalid}]"
    )
    field = f"actuator.{key}[{endpoint}]"
    invalid_path = tmp_path / "invalid-limits.yaml"
    invalid_path.write_text(source.replace(replacement, invalid_range), encoding="utf-8")

    with pytest.raises(ConfigError, match=re.escape(field)):
        load_config(invalid_path)


def test_hardware_config_loads_with_required_physical_fields() -> None:
    config = load_config(Path("configs/hardware.yaml"))

    assert config.camera.device == "/dev/video0"
    assert config.serial is not None
    assert config.serial.device == "/dev/ttyACM0"


def test_v4l2_camera_requires_device(tmp_path: Path) -> None:
    source = Path("configs/hardware.yaml").read_text(encoding="utf-8")
    invalid_path = tmp_path / "missing-camera-device.yaml"
    invalid_path.write_text(source.replace("  device: /dev/video0\n", ""), encoding="utf-8")

    with pytest.raises(ConfigError, match="camera.device"):
        load_config(invalid_path)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            "serial:\n  device: /dev/ttyACM0\n  baudrate: 115200\n  timeout_s: 0.25\n\n",
            "",
        ),
        ("  baudrate: 115200\n", ""),
    ],
)
def test_arduino_serial_requires_serial_settings(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    source = Path("configs/hardware.yaml").read_text(encoding="utf-8")
    invalid_path = tmp_path / "missing-serial.yaml"
    invalid_path.write_text(source.replace(replacement, message), encoding="utf-8")

    with pytest.raises(ConfigError, match="serial"):
        load_config(invalid_path)
