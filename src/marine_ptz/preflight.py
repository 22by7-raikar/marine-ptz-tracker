"""Read-only, mode-specific deployment preflight checks."""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config

PREFLIGHT_MODES = frozenset({"test", "replay", "hardware"})


@dataclass(frozen=True, slots=True)
class PreflightOptions:
    """Inputs checked without constructing a camera, detector, or actuator."""

    mode: str
    config: Path | None = None
    model: Path | None = None
    input_video: Path | None = None
    output_directory: Path | None = None
    target_class: str | None = None
    confidence: float | None = None
    device: str = "cpu"
    runtime_mode: str = "single"
    camera_path: Path | None = None
    serial_path: Path | None = None
    hardware_armed: bool = False


def run_preflight(
    options: PreflightOptions,
    *,
    importer: Callable[[str], Any] = importlib.import_module,
    path_exists: Callable[[Path], bool] = Path.exists,
) -> dict[str, object]:
    """Return a JSON-safe report; no check opens a device or loads a model."""
    checks: list[dict[str, object]] = []

    def check(name: str, action: Callable[[], str]) -> None:
        try:
            detail = action()
        except (ConfigError, ImportError, OSError, RuntimeError, ValueError) as exc:
            checks.append({"name": name, "ok": False, "detail": str(exc)})
        else:
            checks.append({"name": name, "ok": True, "detail": detail})

    if options.mode not in PREFLIGHT_MODES:
        check("mode", lambda: _fail("mode must be test, replay, or hardware"))
        return _result(options.mode, checks)

    check("mode", lambda: options.mode)
    check("base_import", lambda: _import(importer, "marine_ptz"))
    if options.mode == "test":
        return _result(options.mode, checks)

    check("configuration", lambda: _configuration(options))
    check("model", lambda: _readable_file(options.model, "model"))
    check("output_directory", lambda: _writable_directory(options.output_directory))
    check("target_class", lambda: _target_class(options.target_class))
    check("confidence", lambda: _confidence(options.confidence))
    check("runtime_mode", lambda: _runtime_mode(options.runtime_mode))
    for module in ("cv2", "torch", "ultralytics"):
        check(f"import:{module}", lambda module=module: _import(importer, module))
    check("inference_device", lambda: _device(options.device, importer))

    if options.mode == "replay":
        check("input_video", lambda: _readable_file(options.input_video, "input video"))
    else:
        check("hardware_armed", lambda: _armed(options.hardware_armed))
        check(
            "camera_path",
            lambda: _explicit_device(options.camera_path, "camera", path_exists),
        )
        check(
            "serial_path",
            lambda: _explicit_device(options.serial_path, "serial", path_exists),
        )
        check("import:serial", lambda: _import(importer, "serial"))
    return _result(options.mode, checks)


def _result(mode: str, checks: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": mode,
        "ok": all(bool(item["ok"]) for item in checks),
        "checks": checks,
    }


def _configuration(options: PreflightOptions) -> str:
    if options.config is None:
        raise ValueError("configuration path is required")
    load_config(options.config)
    return options.config.name


def _readable_file(path: Path | None, label: str) -> str:
    if path is None:
        raise ValueError(f"{label} path is required")
    if not path.is_file() or not os.access(path, os.R_OK):
        raise ValueError(f"{label} must be an existing readable regular file: {path.name}")
    return path.name


def _writable_directory(path: Path | None) -> str:
    if path is None:
        raise ValueError("output directory is required")
    if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
        raise ValueError(f"output directory must exist and be writable: {path.name}")
    return path.name


def _target_class(value: str | None) -> str:
    if value != "marine_target":
        raise ValueError("target class must be exactly 'marine_target'")
    return value


def _confidence(value: float | None) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be supplied as a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("confidence must be finite and between 0 and 1")
    return str(number)


def _runtime_mode(value: str) -> str:
    if value not in {"single", "concurrent"}:
        raise ValueError("runtime mode must be single or concurrent")
    return value


def _armed(value: bool) -> str:
    if value is not True:
        raise ValueError("hardware mode requires explicit arming acknowledgment")
    return "explicitly acknowledged"


def _explicit_device(
    path: Path | None,
    label: str,
    path_exists: Callable[[Path], bool],
) -> str:
    if path is None:
        raise ValueError(f"explicit {label} path is required; no auto-discovery is performed")
    text = str(path)
    if not path.is_absolute() or not text.startswith("/dev/"):
        raise ValueError(f"{label} path must be an explicit absolute /dev path")
    if not path_exists(path):
        raise ValueError(f"{label} path does not exist: {path.name}")
    return path.name


def _device(value: str, importer: Callable[[str], Any]) -> str:
    if value in {"cpu", "auto"}:
        return value
    if value != "cuda" and not (
        value.startswith("cuda:") and value.removeprefix("cuda:").isdecimal()
    ):
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")
    torch = importer("torch")
    if not bool(torch.cuda.is_available()):
        raise RuntimeError(f"requested {value}, but CUDA is unavailable")
    if value.startswith("cuda:"):
        index = int(value.removeprefix("cuda:"))
        if index >= int(torch.cuda.device_count()):
            raise RuntimeError(f"requested CUDA index {index} is unavailable")
    return value


def _import(importer: Callable[[str], Any], module: str) -> str:
    importer(module)
    return "available"


def _fail(message: str) -> str:
    raise ValueError(message)
