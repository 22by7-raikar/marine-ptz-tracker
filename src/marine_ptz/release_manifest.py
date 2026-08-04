"""Strict, path-private release-candidate manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import AppConfig, load_config
from .serial_protocol import PROTOCOL_VERSION

RELEASE_MANIFEST_SCHEMA_VERSION = 1
EXPECTED_TARGET_CLASS = "marine_target"


class ReleaseManifestError(ValueError):
    """Raised when release evidence is missing, malformed, or unsafe."""


@dataclass(frozen=True, slots=True)
class ReleaseManifestInputs:
    """Validated operator inputs used to assemble one release manifest."""

    repository: Path
    config: Path
    model: Path
    evaluation_report: Path
    benchmark_report: Path
    output: Path
    image_tag: str
    expected_target_class: str = EXPECTED_TARGET_CLASS
    demonstration_video: Path | None = None
    physical_validation_claimed: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.repository, Path):
            raise ReleaseManifestError("repository must be a filesystem path")
        for name in ("config", "model", "evaluation_report", "benchmark_report", "output"):
            if not isinstance(getattr(self, name), Path):
                raise ReleaseManifestError(f"{name} must be a filesystem path")
        if self.demonstration_video is not None and not isinstance(self.demonstration_video, Path):
            raise ReleaseManifestError("demonstration_video must be a filesystem path")
        if not isinstance(self.image_tag, str) or not re.fullmatch(
            r"(?:[a-z0-9]+(?:[._-][a-z0-9]+)*/)*"
            r"[a-z0-9]+(?:[._-][a-z0-9]+)*"
            r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?"
            r"(?:@sha256:[0-9a-f]{64})?",
            self.image_tag,
        ):
            raise ReleaseManifestError("image tag must be a non-empty registry/tag identifier")
        if self.expected_target_class != EXPECTED_TARGET_CLASS:
            raise ReleaseManifestError("expected target class must be exactly 'marine_target'")
        if not isinstance(self.physical_validation_claimed, bool):
            raise ReleaseManifestError("physical validation claim must be boolean")
        if not isinstance(self.overwrite, bool):
            raise ReleaseManifestError("overwrite must be boolean")


def create_release_manifest(
    inputs: ReleaseManifestInputs,
    *,
    timestamp: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    application_version: Callable[[], str] | None = None,
    git_runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
) -> Mapping[str, object]:
    """Validate evidence, atomically publish the manifest, and return its payload."""
    config_path = _required_regular_file(inputs.config, "configuration")
    model_path = _required_regular_file(inputs.model, "model")
    evaluation_path = _required_regular_file(inputs.evaluation_report, "evaluation report")
    benchmark_path = _required_regular_file(inputs.benchmark_report, "benchmark report")
    demonstration_path = (
        None
        if inputs.demonstration_video is None
        else _required_regular_file(inputs.demonstration_video, "demonstration video")
    )
    _load_strict_json(evaluation_path, "evaluation report")
    _load_strict_json(benchmark_path, "benchmark report")
    config = load_config(config_path)
    created = timestamp().astimezone(timezone.utc)
    if not math.isfinite(created.timestamp()):
        raise ReleaseManifestError("creation timestamp must be finite")
    git = _git_metadata(inputs.repository, git_runner=git_runner)
    app_version = (
        application_version() if application_version is not None else _application_version()
    )
    if not isinstance(app_version, str) or not app_version.strip():
        raise ReleaseManifestError("application version must be a non-empty string")

    payload: dict[str, object] = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "application": {
            "version": app_version,
            "git_commit": git["commit"],
            "git_branch": git["branch"],
            "worktree_state": git["worktree_state"],
            "worktree_dirty": git["dirty"],
        },
        "container_image_tag": inputs.image_tag,
        "host_build": {
            "python_version": _python_version(),
            "platform": _platform_description(),
        },
        "runtime": {
            "configured_mode": config.runtime.execution_mode,
            "expected_target_class": inputs.expected_target_class,
            "confidence_threshold": config.detection.confidence_threshold,
            "image_size": config.detection.image_size,
            "configured_target_labels": list(config.detection.target_labels),
        },
        "evidence": {
            "model": _file_record(model_path),
            "configuration": _file_record(config_path),
            "evaluation_report": _file_record(evaluation_path),
            "benchmark_report": _file_record(benchmark_path),
            "demonstration_video": (
                None if demonstration_path is None else _file_record(demonstration_path)
            ),
        },
        "safety": _safety_record(config),
        "firmware_protocol_version": PROTOCOL_VERSION,
        "physical_validation_claimed": inputs.physical_validation_claimed,
    }
    _validate_json_value(payload, "manifest")
    write_manifest_atomic(payload, inputs.output, overwrite=inputs.overwrite)
    return payload


def write_manifest_atomic(
    payload: Mapping[str, object],
    output: Path,
    *,
    overwrite: bool,
) -> None:
    """Publish strict JSON atomically, with race-safe no-overwrite behavior."""
    _validate_json_value(payload, "manifest")
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if output.exists() and not overwrite:
        raise ReleaseManifestError(
            f"release manifest already exists: {_safe_filename(output)}; use --overwrite"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            try:
                os.link(temporary, output)
            except FileExistsError as exc:
                raise ReleaseManifestError(
                    f"release manifest already exists: {_safe_filename(output)}; use --overwrite"
                ) from exc
            temporary.unlink()
        temporary = None
        _fsync_directory(output.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _required_regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ReleaseManifestError(f"{label} is unavailable: {_safe_filename(path)}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ReleaseManifestError(f"{label} must be a regular file: {_safe_filename(path)}")
    if not os.access(path, os.R_OK):
        raise ReleaseManifestError(f"{label} is not readable: {_safe_filename(path)}")
    return path


def _load_strict_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                parse_constant=lambda token: _raise_json_constant(token),
                object_pairs_hook=_unique_json_object,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ReleaseManifestError) as exc:
        raise ReleaseManifestError(f"{label} is not strict JSON: {_safe_filename(path)}") from exc
    if not isinstance(value, dict):
        raise ReleaseManifestError(f"{label} JSON root must be an object")
    _validate_json_value(value, label)
    return value


def _raise_json_constant(token: str) -> Any:
    raise ReleaseManifestError(f"non-finite JSON constant {token!r}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_json_value(value: object, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReleaseManifestError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ReleaseManifestError(f"{path} contains a non-string object key")
            _validate_json_value(nested, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_json_value(nested, f"{path}[{index}]")
        return
    raise ReleaseManifestError(f"{path} contains unsupported JSON type {type(value).__name__}")


def _file_record(path: Path) -> dict[str, str]:
    return {"filename": _safe_filename(path), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReleaseManifestError(f"cannot hash {_safe_filename(path)}") from exc
    return digest.hexdigest()


def _safe_filename(path: Path) -> str:
    name = path.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        return "<redacted>"
    return name


def _git_metadata(
    repository: Path,
    *,
    git_runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None,
) -> dict[str, object]:
    def run(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if git_runner is not None:
            return git_runner(arguments)
        return subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=False,
            capture_output=True,
            text=True,
        )

    commit = run(("rev-parse", "--verify", "HEAD"))
    branch = run(("branch", "--show-current"))
    status_result = run(("status", "--porcelain", "--untracked-files=normal"))
    if commit.returncode != 0 or branch.returncode != 0 or status_result.returncode != 0:
        return {
            "commit": None,
            "branch": None,
            "worktree_state": "unavailable",
            "dirty": None,
        }
    commit_text = commit.stdout.strip()
    branch_text = branch.stdout.strip() or "<detached>"
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit_text):
        raise ReleaseManifestError("Git returned an invalid commit identifier")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", branch_text):
        branch_text = "<redacted>"
    dirty = bool(status_result.stdout)
    return {
        "commit": commit_text.lower(),
        "branch": branch_text,
        "worktree_state": "dirty" if dirty else "clean",
        "dirty": dirty,
    }


def _application_version() -> str:
    try:
        return version("marine-ptz")
    except PackageNotFoundError:
        return "unknown"


def _python_version() -> str:
    import platform

    return platform.python_version()


def _platform_description() -> str:
    import platform

    return f"{platform.system()}-{platform.machine()}"


def _safety_record(config: AppConfig) -> dict[str, object]:
    return {
        "logical_pan_limits_deg": [
            config.actuator.pan_limits.minimum_deg,
            config.actuator.pan_limits.maximum_deg,
        ],
        "logical_tilt_limits_deg": [
            config.actuator.tilt_limits.minimum_deg,
            config.actuator.tilt_limits.maximum_deg,
        ],
        "neutral_pan_deg": config.actuator.initial_pan_deg,
        "neutral_tilt_deg": config.actuator.initial_tilt_deg,
        "maximum_step_deg": config.tracking.max_step_deg,
        "control_rate_hz": config.tracking.update_rate_hz,
        "watchdog_timeout_s": (None if config.serial is None else config.serial.watchdog_timeout_s),
        "physical_pan_limits_deg": (
            None
            if config.serial is None
            else [
                config.serial.physical_pan_limits.minimum_deg,
                config.serial.physical_pan_limits.maximum_deg,
            ]
        ),
        "physical_tilt_limits_deg": (
            None
            if config.serial is None
            else [
                config.serial.physical_tilt_limits.minimum_deg,
                config.serial.physical_tilt_limits.maximum_deg,
            ]
        ),
        "pan_mapping": (
            None
            if config.serial is None
            else {
                "direction": config.serial.pan_direction,
                "offset_deg": config.serial.pan_offset_deg,
            }
        ),
        "tilt_mapping": (
            None
            if config.serial is None
            else {
                "direction": config.serial.tilt_direction,
                "offset_deg": config.serial.tilt_offset_deg,
            }
        ),
    }


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
