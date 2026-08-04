from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from marine_ptz.preflight import PreflightOptions, run_preflight

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "runtime_preflight.py"


def _runtime_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = tmp_path / "best.pt"
    video = tmp_path / "input.mp4"
    output = tmp_path / "artifacts"
    model.write_bytes(b"model fixture")
    video.write_bytes(b"video fixture")
    output.mkdir()
    return model, video, output


def _importer(name: str) -> object:
    if name == "torch":
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        )
    return object()


def _failed_names(report: dict[str, object]) -> set[str]:
    return {str(item["name"]) for item in report["checks"] if not item["ok"]}


def test_test_preflight_does_not_import_optional_modules() -> None:
    imports: list[str] = []

    report = run_preflight(
        PreflightOptions(mode="test"),
        importer=lambda name: imports.append(name) or object(),
    )

    assert report["ok"] is True
    assert imports == ["marine_ptz"]


def test_replay_preflight_validates_only_replay_inputs(tmp_path: Path) -> None:
    model, video, output = _runtime_files(tmp_path)
    report = run_preflight(
        PreflightOptions(
            mode="replay",
            config=ROOT / "configs" / "development.yaml",
            model=model,
            input_video=video,
            output_directory=output,
            target_class="marine_target",
            confidence=0.60,
            device="cuda:0",
            runtime_mode="concurrent",
        ),
        importer=_importer,
    )

    assert report["ok"] is True
    names = {item["name"] for item in report["checks"]}
    assert "camera_path" not in names
    assert "serial_path" not in names
    assert "hardware_armed" not in names


@pytest.mark.parametrize(
    "replacement,failed",
    [
        ({"model": None}, "model"),
        ({"input_video": None}, "input_video"),
        ({"output_directory": None}, "output_directory"),
        ({"target_class": "boat"}, "target_class"),
        ({"confidence": float("nan")}, "confidence"),
    ],
)
def test_replay_preflight_reports_mode_specific_failures(
    tmp_path: Path, replacement: dict[str, object], failed: str
) -> None:
    model, video, output = _runtime_files(tmp_path)
    values: dict[str, object] = {
        "mode": "replay",
        "config": ROOT / "configs" / "development.yaml",
        "model": model,
        "input_video": video,
        "output_directory": output,
        "target_class": "marine_target",
        "confidence": 0.60,
    }
    values.update(replacement)

    report = run_preflight(PreflightOptions(**values), importer=_importer)  # type: ignore[arg-type]

    assert report["ok"] is False
    assert failed in _failed_names(report)


def test_hardware_preflight_requires_explicit_arming_and_device_paths(tmp_path: Path) -> None:
    model, _video, output = _runtime_files(tmp_path)
    report = run_preflight(
        PreflightOptions(
            mode="hardware",
            config=ROOT / "configs" / "hardware.yaml",
            model=model,
            output_directory=output,
            target_class="marine_target",
            confidence=0.60,
            device="cpu",
        ),
        importer=_importer,
    )

    assert report["ok"] is False
    assert {"hardware_armed", "camera_path", "serial_path"} <= _failed_names(report)
    serialized = json.dumps(report)
    assert "/dev/video" not in serialized
    assert "/dev/tty" not in serialized


def test_hardware_preflight_checks_supplied_paths_without_opening_them(tmp_path: Path) -> None:
    model, _video, output = _runtime_files(tmp_path)
    checked: list[Path] = []

    report = run_preflight(
        PreflightOptions(
            mode="hardware",
            config=ROOT / "configs" / "hardware.yaml",
            model=model,
            output_directory=output,
            target_class="marine_target",
            confidence=0.60,
            camera_path=Path("/dev/marine-camera-explicit"),
            serial_path=Path("/dev/marine-serial-explicit"),
            hardware_armed=True,
        ),
        importer=_importer,
        path_exists=lambda path: checked.append(path) or True,
    )

    assert report["ok"] is True
    assert checked == [
        Path("/dev/marine-camera-explicit"),
        Path("/dev/marine-serial-explicit"),
    ]


def test_machine_readable_test_preflight() -> None:
    completed = subprocess.run(
        (sys.executable, str(TOOL), "--mode", "test", "--json"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout) == {
        "checks": [
            {"detail": "test", "name": "mode", "ok": True},
            {"detail": "available", "name": "base_import", "ok": True},
        ],
        "mode": "test",
        "ok": True,
        "schema_version": 1,
    }


def test_importing_preflight_does_not_import_optional_runtime_packages() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import marine_ptz.preflight; "
            "print(','.join(sorted(set(sys.modules) & "
            "{'cv2','torch','torchvision','ultralytics','serial'})))",
        ),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == ""
