from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPILE_HELPER = ROOT / "tools" / "compile_arduino.sh"


def _write_fake_arduino_cli(tmp_path: Path) -> Path:
    fake_cli = tmp_path / "arduino-cli"
    fake_cli.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'printf \'%s\\n\' "$@" >> "$CAPTURE"\n'
        'case "$1" in\n'
        "  version)\n"
        "    printf '%s\\n' \"$FAKE_VERSION_OUTPUT\"\n"
        '    exit "${FAKE_VERSION_EXIT:-0}"\n'
        "    ;;\n"
        "  core) echo 'Arduino AVR Boards arduino:avr 1.8.8' ;;\n"
        "  lib) echo 'Servo 1.3.0' ;;\n"
        "  compile) exit 0 ;;\n"
        "  *) exit 91 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    return fake_cli


def _run_compile_helper(
    tmp_path: Path,
    *,
    version_output: str,
    build_dir: Path | None = None,
    version_exit: int = 0,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake_cli = _write_fake_arduino_cli(tmp_path)
    capture = tmp_path / "arguments.txt"
    environment = os.environ | {
        "ARDUINO_CLI": str(fake_cli),
        "CAPTURE": str(capture),
        "FAKE_VERSION_OUTPUT": version_output,
        "FAKE_VERSION_EXIT": str(version_exit),
    }
    command = [str(COMPILE_HELPER)]
    if build_dir is not None:
        command.append(str(build_dir))
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    return result, capture


def test_compile_helper_validates_prerequisites_and_builds_from_any_directory(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "caller-build"
    build_dir.mkdir()
    sentinel = build_dir / "preserve-me"
    sentinel.write_text("caller-owned", encoding="utf-8")
    result, capture = _run_compile_helper(
        tmp_path,
        version_output="arduino-cli  Version: 1.5.1 Commit: fake",
        build_dir=build_dir,
    )

    assert result.returncode == 0, result.stderr
    assert "--fqbn\narduino:avr:uno" in capture.read_text(encoding="utf-8")
    assert "--warnings\nall" in capture.read_text(encoding="utf-8")
    assert str(ROOT / "firmware" / "ptz_controller") in capture.read_text(encoding="utf-8")
    assert str(build_dir) in capture.read_text(encoding="utf-8")
    assert "upload" not in capture.read_text(encoding="utf-8").casefold()
    assert sentinel.read_text(encoding="utf-8") == "caller-owned"


@pytest.mark.parametrize(
    "version_output",
    [
        "arduino-cli  Version: 1.5.10 Commit: fake",
        "arduino-cli  Version: 1.5.11 Commit: fake",
        "arduino-cli  Version: 1.5.1-rc1 Commit: fake",
        "arduino-cli  Version: 11.5.1 Commit: fake",
        "arduino-cli build information only",
        "arduino-cli  Version:",
        "arduino-cli  Version:1.5.1 Commit: fake",
        "Version: 1.5.1\nVersion: 1.5.10",
    ],
)
def test_compile_helper_rejects_non_exact_or_ambiguous_cli_versions(
    tmp_path: Path,
    version_output: str,
) -> None:
    build_dir = tmp_path / "caller-build"
    build_dir.mkdir()
    sentinel = build_dir / "preserve-me"
    sentinel.write_text("caller-owned", encoding="utf-8")
    result, capture = _run_compile_helper(
        tmp_path,
        version_output=version_output,
        build_dir=build_dir,
    )

    assert result.returncode == 1
    assert "expected 1.5.1" in result.stderr
    captured_arguments = capture.read_text(encoding="utf-8") if capture.exists() else ""
    assert "compile" not in captured_arguments
    assert sentinel.read_text(encoding="utf-8") == "caller-owned"


def test_compile_helper_preserves_a_failing_cli_version_command_status(tmp_path: Path) -> None:
    result, capture = _run_compile_helper(
        tmp_path,
        version_output="version command failed",
        version_exit=23,
    )

    assert result.returncode == 23
    assert "could not query Arduino CLI version" in result.stderr
    assert capture.read_text(encoding="utf-8") == "version\n"


def test_compile_helper_rejects_missing_cli_and_extra_arguments(tmp_path: Path) -> None:
    missing = subprocess.run(
        [str(COMPILE_HELPER)],
        cwd=tmp_path,
        env=os.environ | {"ARDUINO_CLI": "missing-arduino-cli"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 1
    assert "Arduino CLI is required" in missing.stderr

    extra = subprocess.run(
        [str(COMPILE_HELPER), "first", "second"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert extra.returncode == 2
    assert "usage:" in extra.stderr


def test_compile_helper_removes_only_its_own_temporary_build_directory(tmp_path: Path) -> None:
    result, _capture = _run_compile_helper(
        tmp_path,
        version_output="arduino-cli  Version: 1.5.1 Commit: fake",
    )

    assert result.returncode == 0, result.stderr
    match = re.search(r"--build-path (/[^\n]+)", result.stdout)
    assert match is not None
    assert not Path(match.group(1)).exists()
