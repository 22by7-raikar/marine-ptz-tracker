from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import conftest as project_conftest
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CHECKOUT_SHA = "d23441a48e516b6c34aea4fa41551a30e30af803"
SETUP_PYTHON_SHA = "ece7cb06caefa5fff74198d8649806c4678c61a1"
ARDUINO_SETUP_CLI_SHA = "81d310742121c928ea9c8bbd407b4217b432ae02"
COMPILE_HELPER = ROOT / "tools" / "compile_arduino.sh"


def test_ci_workflow_has_hardware_free_base_dev_job() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert set(workflow["on"]) == {"pull_request", "push", "workflow_dispatch"}
    assert workflow["on"]["pull_request"]["branches"] == ["main"]
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    assert workflow["concurrency"]["group"] == "ci-${{ github.workflow }}-${{ github.ref }}"
    job = workflow["jobs"]["test"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == "15"
    assert job["strategy"]["matrix"]["python-version"] == ["3.11", "3.10"]

    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "pip install --constraint constraints/ci.txt setuptools" in commands
    assert "--no-build-isolation" in commands
    assert '--editable ".[dev]"' in commands
    assert "compileall -q src tests tools" in commands
    assert "tools/smoke_check.py" in commands
    assert "ruff format --check src tests tools" in commands
    assert "ruff check src tests tools" in commands
    assert "python -m pytest" in commands
    assert "python -m pip check" in commands
    assert "git diff --check" in commands
    assert ".[vision]" not in commands
    assert ".[hardware]" not in commands


def test_ci_workflow_builds_and_runs_cpu_only_container_target() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["container-test"]

    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == "20"
    checkout, build, run = job["steps"]
    assert checkout["uses"] == f"actions/checkout@{CHECKOUT_SHA}"
    assert checkout["with"] == {"persist-credentials": "false"}
    assert build["run"] == "docker compose --env-file /dev/null -f compose.yaml build test"
    assert run["run"] == "docker compose --env-file /dev/null -f compose.yaml run --rm test"


def test_ci_workflow_uses_pip_cache() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["test"]["steps"]
    checkout = steps[0]
    setup_python = steps[1]

    assert checkout["uses"] == f"actions/checkout@{CHECKOUT_SHA}"
    assert checkout["with"] == {"persist-credentials": "false"}
    assert setup_python["uses"] == f"actions/setup-python@{SETUP_PYTHON_SHA}"
    assert setup_python["with"]["cache"] == "pip"
    assert setup_python["with"]["cache-dependency-path"].splitlines() == [
        "pyproject.toml",
        "constraints/ci.txt",
        "constraints/vision-cu128.txt",
    ]


def test_ci_workflow_has_pinned_compile_only_uno_firmware_job() -> None:
    raw_workflow = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(raw_workflow, Loader=yaml.BaseLoader)
    job = workflow["jobs"]["firmware"]

    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == "10"
    checkout, setup_cli, prerequisites, compile_step = job["steps"]
    assert checkout["uses"] == f"actions/checkout@{CHECKOUT_SHA}"
    assert checkout["with"] == {"persist-credentials": "false"}
    assert setup_cli["uses"] == f"arduino/setup-arduino-cli@{ARDUINO_SETUP_CLI_SHA}"
    assert setup_cli["with"] == {"version": "1.5.1"}
    assert (
        f"arduino/setup-arduino-cli@{ARDUINO_SETUP_CLI_SHA} "
        "# arduino/setup-arduino-cli v2.0.0, verified upstream commit"
    ) in raw_workflow
    assert f"actions/checkout@{CHECKOUT_SHA} # v6.1.0" in raw_workflow
    assert f"actions/setup-python@{SETUP_PYTHON_SHA} # v6.3.0" in raw_workflow

    commands = "\n".join((prerequisites["run"], compile_step["run"]))
    assert "arduino-cli core update-index" in commands
    assert "arduino-cli core install arduino:avr@1.8.8" in commands
    assert "arduino-cli lib install Servo@1.3.0" in commands
    assert "tools/compile_arduino.sh" in commands
    assert "${RUNNER_TEMP}" in commands
    assert "firmware/ptz_controller" not in prerequisites["run"]
    assert "upload" not in commands.casefold()
    assert "serial" not in commands.casefold()
    assert "board list" not in commands.casefold()
    assert "secrets" not in raw_workflow.casefold()
    assert workflow["permissions"] == {"contents": "read"}
    helper_source = COMPILE_HELPER.read_text(encoding="utf-8")
    assert 'FQBN="arduino:avr:uno"' in helper_source
    assert "--warnings all" in helper_source
    assert "firmware/ptz_controller" in helper_source
    for workflow_job in workflow["jobs"].values():
        for step in workflow_job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])
            if step.get("uses") == f"actions/checkout@{CHECKOUT_SHA}":
                assert step["with"] == {"persist-credentials": "false"}


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


class _FakeItem:
    def __init__(self, markers: set[str]) -> None:
        self.markers = markers
        self.added: list[str] = []

    def get_closest_marker(self, name: str) -> object | None:
        return object() if name in self.markers else None

    def add_marker(self, marker: pytest.Mark) -> None:
        self.added.append(marker.name)


@pytest.mark.parametrize("category", ["integration", "hardware", "gpu"])
def test_explicit_category_is_not_reclassified_as_unit(category: str) -> None:
    item = _FakeItem({category})

    project_conftest.pytest_collection_modifyitems([item])  # type: ignore[list-item]

    assert item.added == []


def test_unmarked_test_is_classified_as_unit() -> None:
    item = _FakeItem(set())

    project_conftest.pytest_collection_modifyitems([item])  # type: ignore[list-item]

    assert item.added == ["unit"]


def test_unknown_marker_fails_collection_in_clean_subprocess(tmp_path: Path) -> None:
    candidate = tmp_path / "test_unknown_marker.py"
    candidate.write_text(
        "import pytest\n"
        "@pytest.mark.hardward\n"
        "def test_never_runs():\n"
        "    raise AssertionError('unknown marker was executed')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-m",
            "unit",
            "-c",
            str(ROOT / "pyproject.toml"),
            str(candidate),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "hardward" in output
    assert "not found in `markers` configuration option" in output
