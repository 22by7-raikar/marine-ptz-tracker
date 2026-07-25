from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import conftest as project_conftest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CHECKOUT_SHA = "d23441a48e516b6c34aea4fa41551a30e30af803"
SETUP_PYTHON_SHA = "ece7cb06caefa5fff74198d8649806c4678c61a1"


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
    assert (
        'python -m pip install --constraint constraints/ci.txt --editable ".[dev]"'
        in commands
    )
    assert "compileall -q src tests tools" in commands
    assert "tools/smoke_check.py" in commands
    assert 'pytest -m "not hardware and not gpu"' in commands
    assert "python -m pip check" in commands
    assert ".[vision]" not in commands


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
