from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


OPTIONAL_ROOTS = ("numpy", "cv2", "torch", "torchvision", "ultralytics")


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path("src").resolve())
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _run_subprocess(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=_environment(),
    )


def test_package_import_and_synthetic_demo_do_not_import_vision_packages() -> None:
    optional_names = repr(OPTIONAL_ROOTS)
    result = _run_subprocess(
        f"""
import sys
optional = {optional_names}
import marine_ptz
import marine_ptz.benchmark
assert not (set(optional) & set(sys.modules)), sorted(set(optional) & set(sys.modules))
from marine_ptz.demo import main
assert main([
    "--config", "configs/development.yaml",
    "--steps", "2",
    "--no-sleep",
]) == 0
assert not (set(optional) & set(sys.modules)), sorted(set(optional) & set(sys.modules))
"""
    )

    assert result.returncode == 0, result.stderr


def test_vision_components_report_actionable_missing_dependency_errors() -> None:
    result = _run_subprocess(
        """
import importlib.abc
import sys

blocked = {"cv2", "torch", "torchvision", "ultralytics", "numpy"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ModuleNotFoundError(f"blocked optional dependency: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())

from marine_ptz.opencv_source import OpenCVDependencyError, OpenCVSource
from marine_ptz.yolo_detector import UltralyticsDependencyError, UltralyticsDetector

try:
    OpenCVSource("video.mp4")
except OpenCVDependencyError as exc:
    assert "constrained vision dependencies" in str(exc)
else:
    raise AssertionError("OpenCVSource did not report its missing dependency")

try:
    UltralyticsDetector("local-only.pt", device="cpu")
except UltralyticsDependencyError as exc:
    assert "constrained vision dependencies" in str(exc)
else:
    raise AssertionError("UltralyticsDetector did not report its missing dependency")
"""
    )

    assert result.returncode == 0, result.stderr
