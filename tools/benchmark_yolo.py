#!/usr/bin/env python3
"""Run the camera-free Ultralytics benchmark from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_ptz.benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
