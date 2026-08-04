#!/usr/bin/env python3
"""Compile and import the package without installing dependencies or using hardware."""

from __future__ import annotations

import compileall
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def main() -> int:
    if sys.version_info < (3, 10):
        print("Python 3.10 or newer is required", file=sys.stderr)
        return 2
    if not compileall.compile_dir(SRC, quiet=1):
        return 1
    sys.path.insert(0, str(SRC))
    import marine_ptz  # noqa: F401

    print("marine_ptz compile and import smoke check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
