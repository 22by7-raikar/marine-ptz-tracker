#!/usr/bin/env python3
"""Run a read-only deployment preflight without constructing runtime devices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_ptz.preflight import PreflightOptions, run_preflight  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("test", "replay", "hardware"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input-video", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--target-class")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runtime-mode", default="single")
    parser.add_argument("--camera-path", type=Path)
    parser.add_argument("--serial-path", type=Path)
    parser.add_argument("--arm-hardware", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_preflight(
        PreflightOptions(
            mode=args.mode,
            config=args.config,
            model=args.model,
            input_video=args.input_video,
            output_directory=args.output_directory,
            target_class=args.target_class,
            confidence=args.confidence,
            device=args.device,
            runtime_mode=args.runtime_mode,
            camera_path=args.camera_path,
            serial_path=args.serial_path,
            hardware_armed=args.arm_hardware,
        )
    )
    if args.json:
        print(json.dumps(report, allow_nan=False, sort_keys=True))
    else:
        for check in report["checks"]:
            state = "PASS" if check["ok"] else "FAIL"
            print(f"{state} {check['name']}: {check['detail']}")
        print("preflight passed" if report["ok"] else "preflight failed")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
