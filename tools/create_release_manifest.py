#!/usr/bin/env python3
"""Create one strict atomic manifest for a release candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_ptz.release_manifest import (  # noqa: E402
    ReleaseManifestError,
    ReleaseManifestInputs,
    create_release_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--demonstration-video", type=Path)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--expected-target-class", default="marine_target")
    parser.add_argument("--physical-validation-claimed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = create_release_manifest(
            ReleaseManifestInputs(
                repository=args.repository,
                config=args.config,
                model=args.model,
                evaluation_report=args.evaluation_report,
                benchmark_report=args.benchmark_report,
                demonstration_video=args.demonstration_video,
                output=args.output,
                image_tag=args.image_tag,
                expected_target_class=args.expected_target_class,
                physical_validation_claimed=args.physical_validation_claimed,
                overwrite=args.overwrite,
            )
        )
    except (ReleaseManifestError, OSError, ValueError) as exc:
        print(f"release manifest error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": args.output.name,
                "git_commit": payload["application"]["git_commit"],  # type: ignore[index]
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
