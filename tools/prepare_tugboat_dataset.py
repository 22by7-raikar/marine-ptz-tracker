#!/usr/bin/env python3
"""Validate and split labeled tugboat frames by complete recording session."""

# ruff: noqa: E402, I001 -- add the src layout before importing the local package.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_ptz.dataset import (
    create_tugboat_data_yaml,
    DatasetPreparationError,
    execute_dataset_plan,
    plan_tugboat_dataset,
    validate_tugboat_data_yaml,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("data/tugboat/staging"),
        help="input tree containing SESSION/images and SESSION/labels",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/tugboat"))
    parser.add_argument("--data", type=Path, default=Path("data/tugboat/data.yaml"))
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        data_exists = args.data.exists()
        definition = validate_tugboat_data_yaml(args.data) if data_exists else None
        data_parent = definition.yaml_path.parent if definition is not None else args.data.parent
        if data_parent.resolve() != args.output_root.resolve():
            raise DatasetPreparationError(
                "--data must belong to --output-root so split paths remain coherent"
            )
        plan = plan_tugboat_dataset(
            args.staging_root,
            args.output_root,
            seed=args.seed,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )
        execute_dataset_plan(plan, dry_run=args.dry_run)
        if not args.dry_run and not data_exists:
            create_tugboat_data_yaml(args.data)
    except (DatasetPreparationError, OSError, ValueError) as exc:
        failures = max(1, len(str(exc).splitlines()))
        print(f"dataset preparation failed validation_failures={failures}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    mode = "planned" if args.dry_run else "copied"
    for split in ("train", "val", "test"):
        print(
            f"split={split} sessions={plan.session_count(split)} "
            f"images={plan.sample_count(split)} labels={plan.sample_count(split)}"
        )
    print(
        f"dataset {mode} total_sessions={len(plan.session_splits)} "
        f"total_images={len(plan.copies)} total_labels={len(plan.copies)} "
        f"data_yaml={'validated' if data_exists else ('planned' if args.dry_run else 'created')} "
        "validation_failures=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
