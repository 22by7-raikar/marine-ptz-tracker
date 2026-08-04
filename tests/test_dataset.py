from __future__ import annotations

from pathlib import Path

import pytest

from marine_ptz.dataset import (
    DatasetPreparationError,
    deterministic_session_split,
    execute_dataset_plan,
    inspect_staging,
    plan_tugboat_dataset,
    validate_tugboat_data_yaml,
    validate_yolo_label,
)


def _session(
    root: Path,
    name: str,
    *,
    label: str = "0 0.5 0.5 0.4 0.2\n",
) -> None:
    images = root / name / "images"
    labels = root / name / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "frame.jpg").write_bytes(f"image-{name}".encode())
    (labels / "frame.txt").write_text(label, encoding="utf-8")


def test_checked_in_data_yaml_defines_exact_marine_target_split_paths() -> None:
    definition = validate_tugboat_data_yaml(Path("data/tugboat/data.yaml"))

    assert definition.train_images == Path("data/tugboat/images/train")
    assert definition.val_images == Path("data/tugboat/images/val")
    assert definition.test_images == Path("data/tugboat/images/test")
    assert definition.train_labels == Path("data/tugboat/labels/train")
    assert definition.val_labels == Path("data/tugboat/labels/val")
    assert definition.test_labels == Path("data/tugboat/labels/test")


@pytest.mark.parametrize(
    "names",
    ["names: [boat]", "names: [marine_target, boat]", "names: {1: marine_target}"],
)
def test_data_yaml_rejects_any_schema_other_than_one_marine_target(
    tmp_path: Path,
    names: str,
) -> None:
    data = tmp_path / "data.yaml"
    data.write_text(
        f"train: images/train\nval: images/val\ntest: images/test\n{names}\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetPreparationError, match="exactly one class"):
        validate_tugboat_data_yaml(data)


def test_yolo_label_accepts_class_zero_normalized_boxes_and_empty_negatives(
    tmp_path: Path,
) -> None:
    populated = tmp_path / "populated.txt"
    populated.write_text("0 0.5 0.5 1 1\n0 0.25 0.25 0.5 0.5\n", encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")

    boxes = validate_yolo_label(populated)

    assert len(boxes) == 2
    assert boxes[0].class_id == 0
    assert validate_yolo_label(empty) == ()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("1 0.5 0.5 0.2 0.2", "class id must be exactly 0"),
        ("0.0 0.5 0.5 0.2 0.2", "class id must be exactly 0"),
        ("0 0.5 0.5 0.2", "expected"),
        ("0 nope 0.5 0.2 0.2", "numeric"),
        ("0 nan 0.5 0.2 0.2", "finite"),
        ("0 inf 0.5 0.2 0.2", "finite"),
        ("0 -0.1 0.5 0.2 0.2", "normalized"),
        ("0 0.5 1.1 0.2 0.2", "normalized"),
        ("0 0.5 0.5 0 0.2", "width and height"),
        ("0 0.5 0.5 1.1 0.2", "width and height"),
        ("0 0.05 0.5 0.2 0.2", "extends outside"),
        ("0 0.5 0.95 0.2 0.2", "extends outside"),
    ],
)
def test_yolo_label_rejects_invalid_class_and_coordinates(
    tmp_path: Path,
    line: str,
    message: str,
) -> None:
    label = tmp_path / "invalid.txt"
    label.write_text(f"{line}\n", encoding="utf-8")

    with pytest.raises(DatasetPreparationError, match=message):
        validate_yolo_label(label)


def test_ten_sessions_split_deterministically_without_session_leakage(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "dataset"
    for index in range(1, 11):
        _session(staging, f"session-{index:02d}")

    first = plan_tugboat_dataset(staging, output, seed=42)
    second = plan_tugboat_dataset(staging, output, seed=42)

    assert first == second
    assert first.session_count("train") == 7
    assert first.session_count("val") == 2
    assert first.session_count("test") == 1
    assert {session for session, _ in first.session_splits} == {
        f"session-{index:02d}" for index in range(1, 11)
    }
    for session, split in first.session_splits:
        assert {copy.split for copy in first.copies if copy.session == session} == {split}

    execute_dataset_plan(first)

    for split, expected in (("train", 7), ("val", 2), ("test", 1)):
        assert len(list((output / "images" / split).glob("*.jpg"))) == expected
        assert len(list((output / "labels" / split).glob("*.txt"))) == expected
    assert all(
        (staging / f"session-{index:02d}" / "images" / "frame.jpg").exists()
        for index in range(1, 11)
    )


def test_different_seed_changes_assignment_but_not_split_counts() -> None:
    sessions = [f"session-{index:02d}" for index in range(1, 11)]

    first = deterministic_session_split(sessions, seed=1)
    second = deterministic_session_split(sessions, seed=2)

    assert first != second
    for assignments in (first, second):
        assert list(assignments.values()).count("train") == 7
        assert list(assignments.values()).count("val") == 2
        assert list(assignments.values()).count("test") == 1


def test_staging_reports_missing_or_orphan_and_invalid_labels(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _session(staging, "session-valid")
    _session(staging, "session-invalid", label="2 0.5 0.5 0.2 0.2\n")
    missing = staging / "session-missing"
    (missing / "images").mkdir(parents=True)
    (missing / "labels").mkdir()
    (missing / "images" / "frame.jpg").write_bytes(b"missing-label")
    orphan = staging / "session-orphan"
    (orphan / "images").mkdir(parents=True)
    (orphan / "labels").mkdir()
    (orphan / "images" / "frame.jpg").write_bytes(b"paired")
    (orphan / "labels" / "frame.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (orphan / "labels" / "extra.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    _, issues = inspect_staging(staging)
    rendered = "\n".join(str(issue) for issue in issues)

    assert "class id must be exactly 0" in rendered
    assert "matching label file is missing" in rendered
    assert "label has no matching image" in rendered


def test_dry_run_validates_without_creating_dataset_directories(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "dataset"
    for index in range(3):
        _session(staging, f"session-{index}")
    plan = plan_tugboat_dataset(staging, output)

    execute_dataset_plan(plan, dry_run=True)

    assert output.exists() is False


def test_existing_prepared_sample_is_never_overwritten(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "dataset"
    for index in range(3):
        _session(staging, f"session-{index}")
    plan = plan_tugboat_dataset(staging, output)
    existing = plan.copies[0].destination_image
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"operator-owned")

    with pytest.raises(DatasetPreparationError, match="destinations already exist"):
        execute_dataset_plan(plan)

    assert existing.read_bytes() == b"operator-owned"
    assert not any(copy.destination_label.exists() for copy in plan.copies)
