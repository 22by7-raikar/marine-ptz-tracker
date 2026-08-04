from __future__ import annotations

from pathlib import Path

from marine_ptz.dataset import TUGBOAT_DATA_YAML, validate_tugboat_data_yaml
from tools.prepare_tugboat_dataset import main


def _staging(root: Path) -> Path:
    for index in range(1, 4):
        session = root / f"session-{index:02d}"
        images = session / "images"
        labels = session / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        (images / f"frame-{index}.jpg").write_bytes(f"image-{index}".encode())
        label = "" if index == 2 else "0 0.5 0.5 0.2 0.2\n"
        (labels / f"frame-{index}.txt").write_text(label, encoding="utf-8")
    return root


def _arguments(staging: Path, output: Path, *, dry_run: bool = False) -> list[str]:
    arguments = [
        "--staging-root",
        str(staging),
        "--output-root",
        str(output),
        "--data",
        str(output / "data.yaml"),
    ]
    if dry_run:
        arguments.append("--dry-run")
    return arguments


def test_dry_run_plans_missing_versioned_yaml_without_writing(tmp_path: Path) -> None:
    staging = _staging(tmp_path / "platform-imported-v2")
    output = tmp_path / "tugboat-v2"

    assert main(_arguments(staging, output, dry_run=True)) == 0

    assert output.exists() is False
    assert not (output / "data.yaml").exists()


def test_preparation_creates_v2_local_yaml_and_preserves_v1(tmp_path: Path) -> None:
    staging = _staging(tmp_path / "platform-imported-v2")
    output = tmp_path / "tugboat-v2"
    v1 = tmp_path / "tugboat"
    v1.mkdir()
    v1_marker = v1 / "do-not-touch.txt"
    v1_marker.write_text("v1", encoding="utf-8")

    assert main(_arguments(staging, output)) == 0

    definition = validate_tugboat_data_yaml(output / "data.yaml")
    assert (output / "data.yaml").read_text(encoding="utf-8") == TUGBOAT_DATA_YAML
    assert definition.train_images == output / "images" / "train"
    assert definition.val_images == output / "images" / "val"
    assert definition.test_images == output / "images" / "test"
    assert definition.train_labels == output / "labels" / "train"
    assert definition.val_labels == output / "labels" / "val"
    assert definition.test_labels == output / "labels" / "test"
    assert v1_marker.read_text(encoding="utf-8") == "v1"
    assert any(path.stat().st_size == 0 for path in output.rglob("*.txt"))


def test_existing_versioned_yaml_is_validated_without_overwrite(tmp_path: Path) -> None:
    staging = _staging(tmp_path / "platform-imported-v2")
    output = tmp_path / "tugboat-v2"
    output.mkdir()
    data = output / "data.yaml"
    custom_formatting = TUGBOAT_DATA_YAML + "# reviewed v2 schema\n"
    data.write_text(custom_formatting, encoding="utf-8")

    assert main(_arguments(staging, output, dry_run=True)) == 0

    assert data.read_text(encoding="utf-8") == custom_formatting


def test_data_yaml_outside_output_root_is_rejected_without_writes(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path / "platform-imported-v2")
    output = tmp_path / "tugboat-v2"
    v1_data = tmp_path / "tugboat" / "data.yaml"

    result = main(
        [
            "--staging-root",
            str(staging),
            "--output-root",
            str(output),
            "--data",
            str(v1_data),
            "--dry-run",
        ]
    )

    assert result == 2
    assert output.exists() is False
    assert v1_data.exists() is False
