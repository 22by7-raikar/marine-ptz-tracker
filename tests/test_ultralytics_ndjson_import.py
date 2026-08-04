from __future__ import annotations

import json
from pathlib import Path

import pytest

from marine_ptz.dataset import inspect_staging, plan_tugboat_dataset, validate_yolo_label
from tools import import_ultralytics_ndjson as importer


def _source_image(root: Path, session: str, filename: str) -> Path:
    path = root / session / "images" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"source-{session}-{filename}".encode())
    return path


def _metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "type": "dataset",
        "task": "detect",
        "class_names": {"0": "marine_target"},
    }
    metadata.update(overrides)
    return metadata


def _image_row(
    filename: str,
    *,
    boxes: object = ((0, 0.5, 0.5, 0.25, 0.25),),
    split: str = "train",
) -> dict[str, object]:
    return {
        "type": "image",
        "file": filename,
        "width": 1920,
        "height": 1080,
        "split": split,
        "annotations": {"boxes": boxes},
    }


def _write_ndjson(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_imports_only_annotated_rows_and_feeds_existing_splitter(tmp_path: Path) -> None:
    source = tmp_path / "original-staging"
    filenames = {
        "session-01": "session-01__clip__f0001.jpg",
        "session-02": "session-02__clip__f0002.jpg",
        "session-03": "session-03__clip__f0003.jpg",
    }
    source_paths = {
        session: _source_image(source, session, filename) for session, filename in filenames.items()
    }
    skipped_name = "session-01__clip__unannotated.jpg"
    _source_image(source, "session-01", skipped_name)
    export = tmp_path / "platform.ndjson"
    _write_ndjson(
        export,
        [
            _metadata(),
            _image_row(f"platform/path/{filenames['session-01']}"),
            _image_row(filenames["session-02"], split="val"),
            _image_row(filenames["session-03"], split="test"),
            _image_row(skipped_name, boxes=[]),
        ],
    )
    output = tmp_path / "platform-import"

    plan = importer.plan_import(export, source, output)
    importer.execute_import(plan)

    assert len(plan.items) == 3
    assert plan.session_count == 3
    assert plan.skipped_unannotated_rows == 1
    for session, filename in filenames.items():
        destination_image = output / session / "images" / filename
        destination_label = output / session / "labels" / f"{Path(filename).stem}.txt"
        assert destination_image.read_bytes() == source_paths[session].read_bytes()
        assert validate_yolo_label(destination_label)[0].class_id == 0
        assert source_paths[session].exists()
    assert not (output / "session-01" / "images" / skipped_name).exists()

    manifest = json.loads((output / importer.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["counts"] == {
        "failures": 0,
        "image_rows": 4,
        "imported_images": 3,
        "negative_images": 0,
        "positive_images": 3,
        "label_files": 3,
        "sessions": 3,
        "skipped_unannotated_rows": 1,
    }
    assert manifest["rows"][0]["local_source"].startswith("session-01/images/")
    assert {row["annotation_kind"] for row in manifest["rows"]} == {"positive"}
    assert not Path(manifest["rows"][0]["local_source"]).is_absolute()
    assert inspect_staging(output)[1] == ()
    assert len(plan_tugboat_dataset(output, tmp_path / "prepared").copies) == 3


def test_dry_run_validates_without_creating_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    filename = "session-01__frame.jpg"
    _source_image(source, "session-01", filename)
    export = tmp_path / "platform.ndjson"
    _write_ndjson(export, [_metadata(), _image_row(filename)])
    output = tmp_path / "imported"

    plan = importer.plan_import(export, source, output)
    importer.execute_import(plan, dry_run=True)

    assert output.exists() is False
    assert list(source.rglob("*"))


def test_cli_dry_run_reports_all_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    filename = "session-01__frame.jpg"
    _source_image(source, "session-01", filename)
    export = tmp_path / "platform.ndjson"
    _write_ndjson(
        export,
        [_metadata(), _image_row(filename), _image_row("unannotated.jpg", boxes=[])],
    )
    output = tmp_path / "imported"

    result = importer.main(
        [
            "--ndjson",
            str(export),
            "--source-staging-root",
            str(source),
            "--output-staging-root",
            str(output),
            "--dry-run",
        ]
    )

    assert result == 0
    assert output.exists() is False
    assert (
        capsys.readouterr()
        .out.strip()
        .endswith(
            "positive_images=1 negative_images=0 label_files=1 sessions=1 "
            "skipped_unannotated_rows=1 failures=0"
        )
    )


def test_cli_flag_explicitly_authorizes_reviewed_negative(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    _source_image(source, "session-01", "negative.jpg")
    export = tmp_path / "platform.ndjson"
    _write_ndjson(export, [_metadata(), _image_row("negative.jpg", boxes=[])])
    output = tmp_path / "imported"

    result = importer.main(
        [
            "--ndjson",
            str(export),
            "--source-staging-root",
            str(source),
            "--output-staging-root",
            str(output),
            "--include-unannotated-as-negatives",
            "--dry-run",
        ]
    )

    assert result == 0
    assert output.exists() is False
    assert "positive_images=0 negative_images=1" in capsys.readouterr().out


def test_opt_in_imports_reviewed_negatives_with_zero_byte_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    rows: list[dict[str, object]] = [_metadata()]
    expected_payloads: dict[str, bytes] = {}
    for session, filename, boxes in (
        ("session-01", "session-01__positive.jpg", [[0, 0.5, 0.5, 0.2, 0.2]]),
        ("session-02", "session-02__negative.jpg", []),
        ("session-03", "session-03__positive.jpg", [[0, 0.4, 0.4, 0.2, 0.2]]),
    ):
        source_path = _source_image(source, session, filename)
        expected_payloads[filename] = source_path.read_bytes()
        rows.append(_image_row(filename, boxes=boxes))
    export = tmp_path / "platform.ndjson"
    _write_ndjson(export, rows)
    output = tmp_path / "imported"

    plan = importer.plan_import(
        export,
        source,
        output,
        include_unannotated_as_negatives=True,
    )
    importer.execute_import(plan)

    assert plan.positive_count == 2
    assert plan.negative_count == 1
    assert plan.skipped_unannotated_rows == 0
    negative_label = output / "session-02" / "labels" / "session-02__negative.txt"
    assert negative_label.read_bytes() == b""
    assert validate_yolo_label(negative_label) == ()
    assert inspect_staging(output)[1] == ()
    assert len(plan_tugboat_dataset(output, tmp_path / "prepared").copies) == 3
    assert {path.name: path.read_bytes() for path in source.rglob("*.jpg")} == expected_payloads

    manifest = json.loads((output / importer.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["counts"]["positive_images"] == 2
    assert manifest["counts"]["negative_images"] == 1
    assert manifest["counts"]["label_files"] == 3
    assert manifest["counts"]["sessions"] == 3
    assert manifest["counts"]["skipped_unannotated_rows"] == 0
    assert manifest["counts"]["failures"] == 0
    assert [row["annotation_kind"] for row in manifest["rows"]] == [
        "positive",
        "reviewed_negative",
        "positive",
    ]


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (_metadata(class_names={"0": "boat"}), "class_names must be exactly"),
        (
            _metadata(class_names={"0": "marine_target", "1": "boat"}),
            "class_names must be exactly",
        ),
        (_metadata(task="segment"), "task must be exactly 'detect'"),
    ],
)
def test_rejects_invalid_metadata(
    tmp_path: Path,
    metadata: dict[str, object],
    message: str,
) -> None:
    source = tmp_path / "source"
    filename = "frame.jpg"
    _source_image(source, "session-01", filename)
    export = tmp_path / "platform.ndjson"
    _write_ndjson(export, [metadata, _image_row(filename)])

    with pytest.raises(importer.NdjsonImportError, match=message):
        importer.plan_import(export, source, tmp_path / "output")


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(class_names={"0": "boat"}),
        _metadata(task="segment"),
    ],
)
def test_negative_opt_in_does_not_weaken_metadata_validation(
    tmp_path: Path,
    metadata: dict[str, object],
) -> None:
    source = tmp_path / "source"
    _source_image(source, "session-01", "negative.jpg")
    export = tmp_path / "platform.ndjson"
    _write_ndjson(export, [metadata, _image_row("negative.jpg", boxes=[])])

    with pytest.raises(importer.NdjsonImportError):
        importer.plan_import(
            export,
            source,
            tmp_path / "output",
            include_unannotated_as_negatives=True,
        )


@pytest.mark.parametrize(
    ("boxes", "message"),
    [
        ([[1, 0.5, 0.5, 0.2, 0.2]], "class id must be exactly 0"),
        ([[0.0, 0.5, 0.5, 0.2, 0.2]], "class id must be exactly 0"),
        ([[0, 0.5, 0.5, 0.0, 0.2]], "width and height"),
        ([[0, 0.05, 0.5, 0.2, 0.2]], "extends outside"),
        ([[0, "bad", 0.5, 0.2, 0.2]], "coordinates must be numeric"),
        ([[0, 0.5, 0.5, 0.2]], "expected"),
    ],
)
def test_rejects_invalid_annotations(
    tmp_path: Path,
    boxes: object,
    message: str,
) -> None:
    source = tmp_path / "source"
    filename = "frame.jpg"
    _source_image(source, "session-01", filename)
    export = tmp_path / "platform.ndjson"
    _write_ndjson(export, [_metadata(), _image_row(filename, boxes=boxes)])

    with pytest.raises(importer.NdjsonImportError, match=message):
        importer.plan_import(export, source, tmp_path / "output")


def test_rejects_nonfinite_json_annotation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_image(source, "session-01", "frame.jpg")
    export = tmp_path / "platform.ndjson"
    export.write_text(
        json.dumps(_metadata())
        + "\n"
        + '{"type":"image","file":"frame.jpg","width":1920,"height":1080,'
        '"split":"train","annotations":{"boxes":[[0,NaN,0.5,0.2,0.2]]}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(importer.NdjsonImportError, match="non-finite JSON number NaN"):
        importer.plan_import(export, source, tmp_path / "output")


def test_negative_opt_in_rejects_unexpected_annotation_types(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_image(source, "session-01", "negative.jpg")
    export = tmp_path / "platform.ndjson"
    row = _image_row("negative.jpg", boxes=[])
    row["annotations"] = {"boxes": [], "segments": []}
    _write_ndjson(export, [_metadata(), row])

    with pytest.raises(importer.NdjsonImportError, match="unexpected annotation types"):
        importer.plan_import(
            export,
            source,
            tmp_path / "output",
            include_unannotated_as_negatives=True,
        )


def test_rejects_duplicate_exported_basenames(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_image(source, "session-01", "frame.jpg")
    export = tmp_path / "platform.ndjson"
    _write_ndjson(
        export,
        [
            _metadata(),
            _image_row("first/path/frame.jpg"),
            _image_row("second/path/frame.jpg"),
        ],
    )

    with pytest.raises(importer.NdjsonImportError, match="duplicate filename"):
        importer.plan_import(export, source, tmp_path / "output")


def test_rejects_missing_and_ambiguous_source_images(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_image(source, "session-01", "ambiguous.jpg")
    _source_image(source, "session-02", "ambiguous.jpg")
    export = tmp_path / "platform.ndjson"
    _write_ndjson(
        export,
        [_metadata(), _image_row("ambiguous.jpg"), _image_row("missing.jpg")],
    )

    with pytest.raises(importer.NdjsonImportError) as raised:
        importer.plan_import(export, source, tmp_path / "output")

    assert "ambiguous" in str(raised.value)
    assert "was not found" in str(raised.value)


def test_rejects_existing_output_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_image(source, "session-01", "frame.jpg")
    export = tmp_path / "platform.ndjson"
    _write_ndjson(export, [_metadata(), _image_row("frame.jpg")])
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(importer.NdjsonImportError, match="must be new"):
        importer.plan_import(export, source, output)


def test_negative_validation_failure_does_not_publish_positive_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_image(source, "session-01", "positive.jpg")
    export = tmp_path / "platform.ndjson"
    _write_ndjson(
        export,
        [
            _metadata(),
            _image_row("positive.jpg"),
            _image_row("missing-negative.jpg", boxes=[]),
        ],
    )
    output = tmp_path / "imported"

    with pytest.raises(importer.NdjsonImportError, match="was not found"):
        importer.plan_import(
            export,
            source,
            output,
            include_unannotated_as_negatives=True,
        )

    assert output.exists() is False
    assert (source / "session-01" / "images" / "positive.jpg").exists()


def test_count_mismatch_does_not_publish_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _source_image(source, "session-01", "frame.jpg")
    export = tmp_path / "platform.ndjson"
    _write_ndjson(export, [_metadata(), _image_row("frame.jpg")])
    output = tmp_path / "imported"
    plan = importer.plan_import(export, source, output)
    monkeypatch.setattr(importer.shutil, "copy2", lambda _source, _destination: None)

    with pytest.raises(importer.NdjsonImportError, match="count mismatch"):
        importer.execute_import(plan)

    assert output.exists() is False
    assert not tuple(tmp_path.glob(".imported.import-*"))
