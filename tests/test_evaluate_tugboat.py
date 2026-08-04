from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tools.evaluate_tugboat as evaluation


class FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.detached = False
        self.on_cpu = False

    def detach(self) -> FakeTensor:
        self.detached = True
        return self

    def cpu(self) -> FakeTensor:
        self.on_cpu = True
        return self

    def tolist(self) -> list[float]:
        assert self.detached
        assert self.on_cpu
        return list(self.values)


class FakeModel:
    def __init__(self, negative_confidences: list[list[float]]) -> None:
        self.negative_confidences = negative_confidences
        self.predict_calls: list[dict[str, Any]] = []
        self.val_calls: list[dict[str, Any]] = []

    def val(self, **kwargs: object) -> object:
        self.val_calls.append(kwargs)
        return SimpleNamespace(box=SimpleNamespace(mp=0.998, mr=1.0, map50=0.995, map=0.939))

    def predict(self, **kwargs: Any) -> list[object]:
        self.predict_calls.append(kwargs)
        source = kwargs["source"]
        if isinstance(source, str):
            return [object(), object(), object()]
        if kwargs.get("save") is False:
            return [
                SimpleNamespace(boxes=SimpleNamespace(conf=FakeTensor(values)))
                for values in self.negative_confidences
            ]
        return [object() for _ in source]


def make_dataset(
    tmp_path: Path,
    *,
    include_negatives: bool = True,
) -> tuple[Path, Path]:
    dataset = tmp_path / "tugboat-v3"
    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    (dataset / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n\nnames:\n  - marine_target\n",
        encoding="utf-8",
    )
    samples = [("positive.jpg", "0 0.5 0.5 0.2 0.2\n")]
    if include_negatives:
        samples.extend((("negative-hand.jpg", ""), ("negative-wallet.jpg", "")))
    for name, label in samples:
        (dataset / "images" / "test" / name).write_bytes(b"fixture")
        (dataset / "labels" / "test" / Path(name).with_suffix(".txt")).write_text(
            label,
            encoding="utf-8",
        )
    model = tmp_path / "best.pt"
    model.write_bytes(b"fixture")
    return dataset / "data.yaml", model


def test_evaluation_reports_reviewed_negative_false_positives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, model_path = make_dataset(tmp_path)
    model = FakeModel([[0.2], [0.77, 0.1]])
    monkeypatch.setattr(evaluation, "_load_yolo", lambda: lambda path: model)
    report_path = tmp_path / "evidence" / "report.json"

    result = evaluation.main(
        [
            "--model",
            str(model_path),
            "--data",
            str(data),
            "--device",
            "cpu",
            "--confidence",
            "0.60",
            "--project",
            str(tmp_path / "runs"),
            "--name",
            "v3-test",
            "--report",
            str(report_path),
        ]
    )

    assert result == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["held_out_composition"] == {
        "positive_images": 1,
        "reviewed_negative_images": 2,
    }
    assert report["reviewed_negative_metrics"]["false_positive_images"] == 1
    assert report["reviewed_negative_metrics"]["evaluation_performed"] is True
    assert report["reviewed_negative_metrics"]["status"] == "measured"
    assert report["reviewed_negative_metrics"]["false_positive_image_rate"] == 0.5
    assert report["reviewed_negative_metrics"]["maximum_confidence"] == 0.77
    assert report["confidence_threshold"] == 0.6
    assert model.predict_calls[1]["conf"] == 0.0
    assert model.predict_calls[1]["save"] is False
    assert model.predict_calls[2]["conf"] == 0.6
    assert model.predict_calls[2]["save"] is True
    assert len(model.predict_calls[2]["source"]) == 1
    assert model.val_calls[0]["batch"] == 8


def test_evaluation_skips_false_positive_artifacts_when_negatives_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, model_path = make_dataset(tmp_path)
    model = FakeModel([[], [0.4]])
    monkeypatch.setattr(evaluation, "_load_yolo", lambda: lambda path: model)
    report_path = tmp_path / "report.json"

    assert (
        evaluation.main(
            [
                "--model",
                str(model_path),
                "--data",
                str(data),
                "--device",
                "cpu",
                "--confidence",
                "0.60",
                "--project",
                str(tmp_path / "runs"),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["reviewed_negative_metrics"]["false_positive_images"] == 0
    assert report["reviewed_negative_metrics"]["false_positive_image_rate"] == 0.0
    assert report["reviewed_negative_metrics"]["annotated_false_positive_directory"] is None
    assert len(model.predict_calls) == 2


def test_evaluation_reports_no_reviewed_negative_measurement_as_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data, model_path = make_dataset(tmp_path, include_negatives=False)
    model = FakeModel([])
    monkeypatch.setattr(evaluation, "_load_yolo", lambda: lambda path: model)
    report_path = tmp_path / "report.json"

    assert (
        evaluation.main(
            [
                "--model",
                str(model_path),
                "--data",
                str(data),
                "--device",
                "cpu",
                "--confidence",
                "0.60",
                "--project",
                str(tmp_path / "runs"),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["held_out_composition"] == {
        "positive_images": 1,
        "reviewed_negative_images": 0,
    }
    assert report["reviewed_negative_metrics"] == {
        "evaluation_performed": False,
        "status": "not_measured_no_reviewed_negative_images",
        "false_positive_images": None,
        "false_positive_image_rate": None,
        "maximum_confidence": None,
        "annotated_false_positive_directory": None,
    }
    assert len(model.predict_calls) == 1
    assert "negative_false_positive_rate=not_measured" in capsys.readouterr().out


def test_evaluation_batch_auto_is_rejected_with_actionable_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        evaluation.main(["--batch", "auto"])

    assert raised.value.code == 2
    assert "does not support auto" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_evaluation_batch_requires_positive_integer(
    value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        evaluation.main(["--batch", value])

    assert raised.value.code == 2
    assert "positive integer" in capsys.readouterr().err


def test_evaluation_rejects_missing_test_label_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data, model_path = make_dataset(tmp_path)
    missing = data.parent / "labels" / "test" / "negative-hand.txt"
    missing.unlink()
    loaded = False

    def load() -> object:
        nonlocal loaded
        loaded = True
        return object()

    monkeypatch.setattr(evaluation, "_load_yolo", load)

    assert evaluation.main(["--model", str(model_path), "--data", str(data)]) == 2
    assert loaded is False
    assert "has no label file" in capsys.readouterr().err


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_confidence_values_from_fake_gpu_tensors_are_strict(value: float) -> None:
    with pytest.raises(evaluation.TugboatEvaluationError, match="confidence"):
        evaluation._confidence_values(
            SimpleNamespace(boxes=SimpleNamespace(conf=FakeTensor([value])))
        )
