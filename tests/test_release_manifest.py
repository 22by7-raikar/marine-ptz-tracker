from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

import marine_ptz.release_manifest as release_manifest
from marine_ptz.release_manifest import (
    ReleaseManifestError,
    ReleaseManifestInputs,
    create_release_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )


def _inputs(tmp_path: Path, *, overwrite: bool = False) -> ReleaseManifestInputs:
    private = tmp_path / "home" / "private-user" / "release"
    private.mkdir(parents=True)
    config = private / "hardware.yaml"
    config.write_bytes((ROOT / "configs" / "hardware.yaml").read_bytes())
    model = private / "best.pt"
    model.write_bytes(b"deterministic-model")
    evaluation = private / "evaluation.json"
    evaluation.write_text('{"map50": 0.75}\n', encoding="utf-8")
    benchmark = private / "benchmark.json"
    benchmark.write_text('{"p95_ms": 55.01}\n', encoding="utf-8")
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "config", "user.name", "Test Operator")
    (repository / "tracked.txt").write_text("known\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-q", "-m", "known release")
    return ReleaseManifestInputs(
        repository=repository,
        config=config,
        model=model,
        evaluation_report=evaluation,
        benchmark_report=benchmark,
        output=private / "manifest.json",
        image_tag="registry.invalid/marine-ptz:git-deadbeef",
        overwrite=overwrite,
    )


def test_manifest_records_hashes_clean_git_and_no_private_paths(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    payload = create_release_manifest(
        inputs,
        timestamp=lambda: datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        application_version=lambda: "0.1.0",
    )

    serialized = inputs.output.read_text(encoding="utf-8")
    decoded = json.loads(serialized)
    assert decoded == payload
    assert payload["application"]["worktree_state"] == "clean"
    assert payload["application"]["worktree_dirty"] is False
    assert payload["evidence"]["model"] == {
        "filename": "best.pt",
        "sha256": hashlib.sha256(b"deterministic-model").hexdigest(),
    }
    assert payload["physical_validation_claimed"] is False
    assert payload["runtime"]["expected_target_class"] == "marine_target"
    assert payload["safety"]["logical_pan_limits_deg"] == [75.0, 105.0]
    for private_fragment in (str(tmp_path), "private-user", "/home/", str(inputs.repository)):
        assert private_fragment not in serialized


def test_manifest_reports_dirty_git_without_listing_changed_paths(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    (inputs.repository / "tracked.txt").write_text("changed\n", encoding="utf-8")

    payload = create_release_manifest(inputs, application_version=lambda: "0.1.0")

    assert payload["application"]["worktree_state"] == "dirty"
    assert payload["application"]["worktree_dirty"] is True
    assert "tracked.txt" not in inputs.output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "filename,contents",
    [
        ("evaluation.json", "not json"),
        ("benchmark.json", '{"latency": NaN}'),
        ("benchmark.json", '{"latency": 1, "latency": 2}'),
    ],
)
def test_manifest_rejects_malformed_or_non_strict_reports(
    tmp_path: Path, filename: str, contents: str
) -> None:
    inputs = _inputs(tmp_path)
    (inputs.config.parent / filename).write_text(contents, encoding="utf-8")

    with pytest.raises(ReleaseManifestError, match="strict JSON"):
        create_release_manifest(inputs, application_version=lambda: "0.1.0")
    assert not inputs.output.exists()


def test_manifest_rejects_missing_and_non_regular_inputs(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs.model.unlink()
    with pytest.raises(ReleaseManifestError, match="model is unavailable"):
        create_release_manifest(inputs, application_version=lambda: "0.1.0")

    inputs = _inputs(tmp_path / "second")
    inputs.evaluation_report.unlink()
    inputs.evaluation_report.mkdir()
    with pytest.raises(ReleaseManifestError, match="regular file"):
        create_release_manifest(inputs, application_version=lambda: "0.1.0")


def test_manifest_atomic_overwrite_policy_preserves_existing_file(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs.output.write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(ReleaseManifestError, match="use --overwrite"):
        create_release_manifest(inputs, application_version=lambda: "0.1.0")
    assert inputs.output.read_text(encoding="utf-8") == "operator-owned\n"
    assert list(inputs.output.parent.glob(".manifest.json.*.tmp")) == []

    overwritten = replace(inputs, overwrite=True)
    create_release_manifest(overwritten, application_version=lambda: "0.1.0")
    assert json.loads(inputs.output.read_text(encoding="utf-8"))["schema_version"] == 1
    assert list(inputs.output.parent.glob(".manifest.json.*.tmp")) == []


@pytest.mark.parametrize(
    "image_tag",
    [
        "https://user:pass@example.invalid/marine-ptz:latest",
        "user:pass@example.invalid/marine-ptz:latest",
        "marine-ptz:latest?token=secret",
    ],
)
def test_manifest_rejects_credential_or_url_shaped_image_tags(
    tmp_path: Path, image_tag: str
) -> None:
    inputs = _inputs(tmp_path)

    with pytest.raises(ReleaseManifestError, match="image tag"):
        create_release_manifest(
            replace(inputs, image_tag=image_tag), application_version=lambda: "0.1.0"
        )


def test_manifest_redacts_an_unsafe_evidence_basename(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    unsafe_model = inputs.model.with_name("best.pt?token=secret")
    inputs.model.rename(unsafe_model)

    payload = create_release_manifest(
        replace(inputs, model=unsafe_model), application_version=lambda: "0.1.0"
    )

    assert payload["evidence"]["model"]["filename"] == "<redacted>"
    serialized = inputs.output.read_text(encoding="utf-8")
    assert "token" not in serialized
    assert "secret" not in serialized


def test_failed_atomic_replace_preserves_previous_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, overwrite=True)
    inputs.output.write_text("previous manifest\n", encoding="utf-8")
    monkeypatch.setattr(
        release_manifest.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        create_release_manifest(inputs, application_version=lambda: "0.1.0")

    assert inputs.output.read_text(encoding="utf-8") == "previous manifest\n"
    assert list(inputs.output.parent.glob(".manifest.json.*.tmp")) == []
