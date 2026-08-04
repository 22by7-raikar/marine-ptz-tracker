from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEST_COMPOSE = ROOT / "compose.yaml"
REPLAY_COMPOSE = ROOT / "compose.replay.yaml"
HARDWARE_COMPOSE = ROOT / "compose.hardware.yaml"
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
ENTRYPOINT = ROOT / "scripts" / "container_entrypoint.sh"


def test_container_has_cpu_test_and_headless_runtime_targets() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "AS test" in dockerfile
    assert "AS runtime-headless" in dockerfile
    assert "python:3.11.11-slim-bookworm" in dockerfile
    assert "opencv-python-headless" in dockerfile
    assert "opencv-python==" not in dockerfile
    assert "torch==2.11.0 torchvision==0.26.0" in dockerfile
    assert "https://download.pytorch.org/whl/cu128" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "PYTHONUNBUFFERED=1" in dockerfile
    test_stage, runtime_stage = dockerfile.split("FROM ${PYTHON_IMAGE} AS runtime-headless")
    assert "g++ git" in test_stage
    assert "ruff format --check src tests tools" in test_stage
    assert "ruff check src tests tools" in test_stage
    assert "apt-get" not in runtime_stage
    assert "YOLO_CONFIG_DIR=/tmp/ultralytics" in runtime_stage


def test_entrypoint_creates_yolo_settings_tree_before_python_imports(
    tmp_path: Path,
) -> None:
    config_directory = tmp_path / "ultralytics"
    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    artifact_directory.chmod(0o2770)
    output = artifact_directory / "permission-check"
    environment = os.environ.copy()
    environment["YOLO_CONFIG_DIR"] = str(config_directory)
    environment.pop("MARINE_PTZ_PREFLIGHT_MODE", None)

    completed = subprocess.run(
        (
            "/bin/sh",
            str(ENTRYPOINT),
            "/bin/sh",
            "-c",
            'umask; : > "$1"',
            "permission-check",
            str(output),
        ),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert config_directory.is_dir()
    assert (config_directory / "Ultralytics").is_dir()
    assert stat.S_IMODE(config_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((config_directory / "Ultralytics").stat().st_mode) == 0o700
    assert completed.stdout.strip() == "0027"
    assert stat.S_IMODE(output.stat().st_mode) == 0o640
    assert output.stat().st_gid == artifact_directory.stat().st_gid
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    assert entrypoint.index('mkdir -p -m 0700 "$YOLO_CONFIG_DIR"') < entrypoint.index("python ")


def test_container_context_excludes_private_and_generated_content() -> None:
    patterns = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()

    for pattern in (
        ".git",
        ".env",
        "secrets",
        "*.pem",
        "*.key",
        "*.token",
        "artifacts",
        "runs",
        "*.pt",
        "*.engine",
        "*.mp4",
    ):
        assert pattern in patterns
    assert "!data/tugboat/data.yaml" in patterns


def _service(path: Path, name: str) -> tuple[dict[str, object], str]:
    raw = path.read_text(encoding="utf-8")
    document = yaml.safe_load(raw)
    assert set(document["services"]) == {name}
    return document["services"][name], raw


def _assert_runtime_security(service: dict[str, object]) -> None:
    assert service["read_only"] is True
    assert service["network_mode"] == "none"
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["init"] is True
    assert any(item.startswith("/tmp:") for item in service["tmpfs"])
    assert service.get("privileged") is not True
    assert service["pull_policy"] == "never"


def _tmpfs_options(service: dict[str, object], mountpoint: str) -> set[str]:
    prefix = f"{mountpoint}:"
    entry = next(item for item in service["tmpfs"] if item.startswith(prefix))
    return set(entry.removeprefix(prefix).split(","))


def test_compose_files_isolate_test_replay_and_hardware_services() -> None:
    test, test_raw = _service(TEST_COMPOSE, "test")
    replay, replay_raw = _service(REPLAY_COMPOSE, "replay")
    hardware, hardware_raw = _service(HARDWARE_COMPOSE, "hardware")

    for service in (test, replay, hardware):
        _assert_runtime_security(service)

    assert "devices" not in test
    assert "deploy" not in test
    assert "MARINE_PTZ_CONFIG" not in test_raw
    assert "MARINE_PTZ_GPU" not in test_raw
    assert test["environment"]["TMPDIR"] == "/run/marine-ptz-test-tmp"
    assert "noexec" in _tmpfs_options(test, "/tmp")
    test_exec_tmp = _tmpfs_options(test, "/run/marine-ptz-test-tmp")
    assert {"rw", "exec", "nosuid", "nodev", "size=128m", "mode=0700"} <= test_exec_tmp
    assert {"uid=10001", "gid=10001"} <= test_exec_tmp
    assert "simulated" in replay["command"]
    assert "concurrent" in replay["command"]
    assert replay["environment"]["MARINE_PTZ_TARGET_CLASS"] == "marine_target"
    assert replay["environment"]["YOLO_CONFIG_DIR"] == "/tmp/ultralytics"
    assert PurePosixPath(replay["environment"]["YOLO_CONFIG_DIR"]).is_relative_to(
        PurePosixPath("/tmp")
    )
    assert {"rw", "noexec", "nosuid", "nodev"} <= _tmpfs_options(replay, "/tmp")
    assert replay["group_add"] == [
        "${MARINE_PTZ_ARTIFACT_GID:?set the numeric GID owning the artifact directory}"
    ]
    assert "devices" not in replay
    assert "serial" not in replay_raw.casefold()
    assert "camera" not in replay_raw.casefold()
    assert "MARINE_PTZ_GPU_DEVICE:?" in replay_raw
    assert hardware["profiles"] == ["hardware"]
    assert "--arm-hardware" in hardware["command"]
    assert "--serial-port" in hardware["command"]
    assert "single" in hardware["command"]
    assert replay["image"] == hardware["image"]
    assert "MARINE_PTZ_IMAGE" in replay["image"]
    assert "MARINE_PTZ_CAMERA_DEVICE:?" in hardware_raw
    assert "MARINE_PTZ_SERIAL_DEVICE:?" in hardware_raw
    assert "MARINE_PTZ_ARTIFACT_GID:?" in hardware_raw
    assert "MARINE_PTZ_VIDEO_GID:?" in hardware_raw
    assert "MARINE_PTZ_DIALOUT_GID:?" in hardware_raw
    assert hardware["group_add"] == [
        "${MARINE_PTZ_ARTIFACT_GID:?set the numeric GID owning the artifact directory}",
        "${MARINE_PTZ_VIDEO_GID:?set the host video group numeric GID}",
        "${MARINE_PTZ_DIALOUT_GID:?set the host dialout group numeric GID}",
    ]
    for runtime in (replay, hardware):
        assert "noexec" in _tmpfs_options(runtime, "/tmp")
        assert all(not item.startswith("/run/marine-ptz-test-tmp:") for item in runtime["tmpfs"])
        assert "TMPDIR" not in runtime["environment"]
    for raw in (test_raw, replay_raw, hardware_raw):
        assert "docker.sock" not in raw
        assert "network_mode: host" not in raw


def test_replay_outputs_are_absolute_and_confined_to_artifacts() -> None:
    replay, _raw = _service(REPLAY_COMPOSE, "replay")
    command = replay["command"]
    output_flags = [index for index, argument in enumerate(command) if argument == "--output"]

    assert output_flags
    for index in output_flags:
        output = PurePosixPath(command[index + 1])
        assert output.is_absolute()
        assert output.is_relative_to(PurePosixPath("/artifacts"))
        assert output != PurePosixPath("/artifacts")
    assert replay["environment"]["MARINE_PTZ_OUTPUT_DIRECTORY"] == "/artifacts"


def test_compose_only_writes_to_artifact_and_tmp_mounts() -> None:
    for path, name in ((REPLAY_COMPOSE, "replay"), (HARDWARE_COMPOSE, "hardware")):
        service, _raw = _service(path, name)
        volumes = service["volumes"]
        writable = [volume for volume in volumes if not volume.get("read_only", False)]
        assert len(writable) == 1
        assert writable[0]["target"] == "/artifacts"


def test_cpp_protocol_parity_tests_remain_collected() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-vv",
            "tests/test_cpp_protocol.py",
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "test_python_firmware_matches_golden_vectors" in completed.stdout
    assert "test_cpp_firmware_matches_golden_vectors" in completed.stdout
    assert "tests collected" in completed.stdout


def _compose_environment(values: dict[str, str] | None = None) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("MARINE_PTZ_")
    }
    environment.update(values or {})
    return environment


def _compose_config(
    compose_file: Path,
    *,
    environment: dict[str, str] | None = None,
    hardware_profile: bool = False,
) -> subprocess.CompletedProcess[str]:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is unavailable")
    command = [
        docker,
        "compose",
        "--env-file",
        "/dev/null",
        "-f",
        str(compose_file),
    ]
    if hardware_profile:
        command.extend(("--profile", "hardware"))
    command.extend(("config", "--quiet"))
    return subprocess.run(
        command,
        cwd=ROOT,
        env=_compose_environment(environment),
        check=False,
        capture_output=True,
        text=True,
    )


def test_base_compose_resolves_without_any_marine_ptz_variables() -> None:
    completed = _compose_config(TEST_COMPOSE)

    assert completed.returncode == 0, completed.stderr


def _replay_environment(tmp_path: Path) -> dict[str, str]:
    config = tmp_path / "development.yaml"
    model = tmp_path / "best.pt"
    video = tmp_path / "input.mp4"
    artifacts = tmp_path / "artifacts"
    config.write_bytes((ROOT / "configs" / "development.yaml").read_bytes())
    model.write_bytes(b"model fixture")
    video.write_bytes(b"video fixture")
    artifacts.mkdir()
    return {
        "MARINE_PTZ_CONFIG": str(config),
        "MARINE_PTZ_MODEL": str(model),
        "MARINE_PTZ_INPUT_VIDEO": str(video),
        "MARINE_PTZ_ARTIFACTS": str(artifacts),
        "MARINE_PTZ_ARTIFACT_GID": str(os.getgid()),
        "MARINE_PTZ_GPU_DEVICE": "0",
    }


@pytest.mark.parametrize(
    "missing",
    [
        "MARINE_PTZ_CONFIG",
        "MARINE_PTZ_MODEL",
        "MARINE_PTZ_INPUT_VIDEO",
        "MARINE_PTZ_ARTIFACTS",
        "MARINE_PTZ_ARTIFACT_GID",
        "MARINE_PTZ_GPU_DEVICE",
    ],
)
def test_replay_compose_rejects_each_missing_required_variable(
    tmp_path: Path, missing: str
) -> None:
    environment = _replay_environment(tmp_path)
    environment.pop(missing)

    completed = _compose_config(REPLAY_COMPOSE, environment=environment)

    assert completed.returncode != 0
    assert missing in completed.stderr


def test_replay_compose_resolves_with_explicit_absolute_inputs(tmp_path: Path) -> None:
    completed = _compose_config(REPLAY_COMPOSE, environment=_replay_environment(tmp_path))

    assert completed.returncode == 0, completed.stderr


def _hardware_environment(tmp_path: Path) -> dict[str, str]:
    config = tmp_path / "hardware.yaml"
    model = tmp_path / "best.pt"
    artifacts = tmp_path / "artifacts"
    config.write_bytes((ROOT / "configs" / "hardware.yaml").read_bytes())
    model.write_bytes(b"model fixture")
    artifacts.mkdir()
    return {
        "MARINE_PTZ_CONFIG": str(config),
        "MARINE_PTZ_MODEL": str(model),
        "MARINE_PTZ_ARTIFACTS": str(artifacts),
        "MARINE_PTZ_ARTIFACT_GID": str(os.getgid()),
        "MARINE_PTZ_GPU_DEVICE": "0",
        "MARINE_PTZ_CAMERA_DEVICE": "/dev/verified-camera-fixture",
        "MARINE_PTZ_SERIAL_DEVICE": "/dev/verified-serial-fixture",
        "MARINE_PTZ_VIDEO_GID": "44",
        "MARINE_PTZ_DIALOUT_GID": "20",
    }


@pytest.mark.parametrize(
    "missing",
    [
        "MARINE_PTZ_CONFIG",
        "MARINE_PTZ_MODEL",
        "MARINE_PTZ_ARTIFACTS",
        "MARINE_PTZ_ARTIFACT_GID",
        "MARINE_PTZ_GPU_DEVICE",
        "MARINE_PTZ_CAMERA_DEVICE",
        "MARINE_PTZ_SERIAL_DEVICE",
        "MARINE_PTZ_VIDEO_GID",
        "MARINE_PTZ_DIALOUT_GID",
    ],
)
def test_hardware_compose_rejects_each_missing_required_variable(
    tmp_path: Path, missing: str
) -> None:
    environment = _hardware_environment(tmp_path)
    environment.pop(missing)

    completed = _compose_config(
        HARDWARE_COMPOSE,
        environment=environment,
        hardware_profile=True,
    )

    assert completed.returncode != 0
    assert missing in completed.stderr


def test_hardware_compose_resolves_only_with_explicit_profile_and_inputs(
    tmp_path: Path,
) -> None:
    completed = _compose_config(
        HARDWARE_COMPOSE,
        environment=_hardware_environment(tmp_path),
        hardware_profile=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_entrypoint_exec_propagates_sigterm_to_runtime(tmp_path: Path) -> None:
    marker = tmp_path / "ready"
    child = (
        "from pathlib import Path; import time; "
        f"Path({str(marker)!r}).write_text('ready'); time.sleep(30)"
    )
    process = subprocess.Popen(
        ("/bin/sh", str(ENTRYPOINT), sys.executable, "-c", child),
        cwd=ROOT,
        env={key: value for key, value in os.environ.items() if key != "MARINE_PTZ_PREFLIGHT_MODE"},
    )
    try:
        deadline = time.monotonic() + 5.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=2.0) == -signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)


def test_entrypoint_preflight_preserves_the_runtime_command() -> None:
    completed = subprocess.run(
        (
            "/bin/sh",
            str(ENTRYPOINT),
            sys.executable,
            "-c",
            "print('runtime-started')",
        ),
        cwd=ROOT,
        env=os.environ | {"MARINE_PTZ_PREFLIGHT_MODE": "test", "MARINE_PTZ_APP_ROOT": str(ROOT)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    lines = completed.stdout.splitlines()
    assert '"ok": true' in lines[0]
    assert lines[-1] == "runtime-started"
