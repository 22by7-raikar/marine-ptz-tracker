from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import pytest

from marine_ptz.firmware_model import FirmwareStateMachine, FirmwareStreamModel
from marine_ptz.serial_protocol import parse_response


ROOT = Path(__file__).resolve().parents[1]
VECTOR_PATH = ROOT / "tests" / "fixtures" / "serial_protocol_vectors.tsv"


@dataclass(frozen=True, slots=True)
class GoldenVector:
    name: str
    startup: bool
    events: tuple[tuple[int, bytes], ...]
    expected: bytes


def _unescape(value: str) -> bytes:
    output = bytearray()
    index = 0
    escapes = {"n": 0x0A, "r": 0x0D, "t": 0x09, "\\": 0x5C}
    while index < len(value):
        if value[index] != "\\":
            output.append(ord(value[index]))
            index += 1
            continue
        index += 1
        if index >= len(value):
            raise AssertionError("golden vector ends with an incomplete escape")
        escape = value[index]
        index += 1
        if escape in escapes:
            output.append(escapes[escape])
            continue
        if escape == "x":
            digits = value[index : index + 2]
            if len(digits) != 2:
                raise AssertionError("golden vector has an incomplete hex escape")
            output.append(int(digits, 16))
            index += 2
            continue
        raise AssertionError(f"unsupported golden-vector escape: \\{escape}")
    return bytes(output)


def _load_vectors() -> tuple[GoldenVector, ...]:
    vectors: list[GoldenVector] = []
    for line_number, line in enumerate(
        VECTOR_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise AssertionError(
                f"{VECTOR_PATH}:{line_number}: expected four tab-separated fields"
            )
        name, startup_text, event_text, expected_text = fields
        events: list[tuple[int, bytes]] = []
        if event_text != "-":
            for raw_event in event_text.split("|"):
                millis_text, escaped = raw_event.split("@", 1)
                millis = int(millis_text)
                if not 0 <= millis <= 0xFFFF_FFFF:
                    raise AssertionError(f"{name}: millis must fit uint32")
                events.append((millis, _unescape(escaped)))
        vectors.append(
            GoldenVector(
                name=name,
                startup=startup_text == "1",
                events=tuple(events),
                expected=_unescape(expected_text),
            )
        )
    return tuple(vectors)


VECTORS = _load_vectors()


@pytest.fixture(scope="session")
def cpp_protocol_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.fail("g++ is unavailable; executable C++ protocol parity was not tested")
    output = tmp_path_factory.mktemp("cpp-protocol") / "protocol_harness"
    command = [
        compiler,
        "-std=c++11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-I",
        str(ROOT / "firmware" / "ptz_controller"),
        str(ROOT / "firmware" / "ptz_controller" / "protocol.cpp"),
        str(ROOT / "tests" / "cpp" / "protocol_harness.cpp"),
        "-o",
        str(output),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, (
        "host C++ protocol harness compilation failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stderr == ""
    return output


@pytest.mark.parametrize("vector", VECTORS, ids=lambda vector: vector.name)
def test_python_firmware_matches_golden_vectors(vector: GoldenVector) -> None:
    now = [0.0]
    machine = FirmwareStateMachine(clock=lambda: now[0])
    stream = FirmwareStreamModel(machine)
    output = bytearray(machine.startup_frame() if vector.startup else b"")
    previous_millis: int | None = None
    for millis, payload in vector.events:
        if previous_millis is None:
            now[0] = millis / 1000.0
        else:
            elapsed_ms = (millis - previous_millis) & 0xFFFF_FFFF
            now[0] += elapsed_ms / 1000.0
        previous_millis = millis
        machine.tick()
        output.extend(b"".join(stream.feed(payload)))
        machine.tick()

    assert bytes(output) == vector.expected


@pytest.mark.parametrize("vector", VECTORS, ids=lambda vector: vector.name)
def test_cpp_firmware_matches_golden_vectors(
    vector: GoldenVector,
    cpp_protocol_harness: Path,
) -> None:
    command = [
        str(cpp_protocol_harness),
        "1" if vector.startup else "0",
        *(f"{millis}:{payload.hex()}" for millis, payload in vector.events),
    ]
    result = subprocess.run(command, check=False, capture_output=True)

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert result.stderr == b""
    assert result.stdout == vector.expected


def test_maximum_payload_vectors_have_exact_independent_lengths() -> None:
    by_name = {vector.name: vector for vector in VECTORS}

    assert len(by_name["exact_max_lf"].events[0][1][:-1]) == 63
    assert len(by_name["exact_max_crlf"].events[0][1][:-2]) == 63
    assert len(by_name["one_byte_overflow_recovery"].events[0][1].split(b"\r\n")[0]) == 64


def test_all_golden_firmware_responses_fit_the_fixed_payload_buffer() -> None:
    payloads = [
        payload
        for vector in VECTORS
        for payload in vector.expected.split(b"\r\n")
        if payload
    ]

    assert payloads
    assert max(len(payload) for payload in payloads) <= 63
    assert all(parse_response(payload) is not None for payload in payloads)
