from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

import marine_ptz.serial_transport as serial_transport_module
from marine_ptz.serial_transport import PySerialTransport, SerialTransportError


class FakeSerialConnection:
    def __init__(self) -> None:
        self.port: str | None = None
        self.timeout = 0.25
        self.is_open = False
        self.open_count = 0
        self.close_count = 0
        self.reset_count = 0
        self.flush_count = 0
        self.write_timeout = 0.5
        self.write_results: deque[int | Exception] = deque()
        self.write_calls: list[bytes] = []
        self.reads: deque[bytes | Exception] = deque()

    def open(self) -> None:
        self.open_count += 1
        self.is_open = True

    def close(self) -> None:
        self.close_count += 1
        self.is_open = False

    def write(self, data: bytes) -> int:
        self.write_calls.append(bytes(data))
        if not self.write_results:
            return len(data)
        result = self.write_results.popleft()
        if isinstance(result, Exception):
            raise result
        return result

    def flush(self) -> None:
        self.flush_count += 1
        raise AssertionError("bounded transport writes must never call flush()")

    def read_until(self, expected: bytes, size: int) -> bytes:
        assert expected == b"\n"
        assert size == 65
        if not self.reads:
            return b""
        value = self.reads.popleft()
        if isinstance(value, Exception):
            raise value
        return value

    def reset_input_buffer(self) -> None:
        self.reset_count += 1


def _opened_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    monotonic: object | None = None,
) -> tuple[PySerialTransport, FakeSerialConnection, dict[str, object]]:
    connection = FakeSerialConnection()
    constructor_arguments: dict[str, object] = {}

    def serial_factory(**kwargs: object) -> FakeSerialConnection:
        constructor_arguments.update(kwargs)
        connection.timeout = float(kwargs["timeout"])
        return connection

    module = SimpleNamespace(Serial=serial_factory)
    monkeypatch.setattr(
        serial_transport_module,
        "import_module",
        lambda name: module,
    )
    transport = PySerialTransport(
        "/dev/fake",
        baudrate=115200,
        read_timeout_s=0.25,
        write_timeout_s=0.5,
        **({"monotonic": monotonic} if monotonic is not None else {}),
    )
    transport.open()
    return transport, connection, constructor_arguments


def test_pyserial_open_allows_normal_auto_reset_without_hardware_flow_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, connection, arguments = _opened_transport(monkeypatch)

    assert arguments == {
        "port": None,
        "baudrate": 115200,
        "timeout": 0.25,
        "write_timeout": 0.5,
        "rtscts": False,
        "dsrdtr": False,
    }
    assert connection.port == "/dev/fake"
    assert connection.open_count == 1
    assert transport.may_reset_on_open is True


def test_partial_writes_complete_without_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, connection, _ = _opened_transport(monkeypatch)
    connection.write_results.extend([2, 3, 3])

    transport.write(b"HELLO 1\n")

    assert connection.write_calls == [b"HELLO 1\n", b"LLO 1\n", b" 1\n"]
    assert connection.flush_count == 0


def test_zero_progress_write_is_rejected_without_retry_or_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, connection, _ = _opened_transport(monkeypatch)
    connection.write_results.append(0)

    with pytest.raises(SerialTransportError, match="no progress"):
        transport.write(b"HELLO 1\n")

    assert connection.write_calls == [b"HELLO 1\n"]
    assert connection.flush_count == 0


def test_pyserial_write_timeout_is_reported_without_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, connection, _ = _opened_transport(monkeypatch)
    connection.write_results.append(TimeoutError("write deadline expired"))

    with pytest.raises(SerialTransportError, match="deadline expired"):
        transport.write(b"HELLO 1\n")

    assert connection.write_calls == [b"HELLO 1\n"]
    assert connection.flush_count == 0


def test_partial_write_loop_uses_one_elapsed_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    transport, connection, _ = _opened_transport(
        monkeypatch,
        monotonic=lambda: now[0],
    )
    observed_timeouts: list[float] = []

    def slow_partial_write(data: bytes) -> int:
        connection.write_calls.append(bytes(data))
        observed_timeouts.append(connection.write_timeout)
        now[0] += 0.3
        return 1

    connection.write = slow_partial_write  # type: ignore[method-assign]

    with pytest.raises(SerialTransportError, match="timed out"):
        transport.write(b"HELLO")

    assert observed_timeouts == [pytest.approx(0.5), pytest.approx(0.2)]
    assert now[0] == pytest.approx(0.6)
    assert connection.flush_count == 0


def test_successful_write_and_close_never_enter_a_drain_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, connection, _ = _opened_transport(monkeypatch)

    transport.write(b"DISABLE 2\n")
    transport.close()

    assert connection.flush_count == 0
    assert connection.close_count == 1


def test_fragmented_and_multiple_frames_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, connection, _ = _opened_transport(monkeypatch)
    connection.reads.extend(
        [
            b"ACK 1 SET ",
            b"90 80\r\nSTATUS 2 1 90 80 ARMED\n",
        ]
    )

    assert transport.read_frame() is None
    assert transport.read_frame() == b"ACK 1 SET 90 80"
    assert transport.read_frame() == b"STATUS 2 1 90 80 ARMED"


def test_disconnect_mid_frame_is_reported_and_close_remains_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, connection, _ = _opened_transport(monkeypatch)
    connection.reads.extend([b"ACK 1", OSError("device disconnected")])

    assert transport.read_frame() is None
    with pytest.raises(SerialTransportError, match="disconnected"):
        transport.read_frame()

    transport.close()
    transport.close()
    assert connection.close_count == 1


def test_per_read_timeout_is_applied_then_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, connection, _ = _opened_transport(monkeypatch)
    observed: list[float] = []

    def read_until(expected: bytes, size: int) -> bytes:
        observed.append(connection.timeout)
        return b""

    connection.read_until = read_until  # type: ignore[method-assign]

    assert transport.read_frame(timeout_s=0.05) is None

    assert observed == [0.05]
    assert connection.timeout == 0.25
