"""Small serial transport boundary with lazy pyserial and an in-memory fake."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable
from importlib import import_module
from numbers import Real
from typing import Any, Protocol, runtime_checkable

from .serial_protocol import (
    MAX_FRAME_LENGTH,
    FramingIssue,
    LineFramer,
)


class SerialTransportError(RuntimeError):
    """Raised when a serial transport cannot complete bounded I/O."""


class SerialDependencyError(SerialTransportError):
    """Raised when an explicitly requested pyserial backend is unavailable."""


@runtime_checkable
class SerialTransport(Protocol):
    """Bounded newline-frame transport used by the host actuator."""

    @property
    def is_open(self) -> bool:
        """Return whether transport I/O is currently available."""

    @property
    def may_reset_on_open(self) -> bool:
        """Return whether opening may reset the attached controller."""

    def write(self, data: bytes) -> None:
        """Submit complete frames to the OS/driver within the write deadline.

        A correlated firmware response, not local write completion, provides
        end-to-end command confirmation.
        """

    def read_frame(
        self,
        maximum_length: int = MAX_FRAME_LENGTH,
        *,
        timeout_s: float | None = None,
    ) -> bytes | None:
        """Return one frame payload, or None after the configured timeout."""

    def reset_input_buffer(self) -> None:
        """Discard stale inbound frames before a new handshake."""

    def close(self) -> None:
        """Release transport resources idempotently."""

    def __enter__(self) -> SerialTransport:
        """Return an opened or already-open transport."""

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        """Close transport resources."""


class PySerialTransport:
    """Explicitly opened pyserial transport; importing this module is hardware-free."""

    def __init__(
        self,
        port: str,
        *,
        baudrate: int,
        read_timeout_s: float,
        write_timeout_s: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(port, str) or not port.strip():
            raise ValueError("serial port must be a non-empty string")
        self._port = port
        self._baudrate = baudrate
        self._read_timeout_s = read_timeout_s
        self._write_timeout_s = write_timeout_s
        self._monotonic = monotonic
        self._connection: Any | None = None
        self._framer = LineFramer()
        self._frames: deque[bytes] = deque()

    @property
    def is_open(self) -> bool:
        return self._connection is not None and bool(getattr(self._connection, "is_open", True))

    @property
    def may_reset_on_open(self) -> bool:
        return True

    def open(self) -> None:
        if self.is_open:
            return
        try:
            serial = import_module("serial")
        except (ImportError, ModuleNotFoundError) as exc:
            raise SerialDependencyError(
                "pyserial is unavailable; install the optional hardware dependencies"
            ) from exc
        try:
            connection = serial.Serial(
                port=None,
                baudrate=self._baudrate,
                timeout=self._read_timeout_s,
                write_timeout=self._write_timeout_s,
                rtscts=False,
                dsrdtr=False,
            )
            connection.port = self._port
            connection.open()
            self._connection = connection
        except Exception as exc:
            connection = locals().get("connection")
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            self._connection = None
            raise SerialTransportError(f"unable to open serial port {self._port!r}: {exc}") from exc

    def write(self, data: bytes) -> None:
        connection = self._require_open()
        if not isinstance(data, bytes) or not data:
            raise ValueError("serial write data must be non-empty bytes")
        deadline = self._monotonic() + self._write_timeout_s
        offset = 0
        while offset < len(data):
            remaining = deadline - self._monotonic()
            if remaining <= 0.0:
                raise SerialTransportError(
                    f"serial write timed out after {self._write_timeout_s:g} seconds"
                )
            original_timeout = getattr(
                connection,
                "write_timeout",
                self._write_timeout_s,
            )
            try:
                connection.write_timeout = remaining
                written = connection.write(data[offset:])
            except Exception as exc:
                raise SerialTransportError(f"serial write failed: {exc}") from exc
            finally:
                try:
                    connection.write_timeout = original_timeout
                except Exception:
                    pass
            if (
                isinstance(written, bool)
                or not isinstance(written, int)
                or written < 0
                or written > len(data) - offset
            ):
                raise SerialTransportError(f"serial write returned invalid byte count {written!r}")
            if written == 0:
                raise SerialTransportError("serial write made no progress before its deadline")
            offset += written

    def read_frame(
        self,
        maximum_length: int = MAX_FRAME_LENGTH,
        *,
        timeout_s: float | None = None,
    ) -> bytes | None:
        if self._frames:
            return self._frames.popleft()
        timeout = _optional_timeout(timeout_s)
        connection = self._require_open()
        original_timeout = getattr(connection, "timeout", self._read_timeout_s)
        try:
            if timeout is not None:
                connection.timeout = timeout
            data = connection.read_until(b"\n", maximum_length + 2)
        except Exception as exc:
            raise SerialTransportError(f"serial read failed: {exc}") from exc
        finally:
            if timeout is not None:
                try:
                    connection.timeout = original_timeout
                except Exception:
                    pass
        if not data:
            return None
        for event in self._framer.feed(bytes(data)):
            if isinstance(event, FramingIssue):
                raise SerialTransportError(str(event.error)) from event.error
            self._frames.append(event)
        return self._frames.popleft() if self._frames else None

    def reset_input_buffer(self) -> None:
        connection = self._require_open()
        try:
            connection.reset_input_buffer()
        except Exception as exc:
            raise SerialTransportError(f"unable to reset serial input buffer: {exc}") from exc
        self._framer.reset()
        self._frames.clear()

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        self._framer.reset()
        self._frames.clear()
        if connection is not None:
            try:
                connection.close()
            except Exception as exc:
                raise SerialTransportError(f"serial close failed: {exc}") from exc

    def __enter__(self) -> PySerialTransport:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.close()

    def _require_open(self) -> Any:
        if not self.is_open:
            raise SerialTransportError("serial transport is not open")
        return self._connection


class InMemorySerialTransport:
    """Deterministic frame transport for tests and the default protocol demo."""

    def __init__(
        self,
        responder: Callable[[bytes], Iterable[bytes]] | None = None,
    ) -> None:
        self._responder = responder
        self._host_framer = LineFramer()
        self._response_framer = LineFramer()
        self._responses: deque[bytes] = deque()
        self.writes: list[bytes] = []
        self.reset_count = 0
        self.close_count = 0
        self._open = True

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def may_reset_on_open(self) -> bool:
        return False

    def open(self) -> None:
        self._require_open()

    def write(self, data: bytes) -> None:
        self._require_open()
        self.writes.append(bytes(data))
        for event in self._host_framer.feed(data):
            if isinstance(event, FramingIssue):
                raise SerialTransportError(str(event.error)) from event.error
            if self._responder is not None:
                for response in self._responder(event):
                    self.queue_encoded(response)

    def read_frame(
        self,
        maximum_length: int = MAX_FRAME_LENGTH,
        *,
        timeout_s: float | None = None,
    ) -> bytes | None:
        self._require_open()
        _optional_timeout(timeout_s)
        if not self._responses:
            return None
        frame = self._responses.popleft()
        if len(frame) > maximum_length:
            raise SerialTransportError(f"in-memory response exceeds {maximum_length} bytes")
        return frame

    def queue_frame(self, frame: bytes) -> None:
        self._require_open()
        if b"\n" in frame or b"\r" in frame:
            raise ValueError("queue_frame expects a payload without line endings")
        self._responses.append(bytes(frame))

    def queue_encoded(self, encoded: bytes) -> None:
        self._require_open()
        for event in self._response_framer.feed(encoded):
            if isinstance(event, FramingIssue):
                raise SerialTransportError(str(event.error)) from event.error
            self._responses.append(event)

    def reset_input_buffer(self) -> None:
        self._require_open()
        self.reset_count += 1
        self._responses.clear()
        self._response_framer.reset()

    def close(self) -> None:
        if self._open:
            self.close_count += 1
            self._open = False
            self._host_framer.reset()
            self._response_framer.reset()
            self._responses.clear()

    def __enter__(self) -> InMemorySerialTransport:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.close()

    def _require_open(self) -> None:
        if not self._open:
            raise SerialTransportError("in-memory serial transport is closed")


def _optional_timeout(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("timeout_s must be a finite number or None")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout < 0.0:
        raise ValueError("timeout_s must be a finite number zero or greater")
    return timeout
