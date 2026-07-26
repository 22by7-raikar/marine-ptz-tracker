from __future__ import annotations

from dataclasses import replace
import signal
from typing import Any

import pytest

import marine_ptz.vision_cli as vision_cli
from marine_ptz.actuators import SimulatedPTZActuator
from marine_ptz.cancellation import OperationCancelled
from marine_ptz.config import AppConfig, ConfigError, load_config
from marine_ptz.firmware_model import FirmwareSafetyConfig, FirmwareStateMachine
from marine_ptz.serial_actuator import (
    ArduinoSerialActuator,
    SerialActuatorTimeoutError,
)
from marine_ptz.serial_protocol import (
    AckResponse,
    DisableCommand,
    EnableCommand,
    HelloCommand,
    SetCommand,
    encode_response,
    parse_command,
)
from marine_ptz.serial_transport import InMemorySerialTransport
from marine_ptz.types import Detection, Frame, PTZCommand
from marine_ptz.vision_cli import (
    RuntimeCleanupError,
    UnexpectedCancellationError,
    VisionOptions,
    _RuntimeResources,
    _build_actuator,
    run_vision,
)


pytestmark = pytest.mark.integration


class FakeImage:
    shape = (100, 100, 3)

    def copy(self) -> FakeImage:
        return FakeImage()


class FakeSource:
    def __init__(
        self,
        frames: list[Frame | None | BaseException],
        events: list[str] | None = None,
    ) -> None:
        self._frames = list(frames)
        self._events = events
        self.close_count = 0
        self.read_count = 0

    def read(self) -> Frame | None:
        self.read_count += 1
        value = self._frames.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self) -> None:
        self.close_count += 1
        if self._events is not None:
            self._events.append("source:close")


class FakeDetector:
    active_device = "cpu"
    last_inference_ms = 1.25

    def __init__(
        self,
        outputs: list[tuple[Detection, ...] | BaseException],
        events: list[str] | None = None,
    ) -> None:
        self._outputs = list(outputs)
        self._events = events
        self.calls = 0

    def detect(self, frame: Frame) -> tuple[Detection, ...]:
        self.calls += 1
        if self._events is not None:
            self._events.append(f"detect:{frame.sequence}")
        value = self._outputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class FakeActuator:
    def __init__(
        self,
        events: list[str] | None = None,
        *,
        open_error: Exception | None = None,
        apply_error_at: int | None = None,
        disable_error: Exception | None = None,
    ) -> None:
        self.events = events if events is not None else []
        self.open_error = open_error
        self.apply_error_at = apply_error_at
        self.disable_error = disable_error
        self.pan_deg = 90.0
        self.tilt_deg = 90.0
        self.applied: list[PTZCommand] = []
        self.close_count = 0
        self.disable_count = 0

    def open(self) -> None:
        self.events.append("actuator:open")
        if self.open_error is not None:
            raise self.open_error

    def enable(self) -> None:
        self.events.append("actuator:enable")

    def apply(self, command: PTZCommand) -> None:
        self.events.append(f"actuator:apply:{command.sequence}")
        self.applied.append(command)
        if self.apply_error_at == len(self.applied):
            raise RuntimeError("protocol fault")
        self.pan_deg = command.pan_deg
        self.tilt_deg = command.tilt_deg

    def disable(self) -> None:
        self.disable_count += 1
        self.events.append("actuator:disable")
        if self.disable_error is not None:
            raise self.disable_error

    def close(self) -> None:
        self.close_count += 1
        self.events.append("actuator:close")


class CancellationState:
    def __init__(self) -> None:
        self.requested = False

    def __call__(self) -> bool:
        return self.requested

    def request(self) -> None:
        self.requested = True


def make_frame(sequence: int, *, image: object | None = None) -> Frame:
    return Frame(
        image=FakeImage() if image is None else image,
        timestamp_s=float(sequence),
        width=100,
        height=100,
        sequence=sequence,
    )


def make_options(
    *,
    backend: str = "simulated",
    arm_hardware: bool = False,
    serial_port: str | None = None,
    max_frames: int | None = None,
) -> VisionOptions:
    return VisionOptions(
        source="offline.mp4",
        model="local.pt",
        device="cpu",
        target_classes=("boat",),
        confidence_threshold=0.5,
        iou_threshold=0.45,
        image_size=320,
        max_frames=max_frames,
        display=False,
        output=None,
        actuator_backend=backend,
        serial_port=serial_port,
        arm_hardware=arm_hardware,
    )


def run_fake(
    config: AppConfig,
    options: VisionOptions,
    source: FakeSource,
    detector: FakeDetector,
    actuator: FakeActuator,
    *,
    emit: Any = lambda _line: None,
    stop_requested: Any = lambda: False,
) -> int:
    return run_vision(
        config,
        options,
        source_factory=lambda _source: source,
        detector_factory=lambda *_args, **_kwargs: detector,
        actuator_factory=lambda _config, _options: actuator,
        rate_monotonic=lambda: 10.0,
        sleep=lambda _seconds: None,
        stop_requested=stop_requested,
        emit=emit,
    )


def production_firmware(
    config: AppConfig,
    *,
    clock: Any = lambda: 0.0,
) -> FirmwareStateMachine:
    assert config.serial is not None
    settings = config.serial
    return FirmwareStateMachine(
        FirmwareSafetyConfig(
            pan_min_deg=int(settings.physical_pan_limits.minimum_deg),
            pan_max_deg=int(settings.physical_pan_limits.maximum_deg),
            tilt_min_deg=int(settings.physical_tilt_limits.minimum_deg),
            tilt_max_deg=int(settings.physical_tilt_limits.maximum_deg),
            neutral_pan_deg=int(settings.neutral_pan_deg),
            neutral_tilt_deg=int(settings.neutral_tilt_deg),
            watchdog_timeout_s=settings.watchdog_timeout_s,
        ),
        clock=clock,
    )


def production_actuator(
    config: AppConfig,
    transport: InMemorySerialTransport,
    **kwargs: Any,
) -> ArduinoSerialActuator:
    assert config.serial is not None
    return ArduinoSerialActuator(
        config.serial,
        config.actuator.pan_limits,
        config.actuator.tilt_limits,
        initial_pan_deg=config.actuator.initial_pan_deg,
        initial_tilt_deg=config.actuator.initial_tilt_deg,
        transport=transport,
        **kwargs,
    )


def test_complete_target_and_lost_target_paths_apply_commands() -> None:
    source = FakeSource([make_frame(0), make_frame(1), None])
    detector = FakeDetector(
        [(Detection("boat", 0.9, 70, 70, 90, 90),), ()]
    )
    actuator = FakeActuator()
    telemetry: list[str] = []

    processed = run_fake(
        load_config("configs/development.yaml"),
        make_options(),
        source,
        detector,
        actuator,
        emit=telemetry.append,
    )

    assert processed == 2
    assert [(item.pan_deg, item.tilt_deg) for item in actuator.applied] == [
        (93.0, 93.0),
        (93.0, 93.0),
    ]
    assert "target=boat" in telemetry[0]
    assert "target=none" in telemetry[1]
    assert all("actuator=simulated" in line for line in telemetry)


def test_simulated_actuator_is_the_programmatic_safe_default() -> None:
    config = load_config("configs/development.yaml")

    actuator = _build_actuator(config, make_options())

    assert isinstance(actuator, SimulatedPTZActuator)
    actuator.close()


def test_arduino_backend_is_rejected_without_arm_before_source_creation() -> None:
    source_constructed = False

    def source_factory(_source: object) -> FakeSource:
        nonlocal source_constructed
        source_constructed = True
        return FakeSource([])

    with pytest.raises(ValueError, match="--arm-hardware"):
        run_vision(
            load_config("configs/hardware.yaml"),
            make_options(backend="arduino_serial"),
            source_factory=source_factory,
        )

    assert source_constructed is False


def test_arduino_backend_rejects_missing_explicit_port() -> None:
    config = load_config("configs/hardware.yaml")
    assert config.serial is not None
    config = replace(config, serial=replace(config.serial, device=""))

    with pytest.raises(ValueError, match="explicit serial port"):
        run_vision(
            config,
            make_options(backend="arduino_serial", arm_hardware=True),
        )


@pytest.mark.parametrize(
    "options",
    [
        make_options(arm_hardware=True),
        make_options(serial_port="/dev/meaningless"),
    ],
)
def test_simulated_backend_rejects_hardware_only_options(
    options: VisionOptions,
) -> None:
    with pytest.raises(ValueError, match="incompatible with simulated"):
        run_vision(load_config("configs/development.yaml"), options)


def test_cli_hardware_override_cannot_bypass_validated_hardware_configuration() -> None:
    with pytest.raises(ConfigError, match="actuator.backend: arduino_serial"):
        run_vision(
            load_config("configs/development.yaml"),
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
        )


def test_camera_initialization_failure_occurs_before_actuator_construction() -> None:
    actuator_constructed = False

    def actuator_factory(_config: AppConfig, _options: VisionOptions) -> FakeActuator:
        nonlocal actuator_constructed
        actuator_constructed = True
        return FakeActuator()

    with pytest.raises(RuntimeError, match="camera init"):
        run_vision(
            load_config("configs/hardware.yaml"),
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source_factory=lambda _source: (_ for _ in ()).throw(
                RuntimeError("camera init failed")
            ),
            actuator_factory=actuator_factory,
        )

    assert actuator_constructed is False


def test_model_initialization_failure_closes_source_before_actuator_exists() -> None:
    source = FakeSource([])
    actuator_constructed = False

    def actuator_factory(_config: AppConfig, _options: VisionOptions) -> FakeActuator:
        nonlocal actuator_constructed
        actuator_constructed = True
        return FakeActuator()

    with pytest.raises(RuntimeError, match="model init"):
        run_vision(
            load_config("configs/hardware.yaml"),
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source_factory=lambda _source: source,
            detector_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("model init failed")
            ),
            actuator_factory=actuator_factory,
        )

    assert source.close_count == 1
    assert actuator_constructed is False


def test_hardware_startup_and_normal_eof_use_safe_order() -> None:
    events: list[str] = []
    source = FakeSource([None], events)
    detector = FakeDetector([], events)
    actuator = FakeActuator(events)

    processed = run_vision(
        load_config("configs/hardware.yaml"),
        make_options(
            backend="arduino_serial",
            arm_hardware=True,
        ),
        source_factory=lambda _source: events.append("source:init") or source,
        detector_factory=lambda *_args, **_kwargs: events.append("model:init")
        or detector,
        actuator_factory=lambda _config, _options: events.append("actuator:init")
        or actuator,
        emit=lambda _line: None,
    )

    assert processed == 0
    assert events == [
        "source:init",
        "model:init",
        "actuator:init",
        "actuator:open",
        "actuator:enable",
        "actuator:disable",
        "actuator:close",
        "source:close",
    ]


def test_camera_read_failure_after_enable_stops_before_motion() -> None:
    events: list[str] = []
    source = FakeSource([RuntimeError("camera read failed")], events)
    actuator = FakeActuator(events)

    with pytest.raises(RuntimeError, match="camera read failed"):
        run_fake(
            load_config("configs/hardware.yaml"),
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source,
            FakeDetector([], events),
            actuator,
        )

    assert actuator.applied == []
    assert events[-3:] == [
        "actuator:disable",
        "actuator:close",
        "source:close",
    ]


def test_max_frame_completion_closes_without_reading_an_extra_frame() -> None:
    source = FakeSource([make_frame(0), make_frame(1)])
    detector = FakeDetector([()])
    actuator = FakeActuator()

    processed = run_fake(
        load_config("configs/development.yaml"),
        make_options(max_frames=1),
        source,
        detector,
        actuator,
    )

    assert processed == 1
    assert source.read_count == 1
    assert len(actuator.applied) == 1
    assert actuator.close_count == 1
    assert source.close_count == 1


def test_termination_request_uses_normal_hardware_cleanup_without_motion() -> None:
    source = FakeSource([make_frame(0)])
    detector = FakeDetector([()])
    actuator = FakeActuator()

    processed = run_vision(
        load_config("configs/hardware.yaml"),
        make_options(
            backend="arduino_serial",
            arm_hardware=True,
            serial_port="/dev/fake",
        ),
        source_factory=lambda _source: source,
        detector_factory=lambda *_args, **_kwargs: detector,
        actuator_factory=lambda _config, _options: actuator,
        stop_requested=lambda: True,
        emit=lambda _line: None,
    )

    assert processed == 0
    assert source.read_count == 0
    assert actuator.applied == []
    assert actuator.disable_count == 0
    assert actuator.close_count == 0
    assert source.close_count == 1


def test_termination_requested_during_open_never_enables_and_cleans_once() -> None:
    state = CancellationState()
    events: list[str] = []
    source = FakeSource([make_frame(0)], events)

    class OpenCancellingActuator(FakeActuator):
        def open(self) -> None:
            super().open()
            state.request()

    actuator = OpenCancellingActuator(events)

    processed = run_fake(
        load_config("configs/hardware.yaml"),
        make_options(
            backend="arduino_serial",
            arm_hardware=True,
            serial_port="/dev/fake",
        ),
        source,
        FakeDetector([()], events),
        actuator,
        stop_requested=state,
    )

    assert processed == 0
    assert events == [
        "actuator:open",
        "actuator:disable",
        "actuator:close",
        "source:close",
    ]
    assert actuator.disable_count == actuator.close_count == source.close_count == 1


def test_termination_observed_immediately_after_open_never_enables() -> None:
    events: list[str] = []
    source = FakeSource([make_frame(0)], events)
    actuator = FakeActuator(events)

    processed = run_fake(
        load_config("configs/hardware.yaml"),
        make_options(
            backend="arduino_serial",
            arm_hardware=True,
            serial_port="/dev/fake",
        ),
        source,
        FakeDetector([()], events),
        actuator,
        stop_requested=lambda: "actuator:open" in events,
    )

    assert processed == 0
    assert "actuator:enable" not in events
    assert actuator.disable_count == actuator.close_count == source.close_count == 1


def test_termination_during_detector_inference_prevents_apply() -> None:
    state = CancellationState()
    events: list[str] = []
    source = FakeSource([make_frame(0)], events)

    class CancellingDetector(FakeDetector):
        def detect(self, frame: Frame) -> tuple[Detection, ...]:
            state.request()
            return super().detect(frame)

    actuator = FakeActuator(events)
    processed = run_fake(
        load_config("configs/hardware.yaml"),
        make_options(
            backend="arduino_serial",
            arm_hardware=True,
            serial_port="/dev/fake",
        ),
        source,
        CancellingDetector([()], events),
        actuator,
        stop_requested=state,
    )

    assert processed == 0
    assert actuator.applied == []
    assert actuator.disable_count == actuator.close_count == source.close_count == 1


def test_unexpected_detector_cancellation_fails_and_cleans_up_once() -> None:
    events: list[str] = []
    source = FakeSource([make_frame(0)], events)
    detector = FakeDetector([OperationCancelled("detector cancelled")], events)
    actuator = FakeActuator(events)

    with pytest.raises(UnexpectedCancellationError) as captured:
        run_fake(
            load_config("configs/hardware.yaml"),
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source,
            detector,
            actuator,
        )

    assert isinstance(captured.value.__cause__, OperationCancelled)
    assert str(captured.value.__cause__) == "detector cancelled"
    assert actuator.applied == []
    assert actuator.disable_count == actuator.close_count == source.close_count == 1


def test_unexpected_actuator_cancellation_fails_and_cleans_up_once() -> None:
    events: list[str] = []
    source = FakeSource([make_frame(0)], events)

    class CancellingActuator(FakeActuator):
        def apply(self, command: PTZCommand) -> None:
            super().apply(command)
            raise OperationCancelled("actuator cancelled")

    actuator = CancellingActuator(events)

    with pytest.raises(UnexpectedCancellationError) as captured:
        run_fake(
            load_config("configs/hardware.yaml"),
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source,
            FakeDetector([(Detection("boat", 0.9, 70, 70, 90, 90),)], events),
            actuator,
        )

    assert isinstance(captured.value.__cause__, OperationCancelled)
    assert str(captured.value.__cause__) == "actuator cancelled"
    assert actuator.disable_count == actuator.close_count == source.close_count == 1


def test_termination_during_selection_prevents_controller_and_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = CancellationState()
    events: list[str] = []
    source = FakeSource([make_frame(0)], events)
    actuator = FakeActuator(events)

    class CancellingSelector:
        def __init__(self, _classes: tuple[str, ...]) -> None:
            pass

        def select(
            self,
            frame: Frame,
            detections: tuple[Detection, ...],
        ) -> None:
            state.request()
            return None

    monkeypatch.setattr(vision_cli, "MarineTargetSelector", CancellingSelector)
    processed = run_fake(
        load_config("configs/hardware.yaml"),
        make_options(
            backend="arduino_serial",
            arm_hardware=True,
            serial_port="/dev/fake",
        ),
        source,
        FakeDetector([()], events),
        actuator,
        stop_requested=state,
    )

    assert processed == 0
    assert actuator.applied == []
    assert actuator.disable_count == actuator.close_count == source.close_count == 1


def test_termination_after_control_calculation_prevents_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = CancellationState()
    events: list[str] = []
    source = FakeSource([make_frame(0)], events)
    actuator = FakeActuator(events)

    class CancellingController:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def command_for_target(
            self,
            frame: Frame,
            _target: object,
        ) -> PTZCommand:
            state.request()
            return PTZCommand(90.0, 90.0, frame.sequence)

    monkeypatch.setattr(vision_cli, "ProportionalPTZController", CancellingController)
    processed = run_fake(
        load_config("configs/hardware.yaml"),
        make_options(
            backend="arduino_serial",
            arm_hardware=True,
            serial_port="/dev/fake",
        ),
        source,
        FakeDetector([()], events),
        actuator,
        stop_requested=state,
    )

    assert processed == 0
    assert actuator.applied == []
    assert actuator.disable_count == actuator.close_count == source.close_count == 1


def test_signal_handler_only_records_termination_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: dict[int, Any] = {}

    monkeypatch.setattr(vision_cli.signal, "getsignal", lambda _number: "previous")

    def register(number: int, handler: Any) -> None:
        if callable(handler):
            installed[number] = handler

    monkeypatch.setattr(vision_cli.signal, "signal", register)
    with vision_cli._termination_signals() as request:
        installed[signal.SIGTERM](signal.SIGTERM, None)
        assert request() is True
        assert request.signal_number == signal.SIGTERM


def test_serial_handshake_failure_never_enables_and_closes_everything() -> None:
    events: list[str] = []
    source = FakeSource([None], events)
    actuator = FakeActuator(events, open_error=RuntimeError("handshake failed"))

    with pytest.raises(RuntimeError, match="handshake failed"):
        run_fake(
            load_config("configs/hardware.yaml"),
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source,
            FakeDetector([], events),
            actuator,
        )

    assert "actuator:enable" not in events
    assert "actuator:disable" not in events
    assert events[-2:] == ["actuator:close", "source:close"]


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("detector failed"), KeyboardInterrupt()],
)
def test_detector_failure_after_enable_disables_and_closes(
    failure: BaseException,
) -> None:
    events: list[str] = []
    source = FakeSource([make_frame(0)], events)
    detector = FakeDetector([failure], events)
    actuator = FakeActuator(events)

    with pytest.raises(type(failure)):
        run_fake(
            load_config("configs/hardware.yaml"),
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source,
            detector,
            actuator,
        )

    assert actuator.applied == []
    assert events[-3:] == [
        "actuator:disable",
        "actuator:close",
        "source:close",
    ]


def test_protocol_fault_stops_all_later_motion_and_cleans_up() -> None:
    source = FakeSource([make_frame(0), make_frame(1), make_frame(2), None])
    detector = FakeDetector(
        [
            (Detection("boat", 0.9, 70, 40, 90, 60),),
            (Detection("boat", 0.9, 70, 40, 90, 60),),
            (Detection("boat", 0.9, 70, 40, 90, 60),),
        ]
    )
    actuator = FakeActuator(apply_error_at=2)

    with pytest.raises(RuntimeError, match="protocol fault"):
        run_fake(
            load_config("configs/hardware.yaml"),
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source,
            detector,
            actuator,
        )

    assert len(actuator.applied) == 2
    assert detector.calls == 2
    assert actuator.disable_count == 1
    assert actuator.close_count == 1
    assert source.close_count == 1


def test_malformed_frame_fails_before_detection_or_motion() -> None:
    malformed = Frame(FakeImage(), 0.0, 200, 100, 0)
    source = FakeSource([malformed])
    detector = FakeDetector([()])
    actuator = FakeActuator()

    with pytest.raises(RuntimeError, match="does not match image shape"):
        run_fake(
            load_config("configs/development.yaml"),
            make_options(),
            source,
            detector,
            actuator,
        )

    assert detector.calls == 0
    assert actuator.applied == []
    assert actuator.close_count == 1
    assert source.close_count == 1


def test_cleanup_is_ordered_and_idempotent() -> None:
    events: list[str] = []
    source = FakeSource([], events)
    actuator = FakeActuator(events)
    resources = _RuntimeResources(
        display=False,
        source=source,
        actuator=actuator,
        disable_needed=True,
    )

    assert resources.close() == ()
    assert resources.close() == ()

    assert events == [
        "actuator:disable",
        "actuator:close",
        "source:close",
    ]
    assert actuator.disable_count == 1
    assert actuator.close_count == 1
    assert source.close_count == 1


def test_cleanup_failure_is_reported_without_hiding_tracking_fault() -> None:
    source = FakeSource([make_frame(0)])
    detector = FakeDetector([(Detection("boat", 0.9, 70, 40, 90, 60),)])
    actuator = FakeActuator(
        apply_error_at=1,
        disable_error=RuntimeError("disable failed"),
    )

    with pytest.raises(RuntimeCleanupError, match="cleanup also failed") as captured:
        run_fake(
            load_config("configs/hardware.yaml"),
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source,
            detector,
            actuator,
        )

    assert "protocol fault" in str(captured.value)
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "protocol fault"
    assert actuator.close_count == 1
    assert source.close_count == 1


def test_rate_limiter_uses_tracking_update_rate() -> None:
    source = FakeSource([make_frame(0), make_frame(1), None])
    detector = FakeDetector([(), ()])
    actuator = FakeActuator()
    clock = iter((1.0, 1.05))
    sleeps: list[float] = []

    run_vision(
        load_config("configs/development.yaml"),
        make_options(),
        source_factory=lambda _source: source,
        detector_factory=lambda *_args, **_kwargs: detector,
        actuator_factory=lambda _config, _options: actuator,
        rate_monotonic=lambda: next(clock),
        sleep=sleeps.append,
        emit=lambda _line: None,
    )

    assert sleeps == pytest.approx([0.05])


def test_runtime_composes_real_arduino_actuator_with_firmware_model() -> None:
    config = load_config("configs/hardware.yaml")
    firmware = production_firmware(config)

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        responses = firmware.handle_frame(frame)
        if isinstance(command, HelloCommand):
            return (firmware.startup_frame(), *responses)
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = production_actuator(config, transport)
    source = FakeSource([make_frame(0), None])
    detector = FakeDetector([(Detection("boat", 0.9, 70, 70, 90, 90),)])

    processed = run_vision(
        config,
        make_options(
            backend="arduino_serial",
            arm_hardware=True,
            serial_port="/dev/fake",
        ),
        source_factory=lambda _source: source,
        detector_factory=lambda *_args, **_kwargs: detector,
        actuator_factory=lambda _config, _options: actuator,
        rate_monotonic=lambda: 10.0,
        sleep=lambda _seconds: None,
        emit=lambda _line: None,
    )

    commands = [parse_command(frame) for frame in transport.writes]
    assert processed == 1
    assert actuator.startup_ready_seen is True
    assert isinstance(commands[0], HelloCommand)
    assert isinstance(commands[1], DisableCommand)
    assert isinstance(commands[2], EnableCommand)
    assert isinstance(commands[3], SetCommand)
    assert isinstance(commands[-1], DisableCommand)
    assert [command.sequence for command in commands[:4]] == [1, 2, 3, 4]
    assert (firmware.pan_deg, firmware.tilt_deg) == (93, 93)
    assert firmware.enabled is False
    assert actuator.closed is True


def test_runtime_enable_ack_loss_is_conservative_and_watchdog_recovers() -> None:
    config = load_config("configs/hardware.yaml")
    now = [0.0]
    firmware = production_firmware(config, clock=lambda: now[0])
    enable_accepted = [False]

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        if isinstance(command, HelloCommand) and not enable_accepted[0]:
            return (firmware.startup_frame(), *firmware.handle_frame(frame))
        if isinstance(command, DisableCommand) and not enable_accepted[0]:
            return firmware.handle_frame(frame)
        if isinstance(command, EnableCommand):
            if not enable_accepted[0]:
                enable_accepted[0] = True
                firmware.handle_frame(frame)
            return ()
        return ()

    transport = InMemorySerialTransport(responder)
    actuator = production_actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )

    with pytest.raises(RuntimeCleanupError, match="termination|runtime failed") as captured:
        run_vision(
            config,
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source_factory=lambda _source: FakeSource([make_frame(0)]),
            detector_factory=lambda *_args, **_kwargs: FakeDetector([()]),
            actuator_factory=lambda _config, _options: actuator,
            rate_monotonic=lambda: now[0],
            sleep=lambda _seconds: None,
            emit=lambda _line: None,
        )

    commands = [parse_command(frame) for frame in transport.writes]
    assert isinstance(captured.value.__cause__, SerialActuatorTimeoutError)
    assert enable_accepted[0] is True
    assert firmware.enabled is True
    assert not any(isinstance(command, SetCommand) for command in commands)
    assert any(isinstance(command, DisableCommand) for command in commands[2:])
    assert actuator.closed is True
    assert actuator.connected is False
    assert actuator.enabled is False
    assert actuator.position_uncertain is True
    firmware.tick()
    assert firmware.enabled is False


def test_runtime_set_ack_loss_stops_later_motion_and_preserves_confirmed_state() -> None:
    config = load_config("configs/hardware.yaml")
    now = [0.0]
    firmware = production_firmware(config, clock=lambda: now[0])
    set_commands: list[SetCommand] = []
    set_accepted = [False]

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        if isinstance(command, SetCommand):
            set_commands.append(command)
            firmware.handle_frame(frame)
            set_accepted[0] = True
            return ()
        if set_accepted[0]:
            return ()
        return firmware.handle_frame(frame)

    transport = InMemorySerialTransport(responder)
    actuator = production_actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )
    source = FakeSource([make_frame(0), make_frame(1), None])
    detector = FakeDetector(
        [
            (Detection("boat", 0.9, 70, 70, 90, 90),),
            (Detection("boat", 0.9, 70, 70, 90, 90),),
        ]
    )

    with pytest.raises(RuntimeCleanupError, match="runtime failed") as captured:
        run_vision(
            config,
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source_factory=lambda _source: source,
            detector_factory=lambda *_args, **_kwargs: detector,
            actuator_factory=lambda _config, _options: actuator,
            rate_monotonic=lambda: 10.0,
            sleep=lambda _seconds: None,
            emit=lambda _line: None,
        )

    assert source.read_count == detector.calls == 1
    assert isinstance(captured.value.__cause__, SerialActuatorTimeoutError)
    assert len(set_commands) == config.serial.retries + 1  # type: ignore[union-attr]
    assert len({command.sequence for command in set_commands}) == 1
    assert (actuator.pan_deg, actuator.tilt_deg) == (90.0, 90.0)
    assert actuator.position_uncertain is True
    assert (firmware.pan_deg, firmware.tilt_deg) == (93, 93)
    assert firmware.enabled is True
    firmware.tick()
    assert firmware.enabled is False


def test_runtime_set_protocol_fault_stops_motion_and_keeps_coordinates_unconfirmed() -> None:
    config = load_config("configs/hardware.yaml")
    now = [0.0]
    firmware = production_firmware(config, clock=lambda: now[0])
    set_commands: list[SetCommand] = []
    set_accepted = [False]

    def responder(frame: bytes) -> tuple[bytes, ...]:
        command = parse_command(frame)
        if set_accepted[0]:
            return ()
        responses = firmware.handle_frame(frame)
        if isinstance(command, SetCommand):
            set_commands.append(command)
            set_accepted[0] = True
            return (
                encode_response(
                    AckResponse(command.sequence, "CENTER", command.pan_deg, command.tilt_deg)
                ),
            )
        return responses

    transport = InMemorySerialTransport(responder)
    actuator = production_actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )

    with pytest.raises(RuntimeCleanupError, match="runtime failed"):
        run_vision(
            config,
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source_factory=lambda _source: FakeSource([make_frame(0), make_frame(1)]),
            detector_factory=lambda *_args, **_kwargs: FakeDetector(
                [
                    (Detection("boat", 0.9, 70, 70, 90, 90),),
                    (Detection("boat", 0.9, 70, 70, 90, 90),),
                ]
            ),
            actuator_factory=lambda _config, _options: actuator,
            rate_monotonic=lambda: 10.0,
            sleep=lambda _seconds: None,
            emit=lambda _line: None,
        )

    assert len(set_commands) == 1
    assert (actuator.pan_deg, actuator.tilt_deg) == (90.0, 90.0)
    assert actuator.position_uncertain is True
    assert firmware.enabled is True
    firmware.tick()
    assert firmware.enabled is False


def test_runtime_handshake_failure_never_enables_firmware() -> None:
    config = load_config("configs/hardware.yaml")
    now = [0.0]
    firmware = production_firmware(config, clock=lambda: now[0])
    transport = InMemorySerialTransport(lambda _frame: ())
    actuator = production_actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )

    with pytest.raises(SerialActuatorTimeoutError):
        run_vision(
            config,
            make_options(
                backend="arduino_serial",
                arm_hardware=True,
                serial_port="/dev/fake",
            ),
            source_factory=lambda _source: FakeSource([make_frame(0)]),
            detector_factory=lambda *_args, **_kwargs: FakeDetector([()]),
            actuator_factory=lambda _config, _options: actuator,
            rate_monotonic=lambda: now[0],
            sleep=lambda _seconds: None,
            emit=lambda _line: None,
        )

    assert firmware.enabled is False
    assert all(
        not isinstance(parse_command(frame), EnableCommand)
        for frame in transport.writes
    )


def test_runtime_cancellation_during_real_actuator_rate_wait_writes_no_second_set() -> None:
    config = load_config("configs/hardware.yaml")
    now = [0.0]
    cancellation = CancellationState()
    firmware = production_firmware(config, clock=lambda: now[0])
    transport = InMemorySerialTransport(firmware.handle_frame)

    def serial_sleep(duration: float) -> None:
        now[0] += duration
        cancellation.request()

    actuator = production_actuator(
        config,
        transport,
        monotonic=lambda: now[0],
        sleep=serial_sleep,
    )
    source = FakeSource([make_frame(0), make_frame(1), None])
    detector = FakeDetector(
        [
            (Detection("boat", 0.9, 70, 70, 90, 90),),
            (Detection("boat", 0.9, 70, 70, 90, 90),),
        ]
    )

    processed = run_vision(
        config,
        make_options(
            backend="arduino_serial",
            arm_hardware=True,
            serial_port="/dev/fake",
        ),
        source_factory=lambda _source: source,
        detector_factory=lambda *_args, **_kwargs: detector,
        actuator_factory=lambda _config, _options: actuator,
        rate_monotonic=lambda: 10.0,
        sleep=lambda _seconds: None,
        stop_requested=cancellation,
        emit=lambda _line: None,
    )

    set_commands = [
        parse_command(frame)
        for frame in transport.writes
        if isinstance(parse_command(frame), SetCommand)
    ]
    assert processed == 1
    assert source.read_count == detector.calls == 2
    assert len(set_commands) == 1
    assert (firmware.pan_deg, firmware.tilt_deg) == (93, 93)
    assert firmware.enabled is False
