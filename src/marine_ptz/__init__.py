"""Typed interfaces for the marine PTZ tracker."""

from .actuators import SimulatedPTZActuator
from .config import AppConfig, ConfigError, TrackingConfig, load_config
from .interfaces import Actuator, CameraSource, Detector, PTZController, TargetSelector
from .synthetic import SyntheticCamera, SyntheticTargetDetector
from .tracking import (
    HighestConfidenceTargetSelector,
    MarineTargetSelector,
    ProportionalPTZController,
)
from .types import AngleLimits, Detection, Frame, PTZCommand, Target

__all__ = [
    "Actuator",
    "AngleLimits",
    "AppConfig",
    "CameraSource",
    "ConfigError",
    "Detection",
    "Detector",
    "Frame",
    "HighestConfidenceTargetSelector",
    "MarineTargetSelector",
    "PTZCommand",
    "PTZController",
    "ProportionalPTZController",
    "SimulatedPTZActuator",
    "SyntheticCamera",
    "SyntheticTargetDetector",
    "Target",
    "TargetSelector",
    "TrackingConfig",
    "load_config",
]
