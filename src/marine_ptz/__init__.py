"""Typed interfaces for the marine PTZ tracker."""

from .interfaces import Actuator, CameraSource, Detector, PTZController, TargetSelector
from .types import Detection, Frame, PTZCommand, Target

__all__ = [
    "Actuator",
    "CameraSource",
    "Detection",
    "Detector",
    "Frame",
    "PTZCommand",
    "PTZController",
    "Target",
    "TargetSelector",
]
