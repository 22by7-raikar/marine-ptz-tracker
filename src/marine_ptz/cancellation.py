"""Small, hardware-free cancellation signal shared by runtime components."""

from __future__ import annotations


class OperationCancelled(RuntimeError):
    """Raised before a new motion operation is handed to a transport."""
