"""Pytest collection policy for hardware-safe default test runs."""

from __future__ import annotations

import pytest

_CATEGORY_MARKERS = {"unit", "integration", "hardware", "gpu"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify unmarked tests as hardware-free units by default."""
    for item in items:
        if not any(item.get_closest_marker(name) for name in _CATEGORY_MARKERS):
            item.add_marker(pytest.mark.unit)
