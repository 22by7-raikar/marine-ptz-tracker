from __future__ import annotations

import subprocess
import sys


def test_default_serial_demo_uses_only_in_memory_firmware() -> None:
    result = subprocess.run(
        [sys.executable, "tools/serial_protocol_demo.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "mode=fake" in result.stdout
    assert "connected=True enabled=True" in result.stdout
