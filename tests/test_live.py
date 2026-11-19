import shutil
import subprocess
from pathlib import Path

import pytest

from robot.interfaces.live import PAGE

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js is needed to run the dashboard JavaScript")
def test_dashboard_connection_and_command_delivery():
    result = subprocess.run(
        [NODE, str(Path(__file__).with_name("live_dashboard.cjs"))],
        input=PAGE,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
