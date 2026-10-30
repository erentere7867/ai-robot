"""Suite-wide isolation: no test may read or write the robot's real runtime files."""

from __future__ import annotations

import os

import pytest

from robot.core.config import RobotConfig

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(autouse=True)
def _isolated_runtime_files(monkeypatch, tmp_path):
    """Configs loaded by tests keep the spatial map and recordings under the test's tmp dir."""
    original = RobotConfig.from_yaml.__func__

    def from_yaml(cls, *args, **kwargs):
        cfg = original(cls, *args, **kwargs)
        cfg.spatial_map_path = tmp_path / "isolated_spatial_map.json"
        cfg.recordings_dir = tmp_path / "isolated_recordings"
        cfg.llm.auto_start_server = False
        return cfg

    monkeypatch.setattr(RobotConfig, "from_yaml", classmethod(from_yaml))
