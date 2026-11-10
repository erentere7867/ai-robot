"""The MuJoCo simulator still drives and senses (it becomes the ROS 2 world in phase 2)."""

import numpy as np
import pytest


def test_robot_drives_and_senses_in_mujoco():
    pytest.importorskip("mujoco")
    from robot.body.hal import load_capabilities
    from robot.sim.physics_body import PhysicsBody
    from robot.sim.sandbox import SimulationSandbox
    from robot.sim.sensors import SimLidar

    class Clock:
        t = 0.0

        def monotonic(self):
            self.t += 1.0
            return self.t

    room = SimulationSandbox()
    caps = load_capabilities()
    body = PhysicsBody(caps, sandbox=room)
    start = body.true_pose().x_mm
    for _ in range(100):
        body.command(250.0, 0.0)
        body.advance(0.02)
    assert body.true_pose().x_mm - start > 300.0
    assert room.render_rgb("front", body.true_pose()).shape == (480, 640, 3)
    scan = SimLidar(room, body, Clock(), mount_x_mm=caps.lidar_x_mm, height_mm=caps.lidar_z_mm).latest()
    assert np.isfinite(scan.ranges_mm).sum() > 300
