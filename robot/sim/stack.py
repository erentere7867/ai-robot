"""Sprocket in a Habitat home, the whole way down: the world and body (robot/sim/habitat_server.py), the drivers
(robot/sim/habitat_ros.py), the robot's own software (robot/ros/bringup.py) and the robot itself (RealRobot), as on
the real robot. Nothing of the truth reaches the robot: the first time in a home it explores and builds its own map
(slam_toolbox); afterwards it localises in that map (AMCL), from its dock, where its map's frame begins.

    home = Home("apt_0")                     # explores and saves maps/apt_0.yaml the first time
    robot = home.robot(detector, brain)      # a RealRobot at its dock that knows only its own map
    home.reset()                             # carried back to the dock (between eval tasks)
    home.truth, home.path                    # the judge's: the objects, and where the body really went
    home.close()
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from robot import paths
from robot.ros import bringup
from robot.sim.habitat import Habitat, start

MAPS = paths.MAPS
RUNS = paths.RUNS
RTF = 0.33  # simulated seconds per wall second: the robot's software gets about the board's work done per second
SIMULATOR = ("robot.sim.habitat_ros", "habitat_server.py")  # what stands in for the home and the hardware


def own_home(map_yaml: Path) -> tuple[Any, np.ndarray]:
    """What the robot knows of a home before it looks around: the extent of the map it built (its frame starts at the
    dock) and that map as its planning grid. No objects, rooms or places yet: those come from looking."""
    from robot.agent.api import Pose
    from robot.ros.explore import _pgm, load_grid
    from robot.sim.world import RES, Scene

    text = map_yaml.read_text()
    res = float(re.search(r"resolution: ([\d.]+)", text).group(1))
    ox, oy, _ = (float(v) for v in re.search(r"origin: \[([^\]]+)\]", text).group(1).split(","))
    rows, cols = _pgm(map_yaml.with_name(re.search(r"image: (.+)", text).group(1).strip())).shape
    half_w = max(abs(ox), abs(ox + cols * res)) + 0.5  # the robot's grid is centred on its frame's origin
    half_h = max(abs(oy), abs(oy + rows * res)) + 0.5
    size = (2 * half_w, 2 * half_h)
    shape = (round(size[0] / RES), round(size[1] / RES))
    return Scene(size[0], size[1], [], {}, start=Pose(0.0, 0.0, 0.0)), load_grid(map_yaml, shape, size)


class Home:
    def __init__(self, scene: str = "apt_0", rtf: float = RTF, explore_s: float = 600.0, seed: int = 0) -> None:
        """`explore_s`: simulated seconds the robot may spend mapping a home it has not seen before."""
        bringup.clear(bringup.LEFTOVERS + SIMULATOR)  # whatever an earlier run left behind
        self.scene, self.out = scene, RUNS / f"home_{scene}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.server, port = start(scene)
        self.world = Habitat(port)
        self.truth = self.world.call(op="info")  # the judge's: never handed to the robot
        with open(self.out / "hardware.log", "w") as log:
            self.hardware = subprocess.Popen(
                [sys.executable, "-u", "-m", "robot.sim.habitat_ros", "--port", str(port), "--rtf", str(rtf),
                 "--seed", str(seed)], stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        self.software: bringup.Running | None = None
        self._judge_node()
        self.map = MAPS / f"{scene}.yaml"
        if not self.map.exists():
            self.explored = self._explore(explore_s)
        self.software = bringup.start(self.out, map_yaml=self.map)

    # ---- the judge's view ----------------------------------------------------------------------------------------
    def _judge_node(self) -> None:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from std_msgs.msg import Empty, Int32

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("sprocket_judge", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.path: list[tuple[float, float, float, float]] = []  # t, x, y, yaw: where the body really was
        self.bumps = 0

        def truth(msg: Any) -> None:
            p, s = msg.pose.pose, msg.header.stamp
            self.path.append((s.sec + s.nanosec * 1e-9, p.position.x, p.position.y,
                              2 * math.atan2(p.orientation.z, p.orientation.w)))

        self.node.create_subscription(Odometry, "/ground_truth", truth, 50)
        self.node.create_subscription(Int32, "/ground_truth/bumps", lambda m: setattr(self, "bumps", m.data), 5)
        self._reset = self.node.create_publisher(Empty, "/sim/reset", 1)
        from geometry_msgs.msg import PoseWithCovarianceStamped

        self._initial = self.node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)  # kept: a new one
        # would not be found by AMCL before its few messages were gone
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        threading.Thread(target=executor.spin, daemon=True).start()
        while not self.path:
            time.sleep(0.1)

    def now(self) -> float:
        return self.node.get_clock().now().nanoseconds / 1e9

    def wait(self, seconds: float) -> None:
        """Let `seconds` of simulated time pass."""
        end = self.now() + seconds
        while self.now() < end:
            if self.hardware.poll() is not None:  # it owns the clock: without it, time stands still
                raise RuntimeError(f"the simulated hardware stopped; see {self.out / 'hardware.log'}")
            time.sleep(0.02)

    def dock(self) -> tuple[float, float, float]:
        """Where the robot's frame starts, in the house's frame (the judge's knowledge)."""
        return tuple(self.truth["start"])

    # ---- the robot ---------------------------------------------------------------------------------------------
    def _explore(self, seconds: float) -> dict:
        """The first time in this home: map it with slam_toolbox while exploring, and save the map."""
        from robot.agent.api import Pose
        from robot.ros.explore import Explorer
        from robot.ros.robot import RosRobot
        from robot.sim.world import Scene

        self.software = bringup.start(self.out / "explore")
        robot = RosRobot(Scene(40.0, 40.0, [], {}, start=Pose(0.0, 0.0, 0.0)), max_time=seconds + 600.0)
        explorer = Explorer(robot)
        try:
            stats = explorer.explore(seconds)
        finally:  # whatever ended the exploring, the map so far is the robot's map
            MAPS.mkdir(parents=True, exist_ok=True)
            explorer.save(self.map.with_suffix(""), (0.0, 0.0))
            robot.shutdown()
            self.software.stop()
            self.software = None
        print(f"explored {self.scene}: {stats}", flush=True)
        self.reset(localise=False)
        return stats

    def robot(self, detector: Any, brain: Any = None, store: Any = None) -> Any:
        """A RealRobot at the dock that knows only its own map."""
        from robot.ros.real import RealRobot

        scene, grid = own_home(self.map)
        return RealRobot(scene, detector=detector, brain=brain, store=store, grid=grid,
                         places_file=self.map.with_suffix(".places.json"))

    def reset(self, localise: bool = True) -> None:
        """Carry the robot back to its dock and tell its localisation so (as when it is put back on its dock)."""
        from std_msgs.msg import Empty

        self._reset.publish(Empty())
        self.wait(0.5)
        if localise:
            from geometry_msgs.msg import PoseWithCovarianceStamped

            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = "map"
            msg.pose.pose.orientation.w = 1.0
            msg.pose.covariance[0] = msg.pose.covariance[7] = 0.01
            msg.pose.covariance[35] = 0.005
            for _ in range(3):
                msg.header.stamp = self.node.get_clock().now().to_msg()
                self._initial.publish(msg)
                self.wait(0.3)

    def close(self) -> None:
        if self.software is not None:
            self.software.stop()
        self.hardware.kill()  # a ROS node ignores SIGTERM
        self.server.terminate()
        self.hardware.wait()
        self.server.wait()
        self.node.destroy_node()
