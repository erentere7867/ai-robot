"""Sprocket's own software on ROS, above the drivers: the same on the robot as in the simulator.

    robot_state_publisher   the body's frames (robot/ros/model.py)
    robot_localization EKF  wheels + gyro -> odom -> base_link
    depth node              monocular depth for the three cameras -> synthetic /scan (robot/ros/perceive.py)
    Nav2                    with slam_toolbox the first time in a home (it builds the map while the robot explores),
                            afterwards with AMCL in the map the robot built

Only the drivers differ between the robot and the simulator (robot/sim/habitat_ros.py stands in for them).
Everything runs on simulated time when a /clock is published.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from robot.ros.model import BODY_R, EKF, urdf

NAV2 = Path("/opt/ros/lyrical/share/nav2_bringup")
SLAM_DEFAULTS = Path("/opt/ros/lyrical/share/slam_toolbox/config/mapper_params_online_sync.yaml")
CRUISE, TURN = 0.35, 1.8  # driving limits indoors: m/s, rad/s (the wheels allow 0.5 and 2.0)
SEES = 3.0  # metres: the depth estimate marks obstacles this far out, and clears free space a little further
# What a run starts, by process name: a copy left over from an earlier run fights the new one (a second Nav2
# container answers to the same node names; a second EKF publishes its own odom -> base_link).
LEFTOVERS = ("component_container", "robot_state_publisher", "ekf_node", "robot.ros.perceive", "ros2 launch",
             "slam_toolbox")

# Localisation tuned to this robot: Nav2's defaults suit a lidar and wheel-only odometry. The odometry is fused with
# the gyro (alphas low). The camera scan's measured error (docs/simulation_3d.md): within 15% for 70-80% of returns at
# 0.3-3 m, about a quarter nearer or further than anything there, so a wider hit (0.2 m) and 30% random returns; 60
# beams, as the beams' errors are not independent (one misjudged wall spoils many).
AMCL = {"max_beams": 60, "update_min_d": 0.1, "update_min_a": 0.1, "alpha1": 0.1, "alpha2": 0.1, "alpha3": 0.1,
        "alpha4": 0.1, "z_hit": 0.7, "z_rand": 0.3, "sigma_hit": 0.2, "laser_max_range": SEES + 1.0,
        "transform_tolerance": 1.0, "set_initial_pose": True, "initial_pose": {"x": 0.0, "y": 0.0, "z": 0.0,
                                                                                "yaw": 0.0}}


def clear(names: tuple[str, ...] = LEFTOVERS) -> None:
    """Kill the processes so named (by default the robot's software) that are still running, and wait until they
    are gone."""
    patterns = [f"[{name[0]}]{name[1:]}" for name in names]
    for pattern in patterns:
        subprocess.run(["pkill", "-9", "-f", pattern], check=False)
    for _ in range(20):
        if all(subprocess.run(["pgrep", "-f", p], capture_output=True).returncode for p in patterns):
            return
        time.sleep(0.5)


def params(out: Path) -> Path:
    """Nav2's and slam_toolbox's parameters for this robot, written next to the run's logs."""
    p = yaml.safe_load((NAV2 / "params" / "nav2_params.yaml").read_text())
    p["amcl"]["ros__parameters"].update(AMCL)
    for name in ("local_costmap", "global_costmap"):
        c = p[name][name]["ros__parameters"]
        c["robot_radius"] = BODY_R
        c["inflation_layer"]["inflation_radius"] = 0.45  # Nav2's 0.70 is for a 0.22 m robot in wide halls
        c["transform_tolerance"] = 1.0
        layer = c["voxel_layer" if name == "local_costmap" else "obstacle_layer"]
        # inf is the depth scan's "floor seen clear along this beam": it must clear what an earlier, wrong reading
        # marked there (by default Nav2 drops inf readings, and a fake obstacle then stays for good)
        layer["scan"].update(obstacle_max_range=SEES, raytrace_max_range=SEES + 0.5, inf_is_valid=True)
    ctrl = p["controller_server"]["ros__parameters"]
    ctrl["FollowPath"].update(vx_max=CRUISE, wz_max=TURN)
    ctrl["transform_tolerance"] = 1.0
    smoother = p["velocity_smoother"]["ros__parameters"]
    smoother["max_velocity"], smoother["min_velocity"] = [CRUISE, 0.0, TURN], [-0.25, 0.0, -TURN]
    p["behavior_server"]["ros__parameters"]["transform_tolerance"] = 1.0
    cm = p["collision_monitor"]["ros__parameters"]
    cm["transform_tolerance"], cm["source_timeout"] = 1.0, 5.0
    slam = yaml.safe_load(SLAM_DEFAULTS.read_text())["slam_toolbox"]["ros__parameters"]
    # A cell is wall when this share of the beams through it end there: slam_toolbox's 0.1 suits a lidar, whose
    # rare stray reading is noise; a depth network's wrong readings come and go from frame to frame, a real wall's stay.
    slam.update(use_sim_time=True, base_frame="base_link", max_laser_range=SEES + 1.0, map_update_interval=2.0,
                minimum_travel_distance=0.2, minimum_travel_heading=0.2, transform_timeout=0.5,
                enable_interactive_mode=False, occupancy_threshold=0.3)
    p["slam_toolbox"] = {"ros__parameters": slam}
    path = out / "nav2_params.yaml"
    path.write_text(yaml.safe_dump(p, sort_keys=False))
    return path


class Running:
    """The processes of one bring-up; `stop` ends them all."""

    def __init__(self, procs: list[subprocess.Popen], log: Path) -> None:
        self.procs, self.log = procs, log

    def alive(self) -> bool:
        return all(p.poll() is None for p in self.procs)

    def stop(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
        for proc in self.procs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        clear()  # Nav2's container does not always leave with its launch


def start(out: Path, *, map_yaml: Path | None = None, depth_threads: int = 2) -> Running:
    """The robot's software, logging to out/robot.log: mapping with slam_toolbox when there is no `map_yaml`, else
    localising in it with AMCL from the dock (the map's origin, where the robot starts)."""
    out.mkdir(parents=True, exist_ok=True)
    description = out / "robot_description.yaml"
    description.write_text(yaml.safe_dump({"robot_state_publisher": {"ros__parameters": {
        "robot_description": urdf(), "use_sim_time": True}}}))
    ekf = out / "ekf.yaml"
    ekf.write_text(EKF)
    nav = ["ros2", "launch", str(NAV2 / "launch" / "bringup_launch.py"), "use_sim_time:=True", "autostart:=True",
           f"params_file:={params(out)}", "use_composition:=True", "use_respawn:=False",
           *(["slam:=False", f"map:={map_yaml}"] if map_yaml else ["slam:=True"])]
    log = out / "robot.log"
    sink = open(log, "a")  # noqa: SIM115 - closed with the process group
    procs = [subprocess.Popen(cmd, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
             for cmd in (
                 ["ros2", "run", "robot_state_publisher", "robot_state_publisher", "--ros-args", "--params-file",
                  str(description)],
                 ["ros2", "run", "robot_localization", "ekf_node", "--ros-args", "--params-file", str(ekf)],
                 [sys.executable, "-u", "-m", "robot.ros.perceive", "--depth", "--threads", str(depth_threads)],
                 nav)]
    return Running(procs, log)
