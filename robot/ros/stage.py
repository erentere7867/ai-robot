"""Run suite cases in Gazebo with Nav2: the same room, program and check as in the 2D world.

Each case gets a fresh simulation. Its scene becomes a Gazebo world (walls as boxes, things as cylinders) and a
Nav2 map (the 2D world's occupancy grid), and Nav2's TurtleBot 4 simulation is launched in it with the robot at the
scene's start pose. Then the program goes through the same gate and runner as anywhere else, on `RosRobot`.

    python3 -m robot.ros.stage                 # the hand-written programs whose rooms hold still
    python3 -m robot.ros.stage --cases R1 O3   # some of them

Runs inside WSL (or on the board) with ROS 2 sourced. Imports of ROS stay inside functions: the runner's sandbox
process re-imports this module and must stay small.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from robot.sim.world import SimRobot

WALL_H, THING_H, PERSON_H = 0.6, 0.5, 1.7


def _box(name: str, x: float, y: float, yaw: float, size: tuple[float, float, float]) -> str:
    geometry = f"<box><size>{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}</size></box>"
    return (f'<model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} {size[2] / 2:.3f} 0 0 {yaw:.4f}</pose>'
            f'<link name="l"><collision name="c"><geometry>{geometry}</geometry></collision>'
            f'<visual name="v"><geometry>{geometry}</geometry></visual></link></model>')


def _cylinder(name: str, x: float, y: float, r: float, h: float) -> str:
    geometry = f"<cylinder><radius>{r:.3f}</radius><length>{h:.3f}</length></cylinder>"
    return (f'<model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} {h / 2:.3f} 0 0 0</pose>'
            f'<link name="l"><collision name="c"><geometry>{geometry}</geometry></collision>'
            f'<visual name="v"><geometry>{geometry}</geometry></visual></link></model>')


def export(world: SimRobot, out: Path) -> tuple[Path, Path]:
    """The scene as a Gazebo world and a Nav2 map: one source of truth for both simulators."""
    scene = world.scene
    w, h = scene.width, scene.height
    walls = [((-w / 2, -h / 2), (w / 2, -h / 2)), ((w / 2, -h / 2), (w / 2, h / 2)),
             ((w / 2, h / 2), (-w / 2, h / 2)), ((-w / 2, h / 2), (-w / 2, -h / 2)), *scene.walls]
    models = [_box(f"wall{k}", (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, math.atan2(b[1] - a[1], b[0] - a[0]),
                   (math.dist(a, b) + 0.1, 0.1, WALL_H)) for k, (a, b) in enumerate(walls)]
    for t in world.things.values():
        if t.person or t.solid:
            models.append(_cylinder(f"{t.label}{t.id}", t.x, t.y, t.radius, PERSON_H if t.person else THING_H))
    sdf = out / "world.sdf"
    sdf.write_text(f"""<?xml version="1.0"?>
<sdf version="1.6" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:arg name="headless" default="true"/>
  <world name="room">
    <physics name="1ms" type="ode"><max_step_size>0.003</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <light type="directional" name="sun"><pose>0 0 10 0 0 0</pose><direction>-0.5 0.1 -0.9</direction></light>
    <model name="ground"><static>true</static><link name="l">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></visual>
    </link></model>
    {"".join(models)}
  </world>
</sdf>
""")
    # Map: occupied 0, free 254, first row = largest y (the 2D grid is indexed [x, y]).
    grid = world.occupied.T[::-1]
    pgm = out / "map.pgm"
    pgm.write_bytes(f"P5\n{grid.shape[1]} {grid.shape[0]}\n255\n".encode() + bytes(
        0 if cell else 254 for row in grid for cell in row))
    yaml = out / "map.yaml"
    yaml.write_text(f"image: map.pgm\nmode: trinary\nresolution: {w / world.occupied.shape[0]:.4f}\n"
                    f"origin: [{-w / 2:.3f}, {-h / 2:.3f}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    return sdf, yaml


def launch(world: SimRobot, out: Path) -> subprocess.Popen:
    sdf, yaml = export(world, out)
    s = world.scene.start
    args = [f"world:={sdf}", f"map:={yaml}", f"x_pose:={s.x}", f"y_pose:={s.y}", f"yaw:={math.radians(s.yaw)}",
            "headless:=True", "use_rviz:=False", "use_keepout_zones:=False", "use_speed_zones:=False"]
    log = open(out / "launch.log", "w")  # noqa: SIM115 - closed with the process group
    return subprocess.Popen(["ros2", "launch", "nav2_bringup", "tb4_simulation_launch.py", *args],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def set_initial_pose(node: object, world: SimRobot) -> None:
    from geometry_msgs.msg import PoseWithCovarianceStamped

    s = world.scene.start
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "map"
    msg.pose.pose.position.x, msg.pose.pose.position.y = s.x, s.y
    yaw = math.radians(s.yaw)
    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    msg.pose.covariance[0] = msg.pose.covariance[7] = 0.01
    msg.pose.covariance[35] = 0.005
    pub = node.create_publisher(PoseWithCovarianceStamped, "initialpose", 1)
    for _ in range(5):
        pub.publish(msg)
        time.sleep(0.5)


def run_case(case: object, program: str) -> tuple[bool, str, SimRobot | None]:
    """Like robot.eval.suite.run_case, in Gazebo."""
    import rclpy

    from robot.agent.gate import check
    from robot.agent.runner import Runner
    from robot.ros.robot import RosRobot

    scene = case.scene()
    out = Path(tempfile.mkdtemp(prefix=f"sprocket-{case.id}-"))
    proc = launch(SimRobot(scene), out)
    robot = None
    try:
        rclpy.init()
        helper = rclpy.create_node("sprocket_stage")
        time.sleep(25.0)  # Gazebo, the robot and Nav2's lifecycle take a while to come up
        set_initial_pose(helper, SimRobot(scene))
        robot = RosRobot(case.scene())
        if not robot.actions["navigate_to_pose"].wait_for_server(timeout_sec=120.0):
            return False, "Nav2 did not come up", robot
        time.sleep(3.0)
        gate = check(program, robot)
        if not gate.ok:
            return False, f"gate: {gate}", robot
        result = Runner(robot).run(program)
        if result.status == "error":
            return False, f"run: {result.error} (line {result.line})", robot
        ok = bool(case.check(robot))
        return ok, "passed" if ok else "check failed", robot
    finally:
        if robot is not None:
            robot.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        subprocess.run(["pkill", "-f", "gz sim"], check=False)


def main() -> None:
    import argparse

    from robot.eval.programs import PROGRAMS
    from robot.eval.suite import CASES

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=[])
    args = parser.parse_args()
    moving = {"F1"}  # a walking person needs an actor in Gazebo; not built yet
    chosen = [c for c in CASES if c.id in PROGRAMS and (c.id in args.cases if args.cases else c.id not in moving)]
    passed = 0
    for case in chosen:
        started = time.monotonic()
        ok, why, robot = run_case(case, PROGRAMS[case.id])
        sim_ok = __import__("robot.eval.suite", fromlist=["run_case"]).run_case(case, PROGRAMS[case.id])[0]
        passed += ok
        t = robot.t if robot else 0.0
        print(f"[{'ok' if ok else 'XX'}] {case.id:<4} gazebo {why:<40} 2D world {'ok' if sim_ok else 'XX'}  "
              f"({t:.0f} s simulated, {time.monotonic() - started:.0f} s wall)", flush=True)
        if robot and not ok and robot.trace.poses:
            from robot.eval.suite import end, turned

            x, y, yaw = end(robot)
            print(f"       ended at ({x:.2f}, {y:.2f}) facing {math.degrees(yaw):.0f} deg, turned "
                  f"{math.degrees(turned(robot)):.0f} deg in all", flush=True)
    print(f"{passed}/{len(chosen)} pass in Gazebo")


if __name__ == "__main__":
    main()
