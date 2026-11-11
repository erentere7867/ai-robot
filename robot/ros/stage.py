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

WALL_H, PERSON_H = 2.4, 1.7  # room-height walls: cameras should see a room, not a pen
RGB = {"black": (0.1, 0.1, 0.1), "white": (0.92, 0.92, 0.9), "grey": (0.55, 0.55, 0.55), "gray": (0.55, 0.55, 0.55),
       "brown": (0.5, 0.33, 0.18), "red": (0.8, 0.12, 0.1), "blue": (0.15, 0.3, 0.8), "yellow": (0.95, 0.8, 0.1),
       "green": (0.2, 0.6, 0.25), "orange": (0.95, 0.5, 0.1), "silver": (0.75, 0.75, 0.78), "": (0.6, 0.6, 0.6),
       "wall": (0.9, 0.88, 0.84)}
# label: (shape, height in metres) so the 3D room looks like the room: tables at table height, a tall lamp
SHAPES = {"table": ("box", 0.75), "desk": ("box", 0.75), "sofa": ("box", 0.8), "cabinet": ("box", 1.0),
          "fridge": ("box", 1.7), "tv": ("box", 0.9), "box": ("box", 0.45), "bin": ("cylinder", 0.5),
          "chair": ("box", 0.9), "lamp": ("cylinder", 1.5), "plant": ("cylinder", 0.9), "ball": ("sphere", 0.0),
          "backpack": ("box", 0.45), "basket": ("cylinder", 0.35), "laptop": ("box", 0.06), "pillow": ("box", 0.2),
          "shoe": ("box", 0.12), "bottle": ("cylinder", 0.3), "cup": ("cylinder", 0.12), "person": ("cylinder", 1.7)}


def _steps(lo: float, hi: float, every: float) -> list[float]:
    n = max(1, round((hi - lo) / every))
    return [lo + (k + 0.5) * (hi - lo) / n for k in range(n)]


def _model(name: str, x: float, y: float, yaw: float, geometry: str, z: float, color: str) -> str:
    r, g, b = RGB.get(color, RGB[""])
    material = f"<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>"
    return (f'<model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>'
            f'<link name="l"><collision name="c"><geometry>{geometry}</geometry></collision>'
            f'<visual name="v"><geometry>{geometry}</geometry>{material}</visual></link></model>')


def _box(name: str, x: float, y: float, yaw: float, size: tuple[float, float, float], color: str = "white") -> str:
    return _model(name, x, y, yaw, f"<box><size>{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}</size></box>", size[2] / 2,
                  color)


def _thing(t: object, real: bool = True) -> str:
    """A real 3D model from Fuel when there is one (a detector can recognise it), else a coloured shape."""
    from robot.ros.fuel import include

    name = f"{t.label}{t.id}"
    model = include(t.label, name, t.x, t.y) if real else None
    if model:
        return model
    shape, h = SHAPES.get(t.label, ("cylinder", 0.5))
    if shape == "box":  # a square footprint inside the disc the planner and the map use
        side = t.radius * 1.6
        return _box(name, t.x, t.y, 0.0, (side, side, h), t.color)
    if shape == "sphere":
        return _model(name, t.x, t.y, 0.0, f"<sphere><radius>{t.radius:.3f}</radius></sphere>", t.radius, t.color)
    return _model(name, t.x, t.y, 0.0, f"<cylinder><radius>{t.radius:.3f}</radius><length>{h:.3f}</length></cylinder>",
                  h / 2, t.color or ("blue" if t.label == "person" else ""))


def export(world: SimRobot, out: Path) -> tuple[Path, Path]:
    """The scene as a Gazebo world and a Nav2 map: one source of truth for both simulators."""
    scene = world.scene
    w, h = scene.width, scene.height
    walls = [((-w / 2, -h / 2), (w / 2, -h / 2)), ((w / 2, -h / 2), (w / 2, h / 2)),
             ((w / 2, h / 2), (-w / 2, h / 2)), ((-w / 2, h / 2), (-w / 2, -h / 2)), *scene.walls]
    models = [_box(f"wall{k}", (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, math.atan2(b[1] - a[1], b[0] - a[0]),
                   (math.dist(a, b) + 0.1, 0.1, WALL_H), "wall") for k, (a, b) in enumerate(walls)]
    lights = "".join(  # ceiling lights every 2.5 m, as in a home
        f'<light type="point" name="ceiling{i}_{j}"><pose>{x:.2f} {y:.2f} 2.3 0 0 0</pose>'
        '<diffuse>0.32 0.31 0.29 1</diffuse><specular>0.1 0.1 0.1 1</specular><cast_shadows>false</cast_shadows>'
        '<attenuation><range>7</range><constant>0.6</constant><linear>0.08</linear><quadratic>0.01</quadratic>'
        '</attenuation></light>'
        for i, x in enumerate(_steps(-w / 2, w / 2, 2.5)) for j, y in enumerate(_steps(-h / 2, h / 2, 2.5)))
    for t in world.things.values():
        if (t.person or t.solid or t.label in SHAPES) and t.label != "door" and not t.route and t.appear_at <= 0:
            models.append(_thing(t))
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
    <scene><ambient>0.32 0.32 0.32 1</ambient><background>0.78 0.82 0.86 1</background><shadows>false</shadows></scene>
    <light type="directional" name="sun"><pose>0 0 10 0 0 0</pose><diffuse>0.45 0.45 0.43 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular><direction>-0.4 0.2 -0.9</direction><cast_shadows>false</cast_shadows></light>
    {lights}
    <model name="ground"><static>true</static><link name="l">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
        <material><ambient>0.55 0.45 0.33 1</ambient><diffuse>0.62 0.5 0.36 1</diffuse></material></visual>
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


class Launched:
    """The simulation's processes: Gazebo with Nav2 and Sprocket, and the rear cameras' bridges."""

    def __init__(self, procs: list[subprocess.Popen]) -> None:
        self.procs = procs

    def stop(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
        for proc in self.procs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        subprocess.run(["pkill", "-f", "[g]z sim"], check=False)


def launch(world: SimRobot, out: Path, *, gui: bool = False) -> Launched:
    """Nav2's simulation launch with Sprocket's body (robot/ros/model.py) in the scene; with gui, the Gazebo 3D window
    opens too (RViz stays closed: on a CPU it costs the model and the detector too much, and the live page shows the
    map, the lidar and the cameras). The rear cameras get their own bridges: the TurtleBot launch only knows the front one."""
    from robot.ros.model import REAR_BRIDGES, REAR_INFO, urdf

    sdf, yaml = export(world, out)
    body = out / "sprocket.urdf"
    body.write_text(urdf())
    s = world.scene.start
    args = [f"world:={sdf}", f"map:={yaml}", f"x_pose:={s.x}", f"y_pose:={s.y}", f"yaw:={math.radians(s.yaw)}",
            f"robot_sdf:={body}", "robot_name:=sprocket", f"headless:={not gui}", "use_rviz:=False",
            "use_keepout_zones:=False", "use_speed_zones:=False"]
    env = {**os.environ, "QT_QPA_PLATFORM": "xcb"}  # Qt windows draw through X11 under WSLg
    sim_time = ["--ros-args", "-p", "use_sim_time:=true"]
    for _ in range(3):
        log = open(out / "launch.log", "w")  # noqa: SIM115 - closed with the process group
        procs = [subprocess.Popen(["ros2", "launch", "nav2_bringup", "tb4_simulation_launch.py", *args],
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)]
        for command in (["ros2", "run", "ros_gz_image", "image_bridge", *REAR_BRIDGES, *sim_time],
                        ["ros2", "run", "ros_gz_bridge", "parameter_bridge", *REAR_INFO, *sim_time]):
            procs.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(10.0)
        # Nav2's launch starts `gz sim` on the world file while xacro is still writing it; on a busy machine Gazebo
        # can lose that race and exit at once, taking the launch down. Then start again.
        if "Unable to find or download file /tmp/nav2_" not in (out / "launch.log").read_text(errors="ignore"):
            break
        print("Gazebo started before its world file was written; starting again", flush=True)
        Launched(procs).stop()
    return Launched(procs)


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
    sim = launch(SimRobot(scene), out)
    robot = None
    try:
        rclpy.init()
        helper = rclpy.create_node("sprocket_stage")
        time.sleep(25.0)  # Gazebo, the robot and Nav2's lifecycle take a while to come up
        set_initial_pose(helper, SimRobot(scene))
        robot = RosRobot(case.scene(), nudge=lambda: set_initial_pose(helper, SimRobot(scene)))
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
        sim.stop()


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
        try:
            ok, why, robot = run_case(case, PROGRAMS[case.id])
        except Exception as exc:  # one room that fails to start must not stop the rest
            ok, why, robot = False, f"{type(exc).__name__}: {exc}"[:40], None
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
