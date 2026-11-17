"""Run suite cases closed-loop in Gazebo with Nav2: the same room, agent and check as in the 2D world.

Each case gets a fresh simulation. Its scene becomes a Gazebo world (walls as boxes, things as cylinders) and a
Nav2 map (the 2D world's occupancy grid), and Nav2's TurtleBot 4 simulation is launched in it with Sprocket's 3-CSI
pure-vision URDF at the scene's start pose. Calibrated synthetic /scan from 3-CSI depth feeds AMCL/Nav2 without a
hardware lidar, and `Agent(LocalPlanner(...))` runs closed-loop against `RosRobot`.

    python3 -m robot.ros.stage                 # all static-room suite cases
    python3 -m robot.ros.stage --cases R1 O3   # specific case IDs

Runs inside WSL (or on the board) with ROS 2 sourced. Imports of ROS stay inside functions so this module stays
lightweight when imported elsewhere.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
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
    material = (
        f"<material><ambient>{r * 0.9:.3f} {g * 0.9:.3f} {b * 0.9:.3f} 1</ambient>"
        f"<diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse><specular>0.15 0.15 0.15 1</specular>"
        f"<pbr><metal><roughness>0.65</roughness><metalness>0.05</metalness></metal></pbr></material>"
    )
    return (f'<model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>'
            f'<link name="l"><collision name="c"><geometry>{geometry}</geometry></collision>'
            f'<visual name="v"><cast_shadows>true</cast_shadows><geometry>{geometry}</geometry>{material}</visual></link></model>')


def _box(name: str, x: float, y: float, yaw: float, size: tuple[float, float, float], color: str = "white") -> str:
    return _model(name, x, y, yaw, f"<box><size>{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}</size></box>", size[2] / 2,
                  color)


def _thing(t: object, things: list = ()) -> str:
    """A real 3D model from Fuel when one fits (a detector can recognise it), else a coloured shape."""
    from robot.ros.fuel import include

    name = f"{t.label}{t.id}"
    model = include(t, name, things)
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

    # Perimeter visual-only baseboards (height 0.08 m, depth 0.015 m, white/alder wood, no collision)
    outer_perimeter = [((-w / 2, -h / 2), (w / 2, -h / 2)), ((w / 2, -h / 2), (w / 2, h / 2)),
                       ((w / 2, h / 2), (-w / 2, h / 2)), ((-w / 2, h / 2), (-w / 2, -h / 2))]
    for k, (a, b) in enumerate(outer_perimeter):
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        dist = math.dist(a, b)
        models.append(
            f'<model name="baseboard{k}"><static>true</static><pose>{mx:.3f} {my:.3f} 0.040 0 0 {yaw:.4f}</pose>'
            f'<link name="l"><visual name="v"><cast_shadows>true</cast_shadows>'
            f'<geometry><box><size>{dist:.3f} 0.015 0.080</size></box></geometry>'
            f'<material><ambient>0.85 0.85 0.82 1</ambient><diffuse>0.92 0.92 0.90 1</diffuse><specular>0.1 0.1 0.1 1</specular>'
            f'<pbr><metal><roughness>0.7</roughness><metalness>0.0</metalness></metal></pbr></material></visual></link></model>'
        )

    # Soft visual rug in the living room center (geometry: box 2.2 1.6 0.002, visual only, no collision)
    models.append(
        '<model name="rug"><static>true</static><pose>0.0 0.0 0.001 0 0 0</pose>'
        '<link name="l"><visual name="v"><cast_shadows>false</cast_shadows>'
        '<geometry><box><size>2.2 1.6 0.002</size></box></geometry>'
        '<material><ambient>0.28 0.35 0.42 1</ambient><diffuse>0.32 0.40 0.48 1</diffuse><specular>0.05 0.05 0.05 1</specular>'
        '<pbr><metal><roughness>0.95</roughness><metalness>0.0</metalness></metal></pbr></material></visual></link></model>'
    )
    lights = "".join(  # ceiling lights every 2.5 m, as in a home
        f'<light type="point" name="ceiling{i}_{j}"><pose>{x:.2f} {y:.2f} 2.3 0 0 0</pose>'
        '<diffuse>0.32 0.31 0.29 1</diffuse><specular>0.1 0.1 0.1 1</specular><cast_shadows>false</cast_shadows>'
        '<attenuation><range>7</range><constant>0.6</constant><linear>0.08</linear><quadratic>0.01</quadratic>'
        '</attenuation></light>'
        for i, x in enumerate(_steps(-w / 2, w / 2, 2.5)) for j, y in enumerate(_steps(-h / 2, h / 2, 2.5)))
    things = list(world.things.values())
    for t in things:
        if (t.person or t.solid or t.label in SHAPES) and t.label != "door" and not t.route and t.appear_at <= 0:
            models.append(_thing(t, things))
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
    <scene><ambient>0.4 0.4 0.4 1</ambient><background>0.78 0.82 0.86 1</background><shadows>true</shadows></scene>
    <light type="directional" name="sun"><pose>0 0 10 0 0 0</pose><diffuse>0.8 0.8 0.78 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular><direction>-0.5 0.3 -0.8</direction><cast_shadows>true</cast_shadows></light>
    {lights}
    <model name="ground"><static>true</static><link name="l">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></collision>
      <visual name="v"><cast_shadows>false</cast_shadows><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
        <material><ambient>0.55 0.45 0.33 1</ambient><diffuse>0.65 0.52 0.38 1</diffuse><specular>0.1 0.1 0.1 1</specular>
          <pbr><metal><roughness>0.8</roughness><metalness>0.0</metalness></metal></pbr></material></visual>
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


# What a simulation runs, by process name. A copy left over from an earlier run fights the new one: a second Gazebo
# publishes its own clock and TF, and a second Nav2 container answers to the same node names (one case's Nav2 took
# 170 s to load and never activated while the last run's container was still shutting down).
LEFTOVERS = ("gz-sim-main", "gz-sim-gui", "gz sim", "component_container", "robot_state_publisher", "parameter_bridge",
             "image_bridge", "ros2 launch", "rviz2", "ekf_node", "robot.ros.perceive")


def clear() -> None:
    """Kill whatever an earlier simulation left running, and wait until it is gone."""
    patterns = [f"[{name[0]}]{name[1:]}" for name in LEFTOVERS]
    for pattern in patterns:
        subprocess.run(["pkill", "-9", "-f", pattern], check=False)
    for _ in range(20):
        if all(subprocess.run(["pgrep", "-f", p], capture_output=True).returncode for p in patterns):
            return
        time.sleep(0.5)


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
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        clear()  # Gazebo's GUI and server, and Nav2's container, do not always leave with their launch


def _bridge_config(out: Path) -> Path:
    """Copy Nav2's TB4 ros_gz_bridge config without its lidar /scan entry (Sprocket synthesizes /scan)."""
    import yaml

    path = out / "tb4_bridge.yaml"
    base = Path("/opt/ros/lyrical/share/nav2_minimal_tb4_sim/configs/tb4_bridge.yaml")
    if not base.exists():
        path.write_text("[]\n")
        return path
    entries = yaml.safe_load(base.read_text()) or []
    filtered = [
        e for e in entries
        if not (isinstance(e, dict) and (e.get("ros_topic_name") == "scan" or e.get("gz_topic_name") == "/scan"))
    ]
    path.write_text(yaml.safe_dump(filtered, sort_keys=False))
    return path


def launch(world: SimRobot, out: Path, *, gui: bool = False, slam: bool = False,
           map_file: Path | None = None) -> Launched:
    """Nav2's simulation launch with Sprocket's body (robot/ros/model.py) in the scene; with gui, the Gazebo 3D window
    opens too (RViz stays closed: on a CPU it costs the model and the detector too much, and the live page shows the
    map, synthetic /scan and the cameras). The rear cameras get their own RGB and depth bridges: the TurtleBot launch
    only knows the front one, and robot_localization's EKF fuses the wheels with the gyro into the odometry
    (robot/ros/model.py). The robot localises in `map_file` (a map it built: robot/ros/explore.py), else in the
    simulator's own map of the scene; with `slam` it has no map and builds one as it goes."""
    from robot.ros.model import EKF, GROUND_TRUTH, REAR_BRIDGES, REAR_INFO, TOF_BRIDGES, TOF_INFO, urdf

    clear()
    sdf, yaml = export(world, out)
    body = out / "sprocket.urdf"
    body.write_text(urdf())
    ekf = out / "ekf.yaml"
    ekf.write_text(EKF)
    bridge_cfg = _bridge_config(out)
    s = world.scene.start
    args = [f"world:={sdf}", f"map:={map_file or yaml}", f"x_pose:={s.x}", f"y_pose:={s.y}",
            f"yaw:={math.radians(s.yaw)}", f"robot_sdf:={body}", "robot_name:=sprocket", f"headless:={not gui}",
            "use_rviz:=False", "use_keepout_zones:=False", "use_speed_zones:=False", f"slam:={slam}",
            f"params_file:={_nav2_params(out)}"]
    tb4_launch = Path("/opt/ros/lyrical/share/nav2_bringup/launch/tb4_simulation_launch.py")
    if tb4_launch.exists() and "bridge_config" in tb4_launch.read_text():
        args.append(f"bridge_config:={bridge_cfg}")
    env = {**os.environ, "QT_QPA_PLATFORM": "xcb"}  # Qt windows draw through X11 under WSLg
    if os.path.exists("/dev/dxg"):  # WSL: Mesa falls back to drawing on the CPU (llvmpipe) unless told to use the
        env.update(GALLIUM_DRIVER="d3d12", MESA_D3D12_DEFAULT_ADAPTER_NAME="NVIDIA")  # GPU through Direct3D 12
    env["PATH"] = f"{_wait_for_world(out)}:{env['PATH']}"
    log = open(out / "launch.log", "w")  # noqa: SIM115 - closed with the process group
    procs = [subprocess.Popen(["ros2", "launch", "nav2_bringup", "tb4_simulation_launch.py", *args],
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)]
    image_bridges = list(dict.fromkeys([*REAR_BRIDGES, *TOF_BRIDGES]))
    info_bridges = list(dict.fromkeys([*REAR_INFO, *TOF_INFO, GROUND_TRUTH]))
    sim_time = ["--ros-args", "-p", "use_sim_time:=true"]
    for command in (["ros2", "run", "ros_gz_image", "image_bridge", *image_bridges, *sim_time],
                    ["ros2", "run", "ros_gz_bridge", "parameter_bridge", *info_bridges, *sim_time],
                    ["ros2", "run", "robot_localization", "ekf_node", "--ros-args", "--params-file", str(ekf)],
                    [sys.executable, "-m", "robot.ros.perceive", "--bridge-only"]):
        procs.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
    return Launched(procs)


# Localisation tuned to this robot: Nav2's defaults suit a 60-beam view and wheel-only odometry. With them, AMCL was
# 0.10 m and 1.2 degrees off the true pose while driving (worst 0.25 m and 6 degrees), and every object seen in that
# moment was placed as far off. Sprocket's calibrated 3-CSI depth rig publishes a synthetic 360-degree /scan (360
# beams) without hardware lidar, and its odometry is fused with the gyro.
AMCL = {"max_beams": 150, "update_min_d": 0.1, "update_min_a": 0.1, "alpha1": 0.1, "alpha2": 0.1, "alpha3": 0.1,
        "alpha4": 0.1, "z_hit": 0.9, "z_rand": 0.1, "sigma_hit": 0.1, "transform_tolerance": 1.0}


def _nav2_params(out: Path) -> Path:
    """Nav2's parameters with AMCL tuned (AMCL above), written next to the world."""
    import yaml

    base = Path("/opt/ros/lyrical/share/nav2_bringup/params/nav2_params.yaml")
    params = yaml.safe_load(base.read_text()) if base.exists() else {}

    params.setdefault("amcl", {}).setdefault("ros__parameters", {}).update(AMCL)

    cs = params.setdefault("controller_server", {}).setdefault("ros__parameters", {})
    cs["transform_tolerance"] = 1.0
    fp = cs.setdefault("FollowPath", {})
    fp["transform_tolerance"] = 1.0
    fp["batch_size"] = 800

    lc = params.setdefault("local_costmap", {}).setdefault("local_costmap", {}).setdefault("ros__parameters", {})
    lc["transform_tolerance"] = 1.0

    gc = params.setdefault("global_costmap", {}).setdefault("global_costmap", {}).setdefault("ros__parameters", {})
    gc["transform_tolerance"] = 1.0

    bs = params.setdefault("behavior_server", {}).setdefault("ros__parameters", {})
    bs["transform_tolerance"] = 1.0

    cm = params.setdefault("collision_monitor", {}).setdefault("ros__parameters", {})
    cm["transform_tolerance"] = 1.0
    cm["source_timeout"] = 5.0

    path = out / "nav2_params.yaml"
    path.write_text(yaml.safe_dump(params, sort_keys=False))
    return path


def _wait_for_world(out: Path) -> Path:
    """A folder holding a `gz` that waits for its world file before starting the real one. Nav2's simulation launch
    starts `gz sim` on the world at the same moment as the xacro that writes it; Gazebo started first and exited at
    once in three launches out of three on one evening, taking Nav2 down with it. Written in Ruby, like the real
    `gz`: the launch starts the 3D window as `ruby <gz> sim -g`, and a shell script there killed the window."""
    import shutil

    real = shutil.which("gz")
    folder = out / "bin"
    folder.mkdir(exist_ok=True)
    shim = folder / "gz"
    shim.write_text(f"""#!/usr/bin/env ruby
world = ARGV.last.to_s
if world.start_with?("/tmp/nav2_") && world.end_with?(".sdf")
  300.times {{ break if File.size?(world); sleep 0.1 }}
  sleep 0.5
end
exec("{real}", *ARGV)
""")
    shim.chmod(0o755)
    return folder


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


def run_case(
    case: object,
    planner: object | None = None,
    *,
    brain_url: str = "http://127.0.0.1:8080",
    think: int = 0,
) -> tuple[bool, str, SimRobot | None]:
    """Like robot.eval.suite.run_case, closed-loop in Gazebo on RosRobot."""
    import rclpy

    from robot.agent.agent import Agent
    from robot.agent.planner import LocalPlanner
    from robot.mind.brain import Brain
    from robot.ros.robot import RosRobot

    if planner is None:
        planner = LocalPlanner(Brain(brain_url), think=think)
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
        record = Agent(planner, brain=getattr(planner, "brain", None)).do(case.say, robot)
        checked = bool(case.check(robot))
        ok = checked and record.outcome in ("done", "preempted", "refused")
        if ok:
            why = "passed"
        else:
            why = "; ".join(record.problems[-2:]) or f"{record.outcome or 'failed'}, check failed"
        return ok, why, robot
    finally:
        if robot is not None:
            robot.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        sim.stop()


def main() -> None:
    import argparse

    from robot.agent.planner import LocalPlanner
    from robot.eval.suite import CASES, end, turned
    from robot.eval.suite import run_case as suite_run_case
    from robot.mind.brain import Brain

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--brain", default="http://127.0.0.1:8080")
    parser.add_argument("--think", type=int, default=0)
    args = parser.parse_args()
    planner = LocalPlanner(Brain(args.brain), think=args.think)
    if not planner.brain.healthy():
        raise SystemExit(f"llama-server is not answering on {args.brain}")
    moving = {"F1"}  # a walking person needs an actor in Gazebo; not built yet
    chosen = [c for c in CASES if (c.id in args.cases if args.cases else c.id not in moving)]
    passed = 0
    for case in chosen:
        started = time.monotonic()
        try:
            ok, why, robot = run_case(case, planner)
        except Exception as exc:  # one room that fails to start must not stop the rest
            ok, why, robot = False, f"{type(exc).__name__}: {exc}"[:40], None
        sim_ok = suite_run_case(case, planner)[0]
        passed += ok
        t = robot.t if robot else 0.0
        print(f"[{'ok' if ok else 'XX'}] {case.id:<4} gazebo {why:<40} 2D world {'ok' if sim_ok else 'XX'}  "
              f"({t:.0f} s simulated, {time.monotonic() - started:.0f} s wall)", flush=True)
        if robot and not ok and robot.trace.poses:
            x, y, yaw = end(robot)
            print(f"       ended at ({x:.2f}, {y:.2f}) facing {math.degrees(yaw):.0f} deg, turned "
                  f"{math.degrees(turned(robot)):.0f} deg in all", flush=True)
    print(f"{passed}/{len(chosen)} pass in Gazebo")


if __name__ == "__main__":
    main()
