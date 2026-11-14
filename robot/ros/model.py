"""Sprocket's simulated body: the robot description Gazebo and robot_state_publisher load (architecture section 4).

The body is the one drawn in cad/sprocket_body.py, and the numbers here are that drawing's, in metres: four driven
wheels (hub motors) on sprung trailing arms, a low body, and a head tilted 10 degrees up that carries the three
cameras, with the lidar level on top of it. What the cameras see of the robot is the drawing itself: cad's
`export_meshes` writes robot/ros/meshes, one file per colour. Collisions are plain boxes and cylinders.

Gazebo topic names match Nav2's TurtleBot 4 simulation (/cmd_vel, /odom, tf, scan, imu), so its bridge, Nav2
parameters and launch files drive this body unchanged; the cameras get bridges of their own (CAMERA_BRIDGES).

    python3 -m robot.ros.model > sprocket.urdf
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

WHEEL_R, WHEEL_W, TRACK, WHEELBASE = 0.05, 0.04, 0.30, 0.26
LENGTH, WIDTH = 0.36, 0.343  # over the tyres
FOOTPRINT = [(LENGTH / 2, WIDTH / 2), (LENGTH / 2, -WIDTH / 2), (-LENGTH / 2, -WIDTH / 2), (-LENGTH / 2, WIDTH / 2)]
# Four wheels that cannot steer turn by skidding, so the body turns slower than two wheels this far apart would:
# the drive is told the wheels are TURN_SCRUB times further apart, and then a commanded turn rate is the one it gets.
TURN_SCRUB = 1.0
PIVOT_X, PIVOT_Y, PIVOT_Z = 0.050, 0.099, 0.080  # where each trailing arm hinges on the tub
SPRUNG, WHEEL_MASS, ARM_MASS = 4.1, 0.77, 0.15  # kg: everything the springs carry; one hub motor; one arm
WHEEL_RATE, DAMPING_RATIO = 1200.0, 0.5  # N/m at the wheel (about 8 mm of sag); share of critical damping
WHEEL_TORQUE = 2.0  # N m, the hub motor's stall torque
LIDAR_X, LIDAR_Z = 0.071, 0.375
HEAD, HEAD_TILT = (0.080, 0.0, 0.232), math.radians(10)  # the neck's top, and how far the head looks up
REAR_YAW = math.radians(120)
# name: (place in the head (x, y, z), yaw in the head, horizontal field of view, width, height, rate, depth).
# The front lens is the narrower one: what the robot drives towards gets the most pixels, and the two 155-degree
# rear lenses still overlap it and each other. Set SPROCKET_FRONT_DEPTH=1 to try the body with an RGB-D front camera.
FRONT_DEPTH = os.environ.get("SPROCKET_FRONT_DEPTH", "") == "1"
CAMERAS = {
    "front_camera": ((0.061, 0.0, 0.080), 0.0, 1.92, 1280, 720, 10, FRONT_DEPTH),
    "rear_left_camera": ((-0.031, 0.0805, 0.062), REAR_YAW, 2.7, 1280, 720, 5, False),
    "rear_right_camera": ((-0.031, -0.0805, 0.062), -REAR_YAW, 2.7, 1280, 720, 5, False),
}
MESHES = Path(__file__).parent / "meshes"


def image_topic(name: str) -> str:
    """Where a camera's colour picture arrives (an RGB-D camera uses the TurtleBot's topics, which Nav2 bridges)."""
    return "/rgbd_camera/image" if CAMERAS[name][6] else f"/{name}/image"


def _inertial(mass: float, ixx: float, iyy: float, izz: float, xyz: str = "0 0 0") -> str:
    return (f'<inertial><origin xyz="{xyz}"/><mass value="{mass}"/><inertia ixx="{ixx:.6f}" ixy="0" ixz="0" '
            f'iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/></inertial>')


def _box_inertial(mass: float, x: float, y: float, z: float, xyz: str = "0 0 0") -> str:
    return _inertial(mass, mass * (y * y + z * z) / 12, mass * (x * x + z * z) / 12, mass * (x * x + y * y) / 12, xyz)


def _mesh(folder: Path, name: str, rgb: list[float], rpy: str = "0 0 0") -> str:
    r, g, b = rgb
    return (f'<visual><origin xyz="0 0 0" rpy="{rpy}"/><geometry><mesh filename="file://{folder}/{name}.stl"/>'
            f'</geometry><material name="{name}"><color rgba="{r} {g} {b} 1"/></material></visual>')


def _fixed(parent: str, child: str, xyz: str, rpy: str = "0 0 0") -> str:
    return (f'<joint name="{child}_joint" type="fixed"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{xyz}" rpy="{rpy}"/></joint><gazebo reference="{child}_joint">'
            f'<preserveFixedJoint>true</preserveFixedJoint></gazebo>')


def _frame(name: str, parent: str, xyz: str, rpy: str = "0 0 0") -> str:
    return f'<link name="{name}">{_inertial(0.001, 1e-7, 1e-7, 1e-7)}</link>' + _fixed(parent, name, xyz, rpy)


def _camera(name: str, place: tuple, yaw: float, fov: float, width: int, height: int, rate: int, depth: bool) -> str:
    link, optical = name + "_link", name + "_optical_frame"
    parts = [_frame(link, "head_link", " ".join(f"{v:.4f}" for v in place), f"0 0 {yaw:.4f}"),
             _frame(optical, link, "0 0 0", f"{-math.pi / 2:.4f} 0 {-math.pi / 2:.4f}")]
    camera = (f"<camera><horizontal_fov>{fov}</horizontal_fov><image><width>{width}</width><height>{height}</height>"
              f"</image><clip><near>0.05</near><far>20</far></clip><optical_frame_id>{optical}</optical_frame_id>"
              "</camera>")
    if depth:  # publishes rgbd_camera/image, depth_image and camera_info, as the TurtleBot's does
        sensor = (f'<sensor name="rgbd_camera" type="rgbd_camera"><always_on>1</always_on>{camera}'
                  f"<update_rate>{rate}</update_rate><visualize>true</visualize><topic>rgbd_camera</topic></sensor>")
    else:
        sensor = (f'<sensor name="{name}" type="camera"><always_on>1</always_on>{camera}'
                  f"<update_rate>{rate}</update_rate><visualize>true</visualize><topic>{name}/image</topic>"
                  f"<camera_info_topic>{name}/camera_info</camera_info_topic></sensor>")
    parts.append(f'<gazebo reference="{link}">{sensor}</gazebo>')
    return "".join(parts)


def _corner(name: str, sx: int, sy: int, skins: dict, folder: Path) -> str:
    """One wheel on its sprung trailing arm. The spring is wound up so that it carries its share of the body with
    the arm where the drawing has it."""
    reach = WHEELBASE / 2 - PIVOT_X  # pivot to axle, along the floor
    stiffness = WHEEL_RATE * reach ** 2  # N m per radian at the pivot
    damping = 2 * DAMPING_RATIO * math.sqrt(WHEEL_RATE * SPRUNG / 4) * reach ** 2
    wound = sx * (SPRUNG / 4 * 9.81 * reach) / stiffness  # a wheel ahead of its pivot rises as its arm turns back
    arm, wheel = f"{name}_arm", f"{name}_wheel"
    turned = "0 0 0" if sy > 0 else f"0 0 {math.pi:.4f}"  # the drawing's wheel is a left one: hub cap outwards
    visuals = "".join(_mesh(folder, key, rgb, turned) for key, rgb in skins.items() if key.startswith("wheel_"))
    tyre = f'<origin rpy="{-math.pi / 2:.4f} 0 0"/><geometry><cylinder radius="{WHEEL_R}" length="{WHEEL_W}"/></geometry>'
    spin = WHEEL_MASS * WHEEL_R ** 2 / 2
    side = WHEEL_MASS * (3 * WHEEL_R ** 2 + WHEEL_W ** 2) / 12
    return (
        f'<link name="{arm}">{_box_inertial(ARM_MASS, 0.09, 0.012, 0.026, f"{sx * reach / 2:.4f} 0 -0.015")}</link>'
        f'<joint name="{arm}_joint" type="revolute"><parent link="base_link"/><child link="{arm}"/>'
        f'<origin xyz="{sx * PIVOT_X} {sy * PIVOT_Y} {PIVOT_Z}"/><axis xyz="0 1 0"/>'
        f'<limit lower="-0.24" upper="0.24" effort="100" velocity="20"/><dynamics damping="{damping:.4f}"/></joint>'
        f'<gazebo reference="{arm}_joint"><springStiffness>{stiffness:.4f}</springStiffness>'
        f'<springReference>{wound:.4f}</springReference></gazebo>'
        f'<link name="{wheel}">{visuals}<collision>{tyre}</collision>{_inertial(WHEEL_MASS, side, spin, side)}</link>'
        f'<joint name="{wheel}_joint" type="continuous"><parent link="{arm}"/><child link="{wheel}"/>'
        f'<origin xyz="{sx * reach} {sy * (TRACK / 2 - PIVOT_Y):.4f} {WHEEL_R - PIVOT_Z:.4f}"/><axis xyz="0 1 0"/>'
        f'<limit effort="{WHEEL_TORQUE}" velocity="25"/></joint>'
        f'<gazebo reference="{wheel}"><mu1>1.0</mu1><mu2>1.0</mu2></gazebo>')


def urdf(meshes: Path | str = MESHES) -> str:
    """The robot description, with its pictures read from the folder `meshes` (a copy where Gazebo can reach it)."""
    folder = Path(meshes)
    skins = json.loads((MESHES / "index.json").read_text())
    corners = {"front_left": (1, 1), "front_right": (1, -1), "rear_left": (-1, 1), "rear_right": (-1, -1)}
    body = "".join(_mesh(folder, key, rgb) for key, rgb in skins.items() if not key.startswith("wheel_"))
    hx, hy, hz = HEAD
    parts = [
        # the tub and the shell as one box, 45 mm clear of the floor; the battery keeps the weight low
        f'<link name="base_link">{body}<collision><origin xyz="0 0 0.1185"/><geometry><box size="0.32 0.236 0.147"/>'
        f'</geometry></collision>{_box_inertial(SPRUNG, 0.30, 0.22, 0.16, "0.01 0 0.137")}</link>',
        _frame("base_footprint", "base_link", "0 0 0"),
        f'<link name="head_link"><collision><origin xyz="0.009 0 0.052"/><geometry><box size="0.098 0.16 0.104"/>'
        f'</geometry></collision>{_inertial(0.001, 1e-7, 1e-7, 1e-7)}</link>',
        _fixed("base_link", "head_link", f"{hx} {hy} {hz}", f"0 {-HEAD_TILT:.4f} 0"),
        *(_corner(name, sx, sy, skins, folder) for name, (sx, sy) in corners.items()),
        _frame("lidar_link", "base_link", f"{LIDAR_X} 0 {LIDAR_Z}"),
        '<gazebo reference="lidar_link"><sensor name="lidar" type="gpu_lidar"><update_rate>10</update_rate>'
        "<visualize>true</visualize><always_on>true</always_on><frame_id>lidar_link</frame_id><topic>scan</topic>"
        "<lidar><scan><horizontal><samples>500</samples><resolution>1</resolution><min_angle>-3.14159</min_angle>"
        "<max_angle>3.14159</max_angle></horizontal></scan><range><min>0.12</min><max>12.0</max>"
        "<resolution>0.01</resolution></range></lidar></sensor></gazebo>",
        _frame("imu_link", "base_link", "0 0 0.125"),
        '<gazebo reference="imu_link"><sensor type="imu" name="imu"><always_on>true</always_on>'
        "<update_rate>100</update_rate><frame_id>imu_link</frame_id><topic>imu</topic></sensor></gazebo>",
        *(_camera(name, *spec) for name, spec in CAMERAS.items()),
        f"""<gazebo>
  <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
    <left_joint>front_left_wheel_joint</left_joint><left_joint>rear_left_wheel_joint</left_joint>
    <right_joint>front_right_wheel_joint</right_joint><right_joint>rear_right_wheel_joint</right_joint>
    <wheel_separation>{TRACK * TURN_SCRUB:.4f}</wheel_separation><wheel_radius>{WHEEL_R}</wheel_radius>
    <max_linear_acceleration>2</max_linear_acceleration><min_linear_acceleration>-2</min_linear_acceleration>
    <max_angular_acceleration>3</max_angular_acceleration><min_angular_acceleration>-3</min_angular_acceleration>
    <max_linear_velocity>0.6</max_linear_velocity><min_linear_velocity>-0.6</min_linear_velocity>
    <max_angular_velocity>2.0</max_angular_velocity><min_angular_velocity>-2.0</min_angular_velocity>
    <topic>/cmd_vel</topic><odom_topic>/odom</odom_topic><tf_topic>/model/sprocket/wheel_tf</tf_topic>
    <frame_id>odom</frame_id><child_frame_id>base_link</child_frame_id>
    <odom_publish_frequency>50</odom_publish_frequency>
  </plugin>
  <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">
    {"".join(f"<joint_name>{name}_{part}_joint</joint_name>" for name in corners for part in ("arm", "wheel"))}
    <topic>joint_states</topic><update_rate>30</update_rate>
  </plugin>
  <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
    <odom_frame>world</odom_frame><robot_base_frame>base_link</robot_base_frame><dimensions>2</dimensions>
    <odom_topic>/ground_truth</odom_topic><odom_publish_frequency>20</odom_publish_frequency>
  </plugin>
</gazebo>""",
    ]
    return '<?xml version="1.0"?>\n<robot name="sprocket">\n' + "\n".join(parts) + "\n</robot>\n"


_plain = [name for name, spec in CAMERAS.items() if not spec[6]]
CAMERA_BRIDGES = [f"/{name}/image" for name in _plain]
CAMERA_INFO = [f"/{name}/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" for name in _plain]
# Where the robot really is, from the simulator: for measuring localisation, never used by the robot itself.
GROUND_TRUTH = "/ground_truth@nav_msgs/msg/Odometry[gz.msgs.Odometry"

if __name__ == "__main__":
    print(urdf())


# The odometry the robot drives on: wheels for forward speed (sideways speed held at zero, as on any base that
# cannot drive sideways), the gyro for turning. Wheels slip when the robot turns on the spot, and four wheels that
# cannot steer slip a great deal; the gyro does not. robot_localization's EKF publishes odom -> base_link in
# place of the wheels (whose transform goes to an unused topic above). An indoor compass heading is not trusted, so
# no absolute yaw is fused.
EKF = """ekf_filter_node:
  ros__parameters:
    use_sim_time: true
    frequency: 30.0
    two_d_mode: true
    publish_tf: true
    map_frame: map
    odom_frame: odom
    base_link_frame: base_link
    world_frame: odom
    odom0: /odom
    odom0_config: [false, false, false, false, false, false, true, true, false, false, false, false, false, false, false]
    imu0: /imu
    imu0_config: [false, false, false, false, false, false, false, false, false, false, false, true, false, false, false]
    imu0_remove_gravitational_acceleration: true
"""
