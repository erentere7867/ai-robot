"""Sprocket's simulated body: the robot description Gazebo and robot_state_publisher load (architecture section 4).

A 30 cm round base on two wheels with front and rear casters; a 2D lidar on top seeing all the way round; a front
RGB-D camera and two 155-degree rear cameras facing ±112 degrees, so the three cameras overlap all the way round; an IMU.
Gazebo topic names match Nav2's TurtleBot 4 simulation (/cmd_vel, /odom, tf, scan, imu, rgbd_camera), so its bridge,
Nav2 parameters and launch files drive this body unchanged; only the rear cameras need bridges of their own.

    python3 -m robot.ros.model > sprocket.urdf
"""

from __future__ import annotations

import math

BODY_R, BODY_H, BODY_Z = 0.15, 0.10, 0.04  # base cylinder: radius, height, bottom above the floor
WHEEL_R, WHEEL_W, TRACK = 0.05, 0.03, 0.24
DECK_Z, LIDAR_Z, CAMERA_Z, CAMERA_R = 0.20, 0.27, 0.19, 0.11
REAR_YAW = math.radians(112)  # with 155-degree lenses the rear cameras overlap the front one and each other
CAMERAS = {  # name: (yaw, horizontal field of view, width, height, rate, depth)
    "front_camera": (0.0, 1.5, 640, 480, 10, True),  # RGB-D, like the D435i / Gemini 335 in the parts list
    "rear_left_camera": (REAR_YAW, 2.7, 1280, 720, 5, False),  # wide lens: resolution keeps far things big enough
    "rear_right_camera": (-REAR_YAW, 2.7, 1280, 720, 5, False),
}
ORANGE, DARK, GREY = "0.95 0.55 0.12 1", "0.08 0.08 0.08 1", "0.5 0.5 0.5 1"


def _inertial(mass: float, ixx: float, iyy: float, izz: float, z: float = 0.0) -> str:
    return (f'<inertial><origin xyz="0 0 {z}"/><mass value="{mass}"/><inertia ixx="{ixx:.6f}" ixy="0" ixz="0" '
            f'iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/></inertial>')


def _cylinder_inertial(mass: float, r: float, h: float, z: float = 0.0) -> str:
    side = mass * (3 * r * r + h * h) / 12
    return _inertial(mass, side, side, mass * r * r / 2, z)


def _link(name: str, geometry: str, color: str, mass: float, inertial: str, z: float = 0.0, rpy: str = "0 0 0",
          collide: bool = True) -> str:
    origin = f'<origin xyz="0 0 {z}" rpy="{rpy}"/>'
    collision = f"<collision>{origin}<geometry>{geometry}</geometry></collision>" if collide else ""
    return (f'<link name="{name}"><visual>{origin}<geometry>{geometry}</geometry>'
            f'<material name="{name}_m"><color rgba="{color}"/></material></visual>{collision}{inertial}</link>'
            f'<gazebo reference="{name}"><visual><material><diffuse>{color}</diffuse><ambient>{color}</ambient>'
            f'</material></visual></gazebo>')


def _fixed(parent: str, child: str, xyz: str, rpy: str = "0 0 0") -> str:
    return (f'<joint name="{child}_joint" type="fixed"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{xyz}" rpy="{rpy}"/></joint><gazebo reference="{child}_joint">'
            f'<preserveFixedJoint>true</preserveFixedJoint></gazebo>')


def _camera(name: str, yaw: float, fov: float, width: int, height: int, rate: int, depth: bool) -> str:
    x, y = CAMERA_R * math.cos(yaw), CAMERA_R * math.sin(yaw)
    link = name + "_link"
    optical = name + "_optical_frame"
    parts = [_link(link, '<box size="0.025 0.09 0.03"/>', DARK, 0.05, _inertial(0.05, 1e-5, 1e-5, 1e-5)),
             _fixed("base_link", link, f"{x:.4f} {y:.4f} {CAMERA_Z}", f"0 0 {yaw:.4f}"),
             f'<link name="{optical}">{_inertial(0.001, 1e-7, 1e-7, 1e-7)}</link>',
             _fixed(link, optical, "0 0 0", f"{-math.pi / 2:.4f} 0 {-math.pi / 2:.4f}")]
    camera = (f"<camera><horizontal_fov>{fov}</horizontal_fov><image><width>{width}</width><height>{height}</height>"
              f"</image><clip><near>0.1</near><far>20</far></clip><optical_frame_id>{optical}</optical_frame_id>"
              "</camera>")
    if depth:  # the front RGB-D camera publishes rgbd_camera/image, depth_image and camera_info, as the TB4's does
        sensor = (f'<sensor name="rgbd_camera" type="rgbd_camera"><always_on>1</always_on>{camera}'
                  f"<update_rate>{rate}</update_rate><visualize>true</visualize><topic>rgbd_camera</topic></sensor>")
    else:
        sensor = (f'<sensor name="{name}" type="camera"><always_on>1</always_on>{camera}'
                  f"<update_rate>{rate}</update_rate><visualize>true</visualize><topic>{name}/image</topic>"
                  f"<camera_info_topic>{name}/camera_info</camera_info_topic></sensor>")
    parts.append(f'<gazebo reference="{link}">{sensor}</gazebo>')
    return "".join(parts)


def _wheel(side: str, y: float) -> str:
    name = f"{side}_wheel"
    geometry = f'<cylinder radius="{WHEEL_R}" length="{WHEEL_W}"/>'
    return (_link(name, geometry, DARK, 0.2, _cylinder_inertial(0.2, WHEEL_R, WHEEL_W), rpy=f"{-math.pi / 2:.4f} 0 0")
            + f'<joint name="{name}_joint" type="continuous"><parent link="base_link"/><child link="{name}"/>'
              f'<origin xyz="0 {y} {WHEEL_R}"/><axis xyz="0 1 0"/></joint>'
            + f'<gazebo reference="{name}"><mu1>1.0</mu1><mu2>1.0</mu2></gazebo>')


def _caster(name: str, x: float) -> str:
    r = 0.025
    return (_link(name, f'<sphere radius="{r}"/>', GREY, 0.05, _inertial(0.05, 1e-5, 1e-5, 1e-5))
            + _fixed("base_link", name, f"{x} 0 {r}")
            + f'<gazebo reference="{name}"><mu1>0.0</mu1><mu2>0.0</mu2></gazebo>')


def urdf() -> str:
    body = f'<cylinder radius="{BODY_R}" length="{BODY_H}"/>'
    deck = '<cylinder radius="0.07" length="0.07"/>'
    lidar = '<cylinder radius="0.04" length="0.05"/>'
    parts = [
        _link("base_link", body, ORANGE, 3.0, _cylinder_inertial(3.0, BODY_R, BODY_H, BODY_Z + BODY_H / 2),
              z=BODY_Z + BODY_H / 2),
        f'<link name="base_footprint">{_inertial(0.001, 1e-7, 1e-7, 1e-7)}</link>',
        _fixed("base_link", "base_footprint", "0 0 0"),
        _link("deck", deck, GREY, 0.3, _cylinder_inertial(0.3, 0.07, 0.07)),
        _fixed("base_link", "deck", f"0 0 {DECK_Z}"),
        _wheel("left", TRACK / 2), _wheel("right", -TRACK / 2),
        _caster("front_caster", 0.11), _caster("rear_caster", -0.11),
        _link("lidar_link", lidar, DARK, 0.17, _cylinder_inertial(0.17, 0.04, 0.05)),
        _fixed("base_link", "lidar_link", f"0 0 {LIDAR_Z}"),
        '<gazebo reference="lidar_link"><sensor name="lidar" type="gpu_lidar"><update_rate>10</update_rate>'
        "<visualize>true</visualize><always_on>true</always_on><frame_id>lidar_link</frame_id><topic>scan</topic>"
        "<lidar><scan><horizontal><samples>360</samples><resolution>1</resolution><min_angle>-3.14159</min_angle>"
        "<max_angle>3.14159</max_angle></horizontal></scan><range><min>0.12</min><max>12.0</max>"
        "<resolution>0.01</resolution></range></lidar></sensor></gazebo>",
        f'<link name="imu_link">{_inertial(0.01, 1e-6, 1e-6, 1e-6)}</link>',
        _fixed("base_link", "imu_link", "0 0 0.1"),
        '<gazebo reference="imu_link"><sensor type="imu" name="imu"><always_on>true</always_on>'
        "<update_rate>100</update_rate><topic>imu</topic></sensor></gazebo>",
        *(_camera(name, *spec) for name, spec in CAMERAS.items()),
        f"""<gazebo>
  <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
    <left_joint>left_wheel_joint</left_joint><right_joint>right_wheel_joint</right_joint>
    <wheel_separation>{TRACK}</wheel_separation><wheel_radius>{WHEEL_R}</wheel_radius>
    <max_linear_acceleration>2</max_linear_acceleration><min_linear_acceleration>-2</min_linear_acceleration>
    <max_angular_acceleration>3</max_angular_acceleration><min_angular_acceleration>-3</min_angular_acceleration>
    <max_linear_velocity>0.6</max_linear_velocity><min_linear_velocity>-0.6</min_linear_velocity>
    <max_angular_velocity>2.0</max_angular_velocity><min_angular_velocity>-2.0</min_angular_velocity>
    <topic>/cmd_vel</topic><odom_topic>/odom</odom_topic><tf_topic>tf</tf_topic>
    <frame_id>odom</frame_id><child_frame_id>base_link</child_frame_id>
    <odom_publish_frequency>50</odom_publish_frequency>
  </plugin>
  <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">
    <joint_name>left_wheel_joint</joint_name><joint_name>right_wheel_joint</joint_name>
    <topic>joint_states</topic><update_rate>30</update_rate>
  </plugin>
</gazebo>""",
    ]
    return '<?xml version="1.0"?>\n<robot name="sprocket">\n' + "\n".join(parts) + "\n</robot>\n"


REAR_BRIDGES = [f"/{name}/{part}" for name in ("rear_left_camera", "rear_right_camera") for part in ("image",)]
REAR_INFO = [f"/{name}/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
             for name in ("rear_left_camera", "rear_right_camera")]

if __name__ == "__main__":
    print(urdf())
