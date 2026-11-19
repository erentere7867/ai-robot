"""Sprocket's body: the numbers every other part takes its geometry from, and the robot description
robot_state_publisher loads (architecture section 2).

A 30 cm round base on two wheels with front and rear casters; three wide-angle colour cameras (MIPI-CSI, no depth) at
120-degree separation for 360-degree vision; front and rear 8x8 multi-zone ToF sensors tipped down at the floor; an
IMU. The simulator (robot/sim/habitat_server.py) builds its sensors from these same numbers.

    python3 -m robot.ros.model > sprocket.urdf
"""

from __future__ import annotations

import math

BODY_R, BODY_H, BODY_Z = 0.15, 0.10, 0.04  # base cylinder: radius, height, bottom above the floor
WHEEL_R, WHEEL_W, TRACK = 0.05, 0.03, 0.24
DECK_Z, CAMERA_Z, CAMERA_R = 0.20, 0.22, 0.11
CLEARANCE = 0.30  # the body's top is at 0.235 m (deck and cameras): it fits under anything higher than this
REAR_YAW = 2 * math.pi / 3  # 120 deg
CAMERAS = {  # name: (yaw on the body, horizontal field of view, width, height, frames per second)
    "front_camera": (0.0, 2.44, 1280, 720, 5),
    "rear_left_camera": (REAR_YAW, 2.44, 1280, 720, 5),
    "rear_right_camera": (-REAR_YAW, 2.44, 1280, 720, 5),
}
TOF = {  # side: (x on the body, yaw, height, pitch down, field of view, range) for a VL53L5CX in 8x8 mode
    "front": (BODY_R, 0.0, 0.08, 0.28, 0.7854, 4.0),
    "rear": (-BODY_R, math.pi, 0.08, 0.28, 0.7854, 4.0),
}
MAX_V, MAX_W = 0.5, 2.0  # wheel limits: m/s, rad/s
MAX_DV, MAX_DW = 1.0, 3.0  # the motor controller's acceleration limits: m/s^2, rad/s^2
ORANGE, DARK, GREY = "0.95 0.55 0.12 1", "0.08 0.08 0.08 1", "0.5 0.5 0.5 1"


def _inertial(mass: float, ixx: float, iyy: float, izz: float, z: float = 0.0) -> str:
    return (f'<inertial><origin xyz="0 0 {z}"/><mass value="{mass}"/><inertia ixx="{ixx:.6f}" ixy="0" ixz="0" '
            f'iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/></inertial>')


def _cylinder_inertial(mass: float, r: float, h: float, z: float = 0.0) -> str:
    side = mass * (3 * r * r + h * h) / 12
    return _inertial(mass, side, side, mass * r * r / 2, z)


def _link(name: str, geometry: str, color: str, mass: float, inertial: str, z: float = 0.0,
          rpy: str = "0 0 0") -> str:
    origin = f'<origin xyz="0 0 {z}" rpy="{rpy}"/>'
    return (f'<link name="{name}"><visual>{origin}<geometry>{geometry}</geometry>'
            f'<material name="{name}_m"><color rgba="{color}"/></material></visual>'
            f"<collision>{origin}<geometry>{geometry}</geometry></collision>{inertial}</link>")


def _fixed(parent: str, child: str, xyz: str, rpy: str = "0 0 0") -> str:
    return (f'<joint name="{child}_joint" type="fixed"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{xyz}" rpy="{rpy}"/></joint>')


def _sensor(name: str, xyz: str, rpy: str) -> str:
    """A sensor's mount on the base and its optical frame (z forward, x right, y down), as camera drivers stamp."""
    link, optical = f"{name}_link", f"{name}_optical_frame"
    return "".join([
        _link(link, '<box size="0.025 0.09 0.03"/>', DARK, 0.05, _inertial(0.05, 1e-5, 1e-5, 1e-5)),
        _fixed("base_link", link, xyz, rpy),
        f'<link name="{optical}">{_inertial(0.001, 1e-7, 1e-7, 1e-7)}</link>',
        _fixed(link, optical, "0 0 0", f"{-math.pi / 2:.4f} 0 {-math.pi / 2:.4f}"),
    ])


def _wheel(side: str, y: float) -> str:
    name = f"{side}_wheel"
    geometry = f'<cylinder radius="{WHEEL_R}" length="{WHEEL_W}"/>'
    return (_link(name, geometry, DARK, 0.2, _cylinder_inertial(0.2, WHEEL_R, WHEEL_W), rpy=f"{-math.pi / 2:.4f} 0 0")
            + f'<joint name="{name}_joint" type="continuous"><parent link="base_link"/><child link="{name}"/>'
              f'<origin xyz="0 {y} {WHEEL_R}"/><axis xyz="0 1 0"/></joint>')


def _caster(name: str, x: float) -> str:
    r = 0.025
    return (_link(name, f'<sphere radius="{r}"/>', GREY, 0.05, _inertial(0.05, 1e-5, 1e-5, 1e-5))
            + _fixed("base_link", name, f"{x} 0 {r}"))


def urdf() -> str:
    body = f'<cylinder radius="{BODY_R}" length="{BODY_H}"/>'
    deck = '<cylinder radius="0.07" length="0.07"/>'
    parts = [
        _link("base_link", body, ORANGE, 3.0, _cylinder_inertial(3.0, BODY_R, BODY_H, BODY_Z + BODY_H / 2),
              z=BODY_Z + BODY_H / 2),
        f'<link name="base_footprint">{_inertial(0.001, 1e-7, 1e-7, 1e-7)}</link>',
        _fixed("base_link", "base_footprint", "0 0 0"),
        _link("deck", deck, GREY, 0.3, _cylinder_inertial(0.3, 0.07, 0.07)),
        _fixed("base_link", "deck", f"0 0 {DECK_Z}"),
        _wheel("left", TRACK / 2), _wheel("right", -TRACK / 2),
        _caster("front_caster", 0.11), _caster("rear_caster", -0.11),
        f'<link name="imu_link">{_inertial(0.01, 1e-6, 1e-6, 1e-6)}</link>',
        _fixed("base_link", "imu_link", "0 0 0.1"),
        *(_sensor(name, f"{CAMERA_R * math.cos(yaw):.4f} {CAMERA_R * math.sin(yaw):.4f} {CAMERA_Z}", f"0 0 {yaw:.4f}")
          for name, (yaw, *_) in CAMERAS.items()),
        *(_sensor(f"{side}_tof", f"{x:.4f} 0 {z}", f"0 {pitch:.4f} {yaw:.4f}")
          for side, (x, yaw, z, pitch, *_) in TOF.items()),
    ]
    return '<?xml version="1.0"?>\n<robot name="sprocket">\n' + "\n".join(parts) + "\n</robot>\n"


# The odometry the robot drives on: wheels for forward speed (sideways speed held at zero, as on any two-wheeled
# base), the gyro for turning. Wheels slip when the robot turns on the spot; the gyro does not. robot_localization's
# EKF publishes odom -> base_link. An indoor compass heading is not trusted, so no absolute yaw is fused.
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

if __name__ == "__main__":
    print(urdf())
