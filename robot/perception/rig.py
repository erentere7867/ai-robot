"""The camera rig: where each camera sits on Sprocket and how a pixel maps to a direction, a floor point or a 3D point.

Robot frame (base_link): x forward, y left, z up, metres, origin on the floor under the robot's centre. Cameras are
fixed and level; each has a yaw on the body (front 0, rear-left +120, rear-right -120 degrees) and a horizontal field
of view. The numbers come from robot/ros/model.py, the body the simulator loads, so the two cannot disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robot.ros.model import CAMERA_R, CAMERA_Z, CAMERAS


@dataclass(frozen=True)
class Camera:
    name: str
    yaw: float  # radians on the body, + left
    hfov: float  # radians
    width: int
    height: int
    x: float
    y: float
    z: float
    depth: bool  # an RGB-D camera

    @property
    def focal(self) -> float:
        return (self.width / 2) / math.tan(self.hfov / 2)

    def _local(self, u: float, v: float) -> tuple[float, float]:
        """Right and down per metre forward for pixel (u, v)."""
        return (u - self.width / 2) / self.focal, (v - self.height / 2) / self.focal

    def bearing(self, u: float) -> float:
        """Direction of image column u from the robot, radians, + left."""
        right, _ = self._local(u, self.height / 2)
        return self.yaw + math.atan2(-right, 1.0)

    def to_base(self, forward: float, left: float, up: float) -> tuple[float, float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + forward * c - left * s, self.y + forward * s + left * c, self.z + up

    def point(self, u: float, v: float, depth: float) -> tuple[float, float, float]:
        """The 3D point at pixel (u, v) whose depth along the optical axis is `depth` metres."""
        right, down = self._local(u, v)
        return self.to_base(depth, -right * depth, -down * depth)

    def floor(self, u: float, v: float) -> tuple[float, float] | None:
        """Where the ray through pixel (u, v) meets the floor; None at or above the horizon."""
        right, down = self._local(u, v)
        if down <= 1e-3:
            return None
        t = self.z / down
        x, y, _ = self.to_base(t, -right * t, -self.z)
        return x, y

    def sees(self, bx: float, by: float, margin: float = 0.05) -> bool:
        """Is the robot-frame point (bx, by) inside this camera's horizontal view?"""
        bearing = math.atan2(by - self.y, bx - self.x) - self.yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        return abs(bearing) <= self.hfov / 2 - margin


RIG = {name: Camera(name, yaw, fov, w, h, CAMERA_R * math.cos(yaw), CAMERA_R * math.sin(yaw), CAMERA_Z, depth)
       for name, (yaw, fov, w, h, _, depth) in CAMERAS.items()}


def to_map(pose: tuple[float, float, float], bx: float, by: float) -> tuple[float, float]:
    """A robot-frame point in the map, given the robot's (x, y, yaw)."""
    x, y, yaw = pose
    return x + bx * math.cos(yaw) - by * math.sin(yaw), y + bx * math.sin(yaw) + by * math.cos(yaw)


def to_robot(pose: tuple[float, float, float], mx: float, my: float) -> tuple[float, float]:
    x, y, yaw = pose
    dx, dy = mx - x, my - y
    return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
