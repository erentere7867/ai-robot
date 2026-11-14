"""The camera rig: where each camera sits on Sprocket and how a pixel maps to a direction, a floor point or a 3D point.

Robot frame (base_link): x forward, y left, z up, metres, origin on the floor under the robot's centre. The cameras
ride on the head, which looks 10 degrees up: the front one looks up with it, and the two rear ones, facing 120
degrees to either side, look about 5 degrees down and lean about 9 degrees. So each camera has a full rotation, not
just a yaw, and a pixel's ray is turned by it. `yaw` is where the lens axis points along the floor. The numbers come
from robot/ros/model.py, the body the simulator loads, so the two cannot disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robot.ros.model import CAMERAS, HEAD, HEAD_TILT

Rotation = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class Camera:
    name: str
    hfov: float  # radians
    width: int
    height: int
    x: float
    y: float
    z: float
    turn: Rotation  # camera axes (forward, left, up) in the robot frame, as the columns of a matrix
    depth: bool  # an RGB-D camera

    @property
    def focal(self) -> float:
        return (self.width / 2) / math.tan(self.hfov / 2)

    @property
    def yaw(self) -> float:
        """Where the lens axis points along the floor, radians on the body, + left."""
        return math.atan2(self.turn[1][0], self.turn[0][0])

    def _local(self, u: float, v: float) -> tuple[float, float]:
        """Right and down per metre along the lens axis for pixel (u, v)."""
        return (u - self.width / 2) / self.focal, (v - self.height / 2) / self.focal

    def ray(self, u: float, v: float) -> tuple[float, float, float]:
        """The robot-frame direction of pixel (u, v), per metre along the lens axis."""
        right, down = self._local(u, v)
        t = self.turn
        return (t[0][0] - right * t[0][1] - down * t[0][2], t[1][0] - right * t[1][1] - down * t[1][2],
                t[2][0] - right * t[2][1] - down * t[2][2])

    def bearing(self, u: float, v: float | None = None) -> float:
        """Direction of pixel (u, v) from the robot, radians, + left; v defaults to the picture's middle row."""
        dx, dy, _ = self.ray(u, self.height / 2 if v is None else v)
        return math.atan2(dy, dx)

    def to_base(self, forward: float, left: float, up: float) -> tuple[float, float, float]:
        """A point given level with the floor: `forward` along the lens axis's direction on the floor, `left` of it."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + forward * c - left * s, self.y + forward * s + left * c, self.z + up

    def point(self, u: float, v: float, depth: float) -> tuple[float, float, float]:
        """The 3D point at pixel (u, v) whose depth along the lens axis is `depth` metres."""
        dx, dy, dz = self.ray(u, v)
        return self.x + depth * dx, self.y + depth * dy, self.z + depth * dz

    def pixel(self, bx: float, by: float, bz: float) -> tuple[float, float]:
        """The pixel that shows the robot-frame point (pinhole, as the simulator renders)."""
        t, d = self.turn, (bx - self.x, by - self.y, bz - self.z)
        forward, left, up = (sum(t[k][axis] * d[k] for k in range(3)) for axis in range(3))
        return self.width / 2 - left / forward * self.focal, self.height / 2 - up / forward * self.focal

    def ahead(self, u: float, v: float) -> float:
        """Metres along the floor in the lens axis's direction, per metre of depth, for pixel (u, v)."""
        dx, dy, _ = self.ray(u, v)
        return dx * math.cos(self.yaw) + dy * math.sin(self.yaw)

    def height_at(self, u: float, v: float, forward: float) -> float:
        """How high above the floor the ray through pixel (u, v) is when it has gone `forward` metres along the
        floor in the lens axis's direction."""
        return self.z + forward / max(self.ahead(u, v), 1e-6) * self.ray(u, v)[2]

    def heights(self, top: int, left: int, depth: np.ndarray) -> np.ndarray:
        """The height above the floor of every pixel of a depth patch whose top-left pixel is (left, top)."""
        down = (np.arange(top, top + depth.shape[0], dtype=np.float32)[:, None] - self.height / 2) / self.focal
        right = (np.arange(left, left + depth.shape[1], dtype=np.float32)[None, :] - self.width / 2) / self.focal
        t = self.turn[2]
        return self.z + depth * (t[0] - right * t[1] - down * t[2])

    def floor(self, u: float, v: float) -> tuple[float, float] | None:
        """Where the ray through pixel (u, v) meets the floor; None at or above the horizon."""
        dx, dy, dz = self.ray(u, v)
        if dz >= -1e-3:
            return None
        t = self.z / -dz
        return self.x + t * dx, self.y + t * dy

    def sees(self, bx: float, by: float, margin: float = 0.05) -> bool:
        """Is the robot-frame point (bx, by) inside this camera's horizontal view?"""
        bearing = math.atan2(by - self.y, bx - self.x) - self.yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        return abs(bearing) <= self.hfov / 2 - margin


def _mounted(place: tuple[float, float, float], yaw: float) -> tuple[tuple[float, float, float], Rotation]:
    """A camera's place and rotation on the body, from its place and yaw in the tilted head."""
    c, s = math.cos(HEAD_TILT), math.sin(HEAD_TILT)
    head = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])  # the head's axes in the robot frame: nose up
    spun = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    where = np.array(HEAD) + head @ np.array(place)
    return tuple(float(v) for v in where), tuple(tuple(float(v) for v in row) for row in head @ spun)


RIG = {name: Camera(name, fov, w, h, *_mounted(place, yaw)[0], _mounted(place, yaw)[1], depth)
       for name, (place, yaw, fov, w, h, _, depth) in CAMERAS.items()}


def to_map(pose: tuple[float, float, float], bx: float, by: float) -> tuple[float, float]:
    """A robot-frame point in the map, given the robot's (x, y, yaw)."""
    x, y, yaw = pose
    return x + bx * math.cos(yaw) - by * math.sin(yaw), y + bx * math.sin(yaw) + by * math.cos(yaw)


def to_robot(pose: tuple[float, float, float], mx: float, my: float) -> tuple[float, float]:
    x, y, yaw = pose
    dx, dy = mx - x, my - y
    return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
