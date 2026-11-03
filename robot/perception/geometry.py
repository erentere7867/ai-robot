"""Pinhole camera models for the robot's cameras: image <-> robot/world floor geometry.

The same model serves the simulator and the physical cameras (rectified images); only the mount numbers and
intrinsics change, and they all come from config/body.yaml (`sensors.cameras`). Coordinates: world x forward
at boot, y left, z up (mm); heading is CCW-positive degrees; head tilt is positive upward.

The rig is "fovea + periphery": a 90-degree camera in the tilting head sees ahead in detail (the detector's
range and the VLM's eye), and two 140-degree cameras on the sensor mast cover the rest of the circle (people,
large things, anything behind). Together they see all around without turning.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path

from robot.body.hal import BodyPose


@dataclass(frozen=True)
class CameraModel:
    name: str = "front"
    width: int = 640
    height: int = 480
    hfov_deg: float = 90.0
    # Mount. On the head: position relative to the head pivot, turned by the head tilt (head cameras face
    # forward). On the body: position relative to the robot's centre on the floor. x forward, y left, z up.
    on_head: bool = True
    x_mm: float = 45.0
    y_mm: float = 0.0
    z_mm: float = 88.0
    yaw_deg: float = 0.0
    pitch_deg: float = -9.21  # the lens looks slightly down relative to its mount
    pivot_x_mm: float = 25.0
    pivot_z_mm: float = 196.0
    max_range_mm: float = 3500.0
    device: int = 0  # index of the physical camera (CSI port or /dev/video*)

    @property
    def focal_px(self) -> float:
        return (0.5 * self.width) / math.tan(math.radians(0.5 * self.hfov_deg))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((0.5 * self.height) / self.focal_px))

    def scaled(self, width: int, height: int) -> CameraModel:
        """Same optics at another resolution (square pixels, fixed horizontal FOV)."""
        return CameraModel(self.name, width, height, self.hfov_deg, self.on_head, self.x_mm, self.y_mm, self.z_mm,
                           self.yaw_deg, self.pitch_deg, self.pivot_x_mm, self.pivot_z_mm, self.max_range_mm,
                           self.device)

    def mount(self, head_tilt_deg: float = 0.0) -> tuple[float, float, float, float]:
        """Camera centre in the robot frame (forward, left, up mm) and optical-axis pitch (rad)."""
        if not self.on_head:
            return self.x_mm, self.y_mm, self.z_mm, math.radians(self.pitch_deg)
        tilt = math.radians(head_tilt_deg)
        fwd = self.pivot_x_mm + self.x_mm * math.cos(tilt) - self.z_mm * math.sin(tilt)
        up = self.pivot_z_mm + self.x_mm * math.sin(tilt) + self.z_mm * math.cos(tilt)
        return fwd, self.y_mm, up, tilt + math.radians(self.pitch_deg)

    def extrinsics(self, pose: BodyPose) -> tuple[float, float, float, float, float]:
        """Camera centre (x, y, z mm) plus yaw and optical-axis pitch (rad) in the world."""
        fwd, left, z, pitch = self.mount(pose.head_tilt_deg)
        h = math.radians(pose.heading_deg)
        return (pose.x_mm + fwd * math.cos(h) - left * math.sin(h), pose.y_mm + fwd * math.sin(h) + left * math.cos(h),
                z, h + math.radians(self.yaw_deg), pitch)

    def project(self, pose: BodyPose, x_mm: float, y_mm: float, z_mm: float) -> tuple[float, float] | None:
        """World point -> normalized image (u right, v down), or None when behind the camera."""
        cx, cy, cz, yaw, pitch = self.extrinsics(pose)
        dx, dy, dz = x_mm - cx, y_mm - cy, z_mm - cz
        rx = dx * math.cos(yaw) + dy * math.sin(yaw)
        ry = -dx * math.sin(yaw) + dy * math.cos(yaw)
        fwd = rx * math.cos(pitch) + dz * math.sin(pitch)
        up = -rx * math.sin(pitch) + dz * math.cos(pitch)
        if fwd <= 30.0:
            return None
        f = self.focal_px
        return (0.5 + f * (-ry / fwd) / self.width, 0.5 - f * (up / fwd) / self.height)

    def ray(self, pose: BodyPose, u: float, v: float) -> tuple[float, float, float]:
        """Unit world-frame direction through normalized pixel (u, v)."""
        _, _, _, yaw, pitch = self.extrinsics(pose)
        f = self.focal_px
        right = (u - 0.5) * self.width / f
        up = -(v - 0.5) * self.height / f
        fx = math.cos(pitch) - up * math.sin(pitch)
        fz = math.sin(pitch) + up * math.cos(pitch)
        fy = -right
        wx = fx * math.cos(yaw) - fy * math.sin(yaw)
        wy = fx * math.sin(yaw) + fy * math.cos(yaw)
        n = math.sqrt(wx * wx + wy * wy + fz * fz)
        return wx / n, wy / n, fz / n

    def floor_point(self, pose: BodyPose, u: float, v: float) -> tuple[float, float] | None:
        """Intersect the pixel ray with the floor; None if at/above the horizon or out of range."""
        cx, cy, cz, _, _ = self.extrinsics(pose)
        dx, dy, dz = self.ray(pose, u, v)
        if dz >= -1e-3:
            return None
        t = -cz / dz
        x, y = cx + t * dx, cy + t * dy
        if math.hypot(x - cx, y - cy) > self.max_range_mm:
            return None
        return x, y

    def bearing_deg(self, u: float) -> float:
        """Horizontal angle of an image column relative to the robot's heading (+left)."""
        return self.yaw_deg + math.degrees(math.atan2(-(u - 0.5) * self.width, self.focal_px))

    def locate_box(self, pose: BodyPose, bbox: tuple[float, float, float, float], depth_mm: float | None = None
                   ) -> tuple[float, float, float] | None:
        """Estimate an object's floor centre and radius from a normalized bounding box.

        The bottom-centre of the box touches the floor; the object's centre lies `depth_mm` (default: one
        radius) further along the viewing ray. Width in pixels gives the radius.
        """
        x1, y1, x2, y2 = bbox
        contact = self.floor_point(pose, 0.5 * (x1 + x2), y2)
        if contact is None:
            return None
        cx, cy, _, _, _ = self.extrinsics(pose)
        dist = math.hypot(contact[0] - cx, contact[1] - cy)
        # Width from the floor points under the box's bottom corners: exact anywhere in the image, where pixels
        # times distance over focal length is only right near the optical axis (a wide lens stretches its edges).
        left, right = self.floor_point(pose, x1, y2), self.floor_point(pose, x2, y2)
        if left is not None and right is not None:
            width_mm = math.hypot(left[0] - right[0], left[1] - right[1])
        else:
            width_mm = (x2 - x1) * self.width / self.focal_px * dist
        radius = max(30.0, min(450.0, 0.5 * width_mm))
        ux, uy = (contact[0] - cx) / max(dist, 1e-6), (contact[1] - cy) / max(dist, 1e-6)
        push = radius if depth_mm is None else depth_mm
        return contact[0] + ux * push, contact[1] + uy * push, radius

    def in_view(self, pose: BodyPose, x_mm: float, y_mm: float, z_mm: float = 60.0, margin: float = 0.04) -> bool:
        uv = self.project(pose, x_mm, y_mm, z_mm)
        if uv is None:
            return False
        cx, cy, _, _, _ = self.extrinsics(pose)
        return (margin <= uv[0] <= 1 - margin and margin <= uv[1] <= 1 - margin
                and math.hypot(x_mm - cx, y_mm - cy) <= self.max_range_mm)

    def covers(self, bearing_deg: float, margin_deg: float = 0.0) -> bool:
        """Does this camera see the direction `bearing_deg` (relative to the robot's heading)?"""
        off = (bearing_deg - self.yaw_deg + 180.0) % 360.0 - 180.0
        return abs(off) <= 0.5 * self.hfov_deg - margin_deg


def load_rig(body_yaml: Path | str) -> tuple[CameraModel, ...]:
    """The cameras listed under `sensors.cameras` in body.yaml (the front head camera when none are)."""
    from robot.core.config import load_yaml_file

    raw = load_yaml_file(body_yaml)
    cams = ((raw.get("sensors") or {}).get("cameras")) or []
    if not cams:
        return (CameraModel(),)
    rig = []
    for i, c in enumerate(cams):
        width, height = (int(v) for v in c.get("resolution", [640, 480]))
        rig.append(CameraModel(
            name=str(c.get("name", f"camera_{i}")), width=width, height=height,
            hfov_deg=float(c.get("hfov_deg", 90.0)), on_head=str(c.get("mount", "head")) == "head",
            x_mm=float(c.get("x_mm", 0.0)), y_mm=float(c.get("y_mm", 0.0)), z_mm=float(c.get("z_mm", 0.0)),
            yaw_deg=float(c.get("yaw_deg", 0.0)), pitch_deg=float(c.get("pitch_deg", 0.0)),
            pivot_x_mm=float(c.get("pivot_x_mm", 25.0)), pivot_z_mm=float(c.get("pivot_z_mm", 196.0)),
            max_range_mm=float(c.get("max_range_mm", 3500.0)), device=int(c.get("device", i))))
    if len({c.name for c in rig}) != len(rig):
        raise ValueError("camera names in body.yaml must be unique")
    return tuple(rig)


@functools.cache
def default_rig() -> tuple[CameraModel, ...]:
    from robot.core.config import PROJECT_ROOT

    return load_rig(PROJECT_ROOT / "config" / "body.yaml")


def camera_for(rig: tuple[CameraModel, ...], bearing_deg: float) -> CameraModel:
    """The camera that sees `bearing_deg` best (closest to its optical axis)."""
    return min(rig, key=lambda c: abs((bearing_deg - c.yaw_deg + 180.0) % 360.0 - 180.0) / (0.5 * c.hfov_deg))
