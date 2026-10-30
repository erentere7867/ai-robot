"""Pinhole camera model for the head camera: image <-> robot/world floor geometry.

The same model serves the simulator and a physical camera; only the mount numbers
and intrinsics change. Coordinates: world x forward at boot, y left, z up (mm);
heading is CCW-positive degrees; head tilt is positive upward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robot.body.hal import BodyPose


@dataclass(frozen=True)
class CameraModel:
    width: int = 512
    height: int = 384
    vfov_deg: float = 52.0
    # Head pivot on the base and camera offset from the pivot (companion_r1.xml).
    pivot_x_mm: float = 25.0
    pivot_z_mm: float = 196.0
    cam_forward_mm: float = 45.0
    cam_up_mm: float = 88.0
    # The lens looks slightly down relative to the head (xyaxes 0.16/0.987 => 9.2 deg).
    mount_pitch_deg: float = -9.21
    max_range_mm: float = 3500.0

    @property
    def focal_px(self) -> float:
        return (0.5 * self.height) / math.tan(math.radians(0.5 * self.vfov_deg))

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((0.5 * self.width) / self.focal_px))

    def scaled(self, width: int, height: int) -> CameraModel:
        """Same optics at another resolution (square pixels, fixed vertical FOV)."""
        return CameraModel(width, height, self.vfov_deg, self.pivot_x_mm, self.pivot_z_mm,
                           self.cam_forward_mm, self.cam_up_mm, self.mount_pitch_deg, self.max_range_mm)

    def extrinsics(self, pose: BodyPose) -> tuple[float, float, float, float, float]:
        """Camera centre (x, y, z mm) plus yaw and optical-axis pitch (rad)."""
        tilt = math.radians(pose.head_tilt_deg)
        fwd = self.pivot_x_mm + self.cam_forward_mm * math.cos(tilt) - self.cam_up_mm * math.sin(tilt)
        z = self.pivot_z_mm + self.cam_forward_mm * math.sin(tilt) + self.cam_up_mm * math.cos(tilt)
        yaw = math.radians(pose.heading_deg)
        return (pose.x_mm + fwd * math.cos(yaw), pose.y_mm + fwd * math.sin(yaw), z, yaw,
                tilt + math.radians(self.mount_pitch_deg))

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
        """Horizontal angle of an image column relative to the camera axis (+left)."""
        return math.degrees(math.atan2(-(u - 0.5) * self.width, self.focal_px))

    def locate_box(self, pose: BodyPose, bbox: tuple[float, float, float, float]
                   ) -> tuple[float, float, float] | None:
        """Estimate an object's floor centre and radius from a normalized bounding box.

        The bottom-centre of the box touches the floor; the object's centre lies one
        radius further along the viewing ray. Width in pixels gives the radius.
        """
        x1, y1, x2, y2 = bbox
        contact = self.floor_point(pose, 0.5 * (x1 + x2), y2)
        if contact is None:
            return None
        cx, cy, _, _, _ = self.extrinsics(pose)
        dist = math.hypot(contact[0] - cx, contact[1] - cy)
        width_mm = (x2 - x1) * self.width / self.focal_px * dist
        radius = max(30.0, min(450.0, 0.5 * width_mm))
        ux, uy = (contact[0] - cx) / max(dist, 1e-6), (contact[1] - cy) / max(dist, 1e-6)
        return contact[0] + ux * radius, contact[1] + uy * radius, radius

    def in_view(self, pose: BodyPose, x_mm: float, y_mm: float, z_mm: float = 60.0, margin: float = 0.04) -> bool:
        uv = self.project(pose, x_mm, y_mm, z_mm)
        if uv is None:
            return False
        cx, cy, _, _, _ = self.extrinsics(pose)
        return (margin <= uv[0] <= 1 - margin and margin <= uv[1] <= 1 - margin
                and math.hypot(x_mm - cx, y_mm - cy) <= self.max_range_mm)
