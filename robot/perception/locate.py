"""From a detection box to a place on the floor: depth on the front camera, lidar or floor geometry on the rear ones.

The front RGB-D camera reads the depth inside the box: a low percentile, the object's near face, because through an
open thing (a table's frame, a chair's legs) the middle of the box shows the wall behind. The rear cameras have no
depth (Fable): the lidar return in the box's direction gives the range when it agrees with the floor estimate (the
box's bottom edge, where the object meets the floor), and the floor estimate alone serves for things under the
lidar's plane, such as a ball or a cup. The angle the box spans at that range gives its size (pixels are not a fixed
angle on a wide lens: at the edge of the 155-degree rear lens one pixel is a quarter of what it is in the middle); its
centre lies one radius behind the visible surface. A box cut by the image's side edge is skipped: its middle is not
the object's, and another view will show it whole.
"""

from __future__ import annotations

import math

import numpy as np

from robot.perception.rig import Camera


def lidar_range(scan: tuple[np.ndarray, float, float] | None, bearing: float, spread: float = 0.04) -> float | None:
    """Nearest lidar return within `spread` radians of `bearing` (robot frame); scan is (ranges, angle_min, step)."""
    if scan is None:
        return None
    ranges, angle_min, step = scan
    k = np.arange(len(ranges))
    angles = np.arctan2(np.sin(angle_min + k * step - bearing), np.cos(angle_min + k * step - bearing))
    near = ranges[(np.abs(angles) <= spread) & np.isfinite(ranges) & (ranges > 0.05)]
    return float(near.min()) if near.size else None


def locate(cam: Camera, box: tuple[float, float, float, float], depth: np.ndarray | None = None,
           scan: tuple[np.ndarray, float, float] | None = None) -> tuple[float, float, float] | None:
    """Robot-frame centre (x, y) and radius of the object in pixel box (x1, y1, x2, y2); None when out of reach."""
    x1, y1, x2, y2 = box
    if x1 <= 2 or x2 >= cam.width - 2:
        return None
    u = (x1 + x2) / 2
    bearing = cam.bearing(u)
    surface: float | None = None  # horizontal distance from the camera to the object's near face
    if depth is not None and cam.depth:
        patch = depth[int(y1):max(int(y2), int(y1) + 1), int(x1):max(int(x2), int(x1) + 1)]
        values = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 8.0)]
        if values.size:
            d = float(np.percentile(values, 15))
            px, py, _ = cam.point(u, (y1 + y2) / 2, d)
            surface = math.hypot(px - cam.x, py - cam.y)
    if surface is None:
        floor = cam.floor(u, y2)
        guess = math.hypot(floor[0] - cam.x, floor[1] - cam.y) if floor else None
        lidar = lidar_range(scan, bearing)
        if lidar is not None:  # from the robot's centre; to the camera, roughly the camera's offset less
            lidar -= math.hypot(cam.x, cam.y) * math.cos(bearing - cam.yaw)
        if lidar is not None and guess is not None and abs(lidar - guess) <= 0.35 * guess + 0.2:
            surface = lidar
        else:
            surface = guess
    if surface is None or not 0.15 <= surface <= 6.0:
        return None
    span = abs(cam.bearing(x1) - cam.bearing(x2))
    radius = max(0.04, min(1.0, surface * math.tan(span / 2)))
    reach = surface + radius
    return (cam.x + reach * math.cos(bearing), cam.y + reach * math.sin(bearing), radius)
