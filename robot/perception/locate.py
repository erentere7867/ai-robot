"""From a detection box to a place on the floor: depth on the front camera, lidar or floor geometry on the rear ones.

The front RGB-D camera reads the depth inside the box: the nearest depth a solid cluster of pixels agrees on, the object's near face, because through an
open thing (a table's frame, a chair's legs) the middle of the box shows the wall behind. A box whose depth points all
lie on the floor plane is the floor itself (the detector sometimes calls a stretch of wooden floor a table) and is
dropped. The rear cameras have no depth (Fable): the floor estimate (the box's bottom edge, where the object meets the
floor) gives the range, and the lidar return in the box's direction refines it when the two agree within the floor
estimate's own error. The window is tight because the lidar's beam passes between chair legs and under a table's top
and hits what stands behind: with a 35% window, lidar ranges ran a median 11-18% long while the robot turned, and the
floor estimate alone was within 7%. The lidar is not used at all for things lower than its beam (a ball).

The angle the box spans at that range gives its size (pixels are not a fixed angle on a wide lens: at the edge of the
155-degree rear lens one pixel is a quarter of what it is in the middle); its centre lies one radius behind the visible
surface. A box cut by the image's side edge is placed from its visible part: a wide thing close by (a coffee table
1.5 m away spans 35 degrees) can straddle the seam between two cameras so that neither sees it whole, and the tracker
joins the two parts by place and size. Each place comes with its uncertainty, so the tracker can weigh a depth
reading over a floor estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robot.perception.rig import Camera
from robot.ros.model import LIDAR_Z

LIDAR_AGREES = 0.12  # the lidar is trusted within this fraction of the floor estimate (its middle half was within 7%)
# One standard deviation of a range, metres, by how it was found: a constant part plus a part growing with range.
SIGMA = {"depth": (0.03, 0.03), "lidar": (0.03, 0.02), "floor": (0.05, 0.06)}


@dataclass(frozen=True)
class Place:
    x: float  # robot frame, metres
    y: float
    radius: float
    how: str  # "depth", "lidar" or "floor"
    sigma: float  # one standard deviation of the position, metres


def lidar_range(scan: tuple[np.ndarray, float, float] | None, bearing: float, spread: float = 0.04) -> float | None:
    """Nearest lidar return within `spread` radians of `bearing` (robot frame); scan is (ranges, angle_min, step)."""
    if scan is None:
        return None
    ranges, angle_min, step = scan
    k = np.arange(len(ranges))
    angles = np.arctan2(np.sin(angle_min + k * step - bearing), np.cos(angle_min + k * step - bearing))
    near = ranges[(np.abs(angles) <= spread) & np.isfinite(ranges) & (ranges > 0.05)]
    return float(near.min()) if near.size else None


def near_face(depths: np.ndarray, depth_band: float = 0.15) -> float | None:
    """The nearest depth that a solid share of the pixels agrees on: the object's front, however little of its box
    it fills. A fixed percentile reads the wall behind a chair whose legs and back cover less than that share of the
    box (it ran a median 9% long, a quarter of readings 24% long, while the robot turned)."""
    if depths.size == 0:
        return None
    s = np.sort(depths)
    need = max(20, int(0.03 * s.size))
    within = np.searchsorted(s, s + depth_band) - np.arange(s.size)  # pixels within the band behind each one
    first = int(np.argmax(within >= need))
    if within[first] < need:
        return float(np.median(s))
    return float(np.median(s[first:first + within[first]]))


def locate(cam: Camera, box: tuple[float, float, float, float], depth: np.ndarray | None = None,
           scan: tuple[np.ndarray, float, float] | None = None) -> Place | None:
    """Where the object in pixel box (x1, y1, x2, y2) stands, in the robot frame; None when out of reach or when the
    box holds only floor."""
    x1, y1, x2, y2 = box
    u = (x1 + x2) / 2
    bearing = cam.bearing(u)
    surface: float | None = None  # horizontal distance from the camera to the object's near face
    how = "depth"
    if depth is not None and cam.depth:
        r0, r1 = int(y1), max(int(y2), int(y1) + 1)
        patch = depth[r0:r1, int(x1):max(int(x2), int(x1) + 1)]
        good = np.isfinite(patch) & (patch > 0.1) & (patch < 8.0)
        if good.any():
            down = (np.arange(r0, r0 + patch.shape[0], dtype=np.float32)[:, None] - cam.height / 2) / cam.focal
            heights = cam.z - down * patch  # height above the floor of each pixel's point
            if np.percentile(heights[good], 90) < 0.04:
                return None  # all floor
            d = near_face(patch[good & (heights > 0.03)])
            if d is not None:
                px, py, _ = cam.point(u, (y1 + y2) / 2, d)
                surface = math.hypot(px - cam.x, py - cam.y)
    if surface is None:
        # where it meets the floor, unless the box runs off the bottom of the picture: then the picture's edge is not
        # the thing's foot, and the estimate came out half the true range
        floor = cam.floor(u, y2) if y2 < cam.height - 2 else None
        guess = math.hypot(floor[0] - cam.x, floor[1] - cam.y) if floor else None
        lidar = lidar_range(scan, bearing)
        if floor is not None:  # how tall is it? the lidar's beam passes over anything lower than the lidar
            forward = (floor[0] - cam.x) * math.cos(cam.yaw) + (floor[1] - cam.y) * math.sin(cam.yaw)
            if cam.point(u, y1, forward)[2] < LIDAR_Z:
                lidar = None  # (a 25 cm ball under the 27 cm beam took the wall's range behind it, 1 m too far)
        if lidar is not None:  # from the robot's centre; to the camera, roughly the camera's offset less
            lidar -= math.hypot(cam.x, cam.y) * math.cos(bearing - cam.yaw)
        if lidar is not None and guess is not None and abs(lidar - guess) <= LIDAR_AGREES * guess + 0.05:
            surface, how = lidar, "lidar"
        else:
            surface, how = guess, "floor"
    if surface is None or not 0.15 <= surface <= 6.0:
        return None
    span = abs(cam.bearing(x1) - cam.bearing(x2))
    radius = max(0.04, min(1.0, surface * math.tan(span / 2)))
    reach = surface + radius
    base, per_metre = SIGMA[how]
    return Place(cam.x + reach * math.cos(bearing), cam.y + reach * math.sin(bearing), radius, how,
                 base + per_metre * surface)
