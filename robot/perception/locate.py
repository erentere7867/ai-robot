"""From a detection box to a place on the floor: depth where a camera has it, else lidar or floor geometry.

An RGB-D camera reads the depth inside the box: the nearest depth a solid cluster of pixels agrees on, the object's
near face, because through an open thing (a table's frame, a chair's legs) the middle of the box shows the wall
behind. A box whose depth points all lie on the floor plane is the floor itself (the detector sometimes calls a
stretch of wooden floor a table) and is dropped. The body as drawn has three plain cameras and no depth (an RGB-D
front camera is an option in robot/ros/model.py): the floor estimate gives the range, from where the thing meets the floor.
That is the part of its outline (the detector's mask) whose floor point is nearest along the lens axis: in a level
camera the lowest rows, but the cameras ride on a tilted head and the rear ones lean, and there the lowest corner of a
box's face is not its middle (foot_of). The box's bottom middle served before, and on the 155-degree rear lenses it is not
where the thing stands: a person straight behind the robot, 66-73 degrees off both rear lenses, was put 0.6 m too far
by one camera (a box hid half of them, so the box's middle missed their visible foot by 56 pixels) and 0.4 m too near
by the other (their box cut by the picture's edge), and the track jumped between the two. The lidar refines the range:
of its returns across the thing's span, those that agree with the floor estimate are the thing (its legs, its front);
one beam through the box's middle passed between legs or hit what stood in front. The window is tight because the
beam passes between chair legs and under a table's top and hits what stands behind: with a 35% window, lidar ranges ran
a median 11-18% long while the robot turned, and the floor estimate alone was within 7%. The lidar is not used at all
for things lower than its beam (a ball). The lidar sits on the head, LIDAR_X ahead of the robot's centre: its
bearings and ranges are taken from there.

The angle the box spans at that range gives its size (pixels are not a fixed angle on a wide lens: at the edge of the
155-degree rear lens one pixel is a quarter of what it is in the middle). Its middle lies one radius beyond its nearest
point: along the lens axis for a camera's view (the lowest point of a round thing's outline, and the nearest depth, are
its nearest along the axis), away from the lidar for a lidar return. A person's middle lies BODY beyond, however wide
their arms make the box. A box cut by the image's side edge is placed from its visible part: a wide thing close by (a
coffee table 1.5 m away spans 35 degrees) can straddle the seam between two cameras so that neither sees it whole, and
the tracker joins the two parts by place and size. Each place comes with its uncertainty, so the tracker can weigh a
depth reading over a floor estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robot.perception.rig import Camera
from robot.ros.model import LIDAR_X, LIDAR_Z

LIDAR_AGREES = 0.12  # the lidar is trusted within this fraction of the floor estimate (its middle half was within 7%)
# A person's middle lies this far beyond their nearest point: their arms make their box wide, but the body is about a
# quarter metre deep (the lidar found a standing person's legs 0.12 m nearer than their middle).
BODY = 0.12
# One standard deviation of a range, metres, by how it was found: a constant part plus a part growing with range.
# From driving tours against the true room: depth and lidar ranges held within about 5%, floor estimates spread 15-25%
# (a chair's floor estimates scattered 0.4 m while claiming 0.18 m, and pulled the chair half a metre off).
SIGMA = {"depth": (0.03, 0.04), "lidar": (0.03, 0.04), "floor": (0.05, 0.10)}  # (0.15 merged neighbours)


@dataclass(frozen=True)
class Place:
    x: float  # robot frame, metres
    y: float
    radius: float
    how: str  # "depth", "lidar" or "floor"
    sigma: float  # one standard deviation of the position, metres


def lidar_range(scan: tuple[np.ndarray, float, float] | None, bearing: float, spread: float = 0.04) -> float | None:
    """Nearest lidar return within `spread` radians of `bearing` (from the lidar); scan is (ranges, angle_min, step)."""
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


def lidar_near(scan: tuple[np.ndarray, float, float] | None, lo: float, hi: float,
               guess: float) -> tuple[float, float] | None:
    """The thing's own lidar returns: those between bearings lo and hi (from the lidar) whose range agrees with the floor
    estimate `guess`. Gives the nearest one's range and the middle of their bearings. Across the whole span, not one
    beam through the middle: behind the robot that beam passed a person's legs and hit a box in front of them, while
    their legs showed two degrees aside."""
    if scan is None:
        return None
    ranges, angle_min, step = scan
    angles = angle_min + np.arange(len(ranges)) * step
    gap = math.atan2(math.sin(hi - lo), math.cos(hi - lo))
    off = np.arctan2(np.sin(angles - lo - gap / 2), np.cos(angles - lo - gap / 2))
    mine = (np.abs(off) <= abs(gap) / 2) & np.isfinite(ranges) &         (np.abs(ranges - guess) <= LIDAR_AGREES * guess + 0.05)
    if not mine.any():
        return None
    return float(ranges[mine].min()), lo + gap / 2 + float(np.mean(off[mine]))


def foot_of(cam: Camera, outline: list[tuple[float, float]] | None) -> tuple[float, float] | None:
    """The pixel where a thing meets the floor, from the lower edge of its outline: the middle of those edge points
    whose floor points are nearest along the lens axis. Edge points that are not on the floor (the underside of a
    table top between its legs) look further away and drop out."""
    c, s = math.cos(cam.yaw), math.sin(cam.yaw)
    reach = []
    for u, v in outline or ():
        contact = cam.floor(u, v)
        if contact is not None:
            reach.append(((contact[0] - cam.x) * c + (contact[1] - cam.y) * s, u, v))
    reach = [r for r in reach if r[0] > 0]
    if not reach:
        return None
    nearest = min(r[0] for r in reach)
    near = [r for r in reach if r[0] <= 1.03 * nearest + 0.02]
    return sum(r[1] for r in near) / len(near), sum(r[2] for r in near) / len(near)


def locate(cam: Camera, box: tuple[float, float, float, float], depth: np.ndarray | None = None,
           scan: tuple[np.ndarray, float, float] | None = None, foot: tuple[float, float] | None = None,
           person: bool = False) -> Place | None:
    """Where the object in pixel box (x1, y1, x2, y2) stands, in the robot frame; None when out of reach or when the
    box holds only floor. `foot`: the pixel where its outline meets the floor (foot_of), else the box's bottom
    middle. `person`: their middle lies BODY beyond their nearest point, not a radius."""
    x1, y1, x2, y2 = box
    mid = (y1 + y2) / 2  # a tilted camera's columns are not single bearings: take the box's sides at its middle row
    a, b = cam.bearing(x1, mid), cam.bearing(x2, mid)
    middle, span = (a + b) / 2, abs(a - b)  # the middle direction: on a wide lens the middle pixel is not
    axis = (math.cos(cam.yaw), math.sin(cam.yaw))
    if depth is not None and cam.depth:
        r0, r1 = int(y1), max(int(y2), int(y1) + 1)
        patch = depth[r0:r1, int(x1):max(int(x2), int(x1) + 1)]
        good = np.isfinite(patch) & (patch > 0.1) & (patch < 8.0)
        if good.any():
            heights = cam.heights(r0, int(x1), patch)  # height above the floor of each pixel's point
            if np.percentile(heights[good], 90) < 0.04:
                return None  # all floor
            d = near_face(patch[good & (heights > 0.03)])  # along the lens axis, as depth cameras measure
            if d is not None:
                level = d * cam.ahead((x1 + x2) / 2, mid)  # that depth as metres along the floor
                seen = level / math.cos(middle - cam.yaw)
                radius = _radius(seen, span)
                forward = level + (BODY if person else radius)  # its middle, further along the axis on the middle ray
                x, y, _ = cam.to_base(forward, forward * math.tan(middle - cam.yaw), 0.0)
                return _place(x, y, radius, "depth", seen)
    # where it meets the floor, unless the box runs off the bottom of the picture: then the picture's edge is not the
    # thing's foot, and the estimate came out half the true range
    fu, fv = foot if foot is not None else ((x1 + x2) / 2, y2)
    contact = cam.floor(fu, fv) if y2 < cam.height - 2 else None
    if contact is None:
        return None
    seen = math.hypot(contact[0] - cam.x, contact[1] - cam.y)
    radius = _radius(seen, span)
    behind = BODY if person else radius
    forward = (contact[0] - cam.x) * axis[0] + (contact[1] - cam.y) * axis[1]
    if cam.height_at(fu, y1, forward) >= LIDAR_Z:  # the lidar's beam passes over anything lower than the lidar
        # (a 25 cm ball under a 27 cm beam took the wall's range behind it, 1 m too far)
        lo, hi = (math.atan2(cam.y + seen * math.sin(e), cam.x + seen * math.cos(e) - LIDAR_X) for e in (a, b))
        lidar = lidar_near(scan, lo, hi, math.hypot(contact[0] - LIDAR_X, contact[1]))
        if lidar is not None:  # the nearest of its returns: its middle lies beyond, away from the lidar
            reach, bearing = lidar[0] + behind, lidar[1]
            return _place(LIDAR_X + reach * math.cos(bearing), reach * math.sin(bearing), radius, "lidar", lidar[0])
    # the lowest point of a thing's outline is its nearest along the lens axis: its middle lies further along the axis
    return _place(contact[0] + behind * axis[0], contact[1] + behind * axis[1], radius, "floor", seen)


def _radius(seen: float, span: float) -> float:
    """The radius of a thing whose near side is `seen` metres away and which spans `span` radians."""
    return max(0.04, min(1.0, seen * math.tan(span / 2)))


def _place(x: float, y: float, radius: float, how: str, seen: float) -> Place | None:
    if not 0.15 <= seen <= 6.0:
        return None
    base, per_metre = SIGMA[how]
    return Place(x, y, radius, how, base + per_metre * seen)
