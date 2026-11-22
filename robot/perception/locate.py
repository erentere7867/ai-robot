"""From a detection box to a place on the floor: depth on the front camera, lidar or floor geometry on the rear ones.

The front RGB-D camera reads the depth inside the box: the nearest depth a solid cluster of pixels agrees on, the object's near face, because through an
open thing (a table's frame, a chair's legs) the middle of the box shows the wall behind. A box whose depth points all
lie on the floor plane is the floor itself (the detector sometimes calls a stretch of wooden floor a table) and is
dropped. The rear cameras have no depth: the floor estimate gives the range, from where the thing meets the floor,
the lowest point of its outline (the detector's mask). That point's image row gives its distance along the lens axis
and its column how far to the side. The box's bottom middle served before, and on the 155-degree rear lenses it is not
where the thing stands: a person straight behind the robot, 66-73 degrees off both rear lenses, was put 0.6 m too far
by one camera (a box hid half of them, so the box's middle missed their visible foot by 56 pixels) and 0.4 m too near
by the other (their box cut by the picture's edge), and the track jumped between the two. The lidar refines the range:
of its returns across the thing's span, those that agree with the floor estimate are the thing (its legs, its front);
one beam through the box's middle passed between legs or hit what stood in front. The window is tight because the
beam passes between chair legs and under a table's top and hits what stands behind: with a 35% window, lidar ranges ran
a median 11-18% long while the robot turned, and the floor estimate alone was within 7%. The lidar is not used at all
for things lower than its beam (a ball).

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

SCAN_Z = 0.25  # beam height for scan refinement (replaces legacy spinning lidar)
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
    width: float = 0.0
    depth_extent: float = 0.0
    orientation_yaw: float = 0.0
    z: float = 0.0  # median elevation of non-floor crop pixels (robot frame, metres); 0 for floor-flat targets


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


def lidar_near(scan: tuple[np.ndarray, float, float] | None, lo: float, hi: float,
               guess: float) -> tuple[float, float] | None:
    """The thing's own lidar returns: those between bearings lo and hi (robot frame) whose range agrees with the floor
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


def locate(cam: Camera, box: tuple[float, float, float, float], depth: np.ndarray | None = None,
           scan: tuple[np.ndarray, float, float] | None = None, foot: tuple[float, float] | None = None,
           person: bool = False, mask: np.ndarray | None = None) -> Place | None:
    """Where the object in pixel box (x1, y1, x2, y2) stands, in the robot frame; None when out of reach or when the
    box holds only floor. `foot`: the pixel where its outline meets the floor (the lowest point of the detector's
    mask), else the box's bottom middle. `person`: their middle lies BODY beyond their nearest point, not a radius."""
    x1, y1, x2, y2 = box
    a, b = cam.bearing(x1), cam.bearing(x2)
    middle, span = (a + b) / 2, abs(a - b)  # the middle direction: on a wide lens the middle pixel is not
    axis = (math.cos(cam.yaw), math.sin(cam.yaw))
    fu, fv = foot if foot is not None else ((x1 + x2) / 2, y2)
    contact = cam.floor(fu, fv) if y2 < cam.height - 2 else None
    if contact is not None:
        floor_seen = math.hypot(contact[0] - cam.x, contact[1] - cam.y)
        floor_radius = _radius(floor_seen, span)
        behind = BODY if person else floor_radius
        floor_x, floor_y = contact[0] + behind * axis[0], contact[1] + behind * axis[1]
    if depth is not None:
        r0, r1 = int(y1), max(int(y2), int(y1) + 1)
        c0, c1 = int(x1), max(int(x2), int(x1) + 1)
        patch = depth[r0:r1, c0:c1]
        good = np.isfinite(patch) & (patch > 0.1) & (patch < 8.0)
        if good.any():
            down = (np.arange(r0, r0 + patch.shape[0], dtype=np.float32)[:, None] - cam.height / 2) / cam.focal
            # No return is inf. At the horizon, 0 * inf is undefined even though the pixel is later masked out.
            heights = cam.z - down * np.where(good, patch, 0.0)
            if np.percentile(heights[good], 90) < 0.04:
                return None  # all floor
            valid_mask = good & (heights > 0.03)
            if mask is not None:
                if mask.shape == patch.shape:
                    m_patch = mask.astype(bool)
                elif mask.shape == depth.shape:
                    m_patch = mask[r0:r1, c0:c1].astype(bool)
                else:
                    m_patch = None
                if m_patch is not None:
                    masked = valid_mask & m_patch
                    if np.count_nonzero(masked) >= 12:
                        valid_mask = masked
            else:
                box_w = c1 - c0
                inset_w = int(0.10 * box_w)
                if inset_w > 0:
                    inset_mask = valid_mask.copy()
                    if x1 > 4:
                        inset_mask[:, :inset_w] = False
                    if x2 < cam.width - 4:
                        inset_mask[:, -inset_w:] = False
                    if np.count_nonzero(inset_mask) >= 16:
                        valid_mask = inset_mask

            u_cols = np.arange(c0, c1, dtype=np.float32)[None, :]
            sec_theta = np.sqrt(1.0 + ((u_cols - cam.width / 2.0) / cam.focal) ** 2)
            r_patch = patch * sec_theta
            seen = near_face(r_patch[valid_mask])  # true radial range to the front surface
            if seen is not None:
                r_pts = r_patch[valid_mask]
                d_pts = patch[valid_mask]
                near_pts = (r_pts >= seen - 0.15) & (r_pts <= seen + 0.50)
                v_rel, u_rel = np.where(valid_mask)
                u_abs = c0 + u_rel
                if np.count_nonzero(near_pts) >= 6:
                    u_med = float(np.median(u_abs[near_pts]))
                else:
                    u_med = (x1 + x2) / 2.0
                bearing_eff = cam.bearing(u_med)
                rel = bearing_eff - cam.yaw
                radius = _radius(seen, span)
                reach = seen + (BODY if person else radius)
                if contact is not None and foot is not None and not person and x1 > 4 and x2 < cam.width - 4:
                    # A complete outline gives a second estimate if its foot is consistent with the floor.
                    # Do not project a shelf (or an occluded object's apparent bottom) blindly onto the floor.
                    floor_reach = math.hypot(floor_x - cam.x, floor_y - cam.y)
                    if abs(floor_reach - reach) <= radius + SIGMA["floor"][0] + SIGMA["floor"][1] * reach:
                        reach = (reach + floor_reach) / 2
                        radius = (radius + floor_radius) / 2
                        rel = middle - cam.yaw
                x, y, _ = cam.to_base(reach * math.cos(rel), reach * math.sin(rel), 0.0)

                # 3D extents (width, depth_extent) and orientation_yaw from depth cluster
                width, depth_extent, ori_yaw = 2 * radius, (BODY * 2 if person else 2 * radius), cam.yaw
                if len(v_rel) >= 10 and np.count_nonzero(near_pts) >= 6:
                    u_f = u_abs[near_pts]
                    d_f = d_pts[near_pts]
                    x_opt = (u_f.astype(np.float32) - cam.width / 2) / cam.focal
                    xc = d_f
                    yc = -x_opt * d_f
                    cos_y, sin_y = math.cos(cam.yaw), math.sin(cam.yaw)
                    xb_pts = cam.x + xc * cos_y - yc * sin_y
                    yb_pts = cam.y + xc * sin_y + yc * cos_y
                    coords = np.stack([xb_pts, yb_pts], axis=1)
                    cov = np.cov(coords, rowvar=False)
                    if cov.shape == (2, 2) and np.isfinite(cov).all():
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        main_vec = eigvecs[:, 1]
                        ori_yaw = float(math.atan2(main_vec[1], main_vec[0]))
                        proj1 = coords @ main_vec
                        proj2 = coords @ np.array([-main_vec[1], main_vec[0]])
                        span1 = float(np.percentile(proj1, 95) - np.percentile(proj1, 5))
                        span2 = float(np.percentile(proj2, 95) - np.percentile(proj2, 5))
                        width = max(0.04, span1)
                        depth_extent = max(0.04, span2)

                return _place(x, y, radius, "depth", seen, width, depth_extent, ori_yaw)
    # where it meets the floor, unless the box runs off the bottom of the picture: then the picture's edge is not the
    # thing's foot, and the estimate came out half the true range
    if contact is None:
        return None
    seen, radius = floor_seen, floor_radius
    forward = (contact[0] - cam.x) * axis[0] + (contact[1] - cam.y) * axis[1]
    if scan is not None and cam.point(fu, y1, forward)[2] >= SCAN_Z:  # scan beam passes over low things
        lo, hi = (math.atan2(cam.y + seen * math.sin(e), cam.x + seen * math.cos(e)) for e in (a, b))
        lidar = lidar_near(scan, lo, hi, math.hypot(*contact))
        if lidar is not None:  # the nearest of its returns: its middle lies beyond
            reach, bearing = lidar[0] + behind, lidar[1]
            return _place(reach * math.cos(bearing), reach * math.sin(bearing), radius, "lidar", lidar[0],
                          2 * radius, behind * 2, bearing)
    # the lowest point of a thing's outline is its nearest along the lens axis: its middle lies further along the axis
    return _place(floor_x, floor_y, radius, "floor", seen, 2 * radius, behind * 2, middle)


def locate_crop(cam: Camera, box: tuple[float, float, float, float], depth: np.ndarray | None = None,
                scan: tuple[np.ndarray, float, float] | None = None, on_floor: bool = False) -> Place | None:
    """Range a coarse visual crop (x1, y1, x2, y2) into a robot-frame Place on its front surface or floor spot.

    Unlike `locate()` for tight detector boxes, a coarse crop can include foreground furniture near its side edges
    (sampling the middle half of columns and the 25th depth percentile cut `dock -> kitchen` edge error from 1.50 m
    to 0.00 m). Floor-flat targets (rugs) are kept when at least 60% of below-horizon pixels lie on the floor and
    weighted by r^2 (exact flat-floor Jacobian dr ∝ r^2 dv) over [0.4, 3.5] m: unweighted median fell at 0.72 m near
    the robot's feet, linear r stopped at 0.88 m, r^2 up to 3.0 m landed at 0.69 m (1.02 m after Nav2's 0.25 m goal
    tolerance), and r^2 up to 3.5 m landed the 2-camera centroid's nearest free cell at 0.56 m."""
    x1, y1, x2, y2 = box
    a, b = cam.bearing(x1), cam.bearing(x2)
    u_mid = (x1 + x2) / 2.0
    bearing, span = cam.bearing(u_mid), abs(a - b)
    rel = bearing - cam.yaw
    seen, how, floor_flat, obj_z = None, "", False, 0.0
    if depth is not None:
        dh, dw = depth.shape[:2]
        dcam = cam.at(dw, dh)
        inset = 0.25 * (x2 - x1)
        c0 = max(0, int((x1 + inset) * dw / cam.width))
        c1 = min(dw, max(c0 + 1, int((x2 - inset) * dw / cam.width)))
        r0 = max(0, int(y1 * dh / cam.height))
        r1 = min(dh, max(r0 + 1, int(y2 * dh / cam.height)))
        patch = depth[r0:r1, c0:c1]
        good = np.isfinite(patch) & (patch > 0.15) & (patch < 7.0)
        if good.any():
            rows = np.arange(r0, r1, dtype=np.float32)[:, None]
            down = (rows - dcam.height / 2.0) / dcam.focal
            heights = dcam.z - down * np.where(good, patch, 0.0)
            u_cols = np.arange(c0, c1, dtype=np.float32)[None, :]
            r_patch = patch * np.sqrt(1.0 + ((u_cols - dcam.width / 2.0) / dcam.focal) ** 2)
            below = good & (rows >= dcam.height / 2.0)
            below_floor = below & (heights <= 0.06)
            floor_px = below_floor & (r_patch >= 0.4) & (r_patch <= 3.5)
            obj_px = good & (heights > 0.04)
            below_n, floor_n = int(np.count_nonzero(below)), int(np.count_nonzero(floor_px))
            mostly_floor = floor_n >= 12 and int(np.count_nonzero(below_floor)) >= 0.6 * max(1, below_n)
            if on_floor and mostly_floor:
                vals = r_patch[floor_px]
                seen, how, floor_flat = float(np.average(vals, weights=vals ** 2)), "floor", True
            elif np.count_nonzero(obj_px) >= 12:
                cand = float(np.percentile(r_patch[obj_px], 25))
                if cand <= 6.0:
                    seen, how, obj_z = cand, "depth", float(np.median(heights[obj_px]))
                elif floor_n >= 12:  # far back wall behind a distant floor target
                    vals = r_patch[floor_px]
                    seen, how, floor_flat = float(np.average(vals, weights=vals ** 2)), "floor", True
            elif floor_n >= 12:
                vals = r_patch[floor_px]
                seen, how, floor_flat = float(np.average(vals, weights=vals ** 2)), "floor", True
    if seen is None and scan is not None:
        sr = lidar_range(scan, bearing, spread=max(0.06, span / 3.0))
        if sr is not None:
            return _place(sr * math.cos(bearing), sr * math.sin(bearing), 0.2, "lidar", sr)
    if seen is None:
        fp = cam.floor(u_mid, max(cam.height / 2.0 + 12.0, (y1 + y2) / 2.0))
        if fp is None:
            return None
        seen = math.hypot(fp[0] - cam.x, fp[1] - cam.y)
        return _place(fp[0], fp[1], 0.0 if on_floor else 0.2, "floor", seen)
    x, y, _ = cam.to_base(seen * math.cos(rel), seen * math.sin(rel), 0.0)
    radius = 0.0 if floor_flat else max(0.15, min(0.35, _radius(seen, span) * 0.5))
    return _place(x, y, radius, how, seen, z=0.0 if floor_flat else obj_z)


def _radius(seen: float, span: float) -> float:
    """The radius of a thing whose near side is `seen` metres away and which spans `span` radians."""
    return max(0.04, min(1.0, seen * math.tan(span / 2)))


def _place(x: float, y: float, radius: float, how: str, seen: float,
           width: float = 0.0, depth_extent: float = 0.0, orientation_yaw: float = 0.0,
           z: float = 0.0) -> Place | None:
    if not 0.15 <= seen <= 6.0:
        return None
    base, per_metre = SIGMA[how]
    w = width if width > 0.0 else 2 * radius
    de = depth_extent if depth_extent > 0.0 else 2 * radius
    return Place(x, y, radius, how, base + per_metre * seen, w, de, orientation_yaw, z)
