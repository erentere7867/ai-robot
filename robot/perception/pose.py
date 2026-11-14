"""Human pose estimation from 2D keypoints and depth, gaze/pointing ray resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot.perception.rig import Camera, to_map


@dataclass(frozen=True)
class HumanPose:
    facing_yaw: float  # map-frame radians
    head_yaw: float  # map-frame radians
    posture: str  # "standing" | "sitting" | "crouching" | "waving" | "pointing"
    pointing_ray: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None  # (origin, unit_dir)
    gaze_ray: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None  # (origin, unit_dir)


def estimate_pose_from_keypoints(keypoints_2d: np.ndarray, cam: Camera, depth_m: float,
                                 robot_pose: tuple[float, float, float]) -> HumanPose:
    """Estimate 3D HumanPose from COCO-17 keypoints, camera, depth, and robot map pose."""
    kpts = np.asarray(keypoints_2d, dtype=np.float32)

    def _joint_3d(idx: int) -> tuple[float, float, float] | None:
        if idx >= len(kpts):
            return None
        pt = kpts[idx]
        u, v = float(pt[0]), float(pt[1])
        if math.isnan(u) or math.isnan(v) or (u <= 0 and v <= 0):
            return None
        if len(pt) >= 3 and float(pt[2]) < 0.1:  # low confidence
            return None
        bx, by, bz = cam.point(u, v, depth_m)
        mx, my = to_map(robot_pose, bx, by)
        return float(mx), float(my), float(bz)

    nose = _joint_3d(0)
    l_eye, r_eye = _joint_3d(1), _joint_3d(2)
    l_ear, r_ear = _joint_3d(3), _joint_3d(4)
    l_shoulder, r_shoulder = _joint_3d(5), _joint_3d(6)
    l_wrist, r_wrist = _joint_3d(9), _joint_3d(10)
    l_hip, r_hip = _joint_3d(11), _joint_3d(12)

    # 1. Facing yaw: perpendicular to shoulder line (r_shoulder -> l_shoulder) pointing forward
    facing_yaw = robot_pose[2] + math.pi  # default: facing toward robot
    if l_shoulder is not None and r_shoulder is not None:
        sdx = l_shoulder[0] - r_shoulder[0]
        sdy = l_shoulder[1] - r_shoulder[1]
        n1 = (-sdy, sdx)
        n2 = (sdy, -sdx)
        # Choose forward vector pointing toward nose if available
        if nose is not None:
            sh_mid = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
            ndx = nose[0] - sh_mid[0]
            ndy = nose[1] - sh_mid[1]
            chosen = n1 if n1[0] * ndx + n1[1] * ndy >= 0 else n2
        else:
            # Pointing roughly toward robot
            sh_mid = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
            rdx = robot_pose[0] - sh_mid[0]
            rdy = robot_pose[1] - sh_mid[1]
            chosen = n1 if n1[0] * rdx + n1[1] * rdy >= 0 else n2
        facing_yaw = math.atan2(chosen[1], chosen[0])
    elif nose is not None:
        facing_yaw = math.atan2(robot_pose[1] - nose[1], robot_pose[0] - nose[0])

    # 2. Head yaw and Gaze ray
    head_yaw = facing_yaw
    head_pos = nose
    head_ref = None
    if l_eye is not None and r_eye is not None:
        head_ref = ((l_eye[0] + r_eye[0]) / 2, (l_eye[1] + r_eye[1]) / 2, (l_eye[2] + r_eye[2]) / 2)
    elif l_ear is not None and r_ear is not None:
        head_ref = ((l_ear[0] + r_ear[0]) / 2, (l_ear[1] + r_ear[1]) / 2, (l_ear[2] + r_ear[2]) / 2)

    if nose is not None and head_ref is not None:
        hdx = nose[0] - head_ref[0]
        hdy = nose[1] - head_ref[1]
        if math.hypot(hdx, hdy) > 0.01:
            head_yaw = math.atan2(hdy, hdx)

    gaze_ray: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    if head_pos is not None:
        gaze_ray = (head_pos, (math.cos(head_yaw), math.sin(head_yaw), 0.0))

    # 3. Posture detection: waving, pointing, crouching, sitting, standing
    posture = "standing"
    pointing_ray: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    # Check horizontal arm extension for pointing
    l_ext = math.hypot(l_wrist[0] - l_shoulder[0], l_wrist[1] - l_shoulder[1]) if (l_wrist and l_shoulder) else 0.0
    r_ext = math.hypot(r_wrist[0] - r_shoulder[0], r_wrist[1] - r_shoulder[1]) if (r_wrist and r_shoulder) else 0.0

    # Check vertical wrist height for waving
    l_wave = (l_wrist is not None and l_shoulder is not None and l_wrist[2] > l_shoulder[2] + 0.12)
    r_wave = (r_wrist is not None and r_shoulder is not None and r_wrist[2] > r_shoulder[2] + 0.12)

    if max(l_ext, r_ext) > 0.35:
        posture = "pointing"
        sh, wr = (l_shoulder, l_wrist) if l_ext >= r_ext else (r_shoulder, r_wrist)
        assert sh is not None and wr is not None
        v = (wr[0] - sh[0], wr[1] - sh[1], wr[2] - sh[2])
        v_len = math.hypot(v[0], v[1], v[2])
        if v_len > 1e-4:
            unit_v = (v[0] / v_len, v[1] / v_len, v[2] / v_len)
            pointing_ray = (sh, unit_v)
    elif (l_wave and l_ext <= 0.35) or (r_wave and r_ext <= 0.35):
        posture = "waving"
    elif nose is not None and nose[2] < 1.1:
        posture = "crouching"
    elif (l_hip is not None or r_hip is not None) and (nose is not None and nose[2] < 1.35):
        posture = "sitting"

    return HumanPose(facing_yaw=facing_yaw, head_yaw=head_yaw, posture=posture,
                     pointing_ray=pointing_ray, gaze_ray=gaze_ray)


def resolve_ray_target(ray: tuple[tuple[float, float, float], tuple[float, float, float]],
                       tracks: list[Any], max_cone_rad: float = 0.32) -> Any | tuple[float, float] | None:
    """Find closest non-person track inside ray cone, or intersect ray with floor z=0."""
    origin, direction = ray
    ox, oy, oz = origin
    dx, dy, dz = direction
    dir_len = math.hypot(dx, dy, dz)
    if dir_len < 1e-6:
        return None

    # Find closest non-person Track within cone
    best_track = None
    best_dist = math.inf

    for t in tracks:
        is_person = getattr(t, "person", False) or getattr(t, "label", "") == "person"
        if is_person:
            continue
        tx, ty = float(t.x), float(t.y)
        tz = float(getattr(t, "z", 0.25))

        wx, wy, wz = tx - ox, ty - oy, tz - oz
        dist = math.hypot(wx, wy, wz)
        if dist < 0.25:  # ignore things at origin
            continue

        # 3D angle
        dot = (wx * dx + wy * dy + wz * dz) / (dist * dir_len)
        dot = max(-1.0, min(1.0, dot))
        angle = math.acos(dot)

        # 2D horizontal angle fallback
        dist_2d = math.hypot(wx, wy)
        dir_2d = math.hypot(dx, dy)
        if dir_2d > 1e-4 and dist_2d > 1e-4:
            dot_2d = (wx * dx + wy * dy) / (dist_2d * dir_2d)
            angle_2d = math.acos(max(-1.0, min(1.0, dot_2d)))
            angle = min(angle, angle_2d)

        if angle <= max_cone_rad and dist < best_dist:
            best_track = t
            best_dist = dist

    if best_track is not None:
        return best_track

    # Floor intersection z = 0.0
    if dz < -1e-3:
        t_floor = -oz / dz
        if 0.0 < t_floor <= 8.0:
            fx = ox + t_floor * dx
            fy = oy + t_floor * dy
            return float(fx), float(fy)

    return None
