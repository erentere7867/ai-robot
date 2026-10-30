"""Capability scenarios and their ground-truth graders.

Everything here is pure: graders read a recorded `ScenarioRun` (true poses, true object positions, final
replies) and never touch a running harness. The grader may use simulator ground truth; the robot may not.
Poses are dicts with `x_mm`, `y_mm`, `heading_deg` (+CCW/left) and `head_tilt_deg`.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Pose = Mapping[str, float]
SceneObject = Mapping[str, Any]

ROBOT_HALF_LENGTH_MM = 141.0
ROBOT_HALF_WIDTH_MM = 114.0
ROBOT_RADIUS_MM = math.hypot(ROBOT_HALF_LENGTH_MM, ROBOT_HALF_WIDTH_MM)

FAMILY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "box": ("box", "carton", "cardboard"),
    "ball": ("ball", "sphere"),
    "mug": ("mug", "cup", "cylinder"),
    "cube": ("cube", "block"),
    "duck": ("duck",),
    "dock": ("dock", "charger", "station"),
}
_FAMILY_PATTERNS = {
    family: re.compile(r"\b(?:" + "|".join(words) + r")(?:es|s)?\b", re.IGNORECASE)
    for family, words in FAMILY_SYNONYMS.items()
}
_LOCATION_CUE = re.compile(
    r"\b(?:behind|left|right|front|back|ahead|near|next to|beside|by the|corner|wall|"
    r"north|south|east|west|o'?clock|\d+(?:\.\d+)?\s*(?:mm|cm|m|meters?|millimeters?|degrees?|deg))\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- text helpers


def mentioned_families(text: str) -> set[str]:
    """Object families named in free text, using the synonym table."""
    return {family for family, pattern in _FAMILY_PATTERNS.items() if pattern.search(text or "")}


def family_of(obj: SceneObject) -> str | None:
    """Map a scene object to its family from its label (falling back to its id)."""
    for source in (str(obj.get("label", "")), str(obj.get("id", ""))):
        tokens = set(re.findall(r"[a-z]+", source.lower()))
        for family, words in FAMILY_SYNONYMS.items():
            if tokens.intersection(words):
                return family
    return None


def families_present(objects: Iterable[SceneObject]) -> set[str]:
    return {f for f in (family_of(o) for o in objects) if f}


def has_location_cue(text: str) -> bool:
    return bool(_LOCATION_CUE.search(text or ""))


# --------------------------------------------------------------------------- geometry


def wrap_deg(angle: float) -> float:
    """Wrap into (-180, 180]."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def heading_change_deg(before: Pose, after: Pose) -> float:
    return wrap_deg(float(after["heading_deg"]) - float(before["heading_deg"]))


def distance_mm(a: Pose | SceneObject, b: Pose | SceneObject) -> float:
    return math.hypot(float(b["x_mm"]) - float(a["x_mm"]), float(b["y_mm"]) - float(a["y_mm"]))


def relative_bearing_deg(pose: Pose, x_mm: float, y_mm: float) -> float:
    """Bearing of a world point from the robot's heading (+left)."""
    world = math.degrees(math.atan2(y_mm - float(pose["y_mm"]), x_mm - float(pose["x_mm"])))
    return wrap_deg(world - float(pose["heading_deg"]))


def forward_lateral_mm(before: Pose, after: Pose) -> tuple[float, float]:
    """Displacement expressed in the starting pose's frame (forward, +left)."""
    yaw = math.radians(float(before["heading_deg"]))
    dx = float(after["x_mm"]) - float(before["x_mm"])
    dy = float(after["y_mm"]) - float(before["y_mm"])
    return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)


def angle_between_deg(v1: tuple[float, float], v2: tuple[float, float]) -> float | None:
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def footprint_radius_mm(obj: SceneObject) -> float:
    """Circumscribed floor radius, used for non-overlapping placement."""
    return 0.5 * math.hypot(float(obj.get("width_mm", 100.0)), float(obj.get("depth_mm", 100.0)))


def footprint_overlaps(pose: Pose, obj: SceneObject, *, tolerance_mm: float = 10.0) -> bool:
    """True if the robot's rectangular footprint penetrates the object's footprint by > tolerance."""
    yaw = math.radians(float(pose["heading_deg"]))
    ux, uy = math.cos(yaw), math.sin(yaw)
    vx, vy = -uy, ux
    dx = float(obj["x_mm"]) - float(pose["x_mm"])
    dy = float(obj["y_mm"]) - float(pose["y_mm"])
    half_w = 0.5 * float(obj.get("width_mm", 100.0))
    half_d = 0.5 * float(obj.get("depth_mm", 100.0))
    if obj.get("shape") in ("sphere", "cylinder"):
        lx, ly = dx * ux + dy * uy, dx * vx + dy * vy
        cx = max(-ROBOT_HALF_LENGTH_MM, min(ROBOT_HALF_LENGTH_MM, lx))
        cy = max(-ROBOT_HALF_WIDTH_MM, min(ROBOT_HALF_WIDTH_MM, ly))
        return math.hypot(lx - cx, ly - cy) < max(half_w, half_d) - tolerance_mm
    # Oriented robot rectangle vs axis-aligned object box: separating axis test.
    for ax, ay in ((ux, uy), (vx, vy), (1.0, 0.0), (0.0, 1.0)):
        robot_r = ROBOT_HALF_LENGTH_MM * abs(ux * ax + uy * ay) + ROBOT_HALF_WIDTH_MM * abs(vx * ax + vy * ay)
        obj_r = half_w * abs(ax) + half_d * abs(ay)
        if abs(dx * ax + dy * ay) >= robot_r + obj_r - tolerance_mm:
            return False
    return True


def reflex_obstacle_id(pose: Pose, objects: Iterable[SceneObject], *, max_gap_mm: float = 400.0) -> str | None:
    """Object most likely responsible for a front-obstacle reflex at `pose` (nearest in the ToF cone)."""
    yaw = math.radians(float(pose["heading_deg"]))
    ux, uy = math.cos(yaw), math.sin(yaw)
    ox = float(pose["x_mm"]) + ROBOT_HALF_LENGTH_MM * ux
    oy = float(pose["y_mm"]) + ROBOT_HALF_LENGTH_MM * uy
    best: tuple[float, str] | None = None
    for obj in objects:
        dx, dy = float(obj["x_mm"]) - ox, float(obj["y_mm"]) - oy
        forward = dx * ux + dy * uy
        radius = 0.5 * max(float(obj.get("width_mm", 100.0)), float(obj.get("depth_mm", 100.0)))
        if forward <= -radius:
            continue
        lateral = abs(-dx * uy + dy * ux)
        if lateral > radius + max(0.0, forward) * math.tan(math.radians(14.0)) + 30.0:
            continue
        gap = max(0.0, math.hypot(dx, dy) - radius)
        if gap <= max_gap_mm and (best is None or gap < best[0]):
            best = (gap, str(obj["id"]))
    return best[1] if best else None


# --------------------------------------------------------------------------- camera ground truth


@dataclass(frozen=True)
class CameraModel:
    """Head-camera geometry mirrored from the simulator so graders can decide what was in view."""

    hfov_deg: float = 65.0
    vfov_deg: float = 52.0
    range_mm: float = 2600.0
    image_w: int = 320
    image_h: int = 240
    mast_pivot_x_mm: float = 25.0
    mast_pivot_z_mm: float = 210.0
    cam_forward_mm: float = 42.0
    cam_up_mm: float = 76.0
    base_pitch_deg: float = -9.0

    @classmethod
    def from_sandbox(cls, sandbox: Any) -> CameraModel:
        d = cls()
        return cls(
            hfov_deg=float(getattr(sandbox, "HFOV_DEG", d.hfov_deg)),
            vfov_deg=float(getattr(sandbox, "VFOV_DEG", d.vfov_deg)),
            range_mm=float(getattr(sandbox, "DETECTION_RANGE_MM", d.range_mm)),
            image_w=int(getattr(sandbox, "cam_width", d.image_w)),
            image_h=int(getattr(sandbox, "cam_height", d.image_h)),
            mast_pivot_x_mm=float(getattr(sandbox, "MAST_PIVOT_X_MM", d.mast_pivot_x_mm)),
            mast_pivot_z_mm=float(getattr(sandbox, "MAST_PIVOT_Z_MM", d.mast_pivot_z_mm)),
            cam_forward_mm=float(getattr(sandbox, "CAM_OFFSET_FORWARD_MM", d.cam_forward_mm)),
            cam_up_mm=float(getattr(sandbox, "CAM_OFFSET_UP_MM", d.cam_up_mm)),
        )

    def to_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def visible_object_ids(pose: Pose, objects: Sequence[SceneObject], camera: CameraModel | None = None) -> set[str]:
    """Ids of objects inside the head-camera frustum and range, not occluded by a taller nearer object."""
    cam = camera or CameraModel()
    body_yaw = math.radians(float(pose["heading_deg"]))
    tilt = math.radians(float(pose.get("head_tilt_deg", 0.0)))
    local_fwd = cam.mast_pivot_x_mm + cam.cam_forward_mm * math.cos(tilt) - cam.cam_up_mm * math.sin(tilt)
    # Rounding mirrors the simulator's detector so both agree at frame edges.
    cam_x = round(float(pose["x_mm"]) + local_fwd * math.cos(body_yaw), 1)
    cam_y = round(float(pose["y_mm"]) + local_fwd * math.sin(body_yaw), 1)
    cam_z = round(cam.mast_pivot_z_mm + cam.cam_forward_mm * math.sin(tilt) + cam.cam_up_mm * math.cos(tilt), 1)
    yaw = math.radians(round(float(pose["heading_deg"]), 1))
    pitch = math.radians(round(float(pose.get("head_tilt_deg", 0.0)), 1) + cam.base_pitch_deg)
    fx = 0.5 * cam.image_w / math.tan(math.radians(0.5 * cam.hfov_deg))
    fy = 0.5 * cam.image_h / math.tan(math.radians(0.5 * cam.vfov_deg))
    cos_y, sin_y, cos_p, sin_p = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)

    visible: set[str] = set()
    for obj in objects:
        dx, dy = float(obj["x_mm"]) - cam_x, float(obj["y_mm"]) - cam_y
        height = float(obj.get("height_mm", 100.0))
        dz = float(obj.get("z_mm", 0.0)) + 0.5 * height - cam_z
        rx = dx * cos_y + dy * sin_y
        ry = -dx * sin_y + dy * cos_y
        fwd = rx * cos_p + dz * sin_p
        up = -rx * sin_p + dz * cos_p
        if fwd <= 40.0 or math.hypot(dx, dy) > cam.range_mm:
            continue
        u = 0.5 * cam.image_w + fx * (-ry / fwd)
        v = 0.5 * cam.image_h - fy * (up / fwd)
        proj_w = max(8, round(fx * float(obj.get("width_mm", 100.0)) / fwd))
        proj_h = max(8, round(fy * height / fwd))
        if (
            round(u + 0.5 * proj_w) < 0
            or round(u - 0.5 * proj_w) >= cam.image_w
            or round(v + 0.5 * proj_h) < 0
            or round(v - 0.5 * proj_h) >= cam.image_h
        ):
            continue
        if not _occluded(obj, objects, cam_x, cam_y):
            visible.add(str(obj["id"]))
    return visible


def _occluded(target: SceneObject, objects: Sequence[SceneObject], cam_x: float, cam_y: float) -> bool:
    tx, ty = float(target["x_mm"]) - cam_x, float(target["y_mm"]) - cam_y
    dist = math.hypot(tx, ty)
    if dist < 1e-6:
        return False
    ux, uy = tx / dist, ty / dist
    t_radius = 0.5 * max(float(target.get("width_mm", 100.0)), float(target.get("depth_mm", 100.0)))
    t_height = float(target.get("height_mm", 100.0))
    for other in objects:
        if other["id"] == target["id"]:
            continue
        ox, oy = float(other["x_mm"]) - cam_x, float(other["y_mm"]) - cam_y
        along = ox * ux + oy * uy
        if along <= 0.0 or along >= dist - t_radius:
            continue
        o_radius = 0.5 * max(float(other.get("width_mm", 100.0)), float(other.get("depth_mm", 100.0)))
        if abs(-ox * uy + oy * ux) < 0.9 * o_radius and float(other.get("height_mm", 100.0)) >= 0.7 * t_height:
            return True
    return False


def visible_families(poses: Iterable[Pose], objects: Sequence[SceneObject], camera: CameraModel | None = None) -> set[str]:
    by_id = {str(o["id"]): o for o in objects}
    seen: set[str] = set()
    for pose in poses:
        seen |= visible_object_ids(pose, objects, camera)
    return {f for f in (family_of(by_id[i]) for i in seen) if f}


# --------------------------------------------------------------------------- scene layout


@dataclass(frozen=True)
class SceneSetup:
    """Scenario-specific scene: base-position overrides, removed objects and per-seed jitter radius."""

    overrides: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    remove: tuple[str, ...] = ()
    jitter_mm: float = 100.0


def plan_layout(
    objects: Sequence[SceneObject],
    seed: int,
    setup: SceneSetup | None = None,
    *,
    room_half_mm: float = 1500.0,
    wall_margin_mm: float = 100.0,
    gap_mm: float = 60.0,
    start_clearance_mm: float = ROBOT_RADIUS_MM + 120.0,
) -> dict[str, tuple[float, float]]:
    """Deterministic per-seed placement: jittered, non-overlapping, inside the room, clear of the start."""
    setup = setup or SceneSetup()
    rng = random.Random(seed)
    placed: list[tuple[float, float, float]] = []
    plan: dict[str, tuple[float, float]] = {}
    for obj in sorted((o for o in objects if o["id"] not in setup.remove), key=lambda o: str(o["id"])):
        radius = footprint_radius_mm(obj)
        base_x, base_y = setup.overrides.get(str(obj["id"]), (float(obj["x_mm"]), float(obj["y_mm"])))
        lim = max(0.0, room_half_mm - wall_margin_mm - radius)
        choice: tuple[float, float] | None = None
        for attempt in range(400):
            spread = setup.jitter_mm + 8.0 * attempt
            angle = rng.uniform(0.0, 2.0 * math.pi)
            r = spread * math.sqrt(rng.random())
            x = max(-lim, min(lim, base_x + r * math.cos(angle)))
            y = max(-lim, min(lim, base_y + r * math.sin(angle)))
            if math.hypot(x, y) >= radius + start_clearance_mm and all(
                math.hypot(x - px, y - py) >= radius + pr + gap_mm for px, py, pr in placed
            ):
                choice = (x, y)
                break
        if choice is None:
            choice = (max(-lim, min(lim, base_x)), max(-lim, min(lim, base_y)))
        placed.append((choice[0], choice[1], radius))
        plan[str(obj["id"])] = (round(choice[0], 1), round(choice[1], 1))
    return plan


# --------------------------------------------------------------------------- records and grades


@dataclass
class TurnRecord:
    """One user utterance and what the robot did about it (all poses are ground truth)."""

    utterance: str
    say: str = ""
    pose_before: dict[str, float] = field(default_factory=dict)
    pose_after: dict[str, float] = field(default_factory=dict)
    trajectory: list[dict[str, float]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    reflexes: list[dict[str, Any]] = field(default_factory=list)
    responses: int = 0
    final_status: str = ""
    backend: str = ""
    model_calls: int = 0
    seconds: float = 0.0
    restarted_before: bool = False
    error: str | None = None

    def poses(self) -> list[dict[str, float]]:
        return [p for p in (self.pose_before, *self.trajectory, self.pose_after) if p]

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance": self.utterance,
            "say": self.say,
            "final_status": self.final_status,
            "backend": self.backend,
            "responses": self.responses,
            "model_calls": self.model_calls,
            "seconds": round(self.seconds, 2),
            "restarted_before": self.restarted_before,
            "error": self.error,
            "pose_before": self.pose_before,
            "pose_after": self.pose_after,
            "actions": self.actions,
            "reflexes": self.reflexes,
            "trajectory": [
                [p["x_mm"], p["y_mm"], p["heading_deg"], p.get("head_tilt_deg", 0.0)] for p in self.trajectory
            ],
        }


@dataclass
class ScenarioRun:
    """Everything a grader may look at."""

    scenario: str
    seed: int
    turns: list[TurnRecord]
    objects_start: list[dict[str, Any]]
    objects_end: list[dict[str, Any]]
    camera: CameraModel = field(default_factory=CameraModel)

    @property
    def objects(self) -> list[dict[str, Any]]:
        return self.objects_end or self.objects_start

    def object(self, obj_id: str) -> dict[str, Any] | None:
        return next((o for o in self.objects if o["id"] == obj_id), None)

    def all_poses(self) -> list[dict[str, float]]:
        return [p for turn in self.turns for p in turn.poses()]


@dataclass(frozen=True)
class Grade:
    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


def _verdict(checks: list[tuple[bool, str]], details: dict[str, Any]) -> Grade:
    failed = [msg for ok, msg in checks if not ok]
    return Grade(not failed, "; ".join(failed or [msg for _, msg in checks]), details)


def _missing(name: str) -> Grade:
    return Grade(False, f"scene has no {name}", {})


# --------------------------------------------------------------------------- graders


def grade_greet(run: ScenarioRun) -> Grade:
    turn = run.turns[-1]
    moved = distance_mm(turn.pose_before, turn.pose_after)
    turned = abs(heading_change_deg(turn.pose_before, turn.pose_after))
    details = {"moved_mm": round(moved, 1), "turned_deg": round(turned, 1)}
    return _verdict(
        [
            (bool(turn.say.strip()), "replied" if turn.say.strip() else "empty reply"),
            (moved < 30.0, f"moved {moved:.0f} mm"),
            (turned < 10.0, f"turned {turned:.0f} deg"),
        ],
        details,
    )


def grade_describe(run: ScenarioRun) -> Grade:
    turn = run.turns[-1]
    visible = visible_families(turn.poses(), run.objects, run.camera)
    present = families_present(run.objects)
    mentioned = mentioned_families(turn.say)
    hits = mentioned & visible
    required = max(1, min(2, len(visible)))
    details = {
        "visible": sorted(visible),
        "mentioned": sorted(mentioned),
        "correct": sorted(hits),
        "unseen_mentions": sorted((mentioned & present) - visible),
        "hallucinated": sorted(mentioned - present),
    }
    return _verdict(
        [
            (bool(visible), f"{len(visible)} families in view"),
            (len(hits) >= required, f"named {len(hits)}/{required} visible families ({','.join(sorted(hits))})"),
        ],
        details,
    )


def grade_look_around_report(run: ScenarioRun) -> Grade:
    turn = run.turns[-1]
    present = families_present(run.objects)
    mentioned = mentioned_families(turn.say)
    hits = mentioned & present
    details = {
        "present": sorted(present),
        "mentioned": sorted(mentioned),
        "correct": sorted(hits),
        "seen_during_turn": sorted(visible_families(turn.poses(), run.objects, run.camera)),
        "hallucinated": sorted(mentioned - present),
    }
    return _verdict([(len(hits) >= 4, f"named {len(hits)}/4 families ({','.join(sorted(hits))})")], details)


def _turn_grader(target_deg: float, tolerance_deg: float) -> Callable[[ScenarioRun], Grade]:
    def grade(run: ScenarioRun) -> Grade:
        turn = run.turns[-1]
        change = heading_change_deg(turn.pose_before, turn.pose_after)
        error = abs(wrap_deg(change - target_deg))
        details = {"heading_change_deg": round(change, 1), "error_deg": round(error, 1)}
        return _verdict([(error <= tolerance_deg, f"turned {change:+.0f} deg (want {target_deg:+.0f})")], details)

    return grade


def grade_move_forward(run: ScenarioRun) -> Grade:
    turn = run.turns[-1]
    forward, lateral = forward_lateral_mm(turn.pose_before, turn.pose_after)
    turned = abs(heading_change_deg(turn.pose_before, turn.pose_after))
    details = {"forward_mm": round(forward, 1), "lateral_mm": round(lateral, 1), "turned_deg": round(turned, 1)}
    return _verdict(
        [
            (80.0 <= forward <= 700.0, f"forward {forward:.0f} mm"),
            (turned < 15.0, f"turned {turned:.0f} deg"),
        ],
        details,
    )


def grade_go_to_ball(run: ScenarioRun) -> Grade:
    ball = run.object("red_ball")
    if ball is None:
        return _missing("red_ball")
    end = run.turns[-1].pose_after
    dist = distance_mm(end, ball)
    bearing = relative_bearing_deg(end, float(ball["x_mm"]), float(ball["y_mm"]))
    details = {"distance_mm": round(dist, 1), "bearing_deg": round(bearing, 1)}
    return _verdict(
        [(dist <= 450.0, f"{dist:.0f} mm from ball"), (abs(bearing) <= 40.0, f"ball at {bearing:+.0f} deg")],
        details,
    )


def _collisions(run: ScenarioRun, obj: SceneObject) -> dict[str, int]:
    contacts = sum(1 for pose in run.all_poses() if footprint_overlaps(pose, obj))
    reflexes = sum(
        1
        for turn in run.turns
        for reflex in turn.reflexes
        if reflex.get("reflex") == "obstacle_stop"
        and reflex.get("pose")
        and reflex_obstacle_id(reflex["pose"], run.objects) == obj["id"]
    )
    return {"contacts": contacts, "reflexes": reflexes}


def grade_behind_box(run: ScenarioRun) -> Grade:
    box = run.object("cardboard_box")
    if box is None:
        return _missing("cardboard_box")
    turn = run.turns[-1]
    start, end = turn.pose_before, turn.pose_after
    bx, by = float(box["x_mm"]), float(box["y_mm"])
    approach = (bx - float(start["x_mm"]), by - float(start["y_mm"]))
    beyond = (float(end["x_mm"]) - bx, float(end["y_mm"]) - by)
    angle = angle_between_deg(approach, beyond)
    dist = math.hypot(*beyond)
    hits = _collisions(run, box)
    details = {"angle_deg": None if angle is None else round(angle, 1), "distance_mm": round(dist, 1), **hits}
    side_msg = "ended on the box centre" if angle is None else f"{angle:.0f} deg off the far-side axis"
    return _verdict(
        [
            (angle is not None and angle < 55.0, side_msg),
            (dist <= 800.0, f"{dist:.0f} mm from box"),
            (
                not hits["contacts"] and not hits["reflexes"],
                f"{hits['contacts']} contact poses, {hits['reflexes']} reflex stops at box",
            ),
        ],
        details,
    )


def grade_between(run: ScenarioRun) -> Grade:
    ball, cube = run.object("red_ball"), run.object("blue_cube")
    if ball is None or cube is None:
        return _missing("red_ball/blue_cube")
    mid = {"x_mm": 0.5 * (ball["x_mm"] + cube["x_mm"]), "y_mm": 0.5 * (ball["y_mm"] + cube["y_mm"])}
    dist = distance_mm(run.turns[-1].pose_after, mid)
    details = {"midpoint": [round(mid["x_mm"], 1), round(mid["y_mm"], 1)], "distance_mm": round(dist, 1)}
    return _verdict([(dist <= 300.0, f"{dist:.0f} mm from midpoint")], details)


def grade_find_hidden(run: ScenarioRun) -> Grade:
    duck = run.object("yellow_duck")
    if duck is None:
        return _missing("yellow_duck")
    turn = run.turns[-1]
    bearing = relative_bearing_deg(turn.pose_after, float(duck["x_mm"]), float(duck["y_mm"]))
    seen = any(duck["id"] in visible_object_ids(p, run.objects, run.camera) for p in turn.poses())
    named = "duck" in mentioned_families(turn.say)
    located = has_location_cue(turn.say)
    facing = abs(bearing) <= 40.0
    reported = named and located and seen
    details = {"bearing_deg": round(bearing, 1), "seen": seen, "named": named, "location_cue": located}
    if facing:
        return Grade(True, f"facing duck ({bearing:+.0f} deg)", details)
    if reported:
        return Grade(True, "saw the duck and said where it is", details)
    why = f"duck at {bearing:+.0f} deg"
    if not seen:
        why += ", never in view"
    elif not (named and located):
        why += ", reply does not locate it"
    return Grade(False, why, details)


def grade_go_to_dock(run: ScenarioRun) -> Grade:
    dock = run.object("charging_dock")
    if dock is None:
        return _missing("charging_dock")
    dist = distance_mm(run.turns[-1].pose_after, dock)
    return _verdict([(dist <= 550.0, f"{dist:.0f} mm from dock")], {"distance_mm": round(dist, 1)})


def grade_come_back(run: ScenarioRun) -> Grade:
    start, end = run.turns[0].pose_before, run.turns[-1].pose_after
    dist = distance_mm(start, end)
    excursion = max((distance_mm(start, p) for p in run.all_poses()), default=0.0)
    details = {"distance_mm": round(dist, 1), "max_excursion_mm": round(excursion, 1)}
    return _verdict(
        [(excursion >= 250.0, f"left start by {excursion:.0f} mm"), (dist <= 300.0, f"{dist:.0f} mm from start")],
        details,
    )


def _recall_grader(pattern: str, label: str) -> Callable[[ScenarioRun], Grade]:
    regex = re.compile(pattern, re.IGNORECASE)

    def grade(run: ScenarioRun) -> Grade:
        turn = run.turns[-1]
        restarted = any(t.restarted_before for t in run.turns[1:])
        recalled = bool(regex.search(turn.say))
        details = {"restarted": restarted, "recalled": recalled}
        return _verdict(
            [
                (restarted, "restarted" if restarted else "no restart happened"),
                (recalled, f"recalled {label}" if recalled else f"did not recall {label}"),
            ],
            details,
        )

    return grade


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Scenario:
    """A scene, a sequence of user utterances, and a ground-truth grader."""

    name: str
    description: str
    utterances: tuple[str, ...]
    grader: Callable[[ScenarioRun], Grade]
    setup: SceneSetup = field(default_factory=SceneSetup)
    restart_before: frozenset[int] = frozenset()

    def grade(self, run: ScenarioRun) -> Grade:
        return self.grader(run)


SCENARIO_LIST: tuple[Scenario, ...] = (
    Scenario("greet", "Reply to a greeting without wandering off", ("hi!",), grade_greet),
    Scenario("describe", "Name at least two object families really in view", ("what do you see?",), grade_describe),
    Scenario("turn_around", "Heading changes by 180 +/- 30 deg", ("turn around",), _turn_grader(180.0, 30.0)),
    Scenario("look_left", "Heading changes by +90 +/- 35 deg", ("look to your left",), _turn_grader(90.0, 35.0)),
    Scenario("move_forward", "Drive 80..700 mm forward without turning", ("move forward a little",),
             grade_move_forward),
    Scenario("go_to_ball", "End within 450 mm of the ball, facing it", ("go to the red ball",), grade_go_to_ball),
    Scenario(
        "behind_box",
        "Get to the far side of the box without hitting it",
        ("what do you see?", "go behind that box"),
        grade_behind_box,
        SceneSetup(overrides={"cardboard_box": (900.0, 400.0)}),
    ),
    Scenario(
        "between",
        "End within 300 mm of the ball/cube midpoint",
        ("go between the red ball and the blue cube",),
        grade_between,
        SceneSetup(overrides={"green_mug": (1050.0, -350.0)}),
    ),
    Scenario(
        "find_hidden",
        "Find a duck placed behind the robot",
        ("find the yellow duck",),
        grade_find_hidden,
        SceneSetup(overrides={"yellow_duck": (-800.0, 350.0)}),
    ),
    Scenario("go_to_dock", "End within 550 mm of the charging dock", ("go to your charging dock",),
             grade_go_to_dock),
    Scenario(
        "come_back",
        "Go to the ball, then return within 300 mm of the start",
        ("go to the red ball", "now go back to where you started"),
        grade_come_back,
    ),
    Scenario(
        "look_around_report",
        "Look around and name at least four object families",
        ("look around and tell me everything in the room",),
        grade_look_around_report,
    ),
    Scenario(
        "memory_name",
        "Recall the user's name after a restart on the same database",
        ("my name is Deniz", "what's my name?"),
        _recall_grader(r"\bdeniz\b", "name"),
        restart_before=frozenset({1}),
    ),
    Scenario(
        "memory_pref",
        "Recall a stated preference after a restart (no green objects in the room)",
        ("remember that my favorite color is green", "what's my favorite color?"),
        _recall_grader(r"\bgreen\b", "color"),
        SceneSetup(remove=("green_mug",)),
        restart_before=frozenset({1}),
    ),
)
SCENARIOS: dict[str, Scenario] = {s.name: s for s in SCENARIO_LIST}


def select_scenarios(spec: str) -> list[Scenario]:
    """Parse `all` or a comma-separated list of scenario names."""
    names = [n.strip() for n in (spec or "all").split(",") if n.strip()]
    if not names or names == ["all"]:
        return list(SCENARIO_LIST)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenario(s): {', '.join(unknown)}; choose from {', '.join(SCENARIOS)}")
    return [SCENARIOS[n] for n in dict.fromkeys(names)]
