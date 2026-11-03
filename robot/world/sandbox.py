"""The simulated room: MuJoCo renders of the robot's cameras, simulated range sensors and a person.

The robot model and room are config/companion_r1.xml; the cameras are built from config/body.yaml
(`sensors.cameras`), so the simulator's lenses are exactly the ones the perception code assumes.
Ground truth here (object positions, `detect_objects`, `render_labels`) feeds training data, the dashboard's
"sim truth" overlay and the evaluation graders; the robot itself only gets camera pixels and range readings.
"""

from __future__ import annotations

import atexit
import base64
import concurrent.futures
import math
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robot.body.hal import BodyPose
from robot.core.config import PROJECT_ROOT
from robot.core.events import Event
from robot.perception.camera import encode_png
from robot.perception.geometry import CameraModel, default_rig
from robot.world.spatial_map import direction_words

_MJ_SHARED_STATE: dict[Any, dict[str, Any]] = {}


class _GLThread:
    """Every MuJoCo render runs on this one thread.

    An OpenGL context belongs to the thread that made it current; with cameras captured on the perception
    thread, "what do you see?" on a worker thread and the dashboard on the main loop, renders from several
    threads made WGL refuse the context ("resource is in use") and crashed at exit."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._ident: int | None = None
        self._start_lock = threading.Lock()

    def run(self, fn: Any, timeout: float | None = None) -> Any:
        if threading.get_ident() == self._ident:
            return fn()
        with self._start_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._serve, name="mujoco-gl", daemon=True)
                self._thread.start()
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((fn, future))
        return future.result(timeout)

    def _serve(self) -> None:
        self._ident = threading.get_ident()
        while True:
            fn, future = self._queue.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn())
            except BaseException as exc:  # handed to the caller
                future.set_exception(exc)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


_GL = _GLThread()


def _close_renderers() -> None:
    def close_all() -> None:
        for st in list(_MJ_SHARED_STATE.values()):
            for renderer in [*st["cameras"].values(), *st["seg"].values(), st["over_renderer"]]:
                try:
                    renderer.close()
                except Exception:
                    pass
        _MJ_SHARED_STATE.clear()

    if _GL.alive:
        try:
            _GL.run(close_all, timeout=2.0)  # contexts are closed by the thread that owns them
        except Exception:
            pass
    _MJ_SHARED_STATE.clear()


atexit.register(_close_renderers)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        try:
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        except ValueError:
            pass
    return (56, 189, 248)


def camera_xml(cam: CameraModel) -> str:
    """A MuJoCo <camera> for one camera of the rig (its body: the head for head cameras, else the base)."""
    yaw, pitch = math.radians(cam.yaw_deg), math.radians(cam.pitch_deg)
    right = (math.sin(yaw), -math.cos(yaw), 0.0)
    up = (-math.sin(pitch) * math.cos(yaw), -math.sin(pitch) * math.sin(yaw), math.cos(pitch))
    axes = " ".join(f"{v:.5f}" for v in (*right, *up))
    return (f'<camera name="cam_{cam.name}" pos="{cam.x_mm / 1000:.4f} {cam.y_mm / 1000:.4f} {cam.z_mm / 1000:.4f}" '
            f'xyaxes="{axes}" fovy="{cam.vfov_deg:.3f}"/>')


def scene_xml(xml_path: Path, rig: tuple[CameraModel, ...]) -> str:
    text = xml_path.read_text(encoding="utf-8")
    for mount, on_head in (("head", True), ("body", False)):
        marker = next(line for line in text.splitlines() if f"@cameras:{mount}" in line)
        indent = marker[:len(marker) - len(marker.lstrip())]
        cams = "\n".join(indent + camera_xml(c) for c in rig if c.on_head == on_head)
        text = text.replace(marker, cams or marker)
    return text


def _mujoco_state(xml_path: Path, rig: tuple[CameraModel, ...], over_w: int, over_h: int) -> dict[str, Any]:
    """The MuJoCo model, data and renderers for this scene and rig (built once, on the GL thread)."""
    key = (str(xml_path.resolve()), rig, over_w, over_h)
    state = _MJ_SHARED_STATE.get(key)
    if state is not None:
        return state
    mesh_dir = xml_path.parent / "meshes"
    assets = {p.name: p.read_bytes() for p in mesh_dir.glob("*.stl")} if mesh_dir.exists() else {}
    model = mujoco.MjModel.from_xml_string(scene_xml(xml_path, rig), assets=assets)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def joint(name: str) -> int:
        return int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])

    def geom(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    mocap: dict[str, int] = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name.startswith("obj_") and model.body_mocapid[bid] >= 0:
            mocap[name[4:]] = int(model.body_mocapid[bid])
    cameras = {c.name: mujoco.Renderer(model, height=c.height, width=c.width) for c in rig}
    for renderer in cameras.values():  # the robot's cameras skip shadows and floor reflections: 3-5x faster
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    state = {
        "model": model, "data": data,
        "cameras": cameras,
        "seg": {},
        "over_renderer": mujoco.Renderer(model, height=over_h, width=over_w),
        "qpos": {"base_x": joint("base_x"), "base_y": joint("base_y"), "base_yaw": joint("base_yaw"),
                 "head_tilt": joint("head_tilt"),
                 "wheels": [joint(j) for j in ("fl_spin", "fr_spin", "rl_spin", "rr_spin")]},
        "geoms": {"eye_l": geom("g_eye_left"), "eye_r": geom("g_eye_right"), "mouth": geom("g_mouth"),
                  "cliff_zone": geom("cliff_zone_geom")},
        "mocap": mocap,
        "spawn_geoms": {f"spawn_{i}": geom(f"geom_spawn_{i}") for i in range(1, 5)},
    }
    _MJ_SHARED_STATE[key] = state
    return state


@dataclass
class SandboxObject:
    """A physical object in the simulated room."""

    id: str
    label: str
    shape: str  # "sphere" | "box" | "cylinder" | "person"
    x_mm: float
    y_mm: float
    z_mm: float = 0.0
    width_mm: float = 100.0
    depth_mm: float = 100.0
    height_mm: float = 100.0
    color: str = "#38bdf8"
    movable: bool = True
    yaw_deg: float = 0.0  # which way it faces (people)

    @property
    def radius_mm(self) -> float:
        return 0.5 * max(self.width_mm, self.depth_mm)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "shape": self.shape, "x_mm": round(self.x_mm, 1),
                "y_mm": round(self.y_mm, 1), "z_mm": round(self.z_mm, 1), "width_mm": round(self.width_mm, 1),
                "depth_mm": round(self.depth_mm, 1), "height_mm": round(self.height_mm, 1), "color": self.color,
                "movable": self.movable, "yaw_deg": round(self.yaw_deg, 1)}


PERSON_HEAD_Z_MM = 1580.0
PERSON_LEG_OFFSET_MM, PERSON_LEG_RADIUS_MM = 85.0, 58.0


def default_objects() -> dict[str, SandboxObject]:
    """The 3 m x 3 m playroom: toys, a box, the charging dock, and the person the robot lives with."""
    objs = [
        SandboxObject("red_ball", "Red Ball", "sphere", 680.0, 160.0, width_mm=96.0, depth_mm=96.0,
                      height_mm=96.0, color="#d42020"),
        SandboxObject("blue_cube", "Blue Cube", "box", 620.0, -210.0, width_mm=110.0, depth_mm=110.0,
                      height_mm=110.0, color="#3b82f6"),
        SandboxObject("yellow_duck", "Yellow Toy Duck", "sphere", 470.0, 340.0, width_mm=90.0, depth_mm=90.0,
                      height_mm=96.0, color="#eab308"),
        SandboxObject("green_mug", "Green Mug", "cylinder", 840.0, -30.0, width_mm=86.0, depth_mm=86.0,
                      height_mm=116.0, color="#10b981"),
        SandboxObject("cardboard_box", "Cardboard Box", "box", 950.0, 420.0, width_mm=220.0, depth_mm=220.0,
                      height_mm=190.0, color="#d97706"),
        SandboxObject("charging_dock", "Charging Dock", "box", -680.0, 0.0, width_mm=180.0, depth_mm=220.0,
                      height_mm=130.0, color="#8b5cf6"),
        SandboxObject("person", "Person", "person", -1050.0, -950.0, width_mm=500.0, depth_mm=300.0,
                      height_mm=1700.0, color="#cc5242", yaw_deg=45.0),
    ]
    return {o.id: o for o in objs}


@dataclass
class GroundedDetection:
    """An object's true box in one camera image (simulator ground truth)."""

    id: str
    label: str
    color: str
    bbox: tuple[float, float, float, float]  # normalized x1, y1, x2, y2
    distance_mm: float
    bearing_deg: float  # relative to the robot's heading, + = left

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "color": self.color, "bbox": [round(v, 3) for v in self.bbox],
                "distance_mm": round(self.distance_mm), "direction": direction_words(self.bearing_deg)}


@dataclass
class CameraFrame:
    """One rendered camera image plus what is truly in it (for the dashboard)."""

    camera: str
    width: int
    height: int
    png_bytes: bytes
    visible_objects: list[GroundedDetection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        seen = ", ".join(f"{d.label.lower()} {d.distance_mm / 1000:.1f} m" for d in self.visible_objects[:4])
        return {"camera": self.camera, "width": self.width, "height": self.height,
                "data_url": "data:image/png;base64," + base64.b64encode(self.png_bytes).decode("ascii"),
                "visible_objects": [d.to_dict() for d in self.visible_objects],
                "summary": f"Sim truth, {self.camera} camera: " + (seen or "nothing in view")}


def _extent_points(obj: SandboxObject) -> list[tuple[float, float, float]]:
    """Surface points whose projections bound the image of the object's shape."""
    z0, z1 = obj.z_mm, obj.z_mm + obj.height_mm
    hw, hd = 0.5 * obj.width_mm, 0.5 * obj.depth_mm
    if obj.shape in ("box", "person"):
        return [(obj.x_mm + sx * hw, obj.y_mm + sy * hd, z) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for z in (z0, z1)]
    ring = [(math.cos(math.pi * k / 12.0), math.sin(math.pi * k / 12.0)) for k in range(24)]
    if obj.shape == "cylinder":
        return [(obj.x_mm + c * hw, obj.y_mm + s * hd, z) for c, s in ring for z in (z0, z1)]
    zc, hz = 0.5 * (z0 + z1), 0.5 * obj.height_mm
    points = [(obj.x_mm, obj.y_mm, z0), (obj.x_mm, obj.y_mm, z1)]
    for lat in (-67.5, -45.0, -22.5, 0.0, 22.5, 45.0, 67.5):
        cl, sl = math.cos(math.radians(lat)), math.sin(math.radians(lat))
        points.extend((obj.x_mm + c * hw * cl, obj.y_mm + s * hd * cl, zc + hz * sl) for c, s in ring)
    return points


def _occluded(target: SandboxObject, objects: list[SandboxObject], cam_x: float, cam_y: float) -> bool:
    """True if a nearer, sufficiently tall object blocks the camera's line of sight to `target`."""
    tx, ty = target.x_mm - cam_x, target.y_mm - cam_y
    t_dist = math.hypot(tx, ty)
    if t_dist < 1e-6:
        return False
    ux, uy = tx / t_dist, ty / t_dist
    for other in objects:
        if other.id == target.id:
            continue
        ox, oy = other.x_mm - cam_x, other.y_mm - cam_y
        along = ox * ux + oy * uy
        if 0.0 < along < t_dist - target.radius_mm and abs(-ox * uy + oy * ux) < other.radius_mm * 0.9                 and other.height_mm >= 0.7 * target.height_mm:
            return True
    return False


def visible_in(objects: list[SandboxObject], pose: BodyPose, cam: CameraModel) -> list[GroundedDetection]:
    """What one camera truly sees: projected 3D extents inside the image, within its range, not hidden behind
    a taller nearer object. Simulator ground truth (the dashboard, fake detectors in tests, the graders)."""
    cx, cy, _, _, _ = cam.extrinsics(pose)
    out = []
    for obj in objects:
        if math.hypot(obj.x_mm - cx, obj.y_mm - cy) > cam.max_range_mm or _occluded(obj, objects, cx, cy):
            continue
        uvs = [cam.project(pose, *pt) for pt in _extent_points(obj)]
        if any(uv is None for uv in uvs):
            continue
        us, vs = [uv[0] for uv in uvs], [uv[1] for uv in uvs]
        box = (max(0.0, min(us)), max(0.0, min(vs)), min(1.0, max(us)), min(1.0, max(vs)))
        if box[2] - box[0] < 0.01 or box[3] - box[1] < 0.01:
            continue
        dist = math.hypot(obj.x_mm - pose.x_mm, obj.y_mm - pose.y_mm)
        bearing = math.degrees(math.atan2(obj.y_mm - pose.y_mm, obj.x_mm - pose.x_mm)) - pose.heading_deg
        out.append(GroundedDetection(obj.id, obj.label, obj.color, box, dist, (bearing + 180.0) % 360.0 - 180.0))
    return sorted(out, key=lambda d: d.distance_mm)


class SimulationSandbox:
    """The room, the robot's simulated cameras and sensors, and an interactive overview camera."""

    FRONT_TOF_OFFSET_X_MM = 141.0
    OVERVIEW_FOVY_DEG = 48.0
    ROBOT_FOOTPRINT_RADIUS_MM = 140.0
    DOCK_CHARGE_RADIUS_MM = 230.0

    def __init__(self, bus: Any = None, *, rig: tuple[CameraModel, ...] | None = None,
                 room_half_extent_mm: float = 1500.0, overview_width: int = 520, overview_height: int = 340,
                 xml_path: Path | None = None) -> None:
        self.bus = bus
        self.rig = tuple(rig or default_rig())
        self.cameras = {c.name: c for c in self.rig}
        self.room_half_extent_mm = room_half_extent_mm
        self.overview_width = overview_width
        self.overview_height = overview_height
        self.xml_path = xml_path or (PROJECT_ROOT / "config" / "companion_r1.xml")
        self._objects: dict[str, SandboxObject] = {}
        self.cliff_zone_enabled = False
        self.cliff_zone_rect: tuple[float, float, float, float] = (950.0, 1500.0, -600.0, 600.0)
        self.cam_azimuth_deg, self.cam_elevation_deg = -135.0, -24.0  # the dashboard's orbit camera
        self.cam_distance_m, self.cam_lookat_m = 1.35, [0.25, 0.0, 0.12]
        self._face_state: dict[str, Any] = {"color_rgb": (0, 228, 255), "left_openness": 1.0,
                                            "right_openness": 1.0, "mouth_open": 0.08}
        self._last_pose = BodyPose()
        self._wheel_travel_mm = 0.0
        self._last_frame: CameraFrame | None = None
        self._last_overview_url: str | None = None
        self._viewer_proc: subprocess.Popen[Any] | None = None
        self._sub_ids: list[str] = []
        self._sim_body: Any | None = None
        self.reset_default_scene()

    @property
    def front(self) -> CameraModel:
        return self.cameras.get("front", self.rig[0])

    def bind(self, sim_body: Any | None = None) -> None:
        """Follow `body.pose` to keep the simulated range sensors current."""
        self._sim_body = sim_body
        if sim_body is not None and hasattr(sim_body, "attach_sensor_model"):
            sim_body.attach_sensor_model(self)  # ToF/cliff sampled every control tick, no bus latency
        if self.bus is not None:
            self._sub_ids.append(self.bus.subscribe("body.pose", self._on_body_pose, name="sandbox:pose"))
            self._sync_sensors(self._last_pose)

    def unbind(self) -> None:
        if self.bus is not None:
            for sid in self._sub_ids:
                self.bus.unsubscribe(sid)
        self._sub_ids.clear()
        if self._viewer_proc is not None and self._viewer_proc.poll() is None:
            try:
                self._viewer_proc.terminate()
            except Exception:
                pass
            self._viewer_proc = None

    # ---- scene ----------------------------------------------------------------------------------
    def reset_default_scene(self) -> None:
        self._objects = default_objects()
        self.cliff_zone_enabled = False
        self._changed()

    def _changed(self) -> None:
        self._last_frame = None
        self._last_overview_url = None
        if self._sim_body is not None:
            self._sync_sensors(self._sim_body.pose())

    def list_objects(self) -> list[dict[str, Any]]:
        res = []
        for obj in self._objects.values():
            d = obj.to_dict()
            u, v = self.project_world_to_overview_uv(obj.x_mm, obj.y_mm, obj.z_mm + 0.5 * min(obj.height_mm, 400.0))
            d["overview_uv"] = [round(u, 3), round(v, 3)]
            res.append(d)
        return res

    def get_object(self, obj_id: str) -> SandboxObject | None:
        return self._objects.get(obj_id)

    def move_object(self, obj_id: str, x_mm: float, y_mm: float, yaw_deg: float | None = None
                    ) -> SandboxObject | None:
        obj = self._objects.get(obj_id)
        if obj is None:
            return None
        lim = self.room_half_extent_mm - obj.radius_mm
        obj.x_mm = max(-lim, min(lim, float(x_mm)))
        obj.y_mm = max(-lim, min(lim, float(y_mm)))
        if yaw_deg is not None:
            obj.yaw_deg = float(yaw_deg)
        self._changed()
        return obj

    def move_object_by_overview_ray(self, obj_id: str, u_norm: float, v_norm: float) -> SandboxObject | None:
        """Move an object to where a drag on the overview image meets the floor."""
        obj = self._objects.get(obj_id)
        if obj is None:
            return None
        hit = self.unproject_overview_uv_to_world(u_norm, v_norm, plane_z_mm=0.5 * min(obj.height_mm, 400.0))
        return obj if hit is None else self.move_object(obj_id, hit[0], hit[1])

    def add_object(self, *, obj_id: str | None = None, label: str = "Orange Cone", shape: str = "cylinder",
                   x_mm: float = 550.0, y_mm: float = 0.0, size_mm: float = 100.0, color: str = "#f97316"
                   ) -> SandboxObject:
        clean_id = (obj_id or label.lower().replace(" ", "_")).strip()[:40]
        if clean_id in self._objects:
            idx = 2
            while f"{clean_id}_{idx}" in self._objects:
                idx += 1
            clean_id = f"{clean_id}_{idx}"
        sz = max(40.0, min(400.0, float(size_mm)))
        lim = self.room_half_extent_mm - sz * 0.5
        obj = SandboxObject(clean_id, label[:60], shape if shape in ("sphere", "box", "cylinder") else "box",
                            max(-lim, min(lim, float(x_mm))), max(-lim, min(lim, float(y_mm))),
                            width_mm=sz, depth_mm=sz, height_mm=sz, color=color)
        self._objects[clean_id] = obj
        self._changed()
        return obj

    def remove_object(self, obj_id: str) -> bool:
        removed = self._objects.pop(obj_id, None) is not None
        if removed:
            self._changed()
        return removed

    def set_cliff_zone(self, enabled: bool) -> None:
        self.cliff_zone_enabled = bool(enabled)
        self._changed()

    def person_head(self) -> tuple[float, float, float] | None:
        """Where the person's voice comes from (x, y, z mm), or None when nobody is in the room."""
        person = next((o for o in self._objects.values() if o.shape == "person"), None)
        return None if person is None else (person.x_mm, person.y_mm, PERSON_HEAD_Z_MM)

    # ---- simulated range sensors ------------------------------------------------------------------
    def compute_front_tof_mm(self, pose: BodyPose) -> float:
        """Forward ToF distance (mm) from the front bumper to the nearest object or wall in a 14-degree cone."""
        yaw = math.radians(pose.heading_deg)
        dir_x, dir_y = math.cos(yaw), math.sin(yaw)
        ox = pose.x_mm + self.FRONT_TOF_OFFSET_X_MM * dir_x
        oy = pose.y_mm + self.FRONT_TOF_OFFSET_X_MM * dir_y
        min_dist = 2500.0
        half = self.room_half_extent_mm
        if abs(dir_x) > 1e-6 and (t := ((half if dir_x > 0 else -half) - ox) / dir_x) > 0:
            min_dist = min(min_dist, t)
        if abs(dir_y) > 1e-6 and (t := ((half if dir_y > 0 else -half) - oy) / dir_y) > 0:
            min_dist = min(min_dist, t)
        cone = math.tan(math.radians(14.0))
        for obj in self._objects.values():
            if obj.id == "charging_dock":
                continue
            dx, dy = obj.x_mm - ox, obj.y_mm - oy
            forward = dx * dir_x + dy * dir_y
            if forward > 0 and abs(-dx * dir_y + dy * dir_x) <= obj.radius_mm + forward * cone:
                min_dist = min(min_dist, max(0.0, math.hypot(dx, dy) - obj.radius_mm))
        return round(max(10.0, min(2500.0, min_dist)), 1)

    def lidar_circles(self, min_height_mm: float) -> list[tuple[float, float, float]]:
        """Cross-sections (x, y, radius) of everything that reaches `min_height_mm`: a person is two legs."""
        out = []
        for obj in self._objects.values():
            if obj.z_mm + obj.height_mm < min_height_mm:
                continue
            if obj.shape == "person" and min_height_mm < 860.0:
                yaw = math.radians(obj.yaw_deg)
                for side in (1.0, -1.0):
                    out.append((obj.x_mm - side * PERSON_LEG_OFFSET_MM * math.sin(yaw),
                                obj.y_mm + side * PERSON_LEG_OFFSET_MM * math.cos(yaw), PERSON_LEG_RADIUS_MM))
            else:
                out.append((obj.x_mm, obj.y_mm, obj.radius_mm))
        return out

    def check_collision(self, pose: BodyPose) -> bool:
        """Physical contact of the robot's footprint with a wall or an object (bumper switch)."""
        r = self.ROBOT_FOOTPRINT_RADIUS_MM
        half = self.room_half_extent_mm - 20.0
        if abs(pose.x_mm) + r > half or abs(pose.y_mm) + r > half:
            return True
        for obj in self._objects.values():
            if obj.id == "charging_dock":
                continue  # the dock is a low ramp the robot drives onto
            if math.hypot(obj.x_mm - pose.x_mm, obj.y_mm - pose.y_mm) < r + 0.5 * min(obj.width_mm, obj.depth_mm):
                return True
        return False

    def is_charging(self, pose: BodyPose) -> bool:
        dock = self._objects.get("charging_dock")
        return dock is not None and math.hypot(dock.x_mm - pose.x_mm, dock.y_mm - pose.y_mm) <= self.DOCK_CHARGE_RADIUS_MM

    def check_cliff_triggered(self, pose: BodyPose) -> bool:
        """Is any of the four corner cliff sensors over the cliff zone?"""
        if not self.cliff_zone_enabled:
            return False
        x1, x2, y1, y2 = self.cliff_zone_rect
        yaw = math.radians(pose.heading_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        for lx, ly in ((120.0, 92.0), (120.0, -92.0), (-120.0, 92.0), (-120.0, -92.0)):
            wx, wy = pose.x_mm + lx * c - ly * s, pose.y_mm + lx * s + ly * c
            if x1 <= wx <= x2 and y1 <= wy <= y2:
                return True
        return False

    # ---- ground truth for one camera --------------------------------------------------------------
    def detect_objects(self, pose: BodyPose | None = None, camera: CameraModel | None = None
                       ) -> list[GroundedDetection]:
        """Ground-truth boxes of the objects one camera sees (no render)."""
        return visible_in(list(self._objects.values()), pose or self._last_pose, camera or self.front)

    # ---- rendering -------------------------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        return _mujoco_state(self.xml_path, self.rig, self.overview_width, self.overview_height)

    def _sync_mujoco(self, pose: BodyPose) -> dict[str, Any]:
        """Pose the robot, face and objects in MuJoCo (on the GL thread)."""
        st = self._state()
        model, data, q, g = st["model"], st["data"], st["qpos"], st["geoms"]
        data.qpos[q["base_x"]] = pose.x_mm / 1000.0
        data.qpos[q["base_y"]] = pose.y_mm / 1000.0
        data.qpos[q["base_yaw"]] = math.radians(pose.heading_deg)  # hinge qpos is radians
        data.qpos[q["head_tilt"]] = math.radians(pose.head_tilt_deg)
        wheel = self._wheel_travel_mm / 60.0
        yaw = math.radians(pose.heading_deg)
        for idx, wid in enumerate(q["wheels"]):
            data.qpos[wid] = wheel + yaw * (1.2 if idx % 2 == 0 else -1.2)
        r, gr, b = self._face_state["color_rgb"]
        for k in ("eye_l", "eye_r", "mouth"):
            if g[k] >= 0:
                model.geom_rgba[g[k]] = [r / 255.0, gr / 255.0, b / 255.0, 1.0]
        for k, key in (("eye_l", "left_openness"), ("eye_r", "right_openness")):
            if g[k] >= 0:
                model.geom_size[g[k], 2] = 0.015 * max(0.08, float(self._face_state[key]))
        if g["mouth"] >= 0:
            model.geom_size[g["mouth"], 2] = 0.002 + 0.009 * max(0.08, float(self._face_state["mouth_open"]))
        if g["cliff_zone"] >= 0:
            model.geom_rgba[g["cliff_zone"], 3] = 0.45 if self.cliff_zone_enabled else 0.0
        mocap = st["mocap"]
        for mid in mocap.values():
            data.mocap_pos[mid] = [0.0, 0.0, -3.0]  # hidden below the floor unless placed
        spare = iter(sorted(k for k in mocap if k.startswith("spawn_")))
        for obj in self._objects.values():
            slot = obj.id if obj.id in mocap else next(spare, None)
            if slot is None:
                continue
            on_floor = obj.id in ("yellow_duck", "charging_dock") or obj.shape == "person"
            data.mocap_pos[mocap[slot]] = [obj.x_mm / 1000.0, obj.y_mm / 1000.0,
                                           0.0 if on_floor else obj.height_mm / 2000.0]
            half = math.radians(obj.yaw_deg) / 2.0
            data.mocap_quat[mocap[slot]] = [math.cos(half), 0.0, 0.0, math.sin(half)]
            gid = st["spawn_geoms"].get(slot, -1)
            if gid >= 0:
                cr, cg, cb = _hex_to_rgb(obj.color)
                model.geom_rgba[gid] = [cr / 255.0, cg / 255.0, cb / 255.0, 1.0]
        mujoco.mj_forward(model, data)
        return st

    def set_render_effects(self, *, shadows: bool, reflections: bool) -> None:
        """Shadows and floor reflections in the robot's camera images (off by default; training data varies them)."""
        def apply() -> None:
            for renderer in self._state()["cameras"].values():
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = reflections
        _GL.run(apply)

    def render_rgb(self, camera: str = "front", pose: BodyPose | None = None) -> np.ndarray:
        """What one of the robot's cameras sees (raw RGB, as a physical camera would deliver it)."""
        p = pose or self._last_pose

        def render() -> np.ndarray:
            st = self._sync_mujoco(p)
            renderer: mujoco.Renderer = st["cameras"][camera]
            renderer.update_scene(st["data"], camera=f"cam_{camera}")
            return renderer.render().copy()
        return _GL.run(render)

    def render_labels(self, camera: str = "front", pose: BodyPose | None = None
                      ) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
        """RGB view plus exact per-object boxes from a segmentation render (occlusion included).

        Returns (rgb, {object_id: {"bbox_px": (x1, y1, x2, y2), "pixels": n, "label": str}}): simulator
        ground truth for training and grading a detector.
        """
        p = pose or self._last_pose
        cam = self.cameras[camera]

        def render() -> tuple[Any, np.ndarray, np.ndarray]:
            st = self._sync_mujoco(p)
            model: mujoco.MjModel = st["model"]
            renderer: mujoco.Renderer = st["cameras"][camera]
            renderer.update_scene(st["data"], camera=f"cam_{camera}")
            rgb = renderer.render().copy()
            seg = st["seg"].get(camera)
            if seg is None:
                seg = st["seg"][camera] = mujoco.Renderer(model, height=cam.height, width=cam.width)
                seg.enable_segmentation_rendering()
            seg.update_scene(st["data"], camera=f"cam_{camera}")
            return model, rgb, seg.render()[:, :, 0].copy()  # geom id per pixel, -1 for background
        model, rgb, ids = _GL.run(render)
        mocap = self._state()["mocap"]
        spare = iter(sorted(k for k in mocap if k.startswith("spawn_")))
        body_to_obj: dict[int, str] = {}
        for obj in self._objects.values():
            slot = obj.id if obj.id in mocap else next(spare, None)
            if slot is not None:
                body_to_obj[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{slot}")] = obj.id
        valid = ids >= 0
        bodies = np.full(ids.shape, -1, dtype=np.int64)
        bodies[valid] = model.geom_bodyid[ids[valid]]
        labels: dict[str, dict[str, Any]] = {}
        for bid, obj_id in body_to_obj.items():
            ys, xs = np.nonzero(bodies == bid)
            if len(xs):
                labels[obj_id] = {"bbox_px": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                                  "pixels": len(xs), "label": self._objects[obj_id].label}
        return rgb, labels

    def capture_camera_view(self, pose: BodyPose | None = None, camera: str = "front") -> CameraFrame:
        """A camera image for the dashboard, with what is truly in view (sim truth, for debugging)."""
        p = pose or self._last_pose
        cam = self.cameras[camera]
        frame = CameraFrame(camera, cam.width, cam.height, encode_png(self.render_rgb(camera, p)),
                            self.detect_objects(p, cam))
        if camera == "front":
            self._last_frame = frame
        return frame

    # ---- the dashboard's overview camera ----------------------------------------------------------
    def set_orbit_camera(self, *, azimuth_deg: float | None = None, elevation_deg: float | None = None,
                         distance_m: float | None = None, lookat_m: list[float] | None = None) -> None:
        if azimuth_deg is not None:
            self.cam_azimuth_deg = float(azimuth_deg) % 360.0
        if elevation_deg is not None:
            self.cam_elevation_deg = max(-85.0, min(-5.0, float(elevation_deg)))
        if distance_m is not None:
            self.cam_distance_m = max(0.35, min(3.8, float(distance_m)))
        if lookat_m is not None and len(lookat_m) == 3:
            self.cam_lookat_m = [max(-1.4, min(1.4, float(lookat_m[0]))), max(-1.4, min(1.4, float(lookat_m[1]))),
                                 max(0.0, min(0.8, float(lookat_m[2])))]
        self._last_overview_url = None

    def update_face_state(self, face_dict: dict[str, Any]) -> None:
        """Mirror the live face (eye colour and openness, mouth) on the robot's visor."""
        state = {"color_rgb": tuple(face_dict.get("color_rgb") or (0, 228, 255)),
                 "left_openness": round(float(face_dict.get("left_openness", 1.0)), 2),
                 "right_openness": round(float(face_dict.get("right_openness", 1.0)), 2),
                 "mouth_open": round(float(face_dict.get("mouth_open", 0.08)), 2)}
        if state != self._face_state:
            self._face_state = state
            self._last_overview_url = None

    def _overview_basis(self) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        """(position, forward, right, up) of the orbit camera, in metres."""
        az, el = math.radians(self.cam_azimuth_deg), math.radians(self.cam_elevation_deg)
        fwd = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
        lx, ly, lz = self.cam_lookat_m
        d = self.cam_distance_m
        pos = (lx - d * fwd[0], ly - d * fwd[1], lz - d * fwd[2])
        right = (-math.sin(az), math.cos(az), 0.0)
        up = (-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el))
        return pos, fwd, right, up

    def _overview_tan(self) -> tuple[float, float]:
        ty = math.tan(math.radians(0.5 * self.OVERVIEW_FOVY_DEG))
        return ty * (self.overview_width / float(self.overview_height)), ty

    def project_world_to_overview_uv(self, x_mm: float, y_mm: float, z_mm: float) -> tuple[float, float]:
        """A world point (mm) on the overview image (normalized u, v); (-1, -1) behind the camera."""
        pos, fwd, right, up = self._overview_basis()
        d = (x_mm / 1000.0 - pos[0], y_mm / 1000.0 - pos[1], z_mm / 1000.0 - pos[2])
        zc = sum(a * b for a, b in zip(d, fwd, strict=True))
        if zc <= 0.02:
            return (-1.0, -1.0)
        tx, ty = self._overview_tan()
        xc = sum(a * b for a, b in zip(d, right, strict=True))
        yc = sum(a * b for a, b in zip(d, up, strict=True))
        return 0.5 + 0.5 * xc / (zc * tx), 0.5 - 0.5 * yc / (zc * ty)

    def unproject_overview_uv_to_world(self, u_norm: float, v_norm: float, plane_z_mm: float = 45.0
                                       ) -> tuple[float, float] | None:
        """Where a point on the overview image meets the horizontal plane z = `plane_z_mm`."""
        pos, fwd, right, up = self._overview_basis()
        tx, ty = self._overview_tan()
        nx, ny = (float(u_norm) - 0.5) * 2.0 * tx, (0.5 - float(v_norm)) * 2.0 * ty
        ray = [fwd[i] + nx * right[i] + ny * up[i] for i in range(3)]
        if abs(ray[2]) < 1e-4:
            return None
        t = (plane_z_mm / 1000.0 - pos[2]) / ray[2]
        if t <= 0.0:
            return None
        return (pos[0] + t * ray[0]) * 1000.0, (pos[1] + t * ray[1]) * 1000.0

    def render_overview_3d(self, pose: BodyPose | None = None) -> str:
        """The third-person view of the room as a PNG data URL."""
        p = pose or self._last_pose
        if self._last_overview_url is not None and p == self._last_pose:
            return self._last_overview_url
        az, el, dist, lookat = self.cam_azimuth_deg, self.cam_elevation_deg, self.cam_distance_m, list(self.cam_lookat_m)

        def render() -> np.ndarray:
            st = self._sync_mujoco(p)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
            renderer: mujoco.Renderer = st["over_renderer"]
            renderer.update_scene(st["data"], camera=cam)
            return renderer.render().copy()

        self._last_overview_url = "data:image/png;base64," + base64.b64encode(encode_png(_GL.run(render))).decode()
        return self._last_overview_url

    def launch_native_viewer(self, api_url: str = "http://127.0.0.1:8090") -> bool:
        """Open the native MuJoCo viewer window, synced with the harness through the dashboard API."""
        if self._viewer_proc is not None and self._viewer_proc.poll() is None:
            return True
        script = PROJECT_ROOT / "scripts" / "mujoco_live_viewer.py"
        if not script.exists():
            return False
        try:
            self._viewer_proc = subprocess.Popen([sys.executable, str(script), "--api", api_url, "--xml",
                                                  str(self.xml_path)], stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    # ---- following the robot ----------------------------------------------------------------------
    def _sync_sensors(self, pose: BodyPose) -> None:
        prev = self._last_pose
        self._wheel_travel_mm += math.hypot(pose.x_mm - prev.x_mm, pose.y_mm - prev.y_mm)
        self._last_pose = pose
        if self._sim_body is not None and hasattr(self._sim_body, "set_sandbox_sensors"):
            self._sim_body.set_sandbox_sensors(front_tof_mm=self.compute_front_tof_mm(pose),
                                               cliff=self.check_cliff_triggered(pose))
        self._last_frame = None
        self._last_overview_url = None

    def _on_body_pose(self, event: Event) -> None:
        p = event.payload.get("pose")
        if isinstance(p, dict):
            self._sync_sensors(BodyPose(float(p.get("x_mm", 0.0)), float(p.get("y_mm", 0.0)),
                                        float(p.get("heading_deg", 0.0)), float(p.get("head_tilt_deg", 0.0))))

    def snapshot(self, pose: BodyPose | None = None) -> dict[str, Any]:
        p = pose or self._last_pose
        if self._last_frame is None or p != self._last_pose:
            self._last_pose = p
            self._last_overview_url = None
            frame = self.capture_camera_view(p)
        else:
            frame = self._last_frame
        return {"engine": "mujoco_3d", "room_half_extent_mm": self.room_half_extent_mm,
                "cliff_zone_enabled": self.cliff_zone_enabled, "cliff_zone_rect": list(self.cliff_zone_rect),
                "orbit_camera": {"azimuth_deg": round(self.cam_azimuth_deg, 1),
                                 "elevation_deg": round(self.cam_elevation_deg, 1),
                                 "distance_m": round(self.cam_distance_m, 2),
                                 "lookat_m": [round(x, 2) for x in self.cam_lookat_m],
                                 "data_url": self.render_overview_3d(p)},
                "objects": self.list_objects(), "camera": frame.to_dict()}
