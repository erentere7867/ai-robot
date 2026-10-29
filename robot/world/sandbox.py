"""MuJoCo 3D physics & OpenGL simulation sandbox with `companion_r1.step` CAD meshes and head camera."""

from __future__ import annotations

import atexit
import base64
import math
import struct
import subprocess
import sys
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robot.body.hal import BodyPose
from robot.core.bus import EventBus
from robot.core.config import PROJECT_ROOT
from robot.core.events import Event
from robot.world.state import relative_direction_label

# Shared lock for MuJoCo Renderer calls
_MJ_SHARED_LOCK = threading.Lock()
_MJ_SHARED_STATE: dict[str, Any] = {}


def _cleanup_mujoco_gl_on_exit() -> None:
    for st in list(_MJ_SHARED_STATE.values()):
        for k in ("head_renderer", "over_renderer"):
            r = st.get(k)
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
    _MJ_SHARED_STATE.clear()


atexit.register(_cleanup_mujoco_gl_on_exit)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        try:
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        except ValueError:
            pass
    return (56, 189, 248)


def _encode_rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    """Fast pure-Python/zlib PNG encoder for an (H, W, 3) uint8 RGB image (~2 ms)."""
    h, w, _ = rgb.shape
    filter_col = np.zeros((h, 1), dtype=np.uint8)
    scanlines = np.hstack([filter_col, rgb.reshape(h, w * 3)]).tobytes()

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(scanlines, level=1)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _get_or_init_mujoco_gl(xml_path: Path, head_w: int, head_h: int, over_w: int, over_h: int) -> dict[str, Any]:
    """Initialize or reuse the MuJoCo model, data, and OpenGL renderers on the dedicated GL thread."""
    key = (str(xml_path.resolve()), head_w, head_h, over_w, over_h)
    state = _MJ_SHARED_STATE.get(key)
    if state is not None:
        return state

    xml_text = xml_path.read_text(encoding="utf-8")
    mesh_dir = xml_path.parent / "meshes"
    assets = {p.name: p.read_bytes() for p in mesh_dir.glob("*.stl")} if mesh_dir.exists() else {}
    model = mujoco.MjModel.from_xml_string(xml_text, assets=assets)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    head_renderer = mujoco.Renderer(model, height=head_h, width=head_w)
    over_renderer = mujoco.Renderer(model, height=over_h, width=over_w)

    qpos_ids = {
        "base_x": model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_x")],
        "base_y": model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_y")],
        "base_yaw": model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")],
        "head_tilt": model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_tilt")],
        "wheels": [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
            for jn in ("fl_spin", "fr_spin", "rl_spin", "rr_spin")
        ],
    }

    geom_ids = {
        "eye_l": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_eye_left"),
        "eye_r": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_eye_right"),
        "mouth": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_mouth"),
        "cliff_zone": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cliff_zone_geom"),
    }

    mocap_map: dict[str, int] = {}
    for name in (
        "red_ball",
        "blue_cube",
        "yellow_duck",
        "green_mug",
        "cardboard_box",
        "charging_dock",
        "spawn_1",
        "spawn_2",
        "spawn_3",
        "spawn_4",
    ):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{name}")
        if bid >= 0 and model.body_mocapid[bid] >= 0:
            mocap_map[name] = int(model.body_mocapid[bid])

    spawn_geom_map: dict[str, int] = {}
    for idx in range(1, 5):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_spawn_{idx}")
        if gid >= 0:
            spawn_geom_map[f"spawn_{idx}"] = gid

    state = {
        "model": model,
        "data": data,
        "head_renderer": head_renderer,
        "over_renderer": over_renderer,
        "qpos_ids": qpos_ids,
        "geom_ids": geom_ids,
        "mocap_map": mocap_map,
        "spawn_geom_map": spawn_geom_map,
    }
    _MJ_SHARED_STATE[key] = state
    return state


@dataclass
class SandboxObject:
    """Physical 3D object inside the MuJoCo simulation room."""

    id: str
    label: str
    shape: str  # "sphere" | "box" | "cylinder"
    x_mm: float
    y_mm: float
    z_mm: float = 0.0
    width_mm: float = 100.0
    depth_mm: float = 100.0
    height_mm: float = 100.0
    color: str = "#38bdf8"
    movable: bool = True

    @property
    def radius_mm(self) -> float:
        return 0.5 * max(self.width_mm, self.depth_mm)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "shape": self.shape,
            "x_mm": round(self.x_mm, 1),
            "y_mm": round(self.y_mm, 1),
            "z_mm": round(self.z_mm, 1),
            "width_mm": round(self.width_mm, 1),
            "depth_mm": round(self.depth_mm, 1),
            "height_mm": round(self.height_mm, 1),
            "color": self.color,
            "movable": self.movable,
        }


@dataclass
class GroundedDetection:
    """Object grounded in the head camera's 2D image plane and 3D ego frame."""

    id: str
    label: str
    shape: str
    color: str
    u_norm: float  # [0.0..1.0] left-to-right in image
    v_norm: float  # [0.0..1.0] top-to-bottom in image
    u_px: int
    v_px: int
    bbox_px: tuple[int, int, int, int]
    distance_mm: float
    relative_bearing_deg: float
    elevation_deg: float
    direction: str
    radius_mm: float = 75.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "shape": self.shape,
            "color": self.color,
            "point_norm": [round(self.u_norm, 3), round(self.v_norm, 3)],
            "point_px": [self.u_px, self.v_px],
            "bbox_px": list(self.bbox_px),
            "distance_mm": round(self.distance_mm, 0),
            "relative_bearing_deg": round(self.relative_bearing_deg, 1),
            "elevation_deg": round(self.elevation_deg, 1),
            "direction": self.direction,
            "radius_mm": round(self.radius_mm, 1),
        }


@dataclass
class CameraFrame:
    """Rendered first-person RGB frame and grounded object detections from `head_shell` camera."""

    width: int
    height: int
    png_bytes: bytes
    visible_objects: list[GroundedDetection] = field(default_factory=list)
    camera_pose: dict[str, float] = field(default_factory=dict)
    _data_url_cache: str | None = field(default=None, init=False, repr=False)
    _dict_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @property
    def data_url(self) -> str:
        if self._data_url_cache is None:
            b64 = base64.b64encode(self.png_bytes).decode("ascii")
            self._data_url_cache = f"data:image/png;base64,{b64}"
        return self._data_url_cache

    def render_grounding_summary(self) -> str:
        if not self.visible_objects:
            return "Camera: clear floor ahead."
        parts = []
        for det in self.visible_objects[:4]:
            parts.append(
                f"{det.label} pt=({det.u_norm:.2f},{det.v_norm:.2f}) "
                f"{det.direction} {det.distance_mm:.0f}mm"
            )
        return "Camera: " + "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        if self._dict_cache is None:
            self._dict_cache = {
                "width": self.width,
                "height": self.height,
                "data_url": self.data_url,
                "camera_pose": self.camera_pose,
                "visible_objects": [d.to_dict() for d in self.visible_objects],
                "summary": self.render_grounding_summary(),
            }
        return self._dict_cache


class SimulationSandbox:
    """MuJoCo 3D room sandbox with `companion_r1.step` CAD meshes, raycast sensors, and dual 3D cameras."""

    MAST_PIVOT_X_MM = 25.0
    MAST_PIVOT_Z_MM = 210.0
    CAM_OFFSET_FORWARD_MM = 42.0
    CAM_OFFSET_UP_MM = 76.0
    FRONT_TOF_OFFSET_X_MM = 141.0
    HFOV_DEG = 65.0
    VFOV_DEG = 52.0
    OVERVIEW_FOVY_DEG = 48.0
    DETECTION_RANGE_MM = 2600.0

    def __init__(
        self,
        bus: EventBus | None = None,
        *,
        room_half_extent_mm: float = 1500.0,
        cam_width: int = 320,
        cam_height: int = 240,
        overview_width: int = 520,
        overview_height: int = 340,
        xml_path: Path | None = None,
    ) -> None:
        self.bus = bus
        self.room_half_extent_mm = room_half_extent_mm
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.overview_width = overview_width
        self.overview_height = overview_height
        self.xml_path = xml_path or (PROJECT_ROOT / "config" / "companion_r1.xml")
        self._objects: dict[str, SandboxObject] = {}
        self.cliff_zone_enabled: bool = False
        self.cliff_zone_rect: tuple[float, float, float, float] = (950.0, 1500.0, -600.0, 600.0)

        # Interactive 3D Orbit Camera parameters for the MuJoCo 3D Viewport
        self.cam_azimuth_deg: float = -135.0
        self.cam_elevation_deg: float = -24.0
        self.cam_distance_m: float = 1.35
        self.cam_lookat_m: list[float] = [0.25, 0.0, 0.12]

        # Animated 3D face state on the robot's head visor
        self._face_state: dict[str, Any] = {
            "color_rgb": (0, 228, 255),
            "left_openness": 1.0,
            "right_openness": 1.0,
            "mouth_open": 0.08,
            "gaze_x": 0.0,
            "gaze_y": 0.0,
        }

        self._last_pose = BodyPose()
        self._wheel_travel_mm = 0.0
        self._last_frame: CameraFrame | None = None
        self._last_overview_url: str | None = None
        self._viewer_proc: subprocess.Popen[Any] | None = None
        self._sub_ids: list[str] = []
        self._sim_body: Any | None = None
        self.reset_default_scene()

    def bind(self, sim_body: Any | None = None) -> None:
        """Bind to `body.pose` on the EventBus to update raycast sensors and invalidate 3D camera caches."""
        self._sim_body = sim_body
        if sim_body is not None and hasattr(sim_body, "attach_sensor_model"):
            # Body samples ToF/cliff synchronously every control tick (no bus latency in safety checks)
            sim_body.attach_sensor_model(self)
        if self.bus is not None:
            self._sub_ids.append(
                self.bus.subscribe("body.pose", self._on_body_pose, name="sandbox:pose")
            )
            self._sync_sensors_and_observations(self._last_pose, emit_bus=False)

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

    def reset_default_scene(self) -> None:
        """Populate the 3m x 3m indoor companion playroom with default interactive 3D objects."""
        self._objects = {
            "red_ball": SandboxObject(
                id="red_ball",
                label="Red Ball",
                shape="sphere",
                x_mm=680.0,
                y_mm=160.0,
                width_mm=96.0,
                depth_mm=96.0,
                height_mm=96.0,
                color="#f43f5e",
            ),
            "blue_cube": SandboxObject(
                id="blue_cube",
                label="Blue Cube",
                shape="box",
                x_mm=620.0,
                y_mm=-210.0,
                width_mm=110.0,
                depth_mm=110.0,
                height_mm=110.0,
                color="#3b82f6",
            ),
            "yellow_duck": SandboxObject(
                id="yellow_duck",
                label="Yellow Toy Duck",
                shape="sphere",
                x_mm=470.0,
                y_mm=340.0,
                width_mm=90.0,
                depth_mm=90.0,
                height_mm=96.0,
                color="#eab308",
            ),
            "green_mug": SandboxObject(
                id="green_mug",
                label="Green Mug",
                shape="cylinder",
                x_mm=840.0,
                y_mm=-30.0,
                width_mm=86.0,
                depth_mm=86.0,
                height_mm=116.0,
                color="#10b981",
            ),
            "cardboard_box": SandboxObject(
                id="cardboard_box",
                label="Cardboard Box",
                shape="box",
                x_mm=1120.0,
                y_mm=280.0,
                width_mm=220.0,
                depth_mm=220.0,
                height_mm=190.0,
                color="#d97706",
            ),
            "charging_dock": SandboxObject(
                id="charging_dock",
                label="Charging Dock",
                shape="box",
                x_mm=-680.0,
                y_mm=0.0,
                width_mm=180.0,
                depth_mm=220.0,
                height_mm=130.0,
                color="#8b5cf6",
            ),
        }
        self.cliff_zone_enabled = False
        self._invalidate_renders()
        if self._sim_body is not None:
            self._sync_sensors_and_observations(self._sim_body.pose(), emit_bus=False)

    def _invalidate_renders(self) -> None:
        self._last_frame = None
        self._last_overview_url = None

    def set_orbit_camera(
        self,
        *,
        azimuth_deg: float | None = None,
        elevation_deg: float | None = None,
        distance_m: float | None = None,
        lookat_m: list[float] | None = None,
    ) -> None:
        """Update the interactive 3D MuJoCo free camera angles/distance/lookat."""
        if azimuth_deg is not None:
            self.cam_azimuth_deg = float(azimuth_deg) % 360.0
        if elevation_deg is not None:
            self.cam_elevation_deg = max(-85.0, min(-5.0, float(elevation_deg)))
        if distance_m is not None:
            self.cam_distance_m = max(0.35, min(3.8, float(distance_m)))
        if lookat_m is not None and len(lookat_m) == 3:
            self.cam_lookat_m = [
                max(-1.4, min(1.4, float(lookat_m[0]))),
                max(-1.4, min(1.4, float(lookat_m[1]))),
                max(0.0, min(0.8, float(lookat_m[2]))),
            ]
        self._last_overview_url = None

    def update_face_state(self, face_dict: dict[str, Any]) -> None:
        """Sync the 3D glowing eyes and mouth on the MuJoCo robot's visor with the live FaceFrame."""
        new_rgb = tuple(face_dict.get("color_rgb") or (0, 228, 255))
        lo = round(float(face_dict.get("left_openness", 1.0)), 2)
        ro = round(float(face_dict.get("right_openness", 1.0)), 2)
        mo = round(float(face_dict.get("mouth_open", 0.08)), 2)
        gx = round(float(face_dict.get("gaze_x", 0.0)), 2)
        gy = round(float(face_dict.get("gaze_y", 0.0)), 2)
        new_state = {
            "color_rgb": new_rgb,
            "left_openness": lo,
            "right_openness": ro,
            "mouth_open": mo,
            "gaze_x": gx,
            "gaze_y": gy,
        }
        if new_state != self._face_state:
            self._face_state = new_state
            self._last_overview_url = None

    def launch_native_viewer(self, api_url: str = "http://127.0.0.1:8090") -> bool:
        """Launch the standalone 60 FPS native desktop `mujoco.viewer` window synced with the harness."""
        if self._viewer_proc is not None and self._viewer_proc.poll() is None:
            return True
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "mujoco_live_viewer.py"
        if not script_path.exists():
            return False
        try:
            self._viewer_proc = subprocess.Popen(
                [sys.executable, str(script_path), "--api", api_url, "--xml", str(self.xml_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def list_objects(self) -> list[dict[str, Any]]:
        res = []
        for obj in self._objects.values():
            d = obj.to_dict()
            u_norm, v_norm = self.project_world_to_overview_uv(
                obj.x_mm, obj.y_mm, obj.z_mm + 0.5 * obj.height_mm
            )
            d["overview_uv"] = [round(u_norm, 3), round(v_norm, 3)]
            res.append(d)
        return res

    def get_object(self, obj_id: str) -> SandboxObject | None:
        return self._objects.get(obj_id)

    def find_object_by_query(self, query: str) -> SandboxObject | None:
        q = query.lower().strip()
        for obj in self._objects.values():
            if obj.id.lower() in q or obj.label.lower() in q:
                return obj
            words = [w for w in obj.label.lower().split() if len(w) >= 3]
            if any(w in q for w in words):
                return obj
        return None

    def move_object(self, obj_id: str, x_mm: float, y_mm: float) -> SandboxObject | None:
        obj = self._objects.get(obj_id)
        if obj is None:
            return None
        lim = self.room_half_extent_mm - obj.radius_mm
        obj.x_mm = max(-lim, min(lim, float(x_mm)))
        obj.y_mm = max(-lim, min(lim, float(y_mm)))
        self._invalidate_renders()
        if self._sim_body is not None:
            self._sync_sensors_and_observations(self._sim_body.pose(), emit_bus=False)
        return obj

    def move_object_by_overview_ray(self, obj_id: str, u_norm: float, v_norm: float) -> SandboxObject | None:
        """Unproject a 2D mouse drag `(u_norm, v_norm)` on the 3D MuJoCo viewport onto the floor plane."""
        obj = self._objects.get(obj_id)
        if obj is None:
            return None
        hit = self.unproject_overview_uv_to_world(u_norm, v_norm, plane_z_mm=0.5 * obj.height_mm)
        if hit is None:
            return obj
        return self.move_object(obj_id, hit[0], hit[1])

    def add_object(
        self,
        *,
        obj_id: str | None = None,
        label: str = "Orange Cone",
        shape: str = "cylinder",
        x_mm: float = 550.0,
        y_mm: float = 0.0,
        size_mm: float = 100.0,
        color: str = "#f97316",
    ) -> SandboxObject:
        clean_id = (obj_id or label.lower().replace(" ", "_")).strip()[:40]
        if clean_id in self._objects:
            idx = 2
            while f"{clean_id}_{idx}" in self._objects:
                idx += 1
            clean_id = f"{clean_id}_{idx}"
        sz = max(40.0, min(400.0, float(size_mm)))
        lim = self.room_half_extent_mm - sz * 0.5
        obj = SandboxObject(
            id=clean_id,
            label=label[:60],
            shape=shape if shape in ("sphere", "box", "cylinder") else "box",
            x_mm=max(-lim, min(lim, float(x_mm))),
            y_mm=max(-lim, min(lim, float(y_mm))),
            width_mm=sz,
            depth_mm=sz,
            height_mm=sz,
            color=color,
        )
        self._objects[clean_id] = obj
        self._invalidate_renders()
        if self._sim_body is not None:
            self._sync_sensors_and_observations(self._sim_body.pose(), emit_bus=False)
        return obj

    def remove_object(self, obj_id: str) -> bool:
        removed = self._objects.pop(obj_id, None) is not None
        if removed:
            self._invalidate_renders()
            if self._sim_body is not None:
                self._sync_sensors_and_observations(self._sim_body.pose(), emit_bus=False)
        return removed

    def set_cliff_zone(self, enabled: bool) -> None:
        self.cliff_zone_enabled = bool(enabled)
        self._invalidate_renders()
        if self._sim_body is not None:
            self._sync_sensors_and_observations(self._sim_body.pose(), emit_bus=False)

    def compute_front_tof_mm(self, pose: BodyPose) -> float:
        """Raycast forward ToF distance (mm) from the front bumper against objects and room walls."""
        yaw = math.radians(pose.heading_deg)
        dir_x = math.cos(yaw)
        dir_y = math.sin(yaw)
        origin_x = pose.x_mm + self.FRONT_TOF_OFFSET_X_MM * dir_x
        origin_y = pose.y_mm + self.FRONT_TOF_OFFSET_X_MM * dir_y

        min_dist = 2500.0
        half = self.room_half_extent_mm
        if abs(dir_x) > 1e-6:
            tx = ((half if dir_x > 0 else -half) - origin_x) / dir_x
            if tx > 0:
                min_dist = min(min_dist, tx)
        if abs(dir_y) > 1e-6:
            ty = ((half if dir_y > 0 else -half) - origin_y) / dir_y
            if ty > 0:
                min_dist = min(min_dist, ty)

        cone_rad = math.radians(14.0)
        for obj in self._objects.values():
            dx = obj.x_mm - origin_x
            dy = obj.y_mm - origin_y
            forward = dx * dir_x + dy * dir_y
            if forward <= 0:
                continue
            lateral = abs(-dx * dir_y + dy * dir_x)
            beam_half_width = obj.radius_mm + forward * math.tan(cone_rad)
            if lateral <= beam_half_width:
                surface_dist = max(0.0, math.hypot(dx, dy) - obj.radius_mm)
                min_dist = min(min_dist, surface_dist)

        return round(max(10.0, min(2500.0, min_dist)), 1)

    def check_cliff_triggered(self, pose: BodyPose) -> bool:
        """Check if any of the 4 corner cliff sensors is over the active cliff zone."""
        if not self.cliff_zone_enabled:
            return False
        x1, x2, y1, y2 = self.cliff_zone_rect
        yaw = math.radians(pose.heading_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        for lx, ly in ((120.0, 92.0), (120.0, -92.0), (-120.0, 92.0), (-120.0, -92.0)):
            wx = pose.x_mm + lx * c - ly * s
            wy = pose.y_mm + lx * s + ly * c
            if x1 <= wx <= x2 and y1 <= wy <= y2:
                return True
        return False

    def compute_camera_extrinsics(self, pose: BodyPose) -> dict[str, float]:
        """Compute 3D world pose of the `head_shell` camera from `companion_r1.step` kinematics."""
        yaw_rad = math.radians(pose.heading_deg)
        pitch_rad = math.radians(pose.head_tilt_deg)
        local_fwd = self.MAST_PIVOT_X_MM + (
            self.CAM_OFFSET_FORWARD_MM * math.cos(pitch_rad)
            - self.CAM_OFFSET_UP_MM * math.sin(pitch_rad)
        )
        cam_z = self.MAST_PIVOT_Z_MM + (
            self.CAM_OFFSET_FORWARD_MM * math.sin(pitch_rad)
            + self.CAM_OFFSET_UP_MM * math.cos(pitch_rad)
        )
        cam_x = pose.x_mm + local_fwd * math.cos(yaw_rad)
        cam_y = pose.y_mm + local_fwd * math.sin(yaw_rad)
        return {
            "x_mm": round(cam_x, 1),
            "y_mm": round(cam_y, 1),
            "z_mm": round(cam_z, 1),
            "yaw_deg": round(pose.heading_deg, 1),
            "pitch_deg": round(pose.head_tilt_deg, 1),
        }

    def _overview_camera_basis(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        """Compute (cam_pos, fwd, right, up) in meters for the MuJoCo `MjvCamera` free camera."""
        az = math.radians(self.cam_azimuth_deg)
        el = math.radians(self.cam_elevation_deg)
        fx = math.cos(el) * math.cos(az)
        fy = math.cos(el) * math.sin(az)
        fz = math.sin(el)
        lx, ly, lz = self.cam_lookat_m
        d = self.cam_distance_m
        cam_pos = (lx - d * fx, ly - d * fy, lz - d * fz)
        right = (-math.sin(az), math.cos(az), 0.0)
        up = (-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el))
        return cam_pos, (fx, fy, fz), right, up

    def project_world_to_overview_uv(self, x_mm: float, y_mm: float, z_mm: float) -> tuple[float, float]:
        """Project a 3D world point (mm) into normalized `(u_norm, v_norm)` on the 3D overview viewport."""
        cam_pos, fwd, right, up = self._overview_camera_basis()
        dx = x_mm / 1000.0 - cam_pos[0]
        dy = y_mm / 1000.0 - cam_pos[1]
        dz = z_mm / 1000.0 - cam_pos[2]
        zc = dx * fwd[0] + dy * fwd[1] + dz * fwd[2]
        if zc <= 0.02:
            return (-1.0, -1.0)
        xc = dx * right[0] + dy * right[1] + dz * right[2]
        yc = dx * up[0] + dy * up[1] + dz * up[2]
        tan_half_fovy = math.tan(math.radians(0.5 * self.OVERVIEW_FOVY_DEG))
        tan_half_fovx = tan_half_fovy * (self.overview_width / float(self.overview_height))
        u_norm = 0.5 + 0.5 * (xc / (zc * tan_half_fovx))
        v_norm = 0.5 - 0.5 * (yc / (zc * tan_half_fovy))
        return (u_norm, v_norm)

    def unproject_overview_uv_to_world(
        self, u_norm: float, v_norm: float, plane_z_mm: float = 45.0
    ) -> tuple[float, float] | None:
        """Intersect a ray through `(u_norm, v_norm)` on the 3D overview viewport with `Z = plane_z_mm`."""
        cam_pos, fwd, right, up = self._overview_camera_basis()
        tan_half_fovy = math.tan(math.radians(0.5 * self.OVERVIEW_FOVY_DEG))
        tan_half_fovx = tan_half_fovy * (self.overview_width / float(self.overview_height))
        nx = (float(u_norm) - 0.5) * 2.0 * tan_half_fovx
        ny = (0.5 - float(v_norm)) * 2.0 * tan_half_fovy
        rx = fwd[0] + nx * right[0] + ny * up[0]
        ry = fwd[1] + nx * right[1] + ny * up[1]
        rz = fwd[2] + nx * right[2] + ny * up[2]
        if abs(rz) < 1e-4:
            return None
        target_z_m = plane_z_mm / 1000.0
        t = (target_z_m - cam_pos[2]) / rz
        if t <= 0.0:
            return None
        wx_mm = (cam_pos[0] + t * rx) * 1000.0
        wy_mm = (cam_pos[1] + t * ry) * 1000.0
        return (wx_mm, wy_mm)

    def _compute_visible_detections(self, pose: BodyPose) -> list[GroundedDetection]:
        """Compute grounded `(u, v)` normalized coordinates and 3D ego metrics for visible objects."""
        w, h = self.cam_width, self.cam_height
        ext = self.compute_camera_extrinsics(pose)
        cam_x, cam_y, cam_z = ext["x_mm"], ext["y_mm"], ext["z_mm"]
        yaw_rad = math.radians(ext["yaw_deg"])
        # Note: head_cam in companion_r1.xml has a -9 deg base look-down angle so floor objects are centered
        pitch_rad = math.radians(ext["pitch_deg"] - 9.0)

        fx = (0.5 * w) / math.tan(math.radians(0.5 * self.HFOV_DEG))
        fy = (0.5 * h) / math.tan(math.radians(0.5 * self.VFOV_DEG))
        cx = 0.5 * w
        cy = 0.5 * h

        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        visible: list[GroundedDetection] = []

        for obj in self._objects.values():
            dx = obj.x_mm - cam_x
            dy = obj.y_mm - cam_y
            dz = (obj.z_mm + 0.5 * obj.height_mm) - cam_z

            rx = dx * cos_y + dy * sin_y
            ry = -dx * sin_y + dy * cos_y
            rz = dz

            cam_fwd = rx * cos_p + rz * sin_p
            cam_up = -rx * sin_p + rz * cos_p
            cam_right = -ry

            if cam_fwd <= 40.0:
                continue

            u_f = cx + fx * (cam_right / cam_fwd)
            v_f = cy - fy * (cam_up / cam_fwd)
            proj_w = max(8, int(round(fx * (obj.width_mm / cam_fwd))))
            proj_h = max(8, int(round(fy * (obj.height_mm / cam_fwd))))

            x1 = int(round(u_f - 0.5 * proj_w))
            x2 = int(round(u_f + 0.5 * proj_w))
            y1 = int(round(v_f - 0.5 * proj_h))
            y2 = int(round(v_f + 0.5 * proj_h))

            if x2 < 0 or x1 >= w or y2 < 0 or y1 >= h:
                continue

            # Realistic perception limits: detector range and line-of-sight occlusion
            if math.hypot(obj.x_mm - cam_x, obj.y_mm - cam_y) > self.DETECTION_RANGE_MM:
                continue
            if self._is_occluded(obj, cam_x, cam_y):
                continue

            u_px = max(0, min(w - 1, int(round(u_f))))
            v_px = max(0, min(h - 1, int(round(v_f))))
            u_norm = round(u_px / float(w), 3)
            v_norm = round(v_px / float(h), 3)

            dist_mm = math.hypot(obj.x_mm - pose.x_mm, obj.y_mm - pose.y_mm)
            world_bearing = math.degrees(math.atan2(obj.y_mm - pose.y_mm, obj.x_mm - pose.x_mm))
            rel_bearing = (world_bearing - pose.heading_deg + 180.0) % 360.0 - 180.0
            elev_deg = math.degrees(math.atan2(cam_up, cam_fwd))

            visible.append(
                GroundedDetection(
                    id=obj.id,
                    label=obj.label,
                    shape=obj.shape,
                    color=obj.color,
                    u_norm=u_norm,
                    v_norm=v_norm,
                    u_px=u_px,
                    v_px=v_px,
                    bbox_px=(max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)),
                    distance_mm=dist_mm,
                    relative_bearing_deg=rel_bearing,
                    elevation_deg=elev_deg,
                    direction=relative_direction_label(rel_bearing),
                    radius_mm=0.5 * math.hypot(obj.width_mm, obj.depth_mm),
                )
            )

        visible.sort(key=lambda d: d.distance_mm)
        return visible

    def _is_occluded(self, target: SandboxObject, cam_x: float, cam_y: float) -> bool:
        """True if a nearer, sufficiently tall object blocks the camera's line of sight to `target`."""
        tx, ty = target.x_mm - cam_x, target.y_mm - cam_y
        t_dist = math.hypot(tx, ty)
        if t_dist < 1e-6:
            return False
        ux, uy = tx / t_dist, ty / t_dist
        for other in self._objects.values():
            if other.id == target.id:
                continue
            ox, oy = other.x_mm - cam_x, other.y_mm - cam_y
            along = ox * ux + oy * uy
            if along <= 0.0 or along >= t_dist - target.radius_mm:
                continue
            lateral = abs(-ox * uy + oy * ux)
            # Must cover the target centre ray and be at least ~70% as tall to hide it
            if lateral < other.radius_mm * 0.9 and other.height_mm >= 0.7 * target.height_mm:
                return True
        return False

    def detect_objects(self, pose: BodyPose | None = None) -> list[GroundedDetection]:
        """Cheap geometric object detector for the head camera (no OpenGL render)."""
        return self._compute_visible_detections(pose or self._last_pose)

    def point_in_view(self, pose: BodyPose, x_mm: float, y_mm: float, z_mm: float = 50.0) -> bool:
        """True if a world point is inside the head camera frustum and detection range."""
        ext = self.compute_camera_extrinsics(pose)
        dx, dy = x_mm - ext["x_mm"], y_mm - ext["y_mm"]
        if math.hypot(dx, dy) > self.DETECTION_RANGE_MM:
            return False
        yaw = math.radians(ext["yaw_deg"])
        pitch = math.radians(ext["pitch_deg"] - 9.0)
        rx = dx * math.cos(yaw) + dy * math.sin(yaw)
        ry = -dx * math.sin(yaw) + dy * math.cos(yaw)
        rz = z_mm - ext["z_mm"]
        fwd = rx * math.cos(pitch) + rz * math.sin(pitch)
        up = -rx * math.sin(pitch) + rz * math.cos(pitch)
        if fwd <= 60.0:
            return False
        h_ang = math.degrees(math.atan2(-ry, fwd))
        v_ang = math.degrees(math.atan2(up, fwd))
        return abs(h_ang) <= 0.5 * self.HFOV_DEG - 3.0 and abs(v_ang) <= 0.5 * self.VFOV_DEG

    def unproject_image_point_to_floor(
        self, pose: BodyPose, u_norm: float, v_norm: float
    ) -> tuple[float, float] | None:
        """Cast a ray through a normalized head-camera pixel ("pointing") and intersect the floor (z=0)."""
        ext = self.compute_camera_extrinsics(pose)
        w, h = self.cam_width, self.cam_height
        fx = (0.5 * w) / math.tan(math.radians(0.5 * self.HFOV_DEG))
        fy = (0.5 * h) / math.tan(math.radians(0.5 * self.VFOV_DEG))
        cam_right = (u_norm * w - 0.5 * w) / fx
        cam_up = -(v_norm * h - 0.5 * h) / fy
        pitch = math.radians(ext["pitch_deg"] - 9.0)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        rx = cos_p - cam_up * sin_p
        rz = sin_p + cam_up * cos_p
        ry = -cam_right
        if rz >= -1e-4:
            return None  # ray points at or above the horizon
        t = -ext["z_mm"] / rz
        yaw = math.radians(ext["yaw_deg"])
        wx = ext["x_mm"] + t * (rx * math.cos(yaw) - ry * math.sin(yaw))
        wy = ext["y_mm"] + t * (rx * math.sin(yaw) + ry * math.cos(yaw))
        return (wx, wy)

    def _sync_mujoco_state_on_gl_thread(self, pose: BodyPose) -> dict[str, Any]:
        """Update `MjModel` / `MjData` with current robot pose, 3D face eyes, and sandbox objects."""
        st = _get_or_init_mujoco_gl(
            self.xml_path, self.cam_width, self.cam_height, self.overview_width, self.overview_height
        )
        model: mujoco.MjModel = st["model"]
        data: mujoco.MjData = st["data"]
        qids = st["qpos_ids"]
        gids = st["geom_ids"]
        mocap_map: dict[str, int] = st["mocap_map"]
        spawn_geom_map: dict[str, int] = st["spawn_geom_map"]

        # 1. Robot chassis pose & head tilt & wheel rotation
        data.qpos[qids["base_x"]] = pose.x_mm / 1000.0
        data.qpos[qids["base_y"]] = pose.y_mm / 1000.0
        # MuJoCo hinge-joint qpos is always in radians (compiler angle="degree" only affects XML parsing)
        data.qpos[qids["base_yaw"]] = math.radians(pose.heading_deg)
        data.qpos[qids["head_tilt"]] = math.radians(pose.head_tilt_deg)

        wheel_angle_rad = self._wheel_travel_mm / 60.0
        yaw_rad = math.radians(pose.heading_deg)
        for idx, wid in enumerate(qids["wheels"]):
            data.qpos[wid] = wheel_angle_rad + (yaw_rad * (1.2 if idx % 2 == 0 else -1.2))

        # 2. Glowing 3D eyes & mouth on the head visor
        rgb = self._face_state["color_rgb"]
        rgba = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0]
        for k in ("eye_l", "eye_r", "mouth"):
            gid = gids.get(k, -1)
            if gid >= 0:
                model.geom_rgba[gid] = rgba
        if gids["eye_l"] >= 0:
            model.geom_size[gids["eye_l"], 2] = 0.015 * max(0.08, float(self._face_state["left_openness"]))
        if gids["eye_r"] >= 0:
            model.geom_size[gids["eye_r"], 2] = 0.015 * max(0.08, float(self._face_state["right_openness"]))
        if gids["mouth"] >= 0:
            model.geom_size[gids["mouth"], 2] = 0.002 + 0.009 * max(
                0.08, float(self._face_state["mouth_open"])
            )

        # 3. Cliff hazard zone visibility
        if gids["cliff_zone"] >= 0:
            model.geom_rgba[gids["cliff_zone"], 3] = 0.45 if self.cliff_zone_enabled else 0.0

        # 4. Sandbox objects (default + dynamic spawn slots)
        # First hide all mocap bodies below the floor
        for mid in mocap_map.values():
            data.mocap_pos[mid] = [0.0, 0.0, -1.5]

        extra_idx = 1
        for obj in self._objects.values():
            if obj.id in mocap_map:
                mid = mocap_map[obj.id]
                z_m = 0.0 if obj.id in ("yellow_duck", "charging_dock") else (obj.height_mm * 0.5) / 1000.0
                data.mocap_pos[mid] = [obj.x_mm / 1000.0, obj.y_mm / 1000.0, z_m]
            elif extra_idx <= 4:
                slot_name = f"spawn_{extra_idx}"
                extra_idx += 1
                mid = mocap_map.get(slot_name)
                gid = spawn_geom_map.get(slot_name, -1)
                if mid is not None:
                    data.mocap_pos[mid] = [
                        obj.x_mm / 1000.0,
                        obj.y_mm / 1000.0,
                        (obj.height_mm * 0.5) / 1000.0,
                    ]
                if gid >= 0:
                    cr, cg, cb = _hex_to_rgb(obj.color)
                    model.geom_rgba[gid] = [cr / 255.0, cg / 255.0, cb / 255.0, 1.0]

        mujoco.mj_forward(model, data)
        return st

    def capture_camera_view(self, pose: BodyPose | None = None) -> CameraFrame:
        """Render first-person 3D OpenGL RGB view from `head_cam` in MuJoCo and ground visible objects."""
        p = pose or self._last_pose
        ext = self.compute_camera_extrinsics(p)
        visible_detections = self._compute_visible_detections(p)

        def _render_head() -> np.ndarray:
            with _MJ_SHARED_LOCK:
                st = self._sync_mujoco_state_on_gl_thread(p)
                renderer: mujoco.Renderer = st["head_renderer"]
                renderer.update_scene(st["data"], camera="head_cam")
                return renderer.render().copy()

        img = _render_head()
        # Draw subtle white 3x3 grounding reticle at each detected object center
        h, w, _ = img.shape
        for det in visible_detections:
            uc = max(1, min(w - 2, det.u_px))
            vc = max(1, min(h - 2, det.v_px))
            img[vc - 1 : vc + 2, uc - 1 : uc + 2] = (255, 255, 255)

        frame = CameraFrame(
            width=self.cam_width,
            height=self.cam_height,
            png_bytes=_encode_rgb_to_png_bytes(img),
            visible_objects=visible_detections,
            camera_pose=ext,
        )
        self._last_frame = frame
        return frame

    def render_overview_3d(self, pose: BodyPose | None = None) -> str:
        """Render the interactive 3rd-person 3D MuJoCo scene (`overview_cam`) as a `data:image/png;base64` URL."""
        p = pose or self._last_pose
        if self._last_overview_url is not None and p == self._last_pose:
            return self._last_overview_url

        az = self.cam_azimuth_deg
        el = self.cam_elevation_deg
        dist = self.cam_distance_m
        lookat = list(self.cam_lookat_m)

        def _render_over() -> np.ndarray:
            with _MJ_SHARED_LOCK:
                st = self._sync_mujoco_state_on_gl_thread(p)
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = lookat
                cam.distance = dist
                cam.azimuth = az
                cam.elevation = el
                renderer: mujoco.Renderer = st["over_renderer"]
                renderer.update_scene(st["data"], camera=cam)
                return renderer.render().copy()

        img = _render_over()
        b64 = base64.b64encode(_encode_rgb_to_png_bytes(img)).decode("ascii")
        self._last_overview_url = f"data:image/png;base64,{b64}"
        return self._last_overview_url

    def _sync_sensors_and_observations(self, pose: BodyPose, *, emit_bus: bool = True) -> None:
        prev = self._last_pose
        self._wheel_travel_mm += math.hypot(pose.x_mm - prev.x_mm, pose.y_mm - prev.y_mm)
        self._last_pose = pose
        tof_mm = self.compute_front_tof_mm(pose)
        cliff = self.check_cliff_triggered(pose)
        if self._sim_body is not None and hasattr(self._sim_body, "set_sandbox_sensors"):
            self._sim_body.set_sandbox_sensors(front_tof_mm=tof_mm, cliff=cliff)
        self._invalidate_renders()

    def _on_body_pose(self, event: Event) -> None:
        p = event.payload.get("pose")
        if not isinstance(p, dict):
            return
        pose = BodyPose(
            x_mm=float(p.get("x_mm", 0.0)),
            y_mm=float(p.get("y_mm", 0.0)),
            heading_deg=float(p.get("heading_deg", 0.0)),
            head_tilt_deg=float(p.get("head_tilt_deg", 0.0)),
            linear_vel_mm_s=float(p.get("linear_vel_mm_s", 0.0)),
            angular_vel_deg_s=float(p.get("angular_vel_deg_s", 0.0)),
        )
        self._sync_sensors_and_observations(pose, emit_bus=True)

    def snapshot(self, pose: BodyPose | None = None) -> dict[str, Any]:
        p = pose or self._last_pose
        if (
            self._last_frame is None
            or p.x_mm != self._last_pose.x_mm
            or p.y_mm != self._last_pose.y_mm
            or p.heading_deg != self._last_pose.heading_deg
            or p.head_tilt_deg != self._last_pose.head_tilt_deg
        ):
            self._last_pose = p
            self._invalidate_renders()
            frame = self.capture_camera_view(p)
        else:
            frame = self._last_frame

        overview_url = self.render_overview_3d(p)
        return {
            "engine": "mujoco_3d",
            "room_half_extent_mm": self.room_half_extent_mm,
            "cliff_zone_enabled": self.cliff_zone_enabled,
            "cliff_zone_rect": list(self.cliff_zone_rect),
            "orbit_camera": {
                "azimuth_deg": round(self.cam_azimuth_deg, 1),
                "elevation_deg": round(self.cam_elevation_deg, 1),
                "distance_m": round(self.cam_distance_m, 2),
                "lookat_m": [round(x, 2) for x in self.cam_lookat_m],
                "data_url": overview_url,
            },
            "objects": self.list_objects(),
            "camera": frame.to_dict(),
        }
