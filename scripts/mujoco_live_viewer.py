"""Native desktop 60 FPS MuJoCo 3D interactive viewer synced bidirectionally with the Sprocket harness."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import mujoco
import mujoco.viewer


def _fetch_state(api_url: str) -> dict | None:
    try:
        host = api_url.split("//")[-1]
        req = urllib.request.Request(f"{api_url}/api/state", headers={"Host": host})
        with urllib.request.urlopen(req, timeout=0.4) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _post_move_object(api_url: str, obj_id: str, x_mm: float, y_mm: float) -> None:
    try:
        host = api_url.split("//")[-1]
        payload = json.dumps({"op": "move", "id": obj_id, "x_mm": round(x_mm, 1), "y_mm": round(y_mm, 1)}).encode(
            "utf-8"
        )
        req = urllib.request.Request(
            f"{api_url}/api/sandbox",
            data=payload,
            headers={"Host": host, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=0.3):
            pass
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Live MuJoCo 3D Viewer for Sprocket Companion Robot")
    parser.add_argument("--api", default="http://127.0.0.1:8090", help="Dashboard API base URL")
    parser.add_argument("--xml", default="config/companion_r1.xml", help="Path to MuJoCo XML scene")
    args = parser.parse_args()

    xml_path = Path(args.xml).resolve()
    xml_text = xml_path.read_text(encoding="utf-8")
    mesh_dir = xml_path.parent / "meshes"
    assets = {p.name: p.read_bytes() for p in mesh_dir.glob("*.stl")} if mesh_dir.exists() else {}
    model = mujoco.MjModel.from_xml_string(xml_text, assets=assets)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    qpos_bx = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_x")]
    qpos_by = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_y")]
    qpos_byaw = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")]
    qpos_tilt = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_tilt")]

    wheel_qpos = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)]
        for jname in ("fl_spin", "fr_spin", "rl_spin", "rr_spin")
    ]

    eye_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_eye_left")
    eye_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_eye_right")
    mouth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_mouth")
    cliff_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cliff_zone_geom")

    obj_mocap_map: dict[str, int] = {}
    last_mocap_xy: dict[str, tuple[float, float]] = {}
    for oid in (
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
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{oid}")
        if bid >= 0 and model.body_mocapid[bid] >= 0:
            mid = int(model.body_mocapid[bid])
            obj_mocap_map[oid] = mid
            last_mocap_xy[oid] = (float(data.mocap_pos[mid, 0]), float(data.mocap_pos[mid, 1]))

    spawn_geom_map = {
        f"spawn_{i}": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_spawn_{i}")
        for i in range(1, 5)
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.25, 0.0, 0.14]
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = -135.0
        viewer.cam.elevation = -24.0

        last_poll = 0.0
        wheel_travel = 0.0
        target_pose = {"x_mm": 0.0, "y_mm": 0.0, "heading_deg": 0.0, "head_tilt_deg": 0.0}

        while viewer.is_running():
            now = time.perf_counter()

            # Check if the user moved any mocap object directly inside the MuJoCo 3D viewer
            user_dragged = False
            for oid, mid in obj_mocap_map.items():
                cx, cy = float(data.mocap_pos[mid, 0]), float(data.mocap_pos[mid, 1])
                px, py = last_mocap_xy[oid]
                if math.hypot(cx - px, cy - py) > 0.005:
                    last_mocap_xy[oid] = (cx, cy)
                    _post_move_object(args.api, oid, cx * 1000.0, cy * 1000.0)
                    user_dragged = True

            if now - last_poll >= 0.08 and not user_dragged:
                last_poll = now
                st = _fetch_state(args.api)
                if st:
                    w = st.get("world") or {}
                    p = w.get("pose") or {}
                    target_pose = p

                    # Sync 3D eyes & mouth from live FaceFrame
                    f = st.get("face") or {}
                    rgb = f.get("color_rgb") or [0, 228, 255]
                    rgba = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0]
                    for gid in (eye_l_id, eye_r_id, mouth_id):
                        if gid >= 0:
                            model.geom_rgba[gid] = rgba
                    if eye_l_id >= 0:
                        lo = max(0.08, float(f.get("left_openness", 1.0)))
                        model.geom_size[eye_l_id, 2] = 0.015 * lo
                    if eye_r_id >= 0:
                        ro = max(0.08, float(f.get("right_openness", 1.0)))
                        model.geom_size[eye_r_id, 2] = 0.015 * ro
                    if mouth_id >= 0:
                        mo = max(0.12, float(f.get("mouth_open", 0.1)))
                        model.geom_size[mouth_id, 2] = 0.002 + 0.009 * mo

                    sb = st.get("sandbox") or {}
                    if cliff_id >= 0:
                        model.geom_rgba[cliff_id, 3] = 0.45 if sb.get("cliff_zone_enabled") else 0.0

                    extra_idx = 1
                    for obj in sb.get("objects") or []:
                        oid = obj.get("id", "")
                        nx = float(obj.get("x_mm", 0.0)) / 1000.0
                        ny = float(obj.get("y_mm", 0.0)) / 1000.0
                        h_m = float(obj.get("height_mm", 80.0)) / 1000.0
                        mid = obj_mocap_map.get(oid)
                        if mid is not None:
                            z_m = 0.0 if oid in ("yellow_duck", "charging_dock") else (h_m * 0.5)
                            data.mocap_pos[mid] = [nx, ny, z_m]
                            last_mocap_xy[oid] = (nx, ny)
                        elif extra_idx <= 4:
                            slot_name = f"spawn_{extra_idx}"
                            extra_idx += 1
                            smid = obj_mocap_map.get(slot_name)
                            sgid = spawn_geom_map.get(slot_name, -1)
                            if smid is not None:
                                data.mocap_pos[smid] = [nx, ny, h_m * 0.5]
                                last_mocap_xy[slot_name] = (nx, ny)
                            if sgid >= 0 and "color" in obj:
                                col_hex = str(obj["color"]).lstrip("#")
                                if len(col_hex) == 6:
                                    r = int(col_hex[0:2], 16) / 255.0
                                    g = int(col_hex[2:4], 16) / 255.0
                                    b = int(col_hex[4:6], 16) / 255.0
                                    model.geom_rgba[sgid] = [r, g, b, 1.0]

                    for s_idx in range(extra_idx, 5):
                        slot_name = f"spawn_{s_idx}"
                        smid = obj_mocap_map.get(slot_name)
                        if smid is not None:
                            data.mocap_pos[smid] = [0.0, 0.0, -1.5]

            # Smooth 60 FPS interpolation of robot pose (hinge qpos is radians; wrap yaw the short way)
            tx = float(target_pose.get("x_mm", 0.0)) / 1000.0
            ty = float(target_pose.get("y_mm", 0.0)) / 1000.0
            tyaw = math.radians(float(target_pose.get("heading_deg", 0.0)))
            ttilt = math.radians(float(target_pose.get("head_tilt_deg", 0.0)))

            prev_x, prev_y = float(data.qpos[qpos_bx]), float(data.qpos[qpos_by])
            data.qpos[qpos_bx] += (tx - data.qpos[qpos_bx]) * 0.25
            data.qpos[qpos_by] += (ty - data.qpos[qpos_by]) * 0.25
            yaw_err = (tyaw - data.qpos[qpos_byaw] + math.pi) % (2.0 * math.pi) - math.pi
            data.qpos[qpos_byaw] += yaw_err * 0.25
            data.qpos[qpos_tilt] += (ttilt - data.qpos[qpos_tilt]) * 0.25

            wheel_travel += math.hypot(data.qpos[qpos_bx] - prev_x, data.qpos[qpos_by] - prev_y)
            spin_rad = wheel_travel / 0.060
            for idx, wq in enumerate(wheel_qpos):
                data.qpos[wq] = spin_rad + (data.qpos[qpos_byaw] * (1.2 if idx % 2 == 0 else -1.2))

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.016)


if __name__ == "__main__":
    main()
