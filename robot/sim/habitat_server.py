"""The world and Sprocket's body in Habitat-Sim: what the real robot's hardware would give, and nothing more.

Runs in Habitat's own Python (3.9), apart from the robot code; robot/sim/habitat_ros.py drives it over a local
socket and publishes what it returns on ROS, as the real robot's drivers would. Requests:

    {"op": "reset", "pose": (x, y, yaw)}       put the body there (snapped onto the floor)
    {"op": "step", "v": m/s, "w": rad/s, "dt": s}
                                               drive the wheels for dt; the body stops where it would touch something
                                               (no sliding along walls). Returns the true pose and whether it bumped.
    {"op": "camera", "name": "front_camera"}  one colour picture as the rig's camera takes it (robot/ros/model.py)
    {"op": "tof"}                              the front and rear 8x8 ToF grids: metres, inf where nothing is in range
    {"op": "info"}                             the home's truth, for the judge only: floor grid, objects, start pose
    {"op": "visible", "pose": ..., "cameras": [...]}
                                               the judge's view: pixels of each object in each camera (semantic)

Coordinates are the house frame: metres on the floor, centred on the home, x forward at yaw 0, yaw anticlockwise.
Habitat is y-up with -z forward; the mapping is X = -z, Y = -x (then centred). Keep this file Python 3.9.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import queue
import re
import sys
import threading
import xml.etree.ElementTree as ET
from multiprocessing.connection import Listener

import habitat_sim
import magnum as mn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from robot.ros.model import BODY_R, CAMERA_R, CAMERA_Z, CAMERAS, CLEARANCE, TOF  # noqa: E402 - numbers only

RES = 0.05  # floor grid cell for the judge, m
CLIMB = 0.03  # wheels roll over a rug or a threshold this high
TOF_RENDER = 32  # each ToF zone is the median of a 4 x 4 patch of a 32 x 32 depth picture
JUDGE_WIDTH = 640  # the judge's semantic pictures: which objects are in view, not what the robot gets
NAMES = {  # ReplicaCAD handles -> words people use
    "tv_screen": "tv", "tv_object": "tv", "tvstand": "tv stand", "indoor_plant": "plant", "remote-control": "remote",
    "setupbox": "set-top box", "kitchenCupboard": "kitchen cupboard", "chestOfDrawers": "chest of drawers",
    "door2": "door", "kitchen_counter": "kitchen counter", "small_appliance": "kitchen appliance",
    "monitor_stand": "monitor stand", "choppingboard": "chopping board", "knifeblock": "knife block",
    "sponge_dish": "sponge dish", "wall_cabinet": "wall cabinet", "kitchen_utensil": "kitchen utensil",
}


def label(handle: str) -> str:
    base = handle.split(":")[0].replace("frl_apartment_", "").rstrip("_")
    base = re.sub(r"_\d+$", "", base)
    return NAMES.get(base, base.replace("_", " "))


def _hab_offset(forward: float, left: float, up: float) -> list:
    """A point on the body (robot frame) in Habitat's agent frame: x right, y up, -z forward."""
    return [-left, up, -forward]


def _urdf_visual_matrix(vis: ET.Element, mesh: ET.Element) -> np.ndarray:
    """4x4 local transform for a URDF <visual> element (origin xyz/rpy and mesh scale)."""
    orig = vis.find("origin")
    xyz = [float(v) for v in orig.attrib.get("xyz", "0 0 0").split()] if orig is not None else [0.0, 0.0, 0.0]
    rpy = [float(v) for v in orig.attrib.get("rpy", "0 0 0").split()] if orig is not None else [0.0, 0.0, 0.0]
    sc = [float(v) for v in mesh.attrib.get("scale", "1 1 1").replace(",", " ").split()]
    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = (rz @ ry @ rx) * np.array(sc, float)
    mat[:3, 3] = xyz
    return mat


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    def chain(ordered):
        edge = []
        for x, y in ordered:
            while len(edge) >= 2:
                (ax, ay), (bx, by) = edge[-2:]
                if (bx - ax) * (y - ay) > (by - ay) * (x - ax):
                    break
                edge.pop()
            edge.append((x, y))
        return edge

    ordered = sorted(set(points))
    return chain(ordered)[:-1] + chain(reversed(ordered))[:-1] if len(ordered) > 1 else ordered


class Home:
    def __init__(self, data: str, scene: str) -> None:
        self.data = data
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_dataset_config_file = os.path.join(data, "replica_cad", "replicaCAD.scene_dataset_config.json")
        cfg.scene_id = scene
        cfg.enable_physics = True
        cfg.gpu_device_id = -1  # any GPU by EGL device, not only NVIDIA (CUDA) ones
        cfg.override_scene_light_defaults = True
        cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
        sensors = []
        for name, (yaw, hfov, width, height, _rate) in CAMERAS.items():
            where = _hab_offset(CAMERA_R * math.cos(yaw), CAMERA_R * math.sin(yaw), CAMERA_Z)
            judge = (round(JUDGE_WIDTH * height / width), JUDGE_WIDTH)
            for kind, suffix, size in ((habitat_sim.SensorType.COLOR, "", (height, width)),
                                       (habitat_sim.SensorType.SEMANTIC, ":sem", judge),
                                       (habitat_sim.SensorType.DEPTH, ":depth", judge)):
                s = habitat_sim.CameraSensorSpec()
                s.uuid, s.sensor_type, s.resolution, s.hfov = name + suffix, kind, list(size), math.degrees(hfov)
                s.position, s.orientation = where, [0.0, yaw, 0.0]
                sensors.append(s)
        for side, (x, yaw, z, pitch, hfov, _max_range) in TOF.items():
            s = habitat_sim.CameraSensorSpec()
            s.uuid, s.sensor_type, s.resolution, s.hfov = f"{side}_tof", habitat_sim.SensorType.DEPTH, \
                [TOF_RENDER, TOF_RENDER], math.degrees(hfov)
            s.position = _hab_offset(x, 0.0, z)
            s.orientation = [-pitch, yaw, 0.0]  # tipped down at the floor ahead (behind, for the rear one)
            sensors.append(s)
        agent = habitat_sim.agent.AgentConfiguration(sensor_specifications=sensors)
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
        # ReplicaCAD loads nearly all furniture as dynamic (102 of 113 objects, and every articulated one: fridge,
        # counters, cabinets, doors), and the navigation mesh leaves dynamic objects out: the body would drive
        # through them. Nothing here needs to move, so all of it is solid and fixed where it stands.
        static = habitat_sim.physics.MotionType.STATIC
        for manager in (self.sim.get_rigid_object_manager(), self.sim.get_articulated_object_manager()):
            for handle in manager.get_object_handles():
                manager.get_object_by_handle(handle).motion_type = static
        self.objects = self._tag_objects()
        self._floor()
        self.pose = self.start

    # ---- the home ----------------------------------------------------------------------------------------------
    def _tag_objects(self) -> list:
        """Give every object an id the semantic camera shows; furniture with doors counts as one object."""
        found = []
        rom, aom = self.sim.get_rigid_object_manager(), self.sim.get_articulated_object_manager()
        for handle in sorted(rom.get_object_handles()):
            found.append((handle, rom.get_object_by_handle(handle), False))
        for handle in sorted(aom.get_object_handles()):
            found.append((handle, aom.get_object_by_handle(handle), True))
        objects = []
        for k, (handle, obj, articulated) in enumerate(found, 1):
            if articulated:  # the id goes on the meshes that are drawn, link by link
                for node in obj.visual_scene_nodes:
                    node.semantic_id = k
                box = obj.aabb
            else:
                obj.semantic_id = k
                box = obj.root_scene_node.cumulative_bb
            centre = obj.transformation.transform_point((box.min + box.max) / 2)  # boxes are in the object's frame
            half = (box.max - box.min) / 2
            corners = [obj.transformation.transform_point(mn.Vector3(*p))
                       for p in itertools.product(*zip(box.min, box.max))]
            objects.append({"id": k, "label": label(handle), "hab": (float(centre.x), float(centre.y),
                                                                     float(centre.z)),
                            "corners": corners,
                            "radius": float(max(0.05, min(1.5, max(half.x, half.z)))),
                            "top": float(centre.y + half.y)})
        return objects

    def _navmesh(self, radius: float, height: float) -> habitat_sim.nav.PathFinder:
        ns = habitat_sim.NavMeshSettings()
        ns.set_defaults()
        ns.agent_radius, ns.agent_height, ns.agent_max_climb = radius, height, CLIMB
        # on the default 5 cm grid the mesh rounds obstacles outward: the body's centre was kept 0.21 m (median) from
        # the nearest surface, not its 0.15 m radius, and it "touched" things 6 cm early; on 2.5 cm, about right
        ns.cell_size = 0.025
        ns.include_static_objects = True
        pf = habitat_sim.nav.PathFinder()
        self.sim.recompute_navmesh(pf, ns)
        return pf

    def _floor(self) -> None:
        """The drivable ground floor: where the body fits (its radius, its height, what its wheels climb). The judge
        gets it as a grid; the body's own moves are checked against it."""
        open_floor = self._navmesh(0.3, 1.6)  # where a person could stand: for a start that is not under a table
        self.body = self._navmesh(BODY_R, CLEARANCE)
        pf = self.body
        lo, hi = pf.get_bounds()
        pf.seed(7)
        points = [pf.get_random_navigable_point() for _ in range(2000)]
        self.floor_y = float(np.percentile([p.y for p in points], 20))
        ground = [p for p in points if abs(p.y - self.floor_y) < 0.1]
        islands = [pf.get_island(p) for p in ground]
        self.island = max(set(islands), key=islands.count)
        # The navmesh floats a little above the floor; the body stands on the floor itself, so that its sensors are
        # at their real heights above it (found by dropping a ray from navmesh points onto what is under them).
        hits = []
        for p in ground[:200]:
            ray = habitat_sim.geo.Ray(mn.Vector3(p.x, p.y + 0.3, p.z), mn.Vector3(0.0, -1.0, 0.0))
            result = self.sim.cast_ray(ray, 1.0)
            if result.has_hits():
                hits.append(float(result.hits[0].point.y))
        self.ground_y = float(np.median(hits)) if hits else self.floor_y
        margin = 0.5
        self.cx, self.cy = -(lo.z + hi.z) / 2, -(lo.x + hi.x) / 2
        self.width, self.height = (hi.z - lo.z) + 2 * margin, (hi.x - lo.x) + 2 * margin
        nx, ny = round(self.width / RES), round(self.height / RES)
        free = np.zeros((nx, ny), bool)  # where the robot's centre can be
        for i in range(nx):
            for j in range(ny):
                x, y = i * RES + RES / 2 - self.width / 2, j * RES + RES / 2 - self.height / 2
                h = self.to_hab(x, y)
                snap = pf.snap_point(h, self.island)
                if (not math.isnan(snap.x) and abs(snap.x - h.x) < 0.6 * RES and abs(snap.z - h.z) < 0.6 * RES
                        and abs(snap.y - self.floor_y) < 0.1):
                    free[i, j] = True
        self.free = free
        open_floor.seed(7)
        spots = [p for p in (open_floor.get_random_navigable_point() for _ in range(500))
                 if abs(p.y - self.floor_y) < 0.1 and pf.get_island(pf.snap_point(p)) == self.island]
        best = max(spots, key=lambda p: open_floor.distance_to_closest_obstacle(p, 3.0))
        self.start = (*self.from_hab(best), 0.0)

    def to_hab(self, x: float, y: float) -> mn.Vector3:
        return mn.Vector3(-(y + self.cy), self.floor_y, -(x + self.cx))

    def from_hab(self, p: mn.Vector3) -> tuple:
        return float(-p.z - self.cx), float(-p.x - self.cy)

    def info(self) -> dict:
        objects = []
        for o in self.objects:
            x, y = self.from_hab(mn.Vector3(*o["hab"]))
            objects.append({"id": o["id"], "label": o["label"], "x": x, "y": y, "radius": o["radius"],
                            "footprint": convex_hull([self.from_hab(p) for p in o["corners"]]),
                            "height": round(o["top"] - self.ground_y, 2)})
        return {"width": self.width, "height": self.height, "res": RES, "shape": self.free.shape,
                "navmesh_above_floor": round(self.floor_y - self.ground_y, 3),
                "free": np.packbits(self.free).tobytes(), "objects": objects, "start": self.start}

    def scene_3d(self) -> dict:
        """Render assets and world transforms for the live 3D viewer (viewer-only, never handed to the robot)."""
        rcad = os.path.join(self.data, "replica_cad")

        def resolve(handle: str) -> str:
            for base in (os.path.join(rcad, "configs"), rcad):
                candidate = os.path.normpath(os.path.join(base, handle))
                if os.path.exists(candidate):
                    return candidate
            return os.path.normpath(os.path.join(rcad, handle))

        instances = []
        rom = self.sim.get_rigid_object_manager()
        for handle in sorted(rom.get_object_handles()):
            obj = rom.get_object_by_handle(handle)
            path = resolve(obj.creation_attributes.render_asset_handle)
            if os.path.exists(path):
                mat = np.array(obj.transformation, dtype=float).reshape(-1).tolist()
                instances.append({"path": path, "mat": mat})

        aom = self.sim.get_articulated_object_manager()
        for handle in sorted(aom.get_object_handles()):
            ao = aom.get_object_by_handle(handle)
            urdf_path = os.path.normpath(os.path.join(rcad, "urdf", ao.creation_attributes.urdf_filepath))
            if not os.path.exists(urdf_path):
                continue
            scale = 1.0
            if len(ao.visual_scene_nodes) >= 2:
                scale = float(np.linalg.norm(np.array(ao.visual_scene_nodes[1].absolute_transformation())[:3, 0]))
            scale_mat = np.diag([scale, scale, scale, 1.0])
            links = {lk.attrib["name"]: lk for lk in ET.parse(urdf_path).getroot().findall("link")}
            for lid in [-1, *ao.get_link_ids()]:
                lk = links.get(ao.get_link_name(lid))
                if lk is None:
                    continue
                link_mat = np.array(ao.get_link_scene_node(lid).absolute_transformation(), dtype=float)
                for vis in lk.findall("visual"):
                    mesh = vis.find("geometry/mesh")
                    if mesh is None or "filename" not in mesh.attrib:
                        continue
                    mesh_path = os.path.normpath(os.path.join(os.path.dirname(urdf_path), mesh.attrib["filename"]))
                    if not os.path.exists(mesh_path):
                        continue
                    mat = link_mat @ scale_mat @ _urdf_visual_matrix(vis, mesh)
                    instances.append({"path": mesh_path, "mat": mat.reshape(-1).tolist()})

        stage = resolve(self.sim.get_stage_initialization_template().render_asset_handle)
        return {"cx": self.cx, "cy": self.cy, "ground_y": self.ground_y, "stage": stage, "instances": instances}

    # ---- the body ----------------------------------------------------------------------------------------------
    def _place(self, pose: tuple) -> None:
        x, y, yaw = pose
        agent = self.sim.get_agent(0)
        state = agent.get_state()
        h = self.to_hab(x, y)
        state.position = mn.Vector3(h.x, self.ground_y, h.z)
        state.rotation = habitat_sim.utils.common.quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
        agent.set_state(state)

    def reset(self, pose: tuple) -> tuple:
        snap = self.body.snap_point(self.to_hab(pose[0], pose[1]), self.island)
        self.pose = (*self.from_hab(snap), pose[2])
        self._place(self.pose)
        return self.pose

    def step(self, v: float, w: float, dt: float) -> dict:
        """Wheels at v, w for dt. A move that would put the body into something stops at the contact."""
        x, y, yaw = self.pose
        mid = yaw + w * dt / 2
        want = (x + v * dt * math.cos(mid), y + v * dt * math.sin(mid))
        bumped = False
        if v:
            got = self.body.try_step_no_sliding(self.to_hab(x, y), self.to_hab(*want))
            gx, gy = self.from_hab(got)
            bumped = math.dist((gx, gy), want) > 0.002
            x, y = gx, gy
        self.pose = (x, y, math.atan2(math.sin(yaw + w * dt), math.cos(yaw + w * dt)))
        self._place(self.pose)
        return {"pose": self.pose, "bumped": bumped}

    # ---- the sensors -------------------------------------------------------------------------------------------
    def camera(self, name: str) -> bytes:
        sensor = self.sim._sensors[name]
        sensor.draw_observation()
        return np.ascontiguousarray(np.asarray(sensor.get_observation())[:, :, :3]).tobytes()

    def tof(self) -> dict:
        """Each 8x8 zone is the median of its patch (a zone reports one distance for the cone it covers); distances
        are along the sensor's axis, as the depth picture gives them; beyond the sensor's range there is no reading."""
        out = {}
        k = TOF_RENDER // 8
        for side, (*_, max_range) in TOF.items():
            sensor = self.sim._sensors[f"{side}_tof"]
            sensor.draw_observation()
            depth = np.asarray(sensor.get_observation(), dtype=np.float32)
            zones = np.median(depth.reshape(8, k, 8, k).transpose(0, 2, 1, 3).reshape(8, 8, k * k), axis=2)
            zones[(zones <= 0.0) | (zones > max_range)] = np.inf
            out[side] = zones.astype(np.float32).tobytes()
        return out

    def true_depth(self, pose: tuple, camera: str) -> bytes:
        """For the judge: the true depth (along the camera's axis) a JUDGE_WIDTH-wide copy of `camera` would see."""
        self._place(pose)
        sensor = self.sim._sensors[camera + ":depth"]
        sensor.draw_observation()
        out = np.ascontiguousarray(np.asarray(sensor.get_observation(), dtype=np.float32)).tobytes()
        self._place(self.pose)
        return out

    def visible(self, pose: tuple, cameras: list) -> dict:
        """For the judge: per camera, each object's pixel count in view (of a JUDGE_WIDTH-wide picture)."""
        self._place(pose)
        out = {}
        for cam in cameras:
            sensor = self.sim._sensors[cam + ":sem"]
            sensor.draw_observation()
            ids, counts = np.unique(np.asarray(sensor.get_observation()), return_counts=True)
            out[cam] = {int(k): int(n) for k, n in zip(ids.tolist(), counts.tolist()) if k}
        self._place(self.pose)
        return out


def answer(home: Home, req: dict):  # noqa: ANN201
    op = req.get("op")
    if op == "info":
        return home.info()
    if op == "scene_3d":
        return home.scene_3d()
    if op == "reset":
        return home.reset(tuple(req["pose"]))
    if op == "step":
        return home.step(float(req["v"]), float(req["w"]), float(req["dt"]))
    if op == "camera":
        return home.camera(str(req["name"]))
    if op == "tof":
        return home.tof()
    if op == "true_depth":
        return home.true_depth(tuple(req["pose"]), str(req["camera"]))
    if op == "camera_at":  # the judge's copy of what a camera sees from a pose, without moving the body
        home._place(tuple(req["pose"]))
        picture = home.camera(str(req["name"]))
        home._place(home.pose)
        return picture
    if op == "visible":
        return home.visible(tuple(req["pose"]), list(req["cameras"]))
    return {"error": f"unknown op {op!r}"}


def listen(conn, requests: queue.Queue) -> None:  # noqa: ANN001
    """One client's requests, passed to the main thread (the only one that may draw: the GL context is its own)."""
    with conn:
        while True:
            try:
                req = conn.recv()
            except (EOFError, OSError):
                return
            done, slot = threading.Event(), []
            requests.put((req, slot, done))
            done.wait()
            if req.get("op") == "quit":
                return
            conn.send(slot[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument("--data", default=os.path.expanduser("~/habitat-data"))
    args = parser.parse_args()
    home = Home(args.data, args.scene)
    requests: queue.Queue = queue.Queue()
    listener = Listener(("127.0.0.1", args.port), authkey=b"sprocket")

    def accept() -> None:
        while True:
            threading.Thread(target=listen, args=(listener.accept(), requests), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    print("ready", flush=True)
    while True:
        req, slot, done = requests.get()
        if req.get("op") == "quit":
            done.set()
            return
        try:
            slot.append(answer(home, req))
        except Exception as exc:  # noqa: BLE001 - a bad request must not take the world down
            slot.append({"error": f"{type(exc).__name__}: {exc}"})
        done.set()


if __name__ == "__main__":
    main()
