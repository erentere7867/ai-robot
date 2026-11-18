"""Renders a 3D home for the robot with Habitat-Sim; runs in Habitat's own Python (3.9), apart from the robot code.

The robot side (robot/sim/home3d.py) starts this file with the Habitat environment's Python and talks to it over a
local socket. It answers two requests:

    {"op": "info"}    the home as the 2D world needs it: size, a drivable-floor grid, the objects, a start pose
    {"op": "render", "pose": (x, y, yaw), "cameras": [...], "rgb": bool}
                      what the robot's cameras see from that pose: RGB pictures (if asked) and, per camera, which
                      objects are in view with their pixel boxes and pixel counts (from a semantic camera)

Coordinates on the robot side are the 2D world's: metres on the floor plane, centred on the home, x forward at yaw 0,
yaw anticlockwise in radians. Habitat is y-up with -z forward; the mapping is X = -z, Y = -x (then centred).
Keep this file compatible with Python 3.9.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from multiprocessing.connection import Listener

import habitat_sim
import magnum as mn
import numpy as np

RES = 0.05  # grid cell, m (as the 2D world)
CAMERA_Z = 0.22
CAMERAS = {"front_camera": 0.0, "rear_left_camera": 2 * math.pi / 3, "rear_right_camera": -2 * math.pi / 3}
WIDTH, HEIGHT, HFOV = 512, 288, 120.0
SS = 2  # colour cameras render at SS x the size, then average down: anti-aliasing at no cost to the model
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


class Home:
    def __init__(self, data: str, scene: str) -> None:
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_dataset_config_file = os.path.join(data, "replica_cad", "replicaCAD.scene_dataset_config.json")
        cfg.scene_id = scene
        cfg.enable_physics = True
        cfg.gpu_device_id = -1  # any GPU by EGL device, not only NVIDIA (CUDA) ones
        cfg.override_scene_light_defaults = True
        cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
        sensors = []
        for name, yaw in CAMERAS.items():
            for kind, suffix in ((habitat_sim.SensorType.COLOR, ""), (habitat_sim.SensorType.SEMANTIC, ":sem")):
                s = habitat_sim.CameraSensorSpec()
                ss = SS if kind == habitat_sim.SensorType.COLOR else 1
                s.uuid, s.sensor_type, s.resolution, s.hfov = name + suffix, kind, [HEIGHT * ss, WIDTH * ss], HFOV
                s.position, s.orientation = [0.0, CAMERA_Z, 0.0], [0.0, yaw, 0.0]
                sensors.append(s)
        agent = habitat_sim.agent.AgentConfiguration(sensor_specifications=sensors)
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
        self.objects = self._tag_objects()
        self._floor()

    def _tag_objects(self) -> list[dict]:
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
            objects.append({"id": k, "label": label(handle), "hab": (float(centre.x), float(centre.y),
                                                                     float(centre.z)),
                            "radius": float(max(0.05, min(1.5, max(half.x, half.z)))),
                            "top": float(centre.y + half.y)})
        return objects

    def _navmesh(self, radius: float, height: float) -> habitat_sim.nav.PathFinder:
        ns = habitat_sim.NavMeshSettings()
        ns.set_defaults()
        ns.agent_radius, ns.agent_height, ns.agent_max_climb = radius, height, 0.03  # wheels climb 3 cm
        ns.include_static_objects = True
        pf = habitat_sim.nav.PathFinder()
        self.sim.recompute_navmesh(pf, ns)
        return pf

    def _floor(self) -> None:
        """The drivable ground floor for a wheeled robot (0.3 m tall, climbs 3 cm) as the 2D world's grid."""
        open_floor = self._navmesh(0.3, 1.6)  # where a person could stand: for a start that is not under a table
        pf = self._navmesh(0.02, 0.3)  # the 2D world adds the robot's own radius when it plans
        self.sim.pathfinder = pf
        lo, hi = pf.get_bounds()
        pf.seed(7)
        points = [pf.get_random_navigable_point() for _ in range(2000)]
        self.floor_y = float(np.percentile([p.y for p in points], 20))
        ground = [p for p in points if abs(p.y - self.floor_y) < 0.1]
        islands = [pf.get_island(p) for p in ground]
        self.island = max(set(islands), key=islands.count)
        margin = 0.5
        self.cx, self.cy = -(lo.z + hi.z) / 2, -(lo.x + hi.x) / 2
        self.width, self.height = (hi.z - lo.z) + 2 * margin, (hi.x - lo.x) + 2 * margin
        nx, ny = round(self.width / RES), round(self.height / RES)
        free = np.zeros((nx, ny), bool)
        for i in range(nx):
            for j in range(ny):
                x, y = i * RES + RES / 2 - self.width / 2, j * RES + RES / 2 - self.height / 2
                h = self.to_hab(x, y)
                snap = pf.snap_point(h, self.island)
                if (not math.isnan(snap.x) and abs(snap.x - h.x) < 0.6 * RES and abs(snap.z - h.z) < 0.6 * RES
                        and abs(snap.y - self.floor_y) < 0.1):
                    free[i, j] = True
        self.occupied = ~free
        # start: the most open spot on the ground floor, in the open (not under furniture)
        open_floor.seed(7)
        spots = [p for p in (open_floor.get_random_navigable_point() for _ in range(500))
                 if abs(p.y - self.floor_y) < 0.1 and pf.get_island(pf.snap_point(p)) == self.island]
        best = max(spots, key=lambda p: open_floor.distance_to_closest_obstacle(p, 3.0))
        self.start = (*self.from_hab(best), 0.0)

    def to_hab(self, x: float, y: float) -> mn.Vector3:
        return mn.Vector3(-(y + self.cy), self.floor_y, -(x + self.cx))

    def from_hab(self, p: mn.Vector3) -> tuple[float, float]:
        return float(-p.z - self.cx), float(-p.x - self.cy)

    def info(self) -> dict:
        objects = []
        for o in self.objects:
            x, y = self.from_hab(mn.Vector3(*o["hab"]))
            objects.append({"id": o["id"], "label": o["label"], "x": x, "y": y, "radius": o["radius"],
                            "height": round(o["top"] - self.floor_y, 2)})
        return {"width": self.width, "height": self.height, "res": RES, "shape": self.occupied.shape,
                "occupied": np.packbits(self.occupied).tobytes(), "objects": objects, "start": self.start,
                "cameras": list(CAMERAS), "image": (HEIGHT, WIDTH), "ss": SS}

    def pixels(self, pose: tuple[float, float, float], ids: list[int], most: int = 400) -> dict:
        """Up to `most` RGB pixels of each object in `ids` as the cameras see it from `pose` (for its colour)."""
        out: dict = {}
        reply = self.render(pose, list(CAMERAS), rgb=True)
        for cam in CAMERAS:
            sensor = self.sim._sensors[cam + ":sem"]
            sensor.draw_observation()
            sem = np.asarray(sensor.get_observation())
            img = np.frombuffer(reply["rgb"][cam], np.uint8).reshape(HEIGHT, WIDTH, 3)
            for k in ids:
                px = img[sem == k]
                if len(px):
                    out[k] = np.concatenate([out[k], px]) if k in out else px
        rng = np.random.default_rng(0)
        return {k: (px if len(px) <= most else px[rng.choice(len(px), most, replace=False)]).tobytes()
                for k, px in out.items()}

    def render(self, pose: tuple[float, float, float], cameras: list[str], rgb: bool, full: bool = False) -> dict:
        """Pictures are WIDTH x HEIGHT, drawn at SS times the size and averaged down (anti-aliased); `full` returns
        the SS-times-larger picture itself. Boxes are always in WIDTH x HEIGHT pixels."""
        x, y, yaw = pose
        agent = self.sim.get_agent(0)
        state = agent.get_state()
        state.position = self.to_hab(x, y)
        state.rotation = habitat_sim.utils.common.quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
        agent.set_state(state)
        out: dict = {"rgb": {}, "boxes": {}}
        for cam in cameras:
            sensor = self.sim._sensors[cam + ":sem"]
            sensor.draw_observation()
            sem = np.asarray(sensor.get_observation())
            boxes = {}
            ids, counts = np.unique(sem, return_counts=True)
            for k, n in zip(ids.tolist(), counts.tolist()):
                if k == 0:
                    continue
                rows, cols = np.nonzero(sem == k)
                boxes[k] = (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()), int(n))
            out["boxes"][cam] = boxes
            if rgb:
                sensor = self.sim._sensors[cam]
                sensor.draw_observation()
                big = np.asarray(sensor.get_observation())[:, :, :3].astype(np.float32)
                if not full:  # each output pixel averages SS x SS rendered ones: smooth edges, no jaggies
                    big = big.reshape(HEIGHT, SS, WIDTH, SS, 3).mean(axis=(1, 3))
                out["rgb"][cam] = np.ascontiguousarray(np.clip(big + 0.5, 0, 255).astype(np.uint8)).tobytes()
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument("--data", default=os.path.expanduser("~/habitat-data"))
    args = parser.parse_args()
    home = Home(args.data, args.scene)
    with Listener(("127.0.0.1", args.port), authkey=b"sprocket") as listener:
        print("ready", flush=True)
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        req = conn.recv()
                    except EOFError:
                        break
                    if req.get("op") == "quit":
                        return
                    if req.get("op") == "info":
                        conn.send(home.info())
                    elif req.get("op") == "pixels":
                        conn.send(home.pixels(tuple(req["pose"]), [int(k) for k in req["ids"]]))
                    elif req.get("op") == "render":
                        conn.send(home.render(tuple(req["pose"]), list(req["cameras"]), bool(req.get("rgb")),
                                              bool(req.get("full"))))


if __name__ == "__main__":
    main()
