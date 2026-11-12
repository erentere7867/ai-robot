"""First-time mapping: the robot explores a home it has never seen while slam_toolbox builds the map from its lidar,
then saves the map it will localise in from then on ("slam_toolbox maps once, then localises", docs/architecture.md).

Frontier exploration (Yamauchi, 1997): a frontier is where mapped free floor meets unmapped space. The robot drives to
the nearest worthwhile frontier, its lidar fills in what it sees from there, and it repeats until no frontier is left
or the time is up. A frontier it could not reach is not tried again. The map's frame starts where the robot started
(its dock); the saved map is shifted by the dock's place, so it lines up with the house's frame.
"""

from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

FREE_BELOW = 50  # occupancy-grid values under this are free floor; -1 is unknown


def _box_sum(mask: np.ndarray, half: int) -> np.ndarray:
    """For every cell, how many cells of `mask` lie in the square of side 2*half+1 around it."""
    padded = np.pad(mask.astype(np.int32), half)
    total = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.int32)
    total[1:, 1:] = padded.cumsum(0).cumsum(1)
    k = 2 * half + 1
    return total[k:, k:] - total[:-k, k:] - total[k:, :-k] + total[:-k, :-k]


def frontiers(data: np.ndarray, resolution: float, origin: tuple[float, float], reach: float = 1.0,
              clearance: float = 0.25, min_gain: int = 20) -> list[tuple[float, float, int]]:
    """Places worth driving to, as (x, y, gain): free cells that touch unknown space, at least `clearance` from any
    obstacle (the robot must fit there), each with the number of unknown cells within `reach`. A frontier's cells, not
    its middle: early in a map the frontier is a ring around the robot, and a ring's middle is where the robot stands."""
    free = (data >= 0) & (data < FREE_BELOW)
    unknown = data < 0
    touches = np.zeros_like(free)
    touches[1:, :] |= unknown[:-1, :]
    touches[:-1, :] |= unknown[1:, :]
    touches[:, 1:] |= unknown[:, :-1]
    touches[:, :-1] |= unknown[:, 1:]
    near_wall = _box_sum(data >= FREE_BELOW, max(1, round(clearance / resolution))) > 0
    gain = _box_sum(unknown, max(1, round(reach / resolution)))
    cells = free & touches & ~near_wall & (gain >= min_gain)
    rows, cols = np.nonzero(cells)
    step = max(1, len(rows) // 400)  # a few hundred candidates are plenty
    return [(origin[0] + (c + 0.5) * resolution, origin[1] + (r + 0.5) * resolution, int(gain[r, c]))
            for r, c in zip(rows[::step], cols[::step], strict=True)]


class Explorer:
    def __init__(self, robot: Any) -> None:
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self.robot = robot
        self.grid: tuple[np.ndarray, float, tuple[float, float]] | None = None
        self.updates = 0  # maps received: slam_toolbox publishes one every few seconds
        self.unreachable: list[tuple[float, float]] = []
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        robot.node.create_subscription(OccupancyGrid, "/map", self._map, qos)

    def _map(self, msg: Any) -> None:
        info = msg.info
        data = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
        self.grid = (data, info.resolution, (info.origin.position.x, info.origin.position.y))
        self.updates += 1

    def _next_map(self, timeout: float = 12.0) -> None:
        seen, end = self.updates, time.monotonic() + timeout
        while self.updates == seen and time.monotonic() < end:
            self.robot._step(0.0, 0.0)

    def explore(self, budget: float = 480.0) -> dict[str, float]:
        """Drive to frontiers until none is left or `budget` seconds have passed. "None left" is believed only after
        two more maps: a map can lag what the lidar saw on the way to the last frontier (the first try stopped at 9%
        of the floor, on a map made before the robot had moved)."""
        started, goals, left, quiet = time.monotonic(), 0, -1, 0
        self.robot._rotate(2 * math.pi)  # slam_toolbox adds a scan only after a turn or a move: look all round first
        while time.monotonic() - started < budget:
            if self.grid is None:
                time.sleep(0.5)
                continue
            self.robot._sync()
            here = (self.robot.x, self.robot.y)
            options = [f for f in frontiers(*self.grid) if math.dist(f[:2], here) > 0.4
                       and all(math.dist(f[:2], u) > 0.5 for u in self.unreachable)]
            left = len(options)
            if not options:
                quiet += 1
                if quiet > 2:
                    break
                self._next_map()
                continue
            quiet = 0
            x, y, _ = max(options, key=lambda f: f[2] * math.exp(-math.dist(f[:2], here) / 3.0))  # much unseen, near
            print(f"exploring: to ({x:.1f}, {y:.1f}), {left} frontiers open", flush=True)
            try:
                self.robot._act("navigate_to_pose", pose=self.robot._pose_msg(x, y, math.atan2(y - here[1],
                                                                                              x - here[0])))
                goals += 1
            except Exception as exc:  # noqa: BLE001 - Nav2 could not get there: try another frontier
                print(f"exploring: could not reach ({x:.1f}, {y:.1f}): {exc}", flush=True)
                self.unreachable.append((x, y))
        return {"seconds": time.monotonic() - started, "goals": goals, "unreachable": len(self.unreachable),
                "frontiers left": max(left, 0)}

    def save(self, path: Path, dock: tuple[float, float]) -> Path:
        """Save the map as path.yaml and path.pgm, in the house's frame (`dock`: where the robot started in it)."""
        import re

        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", str(path), "--ros-args", "-p",
                        "use_sim_time:=true"], check=True, capture_output=True, timeout=60)
        yaml = path.with_suffix(".yaml")
        text = yaml.read_text()
        x, y, yaw = (float(v) for v in re.search(r"origin: \[([^\]]+)\]", text).group(1).split(","))
        yaml.write_text(re.sub(r"origin: \[[^\]]+\]", f"origin: [{x + dock[0]:.4f}, {y + dock[1]:.4f}, {yaw:.4f}]",
                               text))
        return yaml


def load_grid(yaml: Path, shape: tuple[int, int], size: tuple[float, float]) -> np.ndarray:
    """A saved map as the planner's occupancy grid: indexed [x, y], `shape` cells over a `size` (width, height) area
    centred on the house frame's origin. Unknown space counts as blocked: the robot has not seen it is free."""
    import re

    text = yaml.read_text()
    resolution = float(re.search(r"resolution: ([\d.]+)", text).group(1))
    ox, oy, _ = (float(v) for v in re.search(r"origin: \[([^\]]+)\]", text).group(1).split(","))
    image = _pgm(yaml.with_name(re.search(r"image: (.+)", text).group(1).strip()))
    nx, ny = shape
    xs = -size[0] / 2 + (np.arange(nx) + 0.5) * size[0] / nx
    ys = -size[1] / 2 + (np.arange(ny) + 0.5) * size[1] / ny
    cols = ((xs - ox) / resolution).astype(int)
    rows = image.shape[0] - 1 - ((ys - oy) / resolution).astype(int)
    inside = (cols[:, None] >= 0) & (cols[:, None] < image.shape[1]) & (rows[None, :] >= 0) & (rows[None, :] < image.shape[0])
    values = image[np.clip(rows, 0, image.shape[0] - 1)[None, :], np.clip(cols, 0, image.shape[1] - 1)[:, None]]
    return ~(inside & (values > 250))


def coverage(yaml: Path, truth: Any) -> float:
    """Share of the floor the simulator (a SimRobot of the true scene) knows to be free that the saved map shows as
    free. The simulator's grid is indexed [x, y]; a map image's first row is its largest y."""
    import re

    text = yaml.read_text()
    resolution = float(re.search(r"resolution: ([\d.]+)", text).group(1))
    ox, oy, _ = (float(v) for v in re.search(r"origin: \[([^\]]+)\]", text).group(1).split(","))
    image = _pgm(yaml.with_name(re.search(r"image: (.+)", text).group(1).strip()))
    occupied = truth.occupied
    nx, ny = occupied.shape
    sx, sy = truth.scene.width / nx, truth.scene.height / ny
    free = mapped = 0
    for i in range(nx):
        for j in range(ny):
            if occupied[i, j]:
                continue
            x, y = -truth.scene.width / 2 + (i + 0.5) * sx, -truth.scene.height / 2 + (j + 0.5) * sy
            row, col = image.shape[0] - 1 - int((y - oy) / resolution), int((x - ox) / resolution)
            free += 1
            if 0 <= row < image.shape[0] and 0 <= col < image.shape[1] and image[row, col] > 250:
                mapped += 1
    return mapped / max(free, 1)


def _pgm(path: Path) -> np.ndarray:
    """A binary (P5) PGM image as an array, top row first."""
    data = path.read_bytes()
    fields, k = [], 0
    while len(fields) < 4:
        while data[k:k + 1].isspace():
            k += 1
        if data[k:k + 1] == b"#":
            k = data.index(b"\n", k) + 1
            continue
        end = k
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[k:end])
        k = end
    w, h = int(fields[1]), int(fields[2])
    return np.frombuffer(data[k + 1:k + 1 + w * h], dtype=np.uint8).reshape(h, w)
