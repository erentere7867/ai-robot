"""Household models from Gazebo Fuel, measured, for 3D rooms a detector can recognise.

`measure()` reads each downloaded model's geometry (collision first, else visual: boxes, cylinders, spheres, Collada
or OBJ meshes, with their scale and unit) and reports its footprint: the size on the floor and the height. The table
below holds those measurements, so a room's object can be given the radius and height its 3D model really has.

    python3 -m robot.ros.fuel           # in WSL, after `gz fuel download` of the models below: prints the table
"""

from __future__ import annotations

import math
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

FUEL = Path("~/.gz/fuel/fuel.gazebosim.org/openrobotics/models").expanduser()
URI = "https://fuel.gazebosim.org/1.0/{}/models/{}"

# label -> (Fuel model, footprint circle radius m, height m, colour as rendered), from `measure()` (11 Nov 2026).
# The large Sofa (1.80 m) and Table (1.50 m) do not fit the test rooms: MiniSofa and CoffeeTable stand in.
MODELS: dict[str, tuple[str, float, float, str]] = {
    "chair": ("OpenRobotics/Chair", 0.39, 0.82, "blue"),  # blue fabric on a wooden frame, as rendered
    "sofa": ("OpenRobotics/MiniSofa", 0.50, 0.61, "grey"),
    "table": ("OpenRobotics/CoffeeTable", 0.51, 0.37, "brown"),
    "desk": ("OpenRobotics/Desk", 0.49, 0.55, "brown"),
    "lamp": ("OpenRobotics/LampAndStand", 0.44, 1.02, "white"),
    "bin": ("OpenRobotics/TrashBin", 0.23, 0.45, "grey"),
    "cabinet": ("OpenRobotics/Cabinet", 0.34, 1.02, "brown"),
    "fridge": ("OpenRobotics/Fridge", 0.40, 1.56, "white"),
    "backpack": ("OpenRobotics/JanSport Backpack Red", 0.22, 0.47, "red"),
    "person": ("OpenRobotics/Standing person", 0.33, 1.93, ""),
    "box": ("OpenRobotics/Cardboard box", 0.32, 0.30, "brown"),
    "ball": ("makerspet/red_ball_10in", 0.13, 0.25, "red"),
    "plant": ("slhdn/plant", 0.20, 0.15, "green"),
}
LIFT = {"box": 0.150, "ball": 0.127}  # models whose origin is their centre: raised so they stand on the floor


def include(label: str, name: str, x: float, y: float, yaw: float = 0.0) -> str | None:
    """An SDF <include> of the label's Fuel model at (x, y), or None when there is no model for it."""
    if label not in MODELS:
        return None
    owner, model = MODELS[label][0].split("/")
    uri = URI.format(owner, model.replace(" ", "%20"))
    return (f"<include><uri>{uri}</uri><name>{name}</name><static>true</static>"
            f"<pose>{x:.3f} {y:.3f} {LIFT.get(label, 0.0):.3f} 0 0 {yaw:.3f}</pose></include>")


def furnish(scene: object) -> object:
    """Give each thing that has a 3D model that model's measured size and colour, so the room the checks judge is
    the room Gazebo shows. Returns the scene (changed in place)."""
    for t in scene.things:
        if t.label in MODELS:
            _, radius, _, color = MODELS[t.label]
            t.radius = radius
            if color:
                t.color = color
    return scene


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def _stl_points(path: Path) -> list[tuple[float, float, float]]:
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:400]:  # ASCII STL
        return [tuple(float(v) for v in line.split()[1:4]) for line in data.decode(errors="ignore").splitlines()
                if line.strip().startswith("vertex")]
    count = struct.unpack_from("<I", data, 80)[0]
    points = []
    for k in range(min(count, (len(data) - 84) // 50)):
        values = struct.unpack_from("<12f", data, 84 + 50 * k)
        points += [values[3:6], values[6:9], values[9:12]]
    return points


def _mesh_points(path: Path) -> list[tuple[float, float, float]]:
    """Vertices of a Collada, OBJ or STL mesh, in metres, z up; none for a format it cannot read."""
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return [tuple(float(v) for v in line.split()[1:4]) for line in path.read_text(errors="ignore").splitlines()
                if line.startswith("v ")]
    if suffix == ".stl":
        return _stl_points(path)
    text = path.read_text(errors="ignore")
    try:
        root = ET.fromstring(re.sub(r'\sxmlns="[^"]+"', "", text, count=1))
    except ET.ParseError:
        return []
    unit = float((root.find("asset/unit").get("meter") if root.find("asset/unit") is not None else 1.0) or 1.0)
    up = (root.findtext("asset/up_axis") or "Z_UP").strip()
    points = []
    for source in root.iter("source"):
        if "position" not in (source.get("id") or "").lower():
            continue
        array = source.find("float_array")
        if array is None or not array.text:
            continue
        values = _floats(array.text)
        for k in range(0, len(values) - 2, 3):
            x, y, z = (v * unit for v in values[k:k + 3])
            points.append((x, -z, y) if up == "Y_UP" else (x, y, z))
    return points


def measure(model_dir: Path) -> tuple[float, float, float, float, float, float]:
    """Footprint of a model: (size x, size y, height, centre x, centre y, bottom z) in metres. A bottom below zero
    means the model sinks into the floor unless it is lifted (LIFT)."""
    sdf = ET.parse(model_dir / "model.sdf").getroot()
    pts: list[tuple[float, float, float]] = []
    for kind in ("collision", "visual"):
        for el in sdf.iter(kind):
            pose = _floats(el.findtext("pose") or "0 0 0 0 0 0") + [0.0] * 6
            ox, oy, oz = pose[:3]
            geometry = el.find("geometry")
            if geometry is None:
                continue
            box, cyl, sph, mesh = (geometry.find(k) for k in ("box", "cylinder", "sphere", "mesh"))
            if box is not None:
                sx, sy, sz = _floats(box.findtext("size") or "0 0 0")
                pts += [(ox + a * sx / 2, oy + b * sy / 2, oz + c * sz / 2) for a in (-1, 1) for b in (-1, 1)
                        for c in (-1, 1)]
            elif cyl is not None:
                r, h = float(cyl.findtext("radius") or 0), float(cyl.findtext("length") or 0)
                pts += [(ox + a * r, oy + b * r, oz + c * h / 2) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
            elif sph is not None:
                r = float(sph.findtext("radius") or 0)
                pts += [(ox + a * r, oy + b * r, oz + c * r) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
            elif mesh is not None:
                uri = (mesh.findtext("uri") or "").strip()
                scale = _floats(mesh.findtext("scale") or "1 1 1")
                name = uri.split("/")[-1]
                found = next(iter(model_dir.rglob(name)), None) if name else None
                if found is not None:
                    pts += [(ox + x * scale[0], oy + y * scale[1], oz + z * scale[2]) for x, y, z in _mesh_points(found)]
        if pts:
            break
    if not pts:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    xs, ys, zs = zip(*pts, strict=True)
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), (max(xs) + min(xs)) / 2,
            (max(ys) + min(ys)) / 2, min(zs))


def main() -> None:
    for model in sorted(FUEL.iterdir()):
        versions = sorted((p for p in model.iterdir() if p.is_dir()), key=lambda p: int(p.name) if p.name.isdigit()
                          else 0)
        if not versions:
            continue
        sx, sy, sz, cx, cy, bottom = measure(versions[-1])
        radius = math.hypot(sx, sy) / 2
        print(f"{model.name.replace('%20', ' '):<24} {sx:5.2f} x {sy:5.2f} m, {sz:4.2f} m tall, centre ({cx:+.2f}, "
              f"{cy:+.2f}), circle radius {radius:.2f}, bottom {bottom:+.3f}")


if __name__ == "__main__":
    main()
