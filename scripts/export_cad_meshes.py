"""Tessellate `companion_r1.step` into optimized binary STL meshes for MuJoCo & WebGL 3D simulation."""

from __future__ import annotations

from pathlib import Path

import gmsh
import numpy as np
import trimesh


def _extract_volume_mesh(
    vol_tags: list[int],
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    """Extract triangulated surface mesh for a set of 3D volume tags from initialized gmsh model."""
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vert_offset = 0

    for vtag in vol_tags:
        _, down = gmsh.model.getAdjacencies(3, vtag)
        for stag in down:
            node_tags, coords, _ = gmsh.model.mesh.getNodes(2, stag, includeBoundary=True)
            if len(node_tags) == 0:
                continue
            pts = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
            pts[:, 0] -= origin_mm[0]
            pts[:, 1] -= origin_mm[1]
            pts[:, 2] -= origin_mm[2]

            tag_to_idx = {int(nt): i + vert_offset for i, nt in enumerate(node_tags)}
            elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, stag)
            for etype, enodes in zip(elem_types, elem_node_tags, strict=True):
                if etype == 2:  # 3-node triangle
                    tris = np.asarray([tag_to_idx[int(n)] for n in enodes], dtype=np.int64).reshape(-1, 3)
                    all_faces.append(tris)

            all_verts.append(pts)
            vert_offset += len(pts)

    if not all_verts or not all_faces:
        return trimesh.Trimesh()

    return trimesh.Trimesh(vertices=np.vstack(all_verts), faces=np.vstack(all_faces), process=True)


def export_all_meshes(step_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 3.2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 22.0)

    gmsh.model.add("companion_r1")
    gmsh.model.occ.importShapes(str(step_path))
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    groups: list[tuple[str, list[int], tuple[float, float, float]]] = [
        # Chassis structural parts (origin at 0, 0, 0)
        ("base_plate", [1], (0.0, 0.0, 0.0)),
        ("battery_tray", [2, 37], (0.0, 0.0, 0.0)),
        ("electronics_deck", [3, 85, 86], (0.0, 0.0, 0.0)),
        ("upper_cover", [4], (0.0, 0.0, 0.0)),
        ("mast", [5], (0.0, 0.0, 0.0)),
        ("frame_hardware", [9, 11, 14, 16, 19, 21, 24, 26, 29, 30, 31, 32], (0.0, 0.0, 0.0)),
        ("rpi5_board", [58, 66, 67, 68, 71, 72, 73, 74], (0.0, 0.0, 0.0)),
        # Head parts (origin at mast head pitch pivot: X=25mm, Y=0mm, Z=196mm)
        ("head_shell", [6, 7], (25.0, 0.0, 196.0)),
        ("head_lens", [8], (25.0, 0.0, 196.0)),
        ("head_display", [87], (25.0, 0.0, 196.0)),
        # 4 Wheels: tires and rims/hubs centered at their respective axle centers
        ("fl_tire", [28], (90.0, 135.0, 60.0)),
        ("fl_rim", [25, 27], (90.0, 135.0, 60.0)),
        ("fr_tire", [23], (90.0, -135.0, 60.0)),
        ("fr_rim", [20, 22], (90.0, -135.0, 60.0)),
        ("rl_tire", [18], (-90.0, 135.0, 60.0)),
        ("rl_rim", [15, 17], (-90.0, 135.0, 60.0)),
        ("rr_tire", [13], (-90.0, -135.0, 60.0)),
        ("rr_rim", [10, 12], (-90.0, -135.0, 60.0)),
    ]

    total_tris = 0
    total_bytes = 0
    for name, tags, origin in groups:
        mesh = _extract_volume_mesh(tags, origin_mm=origin)
        stl_path = out_dir / f"{name}.stl"
        mesh.export(str(stl_path))
        sz = stl_path.stat().st_size
        total_tris += len(mesh.faces)
        total_bytes += sz
        print(f"Exported {stl_path.name:20s}: {len(mesh.faces):6d} tris ({sz/1024:.1f} KB)")

    print(f"Total: {total_tris} tris ({total_bytes/1024:.1f} KB)")
    gmsh.finalize()


if __name__ == "__main__":
    export_all_meshes(Path("companion_r1.step"), Path("config/meshes"))
