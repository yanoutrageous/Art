"""Register the shared mesh to the locked side and rear silhouette landmarks."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT.parent
BASE = REVIEW / "proportion_3d_v23"
GEOMETRY = ROOT / "geometry"
sys.path[:0] = [str(REVIEW / "walk_3d_v15"), str(REVIEW / "walk_3d_v16")]
from repaint_from_turnaround import glb_views, smooth
from fit_scene_materials import write_glb


def fit_tail(points):
    # Retain the attachment and move the outer curve toward the V25 rear
    # silhouette. The upper tip keeps its height while the lower arc rises.
    x, y, _ = points.T.copy()
    points[:, 0] += 15.0 * smooth(23.0, 108.0, -x)
    points[:, 1] += 9.0 * np.clip((13.0 - y) / 79.0, 0, 1) * smooth(23.0, 62.0, -x)


def fit_scarf(points):
    _, y, z = points.T.copy()
    # Lift only the rear lower fold toward the jacket yoke in the approved art.
    rear = smooth(12.0, 33.0, -z)
    points[:, 1] += 5.5 * rear * (1.0 - smooth(19.0, 34.0, y))


def fit_asset(filename, part, move):
    layout, buffers = glb_views(BASE / filename)
    mesh = next(item for item in layout["meshes"] if item["name"] == part)
    for surface in mesh["primitives"]:
        accessor = layout["accessors"][surface["attributes"]["POSITION"]]
        slot = accessor["bufferView"]
        points = np.frombuffer(buffers[slot], dtype="<f4").reshape(-1, 3).copy()
        move(points)
        buffers[slot] = points.tobytes()
        accessor["min"] = points.min(axis=0).tolist()
        accessor["max"] = points.max(axis=0).tolist()
    write_glb(GEOMETRY / filename, layout, buffers)


if __name__ == "__main__":
    GEOMETRY.mkdir(exist_ok=True)
    fit_asset("head_3d.glb", "Scarf", fit_scarf)
    fit_asset("body_3d.glb", "Tail", fit_tail)
    print("V26_MESH_FIT_OK Scarf Tail")
