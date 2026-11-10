"""Trim the cheek margins to fit the approved rear contour."""

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
V27 = ROOT.parent / "head_shape_3d_v27"
sys.path.insert(0, str(V27))
from fit_occiput import actor_mask, glb_views, outline, positions, smooth, write_glb


PROTECTED_FACE = (207, 83, 277, 122)
MAX_LEFT = 5.0
MAX_RIGHT = 2.5
LEFT_GUARD = -36.0
RIGHT_GUARD = 40.0


def fit_width(points, front_grid, rows, target):
    x, y = points[:, :2].T
    scan = 154.0 - y
    front_y = 154.0 - front_grid[:, 0, 1]
    current_left = np.interp(scan, front_y, front_grid[:, :, 0].min(axis=1))
    current_right = np.interp(scan, front_y, front_grid[:, :, 0].max(axis=1))
    desired_left = 240.0 - np.interp(scan, rows + 0.5, target[:, 1])
    desired_right = 240.0 - np.interp(scan, rows + 0.5, target[:, 0])
    height = smooth(89.0, 96.0, scan) * (1.0 - smooth(118.0, 128.0, scan))

    left_span = np.maximum(0.0, LEFT_GUARD - current_left)
    right_span = np.maximum(0.0, current_right - RIGHT_GUARD)
    # Bound strain as the cheek approaches the core. Apply the same map to
    # both halves so their common edge stays welded.
    left_trim = np.clip(desired_left - current_left, 0.0, np.minimum(MAX_LEFT, 0.34 * left_span))
    right_trim = np.clip(current_right - desired_right, 0.0, np.minimum(MAX_RIGHT, 0.34 * right_span))
    left_weight = smooth(0.0, 1.0, (LEFT_GUARD - x) / np.maximum(left_span, 0.01))
    right_weight = smooth(0.0, 1.0, (x - RIGHT_GUARD) / np.maximum(right_span, 0.01))
    result = points.copy()
    result[:, 0] += height * (left_trim * left_weight - right_trim * right_weight)
    return result


def triangle_indices(layout, buffers, surface):
    accessor = layout["accessors"][surface["indices"]]
    return np.frombuffer(buffers[accessor["bufferView"]], "<u2").reshape(-1, 3)


def check_faces(layout, buffers, surface, before, after):
    triangles = triangle_indices(layout, buffers, surface)
    def cross(points):
        a, b, c = np.moveaxis(points[triangles].astype(np.float64), 1, 0)
        return np.cross(b - a, c - a)
    original, updated = cross(before), cross(after)
    old_area = np.linalg.norm(original, axis=1)
    new_area = np.linalg.norm(updated, axis=1)
    regular = old_area > 1e-8
    assert np.array_equal(before[triangles[~regular]], after[triangles[~regular]])
    ratio = (new_area[regular] / old_area[regular]).min()
    cosine = (np.sum(original[regular] * updated[regular], axis=1) /
              (old_area[regular] * new_area[regular])).min()
    assert ratio > 0.25 and cosine > 0.33, (ratio, cosine)
    return {"triangles": len(triangles), "min_area_ratio": float(ratio),
            "min_normal_cosine": float(cosine),
            "unchanged_source_degenerate_triangles": int((~regular).sum())}


def protect_face(layout, buffers, surface, before, after):
    triangles = triangle_indices(layout, buffers, surface)
    l, t, r, b = PROTECTED_FACE
    guarded = np.zeros(len(triangles), dtype=bool)
    for points in (before, after):
        tri = points[triangles]
        x, y = tri[:, :, 0] + 240.0, 154.0 - tri[:, :, 1]
        guarded |= (x.min(axis=1) < r) & (x.max(axis=1) >= l) & (y.min(axis=1) < b) & (y.max(axis=1) >= t)
    core = np.unique(triangles[guarded])
    assert np.array_equal(before[core], after[core]), "Eye, nose or mouth triangle moved"
    return int(len(core))


def main():
    layout, buffers = glb_views(V27 / "head_3d.glb")
    fur = next(mesh for mesh in layout["meshes"] if mesh["name"] == "Fur")
    front, rear = fur["primitives"]
    _, front_points = positions(layout, buffers, front)
    front_grid = front_points.reshape(-1, 65, 3)
    rows = np.arange(91, 120)
    target = outline(actor_mask("back-180"), rows)
    results, shared, shifts = [], [], []
    for surface in (front, rear):
        accessor, points = positions(layout, buffers, surface)
        fitted = fit_width(points, front_grid, rows, target)
        assert np.array_equal(points[:, 1:], fitted[:, 1:])
        mesh_check = check_faces(layout, buffers, surface, points, fitted)
        mesh_check["protected_vertices"] = protect_face(layout, buffers, surface, points, fitted)
        results.append(mesh_check)
        shared.append(fitted.reshape(-1, 65, 3)[:, (0, -1)])
        shifts.append(fitted[:, 0] - points[:, 0])
        buffers[accessor["bufferView"]] = fitted.tobytes()
        accessor["min"] = fitted.min(axis=0).tolist()
        accessor["max"] = fitted.max(axis=0).tolist()
    assert np.array_equal(*shared), "Front and rear cheek boundaries split"
    write_glb(ROOT / "head_3d.glb", layout, buffers)
    shifts = np.concatenate(shifts)
    report = {"shift_x_px": [float(shifts.min()), float(shifts.max())], "surfaces": results}
    (ROOT / "mesh-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("V28_OUTLINE_MESH_OK", report)


if __name__ == "__main__":
    main()
