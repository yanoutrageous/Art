"""Fit the back of the V26 skull to the locked V25 outlines."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT.parent
V26 = REVIEW / "turnaround_3d_v26"
V25 = REVIEW / "side_back_art_v25"
GEOMETRY = ROOT / "geometry"
sys.path.insert(0, str(V26))
from fit_mesh import glb_views, smooth, write_glb


def actor_mask(slug):
    with Image.open(V25 / f"{slug}-480x256.png") as image:
        rgb = np.asarray(image.convert("RGB")).astype(np.int16)
    color = (rgb[:, :, 0] > rgb[:, :, 2] + 6) | (rgb.max(axis=2) < 26)
    return cv2.morphologyEx(color.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def outline(mask, rows):
    spans = [np.flatnonzero(mask[row]) for row in rows]
    edges = np.array([(xs[0], xs[-1] + 1) for xs in spans], dtype=np.float32)
    padded = np.pad(edges, ((1, 1), (0, 0)), mode="edge")
    median = np.median(np.stack([padded[i:i + len(rows)] for i in range(3)]), axis=0)
    return cv2.GaussianBlur(median, (1, 9), sigmaX=0, sigmaY=2.1, borderType=cv2.BORDER_REPLICATE)


def positions(layout, buffers, surface):
    accessor = layout["accessors"][surface["attributes"]["POSITION"]]
    return accessor, np.frombuffer(buffers[accessor["bufferView"]], "<f4").reshape(-1, 3)


def deform_rear(points, side, back):
    # The mesh is a grid of horizontal cross-sections. Its two meridians are
    # shared with the untouched front. Setbacks are weighted by true rear depth
    # so the two meridians and the end poles stay welded.
    count = int(np.count_nonzero(points[:, 1] == points[0, 1]))
    assert count == 65
    rings = points.reshape(-1, count, 3).copy()
    x, y, z = (rings[:, :, i].copy() for i in range(3))
    seam = 0.5 * (z[:, :1] + z[:, -1:])
    deepest = z.min(axis=1, keepdims=True)
    depth = seam - deepest
    ratio = np.clip((seam - z) / np.maximum(depth, 1e-6), 0, 1)
    depth_blend = smooth(1, 9, depth) * ratio**2
    old_row = 154.0 - y
    lower = smooth(103.0, 120.0, old_row)
    height_gain = 2.8 * lower * (1 - smooth(122.0, 130.0, old_row))
    rings[:, :, 1] += height_gain * depth_blend

    row = 154.0 - rings[:, :, 1]
    side_rows, side_edges = side
    rear_edge = np.interp(row, side_rows + 0.5, side_edges[:, 0] + 0.32) - 240.0
    select = smooth(84, 93, old_row) * (1 - smooth(117, 130, old_row))
    shift_z = np.clip(rear_edge - deepest, -0.24 * depth, 0.43 * depth)
    rings[:, :, 2] += select * depth_blend * shift_z

    back_rows, back_edges = back
    left = np.interp(row, back_rows + 0.5, back_edges[:, 0])
    right = np.interp(row, back_rows + 0.5, back_edges[:, 1])
    center = 0.5 * (x.min(axis=1, keepdims=True) + x.max(axis=1, keepdims=True))
    radius = 0.5 * np.ptp(x, axis=1, keepdims=True)
    target_center = 240.0 - 0.5 * (left + right)
    target_radius = 0.5 * (right - left)
    relative = np.clip(target_radius / np.maximum(radius, 1), 0.73, 1.035)
    shift_x = np.clip(target_center - center, -5.0, 8.0) + (x - center) * (relative - 1)
    across = smooth(85, 101, old_row) * (1 - smooth(114, 130, old_row))
    rings[:, :, 0] += across * depth_blend * shift_x
    result = rings.reshape(-1, 3)
    assert np.array_equal(rings[:, (0, -1)], points.reshape(-1, count, 3)[:, (0, -1)])
    return result


def verify_triangles(layout, buffers, surface, before, after):
    accessor = layout["accessors"][surface["indices"]]
    triangles = np.frombuffer(buffers[accessor["bufferView"]], "<u2").reshape(-1, 3)
    def crosses(points):
        a, b, c = np.moveaxis(points[triangles], 1, 0)
        return np.cross(b - a, c - a)
    old, new = crosses(before), crosses(after)
    a, b = (np.linalg.norm(values, axis=1) for values in (old, new))
    ratio = b / np.maximum(a, 1e-8)
    cosine = np.sum(old * new, axis=1) / np.maximum(a * b, 1e-8)
    assert ratio.min() > 0.25 and cosine.min() > 0.33, (ratio.min(), cosine.min())
    return {"triangles": len(triangles), "min_area_ratio": float(ratio.min()), "min_normal_cosine": float(cosine.min())}


def main():
    side_rows, back_rows = np.arange(89, 120), np.arange(93, 120)
    side = side_rows, outline(actor_mask("side-090"), side_rows)
    back = back_rows, outline(actor_mask("back-180"), back_rows)
    layout, buffers = glb_views(V26 / "head_3d.glb")
    mesh = next(item for item in layout["meshes"] if item["name"] == "Fur")
    surface = mesh["primitives"][1]
    accessor, old = positions(layout, buffers, surface)
    fitted = deform_rear(old, side, back)
    quality = verify_triangles(layout, buffers, surface, old, fitted)
    shift = fitted - old
    buffers[accessor["bufferView"]] = fitted.tobytes()
    accessor["min"] = fitted.min(axis=0).tolist()
    accessor["max"] = fitted.max(axis=0).tolist()
    GEOMETRY.mkdir(exist_ok=True)
    write_glb(GEOMETRY / "head_3d.glb", layout, buffers)
    quality["moved_vertices"] = int(np.any(shift != 0, axis=1).sum())
    quality["max_shift_xyz"] = np.max(np.abs(shift), axis=0).tolist()
    (ROOT / "mesh-check.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print("V27_OCCIPUT_MESH_OK", quality)


if __name__ == "__main__":
    main()
