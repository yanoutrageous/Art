"""Keep the V15 geometry and repaint areas where the turnaround has no coverage."""

import hashlib
import io
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
V15 = ROOT.parent / "walk_3d_v15"
sys.path.insert(0, str(V15))
from repaint_from_turnaround import (BACK, HEAD_ROWS, PARTS, REFERENCE, SIDE, V13, V14,
                                      glb_views, projection, read_color, rig_grid, smooth)
from build_model import SOURCE


def reference_fields():
    with Image.open(REFERENCE) as image:
        rgb = np.array(image.convert("RGB").resize((1260, 420), Image.Resampling.LANCZOS))
    colors = rgb.astype(np.int16)
    foreground = (colors[:, :, 0] > colors[:, :, 2] + 3) | (colors.max(axis=2) < 27)
    fields = {}
    for view, regions in (("side", SIDE), ("back", BACK)):
        for part, polygons in regions.items():
            if part == "Tail":
                continue
            mask = np.zeros(rgb.shape[:2], np.uint8)
            if isinstance(polygons[0], tuple):
                polygons = [polygons]
            cv2.fillPoly(mask, [np.array(polygon, np.int32) for polygon in polygons], 1)
            if (view, part) == ("side", "Hat"):
                cv2.ellipse(mask, (606, 44), (23, 31), -15, 0, 360, 0, -1)
            mask &= foreground.astype(np.uint8)
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            # Prefilter the guide at roughly one native pixel and feather only inwards.
            weighted = cv2.GaussianBlur(rgb.astype(np.float32) * mask[:, :, None], (5, 5), 0.85)
            covered = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0.85)
            color = weighted / np.maximum(covered[:, :, None], 1e-4)
            coverage = smooth(0.7, 3.8, distance).astype(np.float32)
            fields[view, part] = color, coverage
    return fields


def sample_field(field, coordinates, fallback, strength=1.0):
    colors, coverage = field
    weight = read_color(coverage, coordinates)[:, :, None] * strength
    sampled = read_color(colors, coordinates)
    return fallback * (1 - weight) + sampled * weight


def fabric_swatch(cloth, x, y, z):
    u = (0.52 * x + 0.84 * z).astype(np.float32)
    v = (y + 0.29 * z).astype(np.float32)
    return cv2.remap(cloth, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def fit_tail(old, native_tail):
    # Keep the shared three-dimensional bands. Reduce subpixel fur noise and
    # mix the stripe colors back toward the artwork already in the menu.
    filtered = cv2.GaussianBlur(old.astype(np.float32), (9, 9), 1.65)
    old_luma = filtered.mean(axis=2)
    band = smooth(50.0, 115.0, old_luma)[:, :, None]
    dark, pale = native_tail
    original = dark * (1 - band) + pale * band
    return 0.69 * filtered + 0.31 * original


def fit(name, rear, current, original, fields, cloth):
    angle_count, slice_count = current.shape[:2]
    rows = (HEAD_ROWS if name in HEAD_ROWS else PARTS)[name]
    slices = rows[0][0] + (rows[-1][0] - rows[0][0]) * (np.arange(slice_count) + 0.5) / slice_count
    theta = np.pi * (np.arange(angle_count) + 0.5) / angle_count
    x, y, z = rig_grid(name, rear, slices, theta)
    part, side_uv, back_uv = projection(name, x, y, z)
    points = np.stack([x, y, z], axis=-1)
    normal = np.cross(np.gradient(points, axis=0), np.gradient(points, axis=1))
    nx, nz = np.abs(normal[:, :, 0]), np.abs(normal[:, :, 2])
    facing = nz / np.maximum(nx + nz, 1e-8)
    native = original.transpose(1, 0, 2).astype(np.float32)
    fallback = native
    if name == "Chest":
        # The guide has virtually no visible center side panel under the arm.
        # Use actual ART21 jacket colors in the hidden area exposed by animation.
        fallback = 0.35 * native + 0.65 * fabric_swatch(cloth, x, y, z)
    strength = 0.82 if name == "Fur" else 0.94
    side = sample_field(fields["side", part], side_uv, fallback, strength)
    if rear:
        other = sample_field(fields["back", part], back_uv, native, strength)
        weight = smooth(0.14, 0.59, facing)
        if name == "Fur":
            boundary = smooth(0.0, 0.23, np.sin(theta))[None, :, None]
            side = native * (1 - boundary) + side * boundary
    else:
        other = current.transpose(1, 0, 2).astype(np.float32)
        weight = smooth(0.08, 0.47, facing)
        if name == "Hat":
            clasp = smooth(144, 150, x) * (1 - smooth(160, 166, x))
            clasp *= smooth(19, 24, y) * (1 - smooth(33, 38, y))
            weight = np.maximum(weight, clasp)
    weight = weight[:, :, None]
    return (other * weight + side * (1 - weight)).transpose(1, 0, 2)


def write_glb(output, data, buffers):
    body = bytearray()
    for view, content in zip(data["bufferViews"], buffers):
        body.extend(b"\0" * (-len(body) % 4))
        view["byteOffset"], view["byteLength"] = len(body), len(content)
        body.extend(content)
    data["buffers"][0]["byteLength"] = len(body)
    body.extend(b"\0" * (-len(body) % 4))
    header = json.dumps(data, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 4)
    output.write_bytes(struct.pack("<4sIIII", b"glTF", 2, 28 + len(header) + len(body), len(header), 0x4E4F534A)
                       + header + struct.pack("<I4s", len(body), b"BIN\0") + body)


def replace_materials(filename, original_path, fields, cloth, tail):
    before, buffers = glb_views(V15 / filename)
    original, original_buffers = glb_views(original_path)
    originals = {item["name"]: original_buffers[item["bufferView"]] for item in original["images"]}
    changed = []
    for item in before["images"]:
        label = item["name"]
        name, side = label.split()
        if side == "front" and name in {"Fur", "EarL", "EarR", "Scarf"}:
            continue
        slot = item["bufferView"]
        with Image.open(io.BytesIO(buffers[slot])) as current:
            with Image.open(io.BytesIO(originals[label])) as initial:
                if name == "Tail":
                    fitted = fit_tail(np.asarray(current), tail)
                else:
                    fitted = fit(name, side == "rear", np.asarray(current), np.asarray(initial), fields, cloth)
        png = io.BytesIO()
        Image.fromarray(np.rint(np.clip(fitted, 0, 255)).astype(np.uint8)).save(png, format="PNG")
        buffers[slot] = png.getvalue()
        changed.append(label)
    write_glb(ROOT / filename, before, buffers)
    _, saved = glb_views(ROOT / filename)
    _, old = glb_views(V15 / filename)
    image_slots = {item["bufferView"] for item in before["images"] if item["name"] in changed}
    assert all(new == prior for i, (new, prior) in enumerate(zip(saved, old)) if i not in image_slots)
    return changed


def main():
    fields = reference_fields()
    with Image.open(SOURCE) as image:
        native = np.asarray(image.convert("RGBA"))
    cloth = native[123:145, 106:120, :3].astype(np.float32)
    assert (native[123:145, 106:120, 3] == 255).all()
    swatch = native[118:177, 18:60]
    rgb = swatch[swatch[:, :, 3] >= 240, :3]
    luma = rgb.mean(axis=1)
    tail = [np.median(rgb[luma < np.percentile(luma, 43)], axis=0),
            np.median(rgb[luma > np.percentile(luma, 63)], axis=0)]
    report = {
        "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "texture_sampling": "Inside-mask only; native material outside",
        "v15_geometry_and_original_front_face_byte_identical": True,
        "changed": {
            "head": replace_materials("head_3d.glb", V13 / "head_v13.glb", fields, cloth, tail),
            "body": replace_materials("body_3d.glb", V14 / "body_3d.glb", fields, cloth, tail),
        },
    }
    (ROOT / "scene-material-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
