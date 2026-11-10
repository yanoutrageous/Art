"""Build the volume and fixed texture coordinates for the head-only sample."""

import io
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
REPO = next(parent for parent in ROOT.parents if (parent / ".git").exists())
SOURCE = REPO / "Godot/GraytailGodot/assets/ui/art21/main_menu/scene/character/idle_00.png"
HAT_REFERENCE = SOURCE.with_name("walk_dungeon_00.png")
WIDTH, HEIGHT = 190, 128
PIVOT_X, BASE_Y = 126, 124
SEGMENTS = 64

# Source pixel edge coordinates. For the hat the slices run from left to
# right through the crown and brim. All other slices run top to bottom.
# Each row is slice, near silhouette edge, far silhouette edge, front
# depth, rear depth, and depth of the axis through that cross-section.
HAT = [
    (58, 65, 65, 0, 0, -5), (63, 54, 71, 16, 15, -5),
    (69, 46, 69, 27, 26, -5), (77, 39, 64, 36, 34, -5),
    (84, 32, 58, 42, 39, -5), (91, 24, 52, 47, 43, -5),
    (99, 16, 48, 50, 46, -5), (108, 9, 44, 52, 48, -5),
    (116, 5, 42, 53, 50, -5), (124, 3, 41, 54, 50, -5),
    (133, 5, 40, 53, 50, -5), (142, 10, 41, 52, 49, -5),
    (149, 17, 43, 49, 47, -5), (156, 24, 46, 46, 43, -5),
    (164, 34, 50, 41, 36, -5), (171, 42, 56, 33, 28, -5),
    (177, 47, 64, 25, 20, -5), (181, 52, 62, 12, 12, -5),
    (182, 58, 58, 0, 0, -5),
]
HEAD = [
    (22, 123, 123, 0, 0, -7), (28, 97, 151, 21, 26, -7),
    (38, 80, 166, 32, 36, -7), (49, 69, 172, 37, 42, -7),
    (58, 62, 175, 37, 43, -6), (64, 58, 176, 38, 43, -6),
    (70, 58, 178, 39, 42, -6), (75, 62, 176, 39, 42, -6),
    (80, 66, 173, 36, 38, -5), (85, 75, 165, 34, 35, -4),
    (90, 82, 157, 31, 30, -4), (94, 90, 151, 26, 25, -4),
    (97, 103, 145, 19, 17, -4), (99, 124, 130, 9, 8, -4),
    (100, 126, 126, 0, 0, -4),
]
EAR_LEFT = [
    (1.7, 73, 73, 0, 0, -15), (3.8, 68, 79, 4.5, 6, -15),
    (8, 66, 87, 7, 9, -16), (14, 65, 94, 8, 11, -16),
    (21, 66, 92, 8, 11, -16), (29, 68, 88, 8, 11, -15),
    (37, 73, 83, 6, 8, -13), (41, 79, 79, 0, 0, -12),
]
EAR_RIGHT = [
    (4, 160, 160, 0, 0, -14), (6, 154, 164, 4, 6, -14),
    (12, 147, 168, 7, 9, -14), (19, 147, 170, 8, 10, -14),
    (26, 148, 169, 8, 10, -13), (34, 154, 166, 5, 8, -11),
    (39, 159, 159, 0, 0, -10),
]
NECK = [
    (83, 125, 125, 0, 0, -8), (89, 100, 151, 15, 17, -8),
    (96, 89, 161, 25, 27, -8), (101, 91, 160, 31, 29, -7),
    (107, 99, 153, 31, 28, -6), (113, 110, 147, 27, 25, -5),
    (118, 120, 139, 22, 20, -3), (121, 129, 134, 16, 14, 0),
    (123, 132, 132, 0, 0, 3),
]


def profile(rows, coordinates):
    data = np.asarray(rows)
    return [np.interp(coordinates, data[:, 0], data[:, column]) for column in range(1, 6)]


def nearest_colors(rgb, mask):
    _, labels = cv2.distanceTransformWithLabels(
        np.where(mask, 0, 255).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    colors = np.empty((int(labels.max()) + 1, 3), dtype=np.uint8)
    colors[labels[mask]] = rgb[mask]
    return colors[labels]


def pigment(kind, rows, axis, x, y, source, noise):
    along = x if axis == "x" else y
    across = y if axis == "x" else x
    lo, hi, _, _, _ = profile(rows, along)
    inside = (across >= lo) & (across <= hi) & (along >= rows[0][0]) & (along <= rows[-1][0])
    valid = inside & (source[:, :, 3] > 128)
    brim_bottom = profile(HAT, x)[1]
    crown_top = profile(HAT, x)[0]
    if kind == "Fur":
        valid &= (y >= brim_bottom - 0.25) & (y < 97)
    elif kind.startswith("Ear"):
        valid &= (y < crown_top + 1)
    elif kind == "Scarf":
        valid &= y >= 94
    front = nearest_colors(source[:, :, :3], valid)

    fraction = np.clip((across - lo) / np.maximum(hi - lo, 0.001), 0, 1)
    edge_distance = 2.0 * np.minimum(fraction, 1.0 - fraction)
    color = np.empty_like(front, dtype=np.float32)
    if kind == "Hat":
        color[:] = (87, 91, 44)
        color += (12 * (1 - fraction))[..., None]
        band = (fraction > 0.68) & (fraction < 0.79)
        color[band] = (111, 77, 38)
        color[(fraction >= 0.79) & (fraction < 0.93)] = (106, 110, 50)
        color[fraction >= 0.93] = (51, 53, 28)
        seam = np.abs(x - 124 - 6 * (fraction - 0.45)) < 1.25
        color[seam & (fraction < 0.68)] *= 0.77
    elif kind == "Fur":
        color[:] = (85, 73, 52)
        fur_band = np.clip(1 - np.abs(y - (70 + 4 * (1 - edge_distance))) / 9, 0, 1)
        color = color * (1 - 0.43 * fur_band[..., None]) + np.array(
            [169, 145, 108], np.float32
        ) * (0.43 * fur_band[..., None])
        color[y > 85] *= 0.85
    elif kind.startswith("Ear"):
        color[:] = (104, 81, 53)
        color[edge_distance < 0.23] = (166, 135, 98)
    else:
        color[:] = (142, 79, 17)
        color += (14 * np.sin(y * 0.87 + x * 0.05))[..., None]
        color[y > 114] *= 0.87
    color += noise[..., None]
    return front, np.clip(color, 0, 255).astype(np.uint8)


def surface_grid(kind, rows, axis, rear, slices, theta):
    s = np.sin(theta)[None, :]
    s[np.abs(s) < 1e-8] = 0
    u = (1 - np.cos(theta))[None, :] * 0.5
    lo, hi, depth_front, depth_rear, axis_depth = (values[:, None] for values in profile(rows, slices))
    q = lo + (hi - lo) * u
    along = np.broadcast_to(slices[:, None], q.shape)
    x, y = (along, q) if axis == "x" else (q, along)
    depth = -depth_rear * s if rear else depth_front * s
    if kind == "Hat":
        # The front lip projects further than the crown. The back remains
        # a closed dome, with a shorter lip, on the same hat.
        cap_width = np.clip(depth_front / 54, 0, 1)
        axis_depth = axis_depth + 30 * cap_width * u**4
        if rear:
            depth = 0.88 * depth - 10 * cap_width * np.exp(-((u - 0.81) / 0.17) ** 2) * s
        else:
            depth = 0.94 * depth + 15 * cap_width * np.exp(-((u - 0.86) / 0.12) ** 2) * s
    elif kind == "Fur" and not rear:
        # Shape of the forehead, eye sockets, muzzle cushion and nose.
        # The original colors are painted once onto this fixed surface.
        def bump(cx, cy, rx, ry):
            return np.exp(-((x - cx) / rx) ** 2 - ((y - cy) / ry) ** 2)

        depth += s * (
            16 * bump(130, 82, 20, 12) + 11 * bump(132.5, 77.5, 7.8, 5.0)
            - 3.8 * bump(110, 64.5, 10, 8) - 3.8 * bump(146, 64, 10, 8)
            + 6.5 * bump(110, 65, 7.5, 7) + 6.5 * bump(146, 64.5, 7.5, 7)
            + 0.72 * (x - 110) * bump(110, 65, 14, 9)
            - 0.72 * (x - 146) * bump(146, 64.5, 14, 9)
            + 4 * bump(128, 54, 9, 11)
        )
    elif kind.startswith("Ear"):
        outward = 1 if kind == "EarL" else -1
        axis_depth = axis_depth + outward * 0.68 * (x - (lo + hi) * 0.5)
        if not rear:
            depth -= 3.2 * s**4

    return x, y, axis_depth + depth


def smooth(a, b, value):
    u = np.clip((value - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def hat_side_colors():
    """Take only cloth from the profile. Repaint the omitted regions in place."""
    source = np.asarray(Image.open(HAT_REFERENCE).convert("RGBA"))
    y, x = np.mgrid[:source.shape[0], :source.shape[1]]
    grid = [22, 30, 40, 50, 60, 70, 80, 90, 100, 108]
    brim = np.interp(x, grid, [34, 38, 40, 44, 49, 53, 57, 61, 65, 63])
    colors = source[:, :, :3].copy()
    r, g, b = (colors[:, :, i].astype(np.int16) for i in range(3))
    olive = (g >= r * 0.82) & (r <= 165) & (b < 92)
    valid = (source[:, :, 3] > 128) & (x >= 22) & (x <= 108) & (y < brim) & olive
    ear = (x >= 66) & (x <= 92) & (y >= 11) & (y <= 37)
    buckle = (x >= 53) & (x <= 72) & (y >= 24) & (y <= 37)
    patch = colors[9:21, 54:65].copy()
    for mask in (ear, buckle):
        row, column = np.where(mask & valid)
        cloth = patch[row % patch.shape[0], column % patch.shape[1]]
        colors[row, column] = np.clip(cloth.astype(np.int16) - (row[:, None] - 14) // 3, 0, 255)
    result = nearest_colors(colors, valid)
    # The source photograph has nothing below the cloth. Paint the underside
    # in its own colors instead of extending a boundary pixel downwards.
    y0, x0 = np.where(y >= brim - 1.0)
    grain = patch[y0 % patch.shape[0], x0 % patch.shape[1]].astype(np.int16)
    result[y0, x0] = np.clip(grain * 0.43 + (4, 6, 0), 0, 255)
    return result


def lookup(pixels, x, y):
    i = np.clip(np.floor(x).astype(np.int32), 0, pixels.shape[1] - 1)
    j = np.clip(np.floor(y).astype(np.int32), 0, pixels.shape[0] - 1)
    return pixels[j, i].astype(np.float32)


def unwrap(kind, rows, axis, rear, paint, profile_hat, noise):
    """Paint one surface in its own coordinates, including the lateral plane."""
    length = math.ceil((rows[-1][0] - rows[0][0]) * 6)
    slices = rows[0][0] + (rows[-1][0] - rows[0][0]) * (np.arange(length) + 0.5) / length
    theta = math.pi * (np.arange(480) + 0.5) / 480
    x, y, z = surface_grid(kind, rows, axis, rear, slices, theta)
    positions = np.stack([x, y, z], axis=-1)
    normal = np.cross(np.gradient(positions, axis=0), np.gradient(positions, axis=1))
    nx, nz = np.abs(normal[:, :, 0]), np.abs(normal[:, :, 2])
    frontal = smooth(0.14, 0.48, nz / np.maximum(nx + nz, 1e-8))
    projected = lookup(paint[rear], x, y)
    shade = lookup(np.repeat(noise[:, :, None], 3, axis=2), z + 92, y)[:, :, 0]

    if kind == "Hat":
        lateral = lookup(profile_hat, 65.8 - 0.73 * z, y * 0.96)
        if not rear:
            # The real front clasp stays attached to its original part of
            # the crown even though the material on the flank is different.
            clasp = smooth(144, 150, x) * (1 - smooth(160, 166, x))
            clasp *= smooth(19, 24, y) * (1 - smooth(33, 38, y))
            frontal = np.maximum(frontal, clasp)
    elif kind == "Fur":
        # The whole face, including the original eye rings and smile, has
        # one source. The flank uses its own fur color outside that area.
        lateral = np.empty_like(projected)
        lateral[:] = (91, 79, 56)
        cheek_line = 71.5 - 0.11 * z + shade * 0.21
        pale = smooth(cheek_line - 4, cheek_line, y) * (1 - smooth(cheek_line + 3, cheek_line + 8, y))
        pale *= smooth(-35, 3, z)
        lateral = lateral * (1 - pale[..., None] * 0.81) + np.array([209, 182, 141]) * pale[..., None] * 0.81
        lateral += (shade * 1.4)[..., None]
        if not rear:
            frontal = np.maximum(frontal, smooth(79, 94, x) * (1 - smooth(159, 174, x)))
    elif kind.startswith("Ear"):
        lateral = np.empty_like(projected)
        lateral[:] = (110, 86, 55)
        lateral += (shade * 1.1)[..., None]
    else:
        lateral = np.empty_like(projected)
        lateral[:] = (152, 82, 15)
        lateral += (11 * np.sin(y * 0.84 - z * 0.09) + shade)[..., None]
    weight = frontal[..., None]
    result = np.clip(projected * weight + lateral * (1 - weight), 0, 255)
    return result.transpose(1, 0, 2).astype(np.uint8)


def geometry(kind, rows, axis, rear):
    slices = np.linspace(rows[0][0], rows[-1][0], math.ceil((rows[-1][0] - rows[0][0]) / 0.8) + 1)
    theta = np.linspace(0, math.pi, SEGMENTS + 1)
    x, y, z = surface_grid(kind, rows, axis, rear, slices, theta)
    vertices = np.stack([x - PIVOT_X, BASE_Y - y, z], axis=-1).reshape(-1, 3)
    first, second = np.meshgrid(np.linspace(0, 1, len(slices)), np.linspace(0, 1, len(theta)), indexing="ij")
    uv = np.stack([first, second], axis=-1).reshape(-1, 2)
    triangles = []
    for row in range(len(slices) - 1):
        for col in range(SEGMENTS):
            i = row * (SEGMENTS + 1) + col
            j = i + SEGMENTS + 1
            if rear == (axis == "x"):
                triangles.extend((i, j, i + 1, j, j + 1, i + 1))
            else:
                triangles.extend((i, i + 1, j, j, i + 1, j + 1))
    faces = np.asarray(triangles).reshape(-1, 3)
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return vertices.astype("<f4"), uv.astype("<f4"), faces[area > 1e-7].astype("<u2").ravel()


class ModelFile:
    def __init__(self):
        self.data = {
            "asset": {"version": "2.0", "generator": "Graytail head study"},
            "extensionsUsed": ["KHR_materials_unlit"],
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"name": "HeadTurn", "children": []}],
            "meshes": [],
            "bufferViews": [],
            "accessors": [],
            "materials": [],
            "images": [],
            "textures": [],
            "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
        }
        self.binary = bytearray()

    def add(self, key, item):
        items = self.data[key]
        items.append(item)
        return len(items) - 1

    def view(self, content, target=None):
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        view = {"buffer": 0, "byteOffset": len(self.binary), "byteLength": len(content)}
        if target:
            view["target"] = target
        self.binary.extend(content)
        return self.add("bufferViews", view)

    def accessor(self, values, shape, component, target):
        record = {
            "bufferView": self.view(values.tobytes(), target),
            "componentType": component,
            "count": len(values),
            "type": shape,
        }
        if shape == "VEC3":
            record["min"] = values.min(axis=0).tolist()
            record["max"] = values.max(axis=0).tolist()
        return self.add("accessors", record)

    def surface(self, kind, rows, axis, pixels, rear):
        png = io.BytesIO()
        Image.fromarray(pixels).save(png, format="PNG", optimize=True)
        label = f"{kind} {'rear' if rear else 'front'}"
        image = self.add("images", {
            "name": label, "mimeType": "image/png", "bufferView": self.view(png.getvalue())
        })
        texture = self.add("textures", {"sampler": 0, "source": image})
        material = self.add("materials", {
            "name": label,
            "extensions": {"KHR_materials_unlit": {}},
            "pbrMetallicRoughness": {"baseColorTexture": {"index": texture}},
            "doubleSided": True,
        })
        vertices, uv, triangles = geometry(kind, rows, axis, rear)
        return {
            "attributes": {
                "POSITION": self.accessor(vertices, "VEC3", 5126, 34962),
                "TEXCOORD_0": self.accessor(uv, "VEC2", 5126, 34962),
            },
            "indices": self.accessor(triangles, "SCALAR", 5123, 34963),
            "material": material,
        }

    def write(self, path):
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        self.data["buffers"] = [{"byteLength": len(self.binary)}]
        header = json.dumps(self.data, separators=(",", ":")).encode()
        header += b" " * (-len(header) % 4)
        size = 12 + 8 + len(header) + 8 + len(self.binary)
        path.write_bytes(
            struct.pack("<4sII", b"glTF", 2, size)
            + struct.pack("<I4s", len(header), b"JSON") + header
            + struct.pack("<I4s", len(self.binary), b"BIN\0") + self.binary
        )


def main():
    source = np.asarray(Image.open(SOURCE).convert("RGBA").crop((0, 0, WIDTH, HEIGHT)))
    y, x = np.mgrid[:HEIGHT, :WIDTH].astype(np.float32) + 0.5
    rng = np.random.default_rng(8173)
    grains = rng.normal(0, 1, (HEIGHT, WIDTH)).astype(np.float32)
    noise = 4 * grains + 12 * cv2.GaussianBlur(grains, (5, 5), 1.0)
    profile_hat = hat_side_colors()
    model = ModelFile()
    parts = (
        ("Scarf", NECK, "y"), ("Fur", HEAD, "y"), ("EarL", EAR_LEFT, "y"),
        ("EarR", EAR_RIGHT, "y"), ("Hat", HAT, "x"),
    )
    for name, rows, axis in parts:
        paint = pigment(name, rows, axis, x, y, source, noise)
        surfaces = []
        for rear in (False, True):
            texture = unwrap(name, rows, axis, rear, paint, profile_hat, noise)
            surfaces.append(model.surface(name, rows, axis, texture, rear))
        mesh = model.add("meshes", {"name": name, "primitives": surfaces})
        node = model.add("nodes", {"name": name, "mesh": mesh})
        model.data["nodes"][0]["children"].append(node)
    model.write(ROOT / "head_v13.glb")


if __name__ == "__main__":
    main()
