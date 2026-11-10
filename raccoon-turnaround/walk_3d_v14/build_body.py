"""Closed body surfaces in the original pixel-art reference coordinates."""

import io
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "head_3d_v13"))
from build_model import BASE_Y, PIVOT_X, SOURCE, ModelFile, lookup, nearest_colors, profile, smooth

# y, left edge, right edge, front radius, rear radius, and center depth.
# Overlap the sleeve, cuff, trouser hem and boot surfaces at their joints.
PARTS = {
    "Chest": [
        (91, 126, 126, 0, 0, -7), (95, 101, 151, 9, 13, -7),
        (101, 84, 164, 16, 22, -7), (110, 80, 166, 22, 26, -7),
        (119, 83, 164, 23, 27, -6), (129, 81, 164, 28, 27, -5),
        (140, 80, 165, 29, 28, -5), (149, 85, 163, 26, 26, -4),
        (155, 92, 156, 22, 23, -4), (161, 126, 126, 0, 0, -4),
    ],
    "Hips": [
        (137, 125, 125, 0, 0, -3), (143, 83, 164, 24, 24, -3),
        (149, 80, 169, 29, 27, -2), (157, 79, 170, 30, 27, -2),
        (164, 83, 163, 28, 27, -2), (172, 88, 160, 24, 23, -2),
        (177, 100, 151, 19, 19, -2), (180, 124, 124, 0, 0, -2),
    ],
    "NearArm": [
        (96, 88, 88, 0, 0, -4), (101, 73, 96, 11, 15, -3),
        (109, 66, 99, 14, 17, -3), (119, 62, 96, 15, 17, -2),
        (130, 59, 92, 15, 15, 0), (139, 59, 90, 14, 14, 1),
        (146, 76, 76, 0, 0, 2),
    ],
    "NearGlove": [
        (126, 75, 75, 0, 0, 0), (132, 61, 89, 12, 13, 0),
        (140, 58, 91, 16, 15, 1), (149, 59, 92, 18, 16, 2),
        (158, 61, 89, 18, 16, 3), (164, 66, 84, 15, 13, 4),
        (169, 76, 76, 0, 0, 4),
    ],
    "FarArm": [
        (95, 155, 155, 0, 0, -4), (101, 145, 164, 11, 14, -4),
        (109, 143, 170, 14, 16, -3), (119, 147, 175, 15, 17, -2),
        (129, 151, 178, 15, 15, 0), (137, 153, 179, 14, 13, 1),
        (144, 166, 166, 0, 0, 2),
    ],
    "FarGlove": [
        (125, 165, 165, 0, 0, 0), (131, 154, 177, 12, 12, 0),
        (139, 154, 180, 15, 14, 1), (148, 155, 180, 17, 15, 2),
        (155, 155, 179, 17, 15, 3), (161, 159, 176, 14, 13, 4),
        (166, 169, 169, 0, 0, 4),
    ],
    "NearThigh": [
        (151, 101, 101, 0, 0, -3), (158, 84, 122, 15, 15, -2),
        (166, 82, 121, 20, 18, 0), (175, 81, 122, 20, 19, 3),
        (183, 82, 118, 18, 17, 5), (188, 86, 115, 14, 14, 4),
        (193, 102, 102, 0, 0, 3),
    ],
    "FarThigh": [
        (151, 147, 147, 0, 0, -3), (158, 128, 163, 15, 15, -2),
        (166, 126, 165, 19, 17, 0), (175, 125, 163, 19, 18, 3),
        (182, 125, 160, 17, 16, 5), (187, 129, 158, 14, 14, 4),
        (192, 147, 147, 0, 0, 3),
    ],
    "NearShin": [
        (174, 103, 103, 0, 0, 4), (179, 85, 117, 15, 15, 4),
        (186, 81, 116, 16, 16, 3), (193, 83, 115, 15, 14, 2),
        (198, 87, 111, 11, 11, 0), (201, 99, 99, 0, 0, 0),
    ],
    "FarShin": [
        (173, 147, 147, 0, 0, 4), (178, 126, 160, 14, 14, 4),
        (185, 125, 161, 15, 15, 3), (191, 127, 159, 14, 13, 1),
        (196, 131, 156, 10, 10, 0), (199, 147, 147, 0, 0, 0),
    ],
    "NearBoot": [
        (182, 101, 101, 0, 0, 0), (186, 81, 116, 11, 11, 0),
        (191, 81, 116, 13, 13, 0), (196, 82, 112, 13, 12, 0),
        (201, 82, 111, 17, 10, 1), (206, 81, 112, 22, 10, 2),
        (211, 83, 111, 19, 10, 2), (213, 88, 106, 14, 8, 1),
        (214, 97, 97, 0, 0, 1),
    ],
    "FarBoot": [
        (181, 145, 145, 0, 0, 0), (185, 125, 161, 11, 11, 0),
        (190, 126, 161, 13, 13, 0), (195, 129, 158, 13, 12, 0),
        (200, 128, 166, 18, 10, 1), (205, 128, 167, 21, 10, 2),
        (209, 135, 165, 18, 9, 2), (210.5, 143, 161, 13, 8, 1),
        (212, 152, 152, 0, 0, 1),
    ],
    "Tail": [
        (111, 24, 24, 0, 0, -70), (115, 18, 34, 13, 14, -69),
        (122, 12, 45, 17, 18, -63), (131, 9, 53, 20, 21, -57),
        (140, 8, 61, 22, 22, -51), (149, 9, 72, 23, 23, -44),
        (157, 11, 86, 23, 24, -34), (164, 13, 92, 22, 23, -31),
        (171, 18, 89, 21, 21, -31), (178, 24, 83, 18, 19, -34),
        (184, 34, 72, 14, 15, -38), (188, 46, 64, 8, 10, -42),
        (190, 55, 55, 0, 0, -45),
    ],
}


def make_surface(name, rows, slices, angles, rear):
    lo, hi, df, dr, center = (v[:, None] for v in profile(rows, slices))
    s = np.sin(angles)[None, :]
    s[np.abs(s) < 1e-9] = 0
    x = lo + (hi - lo) * (1 - np.cos(angles))[None, :] * 0.5
    y = np.broadcast_to(slices[:, None], x.shape)
    z = center + (-dr if rear else df) * s
    if name == "Tail":
        z += 0.43 * (x - (lo + hi) / 2) + 14.0 * (1.0 - smooth(119, 166, y))
    if name == "Hips" and not rear:
        for pouch_x in (98, 155):
            z += (hi > lo) * s * 6.5 * np.exp(-((x - pouch_x) / 10) ** 2 - ((y - 153) / 9) ** 2)
    return x, y, z


def geometry(name, rows, rear):
    count = math.ceil((rows[-1][0] - rows[0][0]) / 0.9) + 1
    segments = 56
    slices = np.linspace(rows[0][0], rows[-1][0], count)
    x, y, z = make_surface(name, rows, slices, np.linspace(0, math.pi, segments + 1), rear)
    points = np.stack([x - PIVOT_X, BASE_Y - y, z], axis=-1).reshape(-1, 3)
    u, v = np.meshgrid(np.linspace(0, 1, count), np.linspace(0, 1, segments + 1), indexing="ij")
    uv = np.stack([u, v], axis=-1).reshape(-1, 2)
    start = (np.arange(count - 1)[:, None] * (segments + 1) + np.arange(segments)).ravel()
    end = start + segments + 1
    faces = np.vstack([np.stack([start, end, start + 1], axis=1),
                       np.stack([end, end + 1, start + 1], axis=1)])
    if rear:
        faces = faces[:, ::-1]
    a, b, c = (points[faces[:, i]] for i in range(3))
    solid = np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-7
    return points.astype("<f4"), uv.astype("<f4"), faces[solid].astype("<u2").ravel()


def tile(source, box, u, v):
    x, y, w, h = box
    return lookup(source, x + np.mod(u, w), y + np.mod(v, h))


def side_paint(name, source, x, y, z, noise):
    fabric = tile(source, (106, 124, 12, 13), z + 38, y).mean(axis=2)
    fold = 8 * np.cos(0.24 * z - 0.19 * y) + 4 * np.sin(0.34 * z + 0.42 * y)
    soft_light = 11 * smooth(-12, 18, z) - 8 * smooth(130, 149, y)
    grain = 0.17 * (fabric - 57) + noise * 1.35
    cloth = np.array([50, 60, 39]) + (soft_light + fold + grain)[..., None]
    trousers = np.array([52, 62, 41]) + (soft_light * 0.50 + fold * 1.34 + grain)[..., None]
    leather = np.array([67, 48, 29]) + (0.25 * (fabric - 57) + grain)[..., None]
    gold = np.array([134, 84, 26]) + (soft_light + grain + 12 * np.cos(y * 1.4))[..., None]
    glove = np.array([50, 42, 28]) + (soft_light * 0.5 + fold + grain)[..., None]
    if name == "Tail":
        length = 0.47 * (x - 23) + 0.31 * (y - 120) + 0.69 * (z + 74)
        stripe = smooth(-0.26, 0.21, np.cos((length - 4) * (2 * math.pi / 27.0) + noise * 0.044))
        color = np.array([50, 40, 28]) + stripe[..., None] * np.array([151, 134, 105])
        color *= (0.9 + 0.1 * smooth(-52, 0, z))[..., None]
        return color + noise[..., None] * 1.5
    if name.endswith("Boot"):
        bottom = 214 if name == "NearBoot" else 212
        toe = smooth(5, 21, z) * (1 - smooth(bottom - 7, bottom - 2, y))
        sole = smooth(bottom - 6, bottom - 3, y)
        cuff = smooth(bottom - 28, bottom - 25, y) * (1 - smooth(bottom - 22, bottom - 20, y))
        return leather * (1 - 0.35 * sole[..., None]) + toe[..., None] * np.array(
            [34, 26, 16]
        ) + cuff[..., None] * np.array([29, 19, 5])
    if name.endswith("Glove"):
        cuff_y = 145 if name == "NearGlove" else 143
        cuff = smooth(cuff_y - 4, cuff_y - 2, y) * (1 - smooth(cuff_y + 3, cuff_y + 5, y))
        hand = smooth(cuff_y + 4, cuff_y + 8, y)
        return cloth * (1 - cuff[..., None] - hand[..., None]) + gold * cuff[..., None] + glove * hand[..., None]
    if name == "Hips":
        belt = smooth(145, 147, y) * (1 - smooth(154, 157, y))
        belt_color = np.array([89, 61, 29]) + (soft_light + grain)[..., None]
        waist = trousers * (1 - belt[..., None]) + belt_color * belt[..., None]
        pouch = lookup(source, 92 + (z - 4) * 0.61, y)
        weight = smooth(4, 9, z) * (1 - smooth(22, 27, z))
        weight *= smooth(146, 149, y) * (1 - smooth(162, 165, y))
        return waist * (1 - weight[..., None]) + pouch * weight[..., None]
    if name.endswith(("Shin", "Thigh")):
        return trousers
    if name.endswith("Arm"):
        upper_seam = 2.4 - 0.17 * (y - 111)
        seam = np.exp(-((z - upper_seam) / 1.7) ** 2)
        seam *= smooth(101, 106, y) * (1 - smooth(130, 135, y))
        cuff_fold = np.exp(-((y - 133 - 0.11 * z) / 2.3) ** 2)
        cloth += (16 * seam - 10 * cuff_fold)[..., None]
    return cloth


def pigment(name, rows, source, front, noise, rear):
    length = math.ceil((rows[-1][0] - rows[0][0]) * 5)
    slices = rows[0][0] + (rows[-1][0] - rows[0][0]) * (np.arange(length) + 0.5) / length
    angles = math.pi * (np.arange(392) + 0.5) / 392
    x, y, z = make_surface(name, rows, slices, angles, rear)
    surface = np.stack([x, y, z], axis=-1)
    normal = np.cross(np.gradient(surface, axis=0), np.gradient(surface, axis=1))
    nx, nz = np.abs(normal[:, :, 0]), np.abs(normal[:, :, 2])
    facing = nz / np.maximum(nx + nz, 1e-8)
    grain = lookup(noise, z + 115, y)[:, :, 0]
    side = side_paint(name, source[:, :, :3], x, y, z, grain)
    if rear:
        color = side * (1 - 0.19 * smooth(0.1, 0.9, facing)[..., None])
    else:
        texture = lookup(front, x, y)
        face_weight = smooth(0.15, 0.51, facing)[..., None]
        color = texture * face_weight + side * (1 - face_weight)
    return np.clip(color, 0, 255).transpose(1, 0, 2).astype(np.uint8)


def surface(model, name, rows, pixels, rear):
    label = name + (" rear" if rear else " front")
    data = io.BytesIO()
    Image.fromarray(pixels).save(data, format="PNG", optimize=True)
    image = model.add("images", {
        "name": label, "mimeType": "image/png", "bufferView": model.view(data.getvalue())
    })
    texture = model.add("textures", {"sampler": 0, "source": image})
    material = model.add("materials", {
        "name": label,
        "extensions": {"KHR_materials_unlit": {}},
        "pbrMetallicRoughness": {"baseColorTexture": {"index": texture}},
        "doubleSided": True,
    })
    points, uv, faces = geometry(name, rows, rear)
    return {
        "attributes": {
            "POSITION": model.accessor(points, "VEC3", 5126, 34962),
            "TEXCOORD_0": model.accessor(uv, "VEC2", 5126, 34962),
        },
        "indices": model.accessor(faces, "SCALAR", 5123, 34963),
        "material": material,
    }


def main():
    source = np.asarray(Image.open(SOURCE).convert("RGBA"))
    y, x = np.mgrid[:source.shape[0], :source.shape[1]] + 0.5
    rng = np.random.default_rng(918)
    noise = rng.normal(0, 1, (216, 230)).astype(np.float32)
    noise = 3.0 * noise + 14.0 * cv2.GaussianBlur(noise, (5, 5), 1.25)
    noise = np.repeat(noise[:, :, None], 3, axis=2)
    model = ModelFile()
    model.data["asset"]["generator"] = "Graytail full-body study"
    model.data["nodes"][0]["name"] = "BodyParts"
    for name, rows in PARTS.items():
        lo, hi, *_ = profile(rows, y)
        valid = (y >= rows[0][0]) & (y <= rows[-1][0]) & (x >= lo) & (x <= hi) & (source[:, :, 3] > 128)
        front = nearest_colors(source[:, :, :3], valid)
        pieces = []
        for rear in (False, True):
            paint = pigment(name, rows, source, front, noise, rear)
            pieces.append(surface(model, name, rows, paint, rear))
        mesh = model.add("meshes", {"name": name, "primitives": pieces})
        node = model.add("nodes", {"name": name, "mesh": mesh})
        model.data["nodes"][0]["children"].append(node)
    model.write(ROOT / "body_3d.glb")


if __name__ == "__main__":
    main()
