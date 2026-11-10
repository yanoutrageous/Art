"""Fit the approved turnaround to the original menu rig in its own UV space."""

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
V14 = ROOT.parent / "walk_3d_v14"
V13 = ROOT.parent / "head_3d_v13"
REFERENCE = V14 / "shape_review/art_reference/raccoon-turnaround-continuity-v2.png"
sys.path[:0] = [str(V14), str(V13)]
from build_body import PARTS, make_surface
from build_model import (BASE_Y, EAR_LEFT, EAR_RIGHT, HAT, HEAD, NECK, PIVOT_X,
                         SEGMENTS, nearest_colors, profile, smooth, surface_grid)


HEAD_ROWS = {"Hat": HAT, "Fur": HEAD, "EarL": EAR_LEFT, "EarR": EAR_RIGHT, "Scarf": NECK}

# Landmarks in a 1260-pixel-wide view of the reference. Exclude neighboring
# pieces at each occlusion so the arm swing never reveals a second cuff or eye.
SIDE = {
    "Hat": [(548, 105), (558, 55), (579, 24), (636, 16), (688, 29), (716, 52),
            (736, 77), (716, 97), (654, 118), (587, 128), (550, 122)],
    "Fur": [(551, 126), (579, 126), (620, 121), (658, 116), (651, 139),
            (654, 157), (642, 169), (601, 171), (570, 159), (553, 143)],
    "Ear": [(592, 22), (604, 17), (619, 22), (628, 39), (627, 56), (619, 74),
            (601, 69), (588, 52), (584, 39)],
    "Scarf": [(582, 175), (596, 169), (619, 170), (641, 175), (665, 181),
              (685, 192), (684, 205), (674, 206), (650, 188), (609, 181), (586, 192)],
    "Chest": [[(587, 201), (604, 191), (614, 190), (606, 209), (602, 226),
               (610, 244), (588, 250), (585, 222)],
              [(661, 200), (674, 205), (686, 224), (685, 247), (668, 248), (663, 233)]],
    "Arm": [(608, 207), (619, 192), (634, 190), (649, 197), (658, 215),
            (657, 244), (618, 251), (608, 240), (605, 224)],
    "Glove": [(612, 247), (655, 242), (663, 265), (657, 274), (660, 289),
              (654, 300), (636, 302), (624, 296), (622, 278), (616, 271)],
    "Hips": [[(588, 249), (613, 247), (617, 272), (612, 285), (591, 276)],
             [(660, 249), (681, 248), (693, 257), (690, 287), (675, 295), (661, 290)]],
    "Leg": [(612, 289), (624, 299), (639, 303), (657, 302), (664, 292),
            (680, 291), (678, 315), (660, 333), (610, 336), (599, 323)],
    "Boot": [(607, 336), (669, 334), (666, 354), (681, 366), (684, 386),
             (658, 389), (606, 387), (601, 377)],
    "Tail": [(495, 238), (507, 213), (529, 202), (543, 204), (561, 221),
             (572, 242), (584, 258), (596, 274), (606, 290), (606, 311),
             (594, 323), (574, 329), (549, 326), (524, 312), (507, 292), (499, 265)],
}

BACK = {
    "Hat": [(902, 109), (924, 83), (937, 54), (965, 27), (993, 19), (1012, 20),
            (1046, 37), (1064, 66), (1077, 92), (1100, 108), (1101, 117),
            (1092, 124), (1057, 121), (1019, 114), (981, 115), (932, 125), (907, 123)],
    "Fur": [(906, 130), (936, 125), (972, 117), (1021, 117), (1063, 123),
            (1094, 132), (1080, 148), (1059, 158), (1029, 166), (971, 168),
            (933, 158), (916, 146)],
    "Ear": [[(909, 25), (922, 19), (940, 27), (955, 46), (935, 67), (916, 59)],
            [(1046, 45), (1061, 27), (1080, 19), (1090, 28), (1091, 48), (1080, 66), (1066, 63)]],
    "Scarf": [(940, 172), (964, 166), (1019, 166), (1055, 171), (1037, 186),
              (1008, 195), (990, 195), (963, 185)],
    "Chest": [(929, 195), (946, 184), (968, 191), (991, 198), (1006, 198),
              (1030, 190), (1051, 183), (1072, 196), (1077, 226), (1062, 248),
              (1043, 253), (956, 252), (931, 244), (922, 220)],
    "Arm": [[(908, 236), (912, 220), (922, 202), (933, 191), (944, 190),
             (954, 212), (947, 244), (935, 250), (907, 243)],
            [(1043, 204), (1052, 189), (1064, 190), (1079, 204), (1088, 218),
             (1093, 237), (1083, 248), (1055, 248)]],
    "Glove": [(902, 244), (931, 247), (930, 270), (925, 282), (922, 295),
              (910, 296), (900, 286), (899, 266)],
    "Hips": [(923, 251), (950, 246), (972, 252), (1024, 252), (1053, 246),
             (1078, 250), (1077, 265), (1068, 275), (1041, 290), (1003, 292),
             (971, 290), (944, 298), (923, 290)],
    "Leg": [(927, 298), (947, 298), (962, 290), (1002, 290), (1043, 292),
            (1035, 312), (1044, 325), (1064, 336), (1022, 340), (1007, 321),
            (991, 320), (980, 339), (940, 336), (928, 322)],
    "Boot": [[(936, 339), (988, 338), (989, 385), (981, 389), (936, 388),
              (932, 383), (937, 359)],
             [(1015, 339), (1063, 337), (1068, 349), (1066, 364), (1073, 379),
              (1070, 388), (1019, 389), (1015, 383)]],
    "Tail": [(1105, 227), (1126, 208), (1145, 203), (1160, 216), (1165, 238),
             (1163, 266), (1150, 292), (1133, 313), (1106, 326), (1081, 332),
             (1054, 327), (1039, 316), (1041, 296), (1054, 274), (1078, 253)],
}

FRONT = {"Tail": [(91, 221), (107, 204), (126, 201), (143, 210), (159, 231),
                  (170, 259), (178, 280), (197, 294), (204, 310), (182, 325),
                  (151, 328), (123, 314), (103, 290), (92, 259)]}
BODY_Y = ([91, 101, 111, 123, 138, 145, 149, 157, 167, 178, 186, 195],
          [164, 182, 199, 219, 244, 249, 256, 275, 292, 317, 337, 344])
TAIL_Y = ([111, 118, 132, 146, 164, 178, 190], [202, 210, 235, 267, 302, 321, 329])


def guides():
    with Image.open(REFERENCE) as image:
        rgb = np.array(image.convert("RGB").resize((1260, 420), Image.Resampling.LANCZOS))
    colors = rgb.astype(np.int16)
    foreground = (colors[:, :, 0] > colors[:, :, 2] + 3) | (colors.max(axis=2) < 27)
    layers = {}
    for view, regions in (("side", SIDE), ("back", BACK), ("front", FRONT)):
        for part, polygons in regions.items():
            mask = np.zeros(rgb.shape[:2], np.uint8)
            if isinstance(polygons[0], tuple):
                polygons = [polygons]
            cv2.fillPoly(mask, [np.array(polygon, np.int32) for polygon in polygons], 1)
            if (view, part) == ("side", "Hat"):
                cv2.ellipse(mask, (606, 44), (23, 31), -15, 0, 360, 0, -1)
            valid = (mask != 0) & foreground
            assert np.count_nonzero(valid) > 450, (view, part)
            if part == "Tail":
                luminance = colors.mean(axis=2).astype(np.float32)
                detail = luminance - cv2.GaussianBlur(luminance, (7, 7), 1.5)
                grain = np.clip(detail, -15, 15) + 128
                layers[view, part] = nearest_colors(
                    np.repeat(grain[:, :, None], 3, axis=2).astype(np.uint8), valid
                )
                low, high = np.percentile(luminance[valid], [48, 56])
                layers[view, "Tail dark"] = nearest_colors(rgb, valid & (luminance <= low))
                layers[view, "Tail light"] = nearest_colors(rgb, valid & (luminance >= high))
            else:
                layers[view, part] = nearest_colors(rgb, valid)
    return layers


def rig_grid(name, rear, slices, theta):
    if name == "Tail":
        x, y, _ = make_surface(name, PARTS[name], slices, theta, rear)
        source_y = np.interp(y, *TAIL_Y)
        side_y = [202, 210, 232, 249, 267, 282, 298, 310, 322, 329]
        left = np.interp(source_y, side_y, [534, 515, 499, 495, 500, 509, 521, 537, 553, 574])
        right = np.interp(source_y, side_y, [534, 555, 568, 579, 590, 602, 609, 607, 595, 574])
        z = ((left + right) * 0.5 - 632) / 1.59
        radius = (right - left) * (0.5 / 1.59)
        z += (-1 if rear else 1) * radius * np.sin(theta)
        # The thick visible part continues into a small attachment under the belt.
        across = (1 - np.cos(theta)) * 0.5
        root = smooth(149, 164, y) * (1 - smooth(164, 177, y)) * across**5
        return x + 34 * root, y, z + 27 * root
    if name not in HEAD_ROWS:
        return make_surface(name, PARTS[name], slices, theta, rear)
    rows = HEAD_ROWS[name]
    axis = "x" if name == "Hat" else "y"
    x, y, z = surface_grid(name, rows, axis, rear, slices, theta)
    if name == "Hat" and rear:
        fraction = (1 - np.cos(theta)) * 0.5
        center = smooth(72, 103, x) * (1 - smooth(148, 176, x))
        brim = center * np.sin(theta) * np.exp(-((fraction - 0.79) / 0.19)**2)
        y = y + 25 * brim
        z = z - 6 * brim
    if name == "Scarf" and rear:
        fold = smooth(96, 106, y) * (1 - smooth(110, 119, y)) * np.sin(theta)
        z = z - 6 * fold
    return x, y, z


def projection(name, x, y, z):
    side_x, back_x = 632 + 1.59 * z, 999 - 1.59 * (x - PIVOT_X)
    side_y = back_y = np.interp(y, *BODY_Y)
    part = name.removeprefix("Near").removeprefix("Far")
    if name == "Hat":
        side_y = np.interp(y, [3, 17, 32, 43, 56, 68, 78], [17, 38, 65, 89, 111, 127, 132])
        back_y = np.interp(y, [3, 18, 36, 46, 53, 62, 74], [18, 44, 79, 97, 112, 124, 128])
    elif name == "Fur":
        side_y = np.interp(y, [45, 57, 66, 78, 90, 100], [103, 118, 131, 148, 164, 172])
        back_y = np.interp(y, [48, 57, 66, 82, 94, 100], [112, 122, 134, 156, 167, 171])
    elif name.startswith("Ear"):
        part = "Ear"
        side_x = 630 + 1.45 * z
        side_y = back_y = np.interp(y, [2, 13, 28, 41], [18, 36, 59, 76])
    elif name == "Scarf":
        side_y = np.interp(y, [89, 95, 100, 106, 115, 123], [165, 171, 178, 184, 197, 208])
        back_y = np.interp(y, [89, 95, 100, 106, 115, 123], [163, 168, 174, 185, 196, 198])
    elif part == "Glove":
        # The right hand is partly hidden in the rear reference. Use its symmetric cuff.
        back_x = 999 - 1.59 * np.abs(x - PIVOT_X)
        top = PARTS[name][0][0]
        side_y = back_y = np.interp(y - top, [0, 11, 16, 22, 29, 38, 44],
                                    [227, 244, 250, 266, 279, 294, 302])
    elif part in {"Thigh", "Shin"}:
        part = "Leg"
    elif part == "Boot":
        bottom = PARTS[name][-1][0]
        side_y = back_y = np.interp(y - bottom, [-33, -27, -20, -11, -5, 0],
                                    [335, 342, 355, 369, 379, 389])
    elif part == "Tail":
        side_y = np.interp(y, *TAIL_Y)
    return part, (side_x, side_y), (back_x, back_y)


def read_color(layer, coordinates):
    x, y = (p.astype(np.float32) for p in coordinates)
    return cv2.remap(layer, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)


def paint_tail(x, y, z, rear, side_uv, facing, layers):
    # Ring boundaries run through the volume, independently of camera angle.
    # Recover fine fur from the corresponding view without projecting its stripes twice.
    side_y = side_uv[1]
    guide_x = 999 - 1.59 * (x - PIVOT_X) if rear else 94 + 1.57 * (x - 8)
    view = "back" if rear else "front"
    weight = smooth(0.12, 0.65, facing)[:, :, None]
    grain, dark, pale = [
        read_color(layers[view, key], (guide_x, side_y)) * weight
        + read_color(layers["side", key], side_uv) * (1 - weight)
        for key in ("Tail", "Tail dark", "Tail light")
    ]
    noise = (0.78 * np.sin(2.9 * x + 1.7 * y + 1.3 * z) + 0.64 * np.cos(2.2 * y - 2.4 * z)
             + 0.48 * np.cos(0.42 * x - 0.63 * y + 0.84 * z))
    along = 0.46 * (x - 24) + 0.59 * (y - 111) + 0.61 * (z + 61)
    ring = smooth(-0.25, 0.22, np.cos((along + noise + 1.8) * (2 * np.pi / 23.0)))
    root_shadow = smooth(85, 116, x) * smooth(-42, -22, z)
    ring *= 1 - 0.85 * root_shadow
    shading = 5.5 * np.sin((x - z) * 0.068) - 8.0 * root_shadow
    dark = 0.62 * dark + 0.38 * np.array([50, 40, 29])
    pale = 0.62 * pale + 0.38 * np.array([203, 177, 140])
    paint = dark + ring[:, :, None] * (pale - dark)
    return np.clip(paint + 0.50 * (grain - 128) + shading[:, :, None], 0, 255)


def paint(name, rear, old, layers):
    length, angles = old.shape[:2][::-1]
    rows = (HEAD_ROWS if name in HEAD_ROWS else PARTS)[name]
    slices = rows[0][0] + (rows[-1][0] - rows[0][0]) * (np.arange(length) + 0.5) / length
    theta = np.pi * (np.arange(angles) + 0.5) / angles
    x, y, z = rig_grid(name, rear, slices, theta)
    part, side_uv, back_uv = projection(name, x, y, z)
    points = np.stack([x, y, z], axis=-1)
    normal = np.cross(np.gradient(points, axis=0), np.gradient(points, axis=1))
    nx, nz = np.abs(normal[:, :, 0]), np.abs(normal[:, :, 2])
    facing = nz / np.maximum(nx + nz, 1e-8)
    if name == "Tail":
        return np.rint(paint_tail(x, y, z, rear, side_uv, facing, layers)).transpose(1, 0, 2).astype(np.uint8)
    side = read_color(layers["side", part], side_uv)
    old = old.transpose(1, 0, 2)
    if rear:
        other = read_color(layers["back", part], back_uv)
        weight = smooth(0.14, 0.59, facing)
        if name == "Fur":
            # Meet the untouched face texture without painting the generated eye on the flank.
            boundary = smooth(0.0, 0.23, np.sin(theta))[None, :, None]
            side = old * (1 - boundary) + side * boundary
    else:
        other = old
        weight = smooth(0.07, 0.40, facing)
        if name == "Hat":
            clasp = smooth(144, 150, x) * (1 - smooth(160, 166, x))
            clasp *= smooth(19, 24, y) * (1 - smooth(33, 38, y))
            weight = np.maximum(weight, clasp)
    weight = weight[:, :, None]
    return np.rint(other * weight + side * (1 - weight)).transpose(1, 0, 2).astype(np.uint8)


def glb_views(path):
    raw = path.read_bytes()
    size, tag = struct.unpack_from("<I4s", raw, 12)
    assert raw[:4] == b"glTF" and tag == b"JSON"
    data = json.loads(raw[20:20 + size])
    payload = 28 + size
    return data, [raw[payload + view["byteOffset"]:payload + view["byteOffset"] + view["byteLength"]]
                  for view in data["bufferViews"]]


def build(source, output, layers):
    data, buffers = glb_views(source)
    changed = []
    for mesh in data["meshes"]:
        name = mesh["name"]
        for rear, primitive in enumerate(mesh["primitives"]):
            if name == "Tail" or rear and name in {"Hat", "Scarf"}:
                accessor = data["accessors"][primitive["attributes"]["POSITION"]]
                rows = HEAD_ROWS.get(name) or PARTS[name]
                segments = SEGMENTS if name in HEAD_ROWS else 56
                count = accessor["count"] // (segments + 1)
                slices = np.linspace(rows[0][0], rows[-1][0], count)
                x, y, z = rig_grid(name, rear, slices, np.linspace(0, np.pi, segments + 1))
                vertices = np.stack([x - PIVOT_X, BASE_Y - y, z], axis=-1).reshape(-1, 3).astype("<f4")
                assert len(vertices) == accessor["count"]
                buffers[accessor["bufferView"]] = vertices.tobytes()
                accessor["min"], accessor["max"] = vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()
                changed.append(f"{name} {'rear' if rear else 'front'}")
    for image in data["images"]:
        name, side = image["name"].split()
        if side == "front" and name in {"Fur", "EarL", "EarR"}:
            continue
        slot = image["bufferView"]
        with Image.open(io.BytesIO(buffers[slot])) as old:
            material = paint(name, side == "rear", np.asarray(old), layers)
        png = io.BytesIO()
        Image.fromarray(material).save(png, format="PNG", compress_level=6)
        buffers[slot] = png.getvalue()

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
    return changed


def main():
    layers = guides()
    report = {"reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest()}
    for source, destination in ((V13 / "head_v13.glb", "head_3d.glb"), (V14 / "body_3d.glb", "body_3d.glb")):
        report[destination] = {"reprofiled_surfaces": build(source, ROOT / destination, layers)}
    (ROOT / "material-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
