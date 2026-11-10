"""Verify the gray renders and draw native-pixel outline comparisons."""

import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fit_outline import ROOT, V27, PROTECTED_FACE, actor_mask, glb_views


ANGLES = (0, 45, 90, 135, 180)
ROWS = range(93, 119)
BG = "#28333c"
FONT = "C:/Windows/Fonts/msyh.ttc"
OLD = "#ffba6c"
NEW = "#eff5fa"
GUIDE = "#65e2d8"


def rgba(path):
    with Image.open(path) as image:
        assert image.size == (480, 256) and image.mode == "RGBA", path
        return np.asarray(image)


def ids(path):
    return (rgba(path)[:, :, 0].astype(np.uint16) + 7) // 14


def edges(mask):
    xs = [np.flatnonzero(mask[y]) for y in ROWS]
    return np.asarray([(x[0], x[-1]) for x in xs])


def verify_geometry():
    baseline, old = glb_views(V27 / "head_3d.glb")
    updated, new = glb_views(ROOT / "head_3d.glb")
    fur = next(mesh for mesh in baseline["meshes"] if mesh["name"] == "Fur")
    expected = {baseline["accessors"][item["attributes"]["POSITION"]]["bufferView"]
                for item in fur["primitives"]}
    assert len(old) == len(new) and baseline["meshes"] == updated["meshes"]
    assert {index for index, (a, b) in enumerate(zip(old, new)) if a != b} == expected


def draw_edges(draw, mask, color, dashed=False):
    outline = edges(mask)
    for column in (0, 1):
        samples = [(int(x), y) for x, y in zip(outline[:, column], ROWS)]
        if dashed:
            for i in range(0, len(samples), 4):
                draw.line(samples[i:i + 2], fill=color, width=1)
        else:
            draw.line(samples, fill=color, width=1)


def run():
    verify_geometry()
    folder = ROOT / "overlays"
    folder.mkdir(exist_ok=True)
    font = ImageFont.truetype(FONT, 13)
    board = Image.new("RGB", (1260, 256), BG)
    guide = {90: actor_mask("side-090"), 180: actor_mask("back-180")}
    l, t, r, b = PROTECTED_FACE
    report = {}
    for i, angle in enumerate(ANGLES):
        name = f"{angle:03d}.png"
        before = rgba(ROOT / "diagnostic/baseline" / name)
        after = rgba(ROOT / "frames" / name)
        old_ids = ids(V27 / "diagnostic/fit-ids" / name)
        new_ids = ids(ROOT / "diagnostic/piece-ids" / name)
        old_mask = before[:, :, 3] != 0
        new_mask = after[:, :, 3] != 0
        assert np.array_equal(old_mask, old_ids != 0) and np.array_equal(new_mask, new_ids != 0)
        for gray in (before, after):
            assert np.array_equal(gray[:, :, 0], gray[:, :, 1])
            assert np.array_equal(gray[:, :, 1], gray[:, :, 2])
        assert cv2.connectedComponents(new_mask.astype(np.uint8), connectivity=8)[0] == 2
        delta = np.any(before != after, axis=2)
        assert not (delta & (old_ids != 4) & (new_ids != 4)).any(), ("Change outside cheek", angle)
        if angle == 0:
            assert not delta[t:b, l:r].any(), "Protected front region changed"
        if angle == 90:
            assert np.array_equal(old_mask, new_mask), "Side silhouette changed"
        if angle == 180:
            right_old = np.abs(edges(old_mask)[:, 1] - edges(guide[angle])[:, 1])
            right_new = np.abs(edges(new_mask)[:, 1] - edges(guide[angle])[:, 1])
            assert right_new.mean() < 0.76 * right_old.mean(), ("Rear silhouette fit", right_old.mean(), right_new.mean())
            report["rear_right_edge_mean_error_px"] = [float(right_old.mean()), float(right_new.mean())]
            report["rear_right_edge_at_y117"] = [int(edges(m)[117 - ROWS.start, 1]) for m in (old_mask, new_mask, guide[angle])]
        report[str(angle)] = {"shape_pixels_changed": int((old_mask != new_mask).sum()),
                             "render_pixels_changed": int(delta.sum())}

        image = Image.new("RGB", (480, 256), BG)
        shaded = Image.fromarray(after)
        image.paste(shaded, (0, 0), shaded)
        draw = ImageDraw.Draw(image)
        draw_edges(draw, new_mask, NEW)
        draw_edges(draw, old_mask, OLD, dashed=True)
        if angle in guide:
            draw_edges(draw, guide[angle], GUIDE, dashed=True)
        text = f"{angle}°   黄 V27 · 白 V28" + (" · 青 V25" if angle in guide else "")
        draw.text((240, 14), text, font=font, fill="#eef0ed", anchor="mm")
        image.save(folder / name, optimize=True)
        with Image.open(folder / name) as saved:
            assert saved.size == (480, 256)
            saved.verify()
        board.paste(image.crop((114, 0, 366, 256)), (252 * i, 0))
    board.save(ROOT / "gray-overlays.png", optimize=True)
    (ROOT / "review-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("V28_GRAY_OVERLAYS_OK", report)


if __name__ == "__main__":
    run()
