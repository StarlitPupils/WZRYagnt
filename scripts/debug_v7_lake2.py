# -*- coding: utf-8 -*-
"""s01/s02 全部蓝环候选(无头像过滤) 的 edge 值 + 与 GT 距离。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import _RING, _CENTER, _BAND_SEL, _has_avatar, RING_RATIO, CENTER_CAP

GT_ALLY = {1: [(0.08, 0.24), (0.26, 0.40), (0.44, 0.47), (0.72, 0.93)],
           2: [(0.16, 0.09), (0.26, 0.91), (0.35, 0.61), (0.44, 0.48)]}
for i in (1, 2):
    img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / f"s{i:02d}.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
    mm = img[:720, :1280][0:232, 0:232]
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    m = ((H >= 70) & (H <= 135) & (S > 60)).astype(np.float32)
    ring_map = cv2.filter2D(m, -1, _RING)
    cen_map = cv2.filter2D(m, -1, _CENTER)
    hits = ((ring_map > RING_RATIO["ally"]) & (cen_map < CENTER_CAP["ally"])).astype(np.uint8) * 255
    n, lab, st, cent = cv2.connectedComponentsWithStats(hits, 8)
    print(f"== s{i:02d} ==")
    for ci in range(1, n):
        cx, cy = int(cent[ci][0]), int(cent[ci][1])
        if not (15 <= cx <= 217 and 15 <= cy <= 217):
            continue
        win = ring_map[cy - 6:cy + 7, cx - 6:cx + 7] - cen_map[cy - 6:cy + 7, cx - 6:cx + 7] * 2.0
        dy, dx = np.unravel_index(np.argmax(win), win.shape)
        px, py = cx - 6 + dx, cy - 6 + dy
        patch = mm[py - 6:py + 7, px - 6:px + 7]
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        ed = float((mag > 60).mean())
        patch2 = hsv[py - 15:py + 16, px - 15:px + 16]
        s90 = np.percentile(patch2[..., 1][_BAND_SEL], 90)
        v90 = np.percentile(patch2[..., 2][_BAND_SEL], 90)
        dgt = min(((px / 232 - x) ** 2 + (py / 232 - y) ** 2) ** 0.5 for x, y in GT_ALLY[i])
        tag = "HERO" if dgt < 0.08 else ""
        print(f"  ({px:3d},{py:3d}) edge={ed:.2f} S90={s90:.0f} V90={v90:.0f} {tag}")
