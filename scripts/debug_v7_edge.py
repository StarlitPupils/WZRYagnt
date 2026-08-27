# -*- coding: utf-8 -*-
"""绿环候选中心纹理(边缘密度)：头像=高纹理，水面/森林=低纹理。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7, _RING, _CENTER, RING_RATIO, CENTER_RATIO

GT_SELF = {1: (0.67, 0.92), 2: (0.74, 0.91), 3: (0.83, 0.89), 4: (0.88, 0.87),
           5: (0.65, 0.81), 6: (0.88, 0.91), 7: (0.87, 0.89), 12: (0.24, 0.78)}

def edge_density(img_bgr, cx, cy, half=6):
    patch = img_bgr[cy - half:cy + half + 1, cx - half:cx + half + 1]
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float((mag > 60).mean())

for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    canvas = img[:720, :1280]
    mm = canvas[0:232, 0:232]
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    H, S = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    m = ((H >= 35) & (H <= 90) & (S > 60)).astype(np.float32)
    ring_map = cv2.filter2D(m, -1, _RING)
    cen_map = cv2.filter2D(m, -1, _CENTER)
    hits = ((ring_map > RING_RATIO["self"]) & (cen_map < CENTER_RATIO)).astype(np.uint8) * 255
    n, lab, st, cent = cv2.connectedComponentsWithStats(hits, 8)
    for ci in range(1, n):
        cx, cy = int(cent[ci][0]), int(cent[ci][1])
        if not (15 <= cx <= 217 and 15 <= cy <= 217):
            continue
        win = ring_map[cy - 6:cy + 7, cx - 6:cx + 7] - cen_map[cy - 6:cy + 7, cx - 6:cx + 7] * 2.0
        dy, dx = np.unravel_index(np.argmax(win), win.shape)
        px, py = cx - 6 + dx, cy - 6 + dy
        ed = edge_density(mm, px, py)
        tag = ""
        if i in GT_SELF:
            d = ((px / 232 - GT_SELF[i][0]) ** 2 + (py / 232 - GT_SELF[i][1]) ** 2) ** 0.5
            tag = "TRUE" if d < 0.12 else ""
        print(f"s{i:02d} ({px:3d},{py:3d}) edge={ed:.2f} {tag}")
