# -*- coding: utf-8 -*-
"""绿环候选：带像素质心相对环心的位移 (整环→小, 弧→大)。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7, _RING, _CENTER, RING_RATIO, CENTER_RATIO

GT_SELF = {1: (0.67, 0.92), 2: (0.74, 0.91), 3: (0.83, 0.89), 4: (0.88, 0.87),
           5: (0.65, 0.81), 6: (0.88, 0.91), 7: (0.87, 0.89), 12: (0.24, 0.78)}

print(f"{'frame':>6} {'pos':>10} {'drift':>6} {'nband':>6}  tag")
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
        # 环带像素 (半径 6-17 内 m>0)，质心漂移
        ys, xs = np.nonzero(m[py - 17:py + 18, px - 17:px + 18])
        if len(xs) == 0:
            continue
        xs = xs - 17 + px - px  # local
        ys = ys - 17
        dd = np.sqrt(xs.astype(float) ** 2 + ys.astype(float) ** 2)
        sel = (dd >= 6) & (dd <= 17)
        if sel.sum() == 0:
            continue
        mx, my = xs[sel].mean(), ys[sel].mean()
        drift = np.sqrt(mx ** 2 + my ** 2)
        tag = ""
        if i in GT_SELF:
            d = ((px / 232 - GT_SELF[i][0]) ** 2 + (py / 232 - GT_SELF[i][1]) ** 2) ** 0.5
            tag = "TRUE" if d < 0.12 else "farGT"
        print(f"s{i:02d} ({px:3d},{py:3d}) {drift:5.1f} {sel.sum():5d}  {tag}")
