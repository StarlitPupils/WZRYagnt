# -*- coding: utf-8 -*-
"""各绿环候选：最佳半径处 12 扇区覆盖率 + 上下半分布。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7, _RING, _CENTER, RING_RATIO, CENTER_RATIO

FRAMES = list(range(1, 13))
for i in FRAMES:
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    canvas = img[:720, :1280]
    mm = canvas[0:232, 0:232]
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    m = ((H >= 35) & (H <= 90) & (S > 60)).astype(np.float32)
    ring_map = cv2.filter2D(m, -1, _RING)
    cen_map = cv2.filter2D(m, -1, _CENTER)
    hits = ((ring_map > RING_RATIO["self"]) & (cen_map < CENTER_RATIO)).astype(np.uint8) * 255
    n, lab, st, cent = cv2.connectedComponentsWithStats(hits, 8)
    lines = []
    for ci in range(1, n):
        cx, cy = int(cent[ci][0]), int(cent[ci][1])
        if not (15 <= cx <= 217 and 15 <= cy <= 217):
            continue
        best, br = 0.0, 0
        for r in range(7, 17):
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            dd = np.sqrt(xx.astype(float) ** 2 + yy.astype(float) ** 2)
            sel = (dd <= r + 2.5) & (dd >= max(0.5, r - 2.5))
            patch = m[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
            if patch.shape != sel.shape:
                continue
            vals = patch[sel]
            ang = (np.degrees(np.arctan2(yy[sel], xx[sel])) + 360) % 360
            sect = np.zeros(12, dtype=bool)
            for a, v in zip(ang, vals):
                if v > 0:
                    sect[int(a // 30)] = True
            cov = sect.sum()
            if cov > best:
                best, br = float(cov), r
        lines.append(f"  ({cx:3d},{cy:3d}) n={st[ci][4]:3d} sectors={best:.0f}/12@r={br}")
    print(f"s{i:02d}:"); print("\n".join(lines))
