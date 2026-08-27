# -*- coding: utf-8 -*-
"""绿环候选中心 9x9 橙肤色占比统计。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7, _RING, _CENTER, RING_RATIO, CENTER_RATIO

GT_SELF = {1: (0.67, 0.92), 2: (0.74, 0.91), 3: (0.83, 0.89), 4: (0.88, 0.87),
           5: (0.65, 0.81), 6: (0.88, 0.91), 7: (0.87, 0.89), 12: (0.24, 0.78)}

for i in range(1, 13):
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
    for ci in range(1, n):
        cx, cy = int(cent[ci][0]), int(cent[ci][1])
        if not (15 <= cx <= 217 and 15 <= cy <= 217):
            continue
        # 中心 9x9 橙色(肤色)占比
        ch = H[cy - 4:cy + 5, cx - 4:cx + 5]
        cs = S[cy - 4:cy + 5, cx - 4:cx + 5]
        cv = V[cy - 4:cy + 5, cx - 4:cx + 5]
        orange = (((ch >= 8) & (ch <= 30) & (cs > 70) & (cv > 110))).mean()
        tag = ""
        if i in GT_SELF:
            d = ((cx / 232 - GT_SELF[i][0]) ** 2 + (cy / 232 - GT_SELF[i][1]) ** 2) ** 0.5
            tag = "TRUE" if d < 0.12 else ""
        print(f"s{i:02d} ({cx:3d},{cy:3d}) orange={orange:.2f} {tag}")
