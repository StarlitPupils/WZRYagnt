# -*- coding: utf-8 -*-
"""s02 绿环候选详细诊断：质心、校正中心、ring/cen 数值。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7, _RING, _CENTER, RING_RATIO, CENTER_RATIO

for i in (2, 5, 12, 8):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
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
    print(f"== s{i:02d} green components ==")
    for ci in range(1, n):
        cx, cy = int(cent[ci][0]), int(cent[ci][1])
        if not (15 <= cx <= 217 and 15 <= cy <= 217):
            continue
        win = ring_map[cy - 6:cy + 7, cx - 6:cx + 7] - cen_map[cy - 6:cy + 7, cx - 6:cx + 7] * 2.0
        dy, dx = np.unravel_index(np.argmax(win), win.shape)
        px, py = cx - 6 + dx, cy - 6 + dy
        print(f"  comp{ci} centroid=({cx},{cy}) size={st[ci][4]} -> corrected=({px},{py}) "
              f"ring={ring_map[py,px]:.2f} cen={cen_map[py,px]:.2f} "
              f"nrm=({px/232:.3f},{py/232:.3f})")
