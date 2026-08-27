# -*- coding: utf-8 -*-
"""B 通道组件调试: s05 (25,30)。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import _RING, _CENTER, _has_avatar

img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s05.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
mm = img[:720, :1280][0:232, 0:232]
hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
H, S = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
m = (((H <= 15) | (H >= 165)) & (S > 60)).astype(np.float32)
ring_map = cv2.filter2D(m, -1, _RING)
cen_map = cv2.filter2D(m, -1, _CENTER)
print("at (25,30): ring=", round(float(ring_map[30, 25]), 3), "cen=", round(float(cen_map[30, 25]), 3))
hits = ((ring_map > 0.62) & (cen_map < 1.0)).astype(np.uint8) * 255
n, lab, st, cent = cv2.connectedComponentsWithStats(hits, 8)
print("components:", n - 1)
for i in range(1, n):
    cx, cy = int(cent[i][0]), int(cent[i][1])
    if not (15 <= cx <= 217 and 15 <= cy <= 217):
        continue
    win = ring_map[cy - 6:cy + 7, cx - 6:cx + 7] - cen_map[cy - 6:cy + 7, cx - 6:cx + 7] * 2.0
    dy, dx = np.unravel_index(np.argmax(win), win.shape)
    px, py = cx - 6 + dx, cy - 6 + dy
    print(f"  comp{i} cent=({cx},{cy}) size={st[i][4]} -> ({px},{py}) ring={ring_map[py,px]:.2f} "
          f"avatar={_has_avatar(mm, px, py)}")
