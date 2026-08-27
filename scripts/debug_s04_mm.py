# -*- coding: utf-8 -*-
"""s04 小地图 4x + 敌人检出标记。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7

img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s04.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
mm = img[:720, :1280][0:232, 0:232]
det = MMDetectorV7()
r = det.detect(img[:720, :1280])
big = cv2.resize(mm, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
for i, d in enumerate(r["dots"]["enemy"]):
    px, py = int(d["n"][0] * 232), int(d["n"][1] * 232)
    cv2.circle(big, (px * 4, py * 4), 16, (0, 255, 255), 3)
    cv2.putText(big, str(i), (px * 4 - 8, py * 4 + 6), 0, 0.8, (255, 0, 255), 3)
# GT 敌对位: (0.18,0.08),(0.54,0.42),(0.95,0.83)
for gx, gy in [(0.18, 0.08), (0.54, 0.42), (0.95, 0.83)]:
    px, py = int(gx * 232), int(gy * 232)
    cv2.circle(big, (px * 4, py * 4), 10, (255, 255, 255), 2)
cv2.imwrite(str(ROOT / "temp" / "v7dbg" / "s04_mm_enemy.png"), big)
print("saved; dets:", [(round(d['n'][0], 2), round(d['n'][1], 2)) for d in r["dots"]["enemy"]])
