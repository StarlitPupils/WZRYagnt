# -*- coding: utf-8 -*-
"""s01 小地图放大: 看小兵点样式。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s01.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
mm = img[:720, :1280][0:232, 0:232]
big = cv2.resize(mm, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(ROOT / "temp" / "v7dbg" / "s01_mm4x.png"), big)
# GT 我兵位置 (map 归一化): (0.06,0.36)(0.06,0.41)(0.07,0.44)(0.37,0.63)(0.34,0.64)(0.31,0.66)(0.53,0.91)(0.58,0.92)(0.61,0.92)
for gx, gy in [(0.06, 0.36), (0.37, 0.63), (0.53, 0.91)]:
    px, py = int(gx * 232), int(gy * 232)
    crop = mm[max(0, py - 10):py + 10, max(0, px - 10):px + 10]
    c2 = cv2.resize(crop, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(ROOT / "temp" / "v7dbg" / f"s01_minion_{int(gx*100)}_{int(gy*100)}.png"), c2)
    print(f"saved ({px},{py})")
