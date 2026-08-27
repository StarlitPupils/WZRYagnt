# -*- coding: utf-8 -*-
"""各候选最优环半径: 真环 vs 假蓝环(发圈)。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def load_mm(i):
    img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / f"s{i:02d}.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img[:720, :1280][0:232, 0:232]

def radius_profile(mm, cx, cy, kind):
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    H, S = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    if kind == "g":
        m = ((H >= 35) & (H <= 90) & (S > 60)).astype(np.float32)
    elif kind == "b":
        m = ((H >= 70) & (H <= 135) & (S > 60)).astype(np.float32)
    else:
        m = (((H <= 15) | (H >= 165)) & (S > 60)).astype(np.float32)
    m = np.pad(m, 40)
    cx, cy = cx + 40, cy + 40
    yy, xx = np.mgrid[-17:18, -17:18]
    d = np.sqrt(xx ** 2 + yy ** 2)
    prof = []
    for r in range(5, 17):
        sel = (d >= r - 1.5) & (d <= r + 1.5)
        prof.append(float(m[cy - 17:cy + 18, cx - 17:cx + 18][sel].mean()))
    return prof

print("=== 真队友(蓝环) 半径5-16占比 ===")
for i, px, py in [(1, 25, 63), (1, 59, 92), (1, 102, 106), (1, 172, 215), (2, 43, 25), (2, 81, 147), (2, 64, 214)]:
    p = radius_profile(load_mm(i), px, py, "b")
    print(f"s{i:02d} ({px},{py}): " + " ".join(f"{v:.2f}" for v in p))
print("=== 假蓝环(发圈) ===")
for i, px, py in [(1, 140, 92), (1, 135, 110), (1, 135, 113), (2, 127, 115)]:
    p = radius_profile(load_mm(i), px, py, "b")
    print(f"s{i:02d} ({px},{py}): " + " ".join(f"{v:.2f}" for v in p))
print("=== 真自己(绿环) ===")
for i, px, py in [(1, 155, 215), (2, 170, 214), (3, 194, 207), (4, 204, 205), (5, 149, 187), (6, 203, 218), (7, 202, 210)]:
    p = radius_profile(load_mm(i), px, py, "g")
    print(f"s{i:02d} ({px},{py}): " + " ".join(f"{v:.2f}" for v in p))
