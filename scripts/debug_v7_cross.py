# -*- coding: utf-8 -*-
"""交叉色环带值: 假蓝环点的 red_ring 值 vs 真敌环点的 green_ring 值。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import _RING

def load_mm(i):
    img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / f"s{i:02d}.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img[:720, :1280][0:232, 0:232]

def ringval(mm, cx, cy, kind):
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    H, S = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    if kind == "g":
        m = ((H >= 35) & (H <= 90) & (S > 60)).astype(np.float32)
    elif kind == "b":
        m = ((H >= 70) & (H <= 135) & (S > 60)).astype(np.float32)
    else:
        m = (((H <= 15) | (H >= 165)) & (S > 60)).astype(np.float32)
    return float(cv2.filter2D(m, -1, _RING)[cy, cx])

# 假蓝环上的 red_ring
print("== 假蓝环点上的 red_ring 值 ==")
for i, px, py in [(1, 140, 92), (1, 135, 110), (1, 135, 113), (2, 127, 115)]:
    print(f"s{i:02d} ({px},{py}): red_ring={ringval(load_mm(i), px, py, 'r'):.2f}")

# 真实敌环 vs 自我的情况 s03: 敌(197,202) 自己的 green_ring
print("== 真实敌环点上的 green_ring 值 (s03) ==")
mm3 = load_mm(3)
print(f"s03 (197,202): green_ring={ringval(mm3, 197, 202, 'g'):.2f}")
print(f"s03 (216,193): green_ring={ringval(mm3, 216, 193, 'g'):.2f}")
# 对照: 敌环上其他点 green_ring
# 自己的红环干扰对照: s04 自己(204,205)附近 敌(220,193)
mm4 = load_mm(4)
print("== s04: 敌人(220,193) green_ring =", round(ringval(mm4, 220, 193, 'g'), 2))
print("== s12: 自己(58,180) red_ring =", round(ringval(load_mm(12), 58, 180, 'r'), 2))
