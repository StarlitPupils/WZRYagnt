# -*- coding: utf-8 -*-
"""s02 自己血条区域诊断: 绿条位置/蓝条宽度/真实MP。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.self_bars import _find_bars, self_hp_mp

img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s02.png"), dtype=np.uint8), cv2.IMREAD_COLOR)

def is_green(H, S, V):
    return (H >= 35) & (H <= 90) & (S > 50)

def is_blue(H, S, V):
    return (H >= 90) & (H <= 135) & (S > 60) & (V > 60)

greens = _find_bars(img, is_green, "self")
greens = [b for b in greens if b["y"] >= 250 and b["y"] <= 550 and not (b["x"] < 240 and b["y"] < 240)]
print("s02 绿条候选:", [(b["x"], b["y"], b["w"]) for b in greens])
best = max(greens, key=lambda b: b["w"])
bx, by, bw = best["x"], best["y"], best["w"]
print(f"最优绿条: x={bx} y={by} w={bw} HP={bw/115:.2f}")

# 看绿条下方 30px 内的蓝色横条
sub = img[by + 2:by + 30, bx - 60:bx + 60]
hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
mask = (is_blue(H, S, V)).astype(np.uint8) * 255
k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
n, lab, st, cent = cv2.connectedComponentsWithStats(mask2, 8)
for i in range(1, n):
    if st[i, cv2.CC_STAT_WIDTH] >= 12 and st[i, cv2.CC_STAT_HEIGHT] <= 14:
        print(f"蓝条: w={st[i, cv2.CC_STAT_WIDTH]} h={st[i, cv2.CC_STAT_HEIGHT]} "
              f"x={int(cent[i][0])} y={int(cent[i][1])} -> MP(w/115)={st[i, cv2.CC_STAT_WIDTH]/115:.2f} "
              f"MP(w/90)={st[i, cv2.CC_STAT_WIDTH]/90:.2f}")

# 保存放大裁剪
crop = img[by - 12:by + 30, bx - 60:bx + 60]
big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(ROOT / "temp" / "v7dbg" / "s02_selfbar.png"), big)
print("裁剪已存 temp/v7dbg/s02_selfbar.png")
hp, mp, pos = self_hp_mp(img)
print("self_hp_mp 输出:", hp, mp, pos)
