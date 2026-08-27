# -*- coding: utf-8 -*-
"""s02 塔/小兵裁剪: 我方 vs 敌方 的颜色对比 + 色占比测量。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s02.png"), dtype=np.uint8), cv2.IMREAD_COLOR)

def meas(name, x1, y1, x2, y2, tag):
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    blue = (((H >= 70) & (H <= 135) & (S > 60))).mean()
    red = ((((H <= 15) | (H >= 165)) & (S > 60))).mean()
    green = (((H >= 35) & (H <= 90) & (S > 60))).mean()
    print(f"{name} {tag}: blue={blue:.3f} red={red:.3f} green={green:.3f}")
    big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(ROOT / "temp" / "v7dbg" / f"full_{name}.png"), big)

meas("turret_ally", 352, 306, 563, 622, "GT我方塔(被标敌塔)")
meas("minion_ally", 611, 423, 696, 503, "GT我方兵(被标敌兵)")
meas("minion_enemy", 959, 279, 1018, 341, "GT敌方兵(被标我兵)")
meas("minion_enemy2", 1163, 163, 1207, 223, "GT敌方兵2(被标我兵)")
