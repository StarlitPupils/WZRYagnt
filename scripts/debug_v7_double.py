# -*- coding: utf-8 -*-
"""s02 双绿环区域放大 + 标记。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s02.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
canvas = img[:720, :1280]
mm = canvas[0:232, 0:232]
# 两个候选 (170,214) 与 (151,216)；裁剪 (125,185)-(200,240)
crop = mm[180:240, 120:205]
big = cv2.resize(crop, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
# 标记: (170,214)->(170-120,214-180)=(50,34)*8=(400,272); (151,216)->(31,36)*8=(248,288)
cv2.circle(big, (400, 272), 12, (255, 0, 255), 2)   # GT自己 (洋红)
cv2.circle(big, (248, 288), 12, (0, 255, 255), 2)   # 第二绿环 (黄)
cv2.imwrite(str(ROOT / "temp" / "v7dbg" / "s02_double.png"), big)
print("saved")
