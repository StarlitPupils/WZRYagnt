# -*- coding: utf-8 -*-
"""s07 自己血条区域放大。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s07.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
# 用检测到的绿条位置: s07 x=625 y=331-55=276 → 裁剪区域 (550..700, 258..310)
crop = img[255:315, 545:705]
big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(ROOT / "temp" / "v7dbg" / "s07_selfbar.png"), big)
print("saved")
