# -*- coding: utf-8 -*-
"""s01 FP 区域 (661,155)-(801,343) 放大。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s01.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
crop = img[100:360, 560:860]
big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(ROOT / "temp" / "v7dbg" / "s01_fphero.png"), big)
print("saved")
