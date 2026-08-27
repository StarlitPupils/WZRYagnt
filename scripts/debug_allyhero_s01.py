# -*- coding: utf-8 -*-
"""s01 我方英雄 GT 框区域放大(含头顶)。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s01.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
# GT (774,271)-(925,457) + 顶部60px
crop = img[220:470, 740:960]
big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(ROOT / "temp" / "v7dbg" / "s01_allyhero.png"), big)
print("saved")
