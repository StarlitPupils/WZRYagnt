# -*- coding: utf-8 -*-
"""s01 (135,110) 蓝环候选放大。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s01.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
mm = img[:720, :1280][0:232, 0:232]
for tag, (px, py) in [("lake135", (135, 110)), ("lake140", (140, 92)), ("lake74", (74, 72))]:
    crop = mm[py - 16:py + 16, px - 16:px + 16]
    big = cv2.resize(crop, None, fx=7, fy=7, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(ROOT / "temp" / "v7dbg" / f"s01_{tag}.png"), big)
    print(tag, "saved")
