# -*- coding: utf-8 -*-
"""QC v4_f05 红框区域放大。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "pseudo1" / "qc_v4_f05.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
crop = img[250:470, 500:720]
big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(ROOT / "temp" / "v7dbg" / "qc_v4f05.png"), big)
print("saved")
