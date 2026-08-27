# -*- coding: utf-8 -*-
"""裁剪 s04/s06/s07 自己区域放大保存，并放松阈值测试。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import _RING, _CENTER

GT_SELF = {4: (0.88, 0.87), 6: (0.88, 0.91), 7: (0.87, 0.89)}

outdir = ROOT / "temp" / "v7dbg"
outdir.mkdir(exist_ok=True)
for i, (gx, gy) in GT_SELF.items():
    img = cv2.imread(str(ROOT / "temp" / "ann" / f"s{i:02d}.png"))
    canvas = img[:720, :1280]
    mm = canvas[0:232, 0:232]
    px, py = int(gx * 232), int(gy * 232)
    crop = mm[py - 25:py + 25, px - 25:px + 25]
    big = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(outdir / f"s{i:02d}_self.png"), big)
    print(f"s{i:02d} saved")
