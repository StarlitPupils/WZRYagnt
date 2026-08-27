# -*- coding: utf-8 -*-
"""漏检敌人环区域放大。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
outdir = ROOT / "temp" / "v7dbg"
for tag, gx, gy in [("s05", 0.11, 0.13), ("s09", 0.25, 0.13)]:
    img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / f"{tag}.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
    canvas = img[:720, :1280]
    px, py = int(gx * 232), int(gy * 232)
    crop = canvas[max(0, py - 18):py + 18, max(0, px - 18):px + 18]
    big = cv2.resize(crop, None, fx=7, fy=7, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(outdir / f"{tag}_enemmm.png"), big)
    print(f"{tag} saved px=({px},{py})")
