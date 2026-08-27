# -*- coding: utf-8 -*-
"""裁剪指定候选位置放大保存。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    ("s05", 0.944, 0.211), ("s05", 0.9397, 0.806), ("s02", 0.6509, 0.931),
    ("s12", 0.181, 0.858), ("s08", 0.1595, 0.9353), ("s03", 0.9353, 0.1724),
]
outdir = ROOT / "temp" / "v7dbg"
outdir.mkdir(exist_ok=True)
for tag, gx, gy in TARGETS:
    fp = ROOT / "temp" / "ann" / f"{tag}.png"
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    canvas = img[:720, :1280]
    px, py = int(gx * 232), int(gy * 232)
    crop = canvas[max(0, py - 20):py + 20, max(0, px - 20):px + 20]
    big = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(outdir / f"{tag}_{int(gx*1000)}_{int(gy*1000)}.png"), big)
    print(f"{tag} ({gx},{gy}) saved")
