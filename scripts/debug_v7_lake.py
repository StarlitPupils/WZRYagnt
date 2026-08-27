# -*- coding: utf-8 -*-
"""s01 蓝环候选: 环带 S/V p90 + 中心边密度分类（湖 vs 英雄）。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import _BAND_SEL, _has_avatar, MMDetectorV7

img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s01.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
mm = img[:720, :1280][0:232, 0:232]
hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
det = MMDetectorV7()
r = det.detect(img[:720, :1280])
GT_ALLY = [(0.08, 0.24), (0.26, 0.40), (0.44, 0.47), (0.72, 0.93)]
for d in r["dots"]["ally"]:
    px, py = int(d["n"][0] * 232), int(d["n"][1] * 232)
    patch = hsv[py - 15:py + 16, px - 15:px + 16]
    s90 = np.percentile(patch[..., 1][_BAND_SEL], 90)
    v90 = np.percentile(patch[..., 2][_BAND_SEL], 90)
    ed = _has_avatar(mm, px, py, thr=0.0)
    tag = "HERO" if min(((d["n"][0] - x) ** 2 + (d["n"][1] - y) ** 2) ** 0.5 for x, y in GT_ALLY) < 0.08 else "lake?"
    print(f"({d['n'][0]:.3f},{d['n'][1]:.3f}) S90={s90:.0f} V90={v90:.0f} edge={ed:.2f} {tag}")
