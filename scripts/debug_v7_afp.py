# -*- coding: utf-8 -*-
"""队友候选与 GT/塔点对比。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7, BLUE_TOWER_PTS

det = MMDetectorV7()

def norm_px(p):
    return (p["n"][0], p["n"][1])

for i in (1, 2):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    r = det.detect(img[:720, :1280])
    ally = [norm_px(d) for d in r["dots"]["ally"]]
    print(f"s{i:02d} ally dets ({len(ally)}):")
    for a in ally:
        near_tw = min(((a[0] - x) ** 2 + (a[1] - y) ** 2) ** 0.5 for x, y in BLUE_TOWER_PTS)
        print(f"  ({a[0]:.3f},{a[1]:.3f}) near_blue_tower={near_tw:.3f}")
