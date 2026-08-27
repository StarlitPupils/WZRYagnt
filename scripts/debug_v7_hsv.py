# -*- coding: utf-8 -*-
"""分析各绿环候选的环带 HSV 分布：亮度/饱和区分真环 vs 装饰底盘。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7

det = MMDetectorV7()
GT_SELF = {1: (0.67, 0.92), 2: (0.74, 0.91), 3: (0.83, 0.89), 4: (0.88, 0.87),
           5: (0.65, 0.81), 6: (0.88, 0.91), 7: (0.87, 0.89), 12: (0.24, 0.78)}

for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    canvas = img[:720, :1280]
    mm = canvas[0:232, 0:232]
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    r = det.detect(canvas)
    cands = r["dots"]["self"]
    gt = GT_SELF.get(i)
    print(f"s{i:02d}:")
    for c in cands:
        px, py = int(c["n"][0] * 232), int(c["n"][1] * 232)
        yy, xx = np.mgrid[-15:16, -15:16]
        d = np.sqrt(xx ** 2 + yy ** 2)
        sel = (d >= 8) & (d <= 14)
        patch = hsv[max(0, py - 15):py + 16, max(0, px - 15):px + 16]
        if patch.shape[0] != 31 or patch.shape[1] != 31:
            continue
        S = patch[..., 1][sel].astype(int)
        V = patch[..., 2][sel].astype(int)
        cen_s = hsv[py - 4:py + 5, px - 4:px + 5, 1].astype(int).mean()
        cen_v = hsv[py - 4:py + 5, px - 4:px + 5, 2].astype(int).mean()
        tag = "GT"
        if gt:
            distv = ((c["n"][0] - gt[0]) ** 2 + (c["n"][1] - gt[1]) ** 2) ** 0.5
            if distv < 0.12:
                tag = "TRUE"
        print(f"  ({c['n'][0]:.3f},{c['n'][1]:.3f}) bandS={S.mean():.0f}(p90={np.percentile(S,90):.0f}) "
              f"bandV={V.mean():.0f}(p90={np.percentile(V,90):.0f}) cenS={cen_s:.0f} cenV={cen_v:.0f} {tag}")
