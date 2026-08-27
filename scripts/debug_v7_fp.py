# -*- coding: utf-8 -*-
"""打印 v7 各帧自己候选坐标 + 裁剪 FP 区域。"""
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

outdir = ROOT / "temp" / "v7dbg"
outdir.mkdir(exist_ok=True)
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    canvas = img[:720, :1280]
    r = det.detect(canvas)
    cands = r["dots"]["self"]
    gt = GT_SELF.get(i)
    print(f"s{i:02d} self candidates: {[(c['n'][0], c['n'][1]) for c in cands]}")
    if gt is None:
        # 无 GT 帧：裁剪前 3 个候选
        for j, c in enumerate(cands[:3]):
            px, py = int(c["n"][0] * 232), int(c["n"][1] * 232)
            crop = canvas[max(0, py - 20):py + 20, max(0, px - 20):px + 20]
            big = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(outdir / f"s{i:02d}_fp{j}.png"), big)
