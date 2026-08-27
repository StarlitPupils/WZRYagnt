# -*- coding: utf-8 -*-
"""真队友环 vs 蓝头画像环的环带 H 分布对比。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import _BAND_SEL, _RING, _CENTER, RING_RATIO, CENTER_CAP

# 真队友: s01 (25,63)(59,92)(102,106)(172,215) | 相机蓝头像: s01 (140,92) s01 (135,110) s02 (127,115)
HERO = [(1, 25, 63), (1, 59, 92), (1, 102, 106), (1, 172, 215), (2, 43, 25), (2, 81, 147), (2, 64, 214), (2, 103, 106)]
FAKE = [(1, 140, 92), (1, 135, 110), (1, 135, 113), (2, 127, 115), (1, 74, 72)]
for i, px, py in HERO + FAKE:
    img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / f"s{i:02d}.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
    mm = img[:720, :1280][0:232, 0:232]
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    H, S = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    # 环带内 S>100 的像素 H 分布
    hs = H[py - 15:py + 16, px - 15:px + 16][_BAND_SEL]
    ss = S[py - 15:py + 16, px - 15:px + 16][_BAND_SEL]
    sel = ss > 100
    if sel.sum() > 5:
        print(f"s{i:02d} ({px},{py}) H_sat: med={np.median(hs[sel]):.0f} p25={np.percentile(hs[sel],25):.0f} p75={np.percentile(hs[sel],75):.0f} n={sel.sum()}")
    else:
        print(f"s{i:02d} ({px},{py}) 饱和像素太少")
