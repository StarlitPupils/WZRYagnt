# -*- coding: utf-8 -*-
"""漏检敌人环 HSV 分布诊断。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]

for tag, cx, cy in [("s05", 25, 30), ("s09", 58, 30)]:
    img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / f"{tag}.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
    mm = img[:720, :1280][0:232, 0:232]
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    yy, xx = np.mgrid[-16:17, -16:17]
    d = np.sqrt(xx ** 2 + yy ** 2)
    sel = (d >= 8) & (d <= 14)
    hb = H[cy - 16:cy + 17, cx - 16:cx + 17][sel]
    sb = S[cy - 16:cy + 17, cx - 16:cx + 17][sel]
    vb = V[cy - 16:cy + 17, cx - 16:cx + 17][sel]
    print(f"{tag} ({cx},{cy}) band H p50={np.median(hb):.0f} p10={np.percentile(hb,10):.0f} p90={np.percentile(hb,90):.0f}")
    print(f"        S p50={np.median(sb):.0f} p90={np.percentile(sb,90):.0f} V p50={np.median(vb):.0f} p90={np.percentile(vb,90):.0f}")
    # 各半径环带占比
    m = (((H <= 15) | (H >= 165)) & (S > 60)).astype(np.float32)
    for r in (7, 8, 9, 10, 11, 12, 13, 14, 15):
        sel2 = (d >= r - 1.5) & (d <= r + 1.5)
        frac = m[cy - 16:cy + 17, cx - 16:cx + 17][sel2].mean()
        print(f"   r={r}: red_frac={frac:.2f}")
