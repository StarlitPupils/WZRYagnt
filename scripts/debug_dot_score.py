# -*- coding: utf-8 -*-
"""小兵点候选: 分数 vs 环带V 分析(s01)。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
mm = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / "s01.png"), dtype=np.uint8), cv2.IMREAD_COLOR)[:720, :1280][0:232, 0:232]
hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
V = hsv[..., 2].astype(int)
tpls = []
for f in sorted((ROOT / "temp" / "mm_dot_templates").glob("ally_*.png")):
    t = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
    if t is not None:
        tpls.append(t)
gt = []
for ln in (ROOT / "temp" / "ann" / "s01.txt").read_text(encoding="utf-8").splitlines():
    p = ln.split()
    if len(p) == 5 and int(p[0]) == 18:
        gt.append((float(p[1]) * 1280, float(p[2]) * 720))
cands = []
for t in tpls:
    r = cv2.matchTemplate(mm, t, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.nonzero(r > 0.62)
    for v, x, y in sorted(zip(r[ys, xs].tolist(), xs.tolist(), ys.tolist()), reverse=True):
        px, py = x + 9, y + 9
        if all((px - a) ** 2 + (py - b) ** 2 > 36 for a, b, _ in cands):
            cands.append((px, py, v))
print("候选数", len(cands))
yy, xx = np.mgrid[-14:15, -14:15]
dd = np.sqrt(xx ** 2 + yy ** 2)
sel = (dd >= 6) & (dd <= 12)
for px, py, v in cands[:70]:
    isgt = any((px - gx) ** 2 + (py - gy) ** 2 <= 12 ** 2 for gx, gy in gt)
    vmean = float(V[py - 14:py + 15, px - 14:px + 15][sel].mean())
    tag = "GT" if isgt else ""
    print(f"({px:3d},{py:3d}) score={v:.3f} ringV={vmean:.0f}  {tag}")
