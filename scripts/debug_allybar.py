# -*- coding: utf-8 -*-
"""蓝血条 vs GT 我方英雄框对应分析(12帧)。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.self_bars import detect_all_bars

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)

hits = total = 0
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    bars = detect_all_bars(img)
    blues = bars["allies"]
    gts = []
    for ln in fp.with_suffix(".txt").read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5 or int(p[0]) != 1:
            continue
        x, y, w, h = [float(v) for v in p[1:5]]
        gts.append(((x - w / 2) * 1280, (y - h / 2) * 720, (x + w / 2) * 1280, (y + h / 2) * 720))
    for g in gts:
        total += 1
        # 蓝条中心是否落在 GT 框内(上下扩展一点)
        ok = any(g[0] - 30 <= b["x"] <= g[2] + 30 and g[1] - 30 <= b["y"] <= g[3] + 30 for b in blues)
        hits += int(ok)
        if not ok:
            print(f"s{i:02d} GT我英{g} 无蓝条命中; 蓝条={[(b['x'],b['y'],b['w']) for b in blues]}")
    print(f"s{i:02d} 蓝条数={len(blues)} GT我英={len(gts)} 蓝条位置={[(b['x'],b['y'],b['w']) for b in blues]}")
print(f"蓝条命中 GT我英: {hits}/{total}")
