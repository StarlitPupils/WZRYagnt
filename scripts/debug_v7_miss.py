# -*- coding: utf-8 -*-
"""找 v7 漏检队友: GT 位置附近无边密度的候选。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7, _BAND_SEL

det = MMDetectorV7()

def edge_at(mm, cx, cy, half=6):
    patch = mm[cy - half:cy + half + 1, cx - half:cx + half + 1]
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float((np.sqrt(gx ** 2 + gy ** 2) > 60).mean())

missing = 0
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    mm = img[:720, :1280][0:232, 0:232]
    r = det.detect(img[:720, :1280])
    ally = r["dots"]["ally"]
    gt_lines = [ln for ln in fp.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
                if ln.strip() and int(ln.split()[0]) == 12]
    for ln in gt_lines:
        p = ln.split()
        x, y = float(p[1]), float(p[2])
        px, py = int(x * 1280), int(y * 720)   # 文件是全帧归一化 -> 设备px(即地图px)
        best = min(((a["n"][0] - px / 232) ** 2 + (a["n"][1] - py / 232) ** 2) ** 0.5 for a in ally)
        if best >= 0.12:
            missing += 1
            try:
                ed = edge_at(mm, px, py)
            except cv2.error:
                ed = -1
            print(f"s{i:02d} ally GT ({px/232:.3f},{py/232:.3f}) 缺失 edge={ed:.2f}")
print("missing:", missing)
