# -*- coding: utf-8 -*-
"""调试 s04/s06/s07 自己环：期望位置附近 3 色环/中心占比。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import _RING, _CENTER, RING_RATIO, CENTER_RATIO

GT_SELF = {1: (0.67, 0.92), 2: (0.74, 0.91), 3: (0.83, 0.89), 4: (0.88, 0.87),
           5: (0.65, 0.81), 6: (0.88, 0.91), 7: (0.87, 0.89), 12: (0.24, 0.78)}

for i in (4, 6, 7, 2, 1):
    if i not in GT_SELF:
        continue
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    img = cv2.imread(str(fp))
    if img is None:
        continue
    canvas = img[:720, :1280]
    mm = canvas[0:232, 0:232]
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    H, S = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    masks = {
        "g": ((H >= 35) & (H <= 90) & (S > 60)).astype(np.float32),
        "b": ((H >= 70) & (H <= 135) & (S > 60)).astype(np.float32),
        "r": (((H <= 15) | (H >= 165)) & (S > 60)).astype(np.float32),
    }
    gx, gy = GT_SELF[i]
    px, py = int(gx * 232), int(gy * 232)
    print(f"s{i:02d} self_gt=({gx},{gy}) px={px},{py}")
    for name, m in masks.items():
        ring_map = cv2.filter2D(m, -1, _RING)
        cen_map = cv2.filter2D(m, -1, _CENTER)
        # 期望位置 9x9 邻域内最大值
        r = ring_map[py - 4:py + 5, px - 4:px + 5].max()
        c = cen_map[py - 4:py + 5, px - 4:px + 5].max()
        hit_r = ring_map[py, px]
        hit_c = cen_map[py, px]
        print(f"  {name}: ring_max={r:.2f} cen_max={c:.2f} (need r>{RING_RATIO}, c<{CENTER_RATIO})")
        # 尝试不同半径
        for rlo, rhi, cr in [(8, 14, 5), (6, 12, 5), (9, 16, 6), (10, 18, 6)]:
            yy, xx = np.mgrid[-20:21, -20:21]
            d = np.sqrt(xx ** 2 + yy ** 2)
            k1 = ((d >= rlo) & (d <= rhi)).astype(np.float32); k1 /= k1.sum()
            k2 = (d <= cr).astype(np.float32); k2 /= k2.sum()
            r2 = cv2.filter2D(m, -1, k1)[py, px]
            c2 = cv2.filter2D(m, -1, k2)[py, px]
            print(f"    r({rlo}-{rhi},c{cr}): ring={r2:.2f} cen={c2:.2f} {'HIT' if r2>RING_RATIO and c2<CENTER_RATIO else ''}")
