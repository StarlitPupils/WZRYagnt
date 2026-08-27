# -*- coding: utf-8 -*-
"""合并分段蓝条检测 vs GT 我方英雄(真实队友)关联重测。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def find_blue_bars_merged(frame, y_top=70, y_bot=635, x_max=1230):
    """蓝色横条(含分段刻度合并)：列投影 + 行高约束。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    mask = ((H >= 90) & (H <= 135) & (S > 60) & (V > 60)).astype(np.uint8)
    bars = []
    # 逐行找蓝色 run; 行高<=12 且水平连续(允许 gap<=9)的 band
    for y in range(y_top, y_bot):
        row = mask[y, :x_max]
        # 找 run
        idx = np.nonzero(row)[0]
        if len(idx) < 20:
            continue
        # 分组: gap<=9 合并
        groups = []
        start = prev = idx[0]
        for i in idx[1:]:
            if i - prev > 9:
                groups.append((start, prev))
                start = i
            prev = i
        groups.append((start, prev))
        for g in groups:
            if 25 <= g[1] - g[0] + 1 <= 150:
                bars.append((y, g[0], g[1] - g[0] + 1))
    # 合并相邻行(bar 高 2-8) → 取每组最低行的代表
    bars.sort()
    merged = []
    for y, x0, w in bars:
        if merged and y - merged[-1][0] <= 3 and abs(x0 - merged[-1][1]) <= 10:
            merged[-1] = (y, x0, max(w, merged[-1][2]))
        else:
            merged.append((y, x0, w))
    return [(x0 + w // 2, y, w) for y, x0, w in merged if w >= 25]

hits = total = 0
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    bars = find_blue_bars_merged(img)
    gts = []
    for ln in fp.with_suffix(".txt").read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5 or int(p[0]) != 1:
            continue
        x, y, w, h = [float(v) for v in p[1:5]]
        gts.append(((x - w / 2) * 1280, (y - h / 2) * 720, (x + w / 2) * 1280, (y + h / 2) * 720))
    # 排除自己(绿条 y 250-550 内, x 600-680) 附近的蓝条
    for g in gts:
        total += 1
        ok = any(g[0] - 40 <= bx <= g[2] + 40 and g[1] - 40 <= by <= g[3] + 40 for bx, by, w in bars)
        hits += int(ok)
        if not ok:
            print(f"s{i:02d} GT我英{g} 未命中; 蓝条={bars}")
    if i in (1, 3, 10, 11):
        print(f"s{i:02d} 蓝条(合并)={bars}")
print(f"合并蓝条命中 GT我英: {hits}/{total}")
