# -*- coding: utf-8 -*-
"""红条 vs GT敌英 / 蓝条(宽>=55) vs GT我英 关联对比。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.self_bars import find_enemy_bars, find_ally_bars

hits_e = tot_e = hits_a = tot_a = 0
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    ebars = [b for b in find_enemy_bars(img) if b["w"] >= 55]
    abars = [b for b in find_ally_bars(img) if b["w"] >= 55]
    for ln in fp.with_suffix(".txt").read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        if c not in (0, 1):
            continue
        x, y, w, h = [float(v) for v in p[1:5]]
        g = ((x - w / 2) * 1280, (y - h / 2) * 720, (x + w / 2) * 1280, (y + h / 2) * 720)
        if c == 0:
            tot_e += 1
            ok = any(g[0] - 50 <= b["cx"] <= g[2] + 50 and g[1] - 50 <= b["y"] <= g[3] + 50 for b in ebars)
            hits_e += int(ok)
            if not ok:
                print(f"s{i:02d} GT敌英{g} 红条未命中; 红条={[(b['cx'],b['y'],b['w']) for b in ebars]}")
        else:
            tot_a += 1
            ok = any(g[0] - 50 <= b["cx"] <= g[2] + 50 and g[1] - 50 <= b["y"] <= g[3] + 50 for b in abars)
            hits_a += int(ok)
            if not ok:
                print(f"s{i:02d} GT我英{g} 蓝条(w>=55)未命中")
print(f"红条(w>=55)命中敌英: {hits_e}/{tot_e}; 蓝条(w>=55)命中我英: {hits_a}/{tot_a}")
