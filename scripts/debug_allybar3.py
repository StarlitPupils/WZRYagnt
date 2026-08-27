# -*- coding: utf-8 -*-
"""干净版蓝条 vs GT我英 关联测试 + 每条条位置。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.self_bars import find_ally_bars

hits = total = 0
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    bars = find_ally_bars(img)
    gts = []
    for ln in fp.with_suffix(".txt").read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5 or int(p[0]) != 1:
            continue
        x, y, w, h = [float(v) for v in p[1:5]]
        gts.append(((x - w / 2) * 1280, (y - h / 2) * 720, (x + w / 2) * 1280, (y + h / 2) * 720))
    print(f"s{i:02d} 蓝条={[(b['cx'], b['y'], b['w']) for b in bars]}")
    for g in gts:
        total += 1
        ok = any(g[0] - 40 <= b["cx"] <= g[2] + 40 and g[1] - 40 <= b["y"] <= g[3] + 40 for b in bars)
        hits += int(ok)
        if not ok:
            print(f"  MISS GT{g}")
print(f"蓝条命中 GT我英: {hits}/{total}")
