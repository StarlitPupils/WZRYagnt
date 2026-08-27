# -*- coding: utf-8 -*-
"""从用户真值提取小兵点模板(19x19)，存 temp/mm_dot_templates/。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "temp" / "mm_dot_templates"
OUT.mkdir(parents=True, exist_ok=True)

imgs = {"ally": [], "enemy": []}
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    gtf = fp.with_suffix(".txt")
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    mm = img[:720, :1280][0:232, 0:232]
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    teal = ((H >= 80) & (H <= 150) & (S > 70) & (V >= 95))
    red = (((H <= 15) | (H >= 160)) & (S > 70) & (V >= 95))
    for ln in gtf.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        if c not in (18, 19):
            continue
        px, py = int(float(p[1]) * 1280), int(float(p[2]) * 720)
        if not (10 <= px <= 222 and 10 <= py <= 222):
            continue
        t = mm[py - 9:py + 10, px - 9:px + 10]
        if t.shape[:2] != (19, 19):
            continue
        key = "ally" if c == 18 else "enemy"
        # 中心 7x7 要有足够色信号 (宽松: 模板本身编码亮度)
        sig = (H >= 80) & (H <= 150) & (S > 60) if key == "ally" else ((H <= 15) | (H >= 160)) & (S > 60)
        center = int(sig[py - 3:py + 4, px - 3:px + 4].sum())
        if center >= 8:
            imgs[key].append(t)

for key, arr in imgs.items():
    print(key, "模板数:", len(arr))
    for j, t in enumerate(arr):
        cv2.imwrite(str(OUT / f"{key}_{j:03d}.png"), t)
print("已存", OUT)
